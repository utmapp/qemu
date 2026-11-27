/*
 * QEMU PCI VGA Emulator.
 *
 * see docs/specs/standard-vga.rst for virtual hardware specs.
 *
 * Copyright (c) 2003 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "hw/pci/pci_device.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "vga_int.h"
#include "ui/pixel_ops.h"
#include "ui/console.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "hw/loader.h"
#include "hw/display/edid.h"
#include "qom/object.h"
#include "hw/acpi/acpi_aml_interface.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "qemu/units.h"
#include "qemu/log.h"
#include "system/reset.h"
#include "trace.h"
#ifdef __APPLE__
#include "qemu/sockets.h"
#include <netinet/tcp.h>
/* The transport header is all static; only the in-process path is used here */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "vmware_passthrough_opengl_transport.h"
#pragma GCC diagnostic pop
#endif

enum vga_pci_flags {
    PCI_VGA_FLAG_ENABLE_MMIO = 1,
    PCI_VGA_FLAG_ENABLE_QEXT = 2,
    PCI_VGA_FLAG_ENABLE_EDID = 3,
    PCI_VGA_FLAG_ENABLE_MAC_ACCEL = 4,
};

/*
 * PVMAC: paravirtual display extension for the Mac OS 9 NDRV (qemu_vga.ndrv
 * built from QemuMacDrivers QemuVGADriver). Opt-in through the
 * "mac-accel" property; documented in docs/specs/standard-vga.rst.
 * Version 1 provides a hardware cursor only.
 *
 * Version 2 adds a byte-stream transport to VirtualIntelMac's in-process
 * PassthroughOpenGL host executor (VIMGLInprocessAPI, library from env
 * VIM_PASSTHROUGH_OPENGL_INPROCESS_LIBRARY), for a guest OpenGL/RAVE library.
 * It is advertised with CAPS bit 1 only when the library loads; otherwise the
 * bit stays clear (one warning at realize) and the GL registers read 0.
 *
 * VRAM layout, from the top (all reserved from the guest's framebuffer):
 *   vram_size - 0x4000                cursor slot (v1)
 *   ... - PVMAC_GL_REPLY_SIZE         reply area (4 MiB; device writes,
 *                                     guest reads; was 1 MiB before E4)
 *   ... - PVMAC_GL_RING_SIZE          command ring (guest writes, device reads)
 * Everything from the ring offset up is reserved; the guest framebuffer and
 * any REPLY_DEST rectangle must lie below RING_OFFSET. VRAM must be at least
 * PVMAC_GL_MIN_VRAM (16 MiB) for the GL stream to be offered.
 *
 * Registers (32-bit LE at BAR2 + 0x800 + offset):
 *   0x00 ID          'PVMC'                   0x04 VERSION   2
 *   0x08 CAPS        bit0 cursor, bit1 GL     0x0c STATUS    (v1 cursor errors)
 *   0x10-0x24        cursor, see v1
 *   0x28 RING_OFFSET  ro  VRAM byte offset of the command ring
 *   0x2c RING_SIZE    ro  ring bytes
 *   0x30 RING_HEAD    rw  guest write index, < RING_SIZE
 *   0x34 RING_TAIL    ro  device read index, == HEAD after a doorbell
 *   0x38 REPLY_OFFSET ro  VRAM byte offset of the reply area
 *   0x3c REPLY_SIZE   ro  reply area bytes
 *   0x40 REPLY_LEN    ro  bytes the last doorbell stored at REPLY_OFFSET
 *   0x44 DOORBELL     wo  any value: consume TAIL..HEAD, then fetch reply
 *   0x48 GL_STATUS    rw1c error bits, PVMAC_GL_ERR_*
 *   0x4c REPLY_EXPECT rw  exact reply bytes to read on the next doorbell
 *                          (<= REPLY_SIZE); 0 = no reply. The executor read
 *                          blocks, so the guest must name the exact span.
 *   0x50 REPLY_DEST   rw  VRAM byte offset of a destination rectangle for
 *                          the reply (used only with REPLY_FLAGS bit3;
 *                          0 is a valid offset)
 *   0x54 REPLY_PITCH  rw  bytes per destination row
 *   0x58 REPLY_ROW_BYTES rw payload bytes per source row (width * 4)
 *   0x5c REPLY_FLAGS  rw  bit0: byte-swap each 32-bit pixel (BGRA8 LE source
 *                          -> big-endian xRGB, as in the r300 path);
 *                          bit1: source rows are bottom-up (flip vertically);
 *                          bit2: convert each BGRA8 LE pixel to big-endian
 *                          RGB555 (0RRRRRGGGGGBBBBB); bit0 is then ignored,
 *                          the destination row is ROW_BYTES / 2 bytes (dest
 *                          and pitch need only 2-byte alignment) and the
 *                          bounds checks use that size
 *   0x60 STREAM_RESET wo  any value: stop the worker, close the executor
 *                          stream (the host frees that client's contexts),
 *                          reset ring head/tail, reply state and GL_STATUS.
 *                          The stream reopens lazily on the next doorbell.
 *                          A guest GL process writes it before its first
 *                          DG_HELLO (the host rejects a second HELLO on one
 *                          stream). One GL app at a time.
 * RING_HEAD/TAIL are byte indexes; a doorbell handles the wrap.
 *
 * Present (E4): with REPLY_FLAGS bit3 (PRESENT) and REPLY_EXPECT > 0 the doorbell
 * streams the reply row by row into VRAM + DEST + row * PITCH instead of the
 * reply area (REPLY_LEN then counts payload bytes written). EXPECT must be a
 * multiple of ROW_BYTES; ROW_BYTES must be a multiple of 4 and <= PITCH and
 * <= 64 KiB; the rectangle must end at or below RING_OFFSET. Any violation
 * sets GL_STATUS PVMAC_GL_ERR_DEST, the commands still run, and the reply is
 * read and dropped. The 16-byte DGReply header read needs bit3 clear.
 * The written span is marked dirty so the display refreshes.
 *
 * Threading (E5): a worker thread owns writes to the executor. A doorbell
 * with REPLY_EXPECT == 0 copies TAIL..HEAD into a queue, advances TAIL and
 * returns. A doorbell with REPLY_EXPECT > 0 first waits (BQL held, the
 * worker never takes the BQL) until the queue is drained, then writes and
 * reads synchronously. TODO: drop the BQL during that wait.
 */
#define PVMAC_OFFSET            0x800
#define PVMAC_SIZE              0x800
#define PVMAC_ID_VALUE          0x50564d43  /* 'PVMC' */
#define PVMAC_VERSION_VALUE     2
#define PVMAC_CAP_HWCURSOR      (1u << 0)
#define PVMAC_CURSOR_SLOT_SIZE  0x4000      /* 64 * 64 * 4, top of VRAM */
#define PVMAC_CAP_GL_STREAM     (1u << 1)
#define PVMAC_GL_RING_SIZE      (1 * MiB)
#define PVMAC_GL_REPLY_SIZE     (4 * MiB)
#define PVMAC_GL_MIN_VRAM       (16 * MiB)  /* leave room for a framebuffer */
#define PVMAC_GL_MAX_ROW_BYTES  (64 * KiB)
#define PVMAC_GL_QUEUE_MAX      (32 * MiB)  /* backpressure on the vCPU */
#define PVMAC_REPLY_FLAG_SWAP32 (1u << 0)
#define PVMAC_REPLY_FLAG_FLIP   (1u << 1)
#define PVMAC_REPLY_FLAG_RGB555 (1u << 2)
#define PVMAC_REPLY_FLAG_PRESENT (1u << 3) /* write the reply rows at REPLY_DEST (0 is a valid offset) */

enum {
    PVMAC_REG_ID         = 0x00,
    PVMAC_REG_VERSION    = 0x04,
    PVMAC_REG_CAPS       = 0x08,
    PVMAC_REG_STATUS     = 0x0c,
    PVMAC_REG_CUR_CTRL   = 0x10,
    PVMAC_REG_CUR_POS    = 0x14,
    PVMAC_REG_CUR_HOT    = 0x18,
    PVMAC_REG_CUR_SIZE   = 0x1c,
    PVMAC_REG_CUR_DEFINE = 0x20,
    PVMAC_REG_CUR_OFFSET = 0x24,
    PVMAC_REG_RING_OFFSET   = 0x28,
    PVMAC_REG_RING_SIZE     = 0x2c,
    PVMAC_REG_RING_HEAD     = 0x30,
    PVMAC_REG_RING_TAIL     = 0x34,
    PVMAC_REG_REPLY_OFFSET  = 0x38,
    PVMAC_REG_REPLY_SIZE    = 0x3c,
    PVMAC_REG_REPLY_LEN     = 0x40,
    PVMAC_REG_DOORBELL      = 0x44,
    PVMAC_REG_GL_STATUS     = 0x48,
    PVMAC_REG_REPLY_EXPECT  = 0x4c,
    PVMAC_REG_REPLY_DEST    = 0x50,
    PVMAC_REG_REPLY_PITCH   = 0x54,
    PVMAC_REG_REPLY_ROW_BYTES = 0x58,
    PVMAC_REG_REPLY_FLAGS   = 0x5c,
    PVMAC_REG_STREAM_RESET  = 0x60,
};

#define PVMAC_GL_ERR_BAD_HEAD   (1u << 0)
#define PVMAC_GL_ERR_BAD_EXPECT (1u << 1)
#define PVMAC_GL_ERR_IO         (1u << 2)   /* write to the executor failed */
#define PVMAC_GL_ERR_REPLY      (1u << 3)   /* reply short or read failed */
#define PVMAC_GL_ERR_OPEN       (1u << 4)   /* executor endpoint won't open */
#define PVMAC_GL_ERR_NO_GL      (1u << 5)   /* doorbell without CAPS bit 1 */
#define PVMAC_GL_ERR_DEST       (1u << 6)   /* bad REPLY_DEST rectangle */

#define PVMAC_CUR_CTRL_VISIBLE  (1u << 0)
#define PVMAC_STATUS_ERROR      (1u << 1)
#define PVMAC_ERR_BAD_SIZE      1
#define PVMAC_ERR_BAD_HOTSPOT   2

typedef struct PVMACState {
    uint32_t status;
    uint32_t cur_ctrl;
    uint32_t cur_pos;
    uint32_t cur_hot;
    uint32_t cur_size;
    bool cur_defined;
    /* v2 GL transport (migrated in the vga/mac-accel-gl subsection) */
    uint32_t ring_head;
    uint32_t ring_tail;
    uint32_t reply_len;
    uint32_t reply_expect;
    uint32_t gl_status;
    /* present destination: set before each present, not migrated */
    uint32_t reply_dest;
    uint32_t reply_pitch;
    uint32_t reply_row_bytes;
    uint32_t reply_flags;
} PVMACState;

#ifdef __APPLE__
typedef struct PVMACGLChunk {
    uint8_t *data;
    uint32_t len;
} PVMACGLChunk;

typedef struct PVMACGLWorker {
    QemuThread thread;
    QemuMutex lock;
    QemuCond work_cond;         /* worker waits for chunks */
    QemuCond idle_cond;         /* vCPU waits for drain */
    GQueue queue;               /* of PVMACGLChunk * */
    uint64_t queued_bytes;      /* queued + in flight */
    bool busy;
    bool quit;
    bool running;
} PVMACGLWorker;
#endif

struct PCIVGAState {
    PCIDevice dev;
    VGACommonState vga;
    uint32_t flags;
    qemu_edid_info edid_info;
    MemoryRegion mmio;
    MemoryRegion mrs[5];
    uint8_t edid[384];
    PVMACState pvmac;
    bool gl_available;          /* executor library loaded; not migrated */
#ifdef __APPLE__
    struct VIMPassthroughOpenGLStream gl_stream;  /* opened on first doorbell */
    PVMACGLWorker gl_worker;                      /* started on first doorbell */
#endif
};

#define TYPE_PCI_VGA "pci-vga"
OBJECT_DECLARE_SIMPLE_TYPE(PCIVGAState, PCI_VGA)

static void pvmac_cursor_define(PCIVGAState *d);

static bool vga_pci_mac_accel_needed(void *opaque)
{
    PCIVGAState *d = opaque;

    return d->flags & (1 << PCI_VGA_FLAG_ENABLE_MAC_ACCEL);
}

static int vga_pci_mac_accel_post_load(void *opaque, int version_id)
{
    PCIVGAState *d = opaque;

    if (d->pvmac.cur_defined) {
        pvmac_cursor_define(d);
    }
    return 0;
}

static const VMStateDescription vmstate_vga_pci_mac_accel = {
    .name = "vga/mac-accel",
    .version_id = 1,
    .minimum_version_id = 1,
    .needed = vga_pci_mac_accel_needed,
    .post_load = vga_pci_mac_accel_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(pvmac.status, PCIVGAState),
        VMSTATE_UINT32(pvmac.cur_ctrl, PCIVGAState),
        VMSTATE_UINT32(pvmac.cur_pos, PCIVGAState),
        VMSTATE_UINT32(pvmac.cur_hot, PCIVGAState),
        VMSTATE_UINT32(pvmac.cur_size, PCIVGAState),
        VMSTATE_BOOL(pvmac.cur_defined, PCIVGAState),
        VMSTATE_END_OF_LIST()
    }
};

static bool vga_pci_mac_accel_gl_needed(void *opaque)
{
    PCIVGAState *d = opaque;

    return vga_pci_mac_accel_needed(d) && d->gl_available;
}

static int vga_pci_mac_accel_gl_post_load(void *opaque, int version_id)
{
    PCIVGAState *d = opaque;

    /* The executor endpoint is not migrated; never trust incoming indexes */
    if (d->pvmac.ring_head >= PVMAC_GL_RING_SIZE ||
        d->pvmac.ring_tail >= PVMAC_GL_RING_SIZE) {
        d->pvmac.ring_head = d->pvmac.ring_tail = 0;
    }
    if (d->pvmac.reply_len > PVMAC_GL_REPLY_SIZE) {
        d->pvmac.reply_len = 0;
    }
    if (d->pvmac.reply_expect > PVMAC_GL_REPLY_SIZE) {
        d->pvmac.reply_expect = 0;
    }
    return 0;
}

static void pvmac_gl_flush(PCIVGAState *d);

static int vga_pci_mac_accel_gl_pre_save(void *opaque)
{
    /* Queued spans are not migrated: drain them so TAIL is truthful */
    pvmac_gl_flush(opaque);
    return 0;
}

static const VMStateDescription vmstate_vga_pci_mac_accel_gl = {
    .name = "vga/mac-accel-gl",
    .version_id = 1,
    .minimum_version_id = 1,
    .needed = vga_pci_mac_accel_gl_needed,
    .pre_save = vga_pci_mac_accel_gl_pre_save,
    .post_load = vga_pci_mac_accel_gl_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(pvmac.ring_head, PCIVGAState),
        VMSTATE_UINT32(pvmac.ring_tail, PCIVGAState),
        VMSTATE_UINT32(pvmac.reply_len, PCIVGAState),
        VMSTATE_UINT32(pvmac.reply_expect, PCIVGAState),
        VMSTATE_UINT32(pvmac.gl_status, PCIVGAState),
        VMSTATE_END_OF_LIST()
    }
};

static const VMStateDescription vmstate_vga_pci = {
    .name = "vga",
    .version_id = 2,
    .minimum_version_id = 2,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(dev, PCIVGAState),
        VMSTATE_STRUCT(vga, PCIVGAState, 0, vmstate_vga_common, VGACommonState),
        VMSTATE_END_OF_LIST()
    },
    .subsections = (const VMStateDescription * const []) {
        &vmstate_vga_pci_mac_accel,
        &vmstate_vga_pci_mac_accel_gl,
        NULL
    }
};

static uint64_t pci_vga_ioport_read(void *ptr, hwaddr addr,
                                    unsigned size)
{
    VGACommonState *s = ptr;
    uint64_t ret = 0;

    switch (size) {
    case 1:
        ret = vga_ioport_read(s, addr + 0x3c0);
        break;
    case 2:
        ret  = vga_ioport_read(s, addr + 0x3c0);
        ret |= vga_ioport_read(s, addr + 0x3c1) << 8;
        break;
    }
    return ret;
}

static void pci_vga_ioport_write(void *ptr, hwaddr addr,
                                 uint64_t val, unsigned size)
{
    VGACommonState *s = ptr;

    switch (size) {
    case 1:
        vga_ioport_write(s, addr + 0x3c0, val);
        break;
    case 2:
        /*
         * Update bytes in little endian order.  Allows to update
         * indexed registers with a single word write because the
         * index byte is updated first.
         */
        vga_ioport_write(s, addr + 0x3c0, val & 0xff);
        vga_ioport_write(s, addr + 0x3c1, (val >> 8) & 0xff);
        break;
    }
}

static const MemoryRegionOps pci_vga_ioport_ops = {
    .read = pci_vga_ioport_read,
    .write = pci_vga_ioport_write,
    .valid.min_access_size = 1,
    .valid.max_access_size = 4,
    .impl.min_access_size = 1,
    .impl.max_access_size = 2,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static uint64_t pci_vga_bochs_read(void *ptr, hwaddr addr,
                                   unsigned size)
{
    VGACommonState *s = ptr;
    int index = addr >> 1;

    vbe_ioport_write_index(s, 0, index);
    return vbe_ioport_read_data(s, 0);
}

static void pci_vga_bochs_write(void *ptr, hwaddr addr,
                                uint64_t val, unsigned size)
{
    VGACommonState *s = ptr;
    int index = addr >> 1;

    vbe_ioport_write_index(s, 0, index);
    vbe_ioport_write_data(s, 0, val);
}

static const MemoryRegionOps pci_vga_bochs_ops = {
    .read = pci_vga_bochs_read,
    .write = pci_vga_bochs_write,
    .valid.min_access_size = 1,
    .valid.max_access_size = 4,
    .impl.min_access_size = 2,
    .impl.max_access_size = 2,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static uint64_t pci_vga_qext_read(void *ptr, hwaddr addr, unsigned size)
{
    VGACommonState *s = ptr;

    switch (addr) {
    case PCI_VGA_QEXT_REG_SIZE:
        return PCI_VGA_QEXT_SIZE;
    case PCI_VGA_QEXT_REG_BYTEORDER:
        return s->big_endian_fb ?
            PCI_VGA_QEXT_BIG_ENDIAN : PCI_VGA_QEXT_LITTLE_ENDIAN;
    default:
        return 0;
    }
}

static void pci_vga_qext_write(void *ptr, hwaddr addr,
                               uint64_t val, unsigned size)
{
    VGACommonState *s = ptr;

    switch (addr) {
    case PCI_VGA_QEXT_REG_BYTEORDER:
        if (val == PCI_VGA_QEXT_BIG_ENDIAN) {
            s->big_endian_fb = true;
        }
        if (val == PCI_VGA_QEXT_LITTLE_ENDIAN) {
            s->big_endian_fb = false;
        }
        break;
    }
}

static bool vga_get_big_endian_fb(Object *obj, Error **errp)
{
    PCIVGAState *d = PCI_VGA(PCI_DEVICE(obj));

    return d->vga.big_endian_fb;
}

static void vga_set_big_endian_fb(Object *obj, bool value, Error **errp)
{
    PCIVGAState *d = PCI_VGA(PCI_DEVICE(obj));

    d->vga.big_endian_fb = value;
}

static const MemoryRegionOps pci_vga_qext_ops = {
    .read = pci_vga_qext_read,
    .write = pci_vga_qext_write,
    .valid.min_access_size = 4,
    .valid.max_access_size = 4,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

/* ------------------------------------------------------------------ PVMAC */

static uint32_t pvmac_cursor_offset(PCIVGAState *d)
{
    return d->vga.vram_size - PVMAC_CURSOR_SLOT_SIZE;
}

static uint32_t pvmac_reply_offset(PCIVGAState *d)
{
    return pvmac_cursor_offset(d) - PVMAC_GL_REPLY_SIZE;
}

static uint32_t pvmac_ring_offset(PCIVGAState *d)
{
    return pvmac_reply_offset(d) - PVMAC_GL_RING_SIZE;
}

#ifdef __APPLE__
/* Write all of buf to the executor; false on error (the data is dropped). */
static bool pvmac_gl_write_all(PCIVGAState *d, const uint8_t *buf, size_t len)
{
    while (len) {
        ssize_t n = vmsvga_gl_stream_write(d->gl_stream, buf, len);

        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        buf += n;
        len -= n;
    }
    return true;
}

/*
 * Worker thread: the only writer to the executor while chunks are queued.
 * It never takes the BQL. gl_stream is only replaced by the vCPU thread
 * while the queue is empty and the worker idle (open before the first
 * enqueue, close after the worker has been joined).
 */
static void *pvmac_gl_worker_fn(void *opaque)
{
    PCIVGAState *d = opaque;
    PVMACGLWorker *w = &d->gl_worker;

    qemu_mutex_lock(&w->lock);
    for (;;) {
        PVMACGLChunk *c;
        bool ok;

        while (g_queue_is_empty(&w->queue) && !w->quit) {
            qemu_cond_wait(&w->work_cond, &w->lock);
        }
        if (w->quit) {
            break;
        }
        c = g_queue_pop_head(&w->queue);
        w->busy = true;
        qemu_mutex_unlock(&w->lock);

        ok = pvmac_gl_write_all(d, c->data, c->len);
        if (!ok) {
            qatomic_or(&d->pvmac.gl_status, PVMAC_GL_ERR_IO);
        }

        qemu_mutex_lock(&w->lock);
        w->busy = false;
        w->queued_bytes -= c->len;
        g_free(c->data);
        g_free(c);
        qemu_cond_broadcast(&w->idle_cond);
    }
    qemu_mutex_unlock(&w->lock);
    return NULL;
}

static void pvmac_gl_worker_start(PCIVGAState *d)
{
    PVMACGLWorker *w = &d->gl_worker;

    if (w->running) {
        return;
    }
    qemu_mutex_init(&w->lock);
    qemu_cond_init(&w->work_cond);
    qemu_cond_init(&w->idle_cond);
    g_queue_init(&w->queue);
    w->queued_bytes = 0;
    w->busy = false;
    w->quit = false;
    qemu_thread_create(&w->thread, "pvmac-gl", pvmac_gl_worker_fn, d,
                       QEMU_THREAD_JOINABLE);
    w->running = true;
}

/* Wait (BQL held; TODO: unlock it) until everything queued is written. */
static void pvmac_gl_flush(PCIVGAState *d)
{
    PVMACGLWorker *w = &d->gl_worker;

    if (!w->running) {
        return;
    }
    qemu_mutex_lock(&w->lock);
    trace_vga_pvmac_gl_flush(w->queued_bytes);
    while (!g_queue_is_empty(&w->queue) || w->busy) {
        qemu_cond_wait(&w->idle_cond, &w->lock);
    }
    qemu_mutex_unlock(&w->lock);
}

/* Stop and join the worker (dropping queued spans), close the stream. */
static void pvmac_gl_close(PCIVGAState *d)
{
    PVMACGLWorker *w = &d->gl_worker;

    if (w->running) {
        PVMACGLChunk *c;

        qemu_mutex_lock(&w->lock);
        w->quit = true;
        qemu_cond_broadcast(&w->work_cond);
        qemu_mutex_unlock(&w->lock);
        qemu_thread_join(&w->thread);
        while ((c = g_queue_pop_head(&w->queue))) {
            g_free(c->data);
            g_free(c);
        }
        qemu_cond_destroy(&w->work_cond);
        qemu_cond_destroy(&w->idle_cond);
        qemu_mutex_destroy(&w->lock);
        w->running = false;
    }
    if (vmsvga_gl_stream_valid(d->gl_stream)) {
        vmsvga_gl_stream_shutdown(d->gl_stream);
        vmsvga_gl_stream_close(d->gl_stream);
    }
    d->gl_stream = (struct VIMPassthroughOpenGLStream){ .fd = -1 };
}

/* Copy TAIL..HEAD out of the ring into the worker queue. */
static void pvmac_gl_enqueue(PCIVGAState *d, uint32_t tail, uint32_t pending)
{
    PVMACGLWorker *w = &d->gl_worker;
    const uint8_t *ring = d->vga.vram_ptr + pvmac_ring_offset(d);
    uint32_t first = MIN(pending, PVMAC_GL_RING_SIZE - tail);
    PVMACGLChunk *c = g_new(PVMACGLChunk, 1);

    c->data = g_malloc(pending);
    c->len = pending;
    memcpy(c->data, ring + tail, first);
    if (pending > first) {
        memcpy(c->data + first, ring, pending - first);
    }
    qemu_mutex_lock(&w->lock);
    while (w->queued_bytes > PVMAC_GL_QUEUE_MAX) {
        qemu_cond_wait(&w->idle_cond, &w->lock);
    }
    g_queue_push_tail(&w->queue, c);
    w->queued_bytes += pending;
    trace_vga_pvmac_gl_enqueue(pending, w->queued_bytes);
    qemu_cond_signal(&w->work_cond);
    qemu_mutex_unlock(&w->lock);
}

/*
 * Check the REPLY_DEST rectangle for a reply of "expect" bytes; on success
 * return the row count in *rows. Every row must end at or below the ring
 * offset (the reserved area starts there).
 */
static bool pvmac_present_valid(PCIVGAState *d, uint32_t expect,
                                uint32_t *rows)
{
    PVMACState *p = &d->pvmac;
    uint64_t limit = pvmac_ring_offset(d);
    bool rgb555 = p->reply_flags & PVMAC_REPLY_FLAG_RGB555;
    uint32_t align = rgb555 ? 1 : 3;
    uint32_t dst_rb = rgb555 ? p->reply_row_bytes / 2 : p->reply_row_bytes;
    uint64_t end;

    if (!p->reply_row_bytes || p->reply_row_bytes > PVMAC_GL_MAX_ROW_BYTES ||
        (p->reply_row_bytes & 3) || p->reply_pitch < dst_rb ||
        (p->reply_dest & align) || (p->reply_pitch & align) ||
        expect % p->reply_row_bytes) {
        return false;
    }
    *rows = expect / p->reply_row_bytes;
    end = (uint64_t)p->reply_dest +
          (uint64_t)(*rows - 1) * p->reply_pitch + dst_rb;
    return end <= limit;
}

/* Stream the reply row by row into the VRAM rectangle (E4). */
static void pvmac_gl_present(PCIVGAState *d, uint32_t expect)
{
    PVMACState *p = &d->pvmac;
    uint8_t *vram = d->vga.vram_ptr;
    uint32_t rb = p->reply_row_bytes, rows = 0, row;
    uint8_t *tmp;
    bool valid = pvmac_present_valid(d, expect, &rows);
    uint32_t done = 0;
    bool rgb555 = p->reply_flags & PVMAC_REPLY_FLAG_RGB555;
    uint32_t dst_rb = rgb555 ? rb / 2 : rb;

    trace_vga_pvmac_gl_present(p->reply_dest, p->reply_pitch, rb, rows,
                               p->reply_flags, valid);
    if (!valid) {
        /* consume and drop the reply so the stream stays in sync */
        p->gl_status |= PVMAC_GL_ERR_DEST;
        VIMGLInprocessReadExpected(vim_passthrough_opengl_inprocess,
                                   d->gl_stream.memory,
                                   vram + pvmac_reply_offset(d), expect);
        return;
    }
    tmp = g_malloc(rb);
    for (row = 0; row < rows; row++) {
        uint32_t dst_row = (p->reply_flags & PVMAC_REPLY_FLAG_FLIP) ?
                           rows - 1 - row : row;
        uint8_t *dst = vram + p->reply_dest + (uint64_t)dst_row * p->reply_pitch;
        ssize_t n = VIMGLInprocessReadExpected(vim_passthrough_opengl_inprocess,
                                               d->gl_stream.memory, tmp, rb);

        if (n != rb) {
            p->gl_status |= PVMAC_GL_ERR_REPLY;
            break;
        }
        if (rgb555) {
            const uint32_t *s32 = (const uint32_t *)tmp;
            uint32_t i;

            for (i = 0; i < rb / 4; i++) {
                uint32_t px = le32_to_cpu(s32[i]);   /* 0xAARRGGBB */
                uint16_t v = ((px >> 9) & 0x7c00) | ((px >> 6) & 0x03e0) |
                             ((px >> 3) & 0x001f);

                stw_be_p(dst + i * 2, v);
            }
        } else if (p->reply_flags & PVMAC_REPLY_FLAG_SWAP32) {
            uint32_t *s32 = (uint32_t *)tmp;   /* g_malloc: aligned */
            uint32_t *d32 = (uint32_t *)dst;   /* dest and pitch are 4-aligned */
            uint32_t i;

            for (i = 0; i < rb / 4; i++) {
                d32[i] = bswap32(s32[i]);
            }
        } else {
            memcpy(dst, tmp, rb);
        }
        done += rb;
    }
    g_free(tmp);
    p->reply_len = done;
    memory_region_set_dirty(&d->vga.vram, p->reply_dest,
                            (uint64_t)(rows - 1) * p->reply_pitch + dst_rb);
}

static void pvmac_gl_doorbell(PCIVGAState *d)
{
    PVMACState *p = &d->pvmac;
    uint8_t *vram = d->vga.vram_ptr;
    uint32_t head = p->ring_head, tail = p->ring_tail;
    uint32_t expect = p->reply_expect;
    uint32_t pending;
    bool ok = true;

    p->reply_len = 0;
    if (!d->gl_available) {
        p->gl_status |= PVMAC_GL_ERR_NO_GL;
        return;
    }
    /* Indexes are only ever stored validated, but check again: cheap. */
    if (head >= PVMAC_GL_RING_SIZE || tail >= PVMAC_GL_RING_SIZE) {
        p->gl_status |= PVMAC_GL_ERR_BAD_HEAD;
        p->ring_tail = p->ring_head = 0;
        return;
    }
    if (expect > PVMAC_GL_REPLY_SIZE) {
        p->gl_status |= PVMAC_GL_ERR_BAD_EXPECT;
        return;
    }
    if (!vmsvga_gl_stream_valid(d->gl_stream)) {
        d->gl_stream = vmsvga_gl_stream_open(0);
        if (!vmsvga_gl_stream_valid(d->gl_stream)) {
            d->gl_stream = (struct VIMPassthroughOpenGLStream){ .fd = -1 };
            p->gl_status |= PVMAC_GL_ERR_OPEN;
            p->ring_tail = head;      /* drop, never replay */
            return;
        }
    }
    pvmac_gl_worker_start(d);

    pending = (head - tail) & (PVMAC_GL_RING_SIZE - 1);
    trace_vga_pvmac_gl_doorbell(tail, head, pending, expect);

    if (!expect) {
        /* asynchronous: hand the span to the worker, free the ring now */
        if (pending) {
            pvmac_gl_enqueue(d, tail, pending);
        }
        p->ring_tail = head;
        trace_vga_pvmac_gl_done(p->reply_len, p->gl_status);
        return;
    }

    /* sync point: everything earlier must be in the executor first */
    pvmac_gl_flush(d);
    if (pending) {
        const uint8_t *ring = vram + pvmac_ring_offset(d);
        uint32_t first = MIN(pending, PVMAC_GL_RING_SIZE - tail);

        ok = pvmac_gl_write_all(d, ring + tail, first);
        if (ok && pending > first) {
            ok = pvmac_gl_write_all(d, ring, pending - first);
        }
        if (!ok) {
            p->gl_status |= PVMAC_GL_ERR_IO;
        }
    }
    p->ring_tail = head;

    if (ok) {
        if (p->reply_flags & PVMAC_REPLY_FLAG_PRESENT) {
            pvmac_gl_present(d, expect);
        } else {
            ssize_t n = VIMGLInprocessReadExpected(
                            vim_passthrough_opengl_inprocess,
                            d->gl_stream.memory,
                            vram + pvmac_reply_offset(d), expect);

            if (n < 0) {
                n = 0;
            }
            p->reply_len = n;
            if (n != expect) {
                p->gl_status |= PVMAC_GL_ERR_REPLY;
            }
        }
    }
    trace_vga_pvmac_gl_done(p->reply_len, p->gl_status);
}

/* Load the executor library once at realize; never abort when absent. */
static void pvmac_gl_probe(PCIVGAState *d)
{
    const char *path = getenv("VIM_PASSTHROUGH_OPENGL_INPROCESS_LIBRARY");

    d->gl_stream = (struct VIMPassthroughOpenGLStream){ .fd = -1 };
    if (d->vga.vram_size < PVMAC_GL_MIN_VRAM) {
        warn_report("vga: mac-accel GL stream disabled: vgamem_mb too small");
        return;
    }
    if (!path) {
        warn_report("vga: mac-accel GL stream disabled: "
                    "VIM_PASSTHROUGH_OPENGL_INPROCESS_LIBRARY not set");
        return;
    }
    vmsvga_gl_load_inprocess(path);
    if (!vim_passthrough_opengl_inprocess) {
        warn_report("vga: mac-accel GL stream disabled: "
                    "executor library failed to load");
        return;
    }
    d->gl_available = true;
}
#else
static void pvmac_gl_close(PCIVGAState *d)
{
}

static void pvmac_gl_flush(PCIVGAState *d)
{
}

static void pvmac_gl_doorbell(PCIVGAState *d)
{
    d->pvmac.reply_len = 0;
    d->pvmac.gl_status |= PVMAC_GL_ERR_NO_GL;
}

static void pvmac_gl_probe(PCIVGAState *d)
{
    warn_report("vga: mac-accel GL stream needs macOS; disabled");
}
#endif

/* STREAM_RESET: drop the executor stream and the ring/reply state. */
static void pvmac_stream_reset(PCIVGAState *d)
{
    PVMACState *p = &d->pvmac;

    trace_vga_pvmac_stream_reset();
    pvmac_gl_flush(d);
    pvmac_gl_close(d);
    p->ring_head = p->ring_tail = 0;
    p->reply_len = 0;
    p->reply_expect = 0;
    p->reply_dest = 0;
    p->reply_pitch = 0;
    p->reply_row_bytes = 0;
    p->reply_flags = 0;
    p->gl_status = 0;
}

static void pvmac_error(PCIVGAState *d, uint32_t code)
{
    d->pvmac.status = PVMAC_STATUS_ERROR | (code << 8);
}

static void pvmac_mouse_set(PCIVGAState *d)
{
    int x = (int16_t)(d->pvmac.cur_pos & 0xffff);
    int y = (int16_t)(d->pvmac.cur_pos >> 16);
    bool visible = d->pvmac.cur_defined &&
                   (d->pvmac.cur_ctrl & PVMAC_CUR_CTRL_VISIBLE);

    trace_vga_pvmac_cursor_pos(x, y, visible);
    dpy_mouse_set(d->vga.con, x, y, visible);
}

/*
 * Build a host cursor from the image the guest left in the cursor slot at the
 * top of VRAM: size x size pixels, 32-bit ARGB, stored big-endian (A, R, G, B
 * byte order), which is the natural layout for a PowerPC guest.
 */
static void pvmac_cursor_define(PCIVGAState *d)
{
    uint32_t size = d->pvmac.cur_size;
    int hot_x = d->pvmac.cur_hot & 0xffff;
    int hot_y = d->pvmac.cur_hot >> 16;
    const uint8_t *src;
    QEMUCursor *c;
    uint32_t i;

    if (size != 32 && size != 64) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vga: mac-accel cursor size %u not supported\n", size);
        pvmac_error(d, PVMAC_ERR_BAD_SIZE);
        return;
    }
    if (hot_x >= size || hot_y >= size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vga: mac-accel cursor hotspot %d,%d outside %ux%u\n",
                      hot_x, hot_y, size, size);
        pvmac_error(d, PVMAC_ERR_BAD_HOTSPOT);
        return;
    }

    src = d->vga.vram_ptr + pvmac_cursor_offset(d);
    c = cursor_alloc(size, size);
    c->hot_x = hot_x;
    c->hot_y = hot_y;
    for (i = 0; i < size * size; i++) {
        c->data[i] = ldl_be_p(src + i * 4);   /* 0xAARRGGBB */
    }
    d->pvmac.cur_defined = true;
    d->pvmac.status = 0;
    trace_vga_pvmac_cursor_define(size, hot_x, hot_y);
    dpy_cursor_define(d->vga.con, c);
    cursor_unref(c);
    pvmac_mouse_set(d);
}

static uint64_t pvmac_read(void *opaque, hwaddr addr, unsigned size)
{
    PCIVGAState *d = opaque;

    switch (addr) {
    case PVMAC_REG_ID:
        return PVMAC_ID_VALUE;
    case PVMAC_REG_VERSION:
        return PVMAC_VERSION_VALUE;
    case PVMAC_REG_CAPS:
        return PVMAC_CAP_HWCURSOR |
               (d->gl_available ? PVMAC_CAP_GL_STREAM : 0);
    case PVMAC_REG_STATUS:
        return d->pvmac.status;
    case PVMAC_REG_CUR_CTRL:
        return d->pvmac.cur_ctrl;
    case PVMAC_REG_CUR_POS:
        return d->pvmac.cur_pos;
    case PVMAC_REG_CUR_HOT:
        return d->pvmac.cur_hot;
    case PVMAC_REG_CUR_SIZE:
        return d->pvmac.cur_size;
    case PVMAC_REG_CUR_OFFSET:
        return pvmac_cursor_offset(d);
    case PVMAC_REG_RING_OFFSET:
        return d->gl_available ? pvmac_ring_offset(d) : 0;
    case PVMAC_REG_RING_SIZE:
        return d->gl_available ? PVMAC_GL_RING_SIZE : 0;
    case PVMAC_REG_RING_HEAD:
        return d->pvmac.ring_head;
    case PVMAC_REG_RING_TAIL:
        return d->pvmac.ring_tail;
    case PVMAC_REG_REPLY_OFFSET:
        return d->gl_available ? pvmac_reply_offset(d) : 0;
    case PVMAC_REG_REPLY_SIZE:
        return d->gl_available ? PVMAC_GL_REPLY_SIZE : 0;
    case PVMAC_REG_REPLY_LEN:
        return d->pvmac.reply_len;
    case PVMAC_REG_GL_STATUS:
        return d->pvmac.gl_status;
    case PVMAC_REG_REPLY_EXPECT:
        return d->pvmac.reply_expect;
    case PVMAC_REG_REPLY_DEST:
        return d->pvmac.reply_dest;
    case PVMAC_REG_REPLY_PITCH:
        return d->pvmac.reply_pitch;
    case PVMAC_REG_REPLY_ROW_BYTES:
        return d->pvmac.reply_row_bytes;
    case PVMAC_REG_REPLY_FLAGS:
        return d->pvmac.reply_flags;
    default:
        return 0;
    }
}

static void pvmac_write(void *opaque, hwaddr addr, uint64_t val, unsigned size)
{
    PCIVGAState *d = opaque;

    switch (addr) {
    case PVMAC_REG_CUR_CTRL:
        d->pvmac.cur_ctrl = val & PVMAC_CUR_CTRL_VISIBLE;
        pvmac_mouse_set(d);
        break;
    case PVMAC_REG_CUR_POS:
        d->pvmac.cur_pos = val;
        pvmac_mouse_set(d);
        break;
    case PVMAC_REG_CUR_HOT:
        d->pvmac.cur_hot = val;
        break;
    case PVMAC_REG_CUR_SIZE:
        d->pvmac.cur_size = val;
        break;
    case PVMAC_REG_CUR_DEFINE:
        pvmac_cursor_define(d);
        break;
    case PVMAC_REG_RING_HEAD:
        if (val >= PVMAC_GL_RING_SIZE) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "vga: mac-accel ring head 0x%" PRIx64 " out of range\n",
                          (uint64_t)val);
            d->pvmac.gl_status |= PVMAC_GL_ERR_BAD_HEAD;
        } else {
            d->pvmac.ring_head = val;
        }
        break;
    case PVMAC_REG_REPLY_EXPECT:
        if (val > PVMAC_GL_REPLY_SIZE) {
            d->pvmac.gl_status |= PVMAC_GL_ERR_BAD_EXPECT;
        } else {
            d->pvmac.reply_expect = val;
        }
        break;
    case PVMAC_REG_REPLY_DEST:
        d->pvmac.reply_dest = val;
        break;
    case PVMAC_REG_REPLY_PITCH:
        d->pvmac.reply_pitch = val;
        break;
    case PVMAC_REG_REPLY_ROW_BYTES:
        d->pvmac.reply_row_bytes = val;
        break;
    case PVMAC_REG_REPLY_FLAGS:
        d->pvmac.reply_flags = val & (PVMAC_REPLY_FLAG_SWAP32 |
                                      PVMAC_REPLY_FLAG_FLIP |
                                      PVMAC_REPLY_FLAG_RGB555 |
                                      PVMAC_REPLY_FLAG_PRESENT);
        break;
    case PVMAC_REG_STREAM_RESET:
        pvmac_stream_reset(d);
        break;
    case PVMAC_REG_DOORBELL:
        pvmac_gl_doorbell(d);
        break;
    case PVMAC_REG_GL_STATUS:
        d->pvmac.gl_status &= ~(uint32_t)val;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vga: mac-accel write to read-only/unknown register "
                      "0x%" HWADDR_PRIx "\n", addr);
        break;
    }
}

static const MemoryRegionOps pvmac_ops = {
    .read = pvmac_read,
    .write = pvmac_write,
    .valid.min_access_size = 4,
    .valid.max_access_size = 4,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void pvmac_reset(void *opaque)
{
    PCIVGAState *d = opaque;

    pvmac_gl_close(d);
    memset(&d->pvmac, 0, sizeof(d->pvmac));
}

void pci_std_vga_mmio_region_init(VGACommonState *s,
                                  Object *owner,
                                  MemoryRegion *parent,
                                  MemoryRegion *subs,
                                  bool qext, bool edid)
{
    PCIVGAState *d = container_of(s, PCIVGAState, vga);

    memory_region_init_io(&subs[0], owner, &pci_vga_ioport_ops, s,
                          "vga ioports remapped", PCI_VGA_IOPORT_SIZE);
    memory_region_add_subregion(parent, PCI_VGA_IOPORT_OFFSET,
                                &subs[0]);

    memory_region_init_io(&subs[1], owner, &pci_vga_bochs_ops, s,
                          "bochs dispi interface", PCI_VGA_BOCHS_SIZE);
    memory_region_add_subregion(parent, PCI_VGA_BOCHS_OFFSET,
                                &subs[1]);

    if (qext) {
        memory_region_init_io(&subs[2], owner, &pci_vga_qext_ops, s,
                              "qemu extended regs", PCI_VGA_QEXT_SIZE);
        memory_region_add_subregion(parent, PCI_VGA_QEXT_OFFSET,
                                    &subs[2]);
    }

    if (edid) {
        qemu_edid_generate(d->edid, sizeof(d->edid), &d->edid_info);
        qemu_edid_region_io(&subs[3], owner, d->edid, sizeof(d->edid));
        memory_region_add_subregion(parent, 0, &subs[3]);
    }
}

static void pci_std_vga_realize(PCIDevice *dev, Error **errp)
{
    PCIVGAState *d = PCI_VGA(dev);
    VGACommonState *s = &d->vga;
    bool qext = false;
    bool edid = false;

    /* vga + console init */
    if (!vga_common_init(s, OBJECT(dev), errp)) {
        return;
    }
    vga_init(s, OBJECT(dev), pci_address_space(dev), pci_address_space_io(dev),
             true);

    s->con = graphic_console_init(DEVICE(dev), 0, s->hw_ops, s);

    /* XXX: VGA_RAM_SIZE must be a power of two */
    pci_register_bar(&d->dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram);

    /* mmio bar for vga register access */
    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_MMIO)) {
        memory_region_init_io(&d->mmio, OBJECT(dev), &unassigned_io_ops, NULL,
                              "vga.mmio", PCI_VGA_MMIO_SIZE);

        if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_QEXT)) {
            qext = true;
            pci_set_byte(&d->dev.config[PCI_REVISION_ID], 2);
        }
        if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_EDID)) {
            edid = true;
        }
        pci_std_vga_mmio_region_init(s, OBJECT(dev), &d->mmio, d->mrs,
                                     qext, edid);

        if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_MAC_ACCEL)) {
            if (s->vram_size < 2 * PVMAC_CURSOR_SLOT_SIZE) {
                error_setg(errp, "mac-accel needs vgamem_mb >= 1");
                return;
            }
            pvmac_gl_probe(d);
            memory_region_init_io(&d->mrs[4], OBJECT(dev), &pvmac_ops, d,
                                  "mac-accel regs", PVMAC_SIZE);
            memory_region_add_subregion(&d->mmio, PVMAC_OFFSET, &d->mrs[4]);
            qemu_register_reset(pvmac_reset, d);
        }

        pci_register_bar(&d->dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);
    } else if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_MAC_ACCEL)) {
        error_setg(errp, "mac-accel requires mmio=on");
    }
}

static void pci_std_vga_exit(PCIDevice *dev)
{
    PCIVGAState *d = PCI_VGA(dev);

    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_MAC_ACCEL)) {
        qemu_unregister_reset(pvmac_reset, d);
        pvmac_gl_close(d);
    }
}

static void pci_secondary_vga_realize(PCIDevice *dev, Error **errp)
{
    PCIVGAState *d = PCI_VGA(dev);
    VGACommonState *s = &d->vga;
    bool qext = false;
    bool edid = false;

    /* vga + console init */
    if (!vga_common_init(s, OBJECT(dev), errp)) {
        return;
    }
    s->con = graphic_console_init(DEVICE(dev), 0, s->hw_ops, s);

    /* mmio bar */
    memory_region_init_io(&d->mmio, OBJECT(dev), &unassigned_io_ops, NULL,
                          "vga.mmio", PCI_VGA_MMIO_SIZE);

    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_QEXT)) {
        qext = true;
        pci_set_byte(&d->dev.config[PCI_REVISION_ID], 2);
    }
    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_EDID)) {
        edid = true;
    }
    pci_std_vga_mmio_region_init(s, OBJECT(dev), &d->mmio, d->mrs, qext, edid);

    pci_register_bar(&d->dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram);
    pci_register_bar(&d->dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);
}

static void pci_secondary_vga_exit(PCIDevice *dev)
{
    PCIVGAState *d = PCI_VGA(dev);
    VGACommonState *s = &d->vga;

    graphic_console_close(s->con);
    memory_region_del_subregion(&d->mmio, &d->mrs[0]);
    memory_region_del_subregion(&d->mmio, &d->mrs[1]);
    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_QEXT)) {
        memory_region_del_subregion(&d->mmio, &d->mrs[2]);
    }
    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_EDID)) {
        memory_region_del_subregion(&d->mmio, &d->mrs[3]);
    }
}

static void pci_secondary_vga_init(Object *obj)
{
    /* Expose framebuffer byteorder via QOM */
    object_property_add_bool(obj, "big-endian-framebuffer",
                             vga_get_big_endian_fb, vga_set_big_endian_fb);
}

static void pci_secondary_vga_reset(DeviceState *dev)
{
    PCIVGAState *d = PCI_VGA(PCI_DEVICE(dev));
    vga_common_reset(&d->vga);
}

static const Property vga_pci_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", PCIVGAState, vga.vram_size_mb, 16),
    DEFINE_PROP_BIT("mmio", PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_MMIO, true),
    DEFINE_PROP_BIT("qemu-extended-regs",
                    PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_QEXT, true),
    DEFINE_PROP_BIT("edid",
                    PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_EDID, true),
    DEFINE_PROP_BIT("mac-accel",
                    PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_MAC_ACCEL, false),
    DEFINE_EDID_PROPERTIES(PCIVGAState, edid_info),
    DEFINE_PROP_BOOL("global-vmstate", PCIVGAState, vga.global_vmstate, false),
};

static const Property secondary_pci_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", PCIVGAState, vga.vram_size_mb, 16),
    DEFINE_PROP_BIT("qemu-extended-regs",
                    PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_QEXT, true),
    DEFINE_PROP_BIT("edid",
                    PCIVGAState, flags, PCI_VGA_FLAG_ENABLE_EDID, true),
    DEFINE_EDID_PROPERTIES(PCIVGAState, edid_info),
};

static void vga_pci_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    AcpiDevAmlIfClass *adevc = ACPI_DEV_AML_IF_CLASS(klass);

    k->vendor_id = PCI_VENDOR_ID_QEMU;
    k->device_id = PCI_DEVICE_ID_QEMU_VGA;
    dc->vmsd = &vmstate_vga_pci;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    adevc->build_dev_aml = build_vga_aml;
}

static const TypeInfo vga_pci_type_info = {
    .name = TYPE_PCI_VGA,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PCIVGAState),
    .abstract = true,
    .class_init = vga_pci_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { TYPE_ACPI_DEV_AML_IF },
        { },
    },
};

static void vga_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->realize = pci_std_vga_realize;
    k->exit = pci_std_vga_exit;
    k->romfile = "vgabios-stdvga.bin";
    k->class_id = PCI_CLASS_DISPLAY_VGA;
    device_class_set_props(dc, vga_pci_properties);
    dc->hotpluggable = false;

    /* Expose framebuffer byteorder via QOM */
    object_class_property_add_bool(klass, "big-endian-framebuffer",
                                   vga_get_big_endian_fb, vga_set_big_endian_fb);
}

static void secondary_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->realize = pci_secondary_vga_realize;
    k->exit = pci_secondary_vga_exit;
    k->class_id = PCI_CLASS_DISPLAY_OTHER;
    device_class_set_props(dc, secondary_pci_properties);
    device_class_set_legacy_reset(dc, pci_secondary_vga_reset);
}

static const TypeInfo vga_info = {
    .name          = "VGA",
    .parent        = TYPE_PCI_VGA,
    .class_init    = vga_class_init,
};

static const TypeInfo secondary_info = {
    .name          = "secondary-vga",
    .parent        = TYPE_PCI_VGA,
    .instance_init = pci_secondary_vga_init,
    .class_init    = secondary_class_init,
};

static void vga_register_types(void)
{
    type_register_static(&vga_pci_type_info);
    type_register_static(&vga_info);
    type_register_static(&secondary_info);
}

type_init(vga_register_types)
