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
#include "qemu/log.h"
#include "system/reset.h"
#include "trace.h"

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
 */
#define PVMAC_OFFSET            0x800
#define PVMAC_SIZE              0x800
#define PVMAC_ID_VALUE          0x50564d43  /* 'PVMC' */
#define PVMAC_VERSION_VALUE     1
#define PVMAC_CAP_HWCURSOR      (1u << 0)
#define PVMAC_CURSOR_SLOT_SIZE  0x4000      /* 64 * 64 * 4, top of VRAM */

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
};

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
} PVMACState;

struct PCIVGAState {
    PCIDevice dev;
    VGACommonState vga;
    uint32_t flags;
    qemu_edid_info edid_info;
    MemoryRegion mmio;
    MemoryRegion mrs[5];
    uint8_t edid[384];
    PVMACState pvmac;
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
        return PVMAC_CAP_HWCURSOR;
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
