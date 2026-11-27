/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI SVGA emulation
 *
 * Copyright (c) 2019 BALATON Zoltan
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#ifndef ATI_INT_H
#define ATI_INT_H

#include "qemu/timer.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/i2c/bitbang_i2c.h"
#include "hw/display/i2c-ddc.h"
#include "vga_int.h"
#include "qom/object.h"

/*#define DEBUG_ATI*/

#ifdef DEBUG_ATI
#define DPRINTF(fmt, ...) printf("%s: " fmt, __func__, ## __VA_ARGS__)
#else
#define DPRINTF(fmt, ...) do {} while (0)
#endif

#define PCI_VENDOR_ID_ATI 0x1002
/* Rage128 Pro GL */
#define PCI_DEVICE_ID_ATI_RAGE128_PF 0x5046
/* Radeon RV100 (VE) */
#define PCI_DEVICE_ID_ATI_RADEON_QY 0x5159
/* Radeon R100 (Radeon DDR / 7200, PCI Radeon Mac Edition: "ATY,Rage6p") */
#define PCI_DEVICE_ID_ATI_RADEON_QD 0x5144

#define ATI_RAGE128_LINEAR_APER_SIZE (64 * MiB)
#define ATI_R100_LINEAR_APER_SIZE (128 * MiB)
#define ATI_HOST_DATA_ACC_BITS 128

#define TYPE_ATI_VGA "ati-vga"
OBJECT_DECLARE_SIMPLE_TYPE(ATIVGAState, ATI_VGA)

typedef struct ATIVGARegs {
    uint32_t mm_index;
    uint32_t bios_scratch[8];
    uint32_t gen_int_cntl;
    uint32_t gen_int_status;
    uint32_t crtc_gen_cntl;
    uint32_t crtc_ext_cntl;
    uint32_t dac_cntl;
    uint32_t gpio_vga_ddc;
    uint32_t gpio_dvi_ddc;
    uint32_t gpio_monid;
    uint32_t config_cntl;
    uint32_t palette[256];
    uint32_t crtc_h_total_disp;
    uint32_t crtc_h_sync_strt_wid;
    uint32_t crtc_v_total_disp;
    uint32_t crtc_v_sync_strt_wid;
    uint32_t crtc_offset;
    uint32_t crtc_offset_cntl;
    uint32_t crtc_pitch;
    uint32_t cur_offset;
    uint32_t cur_hv_pos;
    uint32_t cur_hv_offs;
    uint32_t cur_color0;
    uint32_t cur_color1;
    uint32_t dst_offset;
    uint32_t dst_pitch;
    uint32_t dst_tile;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t src_offset;
    uint32_t src_pitch;
    uint32_t src_tile;
    /* Radeon: the guest's SRC/DST_PITCH_OFFSET while a GUI_MASTER_CNTL with
     * the select bit clear has the default in them (restored when it is set) */
    uint32_t saved_src_offset, saved_src_pitch, saved_src_tile;
    uint32_t saved_dst_offset, saved_dst_pitch, saved_dst_tile;
    bool src_saved, dst_saved;
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t dp_gui_master_cntl;
    uint32_t dp_brush_bkgd_clr;
    uint32_t dp_brush_frgd_clr;
    uint32_t dp_src_frgd_clr;
    uint32_t dp_src_bkgd_clr;
    uint16_t sc_top;
    uint16_t sc_left;
    uint16_t sc_bottom;
    uint16_t sc_right;
    uint16_t src_sc_bottom;
    uint16_t src_sc_right;
    uint32_t dp_cntl;
    uint32_t dp_datatype;
    uint32_t dp_mix;
    uint32_t dp_write_mask;
    uint32_t default_offset;
    uint32_t default_pitch;
    uint16_t default_sc_bottom;
    uint16_t default_sc_right;
    uint32_t default_tile;
} ATIVGARegs;

typedef struct ATICPQueue {
    uint32_t buf[0x4000];
    unsigned n;
} ATICPQueue;

typedef struct ATICPState {
    uint32_t rptr;
    unsigned depth;
    uint8_t me_addr;
    struct { uint32_t hi, lo; } me_ram[256];
    ATICPQueue pio[2];
    uint64_t packets3;
} ATICPState;

enum { ATI_MC_NONE, ATI_MC_VRAM, ATI_MC_BUS };

typedef struct ATI3DState ATI3DState;

typedef struct ATIHostDataState {
    bool active;
    uint32_t row;
    uint32_t col;
    uint32_t next;
    uint32_t acc[4];
    /* RetroBoot: byte stream of the current HOST_DATA blit, whole rows
     * are drawn as soon as they are complete */
    uint8_t buf[65536];
    uint32_t nbytes;
    uint32_t flush_at;          /* draw once this many bytes are buffered */
    uint32_t bit0;              /* bits of buf[0] already consumed */
    uint64_t total_bits;        /* size of the whole source stream */
    uint64_t recv_bits;         /* received so far */
    uint64_t ctx[96];           /* blit parameters latched at the start */
} ATIHostDataState;

/* acceleration counters (QMP/trace evidence that the engine is used) */
typedef struct ATIAccelStats {
    uint64_t ops2d, pixels2d, packets2d;
    uint64_t mmio_w, hd_dw, pio_dw, ring_dw, ib_dw;
    uint64_t vblanks;
} ATIAccelStats;

struct ATIVGAState {
    PCIDevice dev;
    /*
     * RetroBoot: backing store for every MMIO register that has no explicit
     * handler, so drivers read back what they wrote (R100 has many such
     * registers: memory controller, surface control, 3D state, scratch).
     */
    uint32_t regfile[0x4000 / 4];
    uint32_t pll[64];
    VGACommonState vga;
    char *model;
    uint16_t dev_id;
    uint8_t mode;
    uint32_t scan_bpp, scan_stride, scan_h, scan_w;   /* applied scanout */
    int64_t dbg_surface_ns;                           /* ati_display_surface */
    bool lut_loaded;        /* guest wrote the DAC LUT in a direct-colour mode */
    uint64_t lut_sig;                                 /* ati_dac_lut trace */
    uint64_t mode_sig;           /* ati_mode_signature() of the applied mode */
    uint8_t use_pixman;
    bool cursor_guest_mode;
    uint32_t fw_mask;           /* front-buffer watch: writers since last check */
    uint64_t fw_hash;
    unsigned fw_tick;
    bool flip_pending;          /* RetroBoot: display start waits for a readback */
    uint16_t cursor_size;
    uint8_t cursor_mode;
    uint32_t cursor_hv_offs;
    uint32_t cursor_offset;
    QEMUCursor *cursor;
    QEMUTimer vblank_timer;
    bitbang_i2c_interface bbi2c;
    I2CDDCState i2cddc;
    uint64_t linear_aper_sz;
    MemoryRegion linear_aper;
    MemoryRegion linear_aper1;
    MemoryRegion io;
    MemoryRegion mm;
    ATIVGARegs regs;
    ATIHostDataState host_data;
    ATICPState cp;
    uint32_t debug;             /* x-debug: 1 = dump 2D packets */
    uint32_t vblank_hz;         /* x-vblank-hz: emulated vertical refresh (lab) */
    int64_t vblank_next;        /* RetroBoot: deadline of the next vblank */
    ATIAccelStats accel;
    ATI3DState *r3d;
};

const char *ati_reg_name(int num);

static inline bool ati_is_radeon(const ATIVGAState *s)
{
    return s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF;
}

void ati_2d_blt(ATIVGAState *s);
void ati_2d_packet(ATIVGAState *s, uint8_t op, const uint32_t *d, unsigned n);

/* register write as from MMIO, used by the command processor */
void ati_reg_write(ATIVGAState *s, hwaddr addr, uint32_t data);

/* ati_cp.c */
int ati_mc_translate(ATIVGAState *s, uint64_t mc, uint64_t *off,
                     uint64_t *contig);
bool ati_mc_rw(ATIVGAState *s, uint64_t mc, void *buf, uint64_t len,
               bool write);
bool ati_cp_write(ATIVGAState *s, hwaddr addr, uint32_t data);
bool ati_cp_read(ATIVGAState *s, hwaddr addr, uint32_t *val);
void ati_cp_reset(ATIVGAState *s);

/* ati_3d.c */
void ati_3d_init(ATIVGAState *s);
void ati_3d_packet(ATIVGAState *s, uint8_t op, const uint32_t *d, unsigned n);
void ati_3d_reg_write(ATIVGAState *s, hwaddr addr, uint32_t data);
enum { ATI_SYNC_2D, ATI_SYNC_STATUS, ATI_SYNC_VBLANK, ATI_SYNC_N };
void ati_3d_sync(ATIVGAState *s, int why);
void ati_3d_sync_range(ATIVGAState *s, uint64_t mc, uint64_t len);
bool ati_3d_flip(ATIVGAState *s, uint64_t mc, uint64_t len);
bool ati_3d_range_ready(ATIVGAState *s, uint64_t mc, uint64_t len);
void ati_3d_land(ATIVGAState *s);
void ati_3d_mode_changed(ATIVGAState *s);
/* the vCPU waits for the render thread with the BQL released */
bool ati_3d_busy(ATIVGAState *s);
/* the 2D engine wrote VRAM [off, off + len) (VRAM offsets) */
void ati_3d_vram_written(ATIVGAState *s, uint64_t off, uint64_t len);
void ati_3d_watch_2d(ATIVGAState *s, uint64_t off, uint64_t len, int rop,
                     int src_kind);
void ati_3d_watch_present(ATIVGAState *s, uint64_t off, uint64_t len);
bool ati_3d_track_info(ATIVGAState *s, uint64_t off, uint32_t *fmt,
                       uint32_t *pitch, uint32_t *w, uint32_t *h,
                       bool *gl_owned);
/*
 * A solid 2D fill of the rectangle r (dst offset, pitch and bytes per pixel
 * as the 2D engine has them; filler = the 32-bit VRAM word it would store,
 * 16-bit values repeated) done on a 3D render target in GL instead of VRAM.
 * False when it doesn't apply; then nothing has changed.
 */
bool ati_3d_fill(ATIVGAState *s, uint32_t dst_offset, int stride, int bypp,
                 int x, int y, int w, int h, uint32_t filler);
void ati_scanout_range(ATIVGAState *s, uint64_t *mc, uint64_t *len);
/* the CRTC runs an extended (non-VGA) mode */
bool ati_ext_mode(ATIVGAState *s);
void ati_flip_settle(ATIVGAState *s);
/* front-buffer watch: kind 1 = 2D op, 2 = GL readback landing */
void ati_front_touch(ATIVGAState *s, unsigned kind, uint64_t off, uint64_t len);
const char *ati_3d_stats(ATIVGAState *s);
const char *ati_3d_gl_info(ATIVGAState *s);
bool ati_host_data_flush(ATIVGAState *s);
void ati_host_data_finish(ATIVGAState *s);

#endif /* ATI_INT_H */
