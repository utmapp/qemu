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

/*
 * WARNING:
 * This is very incomplete and only enough for Linux console and some
 * unaccelerated X output at the moment.
 * Currently it's little more than a frame buffer with minimal functions,
 * other more advanced features of the hardware are yet to be implemented.
 * We only aim for Rage 128 Pro (and some RV100) and 2D only at first,
 * No 3D at all yet (maybe after 2D works, but feel free to improve it)
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "vga-access.h"
#include "hw/qdev-properties.h"
#include "vga_regs.h"
#include "qemu/bswap.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "ui/console.h"
#include "trace.h"

#define ATI_DEBUG_HW_CURSOR 0

#ifdef CONFIG_PIXMAN
#define DEFAULT_X_PIXMAN 3
#else
#define DEFAULT_X_PIXMAN 0
#endif

static const struct {
    const char *name;
    uint16_t dev_id;
} ati_model_aliases[] = {
    { "rage128p", PCI_DEVICE_ID_ATI_RAGE128_PF },
    { "rv100", PCI_DEVICE_ID_ATI_RADEON_QY },
    { "r100", PCI_DEVICE_ID_ATI_RADEON_QD },
};

enum { VGA_MODE, EXT_MODE };

static void ati_vga_set_offset(VGACommonState *vga, uint32_t offs)
{
    int bypp = DIV_ROUND_UP(vga->vbe_regs[VBE_DISPI_INDEX_BPP], BITS_PER_BYTE);

    if (!bypp ||
        vga->vbe_regs[VBE_DISPI_INDEX_YRES] *
        vga->vbe_regs[VBE_DISPI_INDEX_VIRT_WIDTH] * bypp + offs >
        vga->vbe_size) {
        return;
    }
    vga->vbe_start_addr = offs / 4;
}

static void ati_vga_switch_mode(ATIVGAState *s)
{
    s->flip_pending = false;     /* RetroBoot: a new mode shows its own start */
    DPRINTF("%d -> %d\n",
            s->mode, !!(s->regs.crtc_gen_cntl & CRTC2_EXT_DISP_EN));
    if (s->regs.crtc_gen_cntl & CRTC2_EXT_DISP_EN) {
        /* Extended mode enabled */
        s->mode = EXT_MODE;
        if (s->regs.crtc_gen_cntl & CRTC2_EN) {
            /* CRT controller enabled, use CRTC values */
            /* FIXME Should these be the same as VGA CRTC regs? */
            uint32_t offs = s->regs.crtc_offset & 0x07ffffff;
            int stride = (s->regs.crtc_pitch & 0x7ff) * 8;
            int bpp = 0;
            int h, v;

            if (s->regs.crtc_h_total_disp == 0) {
                s->regs.crtc_h_total_disp = ((640 / 8) - 1) << 16;
            }
            if (s->regs.crtc_v_total_disp == 0) {
                s->regs.crtc_v_total_disp = (480 - 1) << 16;
            }
            h = ((s->regs.crtc_h_total_disp >> 16) + 1) * 8;
            v = (s->regs.crtc_v_total_disp >> 16) + 1;
            switch (s->regs.crtc_gen_cntl & CRTC_PIX_WIDTH_MASK) {
            case CRTC_PIX_WIDTH_4BPP:
                bpp = 4;
                break;
            case CRTC_PIX_WIDTH_8BPP:
                bpp = 8;
                break;
            case CRTC_PIX_WIDTH_15BPP:
                bpp = 15;
                break;
            case CRTC_PIX_WIDTH_16BPP:
                bpp = 16;
                break;
            case CRTC_PIX_WIDTH_24BPP:
                bpp = 24;
                break;
            case CRTC_PIX_WIDTH_32BPP:
                bpp = 32;
                break;
            default:
                qemu_log_mask(LOG_UNIMP, "Unsupported bpp value\n");
                return;
            }
            DPRINTF("Switching to %dx%d %d %d @ %x\n", h, v, stride, bpp, offs);
            vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
            vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_DISABLED);
            s->vga.big_endian_fb = (s->regs.config_cntl & APER_0_ENDIAN ||
                                    s->regs.config_cntl & APER_1_ENDIAN ?
                                    true : false);
            /* reset VBE regs then set up mode */
            s->vga.vbe_regs[VBE_DISPI_INDEX_XRES] = h;
            s->vga.vbe_regs[VBE_DISPI_INDEX_YRES] = v;
            s->vga.vbe_regs[VBE_DISPI_INDEX_BPP] = bpp;
            /* enable mode via ioport so it updates vga regs */
            vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
            vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_ENABLED |
                VBE_DISPI_LFB_ENABLED | VBE_DISPI_NOCLEARMEM |
                (s->regs.dac_cntl & DAC_8BIT_EN ? VBE_DISPI_8BIT_DAC : 0));
            /* now set offset and stride because enable resets these */
            if (stride) {
                vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_VIRT_WIDTH);
                vbe_ioport_write_data(&s->vga, 0, stride);
            }
            ati_vga_set_offset(&s->vga, offs);
        }
    } else {
        /* VGA mode enabled */
        s->mode = VGA_MODE;
        vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
        vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_DISABLED);
    }
}

/* Used by host side hardware cursor */
static void ati_cursor_define(ATIVGAState *s)
{
    uint64_t data[128];
    uint32_t srcoff;

    if (s->regs.cur_offset & BIT(31)) {
        return;
    }
    if (s->cursor_guest_mode) {
        /* Cursor pixels can change at the same address on an unlock. */
        vga_invalidate_scanlines(&s->vga, s->vga.hw_cursor_y,
                                 s->vga.hw_cursor_y + 63);
        return;
    }
    /* FIXME handle cur_hv_offs correctly */
    srcoff = s->regs.cur_offset - (s->regs.cur_hv_offs >> 16) -
             (s->regs.cur_hv_offs & 0xffff) * 16;
    if (srcoff > s->vga.vram_size - 64 * 16) {
        return;
    }
    for (int i = 0; i < 64; i++, srcoff += 16) {
        data[i] = ldq_le_p(&s->vga.vram_ptr[srcoff]);
        data[i + 64] = ldq_le_p(&s->vga.vram_ptr[srcoff + 8]);
    }
    if (!s->cursor) {
        s->cursor = cursor_alloc(64, 64);
    }
    cursor_set_mono(s->cursor, s->regs.cur_color1, s->regs.cur_color0,
                    (uint8_t *)&data[64], 1, (uint8_t *)&data[0]);
    dpy_cursor_define(s->vga.con, s->cursor);
}

/* Alternatively support guest rendered hardware cursor */
static void ati_cursor_invalidate(VGACommonState *vga)
{
    ATIVGAState *s = container_of(vga, ATIVGAState, vga);
    int size = (s->regs.crtc_gen_cntl & CRTC2_CUR_EN) ? 64 : 0;
    unsigned mode = (s->regs.crtc_gen_cntl >> 20) & 7;
    /* CUR_OFFSET already includes the clipped top rows. Horizontal
     * clipping selects pixels within each row, never preceding VRAM. */
    uint32_t offset = s->regs.cur_offset & 0x07ffffff;

    if (s->regs.cur_offset & BIT(31)) {
        return; /* Do not update cursor if locked */
    }
    if (s->cursor_size != size || s->cursor_mode != mode ||
        s->cursor_hv_offs != s->regs.cur_hv_offs ||
        vga->hw_cursor_x != s->regs.cur_hv_pos >> 16 ||
        vga->hw_cursor_y != (s->regs.cur_hv_pos & 0xffff) ||
        s->cursor_offset != offset) {
        /* Remove old cursor then update and show new one if needed */
        vga_invalidate_scanlines(vga, vga->hw_cursor_y, vga->hw_cursor_y + 63);
        vga->hw_cursor_x = s->regs.cur_hv_pos >> 16;
        vga->hw_cursor_y = s->regs.cur_hv_pos & 0xffff;
        s->cursor_offset = offset;
        s->cursor_size = size;
        s->cursor_mode = mode;
        s->cursor_hv_offs = s->regs.cur_hv_offs;
        if (size) {
            vga_invalidate_scanlines(vga,
                                     vga->hw_cursor_y, vga->hw_cursor_y + 63);
        }
    }
}

static void ati_cursor_draw_line(VGACommonState *vga, uint8_t *d, int scr_y)
{
    ATIVGAState *s = container_of(vga, ATIVGAState, vga);
    uint32_t h, srcoff, color;
    uint64_t abits, xbits, mask;
    uint32_t *dp = (uint32_t *)d;

    if (!(s->regs.crtc_gen_cntl & CRTC2_CUR_EN) ||
        scr_y < vga->hw_cursor_y || scr_y >= vga->hw_cursor_y + 64 ||
        scr_y > s->regs.crtc_v_total_disp >> 16) {
        return;
    }
    h = ((s->regs.crtc_h_total_disp >> 16) + 1) * 8;
    if (vga->hw_cursor_x >= h) {
        return;
    }
    if (ati_is_radeon(s) && (s->cursor_mode == 2 || s->cursor_mode == 3)) {
        unsigned xorigin = (s->cursor_hv_offs >> 16) & 63;
        unsigned yorigin = s->cursor_hv_offs & 63;
        unsigned row = scr_y - vga->hw_cursor_y;
        uint64_t source = (uint64_t)s->cursor_offset + row * 256;

        if (row >= 64 - yorigin || source + 256 > vga->vram_size) {
            return;
        }
        dp += vga->hw_cursor_x;
        for (unsigned i = 0; i < 64 - xorigin && vga->hw_cursor_x + i < h; i++) {
            const uint8_t *pixel = vga->vram_ptr + source + (i + xorigin) * 4;
            uint32_t src = vga->big_endian_fb ? ldl_be_p(pixel) : ldl_le_p(pixel);
            unsigned alpha = src >> 24;
            uint32_t out = 0xff000000;

            for (unsigned shift = 0; shift < 24; shift += 8) {
                unsigned value = ((src >> shift) & 255) * alpha +
                                 ((dp[i] >> shift) & 255) * (255 - alpha);
                out |= ((value + 127) / 255) << shift;
            }
            dp[i] = out;
        }
        return;
    }
    /* Legacy two-plane monochrome cursor. CUR_OFFSET points to its
     * first visible row; each plane skips the clipped left pixels. */
    unsigned xorigin = (s->cursor_hv_offs >> 16) & 63;
    unsigned yorigin = s->cursor_hv_offs & 63;
    unsigned row = scr_y - vga->hw_cursor_y;

    if (row >= 64 - yorigin) {
        return;
    }
    srcoff = s->cursor_offset + row * 16;
    if (srcoff > s->vga.vram_size - 16) {
        return;
    }
    dp = &dp[vga->hw_cursor_x];
    h = ((s->regs.crtc_h_total_disp >> 16) + 1) * 8;
    abits = ldq_be_p(&vga->vram_ptr[srcoff]);
    xbits = ldq_be_p(&vga->vram_ptr[srcoff + 8]);
    mask = BIT_ULL(63) >> xorigin;
    for (unsigned i = 0; i < 64 - xorigin; i++, mask >>= 1) {
        if (vga->hw_cursor_x + i >= h) {
            return; /* end of screen, don't span to next line */
        }
        if (abits & mask) {
            if (xbits & mask) {
                color = dp[i] ^ 0xffffffff; /* complement */
            } else {
                continue; /* transparent, no change */
            }
        } else {
            color = (xbits & mask ? s->regs.cur_color1 :
                                    s->regs.cur_color0) | 0xff000000;
        }
        dp[i] = color;
    }
}

static uint64_t ati_i2c(bitbang_i2c_interface *i2c, uint64_t data, int base)
{
    bool c = (data & BIT(base + 17) ? !!(data & BIT(base + 1)) : 1);
    bool d = (data & BIT(base + 16) ? !!(data & BIT(base)) : 1);

    bitbang_i2c_set(i2c, BITBANG_I2C_SCL, c);
    d = bitbang_i2c_set(i2c, BITBANG_I2C_SDA, d);

    data &= ~0xf00ULL;
    if (c) {
        data |= BIT(base + 9);
    }
    if (d) {
        data |= BIT(base + 8);
    }
    return data;
}

static void ati_vga_update_irq(ATIVGAState *s)
{
    pci_set_irq(&s->dev, !!(s->regs.gen_int_status & s->regs.gen_int_cntl));
}

/* RetroBoot: the CRTC runs an extended (non-VGA) mode */
bool ati_ext_mode(ATIVGAState *s)
{
    return s->mode == EXT_MODE;
}

/*
 * RetroBoot: the MC address range the CRTC scans out, or will once a pending
 * flip settles (CRTC_OFFSET as written)
 */
void ati_scanout_range(ATIVGAState *s, uint64_t *mc, uint64_t *len)
{
    *mc = (uint64_t)(s->regfile[MC_FB_LOCATION / 4] & 0xffff) << 16 |
          (s->regs.crtc_offset & 0x07ffffff);
    *len = (uint64_t)s->vga.vbe_line_offset * s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
}

/*
 * RetroBoot: a page flip waiting for its frame takes effect once the frame is
 * in VRAM.  The vblank checks, and so does every synchronous readback (a 2D
 * operation or a status read): the guest goes on as soon as it has written
 * CRTC_OFFSET, and Mac OS's ATI OpenGL clears the next back buffer -- the
 * front buffer being left -- with a 2D fill before it draws.  That fill first
 * lands the new frame (ati_3d_sync), so showing it at once keeps the display
 * off the buffer being cleared; waiting for the vblank showed it black for a
 * frame at most flips (Cro-Mag Rally: about ten black frames a second).
 */
void ati_flip_settle(ATIVGAState *s)
{
    uint64_t mc, len;

    if (!s->flip_pending || s->mode != EXT_MODE) {
        return;
    }
    ati_scanout_range(s, &mc, &len);
    if (ati_3d_range_ready(s, mc, len)) {
        s->flip_pending = false;
        ati_vga_set_offset(&s->vga, s->regs.crtc_offset & 0x07ffffff);
    }
}

static void ati_vga_vblank_irq(void *opaque)
{
    ATIVGAState *s = opaque;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    int64_t period = NANOSECONDS_PER_SECOND / MAX(s->vblank_hz, 1);

    /*
     * Scanout of a 3D render target.  Skipped while the vCPU waits for the
     * render thread with the BQL released (ati_3d_busy): its readback is
     * mid-operation, and it lands frames and settles flips itself on return.
     */
    if (ati_is_radeon(s) && s->mode == EXT_MODE && !ati_3d_busy(s)) {
        uint64_t mc, len;
        ati_3d_land(s);                 /* finished readbacks into VRAM */
        ati_flip_settle(s);
        ati_scanout_range(s, &mc, &len);
        ati_3d_sync_range(s, mc, len);  /* and start the next ones */
    }

    /*
     * RetroBoot: a steady rate.  The next vblank is due a period after this
     * one's deadline, not after the moment the timer ran: a main loop held up
     * for a few milliseconds per frame made every period that much longer
     * (Cro-Mag Rally's menu, which flips once per vblank, ran at 52 Hz
     * instead of 60).  More than a period late starts a new phase.  Enabling
     * the interrupt (GEN_INT_CNTL) while the timer runs keeps its phase.
     */
    if (!timer_pending(&s->vblank_timer)) {
        int64_t next = s->vblank_next + period;
        if (!s->vblank_next || next <= now) {
            next = now + period;
        }
        s->vblank_next = next;
        timer_mod(&s->vblank_timer, next);
    }
    s->regs.gen_int_status |= CRTC_VBLANK_INT;
    s->accel.vblanks++;
    ati_vga_update_irq(s);
}

static uint64_t ati_mm_read(void *opaque, hwaddr addr, unsigned int size);
static void ati_mm_write(void *opaque, hwaddr addr, uint64_t data,
                         unsigned int size);

static inline uint32_t ati_reg_read_offs(uint32_t reg, int offs,
                                         unsigned int size)
{
    if (offs == 0 && size == 4) {
        return reg;
    } else {
        return extract32(reg, offs * BITS_PER_BYTE, size * BITS_PER_BYTE);
    }
}

static uint64_t ati_mm_read_reg(void *opaque, hwaddr addr, unsigned int size)
{
    ATIVGAState *s = opaque;
    uint32_t val = 0;

    switch (addr) {
    case MM_INDEX:
        val = s->regs.mm_index;
        break;
    case MM_DATA ... MM_DATA + 3:
        /* indexed access to regs or memory */
        if (s->regs.mm_index & BIT(31)) {
            uint32_t idx = s->regs.mm_index & ~BIT(31);
            if (idx <= s->vga.vram_size - size) {
                val = ldn_le_p(s->vga.vram_ptr + idx, size);
            }
        } else if (s->regs.mm_index > MM_DATA + 3) {
            val = ati_mm_read(s, s->regs.mm_index + addr - MM_DATA, size);
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                "ati_mm_read: mm_index too small: %u\n", s->regs.mm_index);
        }
        break;
    case BIOS_0_SCRATCH ... BUS_CNTL - 1:
    {
        int i = (addr - BIOS_0_SCRATCH) / 4;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF && i > 3) {
            break;
        }
        val = ati_reg_read_offs(s->regs.bios_scratch[i],
                                addr - (BIOS_0_SCRATCH + i * 4), size);
        break;
    }
    case GEN_INT_CNTL:
        val = s->regs.gen_int_cntl;
        break;
    case GEN_INT_STATUS:
        val = s->regs.gen_int_status;
        break;
    case CRTC_GEN_CNTL ... CRTC_GEN_CNTL + 3:
        val = ati_reg_read_offs(s->regs.crtc_gen_cntl,
                                addr - CRTC_GEN_CNTL, size);
        break;
    case CRTC_EXT_CNTL ... CRTC_EXT_CNTL + 3:
        val = ati_reg_read_offs(s->regs.crtc_ext_cntl,
                                addr - CRTC_EXT_CNTL, size);
        break;
    case DAC_CNTL:
        val = s->regs.dac_cntl;
        break;
    case GPIO_VGA_DDC ... GPIO_VGA_DDC + 3:
        val = ati_reg_read_offs(s->regs.gpio_vga_ddc,
                                addr - GPIO_VGA_DDC, size);
        break;
    case GPIO_DVI_DDC ... GPIO_DVI_DDC + 3:
        val = ati_reg_read_offs(s->regs.gpio_dvi_ddc,
                                addr - GPIO_DVI_DDC, size);
        break;
    case GPIO_MONID ... GPIO_MONID + 3:
        val = ati_reg_read_offs(s->regs.gpio_monid,
                                addr - GPIO_MONID, size);
        break;
    case PALETTE_INDEX:
        /* FIXME unaligned access */
        val = vga_ioport_read(&s->vga, VGA_PEL_IR) << 16;
        val |= vga_ioport_read(&s->vga, VGA_PEL_IW) & 0xff;
        break;
    case PALETTE_DATA:
        val = vga_ioport_read(&s->vga, VGA_PEL_D);
        break;
    case PALETTE_30_DATA:
        val = s->regs.palette[vga_ioport_read(&s->vga, VGA_PEL_IR)];
        break;
    case CNFG_CNTL:
        val = s->regs.config_cntl;
        break;
    case CNFG_MEMSIZE:
        val = s->vga.vram_size;
        break;
    case CONFIG_APER_0_BASE:
    case CONFIG_APER_1_BASE:
        val = pci_default_read_config(&s->dev,
                                      PCI_BASE_ADDRESS_0, size) & 0xfffffff0;
        if (ati_is_radeon(s) && addr == CONFIG_APER_1_BASE) {
            /* Radeon: aperture 1 follows aperture 0 in the same BAR */
            val += memory_region_size(&s->linear_aper) / 2;
        }
        break;
    case CONFIG_APER_SIZE:
        val = memory_region_size(&s->linear_aper) / 2;
        break;
    case CONFIG_REG_1_BASE:
        val = pci_default_read_config(&s->dev,
                                      PCI_BASE_ADDRESS_2, size) & 0xfffffff0;
        break;
    case CONFIG_REG_APER_SIZE:
        val = memory_region_size(&s->mm) / 2;
        break;
    case HOST_PATH_CNTL:
        val = BIT(23); /* Radeon HDP_APER_CNTL */
        break;
    case MC_STATUS:
        /* Radeon: MEM_PWRUP_COMPL_A|B and MC_IDLE; Rage 128: idle */
        val = ati_is_radeon(s) ? 7 : 5;
        break;
    case MEM_SDRAM_MODE_REG:
        if (s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF) {
            val = BIT(28) | BIT(20);
        }
        break;
    case RBBM_STATUS:
    case GUI_STAT:
        if (ati_is_radeon(s)) {
            /* the guest waits for idle: make 3D output visible in memory */
            ati_3d_sync(s, ATI_SYNC_STATUS);
        }
        val = 64; /* free CMDFIFO entries */
        break;
    case CRTC_H_TOTAL_DISP:
        val = s->regs.crtc_h_total_disp;
        break;
    case CRTC_H_SYNC_STRT_WID:
        val = s->regs.crtc_h_sync_strt_wid;
        break;
    case CRTC_V_TOTAL_DISP:
        val = s->regs.crtc_v_total_disp;
        break;
    case CRTC_V_SYNC_STRT_WID:
        val = s->regs.crtc_v_sync_strt_wid;
        break;
    case CRTC_OFFSET:
        val = s->regs.crtc_offset;
        break;
    case CRTC_OFFSET_CNTL:
        val = s->regs.crtc_offset_cntl;
        break;
    case CRTC_PITCH:
        val = s->regs.crtc_pitch;
        break;
    case 0xf00 ... 0xfff:
        val = pci_default_read_config(&s->dev, addr - 0xf00, size);
        break;
    case CUR_OFFSET ... CUR_OFFSET + 3:
        val = ati_reg_read_offs(s->regs.cur_offset, addr - CUR_OFFSET, size);
        break;
    case CUR_HORZ_VERT_POSN ... CUR_HORZ_VERT_POSN + 3:
        val = ati_reg_read_offs(s->regs.cur_hv_pos,
                                addr - CUR_HORZ_VERT_POSN, size);
        if (addr + size > CUR_HORZ_VERT_POSN + 3) {
            val |= (s->regs.cur_offset & BIT(31)) >> (4 - size);
        }
        break;
    case CUR_HORZ_VERT_OFF ... CUR_HORZ_VERT_OFF + 3:
        val = ati_reg_read_offs(s->regs.cur_hv_offs,
                                addr - CUR_HORZ_VERT_OFF, size);
        if (addr + size > CUR_HORZ_VERT_OFF + 3) {
            val |= (s->regs.cur_offset & BIT(31)) >> (4 - size);
        }
        break;
    case CUR_CLR0 ... CUR_CLR0 + 3:
        val = ati_reg_read_offs(s->regs.cur_color0, addr - CUR_CLR0, size);
        break;
    case CUR_CLR1 ... CUR_CLR1 + 3:
        val = ati_reg_read_offs(s->regs.cur_color1, addr - CUR_CLR1, size);
        break;
    case DST_OFFSET:
        val = s->regs.dst_offset;
        break;
    case DST_PITCH:
        val = s->regs.dst_pitch;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            val |= s->regs.dst_tile << 16;
        }
        break;
    case DST_WIDTH:
        val = s->regs.dst_width;
        break;
    case DST_HEIGHT:
        val = s->regs.dst_height;
        break;
    case SRC_X:
        val = s->regs.src_x;
        break;
    case SRC_Y:
        val = s->regs.src_y;
        break;
    case DST_X:
        val = s->regs.dst_x;
        break;
    case DST_Y:
        val = s->regs.dst_y;
        break;
    case DP_GUI_MASTER_CNTL:
        /* DP_GUI_MASTER_CNTL aliases fields from DP_MIX and DP_DATATYPE */
        val = s->regs.dp_gui_master_cntl |
              ((s->regs.dp_datatype & DP_BRUSH_DATATYPE) >> 4) |
              ((s->regs.dp_datatype & DP_DST_DATATYPE) << 8) |
              ((s->regs.dp_datatype & DP_SRC_DATATYPE) >> 4) |
              (s->regs.dp_mix & DP_ROP3) |
              ((s->regs.dp_mix & DP_SRC_SOURCE) << 16);
        break;
    case SRC_OFFSET:
        val = s->regs.src_offset;
        break;
    case SRC_PITCH:
        val = s->regs.src_pitch;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            val |= s->regs.src_tile << 16;
        }
        break;
    case DP_BRUSH_BKGD_CLR:
        val = s->regs.dp_brush_bkgd_clr;
        break;
    case DP_BRUSH_FRGD_CLR:
        val = s->regs.dp_brush_frgd_clr;
        break;
    case DP_SRC_FRGD_CLR:
        val = s->regs.dp_src_frgd_clr;
        break;
    case DP_SRC_BKGD_CLR:
        val = s->regs.dp_src_bkgd_clr;
        break;
    case DP_CNTL:
        val = s->regs.dp_cntl;
        break;
    case DP_DATATYPE:
        val = s->regs.dp_datatype;
        break;
    case DP_MIX:
        val = s->regs.dp_mix;
        break;
    case DP_WRITE_MASK:
        val = s->regs.dp_write_mask;
        break;
    case DEFAULT_OFFSET:
        val = s->regs.default_offset;
        if (s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF) {
            val >>= 10;
            val |= s->regs.default_pitch << 16;
            val |= s->regs.default_tile << 30;
        }
        break;
    case DEFAULT_PITCH:
        val = s->regs.default_pitch;
        val |= s->regs.default_tile << 16;
        break;
    case DEFAULT_SC_BOTTOM_RIGHT:
        val = s->regs.default_sc_right;
        val |= s->regs.default_sc_bottom << 16;
        break;
    case SC_TOP:
        val = s->regs.sc_top;
        break;
    case SC_LEFT:
        val = s->regs.sc_left;
        break;
    case SC_BOTTOM:
        val = s->regs.sc_bottom;
        break;
    case SC_RIGHT:
        val = s->regs.sc_right;
        break;
    case SRC_SC_BOTTOM:
        val = s->regs.src_sc_bottom;
        break;
    case SRC_SC_RIGHT:
        val = s->regs.src_sc_right;
        break;
    case SC_TOP_LEFT:
    case SC_BOTTOM_RIGHT:
    case SRC_SC_BOTTOM_RIGHT:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "Read from write-only register 0x%x\n", (unsigned)addr);
        break;
    case CLOCK_CNTL_INDEX:
        val = s->regfile[CLOCK_CNTL_INDEX / 4];
        break;
    case CLOCK_CNTL_DATA ... CLOCK_CNTL_DATA + 3:
    {
        int idx = s->regfile[CLOCK_CNTL_INDEX / 4] & 0x3f;
        uint32_t pv = s->pll[idx];
        if (idx == PPLL_REF_DIV) {
            pv &= ~PPLL_ATOMIC_UPDATE_R;   /* atomic update always done */
        }
        val = ati_reg_read_offs(pv, addr - CLOCK_CNTL_DATA, size);
        trace_ati_pll_read(idx, pv);
        break;
    }
    default:
        if (ati_is_radeon(s) && addr < sizeof(s->regfile)) {
            uint32_t v;
            if (size == 4 && ati_cp_read(s, addr, &v)) {
                val = v;
            } else {
                val = ati_reg_read_offs(s->regfile[addr / 4], addr & 3, size);
            }
        }
        break;
    }
    if (addr < CUR_OFFSET || addr > CUR_CLR1 || ATI_DEBUG_HW_CURSOR) {
        trace_ati_mm_read(size, addr, ati_reg_name(addr & ~3ULL), val);
    }
    return val;
}

static inline void ati_reg_write_offs(uint32_t *reg, int offs,
                                      uint64_t data, unsigned int size)
{
    if (offs == 0 && size == 4) {
        *reg = data;
    } else {
        *reg = deposit32(*reg, offs * BITS_PER_BYTE, size * BITS_PER_BYTE,
                         data);
    }
}

/* RetroBoot: HOST_DATA0..7/LAST stream into the row buffer; whole bands of
 * rows are drawn as they complete */
static void ati_host_data_write(ATIVGAState *s, hwaddr addr, uint32_t data)
{
    if (!s->host_data.active) {
        return;
    }
    if (s->host_data.nbytes + 4 <= sizeof(s->host_data.buf)) {
        stl_le_p(s->host_data.buf + s->host_data.nbytes, data);
        s->host_data.nbytes += 4;
    }
    s->host_data.recv_bits += 32;
    s->accel.hd_dw++;
    /* the blit completes with its last data, HOST_DATA_LAST or not */
    if (addr == HOST_DATA_LAST ||
        s->host_data.recv_bits >= s->host_data.total_bits) {
        ati_host_data_finish(s);
    } else if (s->host_data.nbytes >= s->host_data.flush_at) {
        ati_host_data_flush(s);
    }
}

static void ati_mm_write_reg(void *opaque, hwaddr addr,
                             uint64_t data, unsigned int size)
{
    ATIVGAState *s = opaque;

    if (addr < CUR_OFFSET || addr > CUR_CLR1 || ATI_DEBUG_HW_CURSOR) {
        trace_ati_mm_write(size, addr, ati_reg_name(addr & ~3ULL), data);
    }
    if (addr >= MM_DATA + 4 && addr < sizeof(s->regfile)) {
        ati_reg_write_offs(&s->regfile[addr / 4], addr & 3, data, size);
    }
    if (ati_is_radeon(s) && size == 4) {
        if (ati_cp_write(s, addr, data)) {
            return;
        }
        if (addr >= 0x1c00 && addr < 0x2000) {
            ati_3d_reg_write(s, addr, data);    /* 3D state */
        }
    }
    switch (addr) {
    case CLOCK_CNTL_DATA ... CLOCK_CNTL_DATA + 3:
    {
        int idx = s->regfile[CLOCK_CNTL_INDEX / 4] & 0x3f;
        if (s->regfile[CLOCK_CNTL_INDEX / 4] & PLL_WR_EN) {
            ati_reg_write_offs(&s->pll[idx], addr - CLOCK_CNTL_DATA,
                               data, size);
            trace_ati_pll_write(idx, s->pll[idx]);
        }
        break;
    }
    case MM_INDEX:
        s->regs.mm_index = data & ~3;
        break;
    case MM_DATA ... MM_DATA + 3:
        /* indexed access to regs or memory */
        if (s->regs.mm_index & BIT(31)) {
            uint32_t idx = s->regs.mm_index & ~BIT(31);
            if (idx <= s->vga.vram_size - size) {
                stn_le_p(s->vga.vram_ptr + idx, size, data);
            }
        } else if (s->regs.mm_index > MM_DATA + 3) {
            ati_mm_write(s, s->regs.mm_index + addr - MM_DATA, data, size);
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                "ati_mm_write: mm_index too small: %u\n", s->regs.mm_index);
        }
        break;
    case BIOS_0_SCRATCH ... BUS_CNTL - 1:
    {
        int i = (addr - BIOS_0_SCRATCH) / 4;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF && i > 3) {
            break;
        }
        ati_reg_write_offs(&s->regs.bios_scratch[i],
                           addr - (BIOS_0_SCRATCH + i * 4), data, size);
        break;
    }
    case GEN_INT_CNTL:
        s->regs.gen_int_cntl = data;
        if (data & CRTC_VBLANK_INT) {
            ati_vga_vblank_irq(s);
        } else {
            timer_del(&s->vblank_timer);
            ati_vga_update_irq(s);
        }
        break;
    case GEN_INT_STATUS:
        if (ati_is_radeon(s) && (data & RADEON_SW_INT_FIRE)) {
            /* software interrupt, usually fired from the command stream */
            s->regs.gen_int_status |= RADEON_SW_INT_TEST;
            data &= ~RADEON_SW_INT_FIRE;
            ati_vga_update_irq(s);
            break;
        }
        data &= (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF ?
                 0x000f040fUL : 0xfc080effUL);
        s->regs.gen_int_status &= ~data;
        ati_vga_update_irq(s);
        break;
    case CRTC_GEN_CNTL ... CRTC_GEN_CNTL + 3:
    {
        uint32_t val = s->regs.crtc_gen_cntl;
        ati_reg_write_offs(&s->regs.crtc_gen_cntl,
                           addr - CRTC_GEN_CNTL, data, size);
        if ((val & CRTC2_CUR_EN) != (s->regs.crtc_gen_cntl & CRTC2_CUR_EN)) {
            ati_vga_switch_mode(s);
            if (s->cursor_guest_mode) {
                s->vga.force_shadow = !!(s->regs.crtc_gen_cntl & CRTC2_CUR_EN);
            } else {
                if (s->regs.crtc_gen_cntl & CRTC2_CUR_EN) {
                    ati_cursor_define(s);
                }
                dpy_mouse_set(s->vga.con, s->regs.cur_hv_pos >> 16,
                              s->regs.cur_hv_pos & 0xffff,
                              (s->regs.crtc_gen_cntl & CRTC2_CUR_EN) != 0);
            }
        }
        if ((val & (CRTC2_EXT_DISP_EN | CRTC2_EN | CRTC_PIX_WIDTH_MASK)) !=
            (s->regs.crtc_gen_cntl &
             (CRTC2_EXT_DISP_EN | CRTC2_EN | CRTC_PIX_WIDTH_MASK))) {
            ati_vga_switch_mode(s);
        }
        break;
    }
    case CRTC_EXT_CNTL ... CRTC_EXT_CNTL + 3:
    {
        uint32_t val = s->regs.crtc_ext_cntl;
        ati_reg_write_offs(&s->regs.crtc_ext_cntl,
                           addr - CRTC_EXT_CNTL, data, size);
        if (s->regs.crtc_ext_cntl & CRT_CRTC_DISPLAY_DIS) {
            DPRINTF("Display disabled\n");
            s->vga.ar_index &= ~BIT(5);
        } else {
            DPRINTF("Display enabled\n");
            s->vga.ar_index |= BIT(5);
            ati_vga_switch_mode(s);
        }
        if ((val & CRT_CRTC_DISPLAY_DIS) !=
            (s->regs.crtc_ext_cntl & CRT_CRTC_DISPLAY_DIS)) {
            ati_vga_switch_mode(s);
        }
        break;
    }
    case DAC_CNTL:
        s->regs.dac_cntl = data & 0xffffe3ff;
        s->vga.dac_8bit = !!(data & DAC_8BIT_EN);
        break;
    /*
     * GPIO regs for DDC access. Because some drivers access these via
     * multiple byte writes we have to be careful when we send bits to
     * avoid spurious changes in bitbang_i2c state. Only do it when either
     * the enable bits are changed or output bits changed while enabled.
     */
    case GPIO_VGA_DDC ... GPIO_VGA_DDC + 3:
        if (s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF) {
            /* FIXME: Maybe add a property to select VGA or DVI port? */
        }
        break;
    case GPIO_DVI_DDC ... GPIO_DVI_DDC + 3:
        if (s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF) {
            ati_reg_write_offs(&s->regs.gpio_dvi_ddc,
                               addr - GPIO_DVI_DDC, data, size);
            if ((addr <= GPIO_DVI_DDC + 2 && addr + size > GPIO_DVI_DDC + 2) ||
                (addr == GPIO_DVI_DDC && (s->regs.gpio_dvi_ddc & 0x30000))) {
                s->regs.gpio_dvi_ddc = ati_i2c(&s->bbi2c,
                                               s->regs.gpio_dvi_ddc, 0);
            }
        }
        break;
    case GPIO_MONID ... GPIO_MONID + 3:
        /* FIXME What does Radeon have here? */
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            /* Rage128p accesses DDC via MONID(1-2) with additional mask bit */
            ati_reg_write_offs(&s->regs.gpio_monid,
                               addr - GPIO_MONID, data, size);
            if ((s->regs.gpio_monid & BIT(25)) &&
                ((addr <= GPIO_MONID + 2 && addr + size > GPIO_MONID + 2) ||
                 (addr == GPIO_MONID && (s->regs.gpio_monid & 0x60000)))) {
                s->regs.gpio_monid = ati_i2c(&s->bbi2c, s->regs.gpio_monid, 1);
            }
        }
        break;
    case PALETTE_INDEX ... PALETTE_INDEX + 3:
        if (size == 4) {
            vga_ioport_write(&s->vga, VGA_PEL_IR, (data >> 16) & 0xff);
            vga_ioport_write(&s->vga, VGA_PEL_IW, data & 0xff);
        } else {
            if (addr == PALETTE_INDEX) {
                vga_ioport_write(&s->vga, VGA_PEL_IW, data & 0xff);
            } else {
                vga_ioport_write(&s->vga, VGA_PEL_IR, data & 0xff);
            }
        }
        break;
    case PALETTE_DATA ... PALETTE_DATA + 3:
        data <<= addr - PALETTE_DATA;
        data = bswap32(data) >> 8;
        vga_ioport_write(&s->vga, VGA_PEL_D, data & 0xff);
        data >>= 8;
        vga_ioport_write(&s->vga, VGA_PEL_D, data & 0xff);
        data >>= 8;
        vga_ioport_write(&s->vga, VGA_PEL_D, data & 0xff);
        break;
    case PALETTE_30_DATA:
        s->regs.palette[vga_ioport_read(&s->vga, VGA_PEL_IW)] = data;
        vga_ioport_write(&s->vga, VGA_PEL_D, (data >> 22) & 0xff);
        vga_ioport_write(&s->vga, VGA_PEL_D, (data >> 12) & 0xff);
        vga_ioport_write(&s->vga, VGA_PEL_D, (data >> 2) & 0xff);
        break;
    case CNFG_CNTL:
        s->regs.config_cntl = data;
        break;
    case CRTC_H_TOTAL_DISP:
    {
        uint32_t old_disp = s->regs.crtc_h_total_disp & 0x07ff0000;
        s->regs.crtc_h_total_disp = data & 0x07ff07ff;
        if (old_disp != (s->regs.crtc_h_total_disp & 0x07ff0000) &&
            s->mode == EXT_MODE) {
            ati_vga_switch_mode(s);
        }
        break;
    }
    case CRTC_H_SYNC_STRT_WID:
        s->regs.crtc_h_sync_strt_wid = data & 0x17bf1fff;
        break;
    case CRTC_V_TOTAL_DISP:
    {
        uint32_t old_disp = s->regs.crtc_v_total_disp & 0x0fff0000;
        s->regs.crtc_v_total_disp = data & 0x0fff0fff;
        if (old_disp != (s->regs.crtc_v_total_disp & 0x0fff0000) &&
            s->mode == EXT_MODE) {
            ati_vga_switch_mode(s);
        }
        break;
    }
    case CRTC_V_SYNC_STRT_WID:
        s->regs.crtc_v_sync_strt_wid = data & 0x9f0fff;
        break;
    case CRTC_OFFSET:
        s->regs.crtc_offset = data & 0x87fffff8;
        if (ati_is_radeon(s) && s->mode == EXT_MODE) {
            /* a page flip onto a 3D render target: its frame is read back
             * asynchronously; the display keeps the old buffer until the
             * frame has landed in VRAM (a later vblank) */
            uint64_t mc, len;
            ati_scanout_range(s, &mc, &len);
            if (ati_3d_flip(s, mc, len)) {
                s->flip_pending = true;
                break;
            }
        }
        s->flip_pending = false;
        ati_vga_set_offset(&s->vga, s->regs.crtc_offset & 0x07ffffff);
        break;
    case CRTC_OFFSET_CNTL:
        s->regs.crtc_offset_cntl = data; /* FIXME */
        break;
    case CRTC_PITCH:
        data &= 0x07ff07ff;
        if (s->regs.crtc_pitch != data) {
            s->regs.crtc_pitch = data;
            ati_vga_switch_mode(s);
        }
        break;
    case 0xf00 ... 0xfff:
        /* read-only copy of PCI config space so ignore writes */
        break;
    case CUR_OFFSET ... CUR_OFFSET + 3:
    {
        uint32_t t = s->regs.cur_offset;

        ati_reg_write_offs(&t, addr - CUR_OFFSET, data, size);
        t &= 0x87fffff0;
        if (s->regs.cur_offset != t) {
            s->regs.cur_offset = t;
            ati_cursor_define(s);
        }
        break;
    }
    case CUR_HORZ_VERT_POSN ... CUR_HORZ_VERT_POSN + 3:
    {
        uint32_t t = s->regs.cur_hv_pos | (s->regs.cur_offset & BIT(31));

        ati_reg_write_offs(&t, addr - CUR_HORZ_VERT_POSN, data, size);
        s->regs.cur_hv_pos = t & 0x3fff0fff;
        if (t & BIT(31)) {
            s->regs.cur_offset |= t & BIT(31);
        } else if (s->regs.cur_offset & BIT(31)) {
            s->regs.cur_offset &= ~BIT(31);
            ati_cursor_define(s);
        }
        if (!s->cursor_guest_mode &&
            (s->regs.crtc_gen_cntl & CRTC2_CUR_EN) && !(t & BIT(31))) {
            dpy_mouse_set(s->vga.con, s->regs.cur_hv_pos >> 16,
                          s->regs.cur_hv_pos & 0xffff, true);
        }
        break;
    }
    case CUR_HORZ_VERT_OFF:
    {
        uint32_t t = s->regs.cur_hv_offs | (s->regs.cur_offset & BIT(31));

        ati_reg_write_offs(&t, addr - CUR_HORZ_VERT_OFF, data, size);
        s->regs.cur_hv_offs = t & 0x3f003f;
        if (t & BIT(31)) {
            s->regs.cur_offset |= t & BIT(31);
        } else if (s->regs.cur_offset & BIT(31)) {
            s->regs.cur_offset &= ~BIT(31);
            ati_cursor_define(s);
        }
        break;
    }
    case CUR_CLR0 ... CUR_CLR0 + 3:
    {
        uint32_t t = s->regs.cur_color0;

        ati_reg_write_offs(&t, addr - CUR_CLR0, data, size);
        t &= 0xffffff;
        if (s->regs.cur_color0 != t) {
            s->regs.cur_color0 = t;
            ati_cursor_define(s);
        }
        break;
    }
    case CUR_CLR1 ... CUR_CLR1 + 3:
        /*
         * Update cursor unconditionally here because some clients set up
         * other registers before actually writing cursor data to memory at
         * offset so we would miss cursor change unless always updating here
         */
        ati_reg_write_offs(&s->regs.cur_color1, addr - CUR_CLR1, data, size);
        s->regs.cur_color1 &= 0xffffff;
        ati_cursor_define(s);
        break;
    case DST_OFFSET:
            s->regs.dst_offset = data & 0xfffffff0;
        break;
    case DST_PITCH:
            s->regs.dst_pitch = data & 0x3fff;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.dst_tile = (data >> 16) & 1;
        }
        break;
    case DST_TILE:
        if (ati_is_radeon(s)) {
            s->regs.dst_tile = data & 3;
        }
        break;
    case DST_WIDTH:
        s->regs.dst_width = data & 0x3fff;
        ati_2d_blt(s);
        break;
    case DST_HEIGHT:
        s->regs.dst_height = data & 0x3fff;
        break;
    case SRC_X:
        s->regs.src_x = data & 0x3fff;
        break;
    case SRC_Y:
        s->regs.src_y = data & 0x3fff;
        break;
    case DST_X:
        s->regs.dst_x = data & 0x3fff;
        break;
    case DST_Y:
        s->regs.dst_y = data & 0x3fff;
        break;
    case SRC_PITCH_OFFSET:
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.src_offset = (data & 0x1fffff) << 5;
            s->regs.src_pitch = (data & 0x7fe00000) >> 21;
            s->regs.src_tile = data >> 31;
        } else {
            s->regs.src_offset = (data & 0x3fffff) << 10;
            s->regs.src_pitch = (data & 0x3fc00000) >> 16;
            s->regs.src_tile = (data >> 30) & 1;
        }
        break;
    case DST_PITCH_OFFSET:
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.dst_offset = (data & 0x1fffff) << 5;
            s->regs.dst_pitch = (data & 0x7fe00000) >> 21;
            s->regs.dst_tile = data >> 31;
        } else {
            s->regs.dst_offset = (data & 0x3fffff) << 10;
            s->regs.dst_pitch = (data & 0x3fc00000) >> 16;
            s->regs.dst_tile = data >> 30;
        }
        break;
    case SRC_Y_X:
        s->regs.src_x = data & 0x3fff;
        s->regs.src_y = (data >> 16) & 0x3fff;
        break;
    case DST_Y_X:
        s->regs.dst_x = data & 0x3fff;
        s->regs.dst_y = (data >> 16) & 0x3fff;
        break;
    case DST_HEIGHT_WIDTH:
        s->regs.dst_width = data & 0x3fff;
        s->regs.dst_height = (data >> 16) & 0x3fff;
        ati_2d_blt(s);
        break;
    case DP_GUI_MASTER_CNTL:
        s->regs.dp_gui_master_cntl = data & 0xf800000f;
        s->regs.dp_datatype = (data & 0x0f00) >> 8 | (data & 0x30f0) << 4 |
                              (data & 0x4000) << 16;
        s->regs.dp_mix = (data & GMC_ROP3_MASK) | (data & 0x7000000) >> 16;

        if (!(data & GMC_SRC_PITCH_OFFSET_CNTL)) {
            s->regs.src_offset = s->regs.default_offset;
            s->regs.src_pitch = s->regs.default_pitch;
        }
        if (!(data & GMC_DST_PITCH_OFFSET_CNTL)) {
            s->regs.dst_offset = s->regs.default_offset;
            s->regs.dst_pitch = s->regs.default_pitch;
        }
        if (!(data & GMC_SRC_CLIPPING)) {
            s->regs.src_sc_right = s->regs.default_sc_right;
            s->regs.src_sc_bottom = s->regs.default_sc_bottom;
        }
        if (!(data & GMC_DST_CLIPPING)) {
            s->regs.sc_top = 0;
            s->regs.sc_left = 0;
            s->regs.sc_right = s->regs.default_sc_right;
            s->regs.sc_bottom = s->regs.default_sc_bottom;
        }
        break;
    case DST_WIDTH_X:
        s->regs.dst_x = data & 0x3fff;
        s->regs.dst_width = (data >> 16) & 0x3fff;
        ati_2d_blt(s);
        break;
    case SRC_X_Y:
        s->regs.src_y = data & 0x3fff;
        s->regs.src_x = (data >> 16) & 0x3fff;
        break;
    case DST_X_Y:
        s->regs.dst_y = data & 0x3fff;
        s->regs.dst_x = (data >> 16) & 0x3fff;
        break;
    case DST_WIDTH_HEIGHT:
        s->regs.dst_height = data & 0x3fff;
        s->regs.dst_width = (data >> 16) & 0x3fff;
        ati_2d_blt(s);
        break;
    case DST_HEIGHT_Y:
        s->regs.dst_y = data & 0x3fff;
        s->regs.dst_height = (data >> 16) & 0x3fff;
        break;
    case SRC_OFFSET:
            s->regs.src_offset = data & 0xfffffff0;
        break;
    case SRC_PITCH:
            s->regs.src_pitch = data & 0x3fff;
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.src_tile = (data >> 16) & 1;
        }
        break;
    case DP_BRUSH_BKGD_CLR:
        s->regs.dp_brush_bkgd_clr = data;
        break;
    case DP_BRUSH_FRGD_CLR:
        s->regs.dp_brush_frgd_clr = data;
        break;
    case DP_CNTL:
        s->regs.dp_cntl = data;
        break;
    case DP_SRC_FRGD_CLR:
        s->regs.dp_src_frgd_clr = data;
        break;
    case DP_SRC_BKGD_CLR:
        s->regs.dp_src_bkgd_clr = data;
        break;
    case DP_DATATYPE:
        s->regs.dp_datatype = data & 0xe0070f0f;
        break;
    case DP_MIX:
        s->regs.dp_mix = data & 0x00ff0700;
        break;
    case DP_WRITE_MASK:
        s->regs.dp_write_mask = data;
        break;
    case DEFAULT_OFFSET:
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.default_offset = data & 0xfffffff0;
        } else {
            /* Radeon has DEFAULT_PITCH_OFFSET here like DST_PITCH_OFFSET */
            s->regs.default_offset = (data & 0x3fffff) << 10;
            s->regs.default_pitch = (data & 0x3fc00000) >> 16;
            s->regs.default_tile = data >> 30;
        }
        break;
    case DEFAULT_PITCH:
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->regs.default_pitch = data & 0x3fff;
            s->regs.default_tile = (data >> 16) & 1;
        }
        break;
    case DEFAULT_SC_BOTTOM_RIGHT:
        s->regs.default_sc_right = data & 0x3fff;
        s->regs.default_sc_bottom = (data >> 16) & 0x3fff;
        break;
    case SC_TOP_LEFT:
        s->regs.sc_left = data & 0x3fff;
        s->regs.sc_top = (data >> 16) & 0x3fff;
        break;
    case SC_LEFT:
        s->regs.sc_left = data & 0x3fff;
        break;
    case SC_TOP:
        s->regs.sc_top = data & 0x3fff;
        break;
    case SC_BOTTOM_RIGHT:
        s->regs.sc_right = data & 0x3fff;
        s->regs.sc_bottom = (data >> 16) & 0x3fff;
        break;
    case SC_RIGHT:
        s->regs.sc_right = data & 0x3fff;
        break;
    case SC_BOTTOM:
        s->regs.sc_bottom = data & 0x3fff;
        break;
    case SRC_SC_BOTTOM_RIGHT:
        s->regs.src_sc_right = data & 0x3fff;
        s->regs.src_sc_bottom = (data >> 16) & 0x3fff;
        break;
    case SRC_SC_RIGHT:
        s->regs.src_sc_right = data & 0x3fff;
        break;
    case SRC_SC_BOTTOM:
        s->regs.src_sc_bottom = data & 0x3fff;
        break;
    case HOST_DATA0:
    case HOST_DATA1:
    case HOST_DATA2:
    case HOST_DATA3:
    case HOST_DATA4:
    case HOST_DATA5:
    case HOST_DATA6:
    case HOST_DATA7:
    case HOST_DATA_LAST:
        ati_host_data_write(s, addr, data);
        break;
    default:
        break;
    }
}

void ati_reg_write(ATIVGAState *s, hwaddr addr, uint32_t data)
{
    if (addr < sizeof(s->regfile)) {
        ati_mm_write_reg(s, addr, data, 4);
    }
}

/*
 * RetroBoot: registers whose handlers take byte/word accesses themselves;
 * every other sub-word access is merged into a full register access, as the
 * hardware does (the Mac OS Radeon driver writes bytes and halves of many
 * registers: MEM_CNTL, SURFACE_CNTL, CRTC_PITCH, MC_FB_LOCATION...).
 */
static bool ati_reg_takes_bytes(hwaddr addr)
{
    switch (addr & ~3ULL) {
    case MM_DATA:
    case BIOS_0_SCRATCH ... BUS_CNTL - 4:
    case CRTC_GEN_CNTL:
    case CRTC_EXT_CNTL:
    case GPIO_VGA_DDC:
    case GPIO_DVI_DDC:
    case GPIO_MONID:
    case PALETTE_INDEX:
    case PALETTE_DATA:
    case CUR_OFFSET:
    case CUR_HORZ_VERT_POSN:
    case CUR_HORZ_VERT_OFF:
    case CUR_CLR0:
    case CUR_CLR1:
    case CLOCK_CNTL_INDEX:
    case CLOCK_CNTL_DATA:
        return true;
    default:
        return false;
    }
}

static hwaddr ati_radeon_alias(hwaddr addr);

static uint64_t ati_mm_read(void *opaque, hwaddr addr, unsigned int size)
{
    ATIVGAState *s = opaque;

    if (ati_is_radeon(s)) {
        addr = ati_radeon_alias(addr);
    }

    if (size < 4 && ati_is_radeon(s) && !ati_reg_takes_bytes(addr)) {
        uint32_t v = ati_mm_read_reg(opaque, addr & ~3ULL, 4);
        return extract32(v, (addr & 3) * BITS_PER_BYTE, size * BITS_PER_BYTE);
    }
    return ati_mm_read_reg(opaque, addr, size);
}

/*
 * Radeon aliases seen from Mac OS's ATI 2D accelerator: the command
 * processor's primary PIO queue port and the HOST_DATA registers also
 * answer 0x2000 higher (0x3000-0x31fc, 0x37c0-0x37e3).
 */
static hwaddr ati_radeon_alias(hwaddr addr)
{
    if ((addr >= 0x3000 && addr < 0x3200) || (addr >= 0x37c0 && addr < 0x37e4)) {
        return addr - 0x2000;
    }
    return addr;
}

static void ati_mm_write(void *opaque, hwaddr addr, uint64_t data,
                         unsigned int size)
{
    ATIVGAState *s = opaque;

    s->accel.mmio_w++;
    if (ati_is_radeon(s)) {
        addr = ati_radeon_alias(addr);
        /* fast path for host data streams (one MMIO write per dword) */
        if (size == 4 && addr >= HOST_DATA0 && addr <= HOST_DATA_LAST &&
            s->host_data.active) {
            ati_host_data_write(s, addr, data);
            return;
        }
    }

    if (size < 4 && ati_is_radeon(s) && !ati_reg_takes_bytes(addr) &&
        addr < sizeof(s->regfile)) {
        /* write-one-to-clear status: the other bytes write zeros */
        uint32_t base = (addr & ~3ULL) == GEN_INT_STATUS ? 0 :
                        s->regfile[addr / 4];
        uint32_t v = deposit32(base, (addr & 3) * BITS_PER_BYTE,
                               size * BITS_PER_BYTE, data);
        ati_mm_write_reg(opaque, addr & ~3ULL, v, 4);
        return;
    }
    ati_mm_write_reg(opaque, addr, data, size);
}

static const MemoryRegionOps ati_mm_ops = {
    .read = ati_mm_read,
    .write = ati_mm_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void ati_vga_realize(PCIDevice *dev, Error **errp)
{
    ATIVGAState *s = ATI_VGA(dev);
    VGACommonState *vga = &s->vga;
    I2CBus *i2cbus;

#ifndef CONFIG_PIXMAN
    if (s->use_pixman != 0) {
        warn_report("x-pixman != 0, not effective without PIXMAN");
    }
#endif

    if (s->model) {
        int i;
        for (i = 0; i < ARRAY_SIZE(ati_model_aliases); i++) {
            if (!strcmp(s->model, ati_model_aliases[i].name)) {
                s->dev_id = ati_model_aliases[i].dev_id;
                break;
            }
        }
        if (i >= ARRAY_SIZE(ati_model_aliases)) {
            warn_report("Unknown ATI VGA model name, "
                        "using default rage128p");
        }
    }
    if (s->dev_id != PCI_DEVICE_ID_ATI_RAGE128_PF &&
        s->dev_id != PCI_DEVICE_ID_ATI_RADEON_QY &&
        s->dev_id != PCI_DEVICE_ID_ATI_RADEON_QD) {
        error_setg(errp, "Unknown ATI VGA device id, "
                   "only 0x5046, 0x5159 and 0x5144 are supported");
        return;
    }
    pci_set_word(dev->config + PCI_DEVICE_ID, s->dev_id);

    if (ati_is_radeon(s) &&
        s->vga.vram_size_mb < 16) {
        warn_report("Too small video memory for device id");
        s->vga.vram_size_mb = 16;
    }

    /* init vga bits */
    if (!vga_common_init(vga, OBJECT(s), errp)) {
        return;
    }
    vga_init(vga, OBJECT(s), pci_address_space(dev),
             pci_address_space_io(dev), true);
    vga->con = graphic_console_init(DEVICE(s), 0, s->vga.hw_ops, vga);
    if (s->cursor_guest_mode) {
        vga->cursor_invalidate = ati_cursor_invalidate;
        vga->cursor_draw_line = ati_cursor_draw_line;
    }

    /* ddc, edid */
    i2cbus = i2c_init_bus(DEVICE(s), "ati-vga.ddc");
    bitbang_i2c_init(&s->bbi2c, i2cbus);
    i2c_slave_set_address(I2C_SLAVE(&s->i2cddc), 0x50);
    /*
     * RetroBoot: i2c-ddc builds its EDID when the object is created, before
     * this device's xres/yres/xmax/ymax/refresh_rate properties are set, so
     * they never reached the monitor and it always reported 1280x800.  Build
     * it again with them: the detailed timing becomes the size asked for,
     * and Mac OS's ATI driver offers that size as a display mode.
     */
    qemu_edid_generate(s->i2cddc.edid_blob, sizeof(s->i2cddc.edid_blob),
                       &s->i2cddc.edid_info);
    qdev_realize(DEVICE(&s->i2cddc), BUS(i2cbus), &error_abort);

    /* mmio register space */
    memory_region_init_io(&s->mm, OBJECT(s), &ati_mm_ops, s,
                          "ati.mmregs", 0x4000);
    /* io space is alias to beginning of mmregs */
    memory_region_init_alias(&s->io, OBJECT(s), "ati.io", &s->mm, 0, 0x100);

    /*
     * The framebuffer is at the beginning of the linear aperture. For
     * Rage128 the upper half of the aperture is reserved for an AGP
     * window (which we do not emulate.)
     */
    if (!s->linear_aper_sz) {
        if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
            s->linear_aper_sz = ATI_RAGE128_LINEAR_APER_SIZE;
        } else if (s->dev_id == PCI_DEVICE_ID_ATI_RADEON_QD) {
            /*
             * Two apertures, each the size of VRAM (32 MB on the Radeon Mac
             * Edition), so the BAR fits mac99's 256 MB PCI hole next to the
             * register BAR.
             */
            s->linear_aper_sz = MAX(2 * s->vga.vram_size, 16 * MiB);
        } else {
            s->linear_aper_sz = ATI_R100_LINEAR_APER_SIZE;
        }
    }
    if (s->linear_aper_sz > 256 * MiB) {
        error_setg(errp, "x-linear-aper-size is too large (maximum 256 MiB)");
        return;
    }
    if (s->linear_aper_sz < 16 * MiB) {
        error_setg(errp, "x-linear-aper-size is too small (minimum 16 MiB)");
        return;
    }
    memory_region_init(&s->linear_aper, OBJECT(dev), "ati-linear-aperture0",
                       s->linear_aper_sz);
    memory_region_add_subregion(&s->linear_aper, 0, &vga->vram);
    if (s->dev_id == PCI_DEVICE_ID_ATI_RADEON_QD &&
        s->linear_aper_sz >= 2 * vga->vram_size) {
        /* aperture 1: the same VRAM (byte swapping per aperture is not
         * modelled; the CPU view stays big-endian as programmed) */
        memory_region_init_alias(&s->linear_aper1, OBJECT(dev),
                                 "ati-linear-aperture1", &vga->vram, 0,
                                 vga->vram_size);
        memory_region_add_subregion(&s->linear_aper, s->linear_aper_sz / 2,
                                    &s->linear_aper1);
    }

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->linear_aper);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_SPACE_IO, &s->io);
    pci_register_bar(dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mm);

    /* most interrupts are not yet emulated but MacOS needs at least VBlank */
    dev->config[PCI_INTERRUPT_PIN] = 1;
    timer_init_ns(&s->vblank_timer, QEMU_CLOCK_VIRTUAL, ati_vga_vblank_irq, s);
    if (ati_is_radeon(s)) {
        ati_3d_init(s);
    }
}

static void ati_vga_reset(DeviceState *dev)
{
    ATIVGAState *s = ATI_VGA(dev);

    timer_del(&s->vblank_timer);
    ati_vga_update_irq(s);

    /* reset vga */
    vga_common_reset(&s->vga);
    s->mode = VGA_MODE;

    s->host_data.active = false;
    s->host_data.next = 0;
    s->host_data.row = 0;
    s->host_data.col = 0;
    ati_cp_reset(s);
    if (ati_is_radeon(s)) {
        /* The classic Mac OpenGL driver polls the idle-clean mask during
         * glFinish. Hand off an idle engine with both cache-clean waits
         * selected; a zero mask makes every finish reach its retry timeout. */
        s->regfile[WAIT_UNTIL / 4] = BIT(16) | BIT(17);
    }
}

static void ati_vga_exit(PCIDevice *dev)
{
    ATIVGAState *s = ATI_VGA(dev);

    timer_del(&s->vblank_timer);
    graphic_console_close(s->vga.con);
}

static const Property ati_vga_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", ATIVGAState, vga.vram_size_mb, 16),
    DEFINE_PROP_STRING("model", ATIVGAState, model),
    DEFINE_PROP_UINT16("x-device-id", ATIVGAState, dev_id,
                       PCI_DEVICE_ID_ATI_RAGE128_PF),
    DEFINE_PROP_BOOL("guest_hwcursor", ATIVGAState, cursor_guest_mode, false),
    /* this is a debug option, prefer PROP_UINT over PROP_BIT for simplicity */
    DEFINE_PROP_UINT8("x-pixman", ATIVGAState, use_pixman, DEFAULT_X_PIXMAN),
    DEFINE_PROP_UINT64("x-linear-aper-size", ATIVGAState, linear_aper_sz, 0),
    DEFINE_PROP_UINT32("x-debug", ATIVGAState, debug, 0),
    DEFINE_PROP_UINT32("x-vblank-hz", ATIVGAState, vblank_hz, 60),
    DEFINE_EDID_PROPERTIES(ATIVGAState, i2cddc.edid_info),
};

static void ati_vga_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    device_class_set_legacy_reset(dc, ati_vga_reset);
    device_class_set_props(dc, ati_vga_properties);
    dc->hotpluggable = false;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);

    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_ATI;
    k->device_id = PCI_DEVICE_ID_ATI_RAGE128_PF;
    k->romfile = "vgabios-ati.bin";
    k->realize = ati_vga_realize;
    k->exit = ati_vga_exit;
}

/* RetroBoot: acceleration counters, evidence that the engines are used */
static char *ati_get_accel_stats(Object *o, Error **errp)
{
    ATIVGAState *s = ATI_VGA(o);

    return g_strdup_printf("ops2d=%" PRIu64 " pixels2d=%" PRIu64
                           " packets2d=%" PRIu64 " packets3=%" PRIu64
                           " mmio_w=%" PRIu64 " hostdata_dw=%" PRIu64
                           " pio_dw=%" PRIu64 " ring_dw=%" PRIu64
                           " ib_dw=%" PRIu64 " vblanks=%" PRIu64 " %s",
                           s->accel.ops2d, s->accel.pixels2d,
                           s->accel.packets2d, s->cp.packets3,
                           s->accel.mmio_w, s->accel.hd_dw, s->accel.pio_dw,
                           s->accel.ring_dw, s->accel.ib_dw, s->accel.vblanks,
                           s->r3d ? ati_3d_stats(s) : "");
}

/* ClassicMac: the host GL renderer the 3D engine uses ("" when 3D is off) */
static char *ati_get_gl_renderer(Object *o, Error **errp)
{
    ATIVGAState *s = ATI_VGA(o);

    return g_strdup(ati_3d_gl_info(s));
}

static void ati_vga_init(Object *o)
{
    ATIVGAState *s = ATI_VGA(o);

    object_property_add_str(o, "x-accel-stats", ati_get_accel_stats, NULL);
    object_property_add_str(o, "x-gl-renderer", ati_get_gl_renderer, NULL);

    object_initialize_child(o, "edid", &s->i2cddc, TYPE_I2CDDC);
    object_property_set_description(o, "x-pixman", "Use pixman for: "
                                    "1: fill, 2: blit");
}

static const TypeInfo ati_vga_info = {
    .name = TYPE_ATI_VGA,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(ATIVGAState),
    .class_init = ati_vga_class_init,
    .instance_init = ati_vga_init,
    .interfaces = (InterfaceInfo[]) {
          { INTERFACE_CONVENTIONAL_PCI_DEVICE },
          { },
    },
};

static void ati_vga_register_types(void)
{
    type_register_static(&ati_vga_info);
}

type_init(ati_vga_register_types)
