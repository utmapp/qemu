/*
 * PPC Mac GPU - ATI Radeon 9200 (RV280) compatible display device for QEMU
 *
 * Emulates enough of the RV280 register interface for Mac OS X Tiger/Leopard
 * ATI kext attachment and accelerated framebuffer operation.
 *
 * Copyright (c) 2024
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef HW_DISPLAY_PPC_MAC_GPU_H
#define HW_DISPLAY_PPC_MAC_GPU_H

#include "hw/pci/pci_device.h"
#include "hw/i2c/bitbang_i2c.h"
#include "hw/display/i2c-ddc.h"
#include "ui/console.h"
#include "qom/object.h"

/* Defined in hw/display/ppc_mac_gpu_renderer.h, private to the device. */
struct R200Vertex;

/* ========================================================================
 * PCI Identity - ATI Radeon 9200 PRO (RV280)
 * ======================================================================== */

#define PPC_MAC_GPU_PCI_VENDOR_ID      0x1002  /* ATI Technologies */
#define PPC_MAC_GPU_PCI_DEVICE_ID      0x5960  /* RV280 [Radeon 9200 PRO] */
#define PPC_MAC_GPU_PCI_REVISION       0x01

/* ATI Radeon 9700 PRO Mac Edition (R300 "ND", FCode ATY,GoldenEye). */
#define PPC_MAC_GPU_R300_DEVICE_ID     0x4E44
/* ATI Radeon 9000 PRO (RV250): an R200-class card Mac OS X 10.2 knows
 * (ATIRadeon8500.kext IOPCIMatch lists 0x49661002; 10.2 has no R300 driver). */
#define PPC_MAC_GPU_RV250_DEVICE_ID    0x4966
#define PPC_MAC_GPU_PCI_CLASS          PCI_CLASS_DISPLAY_VGA  /* 0x0300 */

/* ========================================================================
 * Memory Map - BAR sizes
 * ======================================================================== */

#define PPC_MAC_GPU_VRAM_SIZE_DEFAULT  (128 * 1024 * 1024)  /* 128 MB */
#define PPC_MAC_GPU_VRAM_BAR           0   /* BAR0: VRAM aperture */
#define PPC_MAC_GPU_IO_BAR             1   /* BAR1: I/O register alias */
#define PPC_MAC_GPU_MMIO_BAR           2   /* BAR2: MMIO registers */

#define PPC_MAC_GPU_IO_SIZE            0x100    /* 256 bytes I/O */
#define PPC_MAC_GPU_MMIO_SIZE          0x10000  /* 64 KB MMIO */

/* ========================================================================
 * Register Offsets - R200/RV280 family
 *
 * Reference: Mesa r200_reg.h, xf86-video-ati, Linux radeon_reg.h
 * Only registers needed for MVP are defined here.
 * ======================================================================== */

/* Indexed access */
#define R200_MM_INDEX                  0x0000
#define R200_MM_DATA                   0x0004

/* General */
#define R200_BUS_CNTL                  0x0030
#define R200_GEN_INT_CNTL             0x0040
#define R200_GEN_INT_STATUS           0x0044
#define R200_CNTL_STATUS              0x0048

/* Clock/PLL (stub with fixed values) */
#define R200_CLOCK_CNTL_INDEX         0x0008
#define R200_CLOCK_CNTL_DATA          0x000C

/* Bus/MC */
#define R200_MC_FB_LOCATION           0x0148
#define R200_MC_AGP_LOCATION          0x014C
#define R200_MC_STATUS                0x0150

/* Config */
#define R200_CONFIG_APER_0_BASE       0x0100
#define R200_CONFIG_APER_1_BASE       0x0104
#define R200_CONFIG_APER_SIZE         0x0108
#define R200_CONFIG_REG_1_BASE        0x010C
#define R200_CONFIG_REG_APER_SIZE     0x0110
#define R200_CONFIG_MEMSIZE           0x00F8
#define R200_CONFIG_CNTL              0x00E0

/* PCI config space mirror in MMIO (0x0F00-0x0F3F) */
#define R200_PCI_MIRROR_BASE          0x0F00
#define R200_PCI_MIRROR_END           0x0F40  /* exclusive */

/* GPIO / I2C / DDC */
#define R200_GPIO_VGA_DDC             0x0060
#define R200_GPIO_DVI_DDC             0x0064
#define R200_GPIO_MONID               0x0068

/* RBBM (register block backbone manager) */
#define R200_RBBM_STATUS              0x0E40
#define R200_RBBM_SOFT_RESET          0x00F0

/* GUI engine status */
#define R200_GUI_STAT                  0x1740

/* Host path */
#define R200_HOST_PATH_CNTL           0x0130

/* Surface/tiling */
#define R200_SURFACE_CNTL             0x0B00
#define R200_SURFACE0_INFO            0x0B0C
#define R200_SURFACE0_LOWER_BOUND     0x0B04
#define R200_SURFACE0_UPPER_BOUND     0x0B08

/* CRTC registers */
#define R200_CRTC_GEN_CNTL            0x0050
#define R200_CRTC_EXT_CNTL            0x0054
#define R200_CRTC_STATUS              0x005C
#define R200_CRTC_H_TOTAL_DISP        0x0200
#define R200_CRTC_H_SYNC_STRT_WID     0x0204
#define R200_CRTC_V_TOTAL_DISP        0x0208
#define R200_CRTC_V_SYNC_STRT_WID     0x020C
#define R200_CRTC_OFFSET              0x0224
#define R200_CRTC_OFFSET_CNTL         0x0228
#define R200_CRTC_OFFSET_BIAS         0x20   /* see the CRTC_OFFSET read */
#define R200_CRTC_PITCH               0x022C
#define R200_CRTC_GUI_TRIG_VLINE      0x0218

/* DAC */
#define R200_DAC_CNTL                  0x0058
#define R200_DAC_CNTL2                 0x007C
#define R200_DAC_RANGE_CNTL           0x0484

/* Palette */
#define R200_PALETTE_INDEX             0x00B0
#define R200_PALETTE_DATA              0x00B4
#define R200_PALETTE_30_DATA           0x00B8

/* Cursor */
#define R200_CUR_OFFSET               0x0260
#define R200_CUR_HORZ_VERT_POSN       0x0264
#define R200_CUR_HORZ_VERT_OFF        0x0268
#define R200_CUR_CLR0                 0x026C
#define R200_CUR_CLR1                 0x0270

/* Display */
#define R200_DISP_MISC_CNTL           0x0D00
#define R200_DISP_PWR_MAN             0x0D08
#define R200_DISP_MERGE_CNTL          0x0D60
#define R200_DISP_OUTPUT_CNTL         0x0D64

/* FP (flat panel) */
#define R200_FP_GEN_CNTL              0x0284
#define R200_FP_CRTC_H_TOTAL_DISP     0x0250
#define R200_FP_CRTC_V_TOTAL_DISP     0x0254
#define R200_FP_H_SYNC_STRT_WID       0x02C4
#define R200_FP_V_SYNC_STRT_WID       0x02C8

/* BIOS scratch registers (used by firmware/driver communication) */
#define R200_BIOS_0_SCRATCH           0x0010
#define R200_BIOS_1_SCRATCH           0x0014
#define R200_BIOS_2_SCRATCH           0x0018
#define R200_BIOS_3_SCRATCH           0x001C
#define R200_BIOS_4_SCRATCH           0x0020
#define R200_BIOS_5_SCRATCH           0x0024
#define R200_BIOS_6_SCRATCH           0x0028
#define R200_BIOS_7_SCRATCH           0x002C

/* AGP */
#define R200_AGP_BASE                  0x0170
#define R200_AGP_CNTL                  0x0174
#define R200_AGP_APER_SIZE             0x0178
#define R200_AGP_COMMAND               0x0F60
#define R200_AGP_STATUS                0x0F5C

/* 2D engine (DST/SRC/DP) - Phase 5 */
#define R200_DST_OFFSET               0x1404
#define R200_DST_PITCH                0x1408
#define R200_DST_WIDTH                0x140C
#define R200_DST_HEIGHT               0x1410
#define R200_SRC_PITCH_OFFSET         0x1428  /* combined src pitch+offset */
#define R200_DST_PITCH_OFFSET         0x142C  /* combined dst pitch+offset */
#define R200_SRC_Y_X                  0x1434
#define R200_DST_Y_X                  0x1438
#define R200_SRC_X_Y                  0x1590  /* bits[31:16]=X, bits[15:0]=Y */
#define R200_DST_X_Y                  0x1594  /* bits[31:16]=X, bits[15:0]=Y */
#define R200_DST_HEIGHT_WIDTH         0x143C  /* alt blit trigger: h|w */
#define R200_DST_WIDTH_HEIGHT         0x1598  /* blit trigger: w|h */

#define R200_SRC_OFFSET               0x15AC
#define R200_SRC_PITCH                0x15B0

#define R200_DP_GUI_MASTER_CNTL       0x146C
/*
 * DP_GUI_MASTER_CNTL bits used by the PM4 blit packets.
 *
 * Bits 0 and 1 are the SRC / DST pitch-offset controls.  Set: use an
 * explicit pitch-offset — the SRC_/DST_PITCH_OFFSET register for a
 * register-triggered blit, or a word carried inside a PM4 packet (which
 * makes the packet header variable length).  Clear: use
 * DEFAULT_PITCH_OFFSET (0x16E0).  A clear bit does NOT mean "the
 * SRC_/DST_PITCH_OFFSET register" — treating it that way corrupts the first
 * frame of every window drag.
 */
#define R200_GMC_SRC_PITCH_OFFSET_CNTL (1 << 0)
#define R200_GMC_DST_PITCH_OFFSET_CNTL (1 << 1)
/* Packet clip fields present (ATI radeon_reg.h): bit 2 = one SRC_SC_BOTTOM_RIGHT
 * dword, bit 3 = DST_SC_TOP_LEFT + DST_SC_BOTTOM_RIGHT dwords.  Bit 28 is
 * CLR_CMP_CNTL_DIS, which Apple's driver always sets. */
#define R200_GMC_SRC_CLIPPING          (1 << 2)
#define R200_GMC_DST_CLIPPING          (1 << 3)
#define R200_GMC_CLR_CMP_CNTL_DIS      (1 << 28)

#define R200_DP_BRUSH_FRGD_CLR        0x147C
#define R200_DP_BRUSH_BKGD_CLR        0x1478
#define R200_DP_SRC_FRGD_CLR          0x15D8
#define R200_DP_SRC_BKGD_CLR          0x15DC
#define R200_DP_WRITE_MSK             0x16CC
#define R200_DP_CNTL                  0x16C0
#define R200_DP_DATATYPE              0x16C4
#define R200_DP_MIX                   0x16C8

#define R200_DEFAULT_PITCH_OFFSET     0x16E0  /* combined default pitch+offset */
#define R200_DEFAULT_SC_BOTTOM_RIGHT  0x16E8
#define R200_SC_TOP_LEFT              0x16EC
#define R200_SC_BOTTOM_RIGHT          0x16F0

/* CP (command processor) - Phase 4 */
#define R200_CP_ME_CNTL               0x0086
#define R200_CP_RB_BASE                0x0700
#define R200_CP_RB_CNTL               0x0704
#define R200_CP_RB_RPTR_ADDR          0x070C
#define R200_CP_RB_RPTR               0x0710
#define R200_CP_RB_WPTR               0x0714
#define R200_CP_RB_WPTR_DELAY         0x0718
#define R200_CP_RB_RPTR_WR            0x071C  /* when RB_CNTL.RB_RPTR_WR_ENA */
#define R200_RB_RPTR_WR_ENA           (1u << 31)          /* in CP_RB_CNTL */
#define R200_CP_CSQ_CNTL              0x0740
/* 0x0770/0x0774 are SCRATCH_UMSK/SCRATCH_ADDR on R200 */
#define R200_SCRATCH_UMSK             0x0770
#define R200_SCRATCH_ADDR             0x0774

/* Indirect Buffer (IB) registers */
#define R200_CP_IB_BASE                0x0738
#define R200_CP_IB_BUFSZ              0x073C

/* PM4 PIO FIFO data registers */
/*
 * Where a driver pushes command packets by hand (rather than through the
 * ring): the primary queue fills 0x1000-0x12FF and the second 0x1300-0x13FF.
 * Only the first eight words were accepted here, so every longer burst lost
 * its tail -- including the write that says "this far is done", which Halo
 * then waited on for ever while its sound looped.
 */
#define R200_PM4_FIFO_DATA_BASE       0x1000
#define R200_PM4_FIFO_DATA_END        0x1400  /* exclusive */

/* Scratch registers */
#define R200_SCRATCH_REG0             0x15E0
#define R200_SCRATCH_REG1             0x15E4
#define R200_SCRATCH_REG2             0x15E8
#define R200_SCRATCH_REG3             0x15EC
#define R200_SCRATCH_REG4             0x15F0
#define R200_SCRATCH_REG5             0x15F4
#define R200_CP_ME_RAM_ADDR           0x07D4
#define R200_CP_ME_RAM_RADDR          0x07D8
#define R200_CP_ME_RAM_DATAH          0x07DC
#define R200_CP_ME_RAM_DATAL          0x07E0
#define R200_CP_CSQ_IND_ADDR          0x07F0
#define R200_CP_CSQ_IND_DATA          0x07F4
#define R200_CP_CSQ_STAT              0x07F8

/* HOST_DATA */
#define R200_HOST_DATA0               0x17C0
#define R200_HOST_DATA1               0x17C4
#define R200_HOST_DATA2               0x17C8
#define R200_HOST_DATA3               0x17CC
#define R200_HOST_DATA4               0x17D0
#define R200_HOST_DATA5               0x17D4
#define R200_HOST_DATA6               0x17D8
#define R200_HOST_DATA7               0x17DC
#define R200_HOST_DATA_LAST           0x17E0

/* Wait/sync */
#define R200_WAIT_UNTIL               0x1720
#define R200_ISYNC_CNTL               0x1724

/* Misc registers seen in kext init */
#define R200_RB3D_DSTCACHE_CTLSTAT   0x325C
#define R200_RB3D_DSTCACHE_MODE      0x3258
#define R200_CP_STAT                  0x064C
#define R200_RBBM_CNTL               0x1710

/* GART/AGP extra registers */
#define R200_AGP_BASE_2               0x015C
#define R200_MC_AGP_LOCATION_2        0x01D0
#define R200_AIC_CTRL                 0x01D0  /* AGP Intelligent Controller */
#define R200_AIC_PT_BASE              0x01D8  /* GART page table base (phys) */
#define R200_AIC_LO_ADDR              0x01DC  /* GART aperture low (GPU addr) */
#define R200_AIC_HI_ADDR              0x01E0  /* GART aperture high (GPU addr) */

/* CRTC2 pitch and offset */
#define R200_CRTC2_OFFSET_CNTL       0x0328
#define R200_CRTC2_PITCH              0x023C
#define R200_FP2_GEN_CNTL            0x033C

/* GPIOPAD registers */
#define R200_GPIOPAD_MASK             0x0198
#define R200_GPIOPAD_A                0x019C
#define R200_GPIOPAD_EN               0x0198
#define R200_VIPH_CONTROL             0x0C40

/* Overlay/subpicture */
#define R200_OV0_SCALE_CNTL          0x0420
#define R200_SUBPIC_CNTL             0x0540
#define R200_SUBPIC2_CNTL            0x043C
#define R200_CAP0_TRIG_CNTL          0x0950
#define R200_DISP2_MERGE_CNTL        0x0D68

/* VGA/CRT registers */
#define R200_GENMO_WT                 0x03C2
#define R200_CRTC_STATUS_ALT          0x0500
#define R200_CUR2_OFFSET              0x0360

/* DST_WIDTH_HEIGHT (trigger register) */
#define R200_DST_WIDTH_HEIGHT_ALT    0x1598

/* CRTC_VLINE_CRNT_VLINE - current scanline position */
#define R200_CRTC_VLINE_CRNT_VLINE   0x0210
#define R200_CRTC_CRNT_FRAME         0x0214

/* ========================================================================
 * VBE DISPI registers for qemu_vga.ndrv compatibility
 *
 * qemu_vga.ndrv accesses these at MMIO offsets:
 *   VGA ports at:  boardRegAddress + port + 0x400 - 0x3C0
 *   VBE DISPI at:  boardRegAddress + (reg << 1) + 0x500
 *   Extended at:   boardRegAddress + (reg << 2) + 0x600
 * ======================================================================== */

/* VBE DISPI register indices */
#define VBE_DISPI_INDEX_ID               0x0
#define VBE_DISPI_INDEX_XRES             0x1
#define VBE_DISPI_INDEX_YRES             0x2
#define VBE_DISPI_INDEX_BPP              0x3
#define VBE_DISPI_INDEX_ENABLE           0x4
#define VBE_DISPI_INDEX_BANK             0x5
#define VBE_DISPI_INDEX_VIRT_WIDTH       0x6
#define VBE_DISPI_INDEX_VIRT_HEIGHT      0x7
#define VBE_DISPI_INDEX_X_OFFSET         0x8
#define VBE_DISPI_INDEX_Y_OFFSET         0x9
#define VBE_DISPI_INDEX_VIDEO_MEMORY_64K 0xa
#define VBE_DISPI_INDEX_NB               0xb

#define VBE_DISPI_ID5                    0xB0C5

#define VBE_DISPI_DISABLED               0x00
#define VBE_DISPI_ENABLED                0x01
#define VBE_DISPI_LFB_ENABLED           0x40

/* MMIO offsets for VBE/VGA register blocks */
#define PPC_MAC_GPU_VGA_OFFSET           0x0400
#define PPC_MAC_GPU_VGA_END              0x0420
#define PPC_MAC_GPU_VBE_OFFSET           0x0500
#define PPC_MAC_GPU_VBE_END              0x0516
#define PPC_MAC_GPU_EXT_OFFSET           0x0600
#define PPC_MAC_GPU_EXT_END              0x0620

/* RB3D / 3D pipe registers */
#define R200_RB3D_CNTL               0x1C3C
#define R200_SE_CNTL                 0x1C4C

/* ========================================================================
 * CRTC_GEN_CNTL bit definitions
 * ======================================================================== */

#define R200_CRTC_DBL_SCAN_EN          (1 << 0)
#define R200_CRTC_INTERLACE_EN         (1 << 1)
#define R200_CRTC_CSYNC_EN             (1 << 4)
#define R200_CRTC_PIX_WIDTH_MASK       (0xF << 8)
#define R200_CRTC_PIX_WIDTH_SHIFT      8
#define R200_CRTC_CUR_EN               (1 << 16)
#define R200_CRTC_CUR_MODE_MASK        (7 << 17)
#define R200_CRTC_EXT_DISP_EN          (1 << 24)
#define R200_CRTC_EN                   (1 << 25)
#define R200_CRTC_DISP_REQ_EN_B       (1 << 26)

/* Pixel width encodings */
#define R200_CRTC_PIX_WIDTH_4BPP       (1 << 8)
#define R200_CRTC_PIX_WIDTH_8BPP       (2 << 8)
#define R200_CRTC_PIX_WIDTH_15BPP      (3 << 8)
#define R200_CRTC_PIX_WIDTH_16BPP      (4 << 8)
#define R200_CRTC_PIX_WIDTH_24BPP      (5 << 8)
#define R200_CRTC_PIX_WIDTH_32BPP      (6 << 8)

/* Interrupt bits */
#define R200_CRTC_VBLANK_INT           (1 << 0)
#define R200_CRTC_VLINE_INT            (1 << 1)
#define R200_CRTC_VSYNC_INT            (1 << 2)

/* RBBM_STATUS bits */
#define R200_RBBM_FIFOCNT_MASK         0x007F
#define R200_RBBM_ACTIVE               (1 << 31)

/* CRTC_STATUS bits */
#define R200_CRTC_VBLANK_SAVE          (1 << 1)

/* MC_STATUS bits */
#define R200_MC_IDLE                   (1 << 0)
#define R200_MC_BUSY                   (0)

/* ========================================================================
 * Display mode tracking
 * ======================================================================== */

#define PPC_MAC_GPU_MODE_VGA           0
#define PPC_MAC_GPU_MODE_EXT           1

/* ========================================================================
 * Device state
 * ======================================================================== */

/* Forward declaration for renderer interface */
typedef struct PPCMacGPURenderer PPCMacGPURenderer;

/* Register file - stores all emulated register values */
typedef struct PPCMacGPURegs {
    /* Indexed access */
    uint32_t mm_index;

    /* BIOS scratch */
    uint32_t bios_scratch[8];

    /* Interrupt */
    uint32_t gen_int_cntl;
    uint32_t gen_int_status;

    /* CRTC */
    uint32_t crtc_gen_cntl;
    uint32_t crtc_ext_cntl;
    uint32_t crtc_status;
    uint32_t crtc_h_total_disp;
    uint32_t crtc_h_sync_strt_wid;
    uint32_t crtc_v_total_disp;
    uint32_t crtc_vline;          /* the line the guest asked to be told about */
    uint32_t crtc_v_sync_strt_wid;
    uint32_t crtc_offset;
    uint32_t crtc_offset_cntl;
    uint32_t crtc_pitch;

    /* DAC */
    uint32_t dac_cntl;
    uint32_t dac_cntl2;

    /* Config */
    uint32_t config_cntl;
    uint32_t config_memsize;
    uint32_t mc_fb_location;
    uint32_t mc_agp_location;
    uint32_t mc_status;

    /* Bus */
    uint32_t bus_cntl;
    uint32_t host_path_cntl;

    /* Surface */
    uint32_t surface_cntl;

    /* AGP */
    uint32_t agp_base;
    uint32_t agp_cntl;
    uint32_t agp_command;
    uint32_t agp_status;

    /* RBBM */
    uint32_t rbbm_status;

    /* Display */
    uint32_t disp_misc_cntl;
    uint32_t disp_output_cntl;
    uint32_t disp_merge_cntl;
    uint32_t fp_gen_cntl;

    /* Cursor */
    uint32_t cur_offset;
    uint32_t cur_horz_vert_posn;
    uint32_t cur_horz_vert_off;
    uint32_t cur_clr0;
    uint32_t cur_clr1;

    /* 2D engine */
    uint32_t dst_offset;
    uint32_t dst_pitch;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t dst_y_x;
    uint32_t src_offset;
    uint32_t src_pitch;
    uint32_t src_y_x;
    uint32_t src_pitch_offset;     /* 0x1428: combined src pitch+offset */
    uint32_t dst_pitch_offset;     /* 0x142C: combined dst pitch+offset */
    /* 0x16E0: pitch-offset used by any 2D operation whose
     * DP_GUI_MASTER_CNTL GMC_SRC/DST_PITCH_OFFSET_CNTL bit is CLEAR.
     * Apple's R200 driver points it at the compositor staging buffer. */
    uint32_t default_pitch_offset;
    uint32_t dst_width_height;     /* 0x1598: blit trigger */
    uint32_t dp_gui_master_cntl;
    uint32_t dp_brush_frgd_clr;
    uint32_t dp_brush_bkgd_clr;
    uint32_t dp_src_frgd_clr;
    uint32_t dp_src_bkgd_clr;
    uint32_t dp_write_msk;
    uint32_t dp_cntl;
    uint32_t dp_datatype;
    uint32_t dp_mix;
    uint32_t sc_top_left;
    uint32_t sc_bottom_right;
    uint32_t default_sc_bottom_right;
    uint32_t wait_until;
    uint32_t isync_cntl;
    uint32_t gui_stat;

    /* GART (AGP Intelligent Controller) */
    uint32_t aic_ctrl;             /* 0x01D0: GART enable */
    uint32_t aic_pt_base;          /* 0x01D8: page table phys addr */
    uint32_t aic_lo_addr;          /* 0x01DC: aperture low (GPU addr) */
    uint32_t aic_hi_addr;          /* 0x01E0: aperture high (GPU addr) */

    /* CP (command processor) */
    uint32_t cp_rb_base;
    uint32_t cp_rb_cntl;
    uint32_t cp_rb_rptr;
    uint32_t cp_rb_wptr;
    uint32_t cp_me_cntl;
    uint32_t cp_csq_cntl;
    uint32_t cp_me_ram_addr;
    uint32_t cp_ib_base;           /* 0x0738: Indirect Buffer base address */
    uint32_t cp_ib_bufsz;          /* 0x073C: Indirect Buffer size (dwords) */

    /* Scratch registers + writeback */
    uint32_t scratch_reg[6];       /* SCRATCH_REG0-5 (0x15E0-0x15F4) */
    uint32_t scratch_umsk;         /* 0x0770: which scratch regs to writeback */
    uint32_t scratch_addr;         /* 0x0774: VRAM addr for writeback */
    uint64_t stall_draws;          /* draws carried out, for the stall trace */
    uint64_t stall_pio_dwords;     /* command words pushed in by hand */
    uint64_t stall_ib_done;        /* command buffers fetched and carried out */
    uint64_t stall_ib_dwords;
    uint64_t stall_ib_lost;        /* ones whose memory could not be found */
    bool csq_just_submitted;       /* something was pushed since the last look */
    uint64_t stall_irqs;           /* display interrupts raised */
    uint64_t stall_ring_dwords;    /* command words taken from the ring */
    uint64_t stall_draws_seen;
    uint32_t cp_rb_rptr_addr;      /* 0x070C: where to keep the ring read
                                    * pointer's copy in memory, for a driver
                                    * that reads it from there */

    /* Clock (stub) */
    uint32_t clock_cntl_index;
    uint32_t clock_cntl_data;

    /* GPIO / DDC */
    uint32_t gpio_vga_ddc;
    uint32_t gpio_dvi_ddc;
    uint32_t gpio_monid;

    /* Palette */
    uint32_t palette_index;
    uint32_t palette[256];

    /* VBE DISPI state (for qemu_vga.ndrv compatibility) */
    uint16_t vbe_regs[VBE_DISPI_INDEX_NB];

    /* VGA DAC state (for palette via qemu_vga.ndrv) */
    uint8_t vga_dac_read_index;
    uint8_t vga_dac_write_index;
    uint8_t vga_dac_sub_index;  /* 0=R, 1=G, 2=B cycle */
    uint8_t vga_dac_palette[256][3];  /* 6-bit RGB per entry */
    uint8_t vga_ar_index;       /* Attribute controller index */
    uint8_t vga_ar_flip_flop;   /* AR index/data flip flop */

    /* 3D engine register shadow (Phase 0: instrumentation)
     * Covers registers 0x1C00-0x3FFF in 4-byte slots.
     * Index: (addr - 0x1C00) / 4, total = 0x2400/4 = 0x900 = 2304 entries.
     * Used to store all 3D register writes so reads return stored values. */
    uint32_t regs_3d[2304];

    /* R200 TCL constant memory, written through the SE_TCL_VECTOR_INDX/DATA
     * (0x2200/0x2204) and SCALAR_INDX/DATA (0x2208/0x220C) ports.  Vector
     * address 0x80 + 4*N holds 4x4 matrix N as four row vectors. */
    /* TCL vector memory.  Fixed-function matrices and lights live here,
     * and so do vertex programs (instructions at 0x080 and 0x180,
     * their constants at 0x000 and 0x100), which is why the two must
     * never be read at the same time.  Wider than the 0x1C0 the chip
     * documents, so a driver writing higher can't quietly land on a
     * matrix. */
    uint32_t tcl_vec[0x800][4];
    uint32_t tcl_vec_addr, tcl_vec_stride, tcl_vec_comp;
    uint32_t tcl_scalar[0x200];
    uint32_t tcl_scalar_addr, tcl_scalar_stride;

    /* Direct-R200 asynchronous fences: scratch-register writebacks issued
     * while GPU work is in flight are queued and performed (in order) once
     * the Metal batch they follow has completed. */
    struct {
        uint32_t seq;
        uint32_t reg;
        uint32_t val;
    } r200_fence_q[64];
    uint32_t r200_fence_n;
    uint32_t r200_fence_last_seq;   /* newest batch a queued fence waits on */
    uint32_t r200_fence_done;       /* newest completed batch (atomic) */

    /* Count of ring-buffer status polls, returned in CP_CSQ_STAT so the
     * guest sees the queue draining rather than a constant value. */
    uint32_t cp_csq_stat_counter;

} PPCMacGPURegs;

/* Display mode information derived from CRTC registers */
typedef struct PPCMacGPUDisplayMode {
    uint32_t width;
    uint32_t height;
    uint32_t bpp;        /* bits per pixel: 8, 15, 16, 24, 32 */
    uint32_t stride;     /* bytes per scanline */
    uint32_t offset;     /* VRAM offset for scanout */
    pixman_format_code_t format;
    bool crtc_ext;       /* true when CRTC extended mode active (kext) */
} PPCMacGPUDisplayMode;

#include "../../../hw/display/r300/r300_draw.h"

#define TYPE_PPC_MAC_GPU "ppc-mac-gpu"
#define TYPE_ATI_RADEON_9700 "ati-radeon-9700"
#define TYPE_ATI_RADEON_9000 "ati-radeon-9000"
OBJECT_DECLARE_SIMPLE_TYPE(PPCMacGPUState, PPC_MAC_GPU)

struct PPCMacGPUState {
    /* Parent */
    PCIDevice pci;

    /* Scratch for vertex assembly, reused across draws.
     *
     * These were a g_new0/g_free pair inside the draw path.  Warcraft issues
     * on the order of 7,000 draws a second, so that was 14,000 allocations a
     * second plus a full zeroing of a vertex array each time.
     *
     * The zeroing itself is still needed: the assembly loop only writes the
     * attributes a vertex stream actually carries, and leaves spec[] and the
     * first three components of each tex[] to arrive as zero.  Only the part
     * in use is cleared, rather than the whole high-water allocation.
     *
     * Not thread-safe, and deliberately so: one draw is in flight at a time.
     */
    struct R200Vertex *draw_verts;
    uint32_t draw_verts_cap;      /* in vertices */
    uint32_t *draw_idx;
    size_t draw_idx_cap;          /* in indices */

    /* Display console */
    QemuConsole *con;

    /* Memory regions */
    MemoryRegion vram;          /* BAR0: VRAM aperture */
    MemoryRegion vram_bar;      /* r300: BAR0 = aperture 0 (vram) + aperture 1 */
    MemoryRegion vram_ap1;      /* r300: aperture 1, byte-swapping view of vram */
    uint32_t r300_surf[32];     /* r300: SURFACE0..7 regs (0x0B00 + 4*i) */
    MemoryRegion r300_watch;    /* $R300_SURFWATCH: logs CPU access to surface 0 */
    bool r300_watch_on;
    uint32_t r300_watch_base, r300_watch_logged;
    MemoryRegion io;            /* BAR1: I/O register alias */
    MemoryRegion mmio;          /* BAR2: MMIO register space */
    MemoryRegion cp_wptr_mr;    /* CP_RB_WPTR, lock-free (CP thread) */
    MemoryRegion rom_mr;        /* ROM BAR: traced expansion ROM */

    /* ROM data */
    uint8_t *rom_data;
    uint32_t rom_size;

    /* Full BIOS ROM path (separate from romfile which has the NDRV) */
    char *biosrom;

    /* "renderer" property: metal, vulkan, or unset (the host's default) */
    char *renderer_name;

    /* Device configuration */
    bool r300;                  /* ati-radeon-9700: R300 instead of RV280 */
    bool rv250;                 /* ati-radeon-9000: the card Mac OS X 10.2 drives */
    struct R300State *r3;       /* R300 3D register file (r300 only) */
    struct R300Arrays r3_arrays;    /* last 3D_LOAD_VBPNTR */
    FILE *r3_dump;              /* $R300_DUMP: state dumps at draws */
    uint32_t r3_trace_lines;    /* register trace after the first draw */
    uint32_t r300_src_swap;     /* 0x15D4: 2D upload source endian swap */
    bool r3_zpass_active;       /* ZB_ZPASS_DATA written, count Z-pass samples */
    uint32_t r3_indx_vf;        /* DRAW_INDX_2 waiting for INDX_BUFFER (0: none) */
    uint32_t r3_indx_buf[3];    /* INDX_BUFFER waiting for its DRAW_INDX_2 */
    bool r3_indx_buf_pending;
    uint32_t r3_zb_offset, r3_zb_height;   /* depth buffer extent seen at draws */
    uint32_t r3_zb_ns;          /* GB_AA_CONFIG samples at the last depth draw */
    uint32_t r3_cb_aa_off, r3_cb_aa_ns; /* last multisampled colour buffer, samples */
    bool r3_zconv;              /* depth buffer currently in the card's MSAA layout */
    bool r3_zconv_rows_ns;      /* ... converted from one row per sample (else
                                   from one sample, which all samples copy) */
    uint32_t r3_zconv_off, r3_zconv_pitch, r3_zconv_rows, r3_zconv_ns;
    uint32_t r300_aic_pt_base;  /* PCI GART table base (0x0AB0) */
    uint32_t vram_size_mb;      /* VRAM size in megabytes */
    uint64_t vram_size;         /* VRAM size in bytes (computed) */

    /* Device state */
    uint8_t mode;               /* VGA or extended mode */
    PPCMacGPURegs regs;         /* Register file */
    /* How big that register file is, saved with the machine so a state
     * written by another build cannot be poured into a different layout. */
    uint32_t regs_size;
    PPCMacGPUDisplayMode disp;  /* Current display mode */
    QEMUTimer vblank_timer;     /* VBlank interrupt timer */
    bool display_invalid;       /* Display needs full redraw */

    /* PM4 PIO FIFO for CP command processing */
    /*
     * One command packet pushed in by hand, gathered before it is carried
     * out.  A packet's length field is 14 bits, so this has to hold 16384
     * words; it held 64, and every longer packet quietly lost its tail --
     * which for Halo meant losing the write that marks a frame complete,
     * leaving the game waiting for a number that never came.
     */
    uint32_t pm4_fifo[16384];
    uint32_t pm4_fifo_idx;     /* current write index */
    uint32_t pm4_pkt_count;    /* remaining dwords for current packet */
    uint32_t pm4_pkt_reg;      /* base register for type 0 packet */
    bool     pm4_pkt_one_reg;  /* ONE_REG_WR: all writes to same reg */
    uint32_t pm4_pkt_opcode;   /* type 3 opcode for PIO packets */

    /* HOST_DATA blit state — for CPU-to-VRAM pixel transfers.
     * The kext writes compositor pixel data to HOST_DATA0-HOST_DATA7
     * registers. We store each dword to the destination in VRAM. */
    uint32_t host_data_dst_x;     /* dest rect X origin */
    uint32_t host_data_dst_y;     /* dest rect Y origin */
    uint32_t host_data_w;         /* dest rect width */
    uint32_t host_data_h;         /* dest rect height */
    uint32_t host_data_cur_x;     /* current X position */
    uint32_t host_data_bpp;       /* bytes per pixel of the destination */
    uint32_t host_data_cur_y;     /* current Y position */
    uint32_t host_data_offset;    /* dest surface offset */
    uint32_t host_data_pitch;     /* dest surface pitch */
    bool host_data_macro_tile;    /* dest surface uses macro tiling */
    bool host_data_active;        /* host data transfer in progress */

    /* Shadow buffer for display byte-order conversion.
     * PPC guest writes big-endian XRGB to VRAM: bytes [X,R,G,B].
     * Cocoa backend expects little-endian XRGB: bytes [B,G,R,X].
     * We bswap32 each pixel from VRAM into this buffer. */
    uint8_t *shadow_buf;
    /* What the UI's current surface describes, so it is replaced only when
     * it would actually differ (see ppc_mac_gpu_display_update). */
    uint32_t surface_width, surface_height, surface_stride;
    void *surface_data;
    uint32_t shadow_buf_size;

    /* Dirty-region scanout (see ppc_mac_gpu_display_update): the VRAM dirty
     * log is consumed here, so the last scanout this copy served is remembered
     * to force a full copy across a page flip (the log was cleared while the
     * buffer we are flipping *to* was still the back buffer). */
    uint64_t scanout_offset;      /* crtc_offset the shadow buffer mirrors */
    bool scanout_dirty_valid;     /* shadow_buf/dirty log are in sync */
    bool scanout_log_owner;       /* this device consumes the VGA dirty log */
    int scanout_defers;           /* deferrals of the frame we are after */

    /* Deferred present (PPCGPU_ASYNC_PRESENT): display updates skipped while
     * in-flight GPU work still owned the scanout range. */
    uint32_t present_wait_seq;    /* submit_r200 seq we are waiting on, 0 = no */
    int64_t present_defer_us;     /* monotonic us the current wait started */
    int64_t last_flip_kick;       /* monotonic us of the last refresh kick */
    /* Frame kick (PPCGPU_FRAME_KICK): a refresh shortly after the guest
     * finishes drawing, instead of waiting for the 30 ms UI poll. */
    QEMUTimer frame_kick_timer;
    int64_t last_scanout_ms;      /* QEMU_CLOCK_REALTIME ms of the last display update */

    /* Compositor-direct display: instead of reading the framebuffer
     * (which may have gaps during drag due to missing screen-to-screen
     * scroll), read directly from the compositor buffer that WindowServer
     * writes to.  Detected from SRC_PITCH_OFFSET in 2D blits. */
    uint32_t compositor_base;    /* VRAM offset of compositor (e.g. 0x300000) */
    uint32_t compositor_pitch;   /* stride of compositor buffer */
    bool     compositor_valid;   /* true once we've seen compositor blits */

    /* I2C / DDC for EDID */
    bitbang_i2c_interface bbi2c;

    /* EDID blob for NDRV compatibility (byte reads at MMIO 0x00-0x7F) */
    uint8_t edid_blob[128];
    bool edid_read_done;    /* true after NDRV finishes EDID reading */

    /* 3D vertex buffer pointers (from 3D_LOAD_VBPNTR) */
    uint32_t vb_count;              /* number of active vertex buffers */
    uint32_t vb_stride[4];          /* bytes per vertex for each VB */
    uint32_t vb_addr[4];            /* GPU address for each VB */

    /* MC-style tiled surface registry.
     *
     * R200 macro-tiling is a property of VRAM surface ranges, not of
     * individual blit operations.  When a PITCH_OFFSET register is written
     * with macro_tile=1, the MC (memory controller) remaps ALL accesses
     * to that surface range through the tiling address function.
     *
     * This registry tracks which VRAM surfaces are macro-tiled so that
     * read/write helpers can transparently apply the tiling swizzle. */
#define MC_TILED_SURFACE_MAX 8
    struct {
        uint32_t base;       /* VRAM byte offset of surface start */
        uint32_t pitch;      /* stride in bytes */
        uint32_t size;       /* estimated surface size in bytes (pitch * height_est) */
        bool     active;     /* slot in use */
        char     label[16];  /* source label for diagnostics */
    } tiled_surfaces[MC_TILED_SURFACE_MAX];

    /*
     * Presentation stride override.
     *
     * When the Quartz Extreme compositor blits its back buffer to the
     * visible framebuffer, the blit's destination pitch can differ from
     * CRTC_PITCH.  When the override is active the display path uses the
     * blit pitch instead of the CRTC-derived stride; it is cleared on any
     * resolution change.
     */
    bool     disp_stride_override_active;
    uint32_t disp_stride_override_value;   /* bytes per scanline */

    /*
     * Render targets the Metal backend drew into during the current frame.
     * Used to recognise a present blit whose source was GPU-rendered.
     * Reset at the top of each display update.
     */
    uint32_t metal_rt_offsets[16];
    int      metal_rt_count;
    uint32_t metal_rt_frame_id;

    /* Diagnostic: enable the VRAM write watch in the PM4 processor. */
    bool vram_watch_active;

    /*
     * Shadow storage for R300-only registers (MMIO offsets below 0x5000).
     * ATIRadeon9700.kext writes registers this device does not model; they
     * are absorbed here so the kext can read back what it wrote.
     */
    uint32_t r300_shadow[0x5000 / 4];

    /* Hardware cursor (patched qemu_vga.ndrv, MMIO BAR + 0xFF00) */
    uint32_t hwc_pix[64 * 64];
    uint32_t hwc_w, hwc_h, hwc_idx;
    bool host_aspect_modes;       /* EDID offers the patched NDRV's 1.545 modes */
    bool gl_zero_copy;            /* GL display: show VRAM without a CPU copy */
    int32_t hwc_x, hwc_y;
    /* hotspot inside the image, from HWC reg 0x1C (0,0 with older NDRVs) */
    uint32_t hwc_hot_x, hwc_hot_y;
    bool hwc_visible;
    /* A woken machine's window has never been told what the pointer looks
     * like: the guest uploaded it once, before the machine slept, and will
     * not do it again.  Set when a machine is restored, acted on at the
     * next screen update, by which time the window is listening. */
    bool hwc_announce;

    /*
     * VRAM from the renderer: a shared Metal buffer (zero-copy) or
     * host-visible Vulkan memory.  metal_vram_ptr is the host mapping and
     * metal_vram_opaque the backend's handle.  Both NULL on the fallback
     * path (QEMU-managed RAM).
     */
    void *metal_vram_ptr;
    void *metal_vram_opaque;

    /* Renderer backend */
    PPCMacGPURenderer *renderer;
    void *renderer_opaque;          /* backend-specific state from init() */
};

#endif /* HW_DISPLAY_PPC_MAC_GPU_H */
