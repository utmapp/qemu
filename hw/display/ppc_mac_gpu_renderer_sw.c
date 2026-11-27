/*
 * PPC Mac GPU - Software Renderer Backend
 *
 * Always-available pure C rendering backend. Performs all operations
 * in software on the host CPU. This is the fallback renderer and is
 * guaranteed to work on all host platforms (Intel macOS, Apple Silicon macOS).
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "ppc_mac_gpu_renderer.h"
#include "ui/console.h"
#include "ui/qemu-pixman.h"

typedef struct PPCMacGPURendererSWState {
    uint8_t *vram_ptr;
    uint64_t vram_size;
    DisplaySurface *surface;
} PPCMacGPURendererSWState;

static void *sw_init(uint8_t *vram_ptr, uint64_t vram_size)
{
    PPCMacGPURendererSWState *st = g_new0(PPCMacGPURendererSWState, 1);
    st->vram_ptr = vram_ptr;
    st->vram_size = vram_size;
    st->surface = NULL;
    return st;
}

static void sw_fini(void *opaque)
{
    PPCMacGPURendererSWState *st = opaque;
    /* Surface is owned by the console, don't free it here */
    g_free(st);
}

static void *sw_scanout(void *opaque, const PPCMacGPUScanout *desc)
{
    PPCMacGPURendererSWState *st = opaque;
    pixman_format_code_t format;
    uint8_t *fb_ptr;

    if (!desc || desc->width == 0 || desc->height == 0) {
        return NULL;
    }

    /* Validate that the framebuffer fits in VRAM */
    uint64_t frame_size = (uint64_t)desc->stride * desc->height;
    if (desc->offset + frame_size > st->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "ppc-mac-gpu-sw: framebuffer exceeds VRAM "
                      "(offset=0x%"PRIx64" size=0x%"PRIx64" vram=0x%"PRIx64")\n",
                      desc->offset, frame_size, st->vram_size);
        return NULL;
    }

    fb_ptr = st->vram_ptr + desc->offset;

    /*
     * Determine pixman format based on BPP and endianness.
     *
     * PPC guests write big-endian pixel data to VRAM. The ATI Radeon
     * hardware has a byte-swap engine (SURFACE_CNTL) but we handle
     * endianness at the format level for now.
     *
     * The pixman BE formats handle the byte ordering correctly when
     * the host reads the VRAM data, regardless of whether the host
     * is little-endian (x86) or little-endian (ARM64 macOS).
     */
    switch (desc->bpp) {
    case 32:
        format = desc->big_endian ? PIXMAN_BE_x8r8g8b8 : PIXMAN_LE_x8r8g8b8;
        break;
    case 24:
        format = desc->big_endian ? PIXMAN_BE_r8g8b8 : PIXMAN_r8g8b8;
        break;
    case 16:
        format = PIXMAN_r5g6b5;
        break;
    case 15:
        format = PIXMAN_x1r5g5b5;
        break;
    case 8:
        /* 8bpp indexed color - for now, render as 32bpp placeholder */
        format = PIXMAN_BE_x8r8g8b8;
        break;
    default:
        format = PIXMAN_BE_x8r8g8b8;
        break;
    }

    /*
     * Create a zero-copy DisplaySurface pointing directly into VRAM.
     * This avoids an extra memcpy per frame for the common case.
     * The surface is valid as long as the VRAM region exists.
     */
    st->surface = qemu_create_displaysurface_from(
        desc->width, desc->height,
        format, desc->stride,
        fb_ptr
    );

    return st->surface;
}

static int sw_blit_2d(void *opaque, uint8_t *vram_ptr,
                      const PPCMacGPUBlit *blit)
{
    /* TODO: Phase 5 - implement software 2D blit */
    (void)opaque;
    (void)vram_ptr;
    (void)blit;
    return -1; /* Not yet implemented */
}

static int sw_fill_2d(void *opaque, uint8_t *vram_ptr,
                      const PPCMacGPUBlit *fill)
{
    /* TODO: Phase 5 - implement software 2D fill */
    (void)opaque;
    (void)vram_ptr;
    (void)fill;
    return -1; /* Not yet implemented */
}

static int sw_draw_3d(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                      const PPCMacGPU3DState *state,
                      const PPCMacGPU3DDrawCmd *cmd)
{
    /* Software renderer does not support 3D — fall back to 2D path */
    (void)opaque;
    (void)vram_ptr;
    (void)vram_size;
    (void)state;
    (void)cmd;
    return -1;
}

static void sw_mode_change(void *opaque, const PPCMacGPUScanout *new_mode)
{
    /* Software renderer doesn't need to do anything special on mode change */
    (void)opaque;
    (void)new_mode;
}

static uint32_t sw_get_caps(void *opaque)
{
    (void)opaque;
    /* Software renderer has no acceleration capabilities yet */
    return 0;
}

static PPCMacGPURenderer sw_renderer = {
    .name        = "software",
    .init        = sw_init,
    .fini        = sw_fini,
    .scanout     = sw_scanout,
    .blit_2d     = sw_blit_2d,
    .fill_2d     = sw_fill_2d,
    .draw_3d     = sw_draw_3d,
    .mode_change = sw_mode_change,
    .get_caps    = sw_get_caps,
};

PPCMacGPURenderer *ppc_mac_gpu_renderer_sw(void)
{
    return &sw_renderer;
}
