/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: interface between the 3D state decoder
 * (vCPU thread) and the GL renderer (its own thread).
 *
 * Copyright (c) 2026 RetroBoot contributors
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */
#ifndef ATI_3D_H
#define ATI_3D_H

#include <stdbool.h>
#include <stdint.h>

/* canonical vertex: window-space position plus everything interpolated */
typedef struct ATIVertex {
    float x, y, z, rhw;        /* window x/y in pixels, z in [0,1], 1/w (signed after TCL) */
    float c0[4];               /* diffuse RGBA */
    float c1[4];               /* specular RGB, fog factor in [3] */
    float tc[3][4];            /* s, t, q (r unused) per unit, already /w */
} ATIVertex;

enum {
    ATI_FMT_NONE = 0,
    ATI_FMT_ARGB1555, ATI_FMT_RGB565, ATI_FMT_ARGB8888, ATI_FMT_RGB332,
    ATI_FMT_ARGB4444, ATI_FMT_I8, ATI_FMT_AI88, ATI_FMT_RGBA8888,
    ATI_FMT_Y8, ATI_FMT_VYUY422, ATI_FMT_YVYU422, ATI_FMT_DXT1,
    ATI_FMT_DXT23, ATI_FMT_DXT45, ATI_FMT_AI44,
};

/* a block of guest memory interpreted as a 2D image */
typedef struct ATIImage {
    uint32_t mc;               /* card address of texel/pixel (0,0) */
    uint32_t pitch;            /* bytes per row */
    uint16_t w, h;
    uint8_t fmt;               /* ATI_FMT_* */
    uint8_t swap;              /* 0 none, 1 16-bit, 2 32-bit, 3 half-dword */
} ATIImage;

typedef struct ATITexUnit {
    bool enabled;
    ATIImage img;              /* level 0 */
    uint8_t levels;            /* mip levels present (1 = base only) */
    uint32_t level_offs[12];   /* byte offset of each level from mc */
    uint16_t level_w[12], level_h[12];
    uint32_t level_pitch[12];
    uint16_t min_filter, mag_filter;   /* GL enums */
    uint16_t wrap_s, wrap_t;           /* GL enums */
    float lod_bias;
    float border[4];
    uint64_t key;              /* identity of the uploaded texture */
    uint32_t gen;              /* content generation */
    bool chroma_key;
    bool alpha_one;            /* PP_TXFORMAT ALPHA_IN_MAP clear: texel alpha = 1 */
} ATITexUnit;

typedef struct ATIStage {
    uint32_t cblend, ablend;
    float tfactor[4];
} ATIStage;

typedef struct ATIDrawState {
    ATIImage color;            /* render target */
    uint32_t depth_mc, depth_pitch;
    uint8_t depth_fmt;         /* 16, 24 (with stencil), 32 */
    bool z_test, z_write, stencil;
    uint16_t z_func;
    uint16_t st_func, st_fail, st_zfail, st_zpass;
    uint8_t st_ref, st_mask, st_wmask;
    bool blend;
    uint16_t blend_src, blend_dst, blend_eq;
    bool rop_enable;
    uint8_t rop;
    uint32_t plane_mask;
    uint8_t color_mask;        /* bit 0 R, 1 G, 2 B, 3 A written */
    bool scissor;
    int16_t sc_x0, sc_y0, sc_x1, sc_y1;     /* inclusive */
    uint8_t cull;              /* 0 none, 1 cull CW-front?, see renderer */
    bool cull_enable;
    uint16_t cull_face;        /* GL_FRONT/GL_BACK in screen winding */
    bool front_ccw;
    bool flat;                 /* flat shading of colours */
    uint8_t flat_vtx;          /* provoking vertex */
    bool alpha_test;
    uint16_t alpha_func;
    float alpha_ref;
    bool fog;
    float fog_color[4];
    uint8_t fog_src;           /* 0 vertex fog factor (c1.a), 1 from z */
    bool specular;
    uint8_t tex_enable;        /* bit per unit */
    uint8_t stage_enable;      /* bit per combiner stage */
    ATITexUnit tex[3];
    ATIStage stage[3];
    bool persp_tex;
    float solid[4];            /* RE_SOLID_COLOR */
    bool solid_color;          /* diffuse shade solid */
    float line_width;
    float zbias_factor, zbias_const;
    bool zbias;
    bool tcl;                  /* vertices came through the TCL transform:
                                * clip-space w may be negative or zero */
} ATIDrawState;

/* render thread API (ati_glr.c) */
typedef struct ATIRenderer ATIRenderer;
ATIRenderer *ati_glr_new(char *info, size_t infolen);
void ati_glr_draw(ATIRenderer *r, const ATIDrawState *st, uint16_t prim,
                  const ATIVertex *v, unsigned nv);
/* make the texture identified by key current with these bytes (copied) */
void ati_glr_texture(ATIRenderer *r, uint64_t key, uint32_t gen,
                     const ATITexUnit *t, const uint8_t *data, size_t len);
bool ati_glr_texture_known(ATIRenderer *r, uint64_t key, uint32_t gen);
/* render-target content from guest memory (bytes in guest layout) */
void ati_glr_target_upload(ATIRenderer *r, const ATIImage *img,
                           const uint8_t *data);
/* the same, taking ownership of data (g_malloc()ed, pitch * h bytes) */
void ati_glr_target_upload_take(ATIRenderer *r, const ATIImage *img,
                                uint8_t *data);
/* fill [x, x+w) x [y, y+h) of a target with one pixel (px: guest bytes) */
void ati_glr_fill(ATIRenderer *r, const ATIImage *img, int x, int y, int w,
                  int h, const uint8_t px[4]);
void ati_glr_depth_clear(ATIRenderer *r, const ATIImage *color,
                         uint32_t depth_mc, float z, int stencil,
                         int x, int y, int w, int h);
/*
 * Read the render target back into guest layout; waits.  Returns
 * ati_vram_hash() of the pitch * h bytes at out once written.
 */
uint64_t ati_glr_target_read(ATIRenderer *r, const ATIImage *img, uint8_t *out);
typedef struct ATIReadJob {
    ATIImage img;
    uint8_t *buf;
    size_t len;
    uint64_t hash;              /* ati_vram_hash(buf, len), set before done */
    int done;                   /* set by the render thread (atomic) */
} ATIReadJob;
/*
 * True while the vCPU waits for the render thread with the BQL released
 * (glr_push_wait): the device's 3D state is mid-operation, so other threads
 * holding the BQL (the vblank timer) leave it alone.
 */
bool ati_glr_waiting(ATIRenderer *r);
/* content hash of guest bytes (render-target change detection) */
uint64_t ati_vram_hash(const uint8_t *p, size_t len);
ATIReadJob *ati_glr_target_read_async(ATIRenderer *r, const ATIImage *img);
bool ati_glr_read_done(ATIReadJob *j);
void ati_glr_read_wait(ATIRenderer *r, ATIReadJob *j);
void ati_glr_read_free(ATIReadJob *j);
void ati_glr_finish(ATIRenderer *r);
const char *ati_glr_info(ATIRenderer *r);

/* pixel conversion helpers shared by both sides */
unsigned ati_fmt_bpp(uint8_t fmt);

#endif
