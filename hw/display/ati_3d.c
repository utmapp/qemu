/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: 3D engine.
 *
 * Copyright (c) 2026 RetroBoot contributors
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 *
 * Decodes the R100 setup-engine/pixel-pipeline registers and draw packets
 * (TCL bypass: window-space vertices) into a portable draw description that
 * the GL renderer thread executes (ati_glr.c).  Register semantics follow
 * AMD's R100 documentation and the open radeon drivers.
 *
 * Memory model: VRAM (guest RAM) is authoritative for anything the CPU or the
 * 2D engine touches.  A colour buffer the 3D engine draws into is owned by the
 * renderer from its first draw until a sync point (the guest polling for
 * idle, a 2D operation on it, a CRTC change, ...), where it is read back into
 * VRAM.  Hardware gives the same guarantee: memory is coherent once the
 * engine is idle.
 */

#include "qemu/osdep.h"
#include <math.h>
#include "qemu/log.h"
#include "qemu/error-report.h"
#include "qemu/bswap.h"
#include "qemu/bitmap.h"
#include "exec/memory.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "ati_3d.h"
#include "trace.h"

/* registers (R100) */
#define PP_MISC                 0x1c14
#define PP_FOG_COLOR            0x1c18
#define RE_SOLID_COLOR          0x1c1c
#define RB3D_BLENDCNTL          0x1c20
#define RB3D_DEPTHOFFSET        0x1c24
#define RB3D_DEPTHPITCH         0x1c28
#define RB3D_ZSTENCILCNTL       0x1c2c
#define PP_CNTL                 0x1c38
#define RB3D_CNTL               0x1c3c
#define RB3D_COLOROFFSET        0x1c40
#define RE_WIDTH_HEIGHT         0x1c44
#define RB3D_COLORPITCH         0x1c48
#define SE_CNTL                 0x1c4c
#define SE_COORD_FMT            0x1c50
#define PP_TXFILTER_0           0x1c54
#define PP_TXFORMAT_0           0x1c58
#define PP_TXOFFSET_0           0x1c5c
#define PP_TXCBLEND_0           0x1c60
#define PP_TXABLEND_0           0x1c64
#define PP_TFACTOR_0            0x1c68
#define PP_TEX_STRIDE           0x18
#define PP_TEX_SIZE_0           0x1d04
#define PP_TEX_PITCH_0          0x1d08
#define PP_TEX_NPOT_STRIDE      0x08
#define PP_BORDER_COLOR_0       0x1d40
#define RB3D_STENCILREFMASK     0x1d7c
#define RB3D_DEPTHCLEARVALUE    0x1c30  /* R100 (R200: 0x3230) */
#define RB3D_ROPCNTL            0x1d80
#define RB3D_PLANEMASK          0x1d84
#define SE_VPORT_XSCALE         0x1d98
#define SE_VPORT_XOFFSET        0x1d9c
#define SE_VPORT_YSCALE         0x1da0
#define SE_VPORT_YOFFSET        0x1da4
#define SE_VPORT_ZSCALE         0x1da8
#define SE_VPORT_ZOFFSET        0x1dac
#define SE_ZBIAS_FACTOR         0x1db0
#define SE_ZBIAS_CONSTANT       0x1db4
#define SE_LINE_WIDTH           0x1db8
#define SE_PORT_DATA0           0x2000
#define SE_VTX_FMT              0x2080
#define SE_VF_CNTL              0x2084
#define SE_CNTL_STATUS          0x2140
#define SE_TCL_VECTOR_INDX_REG  0x2200
#define SE_TCL_VECTOR_DATA_REG  0x2204
#define SE_TCL_SCALAR_INDX_REG  0x2208
#define SE_TCL_SCALAR_DATA_REG  0x220c
#define SE_TCL_OUTPUT_VTX_FMT   0x2254
#define SE_TCL_OUTPUT_VTX_SEL   0x2258
#define SE_TCL_MATRIX_SELECT_0  0x225c
#define SE_TCL_MATRIX_SELECT_1  0x2260
#define SE_TCL_UCP_VERT_BLEND_CTL 0x2264
#define SE_TCL_TEXTURE_PROC_CTL 0x2268
#define SE_TCL_LIGHT_MODEL_CTL  0x226c
#define RE_TOP_LEFT             0x26c0
#define RE_MISC                 0x26c4
#define SURFACE_CNTL            0x0b00

/* SE_VTX_FMT */
#define VF_W0       (1u << 0)
#define VF_FPCOLOR  (1u << 1)
#define VF_FPALPHA  (1u << 2)
#define VF_PKCOLOR  (1u << 3)
#define VF_FPSPEC   (1u << 4)
#define VF_FPFOG    (1u << 5)
#define VF_PKSPEC   (1u << 6)
#define VF_ST0      (1u << 7)
#define VF_ST1      (1u << 8)
#define VF_Q1       (1u << 9)
#define VF_ST2      (1u << 10)
#define VF_Q2       (1u << 11)
#define VF_ST3      (1u << 12)
#define VF_Q3       (1u << 13)
#define VF_Q0       (1u << 14)
#define VF_N0       (1u << 18)
#define VF_Z        (1u << 31)

/* TCL vector state addresses (octwords) and sizes, see ATI-R100-TCL.md */
#define TCL_VS_MATRIX_EYE2CLIP  112
#define TCL_VEC_N               256     /* octwords (4 dwords each) */
#define TCL_SCAL_N              512     /* dwords */
#define VFCNTL_TCL_ENABLE       (1u << 9)
#define TCL_VS_LIGHT_AMBIENT    64      /* + 8 * (0 amb, 1 diff, 2 spec, 3 pos, 4 spot dir, 5 atten) + light */
#define TCL_VS_UCP              116
#define TCL_VS_GLOBAL_AMBIENT   122
#define TCL_VS_FOG_PARAM        123
#define TCL_VS_EYE_VECTOR       124
#define TCL_SS_LIGHT_EXPONENT   8       /* + 8 * (0 dcd, 1 exp, 2 cutoff, 3 spec thresh, 4 range, 5 1/const) + light */
#define SE_TCL_MATERIAL_EMISSIVE 0x2210 /* 4 floats; ambient 0x2220, diffuse 0x2230, specular 0x2240 */
#define SE_TCL_SHININESS        0x2250
#define SE_TCL_PER_LIGHT_CTL_0  0x2270

/* per-draw state of the TCL stages after the position transform (T2..T4) */
typedef struct TCLLight {
    uint32_t ctl;               /* 16 bit flags from SE_TCL_PER_LIGHT_CTL */
    float amb[4], dif[4], spec[4], pos[4], spot[4], att[4];
    float exponent, cutoff, const_inv;
} TCLLight;

typedef struct TCLFx {
    bool light, tex;
    uint8_t ucp, fog;           /* enabled planes; fog mode (0 off) */
    uint32_t lm, ucp_ctl, vsel, tpc;
    float mv[4][4], it[4][4];
    float texm[3][4][4];
    float plane[6][4];
    float emis[4], mamb[4], mdif[4], mspec[4], shine, glt[4], eye[4];
    int nl;
    TCLLight L[8];
    float fog_c, fog_d;
} TCLFx;

/* GL enums used in the draw description (match GLES values) */
enum {
    G_NEVER = 0x0200, G_LESS, G_EQUAL, G_LEQUAL, G_GREATER, G_NOTEQUAL,
    G_GEQUAL, G_ALWAYS,
};

typedef struct ATITrack {
    ATIImage img;
    bool gl_owned;
    uint32_t depth_mc;
    /* after a readback GL still holds the same pixels; the next draw takes
     * them back without an upload if VRAM still hashes the same */
    bool gl_valid;
    uint64_t vram_hash;
    /* scanout readback in flight (vblank), and draw generations */
    ATIReadJob *pending;
    uint64_t gen, read_gen;
} ATITrack;

static void watch_land(ATIVGAState *s, uint64_t off, uint64_t len);

#define MAX_TRACK 8

typedef struct ATITexCache {
    uint64_t key;
    uint32_t gen;
    uint32_t mc, len;
    bool valid;
} ATITexCache;

#define TEXC_BITS 12
#define TEXC_N (1 << TEXC_BITS)

struct ATI3DState {
    ATIRenderer *r;
    bool failed;
    ATITrack track[MAX_TRACK];
    ATITexCache texc[TEXC_N];
    uint32_t tex_gen;
    ATIVertex *vbuf;
    unsigned vbuf_cap;
    uint32_t *mbuf;                   /* 3D_RNDR_GEN_INDX_PRIM: fetched vertex dwords */
    size_t mbuf_cap;
    ATIVertex *dec;                   /* ... and the vertices they decode to */
    unsigned dec_cap;
    /* vertex arrays from 3D_LOAD_VBPNTR */
    unsigned narrays;
    uint32_t arr_addr[16], arr_stride[16], arr_size[16];
    /*
     * TCL (hardware transform) state: the vector/scalar memories behind
     * SE_TCL_VECTOR_* / SE_TCL_SCALAR_*, the index registers' auto-increment
     * state, and the transform of the draw being decoded.
     */
    uint32_t tcl_vec[TCL_VEC_N * 4];
    uint32_t tcl_scal[TCL_SCAL_N];
    uint16_t vec_addr, scal_addr;
    uint8_t vec_stride, scal_stride, vec_dw;
    bool tcl_on;                      /* this draw goes through the transform */
    float tcl_m[4][4];                /* clip = tcl_m * (x, y, z, w) */
    TCLFx tx;                         /* lighting / texgen / fog / clip planes */
    uint32_t tcl_sig;                 /* identity of tcl_m, for the traces */
    int tcl_dump_left;                /* draws still to trace after a change */
    unsigned tcl_dumps;               /* matrix traces so far */
    bool tcl_vtx_set;                 /* first vertex of the draw captured */
    float tcl_vtx[8];                 /* its object x y z w, clip x y z w */
    uint64_t tcl_draws;
    uint64_t draws, verts, syncs, readbacks, uploads, texuploads, texhits;
    uint64_t zclears, upload_skips, flips, vblank_waits, readbacks_joined;
    uint64_t gl_fills;          /* RetroBoot: 2D fills done as GL clears */
    uint64_t rb_why[ATI_SYNC_N];      /* readbacks by cause: 2D, status read, vblank */
    uint64_t vb_fetch, vb_idx;        /* vertices fetched and decoded / drawn, 3D_RNDR_GEN_INDX_PRIM */
    char stats[768];
    char gl_info[256];                /* ClassicMac: host GL renderer, "" when off */
};

static inline uint32_t R(ATIVGAState *s, hwaddr a)
{
    return s->regfile[a / 4];
}

static inline float RF(ATIVGAState *s, hwaddr a)
{
    union { uint32_t u; float f; } v = { .u = s->regfile[a / 4] };
    return v.f;
}

static void argb_to_float(uint32_t c, float *o)
{
    o[0] = ((c >> 16) & 0xff) / 255.0f;
    o[1] = ((c >> 8) & 0xff) / 255.0f;
    o[2] = (c & 0xff) / 255.0f;
    o[3] = (c >> 24) / 255.0f;
}

/* ---------------------------------------------------------------------
 * Byte-swap bookkeeping.  QEMU's VRAM holds what the CPU sees through the
 * framebuffer aperture.  The chip's own view is the aperture swap (the CPU's
 * writes went through it) composed with the engine's per-surface swap.
 * Codes form a group: 0 none, 1 swap within 16-bit, 2 swap 16-bit halves,
 * 3 full 32-bit swap; composition is XOR.
 */
static uint8_t swap_code(unsigned r100_endian)
{
    static const uint8_t map[4] = { 0, 1, 3, 2 };   /* none, 16, 32, halfdw */
    return map[r100_endian & 3];
}

static uint8_t aperture_swap(ATIVGAState *s, unsigned bpp)
{
    uint32_t sc = R(s, SURFACE_CNTL);
    /* the Mac driver programs aperture 1 for the CPU's frame buffer view */
    unsigned bits = (sc >> 22) & 3;
    if (!bits) {
        bits = (sc >> 20) & 3;
    }
    /*
     * These bits only say the CPU's view is swapped.  What undoes it is the
     * size of the elements the CPU wrote, whatever the desktop's depth: a
     * 32-bit texture on a 16-bit desktop still needs the 32-bit swap.
     */
    if (bits && bpp == 32) {
        return 3;               /* 32bpp swap */
    }
    if (bits && bpp == 16) {
        return 1;               /* 16bpp swap */
    }
    /* 8-bit formats keep the aperture's swap: no tested guest uses them */
    if (bits & 2) {
        return 3;
    }
    if (bits & 1) {
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------
 * Render-target tracking and sync
 */

static ATITrack *track_find(ATI3DState *t, uint32_t mc, uint32_t pitch,
                            uint8_t fmt)
{
    for (int i = 0; i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        if (k->img.w && k->img.mc == mc && k->img.pitch == pitch &&
            k->img.fmt == fmt) {
            return k;
        }
    }
    return NULL;
}

/* copy a finished scanout readback into VRAM */
static void track_land(ATIVGAState *s, ATITrack *k)
{
    uint64_t off, contig;
    ATIReadJob *j = k->pending;
    bool in_vram;
    uint64_t j_mc, j_hash;
    size_t j_len;

    if (!j || !ati_glr_read_done(j)) {
        return;
    }
    in_vram = ati_mc_translate(s, j->img.mc, &off, &contig) == ATI_MC_VRAM &&
              off + j->len <= s->vga.vram_size;
    if (in_vram) {
        memcpy(s->vga.vram_ptr + off, j->buf, j->len);
        memory_region_set_dirty(&s->vga.vram, off, j->len);
        watch_land(s, off, j->len);
        ati_front_touch(s, 2, off, j->len);
    } else {
        ati_mc_rw(s, j->img.mc, j->buf, j->len, true);
    }
    j_mc = j->img.mc;
    j_len = j->len;
    j_hash = j->hash;           /* of j->buf, now VRAM's bytes */
    ati_glr_read_free(j);
    k->pending = NULL;
    if (k->gen == k->read_gen) {
        /* nothing drawn since: VRAM holds the target; GL keeps a copy that
         * the next draw takes back without an upload if VRAM is unchanged */
        uint64_t off2, c2;
        size_t len = (size_t)k->img.pitch * k->img.h;
        k->gl_owned = false;
        k->gl_valid = ati_mc_translate(s, k->img.mc, &off2, &c2) == ATI_MC_VRAM &&
                      off2 + len <= s->vga.vram_size;
        if (k->gl_valid) {
            k->vram_hash = in_vram && j_mc == k->img.mc && j_len == len ?
                           j_hash : ati_vram_hash(s->vga.vram_ptr + off2, len);
        }
    }
}

/* a synchronous readback supersedes one in flight */
static void track_drop_pending(ATIVGAState *s, ATITrack *k)
{
    if (k->pending) {
        ati_glr_read_wait(s->r3d->r, k->pending);
        ati_glr_read_free(k->pending);
        k->pending = NULL;
    }
}

static void track_readback(ATIVGAState *s, ATITrack *k)
{
    ATI3DState *t = s->r3d;
    uint64_t off, contig;
    size_t len = (size_t)k->img.pitch * k->img.h;

    if (!k->gl_owned) {
        return;
    }
    if (k->pending && k->gen == k->read_gen) {
        /* the read in flight is current: wait for it instead of a new one */
        ati_glr_read_wait(t->r, k->pending);
        track_land(s, k);
        t->readbacks_joined++;
        return;
    }
    track_drop_pending(s, k);
    trace_ati_3d_readback(0, k->img.mc, k->img.w, k->img.h, k->img.pitch,
                          k->img.fmt, k->img.swap);
    if (ati_mc_translate(s, k->img.mc, &off, &contig) == ATI_MC_VRAM &&
        off + len <= s->vga.vram_size) {
        /*
         * Into a copy, then into VRAM with the BQL held again: the vCPU waits
         * without the BQL, and the display must not see a target it shows
         * half rewritten.  The copy starts as VRAM because the renderer
         * writes only the target's pixels, not the rest of each pitch.
         */
        uint8_t *buf = g_memdup2(s->vga.vram_ptr + off, len);
        k->vram_hash = ati_glr_target_read(t->r, &k->img, buf);
        memcpy(s->vga.vram_ptr + off, buf, len);
        g_free(buf);
        memory_region_set_dirty(&s->vga.vram, off, len);
        watch_land(s, off, len);
        ati_front_touch(s, 2, off, len);
        k->gl_valid = true;
    } else {
        uint8_t *buf = g_malloc(len);
        ati_mc_rw(s, k->img.mc, buf, len, false);
        ati_glr_target_read(t->r, &k->img, buf);
        ati_mc_rw(s, k->img.mc, buf, len, true);
        g_free(buf);
    }
    k->gl_owned = false;
    t->readbacks++;
}

void ati_3d_sync(ATIVGAState *s, int why)
{
    ATI3DState *t = s->r3d;

    if (!t || !t->r) {
        return;
    }
    /*
     * Frames in flight first, then the flip waiting for one settles, and only
     * then are other targets read back into VRAM.  With off-screen 2D fills
     * done in GL (ati_3d_fill) the buffer still on screen can be GL-owned
     * while its flip-away is pending; reading it back before the display has
     * left it (the BQL is released while the vCPU waits, so the display can
     * refresh meanwhile) would show the next frame half drawn.
     */
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < MAX_TRACK; i++) {
            ATITrack *k = &t->track[i];
            if (!k->gl_owned || (pass == 0 && !k->pending)) {
                continue;
            }
            t->rb_why[why]++;
            track_readback(s, k);
        }
        /* every frame is in VRAM now: a flip waiting for one shows it before
         * the guest writes to the buffer it is leaving */
        ati_3d_land(s);
        ati_flip_settle(s);
    }
}

/* start an asynchronous readback of the GL-owned targets in [mc, mc+len)
 * that changed since their last read */
static void scanout_read_async(ATIVGAState *s, uint64_t mc, uint64_t len)
{
    ATI3DState *t = s->r3d;

    for (int i = 0; i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        uint64_t a = k->img.mc, b = a + (uint64_t)k->img.pitch * k->img.h;
        track_land(s, k);
        if (k->gl_owned && mc < b && mc + len > a && !k->pending &&
            k->gen != k->read_gen) {
            trace_ati_3d_readback(1, k->img.mc, k->img.w, k->img.h,
                                  k->img.pitch, k->img.fmt, k->img.swap);
            k->pending = ati_glr_target_read_async(t->r, &k->img);
            k->read_gen = k->gen;
            t->rb_why[ATI_SYNC_VBLANK]++;
            t->readbacks++;
        }
    }
}

/*
 * A page flip (CRTC_OFFSET write): the new front buffer is read back
 * asynchronously while the guest goes on with the next frame.  Returns true
 * when a readback for that range is in flight (the display should keep the
 * old buffer until it has landed).
 */
bool ati_3d_flip(ATIVGAState *s, uint64_t mc, uint64_t len)
{
    ATI3DState *t = s->r3d;

    if (!t || !t->r) {
        return false;
    }
    /* at most one frame in flight: the previous flip's readback finishes
     * first, so the guest cannot run ahead of the host renderer */
    for (int i = 0; i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        if (k->pending) {
            if (!ati_glr_read_done(k->pending)) {
                ati_glr_read_wait(t->r, k->pending);
                t->vblank_waits++;
            }
            track_land(s, k);
        }
    }
    scanout_read_async(s, mc, len);
    t->flips++;
    return !ati_3d_range_ready(s, mc, len);
}

/*
 * The scanout's depth or geometry changed: the render targets tracked so far
 * belong to the old mode's buffers.  They must not be read back over the new
 * framebuffer (a stale 32 bpp desktop target landing in a 15 bpp one shows as
 * squeezed, line-doubled garbage), so drop them, in-flight reads included.
 */
void ati_3d_mode_changed(ATIVGAState *s)
{
    ATI3DState *t = s->r3d;
    int n = 0;

    for (int i = 0; t && t->r && i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];

        if (k->pending) {
            ati_glr_read_wait(t->r, k->pending);
            ati_glr_read_free(k->pending);
            k->pending = NULL;
        }
        if (k->img.w) {
            n++;
        }
        memset(k, 0, sizeof(*k));
    }
    trace_ati_3d_mode_drop(n);
}

/* finished asynchronous readbacks into VRAM */
void ati_3d_land(ATIVGAState *s)
{
    ATI3DState *t = s->r3d;

    for (int i = 0; t && t->r && i < MAX_TRACK; i++) {
        track_land(s, &t->track[i]);
    }
}

bool ati_3d_busy(ATIVGAState *s)
{
    return s->r3d && ati_glr_waiting(s->r3d->r);
}

/*
 * The 2D engine wrote VRAM [off, off + len).  A target there whose GL copy
 * matched VRAM (gl_valid) no longer does, so its next draw uploads it without
 * hashing the whole target to find that out (Mac OS's ATI OpenGL 2D-clears
 * the back buffer every frame: a 4 MB hash per frame at 1920x1080 on the
 * vCPU).  CPU writes are still caught by the hash.
 */
/*
 * Debug (ati_watch_*): did anything but the 3D readback write a render target
 * in VRAM between its readback and the 2D blit that presents it?  A copy of
 * each landed target is kept and compared at the blit.
 */
typedef struct ATIWatch {
    uint64_t off, len;
    uint8_t *data;
    unsigned w2d;               /* 2D operations into it since */
} ATIWatch;
static ATIWatch watch[4];
static unsigned watch_next;

static bool watch_on(void)
{
    return trace_event_get_state_backends(TRACE_ATI_WATCH_LAND);
}

static void watch_land(ATIVGAState *s, uint64_t off, uint64_t len)
{
    ATIWatch *w = NULL;

    if (!watch_on() || off + len > s->vga.vram_size) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        if (watch[i].data && watch[i].off == off) {
            w = &watch[i];
        }
    }
    if (!w) {
        w = &watch[watch_next++ & 3];
    }
    g_free(w->data);
    w->off = off;
    w->len = len;
    w->w2d = 0;
    w->data = g_memdup2(s->vga.vram_ptr + off, len);
    trace_ati_watch_land(off, len);
}

/* a 2D operation wrote [off, off+len) */
void ati_3d_watch_2d(ATIVGAState *s, uint64_t off, uint64_t len, int rop,
                     int src_kind)
{
    if (!watch_on()) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        ATIWatch *w = &watch[i];

        if (w->data && off < w->off + w->len && w->off < off + len &&
            ++w->w2d <= 4) {
            trace_ati_watch_2d(w->off, off, len, rop, src_kind, w->w2d);
        }
    }
}

/* a 2D blit reads [off, off+len) as its source: compare with what landed */
void ati_3d_watch_present(ATIVGAState *s, uint64_t off, uint64_t len)
{
    if (!watch_on() || off + len > s->vga.vram_size) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        ATIWatch *w = &watch[i];
        uint64_t n, first = ~0ull, bad = 0;

        if (!w->data || w->off != off) {
            continue;
        }
        n = MIN(len, w->len);
        for (uint64_t k = 0; k + 2 <= n; k += 2) {
            if (memcmp(w->data + k, s->vga.vram_ptr + off + k, 2)) {
                if (first == ~0ull) {
                    first = k;
                }
                bad++;
            }
        }
        trace_ati_watch_present(off, len, w->len, first, bad, w->w2d);
        return;
    }
    trace_ati_watch_present(off, len, 0, ~0ull, 0, ~0u);   /* nothing landed */
}

/* debug: the tracked render target (if any) holding VRAM offset off */
bool ati_3d_track_info(ATIVGAState *s, uint64_t off, uint32_t *fmt,
                       uint32_t *pitch, uint32_t *w, uint32_t *h,
                       bool *gl_owned)
{
    ATI3DState *t = s->r3d;

    for (int i = 0; t && t->r && i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        uint64_t o, contig;

        if (k->img.w &&
            ati_mc_translate(s, k->img.mc, &o, &contig) == ATI_MC_VRAM &&
            off >= o && off < o + (uint64_t)k->img.pitch * k->img.h) {
            *fmt = k->img.fmt;
            *pitch = k->img.pitch;
            *w = k->img.w;
            *h = k->img.h;
            *gl_owned = k->gl_owned;
            return true;
        }
    }
    return false;
}

void ati_3d_vram_written(ATIVGAState *s, uint64_t off, uint64_t len)
{
    ATI3DState *t = s->r3d;

    for (int i = 0; t && i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        uint64_t koff, contig;
        uint64_t klen = (uint64_t)k->img.pitch * k->img.h;

        if (k->gl_valid &&
            ati_mc_translate(s, k->img.mc, &koff, &contig) == ATI_MC_VRAM &&
            off < koff + klen && koff < off + len) {
            k->gl_valid = false;
        }
    }
}

/*
 * A solid 2D fill of an off-screen 3D render target, done as a GL clear.
 *
 * Mac OS's ATI OpenGL 2D-clears the next back buffer -- the buffer the
 * display is just leaving -- every frame.  In VRAM that meant: waiting for the
 * frame in flight to land (so the display could leave the buffer, else it
 * showed black), a 4 MB fill at 1920x1080, and uploading the cleared buffer
 * to GL for the first draw.  In GL it is a scissored clear: the vCPU doesn't
 * wait, the old frame stays intact in VRAM until the new one lands, and the
 * next draw finds the target GL-owned.
 *
 * Only for a rectangle inside one tracked target, with that target's pitch
 * and depth, never the buffer the CRTC shows or is about to show (that keeps
 * the Finder's and single-buffered drawing in VRAM).  A partial fill needs GL
 * to hold the rest of the target already: GL-owned, or a readback that VRAM
 * still matches.  Afterwards the target is GL-owned like one drawn into; VRAM
 * catches up at the next sync point, as for 3D drawing.
 */
bool ati_3d_fill(ATIVGAState *s, uint32_t dst_offset, int stride, int bypp,
                 int x, int y, int w, int h, uint32_t filler)
{
    ATI3DState *t = s->r3d;
    ATITrack *hit = NULL;
    uint64_t smc, slen, soff, koff = 0, contig, start, end, rel;
    int tx, ty;
    bool whole;
    uint8_t px[4];

    if (!t || !t->r || !ati_ext_mode(s) || w <= 0 || h <= 0 || !stride) {
        return false;
    }
    start = dst_offset + (uint64_t)y * stride + (uint64_t)x * bypp;
    end = dst_offset + (uint64_t)(y + h - 1) * stride + (uint64_t)(x + w) * bypp;
    ati_scanout_range(s, &smc, &slen);
    if (ati_mc_translate(s, smc, &soff, &contig) != ATI_MC_VRAM ||
        (start < soff + slen && soff < end)) {
        return false;
    }
    for (int i = 0; i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        uint64_t o;
        uint64_t klen = (uint64_t)k->img.pitch * k->img.h;

        if (!k->img.w ||
            ati_mc_translate(s, k->img.mc, &o, &contig) != ATI_MC_VRAM ||
            !(start < o + klen && o < end)) {
            continue;
        }
        if (hit) {
            return false;                       /* spans two targets */
        }
        hit = k;
        koff = o;
    }
    if (!hit || hit->img.pitch != (uint32_t)stride ||
        ati_fmt_bpp(hit->img.fmt) != (unsigned)bypp || start < koff ||
        end > koff + (uint64_t)hit->img.pitch * hit->img.h) {
        return false;
    }
    rel = start - koff;
    if ((rel % stride) % bypp) {
        return false;
    }
    tx = (rel % stride) / bypp;
    ty = rel / stride;
    if (tx + w > hit->img.w || ty + h > hit->img.h) {
        return false;
    }
    whole = !tx && !ty && w == hit->img.w && h == hit->img.h;
    if (!whole && !hit->gl_owned) {
        /* GL must hold the rest of the target: VRAM unchanged since the
         * readback (the same test as a draw's, track_acquire) */
        if (!hit->gl_valid ||
            ati_vram_hash(s->vga.vram_ptr + koff,
                          (size_t)hit->img.pitch * hit->img.h) != hit->vram_hash) {
            return false;
        }
    }
    stl_he_p(px, filler);
    ati_glr_fill(t->r, &hit->img, tx, ty, w, h, px);
    hit->gl_owned = true;
    hit->gl_valid = false;
    hit->gen++;
    t->gl_fills++;
    return true;
}

/* no readback in flight for [mc, mc+len) */
bool ati_3d_range_ready(ATIVGAState *s, uint64_t mc, uint64_t len)
{
    ATI3DState *t = s->r3d;

    for (int i = 0; t && i < MAX_TRACK; i++) {
        ATITrack *k = &t->track[i];
        uint64_t a = k->img.mc, b = a + (uint64_t)k->img.pitch * k->img.h;
        if (k->pending && mc < b && mc + len > a) {
            return false;
        }
    }
    return true;
}

/*
 * The vblank on the scanout [mc, mc+len): finished readbacks land in VRAM
 * (never waiting for the host), and a target drawn since its last read gets
 * a new asynchronous read (single-buffered drawing shows up a vblank or two
 * later).  2D operations and status reads still read back synchronously
 * (ati_3d_sync).
 */

void ati_3d_sync_range(ATIVGAState *s, uint64_t mc, uint64_t len)
{
    ATI3DState *t = s->r3d;

    if (!t || !t->r) {
        return;
    }
    scanout_read_async(s, mc, len);
    /*
     * Mac OS's ATI RAVE presents by copying the GL back buffer to the front
     * with the CPU through the aperture (no 2D blit, no CRTC flip), so every
     * target drawn since its last landing is read back, not only the
     * scanout: the CPU must see current pixels.
     */
    scanout_read_async(s, 0, 1ULL << 40);
}

static ATITrack *track_acquire(ATIVGAState *s, const ATIImage *img)
{
    ATI3DState *t = s->r3d;
    ATITrack *k = track_find(t, img->mc, img->pitch, img->fmt);
    size_t len = (size_t)img->pitch * img->h;
    uint8_t *buf;

    if (k && k->gl_owned) {
        if (img->h <= k->img.h) {
            return k;
        }
        track_readback(s, k);
    }
    if (k && k->gl_valid && img->h <= k->img.h) {
        uint64_t off, contig;
        size_t klen = (size_t)k->img.pitch * k->img.h;
        if (ati_mc_translate(s, k->img.mc, &off, &contig) == ATI_MC_VRAM &&
            off + klen <= s->vga.vram_size &&
            ati_vram_hash(s->vga.vram_ptr + off, klen) == k->vram_hash) {
            /* nobody wrote it since the readback: GL's copy is current */
            k->gl_owned = true;
            t->upload_skips++;
            return k;
        }
    }
    if (!k) {
        k = &t->track[0];
        for (int i = 0; i < MAX_TRACK; i++) {
            if (!t->track[i].img.w) {
                k = &t->track[i];
                break;
            }
            if (!t->track[i].gl_owned) {
                k = &t->track[i];
            }
        }
        track_readback(s, k);
    }
    k->img = *img;
    /* the colour buffer's current contents become the GL target's */
    buf = g_malloc(len);
    ati_mc_rw(s, img->mc, buf, len, false);
    ati_glr_target_upload_take(t->r, img, buf);  /* buf is the queue's now */
    k->gl_owned = true;
    k->gl_valid = false;
    t->uploads++;
    return k;
}

/* ---------------------------------------------------------------------
 * State decoding
 */

static const uint16_t gl_cmp[8] = {
    G_NEVER, G_LESS, G_LEQUAL, G_EQUAL, G_GEQUAL, G_GREATER, G_NOTEQUAL,
    G_ALWAYS
};

static uint16_t blend_factor(unsigned f)
{
    switch (f) {
    case 32: return 0;              /* GL_ZERO */
    case 33: return 1;              /* GL_ONE */
    case 34: return 0x0300;         /* SRC_COLOR */
    case 35: return 0x0301;
    case 36: return 0x0306;         /* DST_COLOR */
    case 37: return 0x0307;
    case 38: return 0x0302;         /* SRC_ALPHA */
    case 39: return 0x0303;
    case 40: return 0x0304;         /* DST_ALPHA */
    case 41: return 0x0305;
    case 42: return 0x0308;         /* SRC_ALPHA_SATURATE */
    default: return 1;
    }
}

static uint16_t stencil_op(unsigned op)
{
    static const uint16_t m[8] = { 0x1E00, 0, 0x1E01, 0x1E02, 0x1E03, 0x150A,
                                   0x1E00, 0x1E00 };
    return m[op & 7];
}

static uint8_t color_fmt(unsigned f)
{
    switch (f) {
    case 3: return ATI_FMT_ARGB1555;
    case 4: return ATI_FMT_RGB565;
    case 6: return ATI_FMT_ARGB8888;
    case 7: return ATI_FMT_RGB332;
    case 8: return ATI_FMT_Y8;
    case 15: return ATI_FMT_ARGB4444;
    default: return ATI_FMT_NONE;
    }
}

/*
 * RB3D_PLANEMASK holds a mask in the colour buffer's pixel layout; GL masks
 * whole channels, so a channel is written when any of its bits is (partial
 * channel masks are not used by the guest drivers seen so far).
 */
static uint8_t plane_mask_channels(uint8_t fmt, uint32_t m)
{
    /* shift and width of R, G, B, A */
    static const uint8_t lay[][8] = {
        [ATI_FMT_ARGB1555] = { 10, 5, 5, 5, 0, 5, 15, 1 },
        [ATI_FMT_RGB565]   = { 11, 5, 5, 6, 0, 5, 0, 0 },
        [ATI_FMT_ARGB8888] = { 16, 8, 8, 8, 0, 8, 24, 8 },
        [ATI_FMT_RGB332]   = { 5, 3, 2, 3, 0, 2, 0, 0 },
        [ATI_FMT_ARGB4444] = { 8, 4, 4, 4, 0, 4, 12, 4 },
    };
    uint8_t r = 0;

    if (fmt >= ARRAY_SIZE(lay) || !lay[fmt][1]) {
        return m ? 15 : 0;
    }
    for (int c = 0; c < 4; c++) {
        unsigned sh = lay[fmt][2 * c], w = lay[fmt][2 * c + 1];
        if (!w) {
            r |= 1 << c;        /* no such channel: GL's copy may change */
        } else if ((m >> sh) & ((1u << w) - 1)) {
            r |= 1 << c;
        }
    }
    return r;
}

static uint8_t tex_fmt(unsigned f)
{
    switch (f & 31) {
    case 0: return ATI_FMT_I8;
    case 1: return ATI_FMT_AI88;
    case 2: return ATI_FMT_RGB332;
    case 3: return ATI_FMT_ARGB1555;
    case 4: return ATI_FMT_RGB565;
    case 5: return ATI_FMT_ARGB4444;
    case 6: return ATI_FMT_ARGB8888;
    case 7: return ATI_FMT_RGBA8888;
    case 8: return ATI_FMT_Y8;
    case 10: return ATI_FMT_VYUY422;
    case 11: return ATI_FMT_YVYU422;
    case 12: return ATI_FMT_DXT1;
    case 14: return ATI_FMT_DXT23;
    case 15: return ATI_FMT_DXT45;
    default: return ATI_FMT_NONE;
    }
}

static uint16_t wrap_mode(unsigned c)
{
    switch (c & 7) {
    case 0: return 0x2901;              /* REPEAT */
    case 1: return 0x8370;              /* MIRRORED_REPEAT */
    default: return 0x812F;             /* CLAMP_TO_EDGE */
    }
}

/*
 * Has a texture in VRAM been written since we last looked?  Uses the VGA
 * dirty log, which catches CPU writes through the apertures as well as the
 * 2D/3D engines (they mark what they draw).  The texture's own pages are
 * tested and cleared (page-exact, and CPU writes re-arm the log); pages
 * that the scanout uses are left to the display.
 * Returns 1 dirty, 0 clean, -1 not tracked (caller compares contents).
 */
static int tex_vram_dirty(ATIVGAState *s, uint64_t mc, uint32_t len)
{
    uint64_t off, contig, so, sl, a0, a1;
    DirtyBitmapSnapshot *snap;
    bool dirty;

    if (!len || ati_mc_translate(s, mc, &off, &contig) != ATI_MC_VRAM ||
        contig < len) {
        return -1;
    }
    so = s->regs.crtc_offset & 0x07ffffff;
    sl = (uint64_t)s->vga.vbe_line_offset * s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
    a0 = QEMU_ALIGN_DOWN(off, 4096);
    a1 = QEMU_ALIGN_UP(off + len, 4096);
    if (a0 < so + sl && so < a1) {
        return -1;
    }
    /* QEMU 10.0 (UTM): no physical_memory_test_and_clear_dirty() in common code */
    snap = memory_region_snapshot_and_clear_dirty(&s->vga.vram, a0, a1 - a0,
                                                  DIRTY_MEMORY_VGA);
    dirty = memory_region_snapshot_get_dirty(&s->vga.vram, snap, a0, a1 - a0);
    g_free(snap);
    return dirty ? 1 : 0;
}

static void decode_texture(ATIVGAState *s, int u, ATITexUnit *tu)
{
    ATI3DState *t = s->r3d;
    uint32_t filt = R(s, PP_TXFILTER_0 + u * PP_TEX_STRIDE);
    uint32_t fmt = R(s, PP_TXFORMAT_0 + u * PP_TEX_STRIDE);
    uint32_t offs = R(s, PP_TXOFFSET_0 + u * PP_TEX_STRIDE);
    unsigned f = tex_fmt(fmt), bpp = ati_fmt_bpp(f);
    unsigned w, h, maxl, levels;
    uint32_t pos;
    uint8_t swp;

    memset(tu, 0, sizeof(*tu));
    tu->enabled = true;
    if (fmt & (1u << 7)) {      /* NON_POWER2: size from PP_TEX_SIZE */
        uint32_t sz = R(s, PP_TEX_SIZE_0 + u * PP_TEX_NPOT_STRIDE);
        w = (sz & 0x7ff) + 1;
        h = ((sz >> 16) & 0x7ff) + 1;
        maxl = 0;
    } else {
        w = 1u << ((fmt >> 8) & 15);
        h = 1u << ((fmt >> 12) & 15);
        maxl = (filt >> 16) & 15;
    }
    if (!bpp) {
        /* compressed or unsupported: not decoded yet (drawn untextured) */
        static bool warned;
        if (!warned) {
            warned = true;
            qemu_log_mask(LOG_UNIMP, "ati: texture format %u unsupported\n",
                          fmt & 31);
        }
        tu->enabled = false;
        return;
    }
    if (trace_event_get_state_backends(TRACE_ATI_3D_TEX) ||
        trace_event_get_state_backends(TRACE_ATI_3D_TEX_FLAGS)) {
        /* only when a unit's texture state changes (it was one line per draw) */
        static uint32_t last[8][7];
        uint32_t cur[7] = { offs, fmt, filt, w, h,
            R(s, PP_TXCBLEND_0 + u * PP_TEX_STRIDE),
            R(s, PP_TXABLEND_0 + u * PP_TEX_STRIDE) };

        if (u < 8 && memcmp(last[u], cur, sizeof(cur))) {
            memcpy(last[u], cur, sizeof(cur));
            trace_ati_3d_tex(u, offs, fmt, filt, w, h, cur[5], cur[6]);
            /* PP_TXOFFSET low bits: endian swap, macro (bit 2) / micro (bit 3) tile */
            trace_ati_3d_tex_flags(u, offs & ~31u, offs & 3, (offs >> 2) & 1,
                                   (offs >> 3) & 1, fmt & 31, f);
        }
    }
    /* ALPHA_IN_MAP (bit 6): without it the sampler returns alpha 1 (the
     * ATI RAVE engine leaves it clear for 16-bit GWorld textures whose
     * top bit is not alpha, e.g. Nanosaur's GPS map). */
    tu->alpha_one = !(fmt & (1u << 6));
    swp = swap_code((fmt >> 26) & 3) ^ swap_code(offs & 3) ^
          aperture_swap(s, bpp * 8);
    tu->img.mc = offs & ~31u;
    tu->img.fmt = f;
    tu->img.swap = swp;
    tu->img.w = w;
    tu->img.h = h;
    pos = 0;
    levels = 0;
    for (unsigned l = 0; l <= maxl && l < 12; l++) {
        unsigned lw = MAX(w >> l, 1), lh = MAX(h >> l, 1);
        uint32_t pitch;
        if (fmt & (1u << 7)) {
            pitch = (R(s, PP_TEX_PITCH_0 + u * PP_TEX_NPOT_STRIDE) & 0x3fe0) + 32;
        } else {
            pitch = ROUND_UP(lw * bpp, 32);
        }
        tu->level_offs[l] = pos;
        tu->level_w[l] = lw;
        tu->level_h[l] = lh;
        tu->level_pitch[l] = pitch;
        pos += pitch * lh;
        pos = ROUND_UP(pos, 32);
        levels++;
        if (lw == 1 && lh == 1) {
            break;
        }
    }
    tu->levels = levels;
    tu->img.pitch = tu->level_pitch[0];
    {
        static const uint16_t minf[16] = {
            [0] = 0x2600, [1] = 0x2601, [2] = 0x2700, [3] = 0x2702,
            [6] = 0x2701, [7] = 0x2703, [8] = 0x2600, [9] = 0x2601,
            [10] = 0x2700, [11] = 0x2702,
        };
        uint16_t m = minf[(filt >> 1) & 15];
        tu->min_filter = m ? m : 0x2601;
        tu->mag_filter = (filt & 1) ? 0x2601 : 0x2600;
    }
    tu->wrap_s = wrap_mode(filt >> 23);
    tu->wrap_t = wrap_mode(filt >> 27);
    argb_to_float(R(s, PP_BORDER_COLOR_0 + u * 4), tu->border);
    tu->chroma_key = fmt & (1u << 29);

    /* identity and content: a cache entry per (address, layout) */
    {
        uint64_t key = ((uint64_t)tu->img.mc << 32) ^
                       ((uint64_t)f << 24) ^ (w << 12) ^ h ^
                       ((uint64_t)swp << 30) ^ ((uint64_t)levels << 28);
        uint32_t len = pos;
        uint8_t *data;
        uint32_t hash;
        /* the address is in the key's high bits: mix before indexing */
        ATITexCache *c = &t->texc[(key * 0x9E3779B97F4A7C15ULL) >> (64 - TEXC_BITS)];
        int tracked = tex_vram_dirty(s, tu->img.mc, len);

        if (tracked == 0 && c->valid && c->key == key && c->len == len) {
            /* VRAM texture not written since it was last read */
            tu->key = key;
            tu->gen = c->gen;
            t->texhits++;
            return;
        }
        data = g_malloc(len);
        ati_mc_rw(s, tu->img.mc, data, len, false);
        c->len = len;
        /* cheap content hash: guests rewrite textures in place */
        hash = 2166136261u;
        for (uint32_t i = 0; i < len; i += 4) {
            hash = (hash ^ ldl_he_p(data + i)) * 16777619u;
        }
        tu->key = key;
        if (!c->valid || c->key != key || c->gen != hash) {
            c->valid = true;
            c->key = key;
            c->gen = hash;
            ati_glr_texture(t->r, key, hash, tu, data, len);
            t->texuploads++;
        }
        tu->gen = hash;
        g_free(data);
    }
}

/* the colour target only (for clears) */
static bool decode_targets(ATIVGAState *s, ATIDrawState *st)
{
    uint32_t rb3d = R(s, RB3D_CNTL), cpitch = R(s, RB3D_COLORPITCH);
    uint8_t cf = color_fmt((rb3d >> 10) & 15);
    unsigned bpp = ati_fmt_bpp(cf);
    uint32_t wh = R(s, RE_WIDTH_HEIGHT);

    memset(st, 0, sizeof(*st));
    if (!bpp) {
        return false;
    }
    st->color.mc = R(s, RB3D_COLOROFFSET) & ~15u;
    st->color.pitch = (cpitch & 0x1ff8) * bpp;
    st->color.fmt = cf;
    st->color.swap = swap_code((cpitch >> 18) & 3) ^ aperture_swap(s, bpp * 8);
    st->color.w = cpitch & 0x1ff8;
    st->color.h = ((wh >> 16) & 0x7ff) + 1;
    return true;
}

static bool decode_state(ATIVGAState *s, ATIDrawState *st)
{
    uint32_t rb3d = R(s, RB3D_CNTL), pp = R(s, PP_CNTL), se = R(s, SE_CNTL);
    uint32_t zs = R(s, RB3D_ZSTENCILCNTL), misc = R(s, PP_MISC);
    uint32_t cpitch = R(s, RB3D_COLORPITCH);
    uint8_t cf = color_fmt((rb3d >> 10) & 15);
    unsigned bpp = ati_fmt_bpp(cf);
    uint32_t tl = R(s, RE_TOP_LEFT), wh = R(s, RE_WIDTH_HEIGHT);

    memset(st, 0, sizeof(*st));
    if (!bpp) {
        return false;
    }
    st->color.mc = R(s, RB3D_COLOROFFSET) & ~15u;
    st->color.pitch = (cpitch & 0x1ff8) * bpp;
    st->color.fmt = cf;
    st->color.swap = swap_code((cpitch >> 18) & 3) ^ aperture_swap(s, bpp * 8);
    /* surface height: up to the scissor's bottom edge */
    st->color.w = cpitch & 0x1ff8;
    st->color.h = ((wh >> 16) & 0x7ff) + 1;

    st->scissor = true;     /* RE_TOP_LEFT/WIDTH_HEIGHT always clip */
    st->sc_x0 = tl & 0x7ff;
    st->sc_y0 = (tl >> 16) & 0x7ff;
    st->sc_x1 = (wh & 0x7ff);
    st->sc_y1 = (wh >> 16) & 0x7ff;

    if (rb3d & (1 << 8)) {          /* Z_ENABLE */
        st->depth_mc = R(s, RB3D_DEPTHOFFSET) & ~15u;
        st->depth_pitch = R(s, RB3D_DEPTHPITCH) & 0x1ff8;
        st->z_test = true;
        st->z_func = gl_cmp[(zs >> 4) & 7];
        st->z_write = zs & (1u << 30);
        st->depth_fmt = (zs & 15) == 0 ? 16 : 24;
        if (rb3d & (1 << 7)) {      /* STENCIL_ENABLE */
            uint32_t ref = R(s, RB3D_STENCILREFMASK);
            st->stencil = true;
            st->st_func = gl_cmp[(zs >> 12) & 7];
            st->st_fail = stencil_op(zs >> 16);
            st->st_zpass = stencil_op(zs >> 20);
            st->st_zfail = stencil_op(zs >> 24);
            st->st_ref = ref & 0xff;
            st->st_mask = (ref >> 16) & 0xff;
            st->st_wmask = ref >> 24;
        }
    }
    if (rb3d & 1) {                 /* ALPHA_BLEND_ENABLE */
        uint32_t b = R(s, RB3D_BLENDCNTL);
        st->blend = true;
        st->blend_src = blend_factor((b >> 16) & 63);
        st->blend_dst = blend_factor((b >> 24) & 63);
        st->blend_eq = ((b >> 12) & 2) ? 0x800B /* REVERSE_SUBTRACT */ : 0x8006;
        if (((b >> 12) & 3) >= 2) {
            st->blend_eq = 0x800A;  /* FUNC_SUBTRACT: src - dst */
        }
    }
    if (rb3d & (1 << 6)) {
        st->rop_enable = true;
        st->rop = (R(s, RB3D_ROPCNTL) >> 8) & 15;
    }
    /* ATI's OpenGL clears depth alone with a rectangle drawn under plane
     * mask 0 when the target has no fast Z clear (r_mode 0's back buffer) */
    st->plane_mask = (rb3d & 2) ? R(s, RB3D_PLANEMASK) : 0xffffffff;
    st->color_mask = plane_mask_channels(cf, st->plane_mask);
    if (st->rop_enable) {
        switch (st->rop) {
        case 10:                    /* NOOP: the destination stays */
            st->color_mask = 0;
            break;
        case 12:                    /* COPY */
            break;
        default: {
            static bool warned;
            if (!warned) {
                warned = true;
                qemu_log_mask(LOG_UNIMP, "ati: 3D raster op %u drawn as copy\n",
                              st->rop);
            }
        }
        }
    }
    trace_ati_3d_rb(rb3d, st->plane_mask, R(s, RB3D_ROPCNTL), st->color_mask);

    /* culling: SE_CNTL bit 0 = front faces are CCW; b/f solid or cull */
    {
        bool bcull = ((se >> 1) & 3) != 3, fcull = ((se >> 3) & 3) != 3;
        st->front_ccw = se & 1;
        if (bcull || fcull) {
            st->cull_enable = true;
            st->cull_face = (bcull && fcull) ? 0x0408 : bcull ? 0x0405 : 0x0404;
        }
        /*
         * The renderer flips y (row 0 = top of the guest surface), which
         * mirrors the winding; compensate.
         */
        st->front_ccw = !st->front_ccw;
    }
    st->flat = ((se >> 8) & 3) == 1;
    st->solid_color = ((se >> 8) & 3) == 0;
    argb_to_float(R(s, RE_SOLID_COLOR), st->solid);
    st->zbias = (se & (1 << 18)) != 0;
    st->zbias_factor = RF(s, SE_ZBIAS_FACTOR);
    st->zbias_const = RF(s, SE_ZBIAS_CONSTANT);

    if (pp & (1 << 23)) {
        st->alpha_test = true;
        st->alpha_func = (misc >> 8) & 7;
        st->alpha_ref = (misc & 0xff) / 255.0f;
    }
    if (pp & (1 << 22)) {
        uint32_t fc = R(s, PP_FOG_COLOR);
        st->fog = true;
        argb_to_float(fc | 0xff000000, st->fog_color);
        st->fog_src = ((fc >> 24) & 1) ? 1 : (((fc >> 25) & 3) == 2 ? 2 : 0);
    }
    st->specular = pp & (1 << 21);
    st->stage_enable = (pp >> 12) & 7;
    st->tex_enable = (pp >> 4) & 7;
    for (int u = 0; u < 3; u++) {
        st->stage[u].cblend = R(s, PP_TXCBLEND_0 + u * PP_TEX_STRIDE);
        st->stage[u].ablend = R(s, PP_TXABLEND_0 + u * PP_TEX_STRIDE);
        argb_to_float(R(s, PP_TFACTOR_0 + u * PP_TEX_STRIDE),
                      st->stage[u].tfactor);
        if (st->tex_enable & (1 << u)) {
            decode_texture(s, u, &st->tex[u]);
            if (!st->tex[u].enabled) {
                st->tex_enable &= ~(1 << u);
            }
        }
    }
    st->persp_tex = true;
    st->line_width = 1.0f;
    return true;
}

/* ---------------------------------------------------------------------
 * Vertices
 */

static unsigned vtx_dwords(uint32_t f)
{
    unsigned n = 2;
    if (f & VF_Z) n++;
    if (f & VF_W0) n++;
    if (f & VF_FPCOLOR) n += 3;
    if (f & VF_FPALPHA) n++;
    if (f & VF_PKCOLOR) n++;
    if (f & VF_FPSPEC) n += 3;
    if (f & VF_FPFOG) n++;
    if (f & VF_PKSPEC) n++;
    if (f & VF_ST0) n += 2;
    if (f & VF_Q0) n++;
    if (f & VF_ST1) n += 2;
    if (f & VF_Q1) n++;
    if (f & VF_ST2) n += 2;
    if (f & VF_Q2) n++;
    if (f & VF_ST3) n += 2;
    if (f & VF_Q3) n++;
    if (f & VF_N0) n += 3;
    return n;
}

static inline float fl(uint32_t u)
{
    union { uint32_t u; float f; } v = { .u = u };
    return v.f;
}

/* float -> scaled int for the traces, without overflowing */
static inline int tr_i(float f, float k)
{
    float v = f * k;

    return v != v ? 0 : v > 2e9f ? 2000000000 : v < -2e9f ? -2000000000 : (int)v;
}

/* ---------------------------------------------------------------------
 * Hardware TCL (see ATI-R100-TCL.md)
 *
 * T1: the position transform.  The guest loads matrices into the vector
 * memory through SE_TCL_VECTOR_INDX/DATA; a draw with TCL on carries object
 * space positions, which decode_vertex() multiplies by the model-view-
 * projection matrix, divides by w and sends through the viewport transform
 * like any other vertex.  Lighting, texgen, texture matrices, fog and user
 * clip planes are not emulated yet (logged once).
 */

static void tcl_reg_write(ATI3DState *t, hwaddr addr, uint32_t data)
{
    switch (addr) {
    case SE_TCL_VECTOR_INDX_REG:
        /* [7:0] octword address, [23:16] octword stride, [31:28] dword count */
        t->vec_addr = data & 0xff;
        t->vec_stride = (data >> 16) & 0xff ? (data >> 16) & 0xff : 1;
        t->vec_dw = 0;
        break;
    case SE_TCL_VECTOR_DATA_REG:
        t->tcl_vec[(t->vec_addr & (TCL_VEC_N - 1)) * 4 + t->vec_dw] = data;
        if (++t->vec_dw == 4) {
            t->vec_dw = 0;
            t->vec_addr = (t->vec_addr + t->vec_stride) & (TCL_VEC_N - 1);
        }
        break;
    case SE_TCL_SCALAR_INDX_REG:
        t->scal_addr = data & (TCL_SCAL_N - 1);
        t->scal_stride = (data >> 16) & 0xff ? (data >> 16) & 0xff : 1;
        break;
    case SE_TCL_SCALAR_DATA_REG:
        t->tcl_scal[t->scal_addr & (TCL_SCAL_N - 1)] = data;
        t->scal_addr = (t->scal_addr + t->scal_stride) & (TCL_SCAL_N - 1);
        break;
    }
}

/* the four octwords at vector address a are the rows of a 4x4 matrix */
static bool tcl_matrix(ATI3DState *t, unsigned a, float m[4][4])
{
    bool nonzero = false;

    for (unsigned r = 0; r < 4; r++) {
        for (unsigned c = 0; c < 4; c++) {
            uint32_t u = t->tcl_vec[((a + r) & (TCL_VEC_N - 1)) * 4 + c];
            nonzero |= u != 0;
            m[r][c] = fl(u);
        }
    }
    return nonzero;
}

static void tcl_vec4(ATI3DState *t, unsigned a, float *o)
{
    for (int c = 0; c < 4; c++) {
        o[c] = fl(t->tcl_vec[(a & (TCL_VEC_N - 1)) * 4 + c]);
    }
}

static float tcl_scl(ATI3DState *t, unsigned a)
{
    return fl(t->tcl_scal[a & (TCL_SCAL_N - 1)]);
}

static void reg_vec4(ATIVGAState *s, unsigned reg, float *o)
{
    for (int c = 0; c < 4; c++) {
        o[c] = fl(R(s, reg + 4 * c));
    }
}

/*
 * T2..T4: collect the state the post-transform stages need (see
 * ATI-R100-TCL.md for the source of each register).  Done once per draw, and
 * only for the stages that are switched on.
 */
static void tcl_fx_setup(ATIVGAState *s, uint32_t tpc, uint32_t lm,
                         uint32_t ucp, uint32_t sel0, uint32_t sel1)
{
    ATI3DState *t = s->r3d;
    TCLFx *x = &t->tx;
    static bool w_texin;

    memset(x, 0, sizeof(*x));
    x->lm = lm;
    x->ucp_ctl = ucp;
    x->tpc = tpc;
    x->vsel = R(s, SE_TCL_OUTPUT_VTX_SEL);
    x->light = lm & 1;
    x->fog = (ucp >> 8) & 3;
    x->ucp = (ucp >> 2) & 0x3f;
    /* texture: a unit is computed when its output select has bit 3 set */
    for (int u = 0; u < 3; u++) {
        unsigned sel = (x->vsel >> (16 + 4 * u)) & 0xf;

        if (sel & 8) {
            unsigned in = (tpc >> (16 + 4 * u)) & 0xf;

            x->tex = true;
            if (in > 8 && !w_texin) {
                w_texin = true;
                qemu_log_mask(LOG_UNIMP, "ati: TCL texgen input %u unknown "
                              "(texture proc 0x%08x)\n", in, tpc);
            }
            if (tpc & (0x10u << u)) {
                tcl_matrix(t, ((sel1 >> (16 + 4 * u)) & 0xf) * 4, x->texm[u]);
            }
        }
    }
    if (x->light || x->fog || x->ucp || x->tex) {
        /* eye-space matrices */
        tcl_matrix(t, (sel0 & 0xf) * 4, x->mv);
        tcl_matrix(t, ((sel0 >> 16) & 0xf) * 4, x->it);
    }
    if (x->light) {
        tcl_vec4(t, TCL_VS_GLOBAL_AMBIENT, x->glt);
        tcl_vec4(t, TCL_VS_EYE_VECTOR, x->eye);
        reg_vec4(s, SE_TCL_MATERIAL_EMISSIVE, x->emis);
        reg_vec4(s, SE_TCL_MATERIAL_EMISSIVE + 0x10, x->mamb);
        reg_vec4(s, SE_TCL_MATERIAL_EMISSIVE + 0x20, x->mdif);
        reg_vec4(s, SE_TCL_MATERIAL_EMISSIVE + 0x30, x->mspec);
        x->shine = fl(R(s, SE_TCL_SHININESS));
        for (unsigned l = 0; l < 8; l++) {
            uint32_t ctl = (R(s, SE_TCL_PER_LIGHT_CTL_0 + 4 * (l / 2)) >>
                            (16 * (l & 1))) & 0xffff;
            TCLLight *L;

            if (!(ctl & 1)) {
                continue;
            }
            L = &x->L[x->nl++];
            L->ctl = ctl;
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + l, L->amb);
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + 8 + l, L->dif);
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + 16 + l, L->spec);
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + 24 + l, L->pos);
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + 32 + l, L->spot);
            tcl_vec4(t, TCL_VS_LIGHT_AMBIENT + 40 + l, L->att);
            L->exponent = tcl_scl(t, TCL_SS_LIGHT_EXPONENT + l);
            L->cutoff = tcl_scl(t, TCL_SS_LIGHT_EXPONENT + 8 + l);
            L->const_inv = tcl_scl(t, TCL_SS_LIGHT_EXPONENT + 32 + l);
        }
    }
    if (x->ucp) {
        for (int p = 0; p < 6; p++) {
            tcl_vec4(t, TCL_VS_UCP + p, x->plane[p]);
        }
    }
    if (x->fog) {
        float f[4];

        tcl_vec4(t, TCL_VS_FOG_PARAM, f);
        x->fog_c = f[1];
        x->fog_d = f[2];
    }
}

static inline void mat4v(const float m[4][4], const float *v, float *o)
{
    for (int r = 0; r < 4; r++) {
        o[r] = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] +
               m[r][3] * v[3];
    }
}

static inline float dot3(const float *a, const float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void norm3(float *v)
{
    float l = sqrtf(dot3(v, v));

    if (l > 1e-20f) {
        v[0] /= l; v[1] /= l; v[2] /= l;
    }
}

static inline float sat(float v)
{
    return v != v ? 0.0f : v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

/* a material colour source: 0 premultiplied into the light, 1 state, 2/3 vertex */
static void lm_src(unsigned src, const float *state, const float *c0,
                   const float *c1, float *o)
{
    const float *p = src == 1 ? state : src == 2 ? c0 : src == 3 ? c1 : NULL;

    for (int i = 0; i < 4; i++) {
        o[i] = p ? p[i] : 1.0f;
    }
}

/*
 * T2: texture matrices and texgen.  Output unit u is either an input
 * texcoord (select 0..3) or, with bit 3 set, the generated vector
 * (TEXGEN input: 0..3 texcoord, 4 object, 5 eye, 6 eye normal, 7 reflection,
 * 8 normalised eye) run through the unit's texture matrix when
 * TEXMAT_u_ENABLE.  Inputs without a q default to (s, t, 1, 1).
 */
static void tcl_texture(const TCLFx *x, ATIVertex *v, const float tin[3][4],
                        const float *obj, const float *eye, const float *nrm)
{
    for (int u = 0; u < 3; u++) {
        unsigned sel = (x->vsel >> (16 + 4 * u)) & 0xf;
        float in[4], o[4];

        if (!(sel & 8)) {
            if ((sel & 7) < 3 && (sel & 7) != (unsigned)u) {
                memcpy(v->tc[u], tin[sel & 7], 3 * sizeof(float));
            }
            continue;
        }
        switch ((x->tpc >> (16 + 4 * u)) & 0xf) {
        case 4: memcpy(in, obj, sizeof(in)); break;
        case 5: memcpy(in, eye, sizeof(in)); break;
        case 8: memcpy(in, eye, sizeof(in)); norm3(in); in[3] = 1.0f; break;
        case 6:
            memcpy(in, nrm, 3 * sizeof(float)); in[3] = 1.0f;
            break;
        case 7: {       /* reflect the eye vector about the normal */
            float e[3] = { eye[0], eye[1], eye[2] }, d;

            norm3(e);
            d = 2.0f * dot3(e, nrm);
            for (int i = 0; i < 3; i++) {
                in[i] = e[i] - d * nrm[i];
            }
            in[3] = 1.0f;
            break;
        }
        default:
            in[0] = tin[(x->tpc >> (16 + 4 * u)) & 3][0];
            in[1] = tin[(x->tpc >> (16 + 4 * u)) & 3][1];
            in[2] = tin[(x->tpc >> (16 + 4 * u)) & 3][2];
            in[3] = 1.0f;
            break;
        }
        if (x->tpc & (0x10u << u)) {
            mat4v(x->texm[u], in, o);
        } else {
            memcpy(o, in, sizeof(o));
        }
        v->tc[u][0] = o[0];
        v->tc[u][1] = o[1];
        v->tc[u][2] = o[2];
    }
}

/* T3: fixed-function lighting; fills v->c0 (and c1 rgb when not combined) */
static void tcl_light(const TCLFx *x, ATIVertex *v, bool have_n,
                      const float *pos, const float *n)
{
    unsigned es = (x->lm >> 16) & 3, as = (x->lm >> 18) & 3;
    unsigned ds = (x->lm >> 20) & 3, ss = (x->lm >> 22) & 3;
    float em[4], am[4], dm[4], sm[4], col[3], spc[3] = { 0, 0, 0 };
    float vw[3] = { x->eye[0], x->eye[1], -x->eye[2] };
    bool local_viewer = x->lm & (1 << 2);

    lm_src(es, x->emis, v->c0, v->c1, em);
    lm_src(as, x->mamb, v->c0, v->c1, am);
    lm_src(ds, x->mdif, v->c0, v->c1, dm);
    lm_src(ss, x->mspec, v->c0, v->c1, sm);
    for (int i = 0; i < 3; i++) {
        col[i] = (es ? em[i] : 0.0f) + x->glt[i] * am[i];
    }
    if (local_viewer) {
        vw[0] = -pos[0]; vw[1] = -pos[1]; vw[2] = -pos[2];
        norm3(vw);
    }
    for (int l = 0; l < x->nl; l++) {
        const TCLLight *L = &x->L[l];
        float lv[3], hv[3], att = 1.0f, spot = 1.0f, nl, d = 0.0f;

        if (L->ctl & 8) {               /* local light */
            for (int i = 0; i < 3; i++) {
                lv[i] = L->pos[i] - pos[i];
            }
            d = sqrtf(dot3(lv, lv));
            norm3(lv);
            if (L->ctl & 0x40) {        /* range attenuation */
                att = (L->ctl & 0x80) ? L->const_inv :
                      1.0f / (L->att[2] + L->att[1] * d + L->att[0] * d * d);
            }
            if (L->ctl & 0x10) {        /* spot: spot[] = -spot direction */
                float sd = dot3(lv, L->spot);

                spot = sd < L->cutoff || sd <= 0.0f ? 0.0f :
                       L->exponent != 0.0f ? powf(sd, L->exponent) : 1.0f;
            }
        } else {
            memcpy(lv, L->pos, sizeof(lv));
            norm3(lv);
        }
        att *= spot;
        if ((L->ctl & 2)) {
            for (int i = 0; i < 3; i++) {
                col[i] += att * L->amb[i] * am[i];
            }
        }
        if (!have_n) {
            continue;
        }
        nl = dot3(n, lv);
        if (nl <= 0.0f) {
            continue;
        }
        for (int i = 0; i < 3; i++) {
            col[i] += att * nl * L->dif[i] * dm[i];
        }
        if ((x->lm & 0x20) && (L->ctl & 4)) {
            float nh, sp;

            for (int i = 0; i < 3; i++) {
                hv[i] = lv[i] + vw[i];
            }
            norm3(hv);
            nh = dot3(n, hv);
            if (nh > 0.0f) {
                sp = x->shine == 0.0f ? 1.0f : powf(nh, x->shine);
                for (int i = 0; i < 3; i++) {
                    spc[i] += att * sp * L->spec[i] * sm[i];
                }
            }
        }
    }
    if (x->lm & 0x40) {         /* DIFFUSE_SPECULAR_COMBINE */
        for (int i = 0; i < 3; i++) {
            col[i] += spc[i];
            spc[i] = 0.0f;
        }
    }
    for (int i = 0; i < 3; i++) {
        v->c0[i] = sat(col[i]);
        v->c1[i] = sat(spc[i]);
    }
    v->c0[3] = sat(dm[3]);
}

/*
 * Everything after the position transform for one vertex: eye space,
 * texgen/texmat, lighting, fog and the user clip plane outcode (kept in
 * tc[0][3] until draw() has dropped the triangles outside a plane).
 */
static void tcl_post(ATI3DState *t, ATIVertex *v, uint32_t f,
                     const float *obj, const float *nrm, const float *clip,
                     const float tin[3][4])
{
    const TCLFx *x = &t->tx;
    float eye[4] = { obj[0], obj[1], obj[2], obj[3] };
    float n[3] = { nrm[0], nrm[1], nrm[2] }, lp[3];
    bool have_n = f & VF_N0;

    if (!(x->light || x->tex || x->fog || x->ucp)) {
        return;
    }
    mat4v(x->mv, obj, eye);
    if (have_n) {
        float ne[4], i4[4] = { nrm[0], nrm[1], nrm[2], 0.0f };

        mat4v(x->it, i4, ne);
        memcpy(n, (x->lm & 2) ? nrm : ne, sizeof(n));
        if (x->lm & 8) {
            norm3(n);
        } else if (x->lm & 16) {
            for (int i = 0; i < 3; i++) {
                n[i] *= x->eye[3];
            }
        }
    }
    if (x->tex) {
        float ne[3] = { n[0], n[1], n[2] };

        if (x->lm & 2) {    /* texgen wants eye-space normals */
            float i4[4] = { nrm[0], nrm[1], nrm[2], 0.0f }, o[4];

            mat4v(x->it, i4, o);
            memcpy(ne, o, sizeof(ne));
        }
        norm3(ne);
        tcl_texture(x, v, tin, obj, eye, ne);
    }
    if (x->light) {
        if (x->lm & 2) {
            lp[0] = obj[0]; lp[1] = obj[1]; lp[2] = obj[2];
        } else {
            lp[0] = eye[0]; lp[1] = eye[1]; lp[2] = eye[2];
        }
        tcl_light(x, v, have_n, lp, n);
    }
    if (x->fog) {
        float z = x->ucp_ctl & (1u << 10) ?
                  sqrtf(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]) :
                  fabsf(eye[2]);
        float ff;

        switch (x->fog) {
        case 1:  ff = expf(x->fog_d * z); break;
        case 2:  ff = expf(x->fog_d * z * z); break;
        default: ff = x->fog_c + x->fog_d * z; break;
        }
        v->c1[3] = sat(ff);
    }
    if (x->ucp) {
        unsigned out = 0;

        for (int p = 0; p < 6; p++) {
            const float *pl = x->plane[p];
            const float *sp = (x->ucp_ctl & 1) ? clip :
                              (x->ucp_ctl & 2) ? obj : eye;

            if ((x->ucp & (1u << p)) &&
                pl[0] * sp[0] + pl[1] * sp[1] + pl[2] * sp[2] +
                pl[3] * sp[3] < 0.0f) {
                out |= 1u << p;
            }
        }
        v->tc[0][3] = (float)out;
    }
}

/*
 * Decide whether the draw about to be decoded goes through the transform
 * and prepare the matrix.  Bypass (SE_CNTL_STATUS bit 8: RAVE) and
 * primitives without VF_CNTL.TCL_ENABLE are window-space, as before.
 */
static unsigned cur_op;        /* packet3 opcode being executed */

static void tcl_begin(ATIVGAState *s, uint32_t vfmt, uint32_t vfcntl,
                      bool allow)
{
    ATI3DState *t = s->r3d;
    static bool warned_vf;
    float m[4][4], mv[4][4], p[4][4];
    uint32_t sel0, sel1, tpc, lm, ucp, sig;
    bool composed = false;
    unsigned mvp_slot, mv_slot;

    t->tcl_on = false;
    t->tcl_vtx_set = false;
    if (!allow || (R(s, SE_CNTL_STATUS) & (1 << 8))) {
        return;
    }
    {
        static bool first = true;

        if (first) {
            first = false;
            qemu_log_mask(LOG_UNIMP, "ati: first non-bypass draw: packet3 "
                          "0x%02x vfmt 0x%08x vf_cntl 0x%08x se_cntl_status "
                          "0x%08x -> %s\n", cur_op, vfmt, vfcntl,
                          R(s, SE_CNTL_STATUS),
                          (vfcntl & VFCNTL_TCL_ENABLE) ?
                          "TCL transform" : "window space (TCL_ENABLE clear)");
        }
    }
    if (!(vfcntl & VFCNTL_TCL_ENABLE)) {
        if (!warned_vf) {
            warned_vf = true;
            qemu_log_mask(LOG_UNIMP, "ati: TCL not bypassed but the primitive "
                          "has VF_CNTL.TCL_ENABLE clear (0x%x); treating its "
                          "vertices as window coordinates\n", vfcntl);
        }
        return;
    }
    t->tcl_on = true;
    t->tcl_draws++;

    sel0 = R(s, SE_TCL_MATRIX_SELECT_0);
    sel1 = R(s, SE_TCL_MATRIX_SELECT_1);
    mv_slot = sel0 & 0xf;
    mvp_slot = sel1 & 0xf;
    if (!tcl_matrix(t, mvp_slot * 4, m)) {
        /* no model-view-projection matrix loaded: compose the model-view
         * matrix with the eye-to-clip (projection) matrix */
        tcl_matrix(t, mv_slot * 4, mv);
        tcl_matrix(t, TCL_VS_MATRIX_EYE2CLIP, p);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                m[r][c] = p[r][0] * mv[0][c] + p[r][1] * mv[1][c] +
                          p[r][2] * mv[2][c] + p[r][3] * mv[3][c];
            }
        }
        composed = true;
    }
    memcpy(t->tcl_m, m, sizeof(m));

    tpc = R(s, SE_TCL_TEXTURE_PROC_CTL);
    lm = R(s, SE_TCL_LIGHT_MODEL_CTL);
    ucp = R(s, SE_TCL_UCP_VERT_BLEND_CTL);
    tcl_fx_setup(s, tpc, lm, ucp, sel0, sel1);
    {
        static bool w_blend, w_two;

        if ((ucp & ((1u << 16) | (1u << 17))) && !w_blend) {
            w_blend = true;
            qemu_log_mask(LOG_UNIMP, "ati: TCL vertex blending not emulated "
                          "(0x%08x)\n", ucp);
        }
        if ((lm & 1) && (ucp & (1u << 11)) && !w_two) {
            w_two = true;
            qemu_log_mask(LOG_UNIMP, "ati: TCL two-sided lighting not "
                          "emulated (front side only)\n");
        }
    }

    sig = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            sig = sig * 31 + (uint32_t)tr_i(m[r][c], 4096.0f);
        }
    }
    if (sig != t->tcl_sig || !t->tcl_dumps) {
        /* the matrix changed: trace it and the first vertices drawn with it */
        t->tcl_sig = sig;
        t->tcl_dump_left = 3;
        if (t->tcl_dumps < 64 &&
            trace_event_get_state_backends(TRACE_ATI_3D_TCL_MAT)) {
            t->tcl_dumps++;
            for (int r = 0; r < 4; r++) {
                trace_ati_3d_tcl_mat(r, sel0, sel1, composed,
                                     tr_i(m[r][0], 1000), tr_i(m[r][1], 1000),
                                     tr_i(m[r][2], 1000), tr_i(m[r][3], 1000));
            }
        }
    }
}

static void decode_vertex(ATIVGAState *s, uint32_t f, const uint32_t *d,
                          ATIVertex *v, bool rgba_order)
{
    uint32_t coord = R(s, SE_COORD_FMT), se = R(s, SE_CNTL);
    bool tcl = s->r3d->tcl_on;
    unsigned i = 0;
    float w0 = 1.0f, nrm[3] = { 0, 0, 1 }, tin[3][4];

    memset(v, 0, sizeof(*v));
    v->x = fl(d[i++]);
    v->y = fl(d[i++]);
    v->z = (f & VF_Z) ? fl(d[i++]) : 0.0f;
    if (f & VF_W0) {
        w0 = fl(d[i++]);
    }
    if (tcl && (f & VF_N0)) {
        /* the normal sits between the position and the colours */
        nrm[0] = fl(d[i]); nrm[1] = fl(d[i + 1]); nrm[2] = fl(d[i + 2]);
        i += 3;
    }
    v->c0[0] = v->c0[1] = v->c0[2] = v->c0[3] = 1.0f;
    if (f & VF_FPCOLOR) {
        /* float colours follow VF_CNTL's colour order too: B, G, R unless
         * COLOR_ORDER_RGBA (Mac OS's RAVE engine sends BGR) */
        int r = rgba_order ? 0 : 2, b = 2 - r;
        v->c0[r] = fl(d[i]); v->c0[1] = fl(d[i + 1]); v->c0[b] = fl(d[i + 2]);
        i += 3;
    }
    if (f & VF_FPALPHA) {
        v->c0[3] = fl(d[i++]);
    }
    if (f & VF_PKCOLOR) {
        uint32_t c = d[i++];
        if (rgba_order) {
            c = (c & 0xff00ff00) | ((c & 0xff) << 16) | ((c >> 16) & 0xff);
        }
        argb_to_float(c, v->c0);
    }
    v->c1[3] = 1.0f;
    if (f & VF_FPSPEC) {
        int r = rgba_order ? 0 : 2, b = 2 - r;
        v->c1[r] = fl(d[i]); v->c1[1] = fl(d[i + 1]); v->c1[b] = fl(d[i + 2]);
        i += 3;
    }
    if (f & VF_FPFOG) {
        v->c1[3] = fl(d[i++]);
    }
    if (f & VF_PKSPEC) {
        uint32_t c = d[i++];
        if (rgba_order) {
            c = (c & 0xff00ff00) | ((c & 0xff) << 16) | ((c >> 16) & 0xff);
        }
        argb_to_float(c, v->c1);
    }
    for (int u = 0; u < 4; u++) {
        static const uint32_t stb[4] = { VF_ST0, VF_ST1, VF_ST2, VF_ST3 };
        static const uint32_t qb[4] = { VF_Q0, VF_Q1, VF_Q2, VF_Q3 };
        float st0 = 0, st1 = 0, q = 1.0f;
        if (f & stb[u]) {
            st0 = fl(d[i]);
            st1 = fl(d[i + 1]);
            i += 2;
        }
        if (f & qb[u]) {
            q = fl(d[i++]);
        }
        if (u < 3) {
            v->tc[u][0] = tin[u][0] = st0;
            v->tc[u][1] = tin[u][1] = st1;
            v->tc[u][2] = tin[u][2] = q;
        }
    }
    if (tcl) {
        /*
         * Hardware TCL: the position is object space.  Transform it to clip
         * space, then divide by w like the viewport transform engine does
         * for vertices that are not pre-divided; the sign of w is kept so
         * the renderer can clip against the near plane.
         */
        ATI3DState *t = s->r3d;
        float (*m)[4] = t->tcl_m;
        float ox = v->x, oy = v->y, oz = v->z, c[4], iw;

        for (int r = 0; r < 4; r++) {
            c[r] = m[r][0] * ox + m[r][1] * oy + m[r][2] * oz + m[r][3] * w0;
        }
        if (fabsf(c[3]) < 1e-12f) {
            c[3] = c[3] < 0 ? -1e-12f : 1e-12f;
        }
        iw = 1.0f / c[3];
        v->x = c[0] * iw;
        v->y = c[1] * iw;
        v->z = c[2] * iw;
        v->rhw = iw;
        if (t->tcl_dump_left > 0 && !t->tcl_vtx_set) {
            t->tcl_vtx_set = true;
            t->tcl_vtx[0] = ox; t->tcl_vtx[1] = oy;
            t->tcl_vtx[2] = oz; t->tcl_vtx[3] = w0;
            memcpy(&t->tcl_vtx[4], c, sizeof(c));
        }
        {
            float obj[4] = { ox, oy, oz, w0 };

            tcl_post(t, v, f, obj, nrm, c, tin);
        }
    } else {
        /* W0 is 1/w unless the coordinate format says otherwise */
        if (f & VF_W0) {
            v->rhw = (coord & (1 << 16)) ? (w0 != 0 ? 1.0f / w0 : 1.0f) : w0;
        } else {
            v->rhw = 1.0f;
        }
        /*
         * SE_COORD_FMT: *_PRE_MULT_1_OVER_W0 means the value has already been
         * divided by w (projected).  The shader wants projected X/Y/Z and the
         * true texture coordinates (GL interpolates them with w), so
         * pre-divided texture coordinates are multiplied back by w, and X/Y/Z
         * that are not pre-divided are divided here.  Quake III through ATI's
         * OpenGL renderer: X/Y/Z projected, S/T divided by w, W0 = 1/w.
         */
        if (!(coord & 1) && v->rhw) {
            v->x *= v->rhw;
            v->y *= v->rhw;
        }
        if (!(coord & 2) && (f & VF_Z) && v->rhw) {
            v->z *= v->rhw;
        }
    }
    for (int u = 0; u < 3; u++) {
        static const uint32_t qbit[3] = { VF_Q0, VF_Q1, VF_Q2 };
        if (coord & (1u << (17 + 2 * u)) && v->rhw) {
            v->tc[u][0] /= v->rhw;
            v->tc[u][1] /= v->rhw;
            if (f & qbit[u]) {      /* a q sent along was divided as well */
                v->tc[u][2] /= v->rhw;
            }
        }
    }
    /* viewport transform when enabled, after the division by w */
    if (se & (1 << 24)) {
        v->x = v->x * RF(s, SE_VPORT_XSCALE) + RF(s, SE_VPORT_XOFFSET);
        v->y = v->y * RF(s, SE_VPORT_YSCALE) + RF(s, SE_VPORT_YOFFSET);
    }
    if (se & (1 << 25)) {
        v->z = v->z * RF(s, SE_VPORT_ZSCALE) + RF(s, SE_VPORT_ZOFFSET);
    }
    if (se & (1 << 27)) {       /* OpenGL pixel centres */
        /* nothing: GL already samples at pixel centres */
    }
}

/* GL primitive for an R100 prim type, possibly after expansion */
static unsigned expand_prim(unsigned type, const ATIVertex *in, unsigned n,
                            ATIVertex **out, uint16_t *glprim)
{
    ATIVertex *o;
    unsigned k = 0;

    switch (type) {
    case 1: *glprim = 0x0000; *out = (ATIVertex *)in; return n;  /* points */
    case 2: *glprim = 0x0001; *out = (ATIVertex *)in; return n;  /* lines */
    case 3: *glprim = 0x0003; *out = (ATIVertex *)in; return n;  /* strip */
    case 4: *glprim = 0x0004; *out = (ATIVertex *)in; return n - n % 3;
    case 5: *glprim = 0x0006; *out = (ATIVertex *)in; return n;  /* fan */
    case 6: *glprim = 0x0005; *out = (ATIVertex *)in; return n;  /* strip */
    case 12: *glprim = 0x0002; *out = (ATIVertex *)in; return n; /* loop */
    case 15: *glprim = 0x0006; *out = (ATIVertex *)in; return n; /* polygon */
    case 8:     /* rect list: v0 v1 v2 give three corners, v3 = v1+v2-v0 */
        o = g_new(ATIVertex, n / 3 * 6);
        for (unsigned i = 0; i + 2 < n; i += 3) {
            ATIVertex v3 = in[i + 2];
            v3.x = in[i + 1].x + in[i + 2].x - in[i].x;
            v3.y = in[i + 1].y + in[i + 2].y - in[i].y;
            for (int c = 0; c < 3; c++) {
                v3.tc[c][0] = in[i + 1].tc[c][0] + in[i + 2].tc[c][0] -
                              in[i].tc[c][0];
                v3.tc[c][1] = in[i + 1].tc[c][1] + in[i + 2].tc[c][1] -
                              in[i].tc[c][1];
            }
            o[k++] = in[i]; o[k++] = in[i + 1]; o[k++] = in[i + 2];
            o[k++] = in[i + 1]; o[k++] = v3; o[k++] = in[i + 2];
        }
        *glprim = 0x0004;
        *out = o;
        return k;
    case 13:    /* quad list */
        o = g_new(ATIVertex, n / 4 * 6);
        for (unsigned i = 0; i + 3 < n; i += 4) {
            o[k++] = in[i]; o[k++] = in[i + 1]; o[k++] = in[i + 2];
            o[k++] = in[i]; o[k++] = in[i + 2]; o[k++] = in[i + 3];
        }
        *glprim = 0x0004;
        *out = o;
        return k;
    case 14:    /* quad strip */
        *glprim = 0x0005;
        *out = (ATIVertex *)in;
        return n;
    default:
        qemu_log_mask(LOG_UNIMP, "ati: 3D primitive type %u\n", type);
        *out = (ATIVertex *)in;
        *glprim = 0x0004;
        return 0;
    }
}

/*
 * User clip planes (T4): tcl_post() left an outcode per vertex in tc[0][3].
 * Triangles and quads wholly outside one plane are dropped; primitives that
 * only cross a plane are drawn unclipped (and other primitive types are not
 * culled).  The outcodes are cleared again: the shader reads tc[0][3].
 */
static unsigned tcl_ucp_cull(unsigned type, ATIVertex *v, unsigned n)
{
    unsigned k = 0, g = type == 4 ? 3 : type == 13 ? 4 : 0;

    if (g) {
        for (unsigned i = 0; i + g <= n; i += g) {
            unsigned all = ~0u;

            for (unsigned j = 0; j < g; j++) {
                all &= (unsigned)v[i + j].tc[0][3];
            }
            if (!all) {
                for (unsigned j = 0; j < g; j++) {
                    v[k++] = v[i + j];
                }
            }
        }
        n = k;
    }
    for (unsigned i = 0; i < n; i++) {
        v[i].tc[0][3] = 0.0f;
    }
    return n;
}

static void draw(ATIVGAState *s, uint32_t vfmt, uint32_t vfcntl,
                 ATIVertex *v, unsigned n)
{
    ATI3DState *t = s->r3d;
    ATIDrawState st;
    ATIVertex *ev;
    uint16_t glprim;
    unsigned en;

    if (!t->r || !n) {
        return;
    }
    if (!decode_state(s, &st)) {
        t->tcl_on = false;
        return;
    }
    st.tcl = t->tcl_on;
    if (t->tcl_on && t->tcl_dump_left > 0) {
        if (t->tcl_vtx_set &&
            trace_event_get_state_backends(TRACE_ATI_3D_TCL_VTX)) {
            const float *f = t->tcl_vtx;

            trace_ati_3d_tcl_vtx(vfmt, vfcntl, tr_i(f[0], 1000),
                                 tr_i(f[1], 1000), tr_i(f[2], 1000),
                                 tr_i(f[3], 1000), tr_i(f[4], 1000),
                                 tr_i(f[5], 1000), tr_i(f[6], 1000),
                                 tr_i(f[7], 1000));
            trace_ati_3d_tcl_win(tr_i(v[0].x, 1000), tr_i(v[0].y, 1000),
                                 tr_i(v[0].z, 1e6f), tr_i(v[0].rhw, 1e6f),
                                 R(s, SE_CNTL), R(s, SE_COORD_FMT),
                                 tr_i(RF(s, SE_VPORT_XSCALE), 1000),
                                 tr_i(RF(s, SE_VPORT_XOFFSET), 1000),
                                 tr_i(RF(s, SE_VPORT_YSCALE), 1000),
                                 tr_i(RF(s, SE_VPORT_YOFFSET), 1000));
        }
        t->tcl_dump_left--;
    }
    t->tcl_vtx_set = false;
    if (trace_event_get_state_backends(TRACE_ATI_3D_TARGET)) {
        /* the colour target's raw state, whenever it changes */
        static uint32_t last[9];
        uint32_t cur[9] = { R(s, RB3D_COLORPITCH), R(s, RB3D_CNTL),
                            R(s, RE_WIDTH_HEIGHT), R(s, RE_TOP_LEFT),
                            R(s, SE_CNTL), R(s, SE_CNTL_STATUS),
                            st.color.w, st.color.h, st.color.pitch };

        if (memcmp(last, cur, sizeof(cur))) {
            memcpy(last, cur, sizeof(cur));
            trace_ati_3d_target(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5],
                                cur[6], cur[7], cur[8], st.color.fmt);
        }
    }
    if (t->tcl_on && t->tx.ucp) {
        n = tcl_ucp_cull(vfcntl & 15, v, n);
        if (!n) {
            return;
        }
    }
    en = expand_prim(vfcntl & 15, v, n, &ev, &glprim);
    trace_ati_3d_draw(st.color.mc, st.depth_mc, st.z_test, st.z_func, st.z_write,
                      st.blend, st.blend_src, st.blend_dst, vfcntl & 15, n);
    if (trace_event_get_state_backends(TRACE_ATI_3D_DRAW_VTX)) {
        float zmin = 1e9, zmax = -1e9, wmin = 1e9, wmax = -1e9;
        for (unsigned i = 0; i < n; i++) {
            zmin = MIN(zmin, v[i].z); zmax = MAX(zmax, v[i].z);
            wmin = MIN(wmin, v[i].rhw); wmax = MAX(wmax, v[i].rhw);
        }
        trace_ati_3d_draw_vtx((int)(zmin * 1e6), (int)(wmin * 1e6),
                              (int)v[0].x, (int)v[0].y, vfmt, R(s, PP_CNTL),
                              (int)(v[0].tc[0][0] * 1e3), (int)(v[0].tc[0][1] * 1e3),
                              (int)(v[0].tc[1][0] * 1e3), (int)(v[0].tc[1][1] * 1e3));
        (void)zmax; (void)wmax;
    }
    if (en) {
        track_acquire(s, &st.color)->gen++;
        ati_glr_draw(t->r, &st, glprim, ev, en);
        t->draws++;
        t->verts += en;
    }
    if (ev != v) {
        g_free(ev);
    }
}

static ATIVertex *vbuf(ATI3DState *t, unsigned n)
{
    if (n > t->vbuf_cap) {
        t->vbuf_cap = MAX(n, 1024);
        t->vbuf = g_renew(ATIVertex, t->vbuf, t->vbuf_cap);
    }
    return t->vbuf;
}

static void draw_immd(ATIVGAState *s, const uint32_t *d, unsigned n,
                      bool tcl_ok)
{
    ATI3DState *t = s->r3d;
    uint32_t vfmt, vf;
    unsigned vsz, nv;
    ATIVertex *v;

    if (n < 2) {
        return;
    }
    vfmt = d[0];
    vf = d[1];
    vsz = vtx_dwords(vfmt);
    nv = MIN(vf >> 16, (n - 2) / vsz);
    tcl_begin(s, vfmt, vf, tcl_ok);
    v = vbuf(t, nv);
    for (unsigned i = 0; i < nv; i++) {
        decode_vertex(s, vfmt, d + 2 + i * vsz, &v[i], vf & (1 << 6));
    }
    draw(s, vfmt, vf, v, nv);
}

static void draw_vbuf(ATIVGAState *s, uint32_t vfmt, uint32_t vf,
                      const uint16_t *idx, unsigned nidx)
{
    ATI3DState *t = s->r3d;
    unsigned vsz = vtx_dwords(vfmt);
    unsigned nv = idx ? nidx : (vf >> 16);
    uint32_t stride, base;
    bool multi;
    ATIVertex *v;
    uint32_t tmp[64];

    if (!t->narrays || vsz > ARRAY_SIZE(tmp)) {
        return;
    }
    base = t->arr_addr[0];
    stride = t->arr_stride[0] ? t->arr_stride[0] : vsz;
    /*
     * With hardware TCL a driver (Mesa's r100 does) loads one array per
     * vertex component (position, normal, colour, ...).  Each array's word
     * in 3D_LOAD_VBPNTR holds its dword count in bits 7:0 (arr_stride here)
     * and its stride in bits 14:8 (arr_size); a stride of 0 is a constant.
     * When the counts add up to the vertex format's size the vertex is
     * gathered from the arrays; otherwise array 0 is read as interleaved.
     */
    {
        unsigned tot = 0;

        for (unsigned k = 0; k < t->narrays; k++) {
            tot += t->arr_stride[k];
        }
        multi = t->narrays > 1 && tot == vsz;
    }
    tcl_begin(s, vfmt, vf, true);
    v = vbuf(t, nv);
    for (unsigned i = 0; i < nv; i++) {
        unsigned vi = idx ? idx[i] : i;
        if (multi) {
            unsigned o = 0;
            for (unsigned k = 0; k < t->narrays; k++) {
                unsigned comps = t->arr_stride[k];
                unsigned step = t->arr_size[k];
                ati_mc_rw(s, t->arr_addr[k] + (uint64_t)(step ? vi : 0) * step * 4,
                          tmp + o, comps * 4, false);
                o += comps;
            }
        } else {
            ati_mc_rw(s, base + (uint64_t)vi * stride * 4, tmp, vsz * 4, false);
        }
        /* vertex arrays are fetched with the CP's swap setting */
        for (unsigned k = 0; k < vsz; k++) {
            tmp[k] = (R(s, SE_CNTL_STATUS) & 3) == 2 ? ldl_be_p(&tmp[k])
                                                     : ldl_le_p(&tmp[k]);
        }
        decode_vertex(s, vfmt, tmp, &v[i], vf & (1 << 6));
    }
    draw(s, vfmt, vf, v, nv);
}

/*
 * Vertices packed in a buffer at an MC address (GART, AGP or VRAM), read
 * in one piece and decoded by index: 3D_RNDR_GEN_INDX_PRIM.
 *
 * Only the vertices between the lowest and the highest index are fetched.
 * ATI's OpenGL renderer keeps many draws' vertices in one buffer and points
 * each draw's indices into the middle of it: fetching from the buffer's start
 * read and byte-swapped about 16 times the vertices drawn (Quake III), all on
 * the vCPU thread.  When the range holds no more vertices than there are
 * indices (a mesh that reuses its vertices), each vertex is decoded once and
 * copied per index; otherwise each index is decoded as before.  The vertices
 * handed to the renderer are the same either way.
 */
static void draw_membuf(ATIVGAState *s, uint32_t base, uint32_t vfmt,
                        uint32_t vf, const uint16_t *idx, unsigned n)
{
    ATI3DState *t = s->r3d;
    unsigned vsz = vtx_dwords(vfmt), lo = 0, hi = n, nbuf;
    bool be = (R(s, SE_CNTL_STATUS) & 3) == 2;
    bool rgba = vf & (1 << 6);
    size_t ndw;
    uint32_t *buf;
    ATIVertex *v;

    if (!n) {
        return;
    }
    if (idx) {
        lo = 0xffff;
        hi = 0;
        for (unsigned i = 0; i < n; i++) {
            lo = MIN(lo, (unsigned)idx[i]);
            hi = MAX(hi, (unsigned)idx[i] + 1);
        }
    }
    nbuf = hi - lo;
    if (nbuf > 65536) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: vertex buffer of %u vertices\n", nbuf);
        return;
    }
    t->vb_fetch += nbuf;
    t->vb_idx += n;
    ndw = (size_t)nbuf * vsz;
    if (ndw > t->mbuf_cap) {
        t->mbuf_cap = MAX(ndw, 4096);
        t->mbuf = g_renew(uint32_t, t->mbuf, t->mbuf_cap);
    }
    buf = t->mbuf;
    ati_mc_rw(s, base + (uint64_t)lo * vsz * 4, buf, (uint64_t)ndw * 4, false);
    for (size_t k = 0; k < ndw; k++) {
        buf[k] = be ? ldl_be_p(&buf[k]) : ldl_le_p(&buf[k]);
    }
    tcl_begin(s, vfmt, vf, true);
    v = vbuf(t, n);
    if (idx && nbuf <= n) {
        if (nbuf > t->dec_cap) {
            t->dec_cap = MAX(nbuf, 1024);
            t->dec = g_renew(ATIVertex, t->dec, t->dec_cap);
        }
        for (unsigned k = 0; k < nbuf; k++) {
            decode_vertex(s, vfmt, buf + (size_t)k * vsz, &t->dec[k], rgba);
        }
        for (unsigned i = 0; i < n; i++) {
            v[i] = t->dec[idx[i] - lo];
        }
    } else {
        for (unsigned i = 0; i < n; i++) {
            decode_vertex(s, vfmt, buf + (size_t)(idx ? idx[i] - lo : i) * vsz,
                          &v[i], rgba);
        }
    }
    draw(s, vfmt, vf, v, n);
}

/*
 * 3D_CLEAR_ZMASK (first tile, tile count, clear mask): the fast Z clear.
 * Depth lives in the host renderer, so the rows covered by the tiles are
 * set to RB3D_DEPTHCLEARVALUE.  Mac OS's driver clears 2400 tiles for
 * 640x480 with 16-bit Z, i.e. 128 pixels per tile in pixel order; the
 * rows touched are cleared across the whole width.
 */
#define ZMASK_TILE_PIXELS 128

static void clear_zmask(ATIVGAState *s, const uint32_t *d, unsigned n)
{
    ATI3DState *t = s->r3d;
    ATIDrawState st;
    uint32_t zs = R(s, RB3D_ZSTENCILCNTL), cv = R(s, RB3D_DEPTHCLEARVALUE);
    uint64_t p0, p1;
    unsigned y0, y1;
    float z;

    if (n < 2 || !decode_targets(s, &st) || !st.color.w) {
        return;
    }
    if (d[1] == 0) {
        /* ATI's OpenGL driver (Quake III, every glClear of depth) sends
         * first tile 0, tile count 0: the whole surface */
        y0 = 0;
        y1 = MIN(st.color.h, 2048);
    } else {
        p0 = (uint64_t)d[0] * ZMASK_TILE_PIXELS;
        p1 = (uint64_t)(d[0] + d[1]) * ZMASK_TILE_PIXELS;
        y0 = MIN(p0 / st.color.w, 2048);
        y1 = MIN(DIV_ROUND_UP(p1, st.color.w), 2048);
    }
    if (y1 <= y0) {
        return;
    }
    z = (zs & 15) == 0 ? (cv & 0xffff) / 65535.0f : (cv & 0xffffff) / 16777215.0f;
    track_acquire(s, &st.color);
    trace_ati_3d_zclear(st.color.mc, R(s, RB3D_DEPTHOFFSET) & ~15u, st.color.w,
                        y1 - y0, cv, zs);
    ati_glr_depth_clear(t->r, &st.color, R(s, RB3D_DEPTHOFFSET) & ~15u, z,
                        (zs & 15) == 0 ? -1 : (int)(cv >> 24),
                        0, y0, st.color.w, y1 - y0);
    t->zclears++;
}

void ati_3d_packet(ATIVGAState *s, uint8_t op, const uint32_t *d, unsigned n)
{
    ATI3DState *t = s->r3d;

    cur_op = op;
    trace_ati_3d_packet(op, n, n > 0 ? d[0] : 0, n > 1 ? d[1] : 0);
    if (!t || !t->r) {
        return;
    }
    switch (op) {
    case 0x25:  /* used by ATI's Mac OS RAVE engine like 3D_DRAW_IMMD (same
                 * vertex format and VF_CNTL words, vertices inline); RAVE sets
                 * the TCL bypass bit, which tcl_begin() honours */
        draw_immd(s, d, n, true);
        break;
    case 0x29:  /* 3D_DRAW_IMMD */
        draw_immd(s, d, n, true);
        break;
    case 0x2f:  /* 3D_LOAD_VBPNTR */
    {
        unsigned c = d[0] & 0x1f, k = 1;
        t->narrays = MIN(c, 16u);
        for (unsigned i = 0; i < t->narrays && k < n; i += 2) {
            uint32_t sz = d[k++];
            t->arr_size[i] = (sz >> 8) & 0x7f;
            t->arr_stride[i] = sz & 0xff;
            if (k < n) {
                t->arr_addr[i] = d[k++];
            }
            if (i + 1 < t->narrays) {
                t->arr_size[i + 1] = (sz >> 24) & 0x7f;
                t->arr_stride[i + 1] = (sz >> 16) & 0xff;
                if (k < n) {
                    t->arr_addr[i + 1] = d[k++];
                }
            }
        }
        break;
    }
    case 0x28:  /* 3D_DRAW_VBUF */
        if (n >= 2) {
            draw_vbuf(s, d[0], d[1], NULL, 0);
        }
        break;
    case 0x2a:  /* 3D_DRAW_INDX: 16-bit indices packed in the packet */
        if (n >= 2) {
            unsigned ni = MIN(d[1] >> 16, (n - 2) * 2);
            uint16_t *ix = g_new(uint16_t, ni);
            for (unsigned i = 0; i < ni; i++) {
                uint32_t w = d[2 + i / 2];
                ix[i] = (i & 1) ? w >> 16 : w & 0xffff;
            }
            draw_vbuf(s, d[0], d[1], ix, ni);
            g_free(ix);
        }
        break;
    case 0x23:  /* 3D_RNDR_GEN_INDX_PRIM: buffer address, vertex count or
                 * max index, vertex format, VF_CNTL, [16-bit indices] */
        if (n >= 4) {
            uint32_t vf = d[3];
            if (((vf >> 4) & 3) == 1) {         /* WALK_IND */
                unsigned ni = MIN(vf >> 16, (n - 4) * 2);
                uint16_t *ix = g_new(uint16_t, ni ? ni : 1);
                for (unsigned i = 0; i < ni; i++) {
                    uint32_t w = d[4 + i / 2];
                    ix[i] = (i & 1) ? w >> 16 : w & 0xffff;
                }
                draw_membuf(s, d[0], d[2], vf, ix, ni);
                g_free(ix);
            } else {                            /* WALK_LIST */
                draw_membuf(s, d[0], d[2], vf, NULL, vf >> 16);
            }
        }
        break;
    case 0x32:  /* 3D_CLEAR_ZMASK: first tile, tile count, clear mask */
        clear_zmask(s, d, n);
        break;
    default:
        qemu_log_mask(LOG_UNIMP, "ati: 3D packet 0x%02x not emulated\n", op);
        break;
    }
}

const char *ati_3d_stats(ATIVGAState *s)
{
    ATI3DState *t = s->r3d;

    snprintf(t->stats, sizeof(t->stats), "draws3d=%" PRIu64 " verts3d=%" PRIu64
             " readbacks=%" PRIu64 " target_uploads=%" PRIu64
             " tex_uploads=%" PRIu64 " tex_clean_hits=%" PRIu64
             " zclears=%" PRIu64 " upload_skips=%" PRIu64
             " rb_2d=%" PRIu64 " rb_status=%" PRIu64 " rb_scanout=%" PRIu64
             " flips=%" PRIu64 " joined=%" PRIu64 " flip_waits=%" PRIu64
             " vb_fetch=%" PRIu64 " vb_idx=%" PRIu64 " gl_fills=%" PRIu64
             " tcl_draws=%" PRIu64 " gl=%s",
             t->draws, t->verts, t->readbacks, t->uploads, t->texuploads,
             t->texhits, t->zclears, t->upload_skips, t->rb_why[ATI_SYNC_2D],
             t->rb_why[ATI_SYNC_STATUS], t->rb_why[ATI_SYNC_VBLANK], t->flips,
             t->readbacks_joined, t->vblank_waits, t->vb_fetch, t->vb_idx,
             t->gl_fills, t->tcl_draws,
             t->r ? "on" : "off");
    return t->stats;
}

void ati_3d_reg_write(ATIVGAState *s, hwaddr addr, uint32_t data)
{
    /* state lives in the register file; draws decode it there, except for
     * the TCL vector/scalar memories behind their index/data ports */
    if (s->r3d && addr >= SE_TCL_VECTOR_INDX_REG &&
        addr <= SE_TCL_SCALAR_DATA_REG) {
        tcl_reg_write(s->r3d, addr, data);
    }
}

/* ClassicMac: the host renderer for status ("" while 3D is off) */
const char *ati_3d_gl_info(ATIVGAState *s)
{
    return s->r3d && s->r3d->r ? s->r3d->gl_info : "";
}

void ati_3d_init(ATIVGAState *s)
{
    ATI3DState *t = g_new0(ATI3DState, 1);
    char info[256] = "";
    /* ClassicMac's switch, and RetroBoot's, which is kept */
    static const char *const knobs[] = { "CLASSICMAC_ATI_3D", "RETROBOOT_ATI_3D" };

    s->r3d = t;
    for (size_t i = 0; i < ARRAY_SIZE(knobs); i++) {
        const char *env = getenv(knobs[i]);
        if (env && !strcmp(env, "off")) {
            info_report("ati: 3D engine disabled (%s=off)", knobs[i]);
            return;
        }
    }
    t->r = ati_glr_new(info, sizeof(info));
    if (t->r) {
        snprintf(t->gl_info, sizeof(t->gl_info), "%s", info);
        info_report("ati: 3D engine on host GL: %s", info);
    } else {
        warn_report("ati: no host GL (%s); 3D drawing is discarded", info);
    }
}
