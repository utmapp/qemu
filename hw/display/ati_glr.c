/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: OpenGL ES 3.0 / OpenGL 4.1 renderer thread.
 *
 * Copyright (c) 2026 RetroBoot contributors
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 *
 * All GL calls happen on one thread that owns the context.  The device
 * (vCPU thread) queues commands; reads wait for completion.  Guest memory is
 * never touched here: uploads arrive as copies and readbacks are returned to
 * the caller, which writes guest memory itself.
 */

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/atomic.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include "qemu/main-loop.h"
#include "hw/core/cpu.h"
#include "ati_gl.h"
#include "ati_3d.h"

#define MAX_TARGETS 32
#define MAX_TEXTURES 1024
#define TARGET_H 2048

typedef struct GLRTarget {
    bool used;
    uint32_t mc, pitch;
    uint8_t fmt;
    uint16_t w, h;
    GLuint tex, fbo;
    uint32_t depth_mc;         /* depth buffer attached, 0 = none yet */
    GLuint depth_rb;
    uint64_t lru;
} GLRTarget;

typedef struct GLRTexture {
    bool used;
    uint64_t key;
    uint32_t gen;
    GLuint tex;
    uint16_t w, h;
    uint64_t lru;
    /* sampler state last applied */
    uint16_t min, mag, ws, wt;
} GLRTexture;

enum { CMD_DRAW, CMD_TEX, CMD_UPLOAD, CMD_READ, CMD_CLEARZ, CMD_FINISH, CMD_READ_ASYNC,
       CMD_FILL };

typedef struct GLRCmd {
    int kind;
    struct GLRCmd *next;
    ATIDrawState st;
    uint16_t prim;
    unsigned nv;
    ATIVertex *v;
    ATIImage img;
    ATITexUnit tu;
    uint64_t key;
    uint32_t gen;
    uint8_t *data;
    size_t len;
    float z;
    int stencil, x, y, w, h;
    uint32_t depth_mc;
    QemuEvent *done;
    ATIReadJob *job;
    uint64_t hash;              /* CMD_READ: ati_vram_hash of the result */
    uint8_t px[4];              /* CMD_FILL: the pixel, guest bytes */
} GLRCmd;

struct ATIRenderer {
    QemuThread thread;
    QemuMutex lock;
    QemuCond cond;
    GLRCmd *head, *tail;
    bool ok;
    char info[256];
    QemuEvent ready;
    bool waiting;               /* vCPU waits with the BQL released (BQL) */

    /* render thread only */
    uint8_t *scratch;           /* RGBA8 rows for uploads and readbacks */
    size_t scratch_len;
    ATIGL gl;
    GLuint prog, vao, vbo;
    GLint u_fbsize, u_stage_c, u_stage_a, u_tfactor, u_stage_en, u_tex_en,
          u_tex[3], u_alpha, u_alpharef, u_fog, u_fogcol, u_flags, u_solid;
    GLRTarget targets[MAX_TARGETS];
    GLRTexture texs[MAX_TEXTURES];
    uint64_t tick;
    GLRTarget *bound;
    uint32_t bound_depth;
};

/* ---------------------------------------------------------------------
 * Pixel formats.  Guest bytes are first permuted by the swap code
 * (group: 0 none, 1 swap bytes in 16-bit units, 2 swap 16-bit halves,
 * 3 = 1 then 2 = full 32-bit swap) and then read little-endian, as the chip
 * would after its own byte swapper.
 */

unsigned ati_fmt_bpp(uint8_t fmt)
{
    switch (fmt) {
    case ATI_FMT_I8: case ATI_FMT_RGB332: case ATI_FMT_Y8: case ATI_FMT_AI44:
        return 1;
    case ATI_FMT_ARGB1555: case ATI_FMT_RGB565: case ATI_FMT_ARGB4444:
    case ATI_FMT_AI88: case ATI_FMT_VYUY422: case ATI_FMT_YVYU422:
        return 2;
    case ATI_FMT_ARGB8888: case ATI_FMT_RGBA8888:
        return 4;
    default:
        return 0;
    }
}

static inline uint32_t perm_byte_index(unsigned i, uint8_t code)
{
    /* byte i of the little-endian view comes from byte perm(i) in memory */
    if (code & 1) {
        i ^= 1;
    }
    if (code & 2) {
        i ^= 2;
    }
    return i;
}

static inline uint8_t ex5(uint32_t v) { return (v << 3) | (v >> 2); }
static inline uint8_t ex6(uint32_t v) { return (v << 2) | (v >> 4); }
static inline uint8_t ex4(uint32_t v) { return (v << 4) | v; }

static uint8_t clamp255(int v)
{
    return v < 0 ? 0 : v > 255 ? 255 : v;
}

static void yuv_to_rgba(int y, int u, int v, uint8_t *o)
{
    int c = y - 16, d = u - 128, e = v - 128;
    o[0] = clamp255((298 * c + 409 * e + 128) >> 8);
    o[1] = clamp255((298 * c - 100 * d - 208 * e + 128) >> 8);
    o[2] = clamp255((298 * c + 516 * d + 128) >> 8);
    o[3] = 255;
}

/*
 * RetroBoot: fast paths for the per-frame conversions.  A game at 1920x1080
 * converts every frame twice (the cleared back buffer up, the finished frame
 * back), 2 M pixels each way; byte by byte that was most of a frame's time.
 *
 * With the row on a 32-bit boundary, byte b of pixel x's little-endian value
 * is at (bpp * x + b) ^ swap (perm_byte_index), so a 16-bit pixel is a 16-bit
 * load at (2x ^ (swap & 2)), big-endian when swap & 1, and a 32-bit pixel is a
 * little- (swap 0) or big-endian (swap 3) load.  Results equal the generic
 * loops below, which keep every other case.
 */
static uint32_t *dec16_lut[3];      /* render thread only */

static int dec16_index(uint8_t fmt)
{
    switch (fmt) {
    case ATI_FMT_ARGB1555: return 0;
    case ATI_FMT_RGB565: return 1;
    case ATI_FMT_ARGB4444: return 2;
    default: return -1;
    }
}

/* RGBA8 bytes of every 16-bit value, in memory order */
static const uint32_t *dec16_table(int i)
{
    if (!dec16_lut[i]) {
        uint32_t *t = g_new(uint32_t, 65536);
        for (uint32_t v = 0; v < 65536; v++) {
            uint8_t o[4];
            switch (i) {
            case 0:
                o[0] = ex5((v >> 10) & 31); o[1] = ex5((v >> 5) & 31);
                o[2] = ex5(v & 31); o[3] = (v & 0x8000) ? 255 : 0;
                break;
            case 1:
                o[0] = ex5((v >> 11) & 31); o[1] = ex6((v >> 5) & 63);
                o[2] = ex5(v & 31); o[3] = 255;
                break;
            default:
                o[0] = ex4((v >> 8) & 15); o[1] = ex4((v >> 4) & 15);
                o[2] = ex4(v & 15); o[3] = ex4(v >> 12);
                break;
            }
            memcpy(&t[v], o, 4);
        }
        dec16_lut[i] = t;
    }
    return dec16_lut[i];
}

static bool decode_row_fast(const uint8_t *src, uint8_t *dst, unsigned w,
                            uint8_t fmt, uint8_t swap, uint32_t row_base)
{
    int li = dec16_index(fmt);

    if (row_base & 3) {
        return false;
    }
    if (li >= 0) {
        const uint32_t *lut = dec16_table(li);
        unsigned xo = swap & 2;
        if (swap & 1) {
            for (unsigned x = 0; x < w; x++) {
                memcpy(dst + 4 * x, &lut[lduw_be_p(src + ((2 * x) ^ xo))], 4);
            }
        } else {
            for (unsigned x = 0; x < w; x++) {
                memcpy(dst + 4 * x, &lut[lduw_le_p(src + ((2 * x) ^ xo))], 4);
            }
        }
        return true;
    }
    if ((fmt == ATI_FMT_ARGB8888 || fmt == ATI_FMT_RGBA8888) &&
        (swap == 0 || swap == 3)) {
        for (unsigned x = 0; x < w; x++) {
            uint32_t v = swap ? ldl_be_p(src + 4 * x) : ldl_le_p(src + 4 * x);
            uint8_t *o = dst + 4 * x;
            if (fmt == ATI_FMT_ARGB8888) {
                o[0] = v >> 16; o[1] = v >> 8; o[2] = v; o[3] = v >> 24;
            } else {
                o[0] = v >> 24; o[1] = v >> 16; o[2] = v >> 8; o[3] = v;
            }
        }
        return true;
    }
    return false;
}

static inline void enc16_store(uint8_t *dst, unsigned x, unsigned xo, bool be,
                               uint32_t v)
{
    uint8_t *p = dst + ((2 * x) ^ xo);
    if (be) {
        stw_be_p(p, v);
    } else {
        stw_le_p(p, v);
    }
}

static bool encode_row_fast(const uint8_t *src, uint8_t *dst, unsigned w,
                            uint8_t fmt, uint8_t swap, uint32_t row_base)
{
    unsigned xo = swap & 2;
    bool be = swap & 1;

    if (row_base & 3) {
        return false;
    }
    switch (fmt) {
    case ATI_FMT_ARGB1555:
        for (unsigned x = 0; x < w; x++) {
            const uint8_t *p = src + 4 * x;
            enc16_store(dst, x, xo, be, ((p[3] >= 128) << 15) |
                        ((p[0] >> 3) << 10) | ((p[1] >> 3) << 5) | (p[2] >> 3));
        }
        return true;
    case ATI_FMT_RGB565:
        for (unsigned x = 0; x < w; x++) {
            const uint8_t *p = src + 4 * x;
            enc16_store(dst, x, xo, be, ((p[0] >> 3) << 11) |
                        ((p[1] >> 2) << 5) | (p[2] >> 3));
        }
        return true;
    case ATI_FMT_ARGB4444:
        for (unsigned x = 0; x < w; x++) {
            const uint8_t *p = src + 4 * x;
            enc16_store(dst, x, xo, be, ((p[3] >> 4) << 12) |
                        ((p[0] >> 4) << 8) | ((p[1] >> 4) << 4) | (p[2] >> 4));
        }
        return true;
    case ATI_FMT_ARGB8888:
    case ATI_FMT_RGBA8888:
        if (swap == 1 || swap == 2) {
            return false;
        }
        for (unsigned x = 0; x < w; x++) {
            const uint8_t *p = src + 4 * x;
            uint32_t v = fmt == ATI_FMT_ARGB8888 ?
                ((uint32_t)p[3] << 24) | (p[0] << 16) | (p[1] << 8) | p[2] :
                ((uint32_t)p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
            if (swap) {
                stl_be_p(dst + 4 * x, v);
            } else {
                stl_le_p(dst + 4 * x, v);
            }
        }
        return true;
    default:
        return false;
    }
}

/* decode one row of w guest pixels into RGBA8 */
static void decode_row(const uint8_t *src, uint8_t *dst, unsigned w,
                       uint8_t fmt, uint8_t swap, uint32_t row_base)
{
    unsigned bpp = ati_fmt_bpp(fmt);

    if (decode_row_fast(src, dst, w, fmt, swap, row_base)) {
        return;
    }
    for (unsigned x = 0; x < w; x++) {
        uint32_t off = x * bpp, v = 0;
        for (unsigned b = 0; b < bpp; b++) {
            uint32_t a = row_base + off + b;
            uint32_t pa = (a & ~3u) | perm_byte_index(a & 3, swap);
            v |= (uint32_t)src[pa - row_base] << (8 * b);
        }
        uint8_t *o = dst + 4 * x;
        switch (fmt) {
        case ATI_FMT_ARGB1555:
            o[0] = ex5((v >> 10) & 31); o[1] = ex5((v >> 5) & 31);
            o[2] = ex5(v & 31); o[3] = (v & 0x8000) ? 255 : 0;
            break;
        case ATI_FMT_RGB565:
            o[0] = ex5((v >> 11) & 31); o[1] = ex6((v >> 5) & 63);
            o[2] = ex5(v & 31); o[3] = 255;
            break;
        case ATI_FMT_ARGB4444:
            o[0] = ex4((v >> 8) & 15); o[1] = ex4((v >> 4) & 15);
            o[2] = ex4(v & 15); o[3] = ex4(v >> 12);
            break;
        case ATI_FMT_ARGB8888:
            o[0] = v >> 16; o[1] = v >> 8; o[2] = v; o[3] = v >> 24;
            break;
        case ATI_FMT_RGBA8888:
            o[0] = v >> 24; o[1] = v >> 16; o[2] = v >> 8; o[3] = v;
            break;
        case ATI_FMT_RGB332:
            o[0] = ((v >> 5) & 7) * 255 / 7; o[1] = ((v >> 2) & 7) * 255 / 7;
            o[2] = (v & 3) * 85; o[3] = 255;
            break;
        case ATI_FMT_I8:
            o[0] = o[1] = o[2] = o[3] = v;
            break;
        case ATI_FMT_Y8:
            o[0] = o[1] = o[2] = v; o[3] = 255;
            break;
        case ATI_FMT_AI88:
            o[0] = o[1] = o[2] = v & 0xff; o[3] = v >> 8;
            break;
        case ATI_FMT_AI44:
            o[0] = o[1] = o[2] = ex4(v & 15); o[3] = ex4(v >> 4);
            break;
        case ATI_FMT_VYUY422:
        case ATI_FMT_YVYU422:
        {
            /* pairs of pixels share chroma; this pixel's byte 0/1 */
            uint32_t pair = x & ~1u, w2 = 0;
            for (unsigned b = 0; b < 4; b++) {
                uint32_t a = row_base + pair * 2 + b;
                uint32_t pa = (a & ~3u) | perm_byte_index(a & 3, swap);
                w2 |= (uint32_t)src[pa - row_base] << (8 * b);
            }
            int y0, y1, u, vv;
            if (fmt == ATI_FMT_VYUY422) {   /* V Y1 U Y0 from msb */
                y0 = (w2 >> 8) & 0xff; u = w2 & 0xff;
                y1 = (w2 >> 24) & 0xff; vv = (w2 >> 16) & 0xff;
            } else {                        /* Y V Y U */
                y0 = w2 & 0xff; vv = (w2 >> 8) & 0xff;
                y1 = (w2 >> 16) & 0xff; u = (w2 >> 24) & 0xff;
            }
            yuv_to_rgba(x & 1 ? y1 : y0, u, vv, o);
            break;
        }
        default:
            o[0] = 255; o[1] = 0; o[2] = 255; o[3] = 255;
            break;
        }
    }
}

static void encode_row(const uint8_t *src, uint8_t *dst, unsigned w,
                       uint8_t fmt, uint8_t swap, uint32_t row_base)
{
    unsigned bpp = ati_fmt_bpp(fmt);

    if (encode_row_fast(src, dst, w, fmt, swap, row_base)) {
        return;
    }
    for (unsigned x = 0; x < w; x++) {
        const uint8_t *p = src + 4 * x;
        uint32_t v;
        switch (fmt) {
        case ATI_FMT_ARGB1555:
            v = ((p[3] >= 128) << 15) | ((p[0] >> 3) << 10) |
                ((p[1] >> 3) << 5) | (p[2] >> 3);
            break;
        case ATI_FMT_RGB565:
            v = ((p[0] >> 3) << 11) | ((p[1] >> 2) << 5) | (p[2] >> 3);
            break;
        case ATI_FMT_ARGB4444:
            v = ((p[3] >> 4) << 12) | ((p[0] >> 4) << 8) | ((p[1] >> 4) << 4) |
                (p[2] >> 4);
            break;
        case ATI_FMT_ARGB8888:
            v = ((uint32_t)p[3] << 24) | (p[0] << 16) | (p[1] << 8) | p[2];
            break;
        case ATI_FMT_RGBA8888:
            v = ((uint32_t)p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
            break;
        case ATI_FMT_RGB332:
            v = (p[0] & 0xe0) | ((p[1] >> 3) & 0x1c) | (p[2] >> 6);
            break;
        case ATI_FMT_Y8:
        case ATI_FMT_I8:
            v = p[0];
            break;
        default:
            v = 0;
            break;
        }
        uint32_t off = x * bpp;
        for (unsigned b = 0; b < bpp; b++) {
            uint32_t a = row_base + off + b;
            uint32_t pa = (a & ~3u) | perm_byte_index(a & 3, swap);
            dst[pa - row_base] = v >> (8 * b);
        }
    }
}

/* ---------------------------------------------------------------------
 * Shaders.  One program; the R100 pixel pipeline (three combiner stages,
 * alpha test, fog) is selected by uniforms.
 */

/* sources start after the version line, which glr_shader() supplies */
static const char *vs_src =
    "uniform vec2 fbsize;\n"
    "in vec4 a_pos;\n"            /* x, y (pixels), z (0..1), rhw */
    "in vec4 a_c0;\n"
    "in vec4 a_c1;\n"
    "in vec4 a_t0;\n"
    "in vec4 a_t1;\n"
    "in vec4 a_t2;\n"
    "out vec4 v_c0;\n"
    "out vec4 v_c1;\n"
    "flat out vec4 f_c0;\n"
    "flat out vec4 f_c1;\n"
    "out vec4 v_t0;\n"
    "out vec4 v_t1;\n"
    "out vec4 v_t2;\n"
    "out float v_z;\n"
    "void main() {\n"
    "  float w = a_pos.w > 0.0 ? 1.0 / a_pos.w : 1.0;\n"
    "  vec2 ndc = vec2(a_pos.x / fbsize.x * 2.0 - 1.0,\n"
    "                  a_pos.y / fbsize.y * 2.0 - 1.0);\n"
    "  float z = clamp(a_pos.z, 0.0, 1.0) * 2.0 - 1.0;\n"
    "  gl_Position = vec4(ndc * w, z * w, w);\n"
    "  v_c0 = a_c0; v_c1 = a_c1; f_c0 = a_c0; f_c1 = a_c1;\n"
    "  v_t0 = a_t0; v_t1 = a_t1; v_t2 = a_t2; v_z = a_pos.z;\n"
    "}\n";

static const char *fs_src =
    "precision highp float;\n"
    "uniform sampler2D tex0;\n"
    "uniform sampler2D tex1;\n"
    "uniform sampler2D tex2;\n"
    "uniform ivec4 stage_c[3];\n"   /* argA, argB, argC, op|comp<<4|scale<<8|clamp<<12 */
    "uniform ivec4 stage_a[3];\n"
    "uniform vec4 tfactor[3];\n"
    "uniform int stage_en;\n"
    "uniform int tex_en;\n"
    "uniform ivec4 alpha;\n"        /* func (0..7), enabled */
    "uniform vec4 alpha_ref;\n"
    "uniform ivec4 fog;\n"          /* enabled, source */
    "uniform vec4 fogcol;\n"
    "uniform ivec4 flags;\n"        /* flat, specular, solid, persp */
    "uniform vec4 solid;\n"
    "in vec4 v_c0;\n"
    "in vec4 v_c1;\n"
    "flat in vec4 f_c0;\n"
    "flat in vec4 f_c1;\n"
    "in vec4 v_t0;\n"
    "in vec4 v_t1;\n"
    "in vec4 v_t2;\n"
    "in float v_z;\n"
    "layout(location = 0) out vec4 o_c;\n"
    "vec4 T[4];\n"
    "vec4 cur; vec4 dif; vec4 spec;\n"
    "vec3 carg(int a, int s) {\n"
    "  if (a == 0) return vec3(0.0);\n"
    "  if (a == 2) return cur.rgb;\n"
    "  if (a == 3) return vec3(cur.a);\n"
    "  if (a == 4) return dif.rgb;\n"
    "  if (a == 5) return vec3(dif.a);\n"
    "  if (a == 6) return spec.rgb;\n"
    "  if (a == 7) return vec3(spec.a);\n"
    "  if (a == 8) return tfactor[s].rgb;\n"
    "  if (a == 9) return vec3(tfactor[s].a);\n"
    "  if (a >= 10 && a <= 17) {\n"
    "    vec4 t = T[(a - 10) / 2];\n"
    "    return ((a & 1) == 0) ? t.rgb : vec3(t.a);\n"
    "  }\n"
    "  return vec3(0.0);\n"
    "}\n"
    "float aarg(int a, int s) {\n"
    "  if (a == 0) return 0.0;\n"
    "  if (a == 1) return cur.a;\n"
    "  if (a == 2) return dif.a;\n"
    "  if (a == 3) return spec.a;\n"
    "  if (a == 4) return tfactor[s].a;\n"
    "  if (a >= 5 && a <= 8) return T[a - 5].a;\n"
    "  return 0.0;\n"
    "}\n"
    "vec4 sampleu(int u, vec4 tc) {\n"
    "  vec2 st = (flags.w != 0 && tc.z != 0.0) ? tc.xy / tc.z : tc.xy;\n"
    "  if (u == 0) return texture(tex0, st);\n"
    "  if (u == 1) return texture(tex1, st);\n"
    "  return texture(tex2, st);\n"
    "}\n"
    "bool atest(int f, float a, float r) {\n"
    "  if (f == 0) return false;\n"
    "  if (f == 1) return a < r;\n"
    "  if (f == 2) return a <= r;\n"
    "  if (f == 3) return abs(a - r) < 0.5 / 255.0;\n"
    "  if (f == 4) return a >= r;\n"
    "  if (f == 5) return a > r;\n"
    "  if (f == 6) return abs(a - r) >= 0.5 / 255.0;\n"
    "  return true;\n"
    "}\n"
    "void main() {\n"
    "  dif = flags.x != 0 ? f_c0 : v_c0;\n"
    "  spec = flags.x != 0 ? f_c1 : v_c1;\n"
    "  if (flags.z != 0) dif = solid;\n"
    "  T[0] = (tex_en & 1) != 0 ? sampleu(0, v_t0) : vec4(0.0);\n"
    "  T[1] = (tex_en & 2) != 0 ? sampleu(1, v_t1) : vec4(0.0);\n"
    "  T[2] = (tex_en & 4) != 0 ? sampleu(2, v_t2) : vec4(0.0);\n"
    "  T[3] = vec4(0.0);\n"
    "  cur = dif;\n"
    "  for (int s = 0; s < 3; s++) {\n"
    "    if ((stage_en & (1 << s)) == 0) continue;\n"
    "    ivec4 c = stage_c[s]; ivec4 a = stage_a[s];\n"
    "    vec3 A = carg(c.x, s), B = carg(c.y, s), C = carg(c.z, s);\n"
    "    int op = c.w & 15, comp = (c.w >> 4) & 7;\n"
    "    if ((comp & 1) != 0) A = 1.0 - A;\n"
    "    if ((comp & 2) != 0) B = 1.0 - B;\n"
    "    if ((comp & 4) != 0) C = 1.0 - C;\n"
    "    vec3 r;\n"
    "    if (op == 0) r = A * B + C;\n"
    "    else if (op == 1) r = A * B - C;\n"
    "    else if (op == 2) r = A * B + C - 0.5;\n"
    "    else if (op == 3) r = A * C + B * (1.0 - C);\n"
    "    else r = vec3(clamp(4.0 * dot(A - 0.5, B - 0.5), 0.0, 1.0));\n"
    "    r *= float(1 << ((c.w >> 8) & 3));\n"
    "    float aA = aarg(a.x, s), aB = aarg(a.y, s), aC = aarg(a.z, s);\n"
    "    int aop = a.w & 15, acomp = (a.w >> 4) & 7;\n"
    "    if ((acomp & 1) != 0) aA = 1.0 - aA;\n"
    "    if ((acomp & 2) != 0) aB = 1.0 - aB;\n"
    "    if ((acomp & 4) != 0) aC = 1.0 - aC;\n"
    "    float ra;\n"
    "    if (aop == 0) ra = aA * aB + aC;\n"
    "    else if (aop == 1) ra = aA * aB - aC;\n"
    "    else if (aop == 2) ra = aA * aB + aC - 0.5;\n"
    "    else if (aop == 3) ra = aA * aC + aB * (1.0 - aC);\n"
    "    else ra = r.r;\n"
    "    ra *= float(1 << ((a.w >> 8) & 3));\n"
    "    cur = clamp(vec4(r, ra), 0.0, 1.0);\n"
    "  }\n"
    "  if (flags.y != 0) cur.rgb = clamp(cur.rgb + spec.rgb, 0.0, 1.0);\n"
    "  if (alpha.y != 0 && !atest(alpha.x, cur.a, alpha_ref.x)) discard;\n"
    "  if (fog.x != 0) {\n"
    "    float f = fog.y == 0 ? spec.a : (fog.y == 1 ? 1.0 - v_z : dif.a);\n"
    "    cur.rgb = mix(fogcol.rgb, cur.rgb, clamp(f, 0.0, 1.0));\n"
    "  }\n"
    "  o_c = cur;\n"
    "}\n";

static GLuint glr_shader(ATIRenderer *r, GLenum type, const char *src)
{
    ATIGL *gl = &r->gl;
    GLuint s = gl->CreateShader(type);
    GLint ok = 0;
    const GLchar *srcs[2] = { ATI_GLSL_VERSION, src };

    gl->ShaderSource(s, 2, srcs, NULL);
    gl->CompileShader(s);
    gl->GetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        gl->GetShaderInfoLog(s, sizeof(log), NULL, log);
        qemu_log_mask(LOG_UNIMP, "ati: shader compile failed: %s\n", log);
    }
    return s;
}

static bool glr_program(ATIRenderer *r)
{
    ATIGL *gl = &r->gl;
    GLint ok = 0;
    static const char *attrs[] = { "a_pos", "a_c0", "a_c1", "a_t0", "a_t1", "a_t2" };

    r->prog = gl->CreateProgram();
    gl->AttachShader(r->prog, glr_shader(r, GL_VERTEX_SHADER, vs_src));
    gl->AttachShader(r->prog, glr_shader(r, GL_FRAGMENT_SHADER, fs_src));
    for (int i = 0; i < 6; i++) {
        gl->BindAttribLocation(r->prog, i, attrs[i]);
    }
    gl->LinkProgram(r->prog);
    gl->GetProgramiv(r->prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        gl->GetProgramInfoLog(r->prog, sizeof(log), NULL, log);
        snprintf(r->info, sizeof(r->info), "shader link failed: %.200s", log);
        return false;
    }
    gl->UseProgram(r->prog);
    r->u_fbsize = gl->GetUniformLocation(r->prog, "fbsize");
    r->u_stage_c = gl->GetUniformLocation(r->prog, "stage_c");
    r->u_stage_a = gl->GetUniformLocation(r->prog, "stage_a");
    r->u_tfactor = gl->GetUniformLocation(r->prog, "tfactor");
    r->u_stage_en = gl->GetUniformLocation(r->prog, "stage_en");
    r->u_tex_en = gl->GetUniformLocation(r->prog, "tex_en");
    r->u_alpha = gl->GetUniformLocation(r->prog, "alpha");
    r->u_alpharef = gl->GetUniformLocation(r->prog, "alpha_ref");
    r->u_fog = gl->GetUniformLocation(r->prog, "fog");
    r->u_fogcol = gl->GetUniformLocation(r->prog, "fogcol");
    r->u_flags = gl->GetUniformLocation(r->prog, "flags");
    r->u_solid = gl->GetUniformLocation(r->prog, "solid");
    gl->Uniform1i(gl->GetUniformLocation(r->prog, "tex0"), 0);
    gl->Uniform1i(gl->GetUniformLocation(r->prog, "tex1"), 1);
    gl->Uniform1i(gl->GetUniformLocation(r->prog, "tex2"), 2);
    gl->GenVertexArrays(1, &r->vao);
    gl->BindVertexArray(r->vao);
    gl->GenBuffers(1, &r->vbo);
    gl->BindBuffer(GL_ARRAY_BUFFER, r->vbo);
    for (int i = 0; i < 6; i++) {
        static const int offs[] = { 0, 4, 8, 12, 16, 20 };
        gl->VertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, sizeof(ATIVertex),
                                (const void *)(uintptr_t)(offs[i] * 4));
        gl->EnableVertexAttribArray(i);
    }
    return true;
}

/* ---------------------------------------------------------------------
 * Render targets
 */

static GLRTarget *glr_target(ATIRenderer *r, const ATIImage *img, bool create)
{
    ATIGL *gl = &r->gl;
    GLRTarget *t, *victim = NULL;

    for (int i = 0; i < MAX_TARGETS; i++) {
        t = &r->targets[i];
        if (t->used && t->mc == img->mc && t->pitch == img->pitch &&
            t->fmt == img->fmt) {
            t->lru = ++r->tick;
            return t;
        }
        if (!t->used) {
            victim = victim && !victim->used ? victim : t;
        } else if (!victim || (victim->used && t->lru < victim->lru)) {
            victim = t;
        }
    }
    if (!create) {
        return NULL;
    }
    t = victim;
    if (t->used) {
        gl->DeleteFramebuffers(1, &t->fbo);
        gl->DeleteTextures(1, &t->tex);
        if (t->depth_rb) {
            gl->DeleteRenderbuffers(1, &t->depth_rb);
        }
    }
    memset(t, 0, sizeof(*t));
    t->used = true;
    t->mc = img->mc;
    t->pitch = img->pitch;
    t->fmt = img->fmt;
    t->w = MIN(img->pitch / MAX(ati_fmt_bpp(img->fmt), 1), 2048);
    t->h = TARGET_H;
    t->lru = ++r->tick;
    gl->GenTextures(1, &t->tex);
    gl->BindTexture(GL_TEXTURE_2D, t->tex);
    gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, t->w, t->h, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl->GenFramebuffers(1, &t->fbo);
    gl->BindFramebuffer(GL_FRAMEBUFFER, t->fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, t->tex, 0);
    r->bound = NULL;
    return t;
}

static void glr_bind(ATIRenderer *r, GLRTarget *t, uint32_t depth_mc)
{
    ATIGL *gl = &r->gl;

    if (depth_mc && t->depth_mc != depth_mc) {
        if (!t->depth_rb) {
            gl->GenRenderbuffers(1, &t->depth_rb);
            gl->BindRenderbuffer(GL_RENDERBUFFER, t->depth_rb);
            gl->RenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
                                    t->w, t->h);
        }
        gl->BindFramebuffer(GL_FRAMEBUFFER, t->fbo);
        gl->FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                    GL_RENDERBUFFER, t->depth_rb);
        t->depth_mc = depth_mc;
        r->bound = NULL;
    }
    if (r->bound != t) {
        gl->BindFramebuffer(GL_FRAMEBUFFER, t->fbo);
        r->bound = t;
    }
}

/*
 * RGBA8 staging for a whole target, kept between frames: a fresh 8 MB buffer
 * per conversion (1920x1080) cost page faults and kernel zeroing every frame.
 */
static uint8_t *glr_scratch(ATIRenderer *r, size_t len)
{
    if (r->scratch_len < len) {
        g_free(r->scratch);
        r->scratch = g_malloc(len);
        r->scratch_len = len;
    }
    return r->scratch;
}

static void glr_upload(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    GLRTarget *t = glr_target(r, &c->img, true);
    unsigned w = MIN(c->img.w, t->w), h = MIN(c->img.h, t->h);
    uint8_t *rgba = glr_scratch(r, (size_t)w * 4 * h);

    for (unsigned y = 0; y < h; y++) {
        decode_row(c->data + y * c->img.pitch, rgba + (size_t)y * w * 4, w,
                   c->img.fmt, c->img.swap, c->img.mc + y * c->img.pitch);
    }
    gl->BindTexture(GL_TEXTURE_2D, t->tex);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                      rgba);
}

static void glr_read(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    GLRTarget *t = glr_target(r, &c->img, false);
    unsigned w, h;
    uint8_t *rgba;

    if (t) {
        w = MIN(c->img.w, t->w);
        h = MIN(c->img.h, t->h);
        rgba = glr_scratch(r, (size_t)w * 4 * h);
        glr_bind(r, t, t->depth_mc);
        gl->PixelStorei(GL_PACK_ALIGNMENT, 1);
        gl->ReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        for (unsigned y = 0; y < h; y++) {
            encode_row(rgba + (size_t)y * w * 4, c->data + y * c->img.pitch, w,
                       c->img.fmt, c->img.swap, c->img.mc + y * c->img.pitch);
        }
    }
    /* the caller's change detection, here rather than on the vCPU */
    c->hash = ati_vram_hash(c->data, (size_t)c->img.pitch * c->img.h);
}

/*
 * A solid 2D fill on a target (ati_3d_fill).  The pixel is decoded the way an
 * upload decodes VRAM (the 2D engine stores the same value in both halves of
 * a 16-bit word pair, so its position in the word doesn't matter).  Draws set
 * their own scissor and colour mask, as the depth clear relies on too.
 */
static void glr_fill(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    GLRTarget *t = glr_target(r, &c->img, true);
    uint8_t rgba[4];

    decode_row(c->px, rgba, 1, c->img.fmt, c->img.swap, 0);
    glr_bind(r, t, t->depth_mc);
    gl->Enable(GL_SCISSOR_TEST);
    gl->Scissor(c->x, c->y, c->w, c->h);
    gl->ColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    gl->ClearColor(rgba[0] / 255.0f, rgba[1] / 255.0f, rgba[2] / 255.0f,
                   rgba[3] / 255.0f);
    gl->Clear(GL_COLOR_BUFFER_BIT);
}

static void glr_clearz(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    GLRTarget *t = glr_target(r, &c->img, true);

    glr_bind(r, t, c->depth_mc);
    gl->Enable(GL_SCISSOR_TEST);
    gl->Scissor(c->x, c->y, c->w, c->h);
    gl->DepthMask(GL_TRUE);
    gl->StencilMask(0xff);
    gl->ClearDepthf(c->z);
    gl->ClearStencil(c->stencil);
    gl->Clear(GL_DEPTH_BUFFER_BIT | (c->stencil >= 0 ? GL_STENCIL_BUFFER_BIT : 0));
}

/* ---------------------------------------------------------------------
 * Textures
 */

static GLRTexture *glr_tex_find(ATIRenderer *r, uint64_t key)
{
    for (int i = 0; i < MAX_TEXTURES; i++) {
        if (r->texs[i].used && r->texs[i].key == key) {
            r->texs[i].lru = ++r->tick;
            return &r->texs[i];
        }
    }
    return NULL;
}

static void glr_tex(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    GLRTexture *t = glr_tex_find(r, c->key);
    const ATITexUnit *u = &c->tu;

    if (!t) {
        GLRTexture *victim = &r->texs[0];
        for (int i = 0; i < MAX_TEXTURES; i++) {
            if (!r->texs[i].used) {
                victim = &r->texs[i];
                break;
            }
            if (r->texs[i].lru < victim->lru) {
                victim = &r->texs[i];
            }
        }
        t = victim;
        if (t->used) {
            gl->DeleteTextures(1, &t->tex);
        }
        memset(t, 0, sizeof(*t));
        t->used = true;
        t->key = c->key;
        gl->GenTextures(1, &t->tex);
    }
    t->gen = c->gen;
    t->lru = ++r->tick;
    t->w = u->img.w;
    t->h = u->img.h;
    gl->ActiveTexture(GL_TEXTURE0);
    gl->BindTexture(GL_TEXTURE_2D, t->tex);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (unsigned l = 0; l < u->levels; l++) {
        unsigned w = u->level_w[l], h = u->level_h[l];
        uint8_t *rgba = g_malloc((size_t)w * 4 * h);
        for (unsigned y = 0; y < h; y++) {
            size_t off = u->level_offs[l] + (size_t)y * u->level_pitch[l];
            if (off + (size_t)w * ati_fmt_bpp(u->img.fmt) <= c->len) {
                decode_row(c->data + off, rgba + y * w * 4, w, u->img.fmt,
                           u->img.swap, u->img.mc + off);
            } else {
                memset(rgba + y * w * 4, 0, w * 4);
            }
        }
        gl->TexImage2D(GL_TEXTURE_2D, l, GL_RGBA8, w, h, 0, GL_RGBA,
                       GL_UNSIGNED_BYTE, rgba);
        g_free(rgba);
    }
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, u->levels - 1);
    t->min = t->mag = t->ws = t->wt = 0;
}

static void glr_tex_bind(ATIRenderer *r, int unit, const ATITexUnit *u)
{
    ATIGL *gl = &r->gl;
    GLRTexture *t = glr_tex_find(r, u->key);
    uint16_t min = u->min_filter;

    gl->ActiveTexture(GL_TEXTURE0 + unit);
    if (!t) {
        gl->BindTexture(GL_TEXTURE_2D, 0);
        return;
    }
    gl->BindTexture(GL_TEXTURE_2D, t->tex);
    if (u->levels <= 1 && min != GL_NEAREST && min != GL_LINEAR) {
        min = (min == GL_NEAREST_MIPMAP_NEAREST || min == GL_NEAREST_MIPMAP_LINEAR)
              ? GL_NEAREST : GL_LINEAR;
    }
    if (t->min != min) {
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min);
        t->min = min;
    }
    if (t->mag != u->mag_filter) {
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, u->mag_filter);
        t->mag = u->mag_filter;
    }
    if (t->ws != u->wrap_s) {
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, u->wrap_s);
        t->ws = u->wrap_s;
    }
    if (t->wt != u->wrap_t) {
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, u->wrap_t);
        t->wt = u->wrap_t;
    }
}

/* ---------------------------------------------------------------------
 * Drawing
 */

static void glr_draw(ATIRenderer *r, GLRCmd *c)
{
    ATIGL *gl = &r->gl;
    const ATIDrawState *st = &c->st;
    GLRTarget *t = glr_target(r, &st->color, true);
    GLint sc[12], sa[12];
    float tf[12];

    glr_bind(r, t, st->depth_mc);
    gl->UseProgram(r->prog);
    gl->BindVertexArray(r->vao);
    gl->Viewport(0, 0, t->w, t->h);
    gl->Uniform2f(r->u_fbsize, t->w, t->h);

    if (st->scissor) {
        gl->Enable(GL_SCISSOR_TEST);
        gl->Scissor(st->sc_x0, st->sc_y0, MAX(st->sc_x1 - st->sc_x0 + 1, 0),
                    MAX(st->sc_y1 - st->sc_y0 + 1, 0));
    } else {
        gl->Disable(GL_SCISSOR_TEST);
    }
    if (st->depth_mc && (st->z_test || st->z_write)) {
        gl->Enable(GL_DEPTH_TEST);
        gl->DepthFunc(st->z_test ? st->z_func : GL_ALWAYS);
        gl->DepthMask(st->z_write);
    } else {
        gl->Disable(GL_DEPTH_TEST);
    }
    if (st->depth_mc && st->stencil) {
        gl->Enable(GL_STENCIL_TEST);
        gl->StencilFunc(st->st_func, st->st_ref, st->st_mask);
        gl->StencilOp(st->st_fail, st->st_zfail, st->st_zpass);
        gl->StencilMask(st->st_wmask);
    } else {
        gl->Disable(GL_STENCIL_TEST);
    }
    if (st->blend) {
        gl->Enable(GL_BLEND);
        gl->BlendFuncSeparate(st->blend_src, st->blend_dst, st->blend_src,
                              st->blend_dst);
        gl->BlendEquationSeparate(st->blend_eq, st->blend_eq);
    } else {
        gl->Disable(GL_BLEND);
    }
    gl->ColorMask(st->color_mask & 1, (st->color_mask >> 1) & 1,
                  (st->color_mask >> 2) & 1, (st->color_mask >> 3) & 1);
    if (st->cull_enable) {
        gl->Enable(GL_CULL_FACE);
        gl->CullFace(st->cull_face);
        gl->FrontFace(st->front_ccw ? GL_CCW : GL_CW);
    } else {
        gl->Disable(GL_CULL_FACE);
    }
    if (st->zbias) {
        gl->Enable(GL_POLYGON_OFFSET_FILL);
        gl->PolygonOffset(st->zbias_factor, st->zbias_const);
    } else {
        gl->Disable(GL_POLYGON_OFFSET_FILL);
    }

    for (int s = 0; s < 3; s++) {
        uint32_t cb = st->stage[s].cblend, ab = st->stage[s].ablend;
        sc[s * 4 + 0] = cb & 0x1f;
        sc[s * 4 + 1] = (cb >> 5) & 0x1f;
        sc[s * 4 + 2] = (cb >> 10) & 0x1f;
        sc[s * 4 + 3] = ((cb >> 18) & 7) | (((cb >> 15) & 7) << 4) |
                        (((cb >> 21) & 3) << 8);
        sa[s * 4 + 0] = ab & 0xf;
        sa[s * 4 + 1] = (ab >> 4) & 0xf;
        sa[s * 4 + 2] = (ab >> 8) & 0xf;
        sa[s * 4 + 3] = ((ab >> 18) & 7) | (((ab >> 15) & 7) << 4) |
                        (((ab >> 21) & 3) << 8);
        memcpy(&tf[s * 4], st->stage[s].tfactor, 16);
        if (st->tex_enable & (1 << s)) {
            glr_tex_bind(r, s, &st->tex[s]);
        }
    }
    gl->Uniform4iv(r->u_stage_c, 3, sc);
    gl->Uniform4iv(r->u_stage_a, 3, sa);
    gl->Uniform4fv(r->u_tfactor, 3, tf);
    gl->Uniform1i(r->u_stage_en, st->stage_enable);
    gl->Uniform1i(r->u_tex_en, st->tex_enable);
    {
        GLint av[4] = { st->alpha_func, st->alpha_test, 0, 0 };
        GLint fg[4] = { st->fog, st->fog_src, 0, 0 };
        GLint fl[4] = { st->flat, st->specular, st->solid_color, st->persp_tex };
        gl->Uniform4iv(r->u_alpha, 1, av);
        gl->Uniform4f(r->u_alpharef, st->alpha_ref, 0, 0, 0);
        gl->Uniform4iv(r->u_fog, 1, fg);
        gl->Uniform4fv(r->u_fogcol, 1, st->fog_color);
        gl->Uniform4iv(r->u_flags, 1, fl);
        gl->Uniform4fv(r->u_solid, 1, st->solid);
    }
    gl->BindBuffer(GL_ARRAY_BUFFER, r->vbo);
    gl->BufferData(GL_ARRAY_BUFFER, c->nv * sizeof(ATIVertex), c->v,
                   GL_STREAM_DRAW);
    gl->DrawArrays(c->prim, 0, c->nv);
}

/* ---------------------------------------------------------------------
 * Thread and queue
 */

static void *glr_thread(void *opaque)
{
    ATIRenderer *r = opaque;

    r->ok = ati_gl_open(&r->gl, r->info, sizeof(r->info)) && glr_program(r);
    qemu_event_set(&r->ready);
    if (!r->ok) {
        return NULL;
    }
    for (;;) {
        GLRCmd *c;
        qemu_mutex_lock(&r->lock);
        while (!r->head) {
            qemu_cond_wait(&r->cond, &r->lock);
        }
        c = r->head;
        r->head = c->next;
        if (!r->head) {
            r->tail = NULL;
        }
        qemu_mutex_unlock(&r->lock);
        switch (c->kind) {
        case CMD_DRAW:
            glr_draw(r, c);
            break;
        case CMD_TEX:
            glr_tex(r, c);
            break;
        case CMD_UPLOAD:
            glr_upload(r, c);
            break;
        case CMD_READ:
            glr_read(r, c);
            break;
        case CMD_READ_ASYNC:
            glr_read(r, c);
            c->job->hash = c->hash;
            qatomic_store_release(&c->job->done, 1);
            break;
        case CMD_CLEARZ:
            glr_clearz(r, c);
            break;
        case CMD_FILL:
            glr_fill(r, c);
            break;
        case CMD_FINISH:
            r->gl.Finish();
            break;
        }
        if (c->done) {
            qemu_event_set(c->done);
        } else {
            g_free(c->v);
            if (c->kind != CMD_READ && c->kind != CMD_READ_ASYNC) {
                g_free(c->data);
            }
            g_free(c);
        }
    }
    return NULL;
}

static void glr_push(ATIRenderer *r, GLRCmd *c)
{
    qemu_mutex_lock(&r->lock);
    c->next = NULL;
    if (r->tail) {
        r->tail->next = c;
    } else {
        r->head = c;
    }
    r->tail = c;
    qemu_cond_signal(&r->cond);
    qemu_mutex_unlock(&r->lock);
}

/*
 * Wait for the render thread.  The vCPU calls this from MMIO with the BQL
 * held; waiting for a frame at 1920x1080 took ~10 ms, during which the main
 * loop could not run its timers: the Screamer's 1 ms output DMA commands fell
 * behind (Cro-Mag Rally: 43 % of its sound reached the host), and the vblank
 * slipped to 52 Hz.  So a vCPU releases the BQL for the wait.  r->waiting
 * tells the device's other BQL holders (the vblank timer) that the 3D state
 * is mid-operation; the vCPU itself cannot touch the device until it
 * returns.  Any other thread (none waits today) keeps the BQL, so no vCPU can
 * enter the device under it.
 */
static void glr_push_wait(ATIRenderer *r, GLRCmd *c)
{
    QemuEvent ev;
    bool unlock = current_cpu && bql_locked();

    qemu_event_init(&ev, false);
    c->done = &ev;
    glr_push(r, c);
    if (unlock) {
        r->waiting = true;
        bql_unlock();
    }
    qemu_event_wait(&ev);
    if (unlock) {
        bql_lock();
        r->waiting = false;
    }
    qemu_event_destroy(&ev);
}

bool ati_glr_waiting(ATIRenderer *r)
{
    return r && r->waiting;
}

uint64_t ati_vram_hash(const uint8_t *p, size_t len)
{
    /* fast 64-bit content hash (4 lanes, 8 bytes per step) */
    uint64_t h[4] = { 0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
                      0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL };
    size_t i = 0;

    for (; i + 32 <= len; i += 32) {
        for (int l = 0; l < 4; l++) {
            h[l] = (h[l] ^ ldq_he_p(p + i + 8 * l)) * 0x100000001B3ULL;
            h[l] ^= h[l] >> 29;
        }
    }
    for (; i < len; i++) {
        h[0] = (h[0] ^ p[i]) * 0x100000001B3ULL;
    }
    return h[0] ^ (h[1] * 3) ^ (h[2] * 5) ^ (h[3] * 7) ^ len;
}

ATIRenderer *ati_glr_new(char *info, size_t infolen)
{
    ATIRenderer *r = g_new0(ATIRenderer, 1);

    qemu_mutex_init(&r->lock);
    qemu_cond_init(&r->cond);
    qemu_event_init(&r->ready, false);
    qemu_thread_create(&r->thread, "ati-gl", glr_thread, r,
                       QEMU_THREAD_DETACHED);
    qemu_event_wait(&r->ready);
    snprintf(info, infolen, "%s", r->info);
    if (!r->ok) {
        return NULL;    /* the thread has exited; r is leaked deliberately */
    }
    return r;
}

const char *ati_glr_info(ATIRenderer *r)
{
    return r->info;
}

void ati_glr_draw(ATIRenderer *r, const ATIDrawState *st, uint16_t prim,
                  const ATIVertex *v, unsigned nv)
{
    GLRCmd *c = g_new0(GLRCmd, 1);

    c->kind = CMD_DRAW;
    c->st = *st;
    c->prim = prim;
    c->nv = nv;
    c->v = g_memdup2(v, nv * sizeof(*v));
    glr_push(r, c);
}

void ati_glr_texture(ATIRenderer *r, uint64_t key, uint32_t gen,
                     const ATITexUnit *t, const uint8_t *data, size_t len)
{
    GLRCmd *c = g_new0(GLRCmd, 1);

    c->kind = CMD_TEX;
    c->key = key;
    c->gen = gen;
    c->tu = *t;
    c->data = g_memdup2(data, len);
    c->len = len;
    glr_push(r, c);
}

void ati_glr_target_upload_take(ATIRenderer *r, const ATIImage *img,
                                uint8_t *data)
{
    GLRCmd *c = g_new0(GLRCmd, 1);

    c->kind = CMD_UPLOAD;
    c->img = *img;
    c->data = data;             /* freed by the render thread */
    glr_push(r, c);
}

void ati_glr_target_upload(ATIRenderer *r, const ATIImage *img,
                           const uint8_t *data)
{
    ati_glr_target_upload_take(r, img,
                               g_memdup2(data, (size_t)img->pitch * img->h));
}

void ati_glr_fill(ATIRenderer *r, const ATIImage *img, int x, int y, int w,
                  int h, const uint8_t px[4])
{
    GLRCmd *c = g_new0(GLRCmd, 1);

    c->kind = CMD_FILL;
    c->img = *img;
    c->x = x;
    c->y = y;
    c->w = w;
    c->h = h;
    memcpy(c->px, px, 4);
    glr_push(r, c);
}

void ati_glr_depth_clear(ATIRenderer *r, const ATIImage *color,
                         uint32_t depth_mc, float z, int stencil,
                         int x, int y, int w, int h)
{
    GLRCmd *c = g_new0(GLRCmd, 1);

    c->kind = CMD_CLEARZ;
    c->img = *color;
    c->depth_mc = depth_mc;
    c->z = z;
    c->stencil = stencil;
    c->x = x;
    c->y = y;
    c->w = w;
    c->h = h;
    glr_push(r, c);
}

uint64_t ati_glr_target_read(ATIRenderer *r, const ATIImage *img, uint8_t *out)
{
    GLRCmd c = { .kind = CMD_READ, .img = *img, .data = out };

    glr_push_wait(r, &c);
    return c.hash;
}

/*
 * Asynchronous readback for scanout: the render thread reads the target after
 * every command queued before it into job->buf (encoded like VRAM) and sets
 * job->done; the caller copies it into VRAM later (one frame of latency, no
 * wait for the host GPU or llvmpipe at the vblank).
 */
ATIReadJob *ati_glr_target_read_async(ATIRenderer *r, const ATIImage *img)
{
    ATIReadJob *j = g_new0(ATIReadJob, 1);
    GLRCmd *c = g_new0(GLRCmd, 1);

    j->img = *img;
    j->len = (size_t)img->pitch * img->h;
    j->buf = g_malloc0(j->len);
    c->kind = CMD_READ_ASYNC;
    c->img = *img;
    c->data = j->buf;
    c->job = j;
    glr_push(r, c);
    return j;
}

bool ati_glr_read_done(ATIReadJob *j)
{
    return qatomic_load_acquire(&j->done);
}

void ati_glr_read_wait(ATIRenderer *r, ATIReadJob *j)
{
    if (!ati_glr_read_done(j)) {
        ati_glr_finish(r);          /* commands run in order: j is done after it */
    }
}

void ati_glr_read_free(ATIReadJob *j)
{
    if (j) {
        g_free(j->buf);
        g_free(j);
    }
}

void ati_glr_finish(ATIRenderer *r)
{
    GLRCmd c = { .kind = CMD_FINISH };

    glr_push_wait(r, &c);
}
