/*
 * R300-family draw assembly.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_draw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "r300_pvs.h"
#include "r300_sb.h"

/* Registers used here */
#define SE_VPORT_XSCALE             0x1D98  /* then XOFFSET, YSCALE, ... ZOFFSET */
#define GB_MSPOS0                   0x4010
#define GB_MSPOS1                   0x4014
#define GB_SELECT                   0x401C
#define GB_AA_CONFIG                0x4020
#define VAP_CLIP_CNTL               0x221C
#define VAP_PVS_FLOW_CNTL_ADDRS_0   0x2230
#define VAP_PVS_FLOW_CNTL_LOOP_INDEX_0 0x2290
#define VAP_PVS_FLOW_CNTL_OPC       0x22DC
#define VAP_CNTL_STATUS             0x2140
#define VAP_OUTPUT_VTX_FMT_0        0x2090
#define VAP_OUTPUT_VTX_FMT_1        0x2094
#define VAP_VTE_CNTL                0x20B0
#define VAP_VTX_SIZE                0x20B4
#define VAP_PROG_STREAM_CNTL_0      0x2150
#define VAP_PROG_STREAM_CNTL_EXT_0  0x21E0
#define VAP_PVS_CODE_CNTL_0         0x22D0
#define VAP_PVS_CONST_CNTL          0x22D4
#define TX_ENABLE                   0x4104
#define RS_COUNT                    0x4300
#define RS_INST_COUNT               0x4304
#define RS_IP_0                     0x4310
#define RS_INST_0                   0x4330
#define SC_CLIPRECT_TL_0            0x43B0  /* then BR_0, TL_1, BR_1, ... */
#define SC_CLIP_RULE                0x43D0
#define SC_SCISSORS_TL              0x43E0
#define SC_SCISSORS_BR              0x43E4
#define TX_FILTER0_0                0x4400
#define TX_FILTER1_0                0x4440
#define TX_FORMAT0_0                0x4480
#define TX_FORMAT1_0                0x44C0
#define TX_FORMAT2_0                0x4500
#define TX_OFFSET_0                 0x4540
#define TX_BORDER_COLOR_0           0x45C0
#define US_OUT_FMT_0                0x46A4
#define FG_FOG_BLEND                0x4BC0
#define FG_FOG_FACTOR               0x4BC4
#define FG_FOG_COLOR_R              0x4BC8  /* then G, B */
#define FG_ALPHA_FUNC               0x4BD4
#define FG_DEPTH_SRC                0x4BD8
#define PFS_PARAM_0_X               0x4C00
#define RB3D_CBLEND                 0x4E04
#define RB3D_ABLEND                 0x4E08
#define RB3D_COLOR_CHANNEL_MASK     0x4E0C
#define RB3D_BLEND_COLOR            0x4E10
#define RB3D_ROPCNTL                0x4E18
#define RB3D_COLOROFFSET0           0x4E28
#define RB3D_COLORPITCH0            0x4E38
#define SU_POLY_OFFSET_FRONT_SCALE  0x42A4  /* then FRONT_OFFSET, BACK_SCALE, BACK_OFFSET */
#define SU_POLY_OFFSET_ENABLE       0x42B4
#define SU_CULL_MODE                0x42B8
#define ZB_CNTL                     0x4F00
#define ZB_ZSTENCILCNTL             0x4F04
#define ZB_STENCILREFMASK           0x4F08
#define ZB_FORMAT                   0x4F10
#define ZB_DEPTHOFFSET              0x4F20
#define ZB_DEPTHPITCH               0x4F24
#define GA_POINT_S0                 0x4200  /* then T0, S1, T1 */
#define GA_POINT_T0                 0x4204
#define GA_POINT_S1                 0x4208
#define GA_POINT_T1                 0x420C
#define GA_POINT_SIZE               0x421C
#define GA_POINT_MINMAX             0x4230
#define GA_LINE_CNTL                0x4234
#define GA_LINE_STIPPLE_CONFIG      0x4238
#define GA_LINE_STIPPLE_VALUE       0x4260
#define GA_LINE_S0                  0x4264
#define GA_LINE_S1                  0x4268
#define GA_COLOR_CONTROL            0x4278
#define GA_SOLID_RG                 0x427C
#define GA_SOLID_BA                 0x4280
#define GA_POLY_MODE                0x4288
#define GA_FOG_SCALE                0x4294
#define GA_FOG_OFFSET               0x4298

#define SC_COORD_BIAS               1440    /* scissor/cliprect origin */

static float bits_to_float(uint32_t v)
{
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static uint32_t vc_swap(uint32_t v, unsigned mode)
{
    switch (mode & 3) {
    case 1:  return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    case 2:  return __builtin_bswap32(v);
    case 3:  return (v << 16) | (v >> 16);
    default: return v;
    }
}

static float half_to_float(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float f;

    if (e == 0) {
        f = ldexpf((float)m, -24);
    } else if (e == 31) {
        f = m ? NAN : INFINITY;
    } else {
        f = ldexpf(1.0f + m / 1024.0f, (int)e - 15);
    }
    return s ? -f : f;
}

void r300_load_vbpntr(R300Arrays *arr, const uint32_t *d, uint32_t ndw)
{
    unsigned n, i;

    memset(arr, 0, sizeof(*arr));
    if (ndw < 1) {
        return;
    }
    n = d[0] & 0x1F;
    for (i = 0; i < n && i < R300_MAX_ARRAYS; i++) {
        unsigned pair = i / 2, half = i % 2;
        unsigned base = 1 + pair * 3;
        uint32_t fmt;

        if (base + 1 + half >= ndw) {
            break;
        }
        fmt = d[base] >> (16 * half);
        arr->a[i].size_dw = fmt & 0x7F;
        arr->a[i].stride_dw = (fmt >> 8) & 0x7F;
        arr->a[i].addr = d[base + 1 + half];
    }
    arr->count = i;
}

/* ---- vertex fetch --------------------------------------------------- */

typedef struct Fetch {
    const R300State *st;
    const R300Arrays *arr;
    R300ReadFn read;
    void *opaque;
    const uint32_t *immd;       /* immediate vertex data, or NULL */
    uint32_t immd_dw;
    uint32_t vtx_size;          /* VAP_VTX_SIZE, dwords */
    unsigned swap;              /* VAP_CNTL_STATUS.VC_SWAP */
    uint32_t warn;
    const char *why;            /* first fetch failure */
} Fetch;

static unsigned psc_dwords(unsigned type)
{
    static const unsigned n[16] = { 1, 2, 3, 4, 1, 1, 1, 2, 1, 1, 8, 1, 2, 0, 0, 0 };
    return n[type & 15];
}

/* Fetch dwords for PSC element k of vertex vtx starting at *cursor. */
static bool fetch_dwords(Fetch *f, unsigned k, uint32_t vtx, unsigned cursor,
                         unsigned count, uint32_t *out)
{
    if (f->immd) {
        uint64_t at = (uint64_t)vtx * f->vtx_size + cursor;
        if (at + count > f->immd_dw) {
            f->why = "immediate vertex data shorter than VAP_VTX_SIZE says";
            return false;
        }
        memcpy(out, &f->immd[at], count * 4);   /* CP data: already in order */
        return true;
    }
    if (!f->arr->count) {
        f->why = "vertex element with no 3D_LOAD_VBPNTR array";
        return false;
    }
    /* More elements than arrays: the rest are interleaved in the last one
     * (fetch_vertex keeps the cursor running for them). */
    if (k >= f->arr->count) {
        k = f->arr->count - 1;
    }
    uint32_t addr = f->arr->a[k].addr +
                    (vtx * f->arr->a[k].stride_dw + cursor) * 4;
    if (!f->read(f->opaque, addr, out, count * 4)) {
        f->why = "vertex array outside VRAM and the GART";
        return false;
    }
    for (unsigned i = 0; i < count; i++) {
        out[i] = vc_swap(out[i], f->swap);
    }
    return true;
}

static void decode_element(unsigned type, bool sgn, bool norm,
                           const uint32_t *w, float c[4])
{
    c[0] = c[1] = c[2] = 0.0f;
    c[3] = 1.0f;
    switch (type) {
    case 0: case 1: case 2: case 3:         /* FLOAT_1..4 */
        for (unsigned i = 0; i <= type; i++) {
            c[i] = bits_to_float(w[i]);
        }
        break;
    case 4:                                 /* BYTE x4 */
    case 5:                                 /* D3DCOLOR */
        for (unsigned i = 0; i < 4; i++) {
            unsigned b = (w[0] >> (8 * i)) & 0xFF;
            float x = sgn ? (float)(int8_t)b : (float)b;
            if (norm || type == 5) {
                x = sgn ? fmaxf(x / 127.0f, -1.0f) : x / 255.0f;
            }
            c[i] = x;
        }
        if (type == 5) {                    /* stored B,G,R,A */
            float t = c[0];
            c[0] = c[2];
            c[2] = t;
        }
        break;
    case 6:                                 /* SHORT_2 */
    case 7:                                 /* SHORT_4 */
        for (unsigned i = 0; i < (type == 6 ? 2u : 4u); i++) {
            unsigned h = (w[i / 2] >> (16 * (i % 2))) & 0xFFFF;
            float x = sgn ? (float)(int16_t)h : (float)h;
            if (norm) {
                x = sgn ? fmaxf(x / 32767.0f, -1.0f) : x / 65535.0f;
            }
            c[i] = x;
        }
        break;
    case 8:                                 /* VECTOR_3_TTT: 10:10:10 */
    case 9: {                               /* VECTOR_3_EET: 11:11:10 */
        static const uint8_t sh[2][3] = { { 0, 10, 20 }, { 0, 11, 22 } };
        static const uint8_t nb[2][3] = { { 10, 10, 10 }, { 11, 11, 10 } };
        for (unsigned i = 0; i < 3; i++) {
            unsigned n = nb[type - 8][i];
            uint32_t b = (w[0] >> sh[type - 8][i]) & ((1u << n) - 1);
            float x = sgn ? (float)((int32_t)(b << (32 - n)) >> (32 - n)) : (float)b;
            if (norm) {
                float m = (float)((1u << (sgn ? n - 1 : n)) - 1);
                x = sgn ? fmaxf(x / m, -1.0f) : x / m;
            }
            c[i] = x;
        }
        break;
    }
    case 11:                                /* FLT16_2 */
    case 12:                                /* FLT16_4 */
        for (unsigned i = 0; i < (type == 11 ? 2u : 4u); i++) {
            c[i] = half_to_float((w[i / 2] >> (16 * (i % 2))) & 0xFFFF);
        }
        break;
    default:
        break;
    }
}

/* PSC element k's EXT swizzle and write mask into one input vector. */
static void psc_store(unsigned x, const float c[4], float *dst)
{
    for (unsigned i = 0; i < 4; i++) {
        if ((x >> (12 + i)) & 1) {
            unsigned s = (x >> (3 * i)) & 7;
            dst[i] = s < 4 ? c[s] : s == 5 ? 1.0f : 0.0f;
        }
    }
}

/* Fill the input vertex memory for one vertex through the PSC. */
static bool fetch_vertex(Fetch *f, uint32_t vtx, float in[R300_PVS_NUM_INPUTS][4])
{
    unsigned cursor = 0;

    memset(in, 0, sizeof(float) * 4 * R300_PVS_NUM_INPUTS);
    for (unsigned k = 0; k < 16; k++) {
        uint32_t reg = r300_reg(f->st, VAP_PROG_STREAM_CNTL_0 + 4 * (k / 2));
        uint32_t ext = r300_reg(f->st, VAP_PROG_STREAM_CNTL_EXT_0 + 4 * (k / 2));
        unsigned e = (reg >> (16 * (k % 2))) & 0xFFFF;
        unsigned x = (ext >> (16 * (k % 2))) & 0xFFFF;
        unsigned type = e & 0xF, skip = (e >> 4) & 0xF, dst = (e >> 8) & 0x1F;
        unsigned n = psc_dwords(type);
        uint32_t w[8];
        float c[4];

        if (!f->immd && k < f->arr->count) {
            cursor = 0;                 /* each array starts afresh */
        }
        cursor += skip;
        if (!n) {
            f->warn |= R300_WARN_VTXFMT;
        } else if (!fetch_dwords(f, k, vtx, cursor, n, w)) {
            return false;
        } else if (type == 10) {
            /* FLOAT_8: two FLOAT_4 vectors at DST_VEC_LOC and the next */
            decode_element(3, false, false, w, c);
            psc_store(x, c, in[dst]);
            decode_element(3, false, false, w + 4, c);
            psc_store(x, c, in[(dst + 1) % R300_PVS_NUM_INPUTS]);
        } else {
            decode_element(type, (e >> 14) & 1, (e >> 15) & 1, w, c);
            psc_store(x, c, in[dst]);
        }
        cursor += n;
        if (e & (1u << 13)) {           /* LAST_VEC */
            break;
        }
    }
    return true;
}

/* ---- output routing ------------------------------------------------- */

typedef struct Layout {
    int pos, psize, color[4];
    int tex_slot[8];
    unsigned tex_n[8];
    /* texture interpolants as one scalar list: (slot, component) */
    unsigned nscal;
    uint8_t scal_slot[32], scal_comp[32];
} Layout;

static void output_layout(const R300State *st, Layout *l)
{
    uint32_t f0 = r300_reg(st, VAP_OUTPUT_VTX_FMT_0);
    uint32_t f1 = r300_reg(st, VAP_OUTPUT_VTX_FMT_1);
    int slot = 0;

    memset(l, -1, sizeof(*l));
    l->nscal = 0;
    l->pos = slot++;                    /* always present */
    if (f0 & (1u << 16)) {
        l->psize = slot++;              /* point size (x) */
    }
    for (int c = 0; c < 4; c++) {
        if (f0 & (1u << (1 + c))) {
            l->color[c] = slot++;
        }
    }
    for (int t = 0; t < 8; t++) {
        unsigned n = (f1 >> (3 * t)) & 7;
        l->tex_n[t] = n;
        if (n) {
            l->tex_slot[t] = slot++;
            for (unsigned i = 0; i < n && l->nscal < 32; i++) {
                l->scal_slot[l->nscal] = l->tex_slot[t];
                l->scal_comp[l->nscal++] = i;
            }
        }
    }
}

typedef struct Route {
    unsigned nvary;
    struct {
        bool is_tex;
        unsigned ip;
    } vary[R300_NUM_VARYINGS];
} Route;

/* RS_INST -> varyings and the temp routing for the fragment program. */
static void rs_route(const R300State *st, Route *r, R300FSDesc *desc,
                     uint32_t *warn)
{
    unsigned count = (r300_reg(st, RS_INST_COUNT) & 0xF) + 1;

    memset(r, 0, sizeof(*r));
    memset(desc->route, -1, sizeof(desc->route));
    for (unsigned i = 0; i < count; i++) {
        uint32_t inst = r300_reg(st, RS_INST_0 + 4 * i);

        for (int pass = 0; pass < 2; pass++) {
            bool tex = pass == 0;
            bool write = tex ? (inst >> 3) & 1 : (inst >> 14) & 1;
            unsigned ip = tex ? inst & 7 : (inst >> 11) & 7;
            unsigned addr = tex ? (inst >> 6) & 0x1F : (inst >> 17) & 0x1F;

            if (!write) {
                continue;
            }
            if (r->nvary == R300_NUM_VARYINGS) {
                *warn |= R300_WARN_VARYINGS;
                continue;
            }
            r->vary[r->nvary].is_tex = tex;
            r->vary[r->nvary].ip = ip;
            desc->route[addr] = r->nvary++;
        }
    }
}

static float rs_col_fmt(unsigned fmt, const float *c, unsigned i)
{
    static const int8_t tbl[16][4] = {
        /* per component: 0-3 = channel, -1 = 0.0, -2 = 1.0 */
        [0] = { 0, 1, 2, 3 },  [1] = { 0, 1, 2, -1 }, [2] = { 0, 1, 2, -2 },
        [4] = { -1, -1, -1, 3 }, [5] = { -1, -1, -1, -1 }, [6] = { -1, -1, -1, -2 },
        [8] = { -2, -2, -2, 3 }, [9] = { -2, -2, -2, -1 }, [10] = { -2, -2, -2, -2 },
    };
    int s = tbl[fmt & 15][i];
    return s >= 0 ? c[s] : s == -2 ? 1.0f : 0.0f;
}

/* The four rasterized colours of a vertex (RS colour pointer 0-3). */
static void vtx_colors(const Layout *l, float out[][4], float cols[4][4])
{
    for (int cp = 0; cp < 4; cp++) {
        if (l->color[cp] >= 0) {
            memcpy(cols[cp], out[l->color[cp]], sizeof(cols[cp]));
        } else {
            memset(cols[cp], 0, sizeof(cols[cp]));
        }
    }
}

/*
 * Interpolant values of one vertex: texture coordinates from the output
 * vertex, colours from cols (vtx_colors, or substituted for flat shading
 * and back faces), and the rasterizer's generated (s, t) (point sprites,
 * line stipple) past the vertex's own texture components.
 */
static void rs_values(const R300State *st, const Layout *l, const Route *r,
                      float out[][4], const float (*cols)[4],
                      const float *sprite, float v[R300_NUM_VARYINGS][4])
{
    for (unsigned k = 0; k < r->nvary; k++) {
        uint32_t ip = r300_reg(st, RS_IP_0 + 4 * r->vary[k].ip);

        if (r->vary[k].is_tex) {
            unsigned ptr = ip & 0x3F;
            for (unsigned i = 0; i < 4; i++) {
                unsigned sel = (ip >> (13 + 3 * i)) & 7;
                float x = 0.0f;
                if (sel < 4) {
                    unsigned s = ptr + sel;
                    if (s < l->nscal) {
                        x = out[l->scal_slot[s]][l->scal_comp[s]];
                    } else if (sprite) {
                        /* Point sprite: the rasterizer generates (s, t). */
                        unsigned c = s - l->nscal;
                        x = c < 2 ? sprite[c] : c == 3 ? 1.0f : 0.0f;
                    }
                } else if (sel == 5) {
                    x = 1.0f;
                }
                v[k][i] = x;
            }
        } else {
            static const float zero[4];
            unsigned cp = (ip >> 6) & 7;
            const float *c = cp < 4 ? cols[cp] : zero;
            for (unsigned i = 0; i < 4; i++) {
                v[k][i] = rs_col_fmt((ip >> 9) & 0xF, c, i);
            }
        }
    }
}

/* ---- draw ----------------------------------------------------------- */

static void set_uniforms(const R300State *st, R300DrawPacket *pkt)
{
    R300FSUniforms *u = &pkt->uniforms;
    uint32_t outfmt = r300_reg(st, US_OUT_FMT_0);
    uint32_t bc = r300_reg(st, RB3D_BLEND_COLOR);

    memset(u, 0, sizeof(*u));
    for (int i = 0; i < R300_US_NUM_CONSTS; i++) {
        for (int c = 0; c < 4; c++) {
            u->consts[i][c] = r300_float24(r300_reg(st, PFS_PARAM_0_X + 16 * i + 4 * c));
        }
    }
    u->blend_color[0] = ((bc >> 16) & 0xFF) / 255.0f;
    u->blend_color[1] = ((bc >> 8) & 0xFF) / 255.0f;
    u->blend_color[2] = (bc & 0xFF) / 255.0f;
    u->blend_color[3] = (bc >> 24) / 255.0f;
    u->cblend = r300_reg(st, RB3D_CBLEND);
    u->ablend = r300_reg(st, RB3D_ABLEND);
    u->chanmask = r300_reg(st, RB3D_COLOR_CHANNEL_MASK) & 0xF;
    u->alpha_func = r300_reg(st, FG_ALPHA_FUNC);
    for (int n = 0; n < 4; n++) {
        u->out_sel[n] = (outfmt >> (8 + 2 * n)) & 3;
    }
    u->rt_swap32 = ((r300_reg(st, RB3D_COLORPITCH0) >> 19) & 3) == 2;
    u->rt_endian = (r300_reg(st, RB3D_COLORPITCH0) >> 19) & 3;
    u->clip_rule = r300_reg(st, SC_CLIP_RULE) & 0xFFFF;
    for (int i = 0; i < 4; i++) {
        uint32_t tl = r300_reg(st, SC_CLIPRECT_TL_0 + 8 * i);
        uint32_t br = r300_reg(st, SC_CLIPRECT_TL_0 + 8 * i + 4);
        u->cliprect[i][0] = (int32_t)(tl & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][1] = (int32_t)((tl >> 13) & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][2] = (int32_t)(br & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][3] = (int32_t)((br >> 13) & 0x1FFF) - SC_COORD_BIAS;
    }
    u->rop = r300_reg(st, RB3D_ROPCNTL);
    /* Fog: colours and the constant factor are 10-bit fractions. */
    for (int c = 0; c < 3; c++) {
        u->fog_color[c] = (r300_reg(st, FG_FOG_COLOR_R + 4 * c) & 0x3FF) / 1023.0f;
    }
    u->fog_color[3] = (r300_reg(st, FG_FOG_FACTOR) & 0x3FF) / 1023.0f;
    u->fog_blend = r300_reg(st, FG_FOG_BLEND) & 7;
    u->depth_src = r300_reg(st, FG_DEPTH_SRC) & 1;
    /*
     * Polygon offset (SU_POLY_OFFSET_*): scale multiplies the depth slope
     * per 1/12-pixel subpixel and offset is in units of 2^-24 of the depth
     * range.  Mesa programs a 16-bit Z buffer's scale 16 times larger, so
     * the slope there is in 2^-20 units; set_depth adjusts for Z16.
     */
    u->poly_en = r300_reg(st, SU_POLY_OFFSET_ENABLE) & 7;
    for (int i = 0; i < 2; i++) {
        u->poly_offset[2 * i] = bits_to_float(r300_reg(st, SU_POLY_OFFSET_FRONT_SCALE + 8 * i)) / 12.0f;
        u->poly_offset[2 * i + 1] = bits_to_float(r300_reg(st, SU_POLY_OFFSET_FRONT_SCALE + 8 * i + 4)) /
                                    16777216.0f;
    }
}

static uint32_t pot_ceil(uint32_t v)
{
    uint32_t p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

void r300_tex_level_dims(const R300TexDesc *td, uint32_t l, uint32_t *w,
                         uint32_t *h, uint32_t *d)
{
    *w = td->width >> l ? td->width >> l : 1;
    *h = td->height >> l ? td->height >> l : 1;
    if (td->dim == R300_TEXDIM_CUBE) {
        *d = 6;
    } else if (td->dim == R300_TEXDIM_3D) {
        *d = td->depth >> l ? td->depth >> l : 1;
    } else {
        *d = 1;
    }
}

/* The hardware's addressing of linear (untiled) textures: levels one
 * after another, power-of-two heights for mipmapped, 3D and cube
 * textures.  Level 0 rows are 32-byte aligned as in Mesa's
 * r300_texture_desc.c; the levels below it are 16-byte aligned.  Tiger's
 * driver puts them there (a 64x64 RGB565 chain: 8x8 at +0x2a00, 4x4 at
 * +0x2a80, 2x2 at +0x2ac0, all with 16-byte rows; RGBA8 4x4 and smaller
 * the same), and with 32-byte rows every level under 16 bytes wide read
 * other texels: GLQuake's distant floors came out pink and green.
 * Compressed textures follow the same rule per row of 4x4 blocks (a
 * 512x512 DXT1 chain: the 8x8 level's two 8-byte blocks in a 16-byte
 * row, the 4x4 level 32 bytes later); UT2004's distant terrain showed a
 * green and black band with 32-byte rows. */
void r300_tex_layout(R300TexDesc *td, uint32_t bpp, bool dxt, bool pitch_en)
{
    bool pot_rows = td->levels > 1 || td->dim != R300_TEXDIM_2D;
    uint32_t off = 0;

    for (uint32_t l = 0; l < td->levels; l++) {
        uint32_t w, h, d, pitch, rows;

        r300_tex_level_dims(td, l, &w, &h, &d);
        if (pot_rows) {
            h = pot_ceil(h);
        }
        uint32_t row = dxt ? ((w + 3) / 4) * bpp : w * bpp;
        pitch = l == 0 ? (row + 31) & ~31u : (row + 15) & ~15u;
        rows = dxt ? (h + 3) / 4 : h;
        if (l == 0 && pitch_en) {
            pitch = td->pitch_bytes;
        }
        td->lvl_off[l] = off;
        td->lvl_pitch[l] = pitch;
        td->lvl_rows[l] = rows;
        off += pitch * rows * d;
    }
    td->size_bytes = off;
}

static float un_bits(uint32_t v, unsigned sh, unsigned n)
{
    return (float)((v >> sh) & ((1u << n) - 1)) / (float)((1u << n) - 1);
}

static float sn_bits(uint32_t v, unsigned sh, unsigned n)
{
    int32_t x = (int32_t)(((v >> sh) & ((1u << n) - 1)) << (32 - n)) >> (32 - n);
    return fmaxf((float)x / (float)((1u << (n - 1)) - 1), -1.0f);
}

void r300_border_color(uint32_t fmt, uint32_t v, float c[4])
{
    c[0] = c[1] = c[2] = c[3] = 0.0f;
    switch (fmt & 0x1F) {
    case 0x00: c[0] = un_bits(v, 0, 8); break;
    case 0x01: c[0] = un_bits(v, 0, 16); break;
    case 0x02: c[0] = un_bits(v, 0, 4); c[1] = un_bits(v, 4, 4); break;
    case 0x03: c[0] = un_bits(v, 0, 8); c[1] = un_bits(v, 8, 8); break;
    case 0x04: c[0] = un_bits(v, 0, 16); c[1] = un_bits(v, 16, 16); break;
    case 0x05: c[0] = un_bits(v, 0, 2); c[1] = un_bits(v, 2, 3); c[2] = un_bits(v, 5, 3); break;
    case 0x06: c[0] = un_bits(v, 0, 5); c[1] = un_bits(v, 5, 6); c[2] = un_bits(v, 11, 5); break;
    case 0x07: c[0] = un_bits(v, 0, 5); c[1] = un_bits(v, 5, 5); c[2] = un_bits(v, 10, 6); break;
    case 0x08: c[0] = un_bits(v, 0, 10); c[1] = un_bits(v, 10, 11); c[2] = un_bits(v, 21, 11); break;
    case 0x09: c[0] = un_bits(v, 0, 11); c[1] = un_bits(v, 11, 11); c[2] = un_bits(v, 22, 10); break;
    case 0x0A:
        for (int i = 0; i < 4; i++) c[i] = un_bits(v, 4 * i, 4);
        break;
    case 0x0B:
        for (int i = 0; i < 3; i++) c[i] = un_bits(v, 5 * i, 5);
        c[3] = un_bits(v, 15, 1);
        break;
    case 0x0D:
        for (int i = 0; i < 3; i++) c[i] = un_bits(v, 10 * i, 10);
        c[3] = un_bits(v, 30, 2);
        break;
    case 0x0F: case 0x10: case 0x11:        /* DXT: B8G8R8A8 */
        c[0] = un_bits(v, 16, 8); c[1] = un_bits(v, 8, 8);
        c[2] = un_bits(v, 0, 8); c[3] = un_bits(v, 24, 8);
        break;
    case 0x12:                              /* signed formats: R8G8B8A8_SNORM */
        for (int i = 0; i < 4; i++) c[i] = sn_bits(v, 8 * i, 8);
        break;
    case 0x16: case 0x17:                   /* R16G16_SNORM */
        c[0] = sn_bits(v, 0, 16); c[1] = sn_bits(v, 16, 16);
        break;
    case 0x18: case 0x19:                   /* R16G16_FLOAT */
        c[0] = half_to_float(v & 0xFFFF); c[1] = half_to_float(v >> 16);
        break;
    case 0x1B:
        c[0] = bits_to_float(v);
        break;
    default:                                /* 8888 */
        for (int i = 0; i < 4; i++) c[i] = un_bits(v, 8 * i, 8);
        break;
    }
}

static void set_textures(const R300State *st, R300DrawPacket *pkt)
{
    uint32_t enable = r300_reg(st, TX_ENABLE);

    for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
        R300TexDesc *t = &pkt->tex[k];
        R300FSUniforms *u = &pkt->uniforms;
        uint32_t f0 = r300_reg(st, TX_FORMAT0_0 + 4 * k);
        uint32_t f1 = r300_reg(st, TX_FORMAT1_0 + 4 * k);
        uint32_t f2 = r300_reg(st, TX_FORMAT2_0 + 4 * k);
        uint32_t off = r300_reg(st, TX_OFFSET_0 + 4 * k);
        uint32_t fmt = f1 & 0x1F;

        memset(t, 0, sizeof(*t));
        /* swizzle: R 14:12, G 17:15, B 20:18, A 11:9 */
        u->tex_swz[k][0] = (f1 >> 12) & 7;
        u->tex_swz[k][1] = (f1 >> 15) & 7;
        u->tex_swz[k][2] = (f1 >> 18) & 7;
        u->tex_swz[k][3] = (f1 >> 9) & 7;
        /*
         * YUV_TO_RGB: the card hands out the converted colour as is.  Leopard's
         * DVD Player sets R = G = B = X here and gets colour on the real card;
         * r300_yuv leaves B, G, R in X, Y, Z.
         */
        if ((f1 >> 22) & 1 && fmt >= 0x13 && fmt <= 0x15) {
            u->tex_swz[k][0] = 2;
            u->tex_swz[k][1] = 1;
            u->tex_swz[k][2] = 0;
        }
        if (!(enable & (1u << k))) {
            continue;
        }
        unsigned bpp, decode = 0;
        bool dxt = false;
        switch (fmt) {
        case 0x0:                       /* X8 */
            t->kind = R300_TEXK_R8; bpp = 1;
            break;
        case 0x3:                       /* Y8X8: big-endian halfwords */
            t->kind = R300_TEXK_RG8; bpp = 2; decode = 2;
            break;
        case 0x1: case 0x6: case 0x7: case 0xA: case 0xB:
            t->kind = R300_TEXK_CONVERT16; bpp = 2;
            break;
        case 0xC:                       /* W8Z8Y8X8 */
            /* VRAM holds the guest CPU's big-endian words (see r300_tex). */
            t->kind = R300_TEXK_RGBA8; bpp = 4; decode = (off & 3) == 0;
            break;
        /* The card hands the shader DXT1 texels as B,G,R,A and DXT3/5 as
         * A,B,G,R (Tiger's swizzles R=Z,G=Y,B=X and R=W,G=Z,B=Y,A=X);
         * the host decodes to R,G,B,A, so reorder before the swizzle. */
        case 0xF:  t->kind = R300_TEXK_DXT1; bpp = 8;  dxt = true; decode = 3; break;   /* per block */
        case 0x10: t->kind = R300_TEXK_DXT3; bpp = 16; dxt = true; decode = 1; break;
        case 0x11: t->kind = R300_TEXK_DXT5; bpp = 16; dxt = true; decode = 1; break;
        default:
            bpp = r300_tex_raw_bpp(fmt);
            if (!bpp) {
                pkt->warn |= R300_WARN_TEXFMT;
                continue;
            }
            t->kind = R300_TEXK_RAW;
            t->view_bpp = bpp < 4 ? 4 : bpp;
            break;
        }
        t->bound = true;
        t->gpu_addr = off & ~0x1Fu;
        t->endian = off & 3;
        t->width = (f0 & 0x7FF) + 1;
        t->height = ((f0 >> 11) & 0x7FF) + 1;
        t->dim = (f1 >> 25) & 3;
        if (t->dim > R300_TEXDIM_CUBE) {
            t->dim = R300_TEXDIM_2D;
        }
        t->depth = t->dim == R300_TEXDIM_3D ? 1u << ((f0 >> 22) & 0xF) : 1;
        {
            /* NUM_LEVELS counts the levels below the base, up to 1x1x1 */
            uint32_t big = t->width > t->height ? t->width : t->height, chain = 1;
            if (t->depth > big) {
                big = t->depth;
            }
            while (big >> chain) {
                chain++;
            }
            t->levels = ((f0 >> 26) & 0xF) + 1;
            if (t->levels > chain) {
                t->levels = chain;
            }
            if (t->levels > R300_TEX_MAX_LEVELS) {
                t->levels = R300_TEX_MAX_LEVELS;
            }
        }
        if (f0 >> 31) {                 /* TX_PITCH_EN: level 0 pitch in texels */
            uint32_t texels = (f2 & 0x3FFF) + 1;
            t->pitch_bytes = dxt ? ((texels + 3) / 4) * bpp : texels * bpp;
        }
        r300_tex_layout(t, bpp, dxt, f0 >> 31);
        t->pitch_bytes = t->lvl_pitch[0];
        t->format = fmt;
        t->filter0 = r300_reg(st, TX_FILTER0_0 + 4 * k);
        t->filter1 = r300_reg(st, TX_FILTER1_0 + 4 * k);
        u->tex_info[k][0] = 1;
        /* RAW: the uint view's elements per row, as the renderer binds it */
        u->tex_info[k][1] = t->kind == R300_TEXK_RAW ? t->pitch_bytes / t->view_bpp : decode;
        u->tex_info[k][2] = t->width;
        u->tex_info[k][3] = t->height;

        r300_border_color(fmt, r300_reg(st, TX_BORDER_COLOR_0 + 4 * k), u->tex_border[k]);
        {
            int32_t bias = (int32_t)(((t->filter1 >> 3) & 0x3FF) << 22) >> 22;
            uint32_t minl = (t->filter0 >> 17) & 0xF;
            u->tex_lod[k][0] = bias / 32.0f;
            u->tex_lod[k][1] = (float)(minl < t->levels ? minl : t->levels - 1);
            u->tex_lod[k][2] = (float)(t->levels - 1);
            u->tex_lod[k][3] = (float)((t->filter0 >> 13) & 3);
        }
        u->tex_dim[k][0] = t->dim;
        u->tex_dim[k][1] = t->depth;
        u->tex_dim[k][2] = t->pitch_bytes;
        u->tex_dim[k][3] = ((f1 >> 8) & 1) | ((f1 >> 6) & 2) | ((f1 >> 4) & 4) |
                           ((f1 >> 2) & 8) | ((f1 >> 21) & 1 ? R300_TEXF_GAMMA : 0) |
                           (t->levels > 1 || t->dim != R300_TEXDIM_2D ? R300_TEXF_POT_ROWS : 0);
    }
}

static void set_depth(const R300State *st, R300DrawPacket *pkt)
{
    uint32_t cntl = r300_reg(st, ZB_CNTL);
    uint32_t fmt = r300_reg(st, ZB_FORMAT) & 0xF;
    uint32_t pitch = r300_reg(st, ZB_DEPTHPITCH);
    R300DepthDesc *z = &pkt->depth;
    R300FSUniforms *u = &pkt->uniforms;

    memset(z, 0, sizeof(*z));
    /* STENCIL_ENABLE or Z_ENABLE; Z_WRITE_ENABLE alone does nothing. */
    if (!(cntl & 3)) {
        return;
    }
    z->attach = true;
    z->gpu_addr = r300_reg(st, ZB_DEPTHOFFSET) & ~0x1Fu;
    z->pitch = pitch & 0x3FFC;
    z->format = fmt;
    z->bpp = fmt == 2 ? 4 : 2;
    if (fmt > 2) {
        pkt->warn |= R300_WARN_DEPTH;       /* reserved format: as Z24S8 */
        z->bpp = 4;
    } else if (fmt == 1) {
        pkt->warn |= R300_WARN_DEPTH;       /* 13E3 float: kept as 16-bit unorm */
    }
    u->zinfo[0] = cntl;
    u->zinfo[1] = r300_reg(st, ZB_ZSTENCILCNTL);
    u->zinfo[2] = r300_reg(st, ZB_STENCILREFMASK);
    u->zinfo[3] = ((pitch >> 19) & R300_ZFMT_ENDIAN_MASK) |
                  (z->bpp == 2 ? R300_ZFMT_Z16 : 0);
    if (z->bpp == 2) {
        u->poly_offset[0] /= 16.0f;
        u->poly_offset[2] /= 16.0f;
    }
}

static void viewport_xform(const R300State *st, const R300DrawPacket *pkt,
                           const float c[4], float p[4])
{
    uint32_t vte = r300_reg(st, VAP_VTE_CNTL);
    float vp[6];
    float W = pkt->rt_width, H = pkt->rt_height;

    for (int i = 0; i < 6; i++) {
        bool en = (vte >> i) & 1;
        vp[i] = en ? bits_to_float(r300_reg(st, SE_VPORT_XSCALE + 4 * i))
                   : (i % 2 ? 0.0f : 1.0f);
    }
    /* vp: xscale, xoffset, yscale, yoffset, zscale, zoffset */
    if (!((vte >> 8) & 1)) {
        /* Perspective divide by the VTE: stay linear in clip space so Metal
         * clips and interpolates exactly as the card would. */
        float w = c[3];
        p[0] = (2.0f * vp[0] / W) * c[0] + (2.0f * vp[1] / W - 1.0f) * w;
        p[1] = -(2.0f * vp[2] / H) * c[1] + (1.0f - 2.0f * vp[3] / H) * w;
        p[2] = vp[4] * c[2] + vp[5] * w;
        p[3] = w;
    } else {
        float wx = vp[0] * c[0] + vp[1], wy = vp[2] * c[1] + vp[3];
        p[0] = 2.0f * wx / W - 1.0f;
        p[1] = 1.0f - 2.0f * wy / H;
        p[2] = vp[4] * c[2] + vp[5];
        p[3] = 1.0f;
    }
}

/*
 * Emit a primitive list as expanded triangles/lines/points.  With prov,
 * also the flat-shading candidates of each output primitive (the
 * FIRST, SECOND, THIRD and LAST vertex of the source primitive, per
 * GA_COLOR_CONTROL.PROVOKING_VERTEX) and, for triangles, the edge flags
 * polygon-mode lines honour (bit 0: edge 0-1, bit 1: 1-2, bit 2: 2-0;
 * the diagonals the card adds inside quads and polygons are not edges).
 */
uint32_t r300_assemble_prov(unsigned prim, uint32_t n, uint32_t *list,
                            uint32_t *cls, R300Prov *prov)
{
    uint32_t m = 0, np = 0;

#define PUT(a) (list[m++] = (a))
#define PROV(f, s2, t, l, e) do { if (prov) { \
        prov[np].v[0] = (f); prov[np].v[1] = (s2); prov[np].v[2] = (t); \
        prov[np].v[3] = (l); prov[np].edges = (e); } np++; } while (0)
    switch (prim) {
    case 1:                                     /* points */
        *cls = 2;
        for (uint32_t i = 0; i < n; i++) { PUT(i); PROV(i, i, i, i, 0); }
        break;
    case 2:                                     /* lines */
        *cls = 1;
        for (uint32_t i = 0; i + 1 < n; i += 2) {
            PUT(i); PUT(i + 1); PROV(i, i + 1, i + 1, i + 1, 1);
        }
        break;
    case 3:                                     /* line strip */
    case 12:                                    /* line loop */
        *cls = 1;
        for (uint32_t i = 0; i + 1 < n; i++) {
            PUT(i); PUT(i + 1); PROV(i, i + 1, i + 1, i + 1, 1);
        }
        if (prim == 12 && n > 2) { PUT(n - 1); PUT(0); PROV(n - 1, 0, 0, 0, 1); }
        break;
    case 4:                                     /* triangles */
    case 7:                                     /* triangles with wFlags */
        *cls = 0;
        for (uint32_t i = 0; i + 2 < n; i += 3) {
            PUT(i); PUT(i + 1); PUT(i + 2); PROV(i, i + 1, i + 2, i + 2, 7);
        }
        break;
    case 5:                                     /* fan */
        *cls = 0;
        for (uint32_t i = 1; i + 1 < n; i++) {
            PUT(0); PUT(i); PUT(i + 1); PROV(0, i, i + 1, i + 1, 7);
        }
        break;
    case 15:                                    /* polygon */
        *cls = 0;
        for (uint32_t i = 1; i + 1 < n; i++) {
            PUT(0); PUT(i); PUT(i + 1);
            PROV(0, 1, 2, n - 1, (i == 1 ? 1 : 0) | 2 | (i + 2 == n ? 4 : 0));
        }
        break;
    case 6:                                     /* strip */
        *cls = 0;
        for (uint32_t i = 0; i + 2 < n; i++) {
            if (i & 1) { PUT(i + 1); PUT(i); PUT(i + 2); }
            else       { PUT(i); PUT(i + 1); PUT(i + 2); }
            PROV(i, i + 1, i + 2, i + 2, 7);
        }
        break;
    case 13:                                    /* quads */
        *cls = 0;
        for (uint32_t i = 0; i + 3 < n; i += 4) {
            PUT(i); PUT(i + 1); PUT(i + 2); PROV(i, i + 1, i + 2, i + 3, 3);
            PUT(i); PUT(i + 2); PUT(i + 3); PROV(i, i + 1, i + 2, i + 3, 6);
        }
        break;
    case 14:                                    /* quad strip */
        *cls = 0;
        for (uint32_t i = 0; i + 3 < n; i += 2) {
            PUT(i); PUT(i + 1); PUT(i + 3); PROV(i, i + 1, i + 2, i + 3, 3);
            PUT(i); PUT(i + 3); PUT(i + 2); PROV(i, i + 1, i + 2, i + 3, 6);
        }
        break;
    default:
        return 0;
    }
#undef PROV
#undef PUT
    return m;
}

uint32_t r300_assemble(unsigned prim, uint32_t n, uint32_t *list,
                       uint32_t *cls)
{
    return r300_assemble_prov(prim, n, list, cls, NULL);
}

/* ---- primitive emission --------------------------------------------- */

typedef struct Build {
    const R300State *st;
    R300DrawPacket *pkt;
    const Layout *lay;
    const Route *route;
    float (*outs)[R300_PVS_NUM_OUTPUTS][4];     /* per source vertex */
    R300Vertex *xv;                             /* per source vertex */
    uint32_t n;

    uint32_t colctl;            /* GA_COLOR_CONTROL */
    bool fancy_colors;          /* flat/solid shading or two-sided colours */
    bool back_colors;           /* output colours 2/3 are back-face 0/1 */
    float solid[4];             /* GA_SOLID_* */
    uint32_t poly_mode;         /* GA_POLY_MODE, 0 when off */
    uint32_t cull;              /* SU_CULL_MODE for CPU culling, else 0 */
    float line_hw;              /* half line width, pixels */
    uint32_t line_end;          /* GA_LINE_CNTL.END_TYPE */
    float stip_scale;           /* GA_LINE_STIPPLE_CONFIG.STIPPLE_SCALE */
    uint32_t stip_reset;
    float stip_acc;             /* stipple accumulator, subpixels */
    float point_hw, point_hh;   /* GA_POINT_SIZE, half, pixels */
    float point_min, point_max; /* GA_POINT_MINMAX diameters, pixels */
    bool point_psize;           /* per-vertex point size */

    R300Vertex *tri, *line;
    uint32_t ntri, nline, cap_tri, cap_line;
} Build;

static R300Vertex *push(R300Vertex **buf, uint32_t *n, uint32_t *cap, uint32_t k)
{
    if (*n + k > *cap) {
        *cap = (*n + k) * 2 + 16;
        *buf = realloc(*buf, sizeof(R300Vertex) * *cap);
    }
    *n += k;
    return *buf + *n - k;
}

/* Colours of source vertex i as the rasterizer sees them on a face. */
static void face_colors(Build *b, uint32_t i, bool back, float cols[4][4])
{
    vtx_colors(b->lay, b->outs[i], cols);
    if (back && b->back_colors) {
        /* Two-sided lighting: the setup engine picks colours 2 and 3 for
         * back faces (AMD R5xx guide 7.5). */
        for (int c = 0; c < 2; c++) {
            if (b->lay->color[2 + c] >= 0) {
                memcpy(cols[c], cols[2 + c], sizeof(cols[c]));
            }
        }
    }
}

/*
 * One output vertex from source vertex i: prov is its primitive's
 * provoking-vertex candidates (NULL: none), back its facing, sprite the
 * rasterizer's generated (s, t) or NULL.
 */
static void emit_vertex(Build *b, uint32_t i, const R300Prov *prov, bool back,
                        const float *sprite, R300Vertex *o)
{
    *o = b->xv[i];
    if (!b->fancy_colors && !sprite) {
        return;
    }
    float cols[4][4];
    face_colors(b, i, back, cols);
    if (b->fancy_colors) {
        float pc[4][4];
        uint32_t pv = prov ? prov->v[(b->colctl >> 16) & 3] : i;
        face_colors(b, pv < b->n ? pv : i, back, pc);
        for (int c = 0; c < 4; c++) {
            unsigned rgb = (b->colctl >> (4 * c)) & 3, a = (b->colctl >> (4 * c + 2)) & 3;
            for (int k = 0; k < 4; k++) {
                unsigned mode = k < 3 ? rgb : a;
                if (mode == 0) {
                    cols[c][k] = b->solid[k];
                } else if (mode == 1) {
                    cols[c][k] = pc[c][k];
                }
            }
        }
    }
    rs_values(b->st, b->lay, b->route, b->outs[i], (const float (*)[4])cols,
              sprite, o->v);
}

static void ndc_px(const Build *b, const float *p, float *x, float *y)
{
    float w = p[3] != 0.0f ? p[3] : 1.0f;
    *x = (p[0] / w + 1.0f) * b->pkt->rt_width * 0.5f;
    *y = (1.0f - p[1] / w) * b->pkt->rt_height * 0.5f;
}

/* Replace the x, y of clip-space p by pixel position (x, y), keeping w. */
static void px_ndc(const Build *b, float x, float y, float *p)
{
    float w = p[3] != 0.0f ? p[3] : 1.0f;
    p[0] = (2.0f * x / b->pkt->rt_width - 1.0f) * w;
    p[1] = (1.0f - 2.0f * y / b->pkt->rt_height) * w;
}

/*
 * A point: a screen-aligned square GA_POINT_SIZE wide (width in the high
 * half, height in the low half, in sixths of a pixel), or the vertex's
 * point size clamped to GA_POINT_MINMAX, with texture coordinates S0..S1
 * left to right and T0..T1 bottom to top.  Apple's compositor presents
 * the screen as one such point covering the whole frame buffer.
 */
static void emit_point(Build *b, uint32_t i, const R300Prov *prov, bool back)
{
    const R300State *st = b->st;
    const float *p = b->xv[i].pos;
    float hw = b->point_hw, hh = b->point_hh;
    float s0 = bits_to_float(r300_reg(st, GA_POINT_S0));
    float t0 = bits_to_float(r300_reg(st, GA_POINT_T0));
    float s1 = bits_to_float(r300_reg(st, GA_POINT_S1));
    float t1 = bits_to_float(r300_reg(st, GA_POINT_T1));
    static const int corner[6] = { 0, 1, 2, 0, 2, 3 };
    float cx, cy;

    if (b->point_psize) {
        float d = b->outs[i][b->lay->psize][0];
        d = fminf(fmaxf(d, b->point_min), b->point_max);
        hw = hh = d * 0.5f;
    }
    ndc_px(b, p, &cx, &cy);
    /* corners: top-left, top-right, bottom-right, bottom-left */
    float x[4] = { cx - hw, cx + hw, cx + hw, cx - hw };
    float y[4] = { cy - hh, cy - hh, cy + hh, cy + hh };
    float sc[4] = { s0, s1, s1, s0 };
    float tc[4] = { t1, t1, t0, t0 };
    R300Vertex *o = push(&b->tri, &b->ntri, &b->cap_tri, 6);

    for (int k = 0; k < 6; k++) {
        int c = corner[k];
        float st2[2] = { sc[c], tc[c] };

        emit_vertex(b, i, prov, back, st2, &o[k]);
        /* screen-space square at the vertex's depth */
        o[k].pos[2] = p[2] / (p[3] != 0.0f ? p[3] : 1.0f);
        o[k].pos[3] = 1.0f;
        px_ndc(b, x[c], y[c], o[k].pos);
    }
}

/*
 * A line from source vertex i0 to i1.  The rasterizer stuffs (s, t) for
 * the texture components past the vertex's own: s runs GA_LINE_S0 to S1
 * along the line, or with GA_LINE_STIPPLE_CONFIG.STIPPLE_SCALE set, is the
 * stipple accumulator (subpixels, 12 a pixel) times the scale, which the
 * driver's program looks up in its stipple pattern.  Lines wider than a
 * pixel become quads: offset across the major axis (END_TYPE horizontal,
 * vertical, square), or perpendicular to the line (computed).
 */
static void emit_line(Build *b, uint32_t i0, uint32_t i1, const R300Prov *prov,
                      bool back)
{
    const float *p0 = b->xv[i0].pos, *p1 = b->xv[i1].pos;
    float x0, y0, x1, y1;
    float sp0[2] = { 0, 0 }, sp1[2] = { 0, 0 };

    ndc_px(b, p0, &x0, &y0);
    ndc_px(b, p1, &x1, &y1);
    float dx = x1 - x0, dy = y1 - y0, len = sqrtf(dx * dx + dy * dy);
    if (b->stip_scale != 0.0f) {
        if (b->stip_reset == 1) {
            b->stip_acc = 0.0f;
        }
        sp0[0] = b->stip_acc * b->stip_scale;
        b->stip_acc += len * 12.0f;
        sp1[0] = b->stip_acc * b->stip_scale;
    } else {
        sp0[0] = bits_to_float(r300_reg(b->st, GA_LINE_S0));
        sp1[0] = bits_to_float(r300_reg(b->st, GA_LINE_S1));
    }

    if (b->line_hw <= 0.75f) {
        R300Vertex *o = push(&b->line, &b->nline, &b->cap_line, 2);
        emit_vertex(b, i0, prov, back, sp0, &o[0]);
        emit_vertex(b, i1, prov, back, sp1, &o[1]);
        return;
    }
    float ox, oy, hw = b->line_hw;
    if (b->line_end == 3 && len > 0.0f) {
        ox = -dy / len * hw;
        oy = dx / len * hw;
    } else if (fabsf(dx) >= fabsf(dy)) {
        ox = 0.0f; oy = hw;             /* x-major: widen vertically */
    } else {
        ox = hw; oy = 0.0f;
    }
    R300Vertex q[4];
    emit_vertex(b, i0, prov, back, sp0, &q[0]);
    q[1] = q[0];
    emit_vertex(b, i1, prov, back, sp1, &q[2]);
    q[3] = q[2];
    px_ndc(b, x0 - ox, y0 - oy, q[0].pos);
    px_ndc(b, x0 + ox, y0 + oy, q[1].pos);
    px_ndc(b, x1 + ox, y1 + oy, q[2].pos);
    px_ndc(b, x1 - ox, y1 - oy, q[3].pos);
    static const int tri[6] = { 0, 1, 2, 0, 2, 3 };
    R300Vertex *o = push(&b->tri, &b->ntri, &b->cap_tri, 6);
    for (int k = 0; k < 6; k++) {
        o[k] = q[tri[k]];
    }
}

/* Winding on screen, as Metal and SU_CULL_MODE judge it. */
static bool tri_ccw(const Build *b, uint32_t a, uint32_t c, uint32_t d)
{
    const float *p[3] = { b->xv[a].pos, b->xv[c].pos, b->xv[d].pos };
    float x[3], y[3];
    for (int k = 0; k < 3; k++) {
        float w = p[k][3] != 0.0f ? p[k][3] : 1.0f;
        x[k] = p[k][0] / w;
        y[k] = p[k][1] / w;
    }
    float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    /* a vertex behind the eye flips the projected winding */
    if ((p[0][3] < 0) ^ (p[1][3] < 0) ^ (p[2][3] < 0)) {
        area = -area;
    }
    return area > 0.0f;
}

static void emit_tri(Build *b, const uint32_t *ix, const R300Prov *prov)
{
    bool front = tri_ccw(b, ix[0], ix[1], ix[2]) == r300_front_ccw(r300_reg(b->st, SU_CULL_MODE));
    bool back = !front;

    if ((front && (b->cull & R300_CULL_FRONT)) || (back && (b->cull & R300_CULL_BACK))) {
        return;
    }
    /* GA_POLY_MODE.FRONT_PTYPE / BACK_PTYPE: 0 points, 1 lines, 2 triangles */
    unsigned ptype = !b->poly_mode ? 2 : front ? (b->poly_mode >> 4) & 7
                                               : (b->poly_mode >> 7) & 7;
    if (ptype == 0) {
        for (int k = 0; k < 3; k++) {
            emit_point(b, ix[k], prov, back);
        }
    } else if (ptype == 1) {
        for (int k = 0; k < 3; k++) {
            if (!prov || (prov->edges >> k) & 1) {
                emit_line(b, ix[k], ix[(k + 1) % 3], prov, back);
            }
        }
    } else {
        R300Vertex *o = push(&b->tri, &b->ntri, &b->cap_tri, 3);
        for (int k = 0; k < 3; k++) {
            emit_vertex(b, ix[k], prov, back, NULL, &o[k]);
        }
    }
}

/* S3.12 fixed point (GA_SOLID_*). */
static float s312(uint32_t v)
{
    return (int16_t)(v & 0xFFFF) / 4096.0f;
}

/*
 * ---- Vertex shading on the host GPU ----
 *
 * The CPU interpreter (r300_pvs_run) plus the per-vertex work after it --
 * viewport transform, colour and texture interpolant routing, fog, user
 * clip distances -- as one generated vertex shader, for the draws where
 * nothing after the vertex program needs whole primitives: filled
 * triangles with smooth colours.  Everything else stays on the CPU.
 */

/* Which inputs the shader reads, packed in this order per vertex. */
typedef struct GpuVs {
    char *glsl;
    uint32_t id;
    uint32_t in_used;
    uint32_t *key;
    unsigned klen;
} GpuVs;

#define GPU_VS_SLOTS 32
static GpuVs gpu_vs_memo[GPU_VS_SLOTS];     /* most recently used first */
static uint32_t gpu_vs_next_id = 1;

static void key_push(uint32_t **k, unsigned *n, unsigned *cap, uint32_t v)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 256;
        *k = realloc(*k, *cap * sizeof(**k));
    }
    (*k)[(*n)++] = v;
}

static void gpu_vs_emit_post(R300Sb *sb, const R300State *st, const Layout *lay,
                             const Route *route, bool persp, bool fog,
                             uint32_t fogsel, uint32_t clip)
{
    static const char comp[] = "xyzw";

    r300_sb_printf(sb, "    vec4 cp = o%d;\n    vec4 p;\n", lay->pos);
    if (persp) {
        r300_sb_printf(sb,
            "    p = vec4((2.0 * vs.vp0.x / vs.vp1.z) * cp.x + (2.0 * vs.vp0.y / vs.vp1.z - 1.0) * cp.w,\n"
            "             -(2.0 * vs.vp0.z / vs.vp1.w) * cp.y + (1.0 - 2.0 * vs.vp0.w / vs.vp1.w) * cp.w,\n"
            "             vs.vp1.x * cp.z + vs.vp1.y * cp.w, cp.w);\n");
    } else {
        r300_sb_printf(sb,
            "    p = vec4(2.0 * (vs.vp0.x * cp.x + vs.vp0.y) / vs.vp1.z - 1.0,\n"
            "             1.0 - 2.0 * (vs.vp0.z * cp.y + vs.vp0.w) / vs.vp1.w,\n"
            "             vs.vp1.x * cp.z + vs.vp1.y, 1.0);\n");
    }
    r300_sb_printf(sb, "    gl_Position = vec4(p.xy + ms.xy * p.w, p.zw);\n");
    for (int c = 0; c < 4; c++) {
        if (lay->color[c] >= 0) {
            r300_sb_printf(sb, "    vec4 col%d = o%d;\n", c, lay->color[c]);
        } else {
            r300_sb_printf(sb, "    vec4 col%d = vec4(0.0);\n", c);
        }
    }
    for (unsigned k = 0; k < R300_NUM_VARYINGS; k++) {
        char e[4][24];
        if (k >= route->nvary) {
            r300_sb_printf(sb, "    v%u = vec4(0.0);\n", k);
            continue;
        }
        uint32_t ip = r300_reg(st, RS_IP_0 + 4 * route->vary[k].ip);
        for (unsigned i = 0; i < 4; i++) {
            if (route->vary[k].is_tex) {
                unsigned sel = (ip >> (13 + 3 * i)) & 7, sc = (ip & 0x3F) + sel;
                if (sel < 4 && sc < lay->nscal) {
                    snprintf(e[i], sizeof(e[i]), "o%u.%c", lay->scal_slot[sc],
                             comp[lay->scal_comp[sc]]);
                } else {
                    snprintf(e[i], sizeof(e[i]), sel == 5 ? "1.0" : "0.0");
                }
            } else {
                static const int8_t tbl[16][4] = {
                    [0] = { 0, 1, 2, 3 },  [1] = { 0, 1, 2, -1 }, [2] = { 0, 1, 2, -2 },
                    [4] = { -1, -1, -1, 3 }, [5] = { -1, -1, -1, -1 }, [6] = { -1, -1, -1, -2 },
                    [8] = { -2, -2, -2, 3 }, [9] = { -2, -2, -2, -1 }, [10] = { -2, -2, -2, -2 },
                };
                unsigned cp = (ip >> 6) & 7;
                int t = tbl[(ip >> 9) & 0xF][i];
                if (t >= 0 && cp < 4) {
                    snprintf(e[i], sizeof(e[i]), "col%u.%c", cp, comp[t]);
                } else {
                    snprintf(e[i], sizeof(e[i]), t == -2 ? "1.0" : "0.0");
                }
            }
        }
        r300_sb_printf(sb, "    v%u = vec4(%s, %s, %s, %s);\n", k, e[0], e[1], e[2], e[3]);
    }
    if (fog) {
        const char *src;
        char buf[32];
        if (fogsel < 4) {
            if (lay->color[fogsel] >= 0) {
                snprintf(buf, sizeof(buf), "o%d.w", lay->color[fogsel]);
                src = buf;
            } else {
                src = "0.0";
            }
        } else if (fogsel == 4) {
            snprintf(buf, sizeof(buf), "o%d.w", lay->pos);
            src = buf;
        } else {
            src = "p.z / (p.w != 0.0 ? p.w : 1.0)";
        }
        r300_sb_printf(sb, "    aux = vec4(%s * vs.fogp.x + vs.fogp.y, 0.0, 0.0, 0.0);\n", src);
    } else {
        r300_sb_printf(sb, "    aux = vec4(0.0);\n");
    }
    if (clip & 0x3F) {
        for (int k = 0; k < 6; k++) {
            if ((clip >> k) & 1) {
                r300_sb_printf(sb, "    gl_ClipDistance[%d] = dot(cp, vs.ucp[%d]);\n", k, k);
            } else {
                r300_sb_printf(sb, "    gl_ClipDistance[%d] = 1.0;\n", k);
            }
        }
    }
}

/* The vertex shader for the current state (remembered), or NULL. */
static const GpuVs *gpu_vs_get(const R300State *st, const R300PVSProgram *prog,
                               const Layout *lay, const Route *route)
{
    bool persp = !((r300_reg(st, VAP_VTE_CNTL) >> 8) & 1);
    bool fog = r300_reg(st, FG_FOG_BLEND) & 1;
    uint32_t fogsel = r300_reg(st, GB_SELECT) & 7;
    uint32_t clip = r300_reg(st, VAP_CLIP_CNTL) & 0x3F;
    uint32_t *key = NULL;
    unsigned klen = 0, kcap = 0;

    if (prog->last_inst < prog->first_inst || prog->last_inst >= R300_PVS_MAX_INSTS) {
        return NULL;
    }
    key_push(&key, &klen, &kcap, prog->first_inst);
    key_push(&key, &klen, &kcap, prog->last_inst);
    key_push(&key, &klen, &kcap, prog->fc_opc);
    key_push(&key, &klen, &kcap, persp | fog << 1 | fogsel << 2 | clip << 8);
    for (unsigned k = prog->first_inst * 4; k < (prog->last_inst + 1) * 4; k++) {
        key_push(&key, &klen, &kcap, prog->code[k]);
    }
    const uint32_t *lw = (const uint32_t *)lay;
    for (unsigned k = 0; k < sizeof(*lay) / 4; k++) {
        key_push(&key, &klen, &kcap, lw[k]);
    }
    key_push(&key, &klen, &kcap, route->nvary);
    for (unsigned k = 0; k < route->nvary; k++) {
        key_push(&key, &klen, &kcap, route->vary[k].is_tex);
        key_push(&key, &klen, &kcap, r300_reg(st, RS_IP_0 + 4 * route->vary[k].ip));
    }

    for (unsigned k = 0; k < GPU_VS_SLOTS; k++) {
        GpuVs *m = &gpu_vs_memo[k];
        if (m->key && m->klen == klen && !memcmp(m->key, key, klen * 4)) {
            GpuVs hit = *m;
            memmove(&gpu_vs_memo[1], &gpu_vs_memo[0], k * sizeof(GpuVs));
            gpu_vs_memo[0] = hit;
            free(key);
            return hit.glsl ? &gpu_vs_memo[0] : NULL;
        }
    }

    /* Generate.  A program the translation cannot take is remembered as
     * such (glsl NULL), so it isn't retried every draw. */
    R300Sb body, sb;
    uint32_t in_used = 0;
    char *glsl = NULL;
    r300_sb_init(&body);
    if (r300_pvs_to_glsl(prog, &body, &in_used)) {
        r300_sb_init(&sb);
        r300_sb_printf(&sb,
            "#version 450\n"
            "%s"
            "layout(std430, set = 0, binding = 2) readonly buffer R300VI { vec4 vin[]; };\n"
            "layout(std140, set = 0, binding = 3) uniform R300MS { vec4 ms; };\n"
            "layout(std140, set = 0, binding = 6) uniform R300VSU {\n"
            "    vec4 c[256]; vec4 ucp[6]; vec4 vp0; vec4 vp1; vec4 fogp; uvec4 info;\n"
            "} vs;\n"
            "layout(location = 0) out vec4 v0; layout(location = 1) out vec4 v1;\n"
            "layout(location = 2) out vec4 v2; layout(location = 3) out vec4 v3;\n"
            "layout(location = 4) out vec4 v4; layout(location = 5) out vec4 v5;\n"
            "layout(location = 6) out vec4 v6; layout(location = 7) out vec4 v7;\n"
            "layout(location = 8) out vec4 v8; layout(location = 9) out vec4 v9;\n"
            "layout(location = 10) out vec4 aux;\n"
            "%s"
            "vec4 pvs_c(int x) { return x >= 0 && x <= 255 ? vs.c[x] : vec4(0.0); }\n"
            "\n"
            "void main()\n"
            "{\n"
            "    uint vb = uint(gl_VertexIndex) * vs.info.x;\n",
            r300_pvs_glsl_helpers,
            clip ? "out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[6]; };\ninvariant gl_Position;\n"
              : "invariant gl_Position;\n");
        for (unsigned i = 0, slot = 0; i < R300_PVS_NUM_INPUTS; i++) {
            if (in_used & (1u << i)) {
                r300_sb_printf(&sb, "    vec4 i%u = vin[vb + %uu];\n", i, slot++);
            }
        }
        r300_sb_printf(&sb, "%s", body.buf);
        gpu_vs_emit_post(&sb, st, lay, route, persp, fog, fogsel, clip);
        r300_sb_printf(&sb, "}\n");
        glsl = r300_sb_steal(&sb);
    }
    r300_sb_free(&body);

    GpuVs *last = &gpu_vs_memo[GPU_VS_SLOTS - 1];
    free(last->key);
    free(last->glsl);
    memmove(&gpu_vs_memo[1], &gpu_vs_memo[0], (GPU_VS_SLOTS - 1) * sizeof(GpuVs));
    gpu_vs_memo[0] = (GpuVs){ .glsl = glsl, .id = glsl ? gpu_vs_next_id++ : 0,
                              .in_used = in_used, .key = key, .klen = klen };
    return glsl ? &gpu_vs_memo[0] : NULL;
}

/*
 * The GPU-shaded form of a draw: decoded input vertices (each index
 * once), the triangle list as indices into them, and the uniforms.
 * Returns false (pkt untouched) when the draw needs the CPU path;
 * *failed when fetching the vertices failed (the draw is dropped).
 */
static bool gpu_vs_build(const R300State *st, Fetch *f, const R300PVSProgram *prog,
                         const Layout *lay, const Route *route, uint32_t prim,
                         const uint32_t *order, uint32_t n, R300DrawPacket *pkt,
                         bool *failed, const char **err)
{
    uint32_t colctl = r300_reg(st, GA_COLOR_CONTROL);

    *failed = false;
    if ((r300_reg(st, GA_POLY_MODE) & 3) == 1 ||
        (r300_reg(st, VAP_OUTPUT_VTX_FMT_0) & (3u << 3))) {
        return false;               /* polygon mode, two-sided colours */
    }
    for (int c = 0; c < 4; c++) {
        if (((colctl >> (4 * c)) & 3) != 2 || ((colctl >> (4 * c + 2)) & 3) != 2) {
            return false;           /* flat or solid colours */
        }
    }
    uint32_t *list = malloc(sizeof(uint32_t) * (n * 3 + 6));
    uint32_t cls, nidx = r300_assemble_prov(prim, n, list, &cls, NULL);
    if (!nidx || cls != 0) {
        free(list);
        return false;               /* lines and points: expanded on the CPU */
    }
    const GpuVs *vs = gpu_vs_get(st, prog, lay, route);
    if (!vs) {
        free(list);
        return false;
    }

    unsigned nin = 0;
    uint8_t slots[R300_PVS_NUM_INPUTS];
    for (unsigned i = 0; i < R300_PVS_NUM_INPUTS; i++) {
        if (vs->in_used & (1u << i)) {
            slots[nin++] = i;
        }
    }
    unsigned stride = nin ? nin : 1;

    /* Each distinct index once; pos[] maps draw positions to them. */
    uint32_t hn = 8;
    while (hn < 2 * n) {
        hn <<= 1;
    }
    uint32_t *hkey = malloc(hn * sizeof(*hkey)), *hval = malloc(hn * sizeof(*hval));
    uint32_t *pos = malloc(n * sizeof(*pos));
    float (*in)[4] = malloc((size_t)n * stride * sizeof(*in));
    uint32_t nu = 0;
    memset(hval, 0xFF, hn * sizeof(*hval));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t h = (order[i] * 2654435761u) & (hn - 1);
        while (hval[h] != UINT32_MAX && hkey[h] != order[i]) {
            h = (h + 1) & (hn - 1);
        }
        if (hval[h] == UINT32_MAX) {
            float v[R300_PVS_NUM_INPUTS][4];
            if (!fetch_vertex(f, order[i], v)) {
                *err = f->why ? f->why : "vertex data out of range";
                *failed = true;
                free(hkey); free(hval); free(pos); free(in); free(list);
                return false;
            }
            for (unsigned k = 0; k < nin; k++) {
                memcpy(in[nu * stride + k], v[slots[k]], sizeof(v[0]));
            }
            if (!nin) {
                memset(in[nu * stride], 0, sizeof(in[0]));
            }
            hkey[h] = order[i];
            hval[h] = nu++;
        }
        pos[i] = hval[h];
    }
    for (uint32_t k = 0; k < nidx; k++) {
        list[k] = pos[list[k]];
    }
    free(hkey);
    free(hval);
    free(pos);

    R300VSUniforms *u = calloc(1, sizeof(*u));
    /* Reads past VAP_PVS_CONST_CNTL's count return 0: zero those slots
     * here rather than bound the reads in the text, so one program text
     * serves every constant count (each text is a shader compile). */
    memcpy(u->c, prog->consts, (size_t)(prog->max_const + 1) * sizeof(u->c[0]));
    for (int k = 0; k < 6; k++) {
        memcpy(u->ucp[k], &st->pvs_mem[(R300_PVS_UCP_START + k) * 4], sizeof(u->ucp[k]));
    }
    uint32_t vte = r300_reg(st, VAP_VTE_CNTL);
    float vp[6];
    for (int i = 0; i < 6; i++) {
        vp[i] = (vte >> i) & 1 ? bits_to_float(r300_reg(st, SE_VPORT_XSCALE + 4 * i))
                               : (i % 2 ? 0.0f : 1.0f);
    }
    memcpy(u->vp0, vp, sizeof(u->vp0));
    u->vp1[0] = vp[4];
    u->vp1[1] = vp[5];
    u->vp1[2] = pkt->rt_width;
    u->vp1[3] = pkt->rt_height;
    u->fogp[0] = bits_to_float(r300_reg(st, GA_FOG_SCALE));
    u->fogp[1] = bits_to_float(r300_reg(st, GA_FOG_OFFSET));
    u->info[0] = stride;

    pkt->vs_glsl = strdup(vs->glsl);
    pkt->vs_id = vs->id;
    pkt->vs_in = in;
    pkt->vs_in_vecs = nu * stride;
    pkt->vs_idx = list;
    pkt->vs_u = u;
    pkt->verts = NULL;
    pkt->num_verts = nidx;
    pkt->num_line_verts = 0;
    pkt->prim_class = 0;
    pkt->cull = ((1u << prim) & 0xE0F0u) ? r300_reg(st, SU_CULL_MODE) & 7 : 0;
    return true;
}

/*
 * Everything after vertex ordering.  order[] (n entries, malloc'd) is
 * taken over; immd is the DRAW_IMMD_2 vertex data or NULL.
 */
static bool draw_core(const R300State *st, const R300Arrays *arr,
                      uint32_t vf, uint32_t *order, uint32_t n,
                      const uint32_t *immd, uint32_t immd_dw,
                      R300ReadFn read, void *opaque,
                      R300DrawPacket *pkt, const char **err)
{
    uint32_t prim, nidx = 0;
    uint32_t *list = NULL;
    R300Prov *prov = NULL;
    Fetch f = { 0 };
    Layout lay;
    Route route;
    R300FSDesc desc;
    uint32_t pitch_reg = r300_reg(st, RB3D_COLORPITCH0);
    uint32_t cntl0 = r300_reg(st, VAP_PVS_CODE_CNTL_0);
    uint32_t cc = r300_reg(st, VAP_PVS_CONST_CNTL);
    bool bypass = (r300_reg(st, VAP_CNTL_STATUS) >> 8) & 1;
    uint32_t tl = r300_reg(st, SC_SCISSORS_TL), br = r300_reg(st, SC_SCISSORS_BR);
    uint32_t clip = r300_reg(st, VAP_CLIP_CNTL);
    uint32_t fogsel = r300_reg(st, GB_SELECT) & 7;
    bool fog = r300_reg(st, FG_FOG_BLEND) & 1;
    float fog_scale = bits_to_float(r300_reg(st, GA_FOG_SCALE));
    float fog_off = bits_to_float(r300_reg(st, GA_FOG_OFFSET));
    R300PVSProgram prog;
    float consts[256][4];
    Build b;

    prim = vf & 0xF;

    /* Colour buffer */
    pkt->rt_gpu_addr = r300_reg(st, RB3D_COLOROFFSET0) & ~0x1Fu;
    pkt->rt_pitch = pitch_reg & 0x3FFE;
    pkt->rt_format = (pitch_reg >> 21) & 0xF;
    pkt->rt_view = r300_cb_view(pkt->rt_format, r300_reg(st, US_OUT_FMT_0),
                                &pkt->rt_bpp);
    if ((pkt->rt_bpp == 1 && (pitch_reg >> 19) & 3) ||
        (pkt->rt_bpp == 2 && ((pitch_reg >> 19) & 3) >= 2)) {
        pkt->warn |= R300_WARN_RTFMT;   /* swap across pixels: not done */
    }
    if (pkt->rt_view == R300_RTV_NONE) {
        pkt->warn |= R300_WARN_RTFMT;
        *err = "colour buffer format not supported";
        free(order);
        return false;
    }
    /* Render targets B-D (multiple render targets or multiwrites) */
    pkt->num_cb = r300_us_num_targets(st);
    for (uint32_t k = 1; k < pkt->num_cb; k++) {
        R300ColorDesc *c = &pkt->cb[k];
        uint32_t pr = r300_reg(st, RB3D_COLORPITCH0 + 4 * k);
        c->gpu_addr = r300_reg(st, RB3D_COLOROFFSET0 + 4 * k) & ~0x1Fu;
        c->pitch = pr & 0x3FFE;
        c->format = (pr >> 21) & 0xF;
        c->view = r300_cb_view(c->format, r300_us_out_fmt(st, k), &c->bpp);
        if (c->view == R300_RTV_NONE || !c->pitch) {
            pkt->warn |= R300_WARN_RTFMT;
            pkt->num_cb = k;            /* bind the ones before it */
            break;
        }
    }
    pkt->scissor[0] = ((tl & 0x1FFF) > SC_COORD_BIAS) ? (tl & 0x1FFF) - SC_COORD_BIAS : 0;
    pkt->scissor[1] = (((tl >> 13) & 0x1FFF) > SC_COORD_BIAS) ?
                      ((tl >> 13) & 0x1FFF) - SC_COORD_BIAS : 0;
    pkt->scissor[2] = ((br & 0x1FFF) >= SC_COORD_BIAS) ?
                      (br & 0x1FFF) - SC_COORD_BIAS + 1 : 0;
    pkt->scissor[3] = (((br >> 13) & 0x1FFF) >= SC_COORD_BIAS) ?
                      ((br >> 13) & 0x1FFF) - SC_COORD_BIAS + 1 : 0;
    pkt->rt_width = pkt->rt_pitch;
    /*
     * The card has no colour buffer height; take the lowest scissor edge
     * seen for this buffer.  The renderer maps pixels to clip space and
     * back through this height, so it must not change from draw to draw:
     * Doom 3 draws depth full screen and each light inside its own
     * scissor, with an EQUAL depth test, and a per-scissor height rounded
     * the light passes' depths differently -- unlit dots, rectangles and
     * triangles.  (A buffer reused at the same address keeps the larger
     * height, which only widens the area the renderer may touch.)
     */
    {
        static struct { uint32_t addr, pitch, h; } seen[16];
        static unsigned next;
        unsigned k;
        for (k = 0; k < 16; k++) {
            if (seen[k].h && seen[k].addr == pkt->rt_gpu_addr &&
                seen[k].pitch == pkt->rt_pitch) {
                break;
            }
        }
        if (k == 16) {
            k = next++ & 15;
            seen[k].addr = pkt->rt_gpu_addr;
            seen[k].pitch = pkt->rt_pitch;
            seen[k].h = 0;
        }
        if (pkt->scissor[3] > seen[k].h && pkt->scissor[3] <= 4096) {
            seen[k].h = pkt->scissor[3];
        }
        pkt->rt_height = seen[k].h ? seen[k].h : pkt->scissor[3];
    }
    if (!pkt->rt_width || !pkt->rt_height) {
        *err = "empty colour buffer";
        free(order);
        return false;
    }

    pkt->aa_samples = r300_aa_samples(st);
    r300_aa_positions(st, pkt->aa_pos);

    set_uniforms(st, pkt);
    set_textures(st, pkt);
    set_depth(st, pkt);

    /* Vertex order */
    f.st = st;
    f.arr = arr;
    f.read = read;
    f.opaque = opaque;
    f.swap = r300_reg(st, VAP_CNTL_STATUS) & 3;
    f.vtx_size = r300_reg(st, VAP_VTX_SIZE) & 0x7F;
    f.immd = immd;
    f.immd_dw = immd_dw;

    /* Vertex program */
    for (int i = 0; i < 256; i++) {
        memcpy(consts[i], &st->pvs_mem[(R300_PVS_CONST_START + (cc & 0xFF) + i) * 4],
               sizeof(consts[i]));
        if (R300_PVS_CONST_START + (cc & 0xFF) + i + 1 >= R300_PVS_MEM_VECS) {
            break;
        }
    }
    prog.code = &st->pvs_mem[R300_PVS_CODE_START * 4];
    prog.first_inst = cntl0 & 0x3FF;
    prog.last_inst = (cntl0 >> 20) & 0x3FF;
    prog.consts = (const float (*)[4])consts;
    prog.max_const = (cc >> 16) & 0xFF;
    prog.fc_opc = r300_reg(st, VAP_PVS_FLOW_CNTL_OPC);
    for (int i = 0; i < 16; i++) {
        prog.fc_addrs[i] = r300_reg(st, VAP_PVS_FLOW_CNTL_ADDRS_0 + 4 * i);
        prog.fc_loop[i] = r300_reg(st, VAP_PVS_FLOW_CNTL_LOOP_INDEX_0 + 4 * i);
    }

    output_layout(st, &lay);
    rs_route(st, &route, &desc, &pkt->warn);

    /* Rectangle lists: three corners a rectangle, the card completes the
     * parallelogram (v0 + v2 - v1) and draws it as a quad. */
    bool rects = prim == 8;
    uint32_t nsrc = rects ? n / 3 * 4 : n;

    if ((st->glsl_flags & R300_GLSL_GPU_VS) && !rects && !bypass) {
        bool failed;
        if (gpu_vs_build(st, &f, &prog, &lay, &route, prim, order, n, pkt,
                         &failed, err)) {
            pkt->warn |= f.warn;
            free(order);
            pkt->glsl = r300_us_glsl_cached(st, &desc, st->glsl_flags,
                                            &pkt->glsl_id, err);
            if (!pkt->glsl) {
                r300_draw_free(pkt);
                return false;
            }
            return true;
        }
        if (failed) {
            free(order);
            return false;
        }
    }

    /* Transform every vertex in draw order (cheap next to the GPU work). */
    R300Vertex *xv = calloc(nsrc ? nsrc : 1, sizeof(R300Vertex));
    float (*outs)[R300_PVS_NUM_OUTPUTS][4] = calloc(nsrc ? nsrc : 1, sizeof(*outs));

    /*
     * Indexed meshes name most vertices several times; the result depends
     * only on the index, so shade each one once (a post-transform cache,
     * as the hardware has).  Open addressing on order[i] -> slot j.
     */
    uint32_t hmask = 0, *hkey = NULL, *hslot = NULL;
    if (!rects && n > 3) {
        uint32_t hn = 8;
        while (hn < 2 * n) {
            hn <<= 1;
        }
        hmask = hn - 1;
        hkey = malloc(hn * sizeof(*hkey));
        hslot = malloc(hn * sizeof(*hslot));
        memset(hslot, 0xFF, hn * sizeof(*hslot));
    }
    for (uint32_t i = 0, j = 0; i < n && j < nsrc; i++, j++) {
        float in[R300_PVS_NUM_INPUTS][4];
        float (*out)[4] = outs[j];
        uint32_t h = 0;

        if (hslot) {
            h = (order[i] * 2654435761u) & hmask;
            while (hslot[h] != UINT32_MAX && hkey[h] != order[i]) {
                h = (h + 1) & hmask;
            }
            if (hslot[h] != UINT32_MAX) {
                memcpy(out, outs[hslot[h]], sizeof(outs[j]));
                continue;
            }
        }
        if (!fetch_vertex(&f, order[i], in)) {
            free(hkey);
            free(hslot);
            *err = f.why ? f.why : "vertex data out of range";
            free(outs);
            free(xv);
            free(order);
            return false;
        }
        if (bypass) {
            memcpy(out, in, sizeof(outs[j]));
        } else {
            uint32_t u = r300_pvs_run(&prog, (const float (*)[4])in, out);
            if (u & R300_PVS_UNSUP_FLOW) {
                pkt->warn |= R300_WARN_FLOW;
            }
            if (u & ~R300_PVS_UNSUP_FLOW) {
                pkt->warn |= R300_WARN_PVS;
            }
        }
        if (hslot) {
            hkey[h] = order[i];
            hslot[h] = j;
        }
        if (rects && i % 3 == 2) {
            j++;
            for (int o = 0; o < R300_PVS_NUM_OUTPUTS; o++) {
                for (int c = 0; c < 4; c++) {
                    outs[j][o][c] = outs[j - 3][o][c] + outs[j - 1][o][c] - outs[j - 2][o][c];
                }
            }
        }
    }
    free(hkey);
    free(hslot);
    for (uint32_t j = 0; j < nsrc; j++) {
        float (*out)[4] = outs[j];
        float cols[4][4];

        viewport_xform(st, pkt, out[lay.pos], xv[j].pos);
        vtx_colors(&lay, out, cols);
        rs_values(st, &lay, &route, out, (const float (*)[4])cols, NULL, xv[j].v);
        if (fog) {
            float src = fogsel < 4 ? (lay.color[fogsel] >= 0 ? out[lay.color[fogsel]][3] : 0.0f)
                      : fogsel == 4 ? out[lay.pos][3]
                      : xv[j].pos[2] / (xv[j].pos[3] != 0.0f ? xv[j].pos[3] : 1.0f);
            xv[j].aux[0] = src * fog_scale + fog_off;
        }
        /* User clip planes (VAP_CLIP_CNTL.UCP_ENA_n): the distance of the
         * clip-space position from plane n in PVS memory 1024 + n. */
        for (int k = 0; k < 6; k++) {
            const float *pl = (const float *)&st->pvs_mem[(R300_PVS_UCP_START + k) * 4];
            const float *cp = out[lay.pos];
            xv[j].ucp[k] = (clip >> k) & 1 ? cp[0] * pl[0] + cp[1] * pl[1] +
                                             cp[2] * pl[2] + cp[3] * pl[3] : 1.0f;
        }
    }
    if (rects) {
        prim = 13;
        n = nsrc;
    }
    pkt->warn |= f.warn;

    /* Primitive assembly */
    memset(&b, 0, sizeof(b));
    b.st = st;
    b.pkt = pkt;
    b.lay = &lay;
    b.route = &route;
    b.outs = outs;
    b.xv = xv;
    b.n = n;
    b.colctl = r300_reg(st, GA_COLOR_CONTROL);
    b.back_colors = (r300_reg(st, VAP_OUTPUT_VTX_FMT_0) & (3u << 3)) != 0;
    for (int c = 0; c < 4 && !b.fancy_colors; c++) {
        b.fancy_colors = ((b.colctl >> (4 * c)) & 3) != 2 ||
                         ((b.colctl >> (4 * c + 2)) & 3) != 2;
    }
    b.fancy_colors |= b.back_colors;
    b.solid[0] = s312(r300_reg(st, GA_SOLID_RG) >> 16);
    b.solid[1] = s312(r300_reg(st, GA_SOLID_RG));
    b.solid[2] = s312(r300_reg(st, GA_SOLID_BA) >> 16);
    b.solid[3] = s312(r300_reg(st, GA_SOLID_BA));
    b.poly_mode = (r300_reg(st, GA_POLY_MODE) & 3) == 1 ? r300_reg(st, GA_POLY_MODE) : 0;
    {
        uint32_t lc = r300_reg(st, GA_LINE_CNTL);
        uint32_t sc = r300_reg(st, GA_LINE_STIPPLE_CONFIG);
        uint32_t ps = r300_reg(st, GA_POINT_SIZE);
        uint32_t mm = r300_reg(st, GA_POINT_MINMAX);
        b.line_hw = (lc & 0xFFFF) / 12.0f;
        b.line_end = (lc >> 16) & 3;
        b.stip_scale = bits_to_float(sc & ~3u);
        b.stip_reset = sc & 3;
        b.stip_acc = b.stip_reset ? 0.0f : (float)(r300_reg(st, GA_LINE_STIPPLE_VALUE) & 0xFFFFFF);
        b.point_hw = ((ps >> 16) & 0xFFFF) / 12.0f;
        b.point_hh = (ps & 0xFFFF) / 12.0f;
        b.point_min = (mm & 0xFFFF) / 6.0f;
        b.point_max = (mm >> 16) / 6.0f;
        b.point_psize = lay.psize >= 0;
    }
    /* The setup unit culls polygons only (triangles, fans, strips, quads,
     * quad strips, polygons).  Metal culls filled triangles; the CPU does
     * it when polygon mode turns them into lines or points. */
    pkt->cull = ((1u << prim) & 0xE0F0u) ? r300_reg(st, SU_CULL_MODE) & 7 : 0;
    if (b.poly_mode && pkt->cull) {
        b.cull = pkt->cull;
        pkt->cull &= R300_FACE_CW;
    }

    list = malloc(sizeof(uint32_t) * (n * 3 + 6));
    prov = malloc(sizeof(R300Prov) * (n * 2 + 2));
    uint32_t cls;
    nidx = r300_assemble_prov(prim, n, list, &cls, prov);
    if (!nidx) {
        if (prim != 0) {
            pkt->warn |= R300_WARN_PRIM;
        }
        free(prov);
        free(list);
        free(outs);
        free(xv);
        free(order);
        *err = "primitive type not supported";
        return false;
    }
    if (cls == 0) {
        for (uint32_t i = 0, t = 0; i + 2 < nidx; i += 3, t++) {
            emit_tri(&b, &list[i], &prov[t]);
        }
    } else if (cls == 1) {
        if (b.stip_reset == 2) {
            b.stip_acc = 0.0f;
        }
        for (uint32_t i = 0, t = 0; i + 1 < nidx; i += 2, t++) {
            emit_line(&b, list[i], list[i + 1], &prov[t], false);
        }
    } else {
        for (uint32_t i = 0; i < nidx; i++) {
            emit_point(&b, list[i], &prov[i], false);
        }
    }
    if (!b.ntri && b.nline) {
        pkt->verts = b.line;
        pkt->num_verts = b.nline;
        pkt->prim_class = 1;
        free(b.tri);
    } else {
        if (b.nline) {
            b.tri = realloc(b.tri, sizeof(R300Vertex) * (b.ntri + b.nline));
            memcpy(b.tri + b.ntri, b.line, sizeof(R300Vertex) * b.nline);
        }
        free(b.line);
        pkt->verts = b.tri;
        pkt->num_verts = b.ntri;
        pkt->num_line_verts = b.nline;
        pkt->prim_class = 0;
    }
    free(prov);
    free(list);
    free(outs);
    free(xv);
    free(order);

    pkt->glsl = r300_us_glsl_cached(st, &desc, st->glsl_flags, &pkt->glsl_id, err);
    if (!pkt->glsl) {
        r300_draw_free(pkt);
        return false;
    }
    return true;
}

bool r300_draw_build(const R300State *st, const R300Arrays *arr,
                     uint32_t opcode, const uint32_t *d, uint32_t ndw,
                     R300ReadFn read, void *opaque,
                     R300DrawPacket *pkt, const char **err)
{
    const uint32_t *payload = d + 1;
    uint32_t payload_dw = ndw ? ndw - 1 : 0;
    uint32_t vf, n, *order;

    memset(pkt, 0, sizeof(*pkt));
    *err = NULL;
    if (ndw < 1) {
        *err = "empty draw packet";
        return false;
    }
    vf = d[0];
    n = vf >> 16;
    if (!n) {
        *err = "no vertices";
        return false;
    }
    order = malloc(sizeof(uint32_t) * n);
    if (opcode == 0x36) {                       /* DRAW_INDX_2, inline */
        bool i32 = (vf >> 11) & 1;
        for (uint32_t i = 0; i < n; i++) {
            order[i] = r300_index_at(payload, payload_dw, i32, i);
        }
    } else {                                    /* DRAW_VBUF_2, DRAW_IMMD_2 */
        for (uint32_t i = 0; i < n; i++) {
            order[i] = i;
        }
    }
    return draw_core(st, arr, vf, order, n, opcode == 0x35 ? payload : NULL,
                     opcode == 0x35 ? payload_dw : 0, read, opaque, pkt, err);
}

uint32_t r300_aa_samples(const R300State *st)
{
    static const uint8_t n[4] = { 2, 3, 4, 6 };
    uint32_t aa = r300_reg(st, GB_AA_CONFIG);

    return (aa & 1) ? n[(aa >> 1) & 3] : 1;
}

void r300_aa_positions(const R300State *st, float pos[6][2])
{
    uint64_t v = r300_reg(st, GB_MSPOS0) & 0xFFFFFF;

    v |= (uint64_t)(r300_reg(st, GB_MSPOS1) & 0xFFFFFF) << 24;
    for (int k = 0; k < 6; k++, v >>= 8) {
        pos[k][0] = ((float)(v & 0xF) - 6.0f) / 12.0f;
        pos[k][1] = ((float)((v >> 4) & 0xF) - 6.0f) / 12.0f;
    }
}

uint32_t r300_msaa_offset(uint32_t x, uint32_t y, uint32_t ns,
                          uint32_t pitch_px, uint32_t bpp, uint32_t sample)
{
    uint32_t sh = bpp == 2 ? 1 : 2;             /* log2(bytes per sample) */
    uint32_t pitch = pitch_px * ns;             /* in samples, as the driver keeps it */
    uint32_t base, idx;

    if ((ns != 2 && ns != 4) || sample >= ns) {
        return ~0u;
    }
    base = 32 * ((pitch / ns / 4) * ns * sh * 2 * (y >> 3) +
                 ((((x >> 2) << 1) | ((y >> 2) & 1)) * ns * sh));
    if (ns == 2) {
        idx = (((y >> 1) & 1) << 4) | (((x >> 1) & 1) << 3) | (sample << 2);
    } else {
        idx = (((((x >> 2) & 1) ^ ((y >> 1) & 1))) << 5) |
              (((x >> 1) & 1) << 4) | (sample << 2);
    }
    idx |= ((y & 1) << 1) | (x & 1);
    return base + (idx << sh);
}

uint32_t r300_index_at(const uint32_t *dw, uint32_t ndw, bool i32, uint32_t i)
{
    if (i32) {
        return i < ndw ? dw[i] : 0;
    }
    return i / 2 < ndw ? (dw[i / 2] >> (16 * (i & 1))) & 0xFFFF : 0;
}

bool r300_draw_build_indexed(const R300State *st, const R300Arrays *arr,
                             uint32_t vf, const R300Indices *idx,
                             R300ReadFn read, void *opaque,
                             R300DrawPacket *pkt, const char **err)
{
    uint32_t n = vf >> 16, *order, *sw;
    unsigned swap = r300_reg(st, VAP_CNTL_STATUS) & 3;
    bool i32 = (vf >> 11) & 1;

    memset(pkt, 0, sizeof(*pkt));
    *err = NULL;
    if (!n) {
        *err = "no vertices";
        return false;
    }
    if ((uint64_t)(i32 ? n : (n + 1) / 2) > idx->ndw) {
        *err = "index buffer shorter than the draw";
        return false;
    }
    sw = malloc(sizeof(uint32_t) * (idx->ndw ? idx->ndw : 1));
    for (uint32_t k = 0; k < idx->ndw; k++) {
        sw[k] = vc_swap(idx->dw[k], swap);
    }
    order = malloc(sizeof(uint32_t) * n);
    for (uint32_t i = 0; i < n; i++) {
        order[i] = r300_index_at(sw, idx->ndw, i32, i);
    }
    free(sw);
    return draw_core(st, arr, vf, order, n, NULL, 0, read, opaque, pkt, err);
}

void r300_draw_free(R300DrawPacket *pkt)
{
    free(pkt->glsl);
    free(pkt->verts);
    free(pkt->vs_glsl);
    free(pkt->vs_in);
    free(pkt->vs_idx);
    free(pkt->vs_u);
    pkt->glsl = NULL;
    pkt->verts = NULL;
    pkt->vs_glsl = NULL;
    pkt->vs_in = NULL;
    pkt->vs_idx = NULL;
    pkt->vs_u = NULL;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        free(pkt->tex[t].host_data);
        pkt->tex[t].host_data = NULL;
    }
}

/* ---- texel repacking for renderers ------------------------------------ */

/* A 16bpp texel as the card sees it: the guest wrote big-endian halfwords. */
static void r300_decode16(uint32_t fmt, uint16_t v, uint8_t out[4])
{
    uint32_t x, y, z, w;

    switch (fmt) {
    case 0x6:   /* Z5Y6X5 */
        x = (v & 31) * 255 / 31; y = ((v >> 5) & 63) * 255 / 63;
        z = (v >> 11) * 255 / 31; w = 255;
        break;
    case 0x7:   /* Z6Y5X5 */
        x = (v & 31) * 255 / 31; y = ((v >> 5) & 31) * 255 / 31;
        z = (v >> 10) * 255 / 63; w = 255;
        break;
    case 0xA:   /* W4Z4Y4X4 */
        x = (v & 15) * 17; y = ((v >> 4) & 15) * 17;
        z = ((v >> 8) & 15) * 17; w = (v >> 12) * 17;
        break;
    case 0xB:   /* W1Z5Y5X5 */
        x = (v & 31) * 255 / 31; y = ((v >> 5) & 31) * 255 / 31;
        z = ((v >> 10) & 31) * 255 / 31; w = (v >> 15) ? 255 : 0;
        break;
    default:    /* X16 */
        x = v >> 8; y = z = 0; w = 255;
        break;
    }
    out[0] = x; out[1] = y; out[2] = z; out[3] = w;
}

uint64_t r300_hash_bytes(const uint8_t *p, size_t n)
{
    const uint64_t m = 0x9E3779B97F4A7C15ull;
    uint64_t h = n * m, v;
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        memcpy(&v, p + i, 8);
        h = (h ^ v) * m;
        h ^= h >> 29;
    }
    for (; i < n; i++) {
        h = (h ^ p[i]) * m;
    }
    return h ^ (h >> 32);
}

/*
 * Identity of a texture's texels for the renderers' caches.  Hashing every
 * byte of a mip chain on every draw cost a third of the render thread in
 * Doom 3 (megabytes of DXT per draw), so large chains are sampled: both
 * ends plus a spread of words across the rest.  The driver uploads whole
 * textures, which changes every sampled stretch; $R300_TEX_FULLHASH=1
 * hashes everything.
 */
uint64_t r300_hash_tex(const uint8_t *p, size_t n)
{
    static int full = -1;
    uint64_t h;

    if (full < 0) {
        full = getenv("R300_TEX_FULLHASH") != NULL;
    }
    if (full || n <= 32768) {
        return r300_hash_bytes(p, n);
    }
    h = r300_hash_bytes(p, 8192) ^ (r300_hash_bytes(p + n - 8192, 8192) * 3);
    size_t step = (n / 2048) & ~(size_t)7;      /* 2048 spots, 8-byte aligned */
    for (size_t i = 8192; i + 8 <= n - 8192; i += step) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h = (h ^ v) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    return h ^ n;
}

/*
 * Texels as the card reads them: little-endian words.  VRAM holds the
 * big-endian CPU's view (GART copies the other side of the aperture),
 * and the card's dword is the TX_OFFSET endian swap of that, as
 * r300_texel does for shader-decoded formats: with no swap, every word
 * is byte-reversed.  n is a multiple of 4.
 */
static void r300_tex_card_words(const R300TexDesc *td, uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i += 4) {
        uint32_t l, v;
        memcpy(&l, p + i, 4);
        if (td->host_data) {
            l = __builtin_bswap32(l);
        }
        v = __builtin_bswap32(l);                       /* the BE value */
        switch (td->endian & 3) {
        case 1: v = ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu); break;
        case 2: v = __builtin_bswap32(v); break;
        case 3: v = (v << 16) | (v >> 16); break;
        default: break;
        }
        memcpy(p + i, &v, 4);
    }
}

/*
 * r300_hash_tex for a texture: 8-bit texels in full.  QuartzGL keeps its
 * glyphs in an X8 atlas and adds a few at a time, which a sampled hash
 * misses (the stale copy drew later glyphs wrong); such atlases are
 * small next to the DXT chains the sampling is for.
 */
uint64_t r300_tex_identity(const R300TexDesc *td, const uint8_t *src)
{
    if (td->kind == R300_TEXK_R8) {
        return r300_hash_bytes(src, td->size_bytes) ^ td->size_bytes;
    }
    return r300_hash_tex(src, td->size_bytes);
}

uint8_t *r300_tex_level_bytes(const R300TexDesc *td, const uint8_t *src,
                                 uint32_t l, uint32_t w, uint32_t h,
                                 uint32_t *bpr)
{
    uint32_t pitch = td->lvl_pitch[l];
    uint8_t *out;

    switch (td->kind) {
    case R300_TEXK_CONVERT16:
        *bpr = w * 4;
        out = malloc((size_t)w * h * 4);
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t *row = src + (uint64_t)y * pitch;
            for (uint32_t x = 0; x < w; x++) {
                uint16_t v = (uint16_t)(row[2 * x] << 8 | row[2 * x + 1]);
                r300_decode16(td->format, v, out + ((size_t)y * w + x) * 4);
            }
        }
        return out;
    case R300_TEXK_DXT1:
    case R300_TEXK_DXT3:
    case R300_TEXK_DXT5: {
        uint32_t bs = td->kind == R300_TEXK_DXT1 ? 8 : 16;
        uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
        *bpr = bw * bs;
        out = malloc((size_t)bw * bh * bs);
        for (uint32_t y = 0; y < bh; y++) {
            memcpy(out + (size_t)y * bw * bs, src + (uint64_t)y * pitch, (size_t)bw * bs);
        }
        /* (Doom 3 on Tiger: untouched blocks decode to coloured
         * speckle, reversed ones to the art.) */
        r300_tex_card_words(td, out, (size_t)bw * bh * bs);
        return out;
    }
    case R300_TEXK_R8: {
        /*
         * One byte a texel, in the card's word order: each source row
         * (pitch 16-byte aligned) as whole words, then its w texels.
         * QuartzGL's glyph masks are X8 with no swap in VRAM; read as
         * they lie, every four texels came out mirrored (Leopard's
         * System Preferences, which enables QuartzGL, drew garbled text).
         */
        uint32_t wb = (w + 3) & ~3u;
        uint8_t *row = calloc(1, wb);
        *bpr = w;
        out = malloc((size_t)w * h);
        for (uint32_t y = 0; y < h; y++) {
            /* an explicit pitch (TX_PITCH_EN) need not be whole words */
            memcpy(row, src + (uint64_t)y * pitch, wb < pitch ? wb : pitch);
            r300_tex_card_words(td, row, wb);
            memcpy(out + (size_t)y * w, row, w);
        }
        free(row);
        return out;
    }
    default: {                          /* RGBA8, RG8: bytes as they lie */
        uint32_t bpp = td->kind == R300_TEXK_RGBA8 ? 4 : 2;
        *bpr = w * bpp;
        out = malloc((size_t)w * h * bpp);
        for (uint32_t y = 0; y < h; y++) {
            memcpy(out + (size_t)y * w * bpp, src + (uint64_t)y * pitch, (size_t)w * bpp);
        }
        return out;
    }
    }
}
