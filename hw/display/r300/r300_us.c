/*
 * R300-family fragment shader unit (US) -> GLSL (Vulkan dialect).
 *
 * Program layout (US_CONFIG, US_CODE_OFFSET, US_CODE_ADDR_0..3): up to four
 * nodes, stored so that the last node is always in CODE_ADDR_3.  Each node
 * runs a block of texture instructions and then a block of ALU
 * instructions; starts are relative to the offsets in US_CODE_OFFSET.
 *
 * ALU instructions are split into an RGB half and an alpha half, each with
 * three source addresses (temporaries 0-31, constants 32-63), an operand
 * select per argument, a pre-subtract, an op, an output modifier and a
 * clamp.  See AMD "R3xx 3D Registers", US_ALU_{RGB,ALPHA}_{ADDR,INST}.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_us.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "r300_sb.h"

/*
 * While r300_us_glsl_cached() generates a program, every register the
 * generator reads is recorded, so a later draw can reuse the text when
 * those registers still hold the same values.  All reads in this file go
 * through r300_reg(), which the macro below routes here.
 */
#define US_REC_MAX 512
static struct {
    bool on, overflow;
    unsigned n;
    uint16_t idx[US_REC_MAX];
    uint8_t seen[R300_REG_COUNT];
} us_rec;

static inline uint32_t us_reg(const R300State *st, uint32_t addr)
{
    if (us_rec.on) {
        unsigned i = (addr - R300_REG_BASE) / 4;
        if (!us_rec.seen[i]) {
            us_rec.seen[i] = 1;
            if (us_rec.n < US_REC_MAX) {
                us_rec.idx[us_rec.n++] = i;
            } else {
                us_rec.overflow = true;
            }
        }
    }
    return r300_reg(st, addr);
}
#define r300_reg(st, addr) us_reg(st, addr)

#define US_CONFIG           0x4600
#define US_CODE_OFFSET      0x4608
#define US_CODE_ADDR_0      0x4610
#define US_TEX_INST_0       0x4620
#define US_ALU_RGB_ADDR_0   0x46C0
#define US_ALU_ALPHA_ADDR_0 0x47C0
#define US_ALU_RGB_INST_0   0x48C0
#define US_ALU_ALPHA_INST_0 0x49C0
#define US_OUT_FMT_0        0x46A4
#define TX_FILTER0_0        0x4400
#define TX_FORMAT0_0        0x4480
#define TX_FORMAT1_0        0x44C0
#define TX_OFFSET_0         0x4540
#define TX_ENABLE           0x4104
#define RB3D_CCTL           0x4E00
#define RB3D_COLORPITCH0    0x4E38
#define VAP_CLIP_CNTL       0x221C

uint32_t r300_tex_raw_bpp(uint32_t txformat)
{
    switch (txformat & 0x1F) {
    case 0x02: case 0x05:                           /* Y4X4, Z3Y3X2 */
        return 1;
    case 0x12: case 0x14: case 0x15:                /* CxV8U8, VYUY, YVYU */
    case 0x16: case 0x18:                           /* 16 MPEG, 16F */
        return 2;
    case 0x04: case 0x08: case 0x09: case 0x0D:     /* Y16X16, 11/11/10, 2/10/10/10 */
    case 0x13: case 0x17: case 0x19: case 0x1B:     /* AVYU, 16_16 MPEG, 16F x2, 32F */
        return 4;
    case 0x0E: case 0x1A: case 0x1C:                /* W16Z16Y16X16, 16F x4, 32F x2 */
        return 8;
    case 0x1D:                                      /* 32F x4 */
        return 16;
    default:
        return 0;
    }
}

uint32_t r300_cb_view(uint32_t cf, uint32_t of, uint32_t *bpp)
{
    of &= 0x1F;
    switch (cf & 0xF) {
    case 9:                                         /* I8 */
        *bpp = 1;
        return R300_RTV_R8U;
    case 3: case 4: case 13: case 15:               /* 1555, 565, UV88, 4444 */
        *bpp = 2;
        return R300_RTV_R16U;
    case 6:                                         /* ARGB8888 */
        /* C4_10 (Tiger's desktop) still stores 8 bits a channel here:
         * the buffer format decides the layout, and 10-bit storage is
         * the R5xx ARGB2101010.  Other OUT_FMTs are Mesa's aliases
         * (RG16, R32F, RG16F) of a 32-bit pixel. */
        *bpp = 4;
        return of <= 2 ? R300_RTV_RGBA8 : R300_RTV_R32U;
    case 10:                                        /* ARGB16161616 */
        *bpp = 8;
        return R300_RTV_RG32U;
    case 7:                                         /* ARGB32323232 */
        *bpp = 16;
        return R300_RTV_RGBA32U;
    default:                                        /* reserved, YUV 4:2:2 */
        *bpp = 4;
        return R300_RTV_NONE;
    }
}

/* TX_FORMAT1.TXFORMAT 16F..32F x4: the card cannot filter them. */
static bool tex_is_float(uint32_t fmt)
{
    return fmt >= 0x18 && fmt <= 0x1D;
}

float r300_float24(uint32_t v)
{
    uint32_t mant = v & 0xFFFF;
    uint32_t exp = (v >> 16) & 0x7F;
    uint32_t sign = (v >> 23) & 1;
    float f;

    if (!exp && !mant) {
        return sign ? -0.0f : 0.0f;
    }
    /* Bias 63; treat the top exponent as a large finite value, not inf. */
    f = ldexpf(1.0f + mant / 65536.0f, (int)exp - 63);
    return sign ? -f : f;
}

typedef struct USNode {
    unsigned alu_start, alu_count;
    unsigned tex_start, tex_count;
} USNode;

/* Decode the node list; returns the number of nodes. */
static unsigned us_nodes(const R300State *st, USNode nodes[4])
{
    uint32_t config = r300_reg(st, US_CONFIG);
    uint32_t off = r300_reg(st, US_CODE_OFFSET);
    unsigned n = (config & 3) + 1;
    unsigned alu_off = off & 0x3F, tex_off = (off >> 13) & 0x1F;

    for (unsigned i = 0; i < n; i++) {
        uint32_t a = r300_reg(st, US_CODE_ADDR_0 + 4 * (4 - n + i));
        USNode *nd = &nodes[i];

        nd->alu_start = alu_off + (a & 0x3F);
        nd->alu_count = ((a >> 6) & 0x3F) + 1;
        nd->tex_start = tex_off + ((a >> 12) & 0x1F);
        nd->tex_count = ((a >> 17) & 0x1F) + 1;
        if (i == 0 && !(config & (1u << 3))) {
            nd->tex_count = 0;          /* FIRST_NODE_HAS_TEX clear */
        }
    }
    return n;
}

/* ---- source expressions --------------------------------------------- */

static void us_addr_expr(char *buf, size_t len, unsigned addr)
{
    if (addr & 32) {
        snprintf(buf, len, "u.consts[%u]", addr & 31);
    } else {
        snprintf(buf, len, "t[%u]", addr & 31);
    }
}

/* RGB operand select (US_ALU_RGB_INST.SEL_*) as a vec3 expression. */
static void us_rgb_sel(char *buf, size_t len, unsigned sel)
{
    static const char *comp[] = { "rgb", "rrr", "ggg", "bbb" };

    if (sel < 12) {
        snprintf(buf, len, "cs%u.%s", sel / 4 == 0 ? 0 : sel / 4 == 1 ? 1 : 2,
                 comp[sel % 4]);
    } else if (sel < 15) {
        snprintf(buf, len, "vec3(as%u.a)", sel - 12);
    } else if (sel < 19) {
        snprintf(buf, len, "ps.%s", comp[sel - 15]);
    } else if (sel == 19) {
        snprintf(buf, len, "vec3(psa)");
    } else if (sel == 20) {
        snprintf(buf, len, "vec3(0.0)");
    } else if (sel == 21) {
        snprintf(buf, len, "vec3(1.0)");
    } else if (sel == 22) {
        snprintf(buf, len, "vec3(0.5)");
    } else if (sel < 26) {
        snprintf(buf, len, "cs%u.gbr", sel - 23);
    } else if (sel < 29) {
        snprintf(buf, len, "cs%u.brg", sel - 26);
    } else {
        snprintf(buf, len, "vec3(as%u.a, cs%u.b, cs%u.g)",
                 sel - 29, sel - 29, sel - 29);
    }
}

/* Alpha operand select (US_ALU_ALPHA_INST.SEL_*) as a float expression. */
static void us_alpha_sel(char *buf, size_t len, unsigned sel)
{
    static const char comp[] = "rgb";

    if (sel < 9) {
        snprintf(buf, len, "cs%u.%c", sel / 3, comp[sel % 3]);
    } else if (sel < 12) {
        snprintf(buf, len, "as%u.a", sel - 9);
    } else if (sel < 15) {
        snprintf(buf, len, "ps.%c", comp[sel - 12]);
    } else if (sel == 15) {
        snprintf(buf, len, "psa");
    } else if (sel == 16) {
        snprintf(buf, len, "0.0");
    } else if (sel == 17) {
        snprintf(buf, len, "1.0");
    } else {
        snprintf(buf, len, "0.5");
    }
}

static void us_mod(char *out, size_t len, const char *expr, unsigned mod)
{
    switch (mod) {
    case 1:  snprintf(out, len, "(-%s)", expr); break;
    case 2:  snprintf(out, len, "abs(%s)", expr); break;
    case 3:  snprintf(out, len, "(-abs(%s))", expr); break;
    default: snprintf(out, len, "(%s)", expr); break;
    }
}

static const char *us_omod(unsigned omod)
{
    static const char *m[] = { "", " * 2.0", " * 4.0", " * 8.0",
                               " * 0.5", " * 0.25", " * 0.125", "" };
    return m[omod & 7];
}

static const char *us_presub_rgb(unsigned op)
{
    static const char *p[] = { "1.0 - 2.0 * cs0.rgb", "cs1.rgb - cs0.rgb",
                               "cs1.rgb + cs0.rgb", "1.0 - cs0.rgb" };
    return p[op & 3];
}

static const char *us_presub_alpha(unsigned op)
{
    static const char *p[] = { "1.0 - 2.0 * as0.a", "as1.a - as0.a",
                               "as1.a + as0.a", "1.0 - as0.a" };
    return p[op & 3];
}

static void us_mask(char *buf, unsigned m)
{
    int n = 0;

    if (m & 1) buf[n++] = 'r';
    if (m & 2) buf[n++] = 'g';
    if (m & 4) buf[n++] = 'b';
    buf[n] = 0;
}

/* ---- instructions --------------------------------------------------- */

/* The r300_texu() arguments for a unit whose texels the shader decodes. */
static void us_texu_args(char *buf, size_t len, const R300State *st,
                         unsigned unit)
{
    uint32_t f1 = r300_reg(st, TX_FORMAT1_0 + 4 * unit);
    uint32_t f0 = r300_reg(st, TX_FILTER0_0 + 4 * unit);
    uint32_t fmt = f1 & 0x1F;
    /* filt: bit 0 magnify linear, bit 1 minify linear, S wrap 6:4,
     * T wrap 10:8, R wrap 14:12 (TX_FILTER0 CLAMP_S/T/R) */
    uint32_t filt = ((f0 & 7) << 4) | (((f0 >> 3) & 7) << 8) | (((f0 >> 6) & 7) << 12);

    if (!tex_is_float(fmt)) {
        filt |= (((f0 >> 9) & 3) != 1 ? 1 : 0) | (((f0 >> 11) & 3) != 1 ? 2 : 0);
    }
    snprintf(buf, len, "0x%xu, %uu, %uu, 0x%xu, %s", fmt, r300_tex_raw_bpp(fmt),
             r300_reg(st, TX_OFFSET_0 + 4 * unit) & 3, filt,
             (f1 >> 22) & 1 ? "true" : "false");
}

static bool unit_is_raw(const R300State *st, unsigned unit)
{
    return (r300_reg(st, TX_ENABLE) >> unit) & 1 &&
           r300_tex_raw_bpp(r300_reg(st, TX_FORMAT1_0 + 4 * unit));
}

/* TX_FORMAT1.TEX_COORD_TYPE of an enabled unit: 0 2D, 1 3D, 2 cube. */
static unsigned unit_dim(const R300State *st, unsigned unit)
{
    unsigned d = (r300_reg(st, TX_FORMAT1_0 + 4 * unit) >> 25) & 3;
    return (r300_reg(st, TX_ENABLE) >> unit) & 1 && d <= 2 ? d : 0;
}

/*
 * Border handling of a unit sampled through a float texture: per axis
 * 1 = clamp to border, 2 = mirror once to border (S bits 1:0, T 3:2,
 * R 5:4), bit 6 linear filtering.  The Metal sampler clamps to
 * transparent black and r300_tex adds TX_BORDER_COLOR for the missing
 * coverage (Metal has no arbitrary border colours).
 */
static unsigned unit_border(const R300State *st, unsigned unit)
{
    uint32_t f0 = r300_reg(st, TX_FILTER0_0 + 4 * unit);
    unsigned bm = 0, axes = unit_dim(st, unit) == 1 ? 3 : 2;

    if (unit_dim(st, unit) == 2) {
        return 0;                       /* cube maps never reach a border */
    }
    for (unsigned a = 0; a < axes; a++) {
        unsigned m = (f0 >> (3 * a)) & 7;
        bm |= (m == 6 ? 1u : m == 7 ? 2u : 0u) << (2 * a);
    }
    if (bm && (((f0 >> 9) & 3) != 1 || ((f0 >> 11) & 3) != 1)) {
        bm |= 64;
    }
    return bm;
}

static bool us_emit_tex(R300Sb *sb, const R300State *st, uint32_t inst,
                        uint32_t *units_used, const char **err)
{
    unsigned src = inst & 0x1F, dst = (inst >> 6) & 0x1F;
    unsigned unit = (inst >> 11) & 0xF, op = (inst >> 15) & 7;
    const char *proj, *bias;

    switch (op) {
    case 0:         /* NOP */
        return true;
    case 2:         /* KIL */
        r300_sb_printf(sb, "    if (any(lessThan(t[%u], vec4(0.0)))) discard;\n", src);
        return true;
    case 1:         /* LD */
        proj = "false"; bias = "0.0";
        break;
    case 3:         /* TXP */
        proj = "true"; bias = "0.0";
        break;
    case 4:         /* TXB: bias in the source's w */
        proj = "false"; bias = "0.0";
        break;
    default:
        *err = "unknown texture instruction";
        return false;
    }
    char bb[16];
    if (op == 4) {
        snprintf(bb, sizeof(bb), "t[%u].w", src);
        bias = bb;
    }
    if (unit_is_raw(st, unit)) {
        char a[128];
        us_texu_args(a, sizeof(a), st, unit);
        r300_sb_printf(sb, "    t[%u] = r300_texu(tex%u, %uu, t[%u], %s, %s, %s);\n",
                       dst, unit, unit, src, proj, bias, a);
    } else {
        r300_sb_printf(sb, "    t[%u] = r300_tex(tex%u, %uu, t[%u], %s, %s, %uu);\n",
                       dst, unit, unit, src, proj, bias, unit_border(st, unit));
    }
    *units_used |= 1u << unit;
    return true;
}

static void us_emit_alu(R300Sb *sb, const R300State *st, unsigned i)
{
    uint32_t ra = r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * i);
    uint32_t aa = r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * i);
    uint32_t ri = r300_reg(st, US_ALU_RGB_INST_0 + 4 * i);
    uint32_t ai = r300_reg(st, US_ALU_ALPHA_INST_0 + 4 * i);
    char s[3][64], e[3][96], sel[96], m[4];
    unsigned rop = (ri >> 23) & 0xF, aop = (ai >> 23) & 0xF;

    /* No instruction index: the text must not depend on where the
     * program sits in instruction memory (see r300_us_to_glsl). */
    r300_sb_printf(sb, "    { // alu: rgb %08x/%08x alpha %08x/%08x\n",
                   ra, ri, aa, ai);
    for (int k = 0; k < 3; k++) {
        us_addr_expr(s[k], sizeof(s[k]), (ra >> (6 * k)) & 0x3F);
        r300_sb_printf(sb, "        vec4 cs%d = %s;", k, s[k]);
        us_addr_expr(s[k], sizeof(s[k]), (aa >> (6 * k)) & 0x3F);
        r300_sb_printf(sb, " vec4 as%d = %s;\n", k, s[k]);
    }
    r300_sb_printf(sb, "        vec3 ps = %s; float psa = %s;\n",
                   us_presub_rgb((ri >> 21) & 3), us_presub_alpha((ai >> 21) & 3));

    /* RGB arguments A, B, C */
    for (int k = 0; k < 3; k++) {
        us_rgb_sel(sel, sizeof(sel), (ri >> (7 * k)) & 0x1F);
        us_mod(e[k], sizeof(e[k]), sel, (ri >> (7 * k + 5)) & 3);
        r300_sb_printf(sb, "        vec3 r%c = %s;\n", 'A' + k, e[k]);
    }
    for (int k = 0; k < 3; k++) {
        us_alpha_sel(sel, sizeof(sel), (ai >> (7 * k)) & 0x1F);
        us_mod(e[k], sizeof(e[k]), sel, (ai >> (7 * k + 5)) & 3);
        r300_sb_printf(sb, "        float a%c = %s;\n", 'A' + k, e[k]);
    }

    /* The dot product feeds both halves (alpha OP_DP takes it). */
    r300_sb_printf(sb, "        float dp = %s;\n",
                   rop == 2 ? "r300_ldot3(rA, rB) + r300_lm1(aA, aB)" :
                   rop == 3 ? "r300_lm1(rA.r, rB.r) + r300_lm1(rA.g, rB.g) + rC.b" :
                              "r300_ldot3(rA, rB)");

    /* Alpha result first: OP_SOP hands it to the RGB half. */
    switch (aop) {
    case 0:  r300_sb_printf(sb, "        float ar = r300_lm1(aA, aB) + aC;\n"); break;
    case 1:  r300_sb_printf(sb, "        float ar = dp;\n"); break;
    case 2:  r300_sb_printf(sb, "        float ar = min(aA, aB);\n"); break;
    case 3:  r300_sb_printf(sb, "        float ar = max(aA, aB);\n"); break;
    case 5:  r300_sb_printf(sb, "        float ar = aC > 0.5 ? aA : aB;\n"); break;
    case 6:  r300_sb_printf(sb, "        float ar = aC >= 0.0 ? aA : aB;\n"); break;
    case 7:  r300_sb_printf(sb, "        float ar = fract(aA);\n"); break;
    case 8:  r300_sb_printf(sb, "        float ar = exp2(aA);\n"); break;
    /* LN2 and RSQ take |src| (R500 ISA; ARB_fp LG2/RSQ semantics) */
    case 9:  r300_sb_printf(sb, "        float ar = log2(abs(aA));\n"); break;
    case 10: r300_sb_printf(sb, "        float ar = 1.0 / aA;\n"); break;
    case 11: r300_sb_printf(sb, "        float ar = inversesqrt(abs(aA));\n"); break;
    default: r300_sb_printf(sb, "        float ar = 0.0;\n"); break;
    }
    switch (rop) {
    case 0:  r300_sb_printf(sb, "        vec3 rr = r300_lm3(rA, rB) + rC;\n"); break;
    case 1:
    case 2:
    case 3:  r300_sb_printf(sb, "        vec3 rr = vec3(dp);\n"); break;
    case 4:  r300_sb_printf(sb, "        vec3 rr = min(rA, rB);\n"); break;
    case 5:  r300_sb_printf(sb, "        vec3 rr = max(rA, rB);\n"); break;
    case 7:  r300_sb_printf(sb, "        vec3 rr = mix(rB, rA, greaterThan(rC, vec3(0.5)));\n"); break;
    case 8:  r300_sb_printf(sb, "        vec3 rr = mix(rB, rA, greaterThanEqual(rC, vec3(0.0)));\n"); break;
    case 9:  r300_sb_printf(sb, "        vec3 rr = fract(rA);\n"); break;
    case 10: r300_sb_printf(sb, "        vec3 rr = vec3(ar);\n"); break;
    default: r300_sb_printf(sb, "        vec3 rr = vec3(0.0);\n"); break;
    }
    r300_sb_printf(sb, "        rr = rr%s; ar = ar%s;\n",
                   us_omod((ri >> 27) & 7), us_omod((ai >> 27) & 7));
    if ((ri >> 30) & 1) {
        r300_sb_printf(sb, "        rr = clamp(rr, 0.0, 1.0);\n");
    }
    if ((ai >> 30) & 1) {
        r300_sb_printf(sb, "        ar = clamp(ar, 0.0, 1.0);\n");
    }

    /* Writes: temporaries, the output FIFO of render target TARGET, and
     * (alpha OMASK_W) the fragment's depth. */
    us_mask(m, (ra >> 23) & 7);
    if (m[0]) {
        r300_sb_printf(sb, "        t[%u].%s = rr.%s;\n", (ra >> 18) & 0x1F, m, m);
    }
    us_mask(m, (ra >> 26) & 7);
    if (m[0]) {
        r300_sb_printf(sb, "        oc[%u].%s = rr.%s;\n", (ra >> 29) & 3, m, m);
    }
    if ((aa >> 23) & 1) {
        r300_sb_printf(sb, "        t[%u].a = ar;\n", (aa >> 18) & 0x1F);
    }
    if ((aa >> 24) & 1) {
        r300_sb_printf(sb, "        oc[%u].a = ar;\n", (aa >> 25) & 3);
    }
    if ((aa >> 27) & 1) {
        r300_sb_printf(sb, "        ow = ar;\n");
    }
    r300_sb_printf(sb, "    }\n");
}

/* Render targets the active program writes (bit n: target n), and
 * whether it writes depth (OMASK_W). */
static uint32_t us_targets_written(const R300State *st, bool *writes_w)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    uint32_t mask = 0;

    *writes_w = false;
    for (unsigned i = 0; i < n; i++) {
        for (unsigned k = 0; k < nodes[i].alu_count; k++) {
            unsigned a = nodes[i].alu_start + k;
            uint32_t ra, aa;
            if (a >= R300_US_MAX_ALU) {
                break;
            }
            ra = r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * a);
            aa = r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * a);
            if ((ra >> 26) & 7) {
                mask |= 1u << ((ra >> 29) & 3);
            }
            if ((aa >> 24) & 1) {
                mask |= 1u << ((aa >> 25) & 3);
            }
            if ((aa >> 27) & 1) {
                *writes_w = true;
            }
        }
    }
    return mask;
}

uint32_t r300_us_out_fmt(const R300State *st, unsigned k)
{
    uint32_t f = r300_reg(st, US_OUT_FMT_0 + 4 * k);
    /* multiwrite targets without their own format use target A's */
    return k && (f & 0x1F) == 15 ? r300_reg(st, US_OUT_FMT_0) : f;
}

uint32_t r300_us_num_targets(const R300State *st)
{
    bool w;
    uint32_t written = us_targets_written(st, &w);
    uint32_t n = 1, mw = ((r300_reg(st, RB3D_CCTL) >> 5) & 3) + 1;

    for (unsigned k = 1; k < R300_US_MAX_TARGETS; k++) {
        if ((written >> k) & 1 && (r300_reg(st, US_OUT_FMT_0 + 4 * k) & 0x1F) != 15) {
            n = k + 1;
        }
    }
    return n > mw ? n : mw;
}

/* ---- library -------------------------------------------------------- */

/*
 * Resource bindings (descriptor set 0).  r300_spirv.c maps them onto the
 * Metal argument table: see R300_BIND_* in r300_us.h.
 */
static const char us_prelude_vs[] =
"layout(std430, set = 0, binding = 2) readonly buffer R300VB {\n"
"    R300Vertex vb[];\n"
"};\n"
"/* ms.xy: the multisample being drawn, as a clip-space shift */\n"
"layout(std140, set = 0, binding = 3) uniform R300MS { vec4 ms; };\n"
"layout(location = 0) out vec4 v0; layout(location = 1) out vec4 v1;\n"
"layout(location = 2) out vec4 v2; layout(location = 3) out vec4 v3;\n"
"layout(location = 4) out vec4 v4; layout(location = 5) out vec4 v5;\n"
"layout(location = 6) out vec4 v6; layout(location = 7) out vec4 v7;\n"
"layout(location = 8) out vec4 v8; layout(location = 9) out vec4 v9;\n"
"layout(location = 10) out vec4 aux;\n";

static const char us_prelude_common[] =
"#version 450\n"
"\n"
"struct R300Vertex { vec4 pos; vec4 v[10]; vec4 aux; float ucp[8]; };\n"
"\n";

static const char us_prelude[] =
"layout(std140, set = 0, binding = 0) uniform R300FSUniforms {\n"
"    vec4 consts[32];\n"
"    vec4 blend_color;\n"
"    uvec4 tex_swz[16];\n"
"    uvec4 tex_info[16];\n"
"    uint cblend, ablend, chanmask, alpha_func;\n"
"    uvec4 out_sel;\n"
"    uint rt_swap32; uint clip_rule; uint rt_endian, rop;\n"
"    ivec4 cliprect[4];\n"
"    uvec4 zinfo;\n"
"    uint zpass_count, poly_en, pad4, pad5;\n"
"    vec4 tex_border[16];\n"
"    vec4 tex_lod[16];\n"
"    uvec4 tex_dim[16];\n"
"    vec4 fog_color;\n"
"    uint fog_blend, depth_src, pad6, pad7;\n"
"    vec4 poly_offset;\n"
"    uvec4 tex_addr[16];\n"
"} u;\n"
"layout(std430, set = 0, binding = 1) buffer R300ZPass { uint zpass; };\n"
"layout(location = 0) in vec4 v0; layout(location = 1) in vec4 v1;\n"
"layout(location = 2) in vec4 v2; layout(location = 3) in vec4 v3;\n"
"layout(location = 4) in vec4 v4; layout(location = 5) in vec4 v5;\n"
"layout(location = 6) in vec4 v6; layout(location = 7) in vec4 v7;\n"
"layout(location = 8) in vec4 v8; layout(location = 9) in vec4 v9;\n"
"layout(location = 10) in vec4 aux;\n"
"\n"
"/* R300 multiplies: 0 times anything (inf, NaN) is 0, the DX9 rule;\n"
"   IEEE's NaN there turned Leopard's menu bar Apple into a dark box. */\n"
"float r300_lm1(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }\n"
"vec3 r300_lm3(vec3 a, vec3 b)\n"
"{\n"
"    return mix(a * b, vec3(0.0), lessThanEqual(min(abs(a), abs(b)), vec3(0.0)));\n"
"}\n"
"float r300_ldot3(vec3 a, vec3 b) { vec3 p = r300_lm3(a, b); return p.x + p.y + p.z; }\n"
"float r300_swz1(vec4 v, uint s)\n"
"{\n"
"    return s < 4u ? v[s] : (s == 5u ? 1.0 : 0.0);\n"
"}\n"
"vec4 r300_tswz(vec4 hw, uint unit)\n"
"{\n"
"    uvec4 s = u.tex_swz[unit];\n"
"    return vec4(r300_swz1(hw, s.x), r300_swz1(hw, s.y),\n"
"                r300_swz1(hw, s.z), r300_swz1(hw, s.w));\n"
"}\n"
"\n"
"/* TX_FORMAT1.SIGNED_* (8-bit components as two's complement) and GAMMA\n"
"   (sRGB) for units sampled through a float texture. */\n"
"vec4 r300_tfix(vec4 hw, uint unit)\n"
"{\n"
"    uint f = u.tex_dim[unit].w;\n"
"    for (uint i = 0u; i < 4u; i++) {\n"
"        if (((f >> i) & 1u) != 0u) {\n"
"            float b = roundEven(hw[i] * 255.0);\n"
"            hw[i] = max((b > 127.0 ? b - 256.0 : b) / 127.0, -1.0);\n"
"        }\n"
"    }\n"
"    if ((f & 16u) != 0u) {\n"
"        vec3 c = clamp(hw.xyz, 0.0, 1.0);\n"
"        hw.xyz = mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92,\n"
"                     lessThanEqual(c, vec3(0.04045)));\n"
"    }\n"
"    return hw;\n"
"}\n"
"/* VRAM holds what the guest CPU wrote (big-endian words), so a 32-bit\n"
"   texel reads byte-reversed compared with the card's own view. */\n"
"vec4 r300_tpost(vec4 raw, uint unit)\n"
"{\n"
"    uint k = u.tex_info[unit].y;\n"
"    return r300_tfix(k == 1u ? raw.abgr : k == 2u ? raw.grba : k == 3u ? raw.bgra : raw, unit);\n"
"}\n"
"/* Share of a bilinear (or nearest) footprint inside [0, n) texels along\n"
"   one axis; m 2 = mirror once first. */\n"
"float r300_bcov(float c, float n, uint m, bool lin)\n"
"{\n"
"    if (m == 0u) return 1.0;\n"
"    if (m == 2u) c = abs(c);\n"
"    if (!lin) { float i = floor(c * n); return i >= 0.0 && i < n ? 1.0 : 0.0; }\n"
"    float p = c * n - 0.5, i0 = floor(p), f = p - i0;\n"
"    return (i0 >= 0.0 && i0 < n ? 1.0 - f : 0.0) + (i0 + 1.0 >= 0.0 && i0 + 1.0 < n ? f : 0.0);\n"
"}\n"
"vec4 r300_border(vec4 hw, vec3 c, vec3 n, uint bm, uint unit)\n"
"{\n"
"    bool lin = (bm & 64u) != 0u;\n"
"    float cov = r300_bcov(c.x, n.x, bm & 3u, lin) * r300_bcov(c.y, n.y, (bm >> 2) & 3u, lin) *\n"
"                r300_bcov(c.z, n.z, (bm >> 4) & 3u, lin);\n"
"    /* clamp to border sampled transparent black outside; mirror once\n"
"       sampled the edge there */\n"
"    return (bm & 42u) != 0u ? mix(u.tex_border[unit], hw, cov)\n"
"                            : hw + u.tex_border[unit] * (1.0 - cov);\n"
"}\n"
"/* Units sampled through a float texture: c is the source temporary\n"
"   (TXP divides by w), bias the TXB bias on top of TX_FILTER1.LOD_BIAS. */\n"
"vec4 r300_tex(sampler2D tx, uint unit, vec4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0u) return vec4(0.0, 0.0, 0.0, 1.0);\n"
"    vec2 uv = proj ? c.xy / c.w : c.xy;\n"
"    vec4 hw = r300_tpost(texture(tx, uv, u.tex_lod[unit].x + bias), unit);\n"
"    if (bm != 0u) {\n"
"        float l = floor(max(textureQueryLod(tx, uv).x, 0.0));\n"
"        vec2 n = max(floor(vec2(u.tex_info[unit].zw) / exp2(l)), vec2(1.0));\n"
"        hw = r300_border(hw, vec3(uv, 0.5), vec3(n, 1.0), bm, unit);\n"
"    }\n"
"    return r300_tswz(hw, unit);\n"
"}\n"
"vec4 r300_tex(sampler3D tx, uint unit, vec4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0u) return vec4(0.0, 0.0, 0.0, 1.0);\n"
"    vec3 uv = proj ? c.xyz / c.w : c.xyz;\n"
"    vec4 hw = r300_tpost(texture(tx, uv, u.tex_lod[unit].x + bias), unit);\n"
"    if (bm != 0u) {\n"
"        float l = floor(max(textureQueryLod(tx, uv).x, 0.0));\n"
"        vec3 n = max(floor(vec3(vec2(u.tex_info[unit].zw), float(u.tex_dim[unit].y)) / exp2(l)),\n"
"                     vec3(1.0));\n"
"        hw = r300_border(hw, uv, n, bm, unit);\n"
"    }\n"
"    return r300_tswz(hw, unit);\n"
"}\n"
"vec4 r300_tex(samplerCube tx, uint unit, vec4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0u) return vec4(0.0, 0.0, 0.0, 1.0);\n"
"    vec4 hw = r300_tpost(texture(tx, c.xyz, u.tex_lod[unit].x + bias), unit);\n"
"    return r300_tswz(hw, unit);\n"
"}\n"
"\n"
"/* Texels the shader decodes itself (r300_tex_raw_bpp): a uint view of\n"
"   guest memory, 4-byte elements for texels of up to 4 bytes.  The card's\n"
"   dword is the TXO_ENDIAN swap of the big-endian dword in VRAM. */\n"
"uint r300_bswap32(uint v)\n"
"{\n"
"    return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);\n"
"}\n"
"uint r300_cswap(uint v, uint e)\n"
"{\n"
"    switch (e & 3u) {\n"
"    case 1u: return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu);\n"
"    case 2u: return r300_bswap32(v);\n"
"    case 3u: return (v << 16) | (v >> 16);\n"
"    default: return v;\n"
"    }\n"
"}\n"
"float r300_h16(uint v) { return unpackHalf2x16(v & 0xffffu).x; }\n"
"float r300_u8(uint v, uint sh) { return float((v >> sh) & 0xffu) / 255.0; }\n"
"float r300_s8(uint v) { return max(float(int(v << 24) >> 24) / 127.0, -1.0); }\n"
"float r300_s16(uint v) { return max(float(int(v << 16) >> 16) / 32767.0, -1.0); }\n"
"float r300_un(uint v, uint sh, uint bits)\n"
"{\n"
"    uint m = (1u << bits) - 1u;\n"
"    return float((v >> sh) & m) / float(m);\n"
"}\n"
"/* YUV_TO_RGB: BT.601 video range; the result is B, G, R in X, Y, Z (the\n"
"   raw texel has Cr, Y, Cb there, as Mesa's swizzles for UYVY imply). */\n"
"vec4 r300_yuv(vec4 c)\n"
"{\n"
"    float y = 1.164 * (c.y - 16.0 / 255.0), cb = c.z - 128.0 / 255.0, cr = c.x - 128.0 / 255.0;\n"
"    return vec4(clamp(y + 2.018 * cb, 0.0, 1.0), clamp(y - 0.813 * cr - 0.391 * cb, 0.0, 1.0),\n"
"                clamp(y + 1.596 * cr, 0.0, 1.0), c.w);\n"
"}\n"
"vec4 r300_tdecode(uint fmt, uvec4 w, uint odd, bool yuv)\n"
"{\n"
"    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);\n"
"    uint v = w.x;\n"
"    switch (fmt) {\n"
"    case 0x02u: c.x = r300_un(v, 0u, 4u); c.y = r300_un(v, 4u, 4u); break;\n"
"    case 0x04u: c.x = r300_un(v, 0u, 16u); c.y = r300_un(v, 16u, 16u); break;\n"
"    case 0x05u: c.x = r300_un(v, 0u, 2u); c.y = r300_un(v, 2u, 3u); c.z = r300_un(v, 5u, 3u); break;\n"
"    case 0x08u: c.x = r300_un(v, 0u, 10u); c.y = r300_un(v, 10u, 11u); c.z = r300_un(v, 21u, 11u); break;\n"
"    case 0x09u: c.x = r300_un(v, 0u, 11u); c.y = r300_un(v, 11u, 11u); c.z = r300_un(v, 22u, 10u); break;\n"
"    case 0x0Du: c = vec4(r300_un(v, 0u, 10u), r300_un(v, 10u, 10u), r300_un(v, 20u, 10u),\n"
"                         r300_un(v, 30u, 2u)); break;\n"
"    case 0x0Eu: c = vec4(r300_un(w.x, 0u, 16u), r300_un(w.x, 16u, 16u),\n"
"                         r300_un(w.y, 0u, 16u), r300_un(w.y, 16u, 16u)); break;\n"
"    case 0x12u: c.x = r300_s8(v); c.y = r300_s8(v >> 8);\n"
"                c.z = sqrt(clamp(1.0 - c.x * c.x - c.y * c.y, 0.0, 1.0)); break;\n"
"    case 0x13u: c = vec4(r300_u8(v, 16u), r300_u8(v, 8u), r300_u8(v, 0u), r300_u8(v, 24u)); break;\n"
"    /* 4:2:2, the card's dword of a pixel pair (after r300_bswap32): VYUY\n"
"       = Y1 Cr Y0 Cb from the low byte up (the 2vuy bytes Cb Y0 Cr Y1 that\n"
"       Leopard's DVD Player uploads); YVYU = Cr Y1 Cb Y0 (yuvs bytes\n"
"       Y0 Cb Y1 Cr, Tiger's DVD Player). */\n"
"    case 0x14u: c = vec4(r300_u8(v, 8u), r300_u8(v, odd != 0u ? 0u : 16u), r300_u8(v, 24u), 1.0); break;\n"
"    case 0x15u: c = vec4(r300_u8(v, 0u), r300_u8(v, odd != 0u ? 8u : 24u), r300_u8(v, 16u), 1.0); break;\n"
"    case 0x16u: c.x = r300_s16(v); break;\n"
"    case 0x17u: c.x = r300_s16(v); c.y = r300_s16(v >> 16); break;\n"
"    case 0x18u: c.x = r300_h16(v); break;\n"
"    case 0x19u: c.x = r300_h16(v); c.y = r300_h16(v >> 16); break;\n"
"    case 0x1Au: c = vec4(r300_h16(w.x), r300_h16(w.x >> 16), r300_h16(w.y), r300_h16(w.y >> 16)); break;\n"
"    case 0x1Bu: c.x = uintBitsToFloat(v); break;\n"
"    case 0x1Cu: c.x = uintBitsToFloat(w.x); c.y = uintBitsToFloat(w.y); break;\n"
"    case 0x1Du: c = uintBitsToFloat(w); break;\n"
"    default: break;\n"
"    }\n"
"    return yuv && (fmt == 0x13u || fmt == 0x14u || fmt == 0x15u) ? r300_yuv(c) : c;\n"
"}\n"
"/* TX_FILTER0 CLAMP_*: wrap, mirror, clamp to last, mirror once to last,\n"
"   clamp to half-border (as last), mirror once to half-border (as\n"
"   last), clamp to border, mirror once to border. */\n"
"int r300_wrap1(int p, int n, uint m, inout bool border)\n"
"{\n"
"    switch (m & 7u) {\n"
"    case 0u: return ((p % n) + n) % n;\n"
"    case 1u: { int q = ((p % (2 * n)) + 2 * n) % (2 * n); return q < n ? q : 2 * n - 1 - q; }\n"
"    case 3u: case 5u: case 7u: {\n"
"        int q = p < 0 ? -1 - p : p;\n"
"        if (q >= n && (m & 7u) == 7u) border = true;\n"
"        return min(q, n - 1);\n"
"    }\n"
"    case 6u: if (p < 0 || p >= n) border = true; return clamp(p, 0, n - 1);\n"
"    default: return clamp(p, 0, n - 1);\n"
"    }\n"
"}\n"
"/* Level l of a unit's chain as the card lays it out (r300_tex_layout):\n"
"   byte offset, row pitch and face/slice size. */\n"
"uint r300_lvl(uint unit, uint l, uint bpp, out uint pitch, out uint fbytes)\n"
"{\n"
"    uvec4 dm = u.tex_dim[unit];\n"
"    uint off = 0u;\n"
"    pitch = 0u; fbytes = 0u;\n"
"    for (uint i = 0u; i <= l; i++) {\n"
"        uint w = max(u.tex_info[unit].z >> i, 1u), h = max(u.tex_info[unit].w >> i, 1u);\n"
"        uint d = dm.x == 2u ? 6u : dm.x == 1u ? max(dm.y >> i, 1u) : 1u;\n"
"        if ((dm.w & 32u) != 0u && h > 1u) h = 1u << uint(findMSB(h - 1u) + 1);\n"
"        uint p = i == 0u ? dm.z : (w * bpp + 15u) & ~15u;\n"
"        if (i == l) { pitch = p; fbytes = p * h; break; }\n"
"        off += p * h * d;\n"
"    }\n"
"    return off;\n"
"}\n";

/*
 * Raw texel words.  Metal binds a uint texture view of the unit's texels
 * (rows of rowel elements); Vulkan (R300_GLSL_VRAM_SSBO) reads VRAM as a
 * storage buffer from the unit's byte address, since a discrete GPU's
 * images cannot alias guest memory.
 */
static const char us_raw_tex[] =
"#define R300_RAWTEX usampler2D\n"
"uvec4 r300_rawld(usampler2D tx, uint unit, uint el, uint bpp)\n"
"{\n"
"    uint rowel = max(u.tex_info[unit].y, 1u);\n"
"    return texelFetch(tx, ivec2(el % rowel, el / rowel), 0);\n"
"}\n";

static const char us_raw_ssbo[] =
"layout(std430, set = 0, binding = 4) readonly buffer R300VRAM { uint vram[]; };\n"
"layout(std430, set = 0, binding = 5) readonly buffer R300Aux { uint rawaux[]; };\n"
"#define R300_RAWTEX uint\n"
"/* tex_addr.x: byte address of the texels in VRAM, or (tex_addr.y set)\n"
"   in the draw's buffer of texels copied out of the GART */\n"
"uint r300_rawword(uint unit, uint w)\n"
"{\n"
"    return u.tex_addr[unit].y != 0u ? rawaux[w] : vram[w];\n"
"}\n"
"/* el counts elements of bpp bytes (4 for texels of up to 4 bytes) */\n"
"uvec4 r300_rawld(uint tx, uint unit, uint el, uint bpp)\n"
"{\n"
"    uint b = (u.tex_addr[unit].x >> 2) + el * (bpp >> 2);\n"
"    uvec4 r = uvec4(r300_rawword(unit, b), 0u, 0u, 0u);\n"
"    if (bpp >= 8u) r.y = r300_rawword(unit, b + 1u);\n"
"    if (bpp >= 16u) { r.z = r300_rawword(unit, b + 2u); r.w = r300_rawword(unit, b + 3u); }\n"
"    return r;\n"
"}\n";

static const char us_prelude2[] =
"/* One texel: p = (x, y, slice or cube face), sz the level's size. */\n"
"vec4 r300_texel(R300_RAWTEX tx, uint unit,\n"
"                uint fmt, uint bpp, uint e, uint filt, ivec3 p, ivec3 sz,\n"
"                uint base, uint pitch, uint fbytes, bool yuv, bool cube)\n"
"{\n"
"    bool border = false;\n"
"    p.x = r300_wrap1(p.x, sz.x, cube ? 2u : filt >> 4, border);\n"
"    p.y = r300_wrap1(p.y, sz.y, cube ? 2u : filt >> 8, border);\n"
"    if (!cube) p.z = r300_wrap1(p.z, sz.z, filt >> 12, border);\n"
"    if (border) return u.tex_border[unit];\n"
"    uint row0 = base + uint(p.z) * fbytes + uint(p.y) * pitch;\n"
"    uvec4 w;\n"
"    uint odd = 0u;\n"
"    if (bpp <= 4u) {\n"
"        uint x = uint(p.x);\n"
"        if (fmt == 0x14u || fmt == 0x15u) { odd = x & 1u; x &= ~1u; }\n"
"        uint off = row0 + x * bpp, el = off >> 2;\n"
"        uint d = r300_cswap(r300_bswap32(r300_rawld(tx, unit, el, 4u).x), e);\n"
"        w = uvec4(d >> ((off & 3u) * 8u), 0u, 0u, 0u);\n"
"    } else {\n"
"        uint el = (row0 + uint(p.x) * bpp) / bpp;\n"
"        uvec4 r = r300_rawld(tx, unit, el, bpp);\n"
"        w = uvec4(r300_cswap(r300_bswap32(r.x), e), r300_cswap(r300_bswap32(r.y), e),\n"
"                  r300_cswap(r300_bswap32(r.z), e), r300_cswap(r300_bswap32(r.w), e));\n"
"    }\n"
"    return r300_tdecode(fmt, w, odd, yuv);\n"
"}\n"
"/* Nearest or (bi/tri)linear lookup in mip level l; q in [0, 1] units. */\n"
"vec4 r300_texl(R300_RAWTEX tx, uint unit,\n"
"               uint fmt, uint bpp, uint e, uint filt, vec3 q, int face,\n"
"               uint l, bool lin, bool yuv)\n"
"{\n"
"    uint pitch, fbytes;\n"
"    uint base = r300_lvl(unit, l, bpp, pitch, fbytes);\n"
"    uint dim = u.tex_dim[unit].x;\n"
"    bool cube = dim == 2u, vol = dim == 1u;\n"
"    ivec3 sz = ivec3(max(int(u.tex_info[unit].z >> l), 1), max(int(u.tex_info[unit].w >> l), 1),\n"
"                     vol ? max(int(u.tex_dim[unit].y >> l), 1) : 1);\n"
"    vec3 st = q * vec3(sz);\n"
"    if (!lin) {\n"
"        ivec3 p = ivec3(floor(st));\n"
"        if (cube) p.z = face; else if (!vol) p.z = 0;\n"
"        return r300_texel(tx, unit, fmt, bpp, e, filt, p, sz, base, pitch, fbytes, yuv, cube);\n"
"    }\n"
"    vec3 f = st - 0.5;\n"
"    ivec3 p = ivec3(floor(f));\n"
"    vec3 a = f - floor(f);\n"
"    if (cube) { p.z = face; a.z = 0.0; } else if (!vol) { p.z = 0; a.z = 0.0; }\n"
"    vec4 r = vec4(0.0);\n"
"    for (int k = 0; k < (vol ? 2 : 1); k++) {\n"
"        ivec3 b = p + ivec3(0, 0, k);\n"
"        vec4 c00 = r300_texel(tx, unit, fmt, bpp, e, filt, b, sz, base, pitch, fbytes, yuv, cube);\n"
"        vec4 c10 = r300_texel(tx, unit, fmt, bpp, e, filt, b + ivec3(1, 0, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        vec4 c01 = r300_texel(tx, unit, fmt, bpp, e, filt, b + ivec3(0, 1, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        vec4 c11 = r300_texel(tx, unit, fmt, bpp, e, filt, b + ivec3(1, 1, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        vec4 m = mix(mix(c00, c10, a.x), mix(c01, c11, a.x), a.y);\n"
"        r += vol ? m * (k == 0 ? 1.0 - a.z : a.z) : m;\n"
"    }\n"
"    return r;\n"
"}\n"
"/* Cube map face and (s, t) of direction r, as GL and the card pick them\n"
"   (faces +X, -X, +Y, -Y, +Z, -Z). */\n"
"vec2 r300_cubeface(vec3 r, out int face)\n"
"{\n"
"    vec3 a = abs(r);\n"
"    float sc, tc, ma;\n"
"    if (a.x >= a.y && a.x >= a.z) {\n"
"        face = r.x >= 0.0 ? 0 : 1; ma = a.x; sc = r.x >= 0.0 ? -r.z : r.z; tc = -r.y;\n"
"    } else if (a.y >= a.z) {\n"
"        face = r.y >= 0.0 ? 2 : 3; ma = a.y; sc = r.x; tc = r.y >= 0.0 ? r.z : -r.z;\n"
"    } else {\n"
"        face = r.z >= 0.0 ? 4 : 5; ma = a.z; sc = r.z >= 0.0 ? r.x : -r.x; tc = -r.y;\n"
"    }\n"
"    ma = max(ma, 1e-20);\n"
"    return vec2(sc / ma, tc / ma) * 0.5 + 0.5;\n"
"}\n"
"/* filt: bit 0 magnify linear, bit 1 minify linear, wrap S 6:4, T 10:8,\n"
"   R 14:12.  Mip level from the texel-space derivatives, TX_FILTER1\n"
"   LOD_BIAS, MAX_MIP_LEVEL..NUM_LEVELS and the mip filter. */\n"
"vec4 r300_texu(R300_RAWTEX tx, uint unit,\n"
"               vec4 c, bool proj, float bias, uint fmt, uint bpp, uint e,\n"
"               uint filt, bool yuv)\n"
"{\n"
"    uint dim = u.tex_dim[unit].x;\n"
"    vec3 q = proj && dim != 2u ? c.xyz / c.w : c.xyz;\n"
"    int face = 0;\n"
"    if (dim == 2u) q = vec3(r300_cubeface(c.xyz, face), 0.0);\n"
"    vec2 st = q.xy * vec2(max(u.tex_info[unit].zw, uvec2(1u)));\n"
"    float rho = max(length(dFdx(st)), length(dFdy(st)));\n"
"    if (u.tex_info[unit].x == 0u) return vec4(0.0, 0.0, 0.0, 1.0);\n"
"    vec4 lo = u.tex_lod[unit];\n"
"    float lod = log2(max(rho, 1e-8)) + lo.x + bias;\n"
"    bool lin = (filt & (lod > 0.0 ? 2u : 1u)) != 0u;\n"
"    vec4 hw;\n"
"    if (lo.w == 0.0) {\n"
"        hw = r300_texl(tx, unit, fmt, bpp, e, filt, q, face, uint(lo.y), lin, yuv);\n"
"    } else {\n"
"        float l = clamp(lod, lo.y, lo.z);\n"
"        if (lo.w == 1.0) {\n"
"            hw = r300_texl(tx, unit, fmt, bpp, e, filt, q, face, uint(roundEven(l)), lin, yuv);\n"
"        } else {\n"
"            uint l0 = uint(floor(l)), l1 = min(l0 + 1u, uint(lo.z));\n"
"            hw = mix(r300_texl(tx, unit, fmt, bpp, e, filt, q, face, l0, lin, yuv),\n"
"                     r300_texl(tx, unit, fmt, bpp, e, filt, q, face, l1, lin, yuv), fract(l));\n"
"        }\n"
"    }\n"
"    return r300_tswz(hw, unit);\n"
"}\n"
"\n"
"float r300_bfactor(uint f, vec4 src, vec4 dst, vec4 k, uint c)\n"
"{\n"
"    switch (f) {\n"
"    case 32u: return 0.0;\n"
"    case 33u: return 1.0;\n"
"    case 34u: return src[c];\n"
"    case 35u: return 1.0 - src[c];\n"
"    case 36u: return dst[c];\n"
"    case 37u: return 1.0 - dst[c];\n"
"    case 38u: return src.a;\n"
"    case 39u: return 1.0 - src.a;\n"
"    case 40u: return dst.a;\n"
"    case 41u: return 1.0 - dst.a;\n"
"    case 42u: return c == 3u ? 1.0 : min(src.a, 1.0 - dst.a);\n"
"    case 43u: return k[c];\n"
"    case 44u: return 1.0 - k[c];\n"
"    case 45u: return k.a;\n"
"    case 46u: return 1.0 - k.a;\n"
"    default: return 1.0;\n"
"    }\n"
"}\n"
"\n"
"float r300_combine(uint ctl, float s, float sf, float d, float df)\n"
"{\n"
"    switch ((ctl >> 12) & 7u) {\n"
"    case 0u: return clamp(s * sf + d * df, 0.0, 1.0);\n"
"    case 1u: return s * sf + d * df;\n"
"    case 2u: return clamp(s * sf - d * df, 0.0, 1.0);\n"
"    case 3u: return s * sf - d * df;\n"
"    case 4u: return min(s, d);\n"
"    case 5u: return max(s, d);\n"
"    case 6u: return clamp(d * df - s * sf, 0.0, 1.0);\n"
"    default: return d * df - s * sf;\n"
"    }\n"
"}\n"
"\n"
"/* Colour (r,g,b,a) <-> the raw RGBA8 view of the colour buffer.  The\n"
"   card writes channel Cn = sel[n] of the colour (0 A, 1 R, 2 G, 3 B;\n"
"   US_OUT_FMT C0..C3_SEL) to byte n of a little-endian word; with swap\n"
"   (COLOR_ENDIAN 2) the word is stored byte-reversed. */\n"
"float r300_chan(vec4 c, uint s)\n"
"{\n"
"    return s == 0u ? c.a : s == 1u ? c.r : s == 2u ? c.g : c.b;\n"
"}\n"
"vec4 r300_to_hw(vec4 c, uvec4 sel)\n"
"{\n"
"    return vec4(r300_chan(c, sel.x), r300_chan(c, sel.y),\n"
"                r300_chan(c, sel.z), r300_chan(c, sel.w));\n"
"}\n"
"vec4 r300_from_hw(vec4 hw, uvec4 sel)\n"
"{\n"
"    vec4 c = vec4(0.0);\n"
"    for (uint n = 0u; n < 4u; n++) {\n"
"        uint s = sel[n];\n"
"        if (s == 0u) c.a = hw[n]; else if (s == 1u) c.r = hw[n];\n"
"        else if (s == 2u) c.g = hw[n]; else c.b = hw[n];\n"
"    }\n"
"    return c;\n"
"}\n"
"vec4 r300_pack(vec4 c, uvec4 sel, uint swap)\n"
"{\n"
"    vec4 hw = r300_to_hw(c, sel);\n"
"    return swap != 0u ? hw.abgr : hw;\n"
"}\n"
"vec4 r300_unpack(vec4 raw, uvec4 sel, uint swap)\n"
"{\n"
"    return r300_from_hw(swap != 0u ? raw.abgr : raw, sel);\n"
"}\n"
"/* RB3D_ROPCNTL.ROP: a GDI ROP2 code as a truth table over (src, dst)\n"
"   bits: bit 3 both set, 2 src only, 1 dst only, 0 neither. */\n"
"uint r300_rop1(uint s, uint d, uint c)\n"
"{\n"
"    return ((c & 8u) != 0u ? s & d : 0u) | ((c & 4u) != 0u ? s & ~d : 0u) |\n"
"           ((c & 2u) != 0u ? ~s & d : 0u) | ((c & 1u) != 0u ? ~s & ~d : 0u);\n"
"}\n"
"uint r300_rop(uint s, uint d, uint c) { return r300_rop1(s, d, c); }\n"
"uvec2 r300_rop(uvec2 s, uvec2 d, uint c)\n"
"{\n"
"    return uvec2(r300_rop1(s.x, d.x, c), r300_rop1(s.y, d.y, c));\n"
"}\n"
"uvec4 r300_rop(uvec4 s, uvec4 d, uint c)\n"
"{\n"
"    return uvec4(r300_rop1(s.x, d.x, c), r300_rop1(s.y, d.y, c),\n"
"                 r300_rop1(s.z, d.z, c), r300_rop1(s.w, d.w, c));\n"
"}\n"
"vec4 r300_rop(vec4 s, vec4 d, uint c)\n"
"{\n"
"    uvec4 a = uvec4(roundEven(clamp(s, 0.0, 1.0) * 255.0)), b = uvec4(roundEven(clamp(d, 0.0, 1.0) * 255.0));\n"
"    return vec4(r300_rop(a, b, c) & 0xffu) / 255.0;\n"
"}\n"
"/* FG_FOG_BLEND.FN: linear, exp, exp2 of the interpolated fog value, or\n"
"   FG_FOG_FACTOR; the result weights the colour against the fog colour.\n"
"   The interpolated value grows with distance (0 = no fog), so linear\n"
"   mode takes its complement: Tiger's GL driver writes the GL fog factor\n"
"   (1 = no fog) from the vertex shader and sets GA_FOG_SCALE/OFFSET to\n"
"   -1/1 to turn it into that value. */\n"
"float r300_fogf(uint fb, float f, float k)\n"
"{\n"
"    switch ((fb >> 1) & 3u) {\n"
"    case 0u: return clamp(1.0 - f, 0.0, 1.0);\n"
"    case 1u: return clamp(exp(-f), 0.0, 1.0);\n"
"    case 2u: return clamp(exp(-f * f), 0.0, 1.0);\n"
"    default: return k;\n"
"    }\n"
"}\n"
"/* RB3D_CBLEND.DISCARD_SRC_PIXELS: skip blending (keep the destination)\n"
"   for source pixels whose alpha and/or colour is all 0 or all 1. */\n"
"bool r300_discard_src(uint cb, vec4 s)\n"
"{\n"
"    uint m = (cb >> 3) & 7u;\n"
"    bool a0 = s.a == 0.0, c0 = all(equal(s.rgb, vec3(0.0)));\n"
"    bool a1 = s.a == 1.0, c1 = all(equal(s.rgb, vec3(1.0)));\n"
"    switch (m) {\n"
"    case 1u: return a0;        case 2u: return c0;        case 3u: return a0 && c0;\n"
"    case 4u: return a1;        case 5u: return c1;        case 6u: return a1 && c1;\n"
"    default: return false;\n"
"    }\n"
"}\n"
"float r300_upk(uint v, uint sh, uint bits, bool sgn)\n"
"{\n"
"    uint m = (1u << bits) - 1u, x = (v >> sh) & m;\n"
"    if (!sgn) return float(x) / float(m);\n"
"    int sx = int(x << (32u - bits)) >> (32u - bits);\n"
"    return max(float(sx) / float(m >> 1), -1.0);\n"
"}\n"
"uint r300_pk(float f, uint sh, uint bits, bool sgn)\n"
"{\n"
"    uint m = (1u << bits) - 1u;\n"
"    if (!sgn) return uint(roundEven(clamp(f, 0.0, 1.0) * float(m))) << sh;\n"
"    return (uint(int(roundEven(clamp(f, -1.0, 1.0) * float(m >> 1)))) & m) << sh;\n"
"}\n"
"uint r300_ph16(float f) { return packHalf2x16(vec2(f, 0.0)) & 0xffffu; }\n"
"\n";

static const char us_prelude_z[] =
"/* Depth/stencil (ZB_*).  The attachment after the colour buffers is the\n"
"   guest buffer as uint words\n"
"   as they lie in memory; DEPTHENDIAN says how the card swapped them.\n"
"   Z24S8 words hold Z in bits 31:8 and stencil in 7:0. */\n"
"uint r300_zswap(uint v, uint e, bool z16)\n"
"{\n"
"    if (z16) return e == 1u || e == 2u ? ((v >> 8) & 0xffu) | ((v & 0xffu) << 8) : v;\n"
"    switch (e) {\n"
"    case 1u: return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu);\n"
"    case 2u: return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);\n"
"    case 3u: return (v << 16) | (v >> 16);\n"
"    default: return v;\n"
"    }\n"
"}\n"
"bool r300_zcmp(uint f, uint a, uint b)\n"
"{\n"
"    switch (f & 7u) {\n"
"    case 0u: return false;  case 1u: return a < b;   case 2u: return a <= b;\n"
"    case 3u: return a == b; case 4u: return a >= b;  case 5u: return a > b;\n"
"    case 6u: return a != b; default: return true;\n"
"    }\n"
"}\n"
"uint r300_sop(uint op, uint s, uint ref)\n"
"{\n"
"    switch (op & 7u) {\n"
"    case 1u: return 0u;                  case 2u: return ref;\n"
"    case 3u: return min(s + 1u, 255u);   case 4u: return s > 0u ? s - 1u : 0u;\n"
"    case 5u: return ~s & 0xffu;          case 6u: return (s + 1u) & 0xffu;\n"
"    case 7u: return (s - 1u) & 0xffu;    default: return s;\n"
"    }\n"
"}\n"
"/* ZB_CNTL: 0 stencil, 1 Z test, 2 Z write, 4 separate back-face stencil.\n"
"   ZB_ZSTENCILCNTL: Z func 2:0; front stencil func/sfail/zpass/zfail at\n"
"   3, 6, 9, 12; back at 15, 18, 21, 24.  STENCILREFMASK: ref, mask,\n"
"   write mask.  Returns the new buffer word; pass says whether the\n"
"   colour buffers take the fragment.  cnt says whether a pass counts as a\n"
"   Z-pass sample: the early test before shading only asks. */\n"
"uint r300_ztest(uint zb, float fz, float zo, bool front, bool cnt, out bool pass)\n"
"{\n"
"    uint cntl = u.zinfo.x, zs = u.zinfo.y, rm = u.zinfo.z, fmt = u.zinfo.w;\n"
"    bool z16 = (fmt & 4u) != 0u;\n"
"    uint w = r300_zswap(z16 ? (zb & 0xffffu) : zb, fmt & 3u, z16);\n"
"    uint zmax = z16 ? 0xffffu : 0xffffffu;\n"
"    uint zold = z16 ? w : (w >> 8), sold = z16 ? 0u : (w & 0xffu);\n"
"    /* roundEven + clamp: 1.0 * 16777215 + 0.5 rounds to 2^24 in float.\n"
"       The polygon offset zo is added in whole depth steps after the\n"
"       conversion, as the card does: added to fz first, a one-step offset\n"
"       (Doom 3's shadow volumes) was lost or doubled by float rounding\n"
"       near the far end of the range, and the volumes z-fought. */\n"
"    int zi = int(min(uint(roundEven(clamp(fz, 0.0, 1.0) * float(zmax))), zmax)) +\n"
"             int(roundEven(zo * float(zmax)));\n"
"    uint znew = uint(clamp(zi, 0, int(zmax)));\n"
"    bool sten = (cntl & 1u) != 0u, zen = (cntl & 2u) != 0u;\n"
"    uint sf = zs >> ((!front && (cntl & 16u) != 0u) ? 15u : 3u);\n"
"    uint ref = rm & 0xffu, mask = (rm >> 8) & 0xffu, wmask = (rm >> 16) & 0xffu;\n"
"    uint z = zold, sn = sold;\n"
"    pass = false;\n"
"    if (sten && !r300_zcmp(sf, ref & mask, sold & mask)) {\n"
"        sn = r300_sop(sf >> 3, sold, ref);\n"
"    } else if (!zen || r300_zcmp(zs, znew, zold)) {\n"
"        if (zen && (cntl & 4u) != 0u) z = znew;\n"
"        if (sten) sn = r300_sop(sf >> 6, sold, ref);\n"
"        if (cnt && u.zpass_count != 0u) atomicAdd(zpass, 1u);\n"
"        pass = true;\n"
"    } else {\n"
"        if (sten) sn = r300_sop(sf >> 9, sold, ref);\n"
"    }\n"
"    sn = (sold & ~wmask) | (sn & wmask);\n"
"    return r300_zswap(z16 ? z : ((z << 8) | sn), fmt & 3u, z16);\n"
"}\n"
"\n"
"bool r300_alpha_pass(uint af, float a)\n"
"{\n"
"    if ((af & (1u << 11)) == 0u) return true;\n"
"    float ref = float(af & 0xffu) / 255.0;\n"
"    switch ((af >> 8) & 7u) {\n"
"    case 0u: return false;\n"
"    case 1u: return a < ref;\n"
"    case 2u: return a == ref;\n"
"    case 3u: return a <= ref;\n"
"    case 4u: return a > ref;\n"
"    case 5u: return a != ref;\n"
"    case 6u: return a >= ref;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n";

/*
 * Early Z/stencil rejection.  r300_shade() runs the whole fragment program
 * -- every texture fetch -- before the test that decides whether the
 * fragment lands, and the buffers are plain colour attachments, so the GPU
 * cannot reject it for us.  When the depth does not come from the program,
 * the test can be made first: a fragment that fails it without changing the
 * buffer's word (a stencil op that rewrites the stencil still has to run,
 * so those take the full path) leaves everything as it was whether or not
 * the program would have discarded it, which is exactly what discard does.
 * Only worth the extra test for programs that fetch textures or are long.
 */
static const char us_early_z[] =
"    if (!(u.depth_src != 0u && R300_WRITES_W)) {\n"
"        bool p0;\n"
"        uint nz = r300_ztest(zb, gl_FragCoord.z, zoff, front, false, p0);\n"
"        if (!p0 && nz == zb) discard;\n"
"    }\n";

/*
 * Colour buffer k's pixel type (r300_cbK_t) and its conversions to and
 * from the blender's colour (r300_cb_view).  Target A reads its channel
 * selects and swap from the uniforms; B-D have them folded into the text.
 * fbinK is the buffer as it was (a framebuffer-fetch input attachment).
 */
static void us_emit_cb(R300Sb *sb, const R300State *st, unsigned k)
{
    uint32_t pitch = r300_reg(st, RB3D_COLORPITCH0 + 4 * k);
    uint32_t cf = (pitch >> 21) & 0xF, e = (pitch >> 19) & 3;
    uint32_t ofr = r300_us_out_fmt(st, k), of = ofr & 0x1F;
    uint32_t bpp, view = r300_cb_view(cf, of, &bpp);
    bool sg[4];
    const char *type, *load, *pad;
    char sel[64], swap[16];
    /* The R300 blends and clamps only its 8-bit-per-channel formats. */
    bool fp = of >= 16 && of <= 21;
    bool blend = view == R300_RTV_RGBA8 || cf == 3 || cf == 4 || cf == 15 ||
                 cf == 9 || (cf == 13 && of == 0);

    if (k == 0) {
        snprintf(sel, sizeof(sel), "u.out_sel");
        snprintf(swap, sizeof(swap), "u.rt_swap32");
    } else {
        snprintf(sel, sizeof(sel), "uvec4(%uu, %uu, %uu, %uu)", (ofr >> 8) & 3,
                 (ofr >> 10) & 3, (ofr >> 12) & 3, (ofr >> 14) & 3);
        snprintf(swap, sizeof(swap), "%uu", e == 2);
    }
    for (int n = 0; n < 4; n++) {
        sg[n] = (ofr >> (16 + n)) & 1;
    }
    if (view == R300_RTV_RGBA8 || view == R300_RTV_NONE) {
        r300_sb_printf(sb,
            "#define r300_cb%u_t vec4\n#define r300_cb%u_o vec4\n#define R300_CB%u_OUT(x) (x)\n"
            "#define R300_CB%u_CLAMP true\n#define R300_CB%u_BLEND true\n"
            "layout(input_attachment_index = %u, set = 0, binding = %u) uniform subpassInput fbin%u;\n"
            "#define R300_FB%u_LOAD subpassLoad(fbin%u)\n"
            "vec4 r300_cb%u_unpack(vec4 fb) { return r300_unpack(fb, %s, %s); }\n"
            "vec4 r300_cb%u_pack(vec4 c) { return r300_pack(c, %s, %s); }\n\n",
            k, k, k, k, k, k, R300_BIND_FB0 + k, k, k, k, k, sel, swap, k, sel, swap);
        return;
    }
    type = view == R300_RTV_RG32U ? "uvec2" : view == R300_RTV_RGBA32U ? "uvec4" : "uint";
    load = view == R300_RTV_RG32U ? ".xy" : view == R300_RTV_RGBA32U ? "" : ".x";
    /* Outputs are uvec4 like the input attachment: Metal wants a fetched
     * attachment's input and output types to agree. */
    pad = view == R300_RTV_RG32U ? "uvec4(x, 0u, 0u)" : view == R300_RTV_RGBA32U ? "(x)"
                                 : "uvec4(x, 0u, 0u, 0u)";
    r300_sb_printf(sb, "// colour buffer %u: format %u, US_OUT_FMT %08x, endian %u, %u bytes\n"
                   "#define r300_cb%u_t %s\n#define r300_cb%u_o uvec4\n#define R300_CB%u_OUT(x) %s\n"
                   "#define R300_CB%u_CLAMP %s\n#define R300_CB%u_BLEND %s\n"
                   "layout(input_attachment_index = %u, set = 0, binding = %u) uniform usubpassInput fbin%u;\n"
                   "#define R300_FB%u_LOAD subpassLoad(fbin%u)%s\n",
                   k, cf, ofr, e, bpp, k, type, k, k, pad, k, fp ? "false" : "true",
                   k, blend ? "true" : "false", k, R300_BIND_FB0 + k, k, k, k, load);

    /* The card's word(s): swaps are their own inverse. */
    char swb[200];
    if (bpp == 1) {
        snprintf(swb, sizeof(swb), "uint w = fb;");
    } else if (bpp == 2) {
        snprintf(swb, sizeof(swb), "%s", e == 1 || e == 2 ?
                 "uint w = ((fb >> 8) & 0xffu) | ((fb & 0xffu) << 8);" : "uint w = fb & 0xffffu;");
    } else if (bpp == 4) {
        snprintf(swb, sizeof(swb), "uint w = r300_cswap(fb, %uu);", e);
    } else if (bpp == 8) {
        snprintf(swb, sizeof(swb), "uvec2 w = uvec2(r300_cswap(fb.x, %uu), r300_cswap(fb.y, %uu));", e, e);
    } else {
        snprintf(swb, sizeof(swb), "uvec4 w = uvec4(r300_cswap(fb.x, %uu), r300_cswap(fb.y, %uu), "
                 "r300_cswap(fb.z, %uu), r300_cswap(fb.w, %uu));", e, e, e, e);
    }

    char ub[640], pb[640];
#define S(n) (sg[n] ? "true" : "false")
    switch (cf) {
    case 9:     /* I8 stores C2 (Mesa r300_translate_out_fmt) */
        snprintf(ub, sizeof(ub), "h.z = r300_upk(w, 0u, 8u, %s);", S(2));
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.z, 0u, 8u, %s);", S(2));
        break;
    case 4:     /* C_5_6_5 */
        snprintf(ub, sizeof(ub), "h = vec4(r300_upk(w, 0u, 5u, false), r300_upk(w, 5u, 6u, false), r300_upk(w, 11u, 5u, false), 1.0);");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 5u, false) | r300_pk(h.y, 5u, 6u, false) | r300_pk(h.z, 11u, 5u, false);");
        break;
    case 3:     /* C_1_5_5_5 */
        snprintf(ub, sizeof(ub), "h = vec4(r300_upk(w, 0u, 5u, false), r300_upk(w, 5u, 5u, false), r300_upk(w, 10u, 5u, false), r300_upk(w, 15u, 1u, false));");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 5u, false) | r300_pk(h.y, 5u, 5u, false) | r300_pk(h.z, 10u, 5u, false) | r300_pk(h.w, 15u, 1u, false);");
        break;
    case 15:    /* C4_4 */
        snprintf(ub, sizeof(ub), "h = vec4(r300_upk(w, 0u, 4u, false), r300_upk(w, 4u, 4u, false), r300_upk(w, 8u, 4u, false), r300_upk(w, 12u, 4u, false));");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 4u, false) | r300_pk(h.y, 4u, 4u, false) | r300_pk(h.z, 8u, 4u, false) | r300_pk(h.w, 12u, 4u, false);");
        break;
    case 13:    /* UV88: C2 low byte, C0 high (Mesa); C_16 / C_16_FP: C0 */
        if (of == 3) {
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0u, 16u, %s);", S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 16u, %s);", S(0));
        } else if (of == 16) {
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x);");
        } else {
            snprintf(ub, sizeof(ub), "h.z = r300_upk(w, 0u, 8u, %s); h.x = r300_upk(w, 8u, 8u, %s);", S(2), S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.z, 0u, 8u, %s) | r300_pk(h.x, 8u, 8u, %s);", S(2), S(0));
        }
        break;
    case 6:     /* 32-bit pixels other than C4_8 */
        switch (of) {
        case 3:
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0u, 16u, %s);", S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 16u, %s);", S(0));
            break;
        case 16:
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x);");
            break;
        case 17:
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w); h.y = r300_h16(w >> 16);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x) | (r300_ph16(h.y) << 16);");
            break;
        case 19:
            snprintf(ub, sizeof(ub), "h.x = uintBitsToFloat(w);");
            snprintf(pb, sizeof(pb), "uint w = floatBitsToUint(h.x);");
            break;
        default:    /* C2_16 */
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0u, 16u, %s); h.y = r300_upk(w, 16u, 16u, %s);", S(0), S(1));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0u, 16u, %s) | r300_pk(h.y, 16u, 16u, %s);", S(0), S(1));
            break;
        }
        break;
    case 10:    /* 64-bit pixels */
        if (of == 18) {
            snprintf(ub, sizeof(ub), "h = vec4(r300_h16(w.x), r300_h16(w.x >> 16), r300_h16(w.y), r300_h16(w.y >> 16));");
            snprintf(pb, sizeof(pb), "uvec2 w = uvec2(r300_ph16(h.x) | (r300_ph16(h.y) << 16), r300_ph16(h.z) | (r300_ph16(h.w) << 16));");
        } else if (of == 20) {
            snprintf(ub, sizeof(ub), "h.x = uintBitsToFloat(w.x); h.y = uintBitsToFloat(w.y);");
            snprintf(pb, sizeof(pb), "uvec2 w = uvec2(floatBitsToUint(h.x), floatBitsToUint(h.y));");
        } else {    /* C4_16 */
            snprintf(ub, sizeof(ub), "h = vec4(r300_upk(w.x, 0u, 16u, %s), r300_upk(w.x, 16u, 16u, %s), r300_upk(w.y, 0u, 16u, %s), r300_upk(w.y, 16u, 16u, %s));",
                     S(0), S(1), S(2), S(3));
            snprintf(pb, sizeof(pb), "uvec2 w = uvec2(r300_pk(h.x, 0u, 16u, %s) | r300_pk(h.y, 16u, 16u, %s), r300_pk(h.z, 0u, 16u, %s) | r300_pk(h.w, 16u, 16u, %s));",
                     S(0), S(1), S(2), S(3));
        }
        break;
    default:    /* 7: 128-bit pixels, C4_32_FP (C2_32_FP fills two) */
        if (of == 20) {
            snprintf(ub, sizeof(ub), "h.x = uintBitsToFloat(w.x); h.y = uintBitsToFloat(w.y);");
            snprintf(pb, sizeof(pb), "uvec4 w = uvec4(floatBitsToUint(h.x), floatBitsToUint(h.y), 0u, 0u);");
        } else {
            snprintf(ub, sizeof(ub), "h = uintBitsToFloat(w);");
            snprintf(pb, sizeof(pb), "uvec4 w = floatBitsToUint(h);");
        }
        break;
    }
#undef S

    /* Pack: the word(s), then the same swap back. */
    char back[200];
    if (bpp == 1) {
        snprintf(back, sizeof(back), "return w & 0xffu;");
    } else if (bpp == 2) {
        snprintf(back, sizeof(back), "%s", e == 1 || e == 2 ?
                 "return ((w >> 8) & 0xffu) | ((w & 0xffu) << 8);" : "return w & 0xffffu;");
    } else if (bpp == 4) {
        snprintf(back, sizeof(back), "return r300_cswap(w, %uu);", e);
    } else if (bpp == 8) {
        snprintf(back, sizeof(back), "return uvec2(r300_cswap(w.x, %uu), r300_cswap(w.y, %uu));", e, e);
    } else {
        snprintf(back, sizeof(back), "return uvec4(r300_cswap(w.x, %uu), r300_cswap(w.y, %uu), "
                 "r300_cswap(w.z, %uu), r300_cswap(w.w, %uu));", e, e, e, e);
    }
    r300_sb_printf(sb,
        "vec4 r300_cb%u_unpack(r300_cb%u_t fb)\n"
        "{\n    %s\n    vec4 h = vec4(0.0, 0.0, 0.0, 1.0);\n    %s\n"
        "    return r300_from_hw(h, %s);\n}\n"
        "r300_cb%u_t r300_cb%u_pack(vec4 c)\n"
        "{\n    vec4 h = r300_to_hw(c, %s);\n    %s\n    %s\n}\n\n",
        k, k, swb, ub, sel, k, k, sel, pb, back);
}

/* Blend, ROP and write mask for colour buffer k from shader output src. */
static void us_emit_target(R300Sb *sb, unsigned k, int src)
{
    if (src < 0) {
        r300_sb_printf(sb, "    o.c%u = fb%u;      // target %u: not written\n", k, k, k);
        return;
    }
    r300_sb_printf(sb,
        "    {   // target %u\n"
        "        vec4 s = oc[%d];\n"
        "        vec4 d = r300_cb%u_unpack(fb%u);\n"
        "        vec4 res = R300_CB%u_CLAMP ? clamp(s, 0.0, 1.0) : s;\n"
        "        if (R300_CB%u_BLEND && (u.cblend & 1u) != 0u) {\n"
        "            uint ab = (u.cblend & 2u) != 0u ? u.ablend : u.cblend;\n"
        "            uint cs = (u.cblend >> 16) & 63u, cd = (u.cblend >> 24) & 63u;\n"
        "            uint as_ = (ab >> 16) & 63u, ad = (ab >> 24) & 63u;\n"
        "            for (uint c = 0u; c < 3u; c++)\n"
        "                res[c] = r300_combine(u.cblend, s[c], r300_bfactor(cs, s, d, u.blend_color, c),\n"
        "                                      d[c], r300_bfactor(cd, s, d, u.blend_color, c));\n"
        "            res.a = r300_combine(ab, s.a, r300_bfactor(as_, s, d, u.blend_color, 3u),\n"
        "                                 d.a, r300_bfactor(ad, s, d, u.blend_color, 3u));\n"
        "            if (keep) res = d;\n"
        "        }\n"
        "        if ((u.rop & 4u) != 0u)\n"
        "            res = r300_cb%u_unpack(r300_rop(r300_cb%u_pack(R300_CB%u_CLAMP ? clamp(s, 0.0, 1.0) : s),\n"
        "                                            fb%u, (u.rop >> 8) & 15u));\n"
        "        /* RB3D_COLOR_CHANNEL_MASK: B G R A in bits 0..3 */\n"
        "        uint cm = u.chanmask;\n"
        "        res = vec4((cm & 4u) != 0u ? res.r : d.r, (cm & 2u) != 0u ? res.g : d.g,\n"
        "                   (cm & 1u) != 0u ? res.b : d.b, (cm & 8u) != 0u ? res.a : d.a);\n"
        "        o.c%u = r300_cb%u_pack(res);\n"
        "    }\n",
        k, src, k, k, k, k, k, k, k, k, k, k);
}

char *r300_us_to_glsl(const R300State *st, const R300FSDesc *desc,
                      uint32_t flags, const char **err)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    uint32_t units = 0;
    R300Sb body, sb;
    bool writes_w;
    uint32_t written = us_targets_written(st, &writes_w);
    uint32_t nt = r300_us_num_targets(st);
    uint32_t mw = ((r300_reg(st, RB3D_CCTL) >> 5) & 3) + 1;
    uint32_t ucp = r300_reg(st, VAP_CLIP_CNTL) & 0x3F;

    unsigned work = 0;

    *err = NULL;
    r300_sb_init(&body);
    for (unsigned i = 0; i < n; i++) {
        const USNode *nd = &nodes[i];
        work += nd->alu_count + 4 * nd->tex_count;

        if (nd->alu_start + nd->alu_count > R300_US_MAX_ALU ||
            nd->tex_start + nd->tex_count > R300_US_MAX_TEX) {
            *err = "US program runs past instruction memory";
            r300_sb_free(&body);
            return NULL;
        }
        /*
         * Counts only, no instruction addresses: the renderers key their
         * shader and pipeline caches on this text, and the driver loads
         * the same program at different places in instruction memory.
         * With the addresses in, UT2004 converted one shader hundreds of
         * times (~7 ms each, a stutter on every new combination).
         */
        r300_sb_printf(&body, "    // node %u: %u tex, %u alu\n", i,
                       nd->tex_count, nd->alu_count);
        for (unsigned k = 0; k < nd->tex_count; k++) {
            if (!us_emit_tex(&body, st, r300_reg(st, US_TEX_INST_0 +
                                                 4 * (nd->tex_start + k)),
                             &units, err)) {
                r300_sb_free(&body);
                return NULL;
            }
        }
        for (unsigned k = 0; k < nd->alu_count; k++) {
            us_emit_alu(&body, st, nd->alu_start + k);
        }
    }

    bool early = units != 0 || work >= 8;

    r300_sb_init(&sb);
    r300_sb_printf(&sb, "%s// flags %x\n", us_prelude_common, flags);

    /* Vertex stage: post-transform vertices as they are (shifted by the
     * sample offset when multisampling), plus the user clip plane
     * distances when VAP_CLIP_CNTL enables any. */
    r300_sb_printf(&sb,
        "#ifdef R300_VS\n%s"
        "%s"
        "void main()\n"
        "{\n"
        "    R300Vertex x = vb[gl_VertexIndex];\n"
        "    gl_Position = vec4(x.pos.xy + ms.xy * x.pos.w, x.pos.zw);\n"
        "    v0 = x.v[0]; v1 = x.v[1]; v2 = x.v[2]; v3 = x.v[3];\n"
        "    v4 = x.v[4]; v5 = x.v[5]; v6 = x.v[6]; v7 = x.v[7];\n"
        "    v8 = x.v[8]; v9 = x.v[9]; aux = x.aux;\n"
        "%s"
        "}\n\n"
        "#else\n\n",
        us_prelude_vs,
        ucp ? "out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[6]; };\ninvariant gl_Position;\n"
              : "invariant gl_Position;\n",
        ucp ? "    for (int i = 0; i < 6; i++) gl_ClipDistance[i] = x.ucp[i];\n" : "");

    r300_sb_printf(&sb, "%s%s%s", us_prelude,
                   flags & R300_GLSL_VRAM_SSBO ? us_raw_ssbo : us_raw_tex,
                   us_prelude2);
    for (unsigned k = 0; k < nt; k++) {
        us_emit_cb(&sb, st, k);
    }
    r300_sb_printf(&sb, "%s", us_prelude_z);

    /* Fragment outputs: colour buffers 0..nt-1, then the depth buffer. */
    R300Sb outs, fbp, fbarg, fbin, fbout, zsel;
    r300_sb_init(&outs);
    r300_sb_init(&fbp);
    r300_sb_init(&fbarg);
    r300_sb_init(&fbin);
    r300_sb_init(&fbout);
    r300_sb_init(&zsel);
    for (unsigned k = 0; k < nt; k++) {
        r300_sb_printf(&outs, "r300_cb%u_t c%u; ", k, k);
        r300_sb_printf(&fbp, "%sr300_cb%u_t fb%u", k ? ", " : "", k, k);
        r300_sb_printf(&fbarg, ", fb%u", k);
        r300_sb_printf(&fbin, "    r300_cb%u_t fb%u = R300_FB%u_LOAD;\n", k, k, k);
        r300_sb_printf(&fbout, "layout(location = %u) out r300_cb%u_o o_c%u;\n", k, k, k);
        r300_sb_printf(&zsel, "    o_c%u = R300_CB%u_OUT(pass ? c.c%u : fb%u);\n", k, k, k, k);
    }
    r300_sb_printf(&sb, "struct R300FOut { %s};\n%s\n", outs.buf, fbout.buf);

    /* Texture units: unit k at binding R300_BIND_TEX0 + k. */
    for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
        if (units & (1u << k)) {
            static const char *dims[3] = { "sampler2D", "sampler3D", "samplerCube" };
            if (unit_is_raw(st, k)) {
                if (flags & R300_GLSL_VRAM_SSBO) {
                    r300_sb_printf(&sb, "#define tex%u %uu\n", k, k);
                } else {
                    r300_sb_printf(&sb, "layout(set = 0, binding = %u) uniform usampler2D tex%u;\n",
                                   R300_BIND_TEX0 + k, k);
                }
            } else {
                r300_sb_printf(&sb, "layout(set = 0, binding = %u) uniform %s tex%u;\n",
                               R300_BIND_TEX0 + k, dims[unit_dim(st, k)], k);
            }
        }
    }
    r300_sb_printf(&sb,
        "\n#define R300_WRITES_W %s\n"
        "R300FOut r300_shade(%s, inout float ow)\n"
        "{\n"
        "    /* SC_CLIP_RULE: a 16-entry truth table over which of the four\n"
        "       clip rectangles contain the pixel. */\n"
        "    {\n"
        "        ivec2 p = ivec2(gl_FragCoord.xy);\n"
        "        uint idx = 0u;\n"
        "        for (uint i = 0u; i < 4u; i++) {\n"
        "            ivec4 r = u.cliprect[i];\n"
        "            if (p.x >= r.x && p.y >= r.y && p.x <= r.z && p.y <= r.w) idx |= 1u << i;\n"
        "        }\n"
        "        if (((u.clip_rule >> idx) & 1u) == 0u) discard;\n"
        "    }\n"
        "    vec4 t[32];\n"
        "    for (int i = 0; i < 32; i++) t[i] = vec4(0.0);\n"
        "    vec4 vin[10] = vec4[10](v0, v1, v2, v3, v4, v5, v6, v7, v8, v9);\n",
        writes_w ? "true" : "false", fbp.buf);
    for (unsigned k = 0; k < R300_US_NUM_TEMPS; k++) {
        if (desc->route[k] >= 0) {
            r300_sb_printf(&sb, "    t[%u] = vin[%d];\n", k, desc->route[k]);
        }
    }
    r300_sb_printf(&sb,
        "    vec4 oc[4] = vec4[4](vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0),\n"
        "                         vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0));\n%s",
        body.buf ? body.buf : "");
    r300_sb_printf(&sb,
        "    /* FG: fog, then the alpha test on render target A. */\n"
        "    if ((u.fog_blend & 1u) != 0u) {\n"
        "        float f = r300_fogf(u.fog_blend, aux.x, u.fog_color.w);\n"
        "        for (int k = 0; k < 4; k++) oc[k].rgb = mix(u.fog_color.rgb, oc[k].rgb, f);\n"
        "    }\n"
        "    if (!r300_alpha_pass(u.alpha_func, clamp(oc[0].a, 0.0, 1.0))) discard;\n"
        "    bool keep = r300_discard_src(u.cblend, clamp(oc[0], 0.0, 1.0));\n"
        "    R300FOut o;\n");
    for (unsigned k = 0; k < nt; k++) {
        int src = mw > 1 && k < mw ? 0 : ((written >> k) & 1) || k == 0 ? (int)k : -1;
        us_emit_target(&sb, k, src);
    }
    r300_sb_printf(&sb,
        "    return o;\n"
        "}\n"
        "\n"
        "#ifdef R300_FS_Z\n"
        "layout(input_attachment_index = %u, set = 0, binding = %u) uniform usubpassInput zin;\n"
        "layout(location = %u) out uvec4 o_z;\n"
        "#endif\n"
        "\n"
        "void main()\n"
        "{\n"
        "%s"
        "    float ow = gl_FragCoord.z;\n"
        "#ifndef R300_FS_Z\n"
        "    R300FOut c = r300_shade(%s, ow);\n"
        "    if (u.zpass_count != 0u) atomicAdd(zpass, 1u);\n",
        nt, R300_BIND_FB0 + nt, nt, fbin.buf, fbarg.buf + 2);
    for (unsigned k = 0; k < nt; k++) {
        r300_sb_printf(&sb, "    o_c%u = R300_CB%u_OUT(c.c%u);\n", k, k, k);
    }
    r300_sb_printf(&sb,
        "#else\n"
        "    uint zb = subpassLoad(zin).x;\n"
        "    bool front = gl_FrontFacing;\n"
        "    /* SU_POLY_OFFSET_*: slope in depth per pixel */\n"
        "    float slope = max(abs(dFdx(gl_FragCoord.z)), abs(dFdy(gl_FragCoord.z)));\n"
        "    float zoff = 0.0;\n"
        "    if (front ? (u.poly_en & 1u) != 0u : (u.poly_en & 2u) != 0u)\n"
        "        zoff = front ? u.poly_offset.x * slope + u.poly_offset.y\n"
        "                     : u.poly_offset.z * slope + u.poly_offset.w;\n"
        "%s"
        "    R300FOut c = r300_shade(%s, ow);\n"
        "    /* FG_DEPTH_SRC: the program's OMASK_W output replaces the depth */\n"
        "    float fz = (u.depth_src != 0u && R300_WRITES_W) ? ow : gl_FragCoord.z;\n"
        "    bool pass;\n"
        "    o_z = uvec4(r300_ztest(zb, fz, zoff, front, true, pass), 0u, 0u, 0u);\n"
        "%s"
        "#endif\n"
        "}\n"
        "#endif\n",
        early ? us_early_z : "", fbarg.buf + 2, zsel.buf);
    r300_sb_free(&outs);
    r300_sb_free(&fbp);
    r300_sb_free(&fbarg);
    r300_sb_free(&fbin);
    r300_sb_free(&fbout);
    r300_sb_free(&zsel);
    r300_sb_free(&body);
    return r300_sb_steal(&sb);
}

char *r300_us_disasm(const R300State *st)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    R300Sb sb;

    r300_sb_init(&sb);
    r300_sb_printf(&sb, "US config %08x offset %08x, %u node(s)\n",
                   r300_reg(st, US_CONFIG), r300_reg(st, US_CODE_OFFSET), n);
    for (unsigned i = 0; i < n; i++) {
        const USNode *nd = &nodes[i];
        r300_sb_printf(&sb, " node %u\n", i);
        for (unsigned k = 0; k < nd->tex_count; k++) {
            uint32_t t = r300_reg(st, US_TEX_INST_0 + 4 * (nd->tex_start + k));
            r300_sb_printf(&sb, "  tex%-2u op%u t%u <- unit%u t%u\n",
                           nd->tex_start + k, (t >> 15) & 7, (t >> 6) & 0x1F,
                           (t >> 11) & 0xF, t & 0x1F);
        }
        for (unsigned k = 0; k < nd->alu_count; k++) {
            unsigned a = nd->alu_start + k;
            r300_sb_printf(&sb, "  alu%-2u rgb %08x %08x  alpha %08x %08x\n", a,
                           r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * a),
                           r300_reg(st, US_ALU_RGB_INST_0 + 4 * a),
                           r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * a),
                           r300_reg(st, US_ALU_ALPHA_INST_0 + 4 * a));
        }
    }
    return r300_sb_steal(&sb);
}

/*
 * A remembered program: the registers its generation read, their values,
 * the other inputs and the text.  Games and the window server cycle
 * through a handful of programs, thousands of draws a second; generating
 * the text each time (printf-heavy) was about a sixth of the emulated
 * CPU's time in Chess.
 */
typedef struct USMemo {
    uint32_t flags, id;
    R300FSDesc desc;
    unsigned n;
    uint16_t *idx;
    uint32_t *val;
    char *glsl;
} USMemo;

#define US_MEMO_SLOTS 64
static USMemo us_memo[US_MEMO_SLOTS];   /* most recently used first */
static uint32_t us_memo_next_id = 1;

static bool us_memo_match(const USMemo *m, const R300State *st,
                          const R300FSDesc *desc, uint32_t flags)
{
    if (!m->glsl || m->flags != flags ||
        memcmp(&m->desc, desc, sizeof(*desc)) != 0) {
        return false;
    }
    for (unsigned i = 0; i < m->n; i++) {
        if (st->regs[m->idx[i]] != m->val[i]) {
            return false;
        }
    }
    return true;
}

char *r300_us_glsl_cached(const R300State *st, const R300FSDesc *desc,
                          uint32_t flags, uint32_t *id, const char **err)
{
    for (unsigned k = 0; k < US_MEMO_SLOTS; k++) {
        if (us_memo_match(&us_memo[k], st, desc, flags)) {
            USMemo hit = us_memo[k];
            memmove(&us_memo[1], &us_memo[0], k * sizeof(USMemo));
            us_memo[0] = hit;
            *id = hit.id;
            *err = NULL;
            return strdup(hit.glsl);
        }
    }

    memset(us_rec.seen, 0, sizeof(us_rec.seen));
    us_rec.n = 0;
    us_rec.overflow = false;
    us_rec.on = true;
    char *glsl = r300_us_to_glsl(st, desc, flags, err);
    us_rec.on = false;
    *id = 0;
    if (!glsl || us_rec.overflow) {
        return glsl;
    }

    USMemo *last = &us_memo[US_MEMO_SLOTS - 1];
    free(last->idx);
    free(last->val);
    free(last->glsl);
    memmove(&us_memo[1], &us_memo[0], (US_MEMO_SLOTS - 1) * sizeof(USMemo));
    USMemo *m = &us_memo[0];
    m->flags = flags;
    m->desc = *desc;
    m->n = us_rec.n;
    m->idx = malloc(m->n * sizeof(*m->idx) + 1);
    m->val = malloc(m->n * sizeof(*m->val) + 1);
    for (unsigned i = 0; i < m->n; i++) {
        m->idx[i] = us_rec.idx[i];
        m->val[i] = st->regs[us_rec.idx[i]];
    }
    m->glsl = strdup(glsl);
    m->id = us_memo_next_id++;
    *id = m->id;
    return glsl;
}
