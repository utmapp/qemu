/*
 * R200 vertex programs: decoding, disassembly and execution.
 *
 * The chip can transform each vertex with a small program the driver
 * uploads, instead of with the fixed matrices and lights.  Every Mac
 * OpenGL program that asks for ARB_vertex_program uses this, and some
 * games (Halo) use it for everything they draw, so without it their
 * screens stay black.
 *
 * The programs live in the chip's vector memory, which also holds the
 * fixed-function matrices -- the two are never in use at once.  An
 * instruction is four 32-bit words: what to do and where to put it,
 * then three source operands.  There is no branching and no loop; the
 * program runs from first to last instruction for every vertex.
 *
 * Instruction word 0:
 *   [5:0]   opcode, within the engine chosen by bit 6
 *   [6]     0 = vector engine, 1 = maths engine
 *   [7]     macro (a two-cycle MAD; it means the same as MAD)
 *   [11:8]  where the result goes: 0 temporary, 3 address register,
 *           4 position, 5 colour, 6 texture coordinate, 7 fog, 8 point size
 *   [19:13] which register of that kind
 *   [23:20] which of x, y, z, w to write
 *
 * Source words 1-3:
 *   [3:0]   what it reads: 0 temporary, 1 vertex attribute, 2 constant,
 *           9 nothing
 *   [4]     add the address register to the index
 *   [12:5]  index
 *   [15:13] [18:16] [21:19] [24:22]  which component feeds x, y, z, w
 *           (0-3 = x,y,z,w; 4 = 0.0; 5 = 1.0)
 *   [28:25] negate x, y, z, w, after the choice above
 *
 * There is no public register manual for this chip; this follows the
 * open-source Linux driver (Mesa's r200_vertprog.c and r200_reg.h,
 * reverse-engineered from ATI's own driver) cross-checked against AMD's
 * published manual for the next chip generation, which shares the
 * instruction word.  Fields that neither source pins down are marked
 * where they are used.
 *
 * Copyright (c) 2026 Spartan0285
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include <math.h>
#include "hw/display/ppc_mac_gpu_vp.h"

/* Where things sit in the chip's vector memory, in vec4 entries. */
#define VP_PROG0    0x080       /* instructions 0-63    */
#define VP_PROG1    0x180       /* instructions 64-127  */
#define VP_PARAM0   0x000       /* constants 0-95       */
#define VP_PARAM1   0x100       /* constants 96-191     */

/* Vector engine. */
enum {
    VP_OP_NOP = 0, VP_OP_DOT = 1, VP_OP_MUL = 2, VP_OP_ADD = 3,
    VP_OP_MAD = 4, VP_OP_DST = 5, VP_OP_FRC = 6, VP_OP_MAX = 7,
    VP_OP_MIN = 8, VP_OP_SGE = 9, VP_OP_SLT = 10, VP_OP_ARL = 13,
};

/* Maths engine (bit 6 set; these are the values with bit 6 included). */
enum {
    VP_OP_EXP = 65, VP_OP_LOG = 66, VP_OP_EXP_E = 67, VP_OP_LIT = 68,
    VP_OP_POW = 69, VP_OP_RCP = 70, VP_OP_RCP_FF = 71, VP_OP_RSQ = 72,
    VP_OP_RSQ_FF = 73, VP_OP_EX2 = 75, VP_OP_LG2 = 76,
};

#define VP_OP_MAD_2 128         /* a MAD that takes two cycles */

static const char *vp_op_name(uint32_t op)
{
    switch (op) {
    case VP_OP_NOP:   return "nop";
    case VP_OP_DOT:   return "dot";
    case VP_OP_MUL:   return "mul";
    case VP_OP_ADD:   return "add";
    case VP_OP_MAD:   return "mad";
    case VP_OP_DST:   return "dst";
    case VP_OP_FRC:   return "frc";
    case VP_OP_MAX:   return "max";
    case VP_OP_MIN:   return "min";
    case VP_OP_SGE:   return "sge";
    case VP_OP_SLT:   return "slt";
    case VP_OP_ARL:   return "arl";
    case VP_OP_EXP:   return "exp";
    case VP_OP_LOG:   return "log";
    case VP_OP_EXP_E: return "expe";
    case VP_OP_LIT:   return "lit";
    case VP_OP_POW:   return "pow";
    case VP_OP_RCP:   return "rcp";
    case VP_OP_RCP_FF: return "rcpf";
    case VP_OP_RSQ:   return "rsq";
    case VP_OP_RSQ_FF: return "rsqf";
    case VP_OP_EX2:   return "ex2";
    case VP_OP_LG2:   return "lg2";
    case VP_OP_MAD_2: return "mad2";
    default:          return "?";
    }
}

static const char *vp_dest_name(uint32_t class)
{
    switch (class) {
    case VP_DST_TEMP:      return "r";
    case VP_DST_ADDR:      return "a0";
    case VP_DST_POS:       return "pos";
    case VP_DST_COLOR:     return "col";
    case VP_DST_TEXCOORD:  return "tex";
    case VP_DST_FOG:       return "fog";
    case VP_DST_POINTSIZE: return "psize";
    default:               return "dst?";
    }
}

/* How many source operands an instruction reads. */
static int vp_num_sources(uint32_t op)
{
    switch (op) {
    case VP_OP_NOP:                             return 0;
    case VP_OP_MAD: case VP_OP_MAD_2:           return 3;
    case VP_OP_DOT: case VP_OP_MUL: case VP_OP_ADD: case VP_OP_DST:
    case VP_OP_MAX: case VP_OP_MIN: case VP_OP_SGE: case VP_OP_SLT:
                                                return 2;
    default:                                    return 1;
    }
}

void ppc_mac_gpu_vp_decode(const uint32_t w[4], PPCMacGPUVPInst *out)
{
    memset(out, 0, sizeof(*out));
    out->op        = w[0] & 0xFF;      /* opcode with the engine and macro bits */
    out->dst_class = (w[0] >> 8) & 0xF;
    out->dst_index = (w[0] >> 13) & 0x7F;
    out->write     = (w[0] >> 20) & 0xF;
    out->sources   = vp_num_sources(out->op);
    for (int i = 0; i < 3; i++) {
        uint32_t s = w[1 + i];
        out->src[i].class    = s & 0xF;
        out->src[i].relative = (s >> 4) & 1;
        out->src[i].index    = (s >> 5) & 0xFF;
        for (int c = 0; c < 4; c++) {
            out->src[i].select[c] = (s >> (13 + 3 * c)) & 7;
            out->src[i].negate[c] = (s >> (25 + c)) & 1;
        }
    }
}

/* "r3.xy", "pos.xyzw" */
static void vp_print_dest(const PPCMacGPUVPInst *in, char *buf, size_t len)
{
    char mask[5] = { 0 };
    int n = 0;
    for (int c = 0; c < 4; c++) {
        if (in->write & (1 << c)) {
            mask[n++] = "xyzw"[c];
        }
    }
    if (in->dst_class == VP_DST_ADDR || in->dst_class == VP_DST_POS ||
        in->dst_class == VP_DST_FOG || in->dst_class == VP_DST_POINTSIZE) {
        snprintf(buf, len, "%s.%s", vp_dest_name(in->dst_class), mask);
    } else {
        snprintf(buf, len, "%s%u.%s", vp_dest_name(in->dst_class),
                 in->dst_index, mask);
    }
}

/* "-v0.xyzw", "c12.wwww", "r1.xy01" */
static void vp_print_source(const PPCMacGPUVPSrc *s, char *buf, size_t len)
{
    static const char kind[16] = { 'r', 'v', 'c' };
    char sw[16];
    int n = 0;
    bool all_neg = s->negate[0] && s->negate[1] && s->negate[2] && s->negate[3];

    if (s->class == VP_SRC_NONE) {
        snprintf(buf, len, "-");
        return;
    }
    for (int c = 0; c < 4; c++) {
        if (s->negate[c] && !all_neg) {
            sw[n++] = '-';
        }
        sw[n++] = s->select[c] < 4 ? "xyzw"[s->select[c]]
                : s->select[c] == VP_SEL_ZERO ? '0'
                : s->select[c] == VP_SEL_ONE ? '1' : '?';
    }
    sw[n] = 0;
    snprintf(buf, len, "%s%c%u%s.%s", all_neg ? "-" : "",
             s->class < 3 ? kind[s->class] : '?', s->index,
             s->relative ? "[a0]" : "", sw);
}

void ppc_mac_gpu_vp_disasm(const PPCMacGPUVPInst *in, char *buf, size_t len)
{
    char dst[32], src[3][40];
    int n;

    vp_print_dest(in, dst, sizeof(dst));
    n = snprintf(buf, len, "%-5s %s", vp_op_name(in->op), dst);
    for (int i = 0; i < in->sources && n < (int)len; i++) {
        vp_print_source(&in->src[i], src[i], sizeof(src[i]));
        n += snprintf(buf + n, len - n, ", %s", src[i]);
    }
}

/* Where instruction `i` of a program lives in vector memory. */
uint32_t ppc_mac_gpu_vp_inst_addr(uint32_t i)
{
    return i < 64 ? VP_PROG0 + i : VP_PROG1 + (i - 64);
}

/* Where constant `i` lives. */
uint32_t ppc_mac_gpu_vp_const_addr(uint32_t i)
{
    return i < 96 ? VP_PARAM0 + i : VP_PARAM1 + (i - 96);
}

/* ---- running a program ---- */

static float vp_src_comp(const PPCMacGPUVPState *st, const PPCMacGPUVPSrc *s,
                         int c, int a0)
{
    uint32_t sel = s->select[c];
    float v;
    uint32_t i;

    if (sel == VP_SEL_ZERO) {
        v = 0.0f;
    } else if (sel == VP_SEL_ONE) {
        v = 1.0f;
    } else {
        i = s->index + (s->relative ? a0 : 0);
        switch (s->class) {
        case VP_SRC_TEMP:  v = st->temp[i & 15][sel]; break;
        case VP_SRC_ATTR:  v = st->in[i & 15][sel]; break;
        case VP_SRC_CONST:
            /* Out of range reads zero, as the chip documents. */
            v = i < PPC_MAC_GPU_VP_CONSTS ? st->constant[i][sel] : 0.0f;
            break;
        default:           v = 0.0f; break;
        }
    }
    return s->negate[c] ? -v : v;
}

static void vp_read(const PPCMacGPUVPState *st, const PPCMacGPUVPSrc *s,
                    int a0, float out[4])
{
    for (int c = 0; c < 4; c++) {
        out[c] = vp_src_comp(st, s, c, a0);
    }
}

/* Where a result goes.  An output the chip isn't emitting is harmless. */
static float *vp_dest(PPCMacGPUVPState *st, const PPCMacGPUVPInst *in)
{
    switch (in->dst_class) {
    case VP_DST_TEMP:      return st->temp[in->dst_index & 15];
    case VP_DST_POS:       return st->out_pos;
    case VP_DST_COLOR:     return in->dst_index ? st->out_color1 : st->out_color0;
    case VP_DST_TEXCOORD:  return in->dst_index < 6 ? st->out_tex[in->dst_index] : NULL;
    case VP_DST_FOG:       return &st->out_fog[0];
    case VP_DST_POINTSIZE: return &st->out_psize[0];
    case VP_DST_ADDR:      return NULL;          /* handled by the caller */
    default:               return NULL;
    }
}

static float vp_rcp(float x) { return x == 0.0f ? INFINITY : 1.0f / x; }
static float vp_rsq(float x)
{
    float a = fabsf(x);
    return a == 0.0f ? INFINITY : 1.0f / sqrtf(a);
}

void ppc_mac_gpu_vp_run(PPCMacGPUVPState *st, const PPCMacGPUVPInst *prog,
                        uint32_t count)
{
    int a0 = 0;

    for (uint32_t pc = 0; pc < count; pc++) {
        const PPCMacGPUVPInst *in = &prog[pc];
        float s0[4], s1[4], s2[4], r[4] = { 0, 0, 0, 0 };
        float *dst;

        if (in->sources > 0) { vp_read(st, &in->src[0], a0, s0); }
        if (in->sources > 1) { vp_read(st, &in->src[1], a0, s1); }
        if (in->sources > 2) { vp_read(st, &in->src[2], a0, s2); }

        switch (in->op) {
        case VP_OP_NOP:
            continue;
        case VP_OP_ARL:
            /* The address register: a whole number, used to index constants
             * (Halo's skinning picks a bone matrix with it). */
            a0 = (int)s0[0];
            continue;
        case VP_OP_MUL:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] * s1[c]; }
            break;
        case VP_OP_ADD:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] + s1[c]; }
            break;
        case VP_OP_MAD:
        case VP_OP_MAD_2:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] * s1[c] + s2[c]; }
            break;
        case VP_OP_DOT: {
            float d = s0[0] * s1[0] + s0[1] * s1[1] + s0[2] * s1[2] + s0[3] * s1[3];
            for (int c = 0; c < 4; c++) { r[c] = d; }
            break;
        }
        case VP_OP_MAX:
            for (int c = 0; c < 4; c++) { r[c] = MAX(s0[c], s1[c]); }
            break;
        case VP_OP_MIN:
            for (int c = 0; c < 4; c++) { r[c] = MIN(s0[c], s1[c]); }
            break;
        case VP_OP_SGE:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] >= s1[c] ? 1.0f : 0.0f; }
            break;
        case VP_OP_SLT:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] < s1[c] ? 1.0f : 0.0f; }
            break;
        case VP_OP_FRC:
            for (int c = 0; c < 4; c++) { r[c] = s0[c] - floorf(s0[c]); }
            break;
        case VP_OP_DST:              /* distance vector, for attenuation */
            r[0] = 1.0f;
            r[1] = s0[1] * s1[1];
            r[2] = s0[2];
            r[3] = s1[3];
            break;
        case VP_OP_RCP:
        case VP_OP_RCP_FF:
            for (int c = 0; c < 4; c++) { r[c] = vp_rcp(s0[0]); }
            break;
        case VP_OP_RSQ:
        case VP_OP_RSQ_FF:
            for (int c = 0; c < 4; c++) { r[c] = vp_rsq(s0[0]); }
            break;
        case VP_OP_EX2:
            for (int c = 0; c < 4; c++) { r[c] = exp2f(s0[0]); }
            break;
        case VP_OP_LG2:
            for (int c = 0; c < 4; c++) { r[c] = s0[0] > 0.0f ? log2f(s0[0]) : -INFINITY; }
            break;
        case VP_OP_EXP_E:
            for (int c = 0; c < 4; c++) { r[c] = expf(s0[0]); }
            break;
        case VP_OP_EXP:              /* the older, split form */
            r[0] = exp2f(floorf(s0[0]));
            r[1] = s0[0] - floorf(s0[0]);
            r[2] = exp2f(s0[0]);
            r[3] = 1.0f;
            break;
        case VP_OP_LOG: {
            float a = fabsf(s0[0]);
            float e = a == 0.0f ? -INFINITY : floorf(log2f(a));
            r[0] = e;
            r[1] = a == 0.0f ? 1.0f : a / exp2f(e);
            r[2] = a == 0.0f ? -INFINITY : log2f(a);
            r[3] = 1.0f;
            break;
        }
        case VP_OP_POW:              /* base in x, exponent in z */
            for (int c = 0; c < 4; c++) { r[c] = powf(s0[0], s0[2]); }
            break;
        case VP_OP_LIT: {            /* lighting coefficients */
            float d = s0[0], sp = s0[1], p = s0[3];
            p = MIN(MAX(p, -127.9961f), 127.9961f);
            r[0] = 1.0f;
            r[1] = MAX(d, 0.0f);
            r[2] = d > 0.0f ? powf(MAX(sp, 0.0f), p) : 0.0f;
            r[3] = 1.0f;
            break;
        }
        default:
            st->unknown_op = in->op;
            continue;
        }

        dst = vp_dest(st, in);
        if (!dst) {
            continue;
        }
        for (int c = 0; c < 4; c++) {
            if (in->write & (1 << c)) {
                dst[c] = r[c];
            }
        }
        if (in->dst_class == VP_DST_POS) {
            st->wrote_pos = true;
        }
    }
}
