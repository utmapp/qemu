/*
 * R300-family Programmable Vertex Shader (PVS) interpreter.
 *
 * Semantics: AMD "R5xx Acceleration" v1.5, 7.5.7 (vector ops) and 7.5.8
 * (math ops), and the PVS instruction tables that follow.  Math ops read
 * the W channel of their operands (after swizzle): A.w is the argument or
 * base, B.w a clamp, C.w an exponent.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_pvs.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "r300_sb.h"

/* Vector engine opcodes */
enum {
    VE_NOP = 0, VE_DOT_PRODUCT, VE_MULTIPLY, VE_ADD, VE_MULTIPLY_ADD,
    VE_DISTANCE_VECTOR, VE_FRACTION, VE_MAXIMUM, VE_MINIMUM,
    VE_SET_GREATER_THAN_EQUAL, VE_SET_LESS_THAN, VE_MULTIPLYX2_ADD,
    VE_MULTIPLY_CLAMP, VE_FLT2FIX_DX, VE_FLT2FIX_DX_RND,
    /* R5xx, harmless to implement */
    VE_COND_MUX_EQ = 23, VE_COND_MUX_GT, VE_COND_MUX_GTE,
    VE_SET_GREATER_THAN, VE_SET_EQUAL, VE_SET_NOT_EQUAL,
};

/* Math engine opcodes */
enum {
    ME_NOP = 0, ME_EXP_BASE2_DX, ME_LOG_BASE2_DX, ME_EXP_BASEE_FF,
    ME_LIGHT_COEFF_DX, ME_POWER_FUNC_FF, ME_RECIP_DX, ME_RECIP_FF,
    ME_RECIP_SQRT_DX, ME_RECIP_SQRT_FF, ME_MULTIPLY, ME_EXP_BASE2_FULL_DX,
    ME_LOG_BASE2_FULL_DX, ME_POWER_FUNC_FF_CLAMP_B,
    ME_POWER_FUNC_FF_CLAMP_B1, ME_POWER_FUNC_FF_CLAMP_01, ME_SIN, ME_COS,
    ME_LOG_BASE2_IEEE, ME_RECIP_IEEE, ME_RECIP_SQRT_IEEE,
};

enum { SRC_TEMP = 0, SRC_INPUT, SRC_CONST, SRC_ALT };
enum { DST_TEMP = 0, DST_A0, DST_OUT, DST_OUT_REPL_X, DST_ALT, DST_INPUT };

typedef struct PVSRegs {
    float temp[R300_PVS_NUM_TEMPS][4];
    float alt[R300_PVS_NUM_ALT_TEMPS][4];
    int a0[4];
} PVSRegs;

typedef struct PVSCtx {
    const R300PVSProgram *prog;
    const float (*in)[4];
    float (*out)[4];
    PVSRegs r;
    int al;                 /* fixed-point loop index of the innermost loop */
    uint32_t unsup;
} PVSCtx;

static const float zero4[4];

static const float *pvs_mem(PVSCtx *c, unsigned type, int index)
{
    switch (type) {
    case SRC_TEMP:
        return index >= 0 && index < R300_PVS_NUM_TEMPS ? c->r.temp[index] : zero4;
    case SRC_INPUT:
        return index >= 0 && index < R300_PVS_NUM_INPUTS ? c->in[index] : zero4;
    case SRC_CONST:
        return index >= 0 && index <= c->prog->max_const ?
               c->prog->consts[index] : zero4;
    default:
        return index >= 0 && index < R300_PVS_NUM_ALT_TEMPS ? c->r.alt[index] : zero4;
    }
}

/* ADDR_MODE: 0 absolute, 1 relative to A0.ADDR_SEL, 2 relative to the
 * loop index (R5xx guide, PVS source and destination operands). */
static int pvs_rel(PVSCtx *c, unsigned mode, unsigned sel, unsigned offset)
{
    if (mode == 1) {
        return (int)offset + c->r.a0[sel & 3];
    }
    if (mode == 2) {
        return (int)offset + c->al;
    }
    return offset;
}

static int pvs_addr(PVSCtx *c, uint32_t s, unsigned offset)
{
    return pvs_rel(c, ((s >> 4) & 1) | ((s >> 31) << 1), s >> 29, offset);
}

static float pvs_select(const float *v, unsigned sel)
{
    return sel < 4 ? v[sel] : sel == 5 ? 1.0f : 0.0f;
}

/* A regular source operand, swizzled, with abs and negate applied. */
static void pvs_src(PVSCtx *c, uint32_t s, float o[4])
{
    const float *v = pvs_mem(c, s & 3, (s & ((1u << 4) | (1u << 31))) ?
                                       pvs_addr(c, s, (s >> 5) & 0xFF) :
                                       (int)((s >> 5) & 0xFF));
    /* Swizzle selects 0-3 a component, 5 one, 4/6/7 zero; abs and negate
     * are sign-bit operations (the hottest code of the interpreter). */
    float ext[8] = { v[0], v[1], v[2], v[3], 0.0f, 1.0f, 0.0f, 0.0f };
    uint32_t clear = (s & (1u << 3)) ? 0x7FFFFFFFu : 0xFFFFFFFFu;

    for (int i = 0; i < 4; i++) {
        uint32_t bits;
        float x = ext[(s >> (13 + 3 * i)) & 7];
        memcpy(&bits, &x, 4);
        bits = (bits & clear) ^ (((s >> (25 + i)) & 1u) << 31);
        memcpy(&o[i], &bits, 4);
    }
}

static float pvs_pow_ff(float base, float e)
{
    /* Special cases, in the documented order of detection. */
    if (base == 0.0f) {
        return e < 0.0f ? INFINITY : 0.0f;
    }
    if (e == 0.0f) {
        return 1.0f;
    }
    if (base < 0.0f) {
        return -powf(-base, e);
    }
    return powf(base, e);
}

/* Math engine: scalar operands A, B, C (the .w channels). */
static void pvs_math(PVSCtx *c, unsigned op, float a, float b, float cc,
                     float r[4])
{
    float x;

    switch (op) {
    case ME_EXP_BASE2_DX:
        r[0] = exp2f(floorf(a));
        r[1] = a > 128.0f ? 0.0f : a - floorf(a);
        r[2] = exp2f(a);
        r[3] = 1.0f;
        return;
    case ME_LOG_BASE2_DX:
        if (a == 0.0f) {
            r[0] = -FLT_MAX; r[1] = 1.0f; r[2] = -FLT_MAX; r[3] = 1.0f;
        } else {
            int e;
            float m = frexpf(fabsf(a), &e);    /* m in [0.5, 1) */
            r[0] = (float)(e - 1);
            r[1] = m * 2.0f;
            r[2] = log2f(fabsf(a));
            r[3] = 1.0f;
        }
        return;
    case ME_LIGHT_COEFF_DX:
        r[0] = 1.0f;
        r[1] = fmaxf(b, 0.0f);
        r[2] = b > 0.0f ? pvs_pow_ff(fmaxf(a, 0.0f),
                                     fminf(fmaxf(cc, -128.0f), 128.0f)) : 0.0f;
        r[3] = 1.0f;
        return;
    case ME_EXP_BASEE_FF:       x = expf(a); break;
    case ME_POWER_FUNC_FF:      x = pvs_pow_ff(a, cc); break;
    case ME_RECIP_DX:           x = a == 0.0f ? FLT_MAX : 1.0f / a; break;
    case ME_RECIP_FF:           x = a == 0.0f ? 0.0f : 1.0f / a; break;
    case ME_RECIP_SQRT_DX:      x = a == 0.0f ? FLT_MAX : 1.0f / sqrtf(fabsf(a)); break;
    case ME_RECIP_SQRT_FF:      x = a == 0.0f ? 0.0f : 1.0f / sqrtf(fabsf(a)); break;
    case ME_MULTIPLY:           x = a * b; break;
    case ME_EXP_BASE2_FULL_DX:  x = exp2f(a); break;
    case ME_LOG_BASE2_FULL_DX:  x = a == 0.0f ? -FLT_MAX : log2f(fabsf(a)); break;
    case ME_POWER_FUNC_FF_CLAMP_B:
        x = a < b ? 0.0f : pvs_pow_ff(a, cc);
        break;
    case ME_POWER_FUNC_FF_CLAMP_B1:
        x = a < b ? 0.0f : a > 1.0f ? 1.0f : pvs_pow_ff(a, cc);
        break;
    case ME_POWER_FUNC_FF_CLAMP_01:
        x = a <= 0.0f ? 0.0f : a > 1.0f ? 1.0f : pvs_pow_ff(a, cc);
        break;
    case ME_SIN:
    case ME_COS: {
        /* Inputs outside [-pi, pi] clamp: sin 0, cos -1. */
        float t = fminf(fmaxf(a, (float)-M_PI), (float)M_PI);
        x = op == ME_SIN ? sinf(t) : cosf(t);
        break;
    }
    case ME_LOG_BASE2_IEEE:     x = log2f(fabsf(a)); break;
    case ME_RECIP_IEEE:         x = 1.0f / a; break;
    case ME_RECIP_SQRT_IEEE:    x = 1.0f / sqrtf(fabsf(a)); break;
    case ME_NOP:                x = 0.0f; break;
    default:
        c->unsup |= R300_PVS_UNSUP_OPCODE;
        x = 0.0f;
        break;
    }
    r[0] = r[1] = r[2] = r[3] = x;
}

static void pvs_vector(PVSCtx *c, unsigned op, const float *a, const float *b,
                       const float *cc, float r[4])
{
    switch (op) {
    case VE_DOT_PRODUCT: {
        float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        r[0] = r[1] = r[2] = r[3] = d;
        return;
    }
    case VE_DISTANCE_VECTOR:
        r[0] = 1.0f; r[1] = a[1] * b[1]; r[2] = a[2]; r[3] = b[3];
        return;
    case VE_MULTIPLY_CLAMP: {
        float x;
        if (cc[3] < a[3] * b[3]) {
            x = cc[3];
        } else if (cc[0] >= a[0] * b[0]) {
            x = cc[0];
        } else {
            x = a[0] * b[0];
        }
        r[0] = r[1] = r[2] = r[3] = x;
        return;
    }
    }
    for (int i = 0; i < 4; i++) {
        float x;
        switch (op) {
        case VE_NOP:                    x = 0.0f; break;
        case VE_MULTIPLY:               x = a[i] * b[i]; break;
        case VE_ADD:                    x = a[i] + b[i]; break;
        case VE_MULTIPLY_ADD:           x = a[i] * b[i] + cc[i]; break;
        case VE_FRACTION:               x = a[i] - floorf(a[i]); break;
        case VE_MAXIMUM:                x = fmaxf(a[i], b[i]); break;
        case VE_MINIMUM:                x = fminf(a[i], b[i]); break;
        case VE_SET_GREATER_THAN_EQUAL: x = a[i] >= b[i]; break;
        case VE_SET_LESS_THAN:          x = a[i] < b[i]; break;
        case VE_MULTIPLYX2_ADD:         x = 2.0f * (a[i] * b[i]) + cc[i]; break;
        case VE_FLT2FIX_DX:             x = floorf(a[i]); break;
        case VE_FLT2FIX_DX_RND:         x = floorf(a[i] + 0.5f); break;
        case VE_COND_MUX_EQ:            x = a[i] == 0.0f ? b[i] : cc[i]; break;
        case VE_COND_MUX_GT:            x = a[i] > 0.0f ? b[i] : cc[i]; break;
        case VE_COND_MUX_GTE:           x = a[i] >= 0.0f ? b[i] : cc[i]; break;
        case VE_SET_GREATER_THAN:       x = a[i] > b[i]; break;
        case VE_SET_EQUAL:              x = a[i] == b[i]; break;
        case VE_SET_NOT_EQUAL:          x = a[i] != b[i]; break;
        default:
            c->unsup |= R300_PVS_UNSUP_OPCODE;
            x = 0.0f;
            break;
        }
        r[i] = x;
    }
}

static void pvs_sat(float r[4])
{
    for (int i = 0; i < 4; i++) {
        r[i] = fminf(fmaxf(r[i], 0.0f), 1.0f);
    }
}

static void pvs_write(PVSCtx *c, uint32_t d0, const float r[4])
{
    unsigned type = (d0 >> 8) & 0xF;
    unsigned mask = (d0 >> 20) & 0xF;
    /* ADDR_MODE_0 is bit 31, ADDR_MODE_1 bit 12 */
    unsigned mode = ((d0 >> 31) & 1) | (((d0 >> 12) & 1) << 1);
    int rel = pvs_rel(c, mode, d0 >> 29, (d0 >> 13) & 0x7F);
    unsigned index = rel < 0 ? ~0u : (unsigned)rel;
    float *dst;

    switch (type) {
    case DST_TEMP:
        dst = index < R300_PVS_NUM_TEMPS ? c->r.temp[index] : NULL;
        break;
    case DST_A0:
        for (int i = 0; i < 4; i++) {
            if (mask & (1u << i)) {
                float f = floorf(r[i]);
                c->r.a0[i] = f < -256.0f ? -256 : f > 255.0f ? 255 : (int)f;
            }
        }
        return;
    case DST_OUT:
    case DST_OUT_REPL_X:
        dst = index < R300_PVS_NUM_OUTPUTS ? c->out[index] : NULL;
        break;
    case DST_ALT:
        dst = index < R300_PVS_NUM_ALT_TEMPS ? c->r.alt[index] : NULL;
        break;
    default:
        return;
    }
    if (!dst) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        if (mask & (1u << i)) {
            dst[i] = type == DST_OUT_REPL_X ? r[0] : r[i];
        }
    }
}

/*
 * The instruction after pc, following the flow control instructions
 * (AMD R5xx guide 7.5.5; VAP_PVS_FLOW_CNTL_*), once pc has run.  A
 * loop's end and a subroutine's
 * return are live only while that loop or call is.
 *
 *   ADDRS: 7:0 activation instruction (the last before the redirect),
 *          15:8 JUMP/JSR target or LOOP count, 23:16 last instruction of
 *          the loop or subroutine, 31:24 loop start or return address.
 *   LOOP_INDEX: 7:0 initial loop index, 15:8 signed step.
 */
typedef struct PVSLoop {
    int fc;                 /* its FLOW_CNTL instruction */
    unsigned count;         /* iterations left */
    int saved_al;           /* the enclosing loop's index */
} PVSLoop;

static unsigned pvs_flow(const R300PVSProgram *prog, PVSCtx *c, unsigned pc,
                         PVSLoop *loop, unsigned *nloop, int *jsr, unsigned *njsr)
{
    if (*nloop) {
        int k = loop[*nloop - 1].fc;
        if (((prog->fc_addrs[k] >> 16) & 0xFF) == pc) {
            c->al += (int8_t)(prog->fc_loop[k] >> 8);
            if (--loop[*nloop - 1].count) {
                return (prog->fc_addrs[k] >> 24) & 0xFF;
            }
            c->al = loop[--*nloop].saved_al;
            return pc + 1;
        }
    }
    if (*njsr) {
        int k = jsr[*njsr - 1];
        if (((prog->fc_addrs[k] >> 16) & 0xFF) == pc) {
            --*njsr;
            return (prog->fc_addrs[k] >> 24) & 0xFF;
        }
    }
    for (int k = 0; k < 16; k++) {
        unsigned op = (prog->fc_opc >> (2 * k)) & 3;
        uint32_t a = prog->fc_addrs[k];

        if (!op || (a & 0xFF) != pc) {
            continue;
        }
        switch (op) {
        case 1:                                 /* JUMP */
            return (a >> 8) & 0xFF;
        case 2:                                 /* LOOP */
            if (!((a >> 8) & 0xFF) || *nloop == 16) {
                return pc + 1;
            }
            loop[*nloop].fc = k;
            loop[*nloop].count = (a >> 8) & 0xFF;
            loop[*nloop].saved_al = c->al;
            ++*nloop;
            c->al = prog->fc_loop[k] & 0xFF;
            return (a >> 24) & 0xFF;
        default:                                /* JSR */
            if (*njsr == 16) {
                return pc + 1;
            }
            jsr[(*njsr)++] = k;
            return (a >> 8) & 0xFF;
        }
    }
    return pc + 1;
}

/* One instruction. */
static void pvs_exec(PVSCtx *c, const uint32_t *d)
{
    uint32_t d0 = d[0];
    unsigned op = d0 & 0x3F;
    bool math = (d0 >> 6) & 1;
    bool macro = (d0 >> 7) & 1;
    bool dual = !math && !macro && ((d0 >> 28) & 1);
    float a[4], b[4], cc[4] = { 0 }, r[4];     /* dual: no third operand */

    if ((d0 >> 26) & 1) {
        c->unsup |= R300_PVS_UNSUP_PRED;
    }
    pvs_src(c, d[1], a);
    pvs_src(c, d[2], b);
    if (!dual) {
        pvs_src(c, d[3], cc);
    }

    if (math) {
        pvs_math(c, op, a[3], b[3], cc[3], r);
        if ((d0 >> 25) & 1) {
            pvs_sat(r);
        }
        pvs_write(c, d0, r);
        return;
    }

    if (macro) {
        /* Two-clock MAD/M2X_ADD with three distinct temporaries. */
        pvs_vector(c, (op & 1) ? VE_MULTIPLYX2_ADD : VE_MULTIPLY_ADD,
                   a, b, cc, r);
    } else {
        pvs_vector(c, op, a, b, cc, r);
    }
    if ((d0 >> 24) & 1) {
        pvs_sat(r);
    }

    if (dual) {
        /* The third source dword describes a math op writing ATRM 0-3. */
        uint32_t s = d[3];
        unsigned mop = ((s >> 21) & 0xF) | (((s >> 2) & 1) << 4);
        const float *v = pvs_mem(c, s & 3, pvs_addr(c, s, (s >> 5) & 0xFF));
        float x = pvs_select(v, (s >> 13) & 7);
        float y = pvs_select(v, (s >> 16) & 7);
        float mr[4];

        if (s & (1u << 3)) {
            x = fabsf(x);
            y = fabsf(y);
        }
        if (s & (1u << 25)) {
            x = -x;
        }
        if (s & (1u << 26)) {
            y = -y;
        }
        pvs_math(c, mop, x, y, y, mr);
        if ((d0 >> 25) & 1) {
            pvs_sat(mr);
        }
        pvs_write(c, d0, r);   /* vector result first... */
        {
            unsigned comp = (s >> 27) & 3;
            c->r.alt[(s >> 19) & 3][comp] = mr[comp];  /* ...then math */
        }
        return;
    }
    pvs_write(c, d0, r);
}

uint32_t r300_pvs_run(const R300PVSProgram *prog,
                      const float in[R300_PVS_NUM_INPUTS][4],
                      float out[R300_PVS_NUM_OUTPUTS][4])
{
    PVSCtx c;

    memset(&c.r, 0, sizeof(c.r));
    c.prog = prog;
    c.in = in;
    c.out = out;
    c.unsup = 0;

    /* Flow control state: active loops (innermost last) and subroutine
     * returns, as indices of their FLOW_CNTL instruction. */
    PVSLoop loop[16];
    int jsr[16];
    unsigned nloop = 0, njsr = 0, steps = 0;
    c.al = 0;

    for (unsigned pc = prog->first_inst, next;
         pc <= prog->last_inst && pc < R300_PVS_MAX_INSTS; pc = next) {
        if (++steps > 65536) {
            c.unsup |= R300_PVS_UNSUP_FLOW;     /* runaway loop */
            break;
        }
        pvs_exec(&c, &prog->code[pc * 4]);
        next = prog->fc_opc ? pvs_flow(prog, &c, pc, loop, &nloop, jsr, &njsr)
                            : pc + 1;
    }
    return c.unsup;
}

void r300_pvs_disasm_inst(const uint32_t d[4], char *buf, unsigned len)
{
    static const char *dst_type[] = { "t", "a0", "o", "ox", "at", "i", "?", "?" };
    static const char *src_type[] = { "t", "i", "c", "at" };
    static const char swz[] = "xyzw01??";
    uint32_t d0 = d[0];
    int n;

    n = snprintf(buf, len, "%s%s op%-2u %s%u.%s%s%s%s <-",
                 (d0 >> 6) & 1 ? "ME" : (d0 >> 7) & 1 ? "MACRO" : "VE",
                 !((d0 >> 6) & 1) && ((d0 >> 28) & 1) ? "+dual" : "",
                 d0 & 0x3F, dst_type[(d0 >> 8) & 7], (d0 >> 13) & 0x7F,
                 (d0 >> 20) & 1 ? "x" : "", (d0 >> 21) & 1 ? "y" : "",
                 (d0 >> 22) & 1 ? "z" : "", (d0 >> 23) & 1 ? "w" : "");
    for (int k = 1; k < 4 && n > 0 && (unsigned)n < len; k++) {
        uint32_t s = d[k];
        char sw[5];
        for (int i = 0; i < 4; i++) {
            sw[i] = swz[(s >> (13 + 3 * i)) & 7];
            if (s & (1u << (25 + i))) {
                sw[i] = sw[i] >= 'a' ? sw[i] - 32 : sw[i];   /* upper = negated */
            }
        }
        sw[4] = 0;
        n += snprintf(buf + n, len - n, " %s%s[%u%s].%s",
                      s & 8 ? "|" : "", src_type[s & 3], (s >> 5) & 0xFF,
                      s & 16 ? "+a0" : "", sw);
    }
}

/*
 * ---- PVS -> GLSL, for running the vertex program on the host GPU ----
 *
 * A translation of what r300_pvs_run() interprets, instruction for
 * instruction, for straight-line programs.  Registers become locals
 * (tN temporaries, atN alternate temporaries, oN outputs, iN inputs, a0),
 * constants the uniform array vs.c[].  Anything the translation does not
 * cover makes it return false and the draw stays on the interpreter.
 */
typedef struct PVSGlsl {
    const R300PVSProgram *prog;
    R300Sb *sb;
    uint32_t in_used;
    bool temp[R300_PVS_NUM_TEMPS], alt[R300_PVS_NUM_ALT_TEMPS];
    bool ok;
} PVSGlsl;

static const char pvs_comp[] = "xyzw";

/* The unswizzled operand, as an expression. */
static void pg_base(PVSGlsl *g, uint32_t s, char *buf, size_t len)
{
    unsigned type = s & 3, idx = (s >> 5) & 0xFF;
    unsigned mode = ((s >> 4) & 1) | ((s >> 31) << 1);

    if (mode == 1) {
        /* A0-relative: only for constants (bounds-checked at run time) */
        if (type != SRC_CONST) {
            g->ok = false;
        }
        snprintf(buf, len, "pvs_c(%d + a0[%u])", (int)idx, (s >> 29) & 3);
        return;
    }
    /* Absolute; mode 2 adds the loop index, 0 without flow control. */
    switch (type) {
    case SRC_TEMP:
        if (idx < R300_PVS_NUM_TEMPS) {
            g->temp[idx] = true;
            snprintf(buf, len, "t%u", idx);
            return;
        }
        break;
    case SRC_INPUT:
        if (idx < R300_PVS_NUM_INPUTS) {
            g->in_used |= 1u << idx;
            snprintf(buf, len, "i%u", idx);
            return;
        }
        break;
    case SRC_CONST:
        /* constants past the count are zero in the uniforms (gpu_vs) */
        if (idx < 256) {
            snprintf(buf, len, "vs.c[%u]", idx);
            return;
        }
        break;
    default:
        if (idx < R300_PVS_NUM_ALT_TEMPS) {
            g->alt[idx] = true;
            snprintf(buf, len, "at%u", idx);
            return;
        }
        break;
    }
    snprintf(buf, len, "vec4(0.0)");
}

static void pg_comp(char *buf, size_t len, unsigned sel, bool abs_, bool neg)
{
    const char *x = sel < 4 ? (const char[][5]){ "pb.x", "pb.y", "pb.z", "pb.w" }[sel]
                  : sel == 5 ? "1.0" : "0.0";
    snprintf(buf, len, "%s%s%s%s%s", neg ? "-(" : "", abs_ ? "abs(" : "", x,
             abs_ ? ")" : "", neg ? ")" : "");
}

/* var = the swizzled operand s (abs, negate) */
static void pg_src(PVSGlsl *g, const char *var, uint32_t s)
{
    char base[64], c[4][24];

    pg_base(g, s, base, sizeof(base));
    for (int i = 0; i < 4; i++) {
        pg_comp(c[i], sizeof(c[i]), (s >> (13 + 3 * i)) & 7, s & (1u << 3),
                s & (1u << (25 + i)));
    }
    r300_sb_printf(g->sb, "    pb = %s; %s = vec4(%s, %s, %s, %s);\n",
                   base, var, c[0], c[1], c[2], c[3]);
}

/* The math engine on scalars a, b, c (expressions), as a vec4 expression. */
static bool pg_math(unsigned op, const char *a, const char *b, const char *c,
                    char *buf, size_t len)
{
    const char *f;

    switch (op) {
    case ME_EXP_BASE2_DX:
        snprintf(buf, len, "vec4(exp2(floor(%s)), %s > 128.0 ? 0.0 : %s - floor(%s), "
                 "exp2(%s), 1.0)", a, a, a, a, a);
        return true;
    case ME_LOG_BASE2_DX:
        snprintf(buf, len, "pvs_log_dx(%s)", a);
        return true;
    case ME_LIGHT_COEFF_DX:
        snprintf(buf, len, "vec4(1.0, max(%s, 0.0), %s > 0.0 ? pvs_pow_ff(max(%s, 0.0), "
                 "clamp(%s, -128.0, 128.0)) : 0.0, 1.0)", b, b, a, c);
        return true;
    case ME_EXP_BASEE_FF:       f = "exp(A)"; break;
    case ME_POWER_FUNC_FF:      f = "pvs_pow_ff(A, C)"; break;
    case ME_RECIP_DX:           f = "(A == 0.0 ? PVS_FLT_MAX : 1.0 / A)"; break;
    case ME_RECIP_FF:           f = "(A == 0.0 ? 0.0 : 1.0 / A)"; break;
    case ME_RECIP_SQRT_DX:      f = "(A == 0.0 ? PVS_FLT_MAX : inversesqrt(abs(A)))"; break;
    case ME_RECIP_SQRT_FF:      f = "(A == 0.0 ? 0.0 : inversesqrt(abs(A)))"; break;
    case ME_MULTIPLY:           f = "(A * B)"; break;
    case ME_EXP_BASE2_FULL_DX:  f = "exp2(A)"; break;
    case ME_LOG_BASE2_FULL_DX:  f = "(A == 0.0 ? -PVS_FLT_MAX : log2(abs(A)))"; break;
    case ME_POWER_FUNC_FF_CLAMP_B:
        f = "(A < B ? 0.0 : pvs_pow_ff(A, C))"; break;
    case ME_POWER_FUNC_FF_CLAMP_B1:
        f = "(A < B ? 0.0 : A > 1.0 ? 1.0 : pvs_pow_ff(A, C))"; break;
    case ME_POWER_FUNC_FF_CLAMP_01:
        f = "(A <= 0.0 ? 0.0 : A > 1.0 ? 1.0 : pvs_pow_ff(A, C))"; break;
    case ME_SIN:                f = "sin(clamp(A, -3.14159265, 3.14159265))"; break;
    case ME_COS:                f = "cos(clamp(A, -3.14159265, 3.14159265))"; break;
    case ME_LOG_BASE2_IEEE:     f = "log2(abs(A))"; break;
    case ME_RECIP_IEEE:         f = "(1.0 / A)"; break;
    case ME_RECIP_SQRT_IEEE:    f = "inversesqrt(abs(A))"; break;
    case ME_NOP:                f = "0.0"; break;
    default:
        return false;
    }
    /* Substitute the operands for A, B and C. */
    size_t n = 0;
    n += snprintf(buf + n, len - n, "vec4(");
    for (const char *p = f; *p && n < len; p++) {
        const char *r = *p == 'A' ? a : *p == 'B' ? b : *p == 'C' ? c : NULL;
        if (r && (p == f || !isalnum((unsigned char)p[-1]) || p[-1] == '(') &&
            !isalnum((unsigned char)p[1]) && p[1] != '_') {
            n += snprintf(buf + n, len - n, "%s", r);
        } else {
            buf[n++] = *p;
            buf[n] = 0;
        }
    }
    if (n < len) {
        snprintf(buf + n, len - n, ")");
    }
    return n + 1 < len;
}

static bool pg_vector(unsigned op, char *buf, size_t len)
{
    const char *f;

    switch (op) {
    case VE_NOP:                    f = "vec4(0.0)"; break;
    case VE_DOT_PRODUCT:            f = "vec4(dot(A, B))"; break;
    case VE_MULTIPLY:               f = "A * B"; break;
    case VE_ADD:                    f = "A + B"; break;
    case VE_MULTIPLY_ADD:           f = "A * B + C"; break;
    case VE_DISTANCE_VECTOR:        f = "vec4(1.0, A.y * B.y, A.z, B.w)"; break;
    case VE_FRACTION:               f = "A - floor(A)"; break;
    case VE_MAXIMUM:                f = "max(A, B)"; break;
    case VE_MINIMUM:                f = "min(A, B)"; break;
    case VE_SET_GREATER_THAN_EQUAL: f = "vec4(greaterThanEqual(A, B))"; break;
    case VE_SET_LESS_THAN:          f = "vec4(lessThan(A, B))"; break;
    case VE_MULTIPLYX2_ADD:         f = "2.0 * (A * B) + C"; break;
    case VE_MULTIPLY_CLAMP:
        f = "vec4(C.w < A.w * B.w ? C.w : C.x >= A.x * B.x ? C.x : A.x * B.x)"; break;
    case VE_FLT2FIX_DX:             f = "floor(A)"; break;
    case VE_FLT2FIX_DX_RND:         f = "floor(A + 0.5)"; break;
    case VE_COND_MUX_EQ:            f = "mix(C, B, equal(A, vec4(0.0)))"; break;
    case VE_COND_MUX_GT:            f = "mix(C, B, greaterThan(A, vec4(0.0)))"; break;
    case VE_COND_MUX_GTE:           f = "mix(C, B, greaterThanEqual(A, vec4(0.0)))"; break;
    case VE_SET_GREATER_THAN:       f = "vec4(greaterThan(A, B))"; break;
    case VE_SET_EQUAL:              f = "vec4(equal(A, B))"; break;
    case VE_SET_NOT_EQUAL:          f = "vec4(notEqual(A, B))"; break;
    default:
        return false;
    }
    snprintf(buf, len, "%s", f);
    return true;
}

/* pvs_write(): r (a vec4 variable) into the destination d0 names. */
static void pg_write(PVSGlsl *g, uint32_t d0, const char *r)
{
    unsigned type = (d0 >> 8) & 0xF;
    unsigned mask = (d0 >> 20) & 0xF;
    unsigned mode = ((d0 >> 31) & 1) | (((d0 >> 12) & 1) << 1);
    unsigned idx = (d0 >> 13) & 0x7F;
    char dst[16];

    if (mode == 1) {
        g->ok = false;              /* A0-relative destination */
        return;
    }
    switch (type) {
    case DST_TEMP:
        if (idx >= R300_PVS_NUM_TEMPS) {
            return;
        }
        g->temp[idx] = true;
        snprintf(dst, sizeof(dst), "t%u", idx);
        break;
    case DST_A0:
        for (int i = 0; i < 4; i++) {
            if (mask & (1u << i)) {
                r300_sb_printf(g->sb, "    a0[%d] = int(clamp(floor(%s.%c), -256.0, 255.0));\n",
                               i, r, pvs_comp[i]);
            }
        }
        return;
    case DST_OUT:
    case DST_OUT_REPL_X:
        if (idx >= R300_PVS_NUM_OUTPUTS) {
            return;
        }
        snprintf(dst, sizeof(dst), "o%u", idx);
        break;
    case DST_ALT:
        if (idx >= R300_PVS_NUM_ALT_TEMPS) {
            return;
        }
        g->alt[idx] = true;
        snprintf(dst, sizeof(dst), "at%u", idx);
        break;
    default:
        return;
    }
    for (int i = 0; i < 4; i++) {
        if (mask & (1u << i)) {
            r300_sb_printf(g->sb, "    %s.%c = %s.%c;\n", dst, pvs_comp[i], r,
                           type == DST_OUT_REPL_X ? 'x' : pvs_comp[i]);
        }
    }
}

static void pg_inst(PVSGlsl *g, const uint32_t *d)
{
    uint32_t d0 = d[0];
    unsigned op = d0 & 0x3F;
    bool math = (d0 >> 6) & 1;
    bool macro = (d0 >> 7) & 1;
    bool dual = !math && !macro && ((d0 >> 28) & 1);
    char e[512];

    if ((d0 >> 26) & 1) {
        g->ok = false;              /* predication */
        return;
    }
    pg_src(g, "A", d[1]);
    pg_src(g, "B", d[2]);
    if (!dual) {
        pg_src(g, "C", d[3]);
    } else {
        r300_sb_printf(g->sb, "    C = vec4(0.0);\n");
    }
    if (math) {
        if (!pg_math(op, "A.w", "B.w", "C.w", e, sizeof(e))) {
            g->ok = false;
            return;
        }
        r300_sb_printf(g->sb, "    R = %s;\n", e);
        if ((d0 >> 25) & 1) {
            r300_sb_printf(g->sb, "    R = clamp(R, 0.0, 1.0);\n");
        }
        pg_write(g, d0, "R");
        return;
    }
    if (!pg_vector(macro ? ((op & 1) ? VE_MULTIPLYX2_ADD : VE_MULTIPLY_ADD) : op,
                   e, sizeof(e))) {
        g->ok = false;
        return;
    }
    r300_sb_printf(g->sb, "    R = %s;\n", e);
    if ((d0 >> 24) & 1) {
        r300_sb_printf(g->sb, "    R = clamp(R, 0.0, 1.0);\n");
    }
    if (dual) {
        uint32_t s = d[3];
        unsigned mop = ((s >> 21) & 0xF) | (((s >> 2) & 1) << 4);
        char base[64], x[24], y[24];

        pg_base(g, s, base, sizeof(base));
        pg_comp(x, sizeof(x), (s >> 13) & 7, s & (1u << 3), s & (1u << 25));
        pg_comp(y, sizeof(y), (s >> 16) & 7, s & (1u << 3), s & (1u << 26));
        r300_sb_printf(g->sb, "    pb = %s; M = vec4(%s, %s, 0.0, 0.0);\n", base, x, y);
        if (!pg_math(mop, "M.x", "M.y", "M.y", e, sizeof(e))) {
            g->ok = false;
            return;
        }
        r300_sb_printf(g->sb, "    M = %s;\n", e);
        if ((d0 >> 25) & 1) {
            r300_sb_printf(g->sb, "    M = clamp(M, 0.0, 1.0);\n");
        }
        pg_write(g, d0, "R");
        unsigned at = (s >> 19) & 3, comp = (s >> 27) & 3;
        g->alt[at] = true;
        r300_sb_printf(g->sb, "    at%u.%c = M.%c;\n", at, pvs_comp[comp], pvs_comp[comp]);
        return;
    }
    pg_write(g, d0, "R");
}

const char r300_pvs_glsl_helpers[] =
"#define PVS_FLT_MAX 3.40282347e38\n"
"float pvs_pow_ff(float b, float e)\n"
"{\n"
"    if (b == 0.0) return e < 0.0 ? uintBitsToFloat(0x7F800000u) : 0.0;\n"
"    if (e == 0.0) return 1.0;\n"
"    return b < 0.0 ? -pow(-b, e) : pow(b, e);\n"
"}\n"
"vec4 pvs_log_dx(float a)\n"
"{\n"
"    if (a == 0.0) return vec4(-PVS_FLT_MAX, 1.0, -PVS_FLT_MAX, 1.0);\n"
"    int e; float m = frexp(abs(a), e);\n"
"    return vec4(float(e - 1), m * 2.0, log2(abs(a)), 1.0);\n"
"}\n";

bool r300_pvs_to_glsl(const R300PVSProgram *prog, R300Sb *sb, uint32_t *in_used)
{
    PVSGlsl g = { .prog = prog, .ok = prog->fc_opc == 0 };
    R300Sb body;

    if (!g.ok) {
        return false;               /* flow control: the interpreter's */
    }
    r300_sb_init(&body);
    g.sb = &body;
    for (unsigned pc = prog->first_inst;
         pc <= prog->last_inst && pc < R300_PVS_MAX_INSTS && g.ok; pc++) {
        pg_inst(&g, &prog->code[pc * 4]);
    }
    if (g.ok) {
        r300_sb_printf(sb, "    vec4 pb, A, B, C, R, M;\n    ivec4 a0 = ivec4(0);\n");
        for (unsigned k = 0; k < R300_PVS_NUM_TEMPS; k++) {
            if (g.temp[k]) {
                r300_sb_printf(sb, "    vec4 t%u = vec4(0.0);\n", k);
            }
        }
        for (unsigned k = 0; k < R300_PVS_NUM_ALT_TEMPS; k++) {
            if (g.alt[k]) {
                r300_sb_printf(sb, "    vec4 at%u = vec4(0.0);\n", k);
            }
        }
        for (unsigned k = 0; k < R300_PVS_NUM_OUTPUTS; k++) {
            r300_sb_printf(sb, "    vec4 o%u = vec4(0.0);\n", k);
        }
        r300_sb_printf(sb, "%s", body.buf);
        *in_used = g.in_used;
    }
    r300_sb_free(&body);
    return g.ok;
}
