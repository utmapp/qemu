/*
 * Bare-metal AltiVec check for the inline (host vector) translations in
 * target/ppc/translate/vmx-impl.c.inc (vperm, vsldoi, merges, packs,
 * unpacks, even/odd multiplies, modulo multiply-sums) and the host-FPU
 * fast paths of the float helpers in target/ppc/int_helper.c.
 *
 * Each op runs on random and edge-case inputs and is compared with a
 * scalar reference (scalar integer and FPU code go through the ordinary,
 * trusted TCG and softfloat paths).  Results go to guest RAM at RESULTS for run.py to read:
 *   +0  magic 'VMXT' once finished
 *   +4  checks run            +8  failures
 *   +12 failures per op (NOPS words)
 *   +0x100 first MAXLOG failures: op, iteration, a, b, c, got, expected
 *   +0x800 benchmark: timebase ticks per op for BENCH_ITERS iterations
 *
 * Built by run.py with clang --target=powerpc-unknown-none -maltivec.
 */
#include <stdint.h>

#define RESULTS     0x10000u
#define ITERS       4000
#define MAXLOG      8
#define BENCH_ITERS 2000000

typedef vector unsigned char vuc;

typedef union {
    vuc v;
    uint8_t b[16];
    int8_t sb[16];
    uint16_t h[8];
    int16_t sh[8];
    uint32_t w[4];
    int32_t sw[4];
    float f[4];
} V;

#define FIRST_FP_OP OP_VADDFP

enum {
    OP_VPERM, OP_VPERM_DA, OP_VPERM_DC, OP_VPERM_AA, OP_VSLDOI, OP_VSLDOI_DB,
    OP_VMRGHB, OP_VMRGHH, OP_VMRGHW, OP_VMRGLB, OP_VMRGLH, OP_VMRGLW,
    OP_VPKUHUM, OP_VPKUWUM, OP_VPKUHUS, OP_VPKUWUS, OP_VPKSHUS, OP_VPKSWUS,
    OP_VPKSHSS, OP_VPKSWSS,
    OP_VUPKHSB, OP_VUPKHSH, OP_VUPKLSB, OP_VUPKLSH,
    OP_VMULEUB, OP_VMULOUB, OP_VMULESB, OP_VMULOSB,
    OP_VMULEUH, OP_VMULOUH, OP_VMULESH, OP_VMULOSH,
    OP_VMSUMUBM, OP_VMSUMMBM, OP_VMSUMUHM, OP_VMSUMSHM,
    OP_VADDFP, OP_VSUBFP, OP_VMAXFP, OP_VMINFP, OP_VMADDFP, OP_VNMSUBFP,
    OP_VREFP, OP_VCFSX, OP_VCFUX, OP_VCTSXS, OP_VCTUXS,
    NOPS
};

static volatile uint32_t *const res = (volatile uint32_t *)RESULTS;

void *memcpy(void *d, const void *s, unsigned long n)
{
    uint8_t *dp = d;
    const uint8_t *sp = s;
    while (n--) {
        *dp++ = *sp++;
    }
    return d;
}

void *memset(void *d, int c, unsigned long n)
{
    uint8_t *dp = d;
    while (n--) {
        *dp++ = c;
    }
    return d;
}

static uint32_t rnd(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

/* Random bytes, with an eighth of them edge values for the saturations. */
static V rand_vec(uint32_t *s)
{
    static const uint8_t edge[8] = { 0, 1, 0x7f, 0x80, 0xff, 0xfe, 0x81, 0 };
    V r;
    int i;

    for (i = 0; i < 16; i++) {
        uint32_t x = rnd(s);
        r.b[i] = (x & 7) == 0 ? edge[(x >> 3) & 7] : x >> 8;
    }
    return r;
}

/*
 * Random floats: mostly moderate exponents (the host fast path), some
 * anything at all (overflow, NaN, infinity, denormal: softfloat), some
 * zeros and small integers.
 */
static V rand_fvec(uint32_t *s)
{
    V r;
    int i;

    for (i = 0; i < 4; i++) {
        uint32_t x = rnd(s), y = rnd(s);
        switch (y & 7) {
        case 0:
            r.w[i] = x;
            break;
        case 1:
            r.w[i] = x & 0x80000000;
            break;
        case 2:
            r.w[i] = x & 0x807fffff;            /* denormal */
            break;
        case 3:
            r.sw[i] = (int32_t)x >> (y >> 27);  /* integers for vcfsx */
            break;
        default:
            r.w[i] = (x & 0x807fffff) | (((y >> 8) % 40 + 108) << 23);
            break;
        }
    }
    return r;
}

static int is_nan(uint32_t w)
{
    return (w & 0x7f800000) == 0x7f800000 && (w & 0x007fffff);
}

static float fmadds(float a, float c, float b)
{
    float r;
    asm("fmadds %0,%1,%2,%3" : "=f"(r) : "f"(a), "f"(c), "f"(b));
    return r;
}

static float fnmsubs(float a, float c, float b)
{
    float r;
    asm("fnmsubs %0,%1,%2,%3" : "=f"(r) : "f"(a), "f"(c), "f"(b));
    return r;
}

static float fmax_ref(float a, float b, int is_max)
{
    if (a != a || b != b) {
        return a + b;           /* some NaN; any NaN result passes */
    }
    if (a == 0 && b == 0) {
        /* Signed zeros: max prefers +0, min prefers -0. */
        union { float f; uint32_t w; } ua = { a }, ub = { b };
        return is_max ? ((ua.w & ub.w) >> 31 ? a : (ua.w >> 31 ? b : a))
                      : ((ua.w | ub.w) >> 31 ? (ua.w >> 31 ? a : b) : a);
    }
    return is_max ? (a > b ? a : b) : (a < b ? a : b);
}

static double pow2(int n)
{
    double d = 1;
    while (n > 0) {
        d *= 2;
        n--;
    }
    while (n < 0) {
        d /= 2;
        n++;
    }
    return d;
}

/* vctsxs/vctuxs reference: saturate(trunc(f * 2^uim)). */
static uint32_t ct_ref(float f, int uim, int sign, int *sat)
{
    double d = (double)f * pow2(uim);
    double lo = sign ? -2147483648.0 : 0.0;
    double hi = sign ? 2147483647.0 : 4294967295.0;

    if (f != f) {
        return 0;
    }
    if (d <= lo - 1 || d >= hi + 1) {
        *sat = 1;
        return d < 0 ? (sign ? 0x80000000u : 0) : (sign ? 0x7fffffffu
                                                        : 0xffffffffu);
    }
    if (sign) {
        return (uint32_t)(int32_t)d;
    }
    return d < 0 ? 0 : (uint32_t)d;
}

static uint32_t tb(void)
{
    uint32_t t;
    asm volatile("mftb %0" : "=r"(t));
    return t;
}

static void clear_sat(void)
{
    V z;
    memset(&z, 0, sizeof(z));
    asm volatile("mtvscr %0" : : "v"(z.v));
}

static int get_sat(void)
{
    V r;
    asm volatile("mfvscr %0" : "=v"(r.v));
    return r.w[3] & 1;
}

#define ASM3(insn, r, a, b) \
    asm volatile(insn " %0,%1,%2" : "=v"(r.v) : "v"(a.v), "v"(b.v))
#define ASM4(insn, r, a, b, c) \
    asm volatile(insn " %0,%1,%2,%3" : "=v"(r.v) : "v"(a.v), "v"(b.v), "v"(c.v))
#define ASM2(insn, r, b) \
    asm volatile(insn " %0,%1" : "=v"(r.v) : "v"(b.v))

static int32_t clamp(int64_t x, int64_t lo, int64_t hi, int *sat)
{
    if (x < lo) {
        *sat = 1;
        return lo;
    }
    if (x > hi) {
        *sat = 1;
        return hi;
    }
    return x;
}

/* Run op on a, b, c; fill in the result and the reference. */
static void run_op(int op, V a, V b, V c, int sh, V *got, V *exp,
                   int *got_sat, int *exp_sat)
{
    V r, e, t;
    int i, j, sat = 0;

    memset(&e, 0, sizeof(e));
    clear_sat();
    switch (op) {
    case OP_VPERM:
        ASM4("vperm", r, a, b, c);
        goto ref_perm;
    case OP_VPERM_DA:
        r = a;
        asm volatile("vperm %0,%0,%1,%2" : "+v"(r.v) : "v"(b.v), "v"(c.v));
        goto ref_perm;
    case OP_VPERM_DC:
        r = c;
        asm volatile("vperm %0,%1,%2,%0" : "+v"(r.v) : "v"(a.v), "v"(b.v));
        goto ref_perm;
    case OP_VPERM_AA:
        b = a;
        asm volatile("vperm %0,%1,%1,%2" : "=v"(r.v) : "v"(a.v), "v"(c.v));
    ref_perm:
        for (i = 0; i < 16; i++) {
            int s = c.b[i] & 0x1f;
            e.b[i] = s < 16 ? a.b[s] : b.b[s - 16];
        }
        break;
    case OP_VSLDOI:
    case OP_VSLDOI_DB:
#define SLD(n)                                                              \
    case n:                                                                 \
        if (op == OP_VSLDOI) {                                              \
            asm volatile("vsldoi %0,%1,%2,%3"                               \
                         : "=v"(r.v) : "v"(a.v), "v"(b.v), "i"(n));         \
        } else {                                                            \
            r = b;                                                          \
            asm volatile("vsldoi %0,%1,%0,%2"                               \
                         : "+v"(r.v) : "v"(a.v), "i"(n));                   \
        }                                                                   \
        break;
        switch (sh) {
            SLD(0) SLD(1) SLD(2) SLD(3) SLD(4) SLD(5) SLD(6) SLD(7)
            SLD(8) SLD(9) SLD(10) SLD(11) SLD(12) SLD(13) SLD(14) SLD(15)
        }
        for (i = 0; i < 16; i++) {
            j = i + sh;
            e.b[i] = j < 16 ? a.b[j] : b.b[j - 16];
        }
        break;
    case OP_VMRGHB:
        ASM3("vmrghb", r, a, b);
        for (i = 0; i < 8; i++) {
            e.b[2 * i] = a.b[i];
            e.b[2 * i + 1] = b.b[i];
        }
        break;
    case OP_VMRGHH:
        ASM3("vmrghh", r, a, b);
        for (i = 0; i < 4; i++) {
            e.h[2 * i] = a.h[i];
            e.h[2 * i + 1] = b.h[i];
        }
        break;
    case OP_VMRGHW:
        ASM3("vmrghw", r, a, b);
        for (i = 0; i < 2; i++) {
            e.w[2 * i] = a.w[i];
            e.w[2 * i + 1] = b.w[i];
        }
        break;
    case OP_VMRGLB:
        ASM3("vmrglb", r, a, b);
        for (i = 0; i < 8; i++) {
            e.b[2 * i] = a.b[i + 8];
            e.b[2 * i + 1] = b.b[i + 8];
        }
        break;
    case OP_VMRGLH:
        ASM3("vmrglh", r, a, b);
        for (i = 0; i < 4; i++) {
            e.h[2 * i] = a.h[i + 4];
            e.h[2 * i + 1] = b.h[i + 4];
        }
        break;
    case OP_VMRGLW:
        ASM3("vmrglw", r, a, b);
        for (i = 0; i < 2; i++) {
            e.w[2 * i] = a.w[i + 2];
            e.w[2 * i + 1] = b.w[i + 2];
        }
        break;
    case OP_VPKUHUM:
        ASM3("vpkuhum", r, a, b);
        for (i = 0; i < 16; i++) {
            e.b[i] = i < 8 ? a.h[i] : b.h[i - 8];
        }
        break;
    case OP_VPKUWUM:
        ASM3("vpkuwum", r, a, b);
        for (i = 0; i < 8; i++) {
            e.h[i] = i < 4 ? a.w[i] : b.w[i - 4];
        }
        break;
    case OP_VPKUHUS:
        ASM3("vpkuhus", r, a, b);
        for (i = 0; i < 16; i++) {
            e.b[i] = clamp(i < 8 ? a.h[i] : b.h[i - 8], 0, 0xff, &sat);
        }
        break;
    case OP_VPKUWUS:
        ASM3("vpkuwus", r, a, b);
        for (i = 0; i < 8; i++) {
            e.h[i] = clamp(i < 4 ? a.w[i] : b.w[i - 4], 0, 0xffff, &sat);
        }
        break;
    case OP_VPKSHUS:
        ASM3("vpkshus", r, a, b);
        for (i = 0; i < 16; i++) {
            e.b[i] = clamp(i < 8 ? a.sh[i] : b.sh[i - 8], 0, 0xff, &sat);
        }
        break;
    case OP_VPKSWUS:
        ASM3("vpkswus", r, a, b);
        for (i = 0; i < 8; i++) {
            e.h[i] = clamp(i < 4 ? a.sw[i] : b.sw[i - 4], 0, 0xffff, &sat);
        }
        break;
    case OP_VPKSHSS:
        ASM3("vpkshss", r, a, b);
        for (i = 0; i < 16; i++) {
            e.b[i] = clamp(i < 8 ? a.sh[i] : b.sh[i - 8], -0x80, 0x7f, &sat);
        }
        break;
    case OP_VPKSWSS:
        ASM3("vpkswss", r, a, b);
        for (i = 0; i < 8; i++) {
            e.h[i] = clamp(i < 4 ? a.sw[i] : b.sw[i - 4],
                           -0x8000, 0x7fff, &sat);
        }
        break;
    case OP_VUPKHSB:
        ASM2("vupkhsb", r, b);
        for (i = 0; i < 8; i++) {
            e.sh[i] = b.sb[i];
        }
        break;
    case OP_VUPKHSH:
        ASM2("vupkhsh", r, b);
        for (i = 0; i < 4; i++) {
            e.sw[i] = b.sh[i];
        }
        break;
    case OP_VUPKLSB:
        ASM2("vupklsb", r, b);
        for (i = 0; i < 8; i++) {
            e.sh[i] = b.sb[i + 8];
        }
        break;
    case OP_VUPKLSH:
        ASM2("vupklsh", r, b);
        for (i = 0; i < 4; i++) {
            e.sw[i] = b.sh[i + 4];
        }
        break;
#define MUL(OPC, insn, ew, rw, off)                                         \
    case OPC:                                                               \
        ASM3(insn, r, a, b);                                                \
        for (i = 0; i < 16 / sizeof(e.rw[0]); i++) {                        \
            e.rw[i] = a.ew[2 * i + off] * b.ew[2 * i + off];                \
        }                                                                   \
        break;
    MUL(OP_VMULEUB, "vmuleub", b, h, 0)
    MUL(OP_VMULOUB, "vmuloub", b, h, 1)
    MUL(OP_VMULESB, "vmulesb", sb, sh, 0)
    MUL(OP_VMULOSB, "vmulosb", sb, sh, 1)
    MUL(OP_VMULEUH, "vmuleuh", h, w, 0)
    MUL(OP_VMULOUH, "vmulouh", h, w, 1)
    MUL(OP_VMULESH, "vmulesh", sh, sw, 0)
    MUL(OP_VMULOSH, "vmulosh", sh, sw, 1)
    case OP_VMSUMUBM:
        ASM4("vmsumubm", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.w[i] = c.w[i];
            for (j = 0; j < 4; j++) {
                e.w[i] += (uint32_t)a.b[4 * i + j] * b.b[4 * i + j];
            }
        }
        break;
    case OP_VMSUMMBM:
        ASM4("vmsummbm", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.w[i] = c.w[i];
            for (j = 0; j < 4; j++) {
                e.w[i] += (int32_t)a.sb[4 * i + j] * b.b[4 * i + j];
            }
        }
        break;
    case OP_VMSUMUHM:
        ASM4("vmsumuhm", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.w[i] = c.w[i] + (uint32_t)a.h[2 * i] * b.h[2 * i]
                     + (uint32_t)a.h[2 * i + 1] * b.h[2 * i + 1];
        }
        break;
#define FP3(OPC, insn, expr)                                               \
    case OPC:                                                               \
        ASM3(insn, r, a, b);                                                \
        for (i = 0; i < 4; i++) {                                           \
            e.f[i] = expr;                                                  \
        }                                                                   \
        break;
    FP3(OP_VADDFP, "vaddfp", a.f[i] + b.f[i])
    FP3(OP_VSUBFP, "vsubfp", a.f[i] - b.f[i])
    FP3(OP_VMAXFP, "vmaxfp", fmax_ref(a.f[i], b.f[i], 1))
    FP3(OP_VMINFP, "vminfp", fmax_ref(a.f[i], b.f[i], 0))
    case OP_VMADDFP:
        ASM4("vmaddfp", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.f[i] = fmadds(a.f[i], b.f[i], c.f[i]);
        }
        break;
    case OP_VNMSUBFP:
        ASM4("vnmsubfp", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.f[i] = fnmsubs(a.f[i], b.f[i], c.f[i]);
        }
        break;
    case OP_VREFP:
        ASM2("vrefp", r, b);
        for (i = 0; i < 4; i++) {
            e.f[i] = 1.0f / b.f[i];
        }
        break;
#define UIMS(X) X(0) X(1) X(7) X(16) X(31)
#define CF(n)                                                               \
    case n:                                                                 \
        if (op == OP_VCFSX) {                                               \
            asm volatile("vcfsx %0,%1,%2" : "=v"(r.v) : "v"(b.v), "i"(n));  \
        } else if (op == OP_VCFUX) {                                        \
            asm volatile("vcfux %0,%1,%2" : "=v"(r.v) : "v"(b.v), "i"(n));  \
        } else if (op == OP_VCTSXS) {                                       \
            asm volatile("vctsxs %0,%1,%2" : "=v"(r.v) : "v"(b.v), "i"(n)); \
        } else {                                                            \
            asm volatile("vctuxs %0,%1,%2" : "=v"(r.v) : "v"(b.v), "i"(n)); \
        }                                                                   \
        uim = n;                                                            \
        break;
    case OP_VCFSX:
    case OP_VCFUX:
    case OP_VCTSXS:
    case OP_VCTUXS: {
        static const int uim_of[5] = { 0, 1, 7, 16, 31 };
        int uim = 0;
        switch (uim_of[sh % 5]) {
            UIMS(CF)
        }
        for (i = 0; i < 4; i++) {
            if (op == OP_VCFSX) {
                e.f[i] = (float)b.sw[i] * (float)pow2(-uim);
            } else if (op == OP_VCFUX) {
                e.f[i] = (float)b.w[i] * (float)pow2(-uim);
            } else {
                e.w[i] = ct_ref(b.f[i], uim, op == OP_VCTSXS, &sat);
            }
        }
        break;
    }
    case OP_VMSUMSHM:
        ASM4("vmsumshm", r, a, b, c);
        for (i = 0; i < 4; i++) {
            e.w[i] = c.w[i] + (uint32_t)((int32_t)a.sh[2 * i] * b.sh[2 * i])
                     + (uint32_t)((int32_t)a.sh[2 * i + 1] * b.sh[2 * i + 1]);
        }
        break;
    }
    t = r;
    *got = t;
    *exp = e;
    *got_sat = get_sat();
    *exp_sat = sat;
}

static void log_vec(volatile uint32_t *p, V v)
{
    int i;
    for (i = 0; i < 4; i++) {
        p[i] = v.w[i];
    }
}

/* Dependent chains of the hot ops, for timing against the helper path. */
static void bench(void)
{
    V a, b, c;
    uint32_t t0;
    int i;

    memset(&a, 0x12, sizeof(a));
    memset(&b, 0x34, sizeof(b));
    memset(&c, 0x05, sizeof(c));

    t0 = tb();
    for (i = 0; i < BENCH_ITERS; i++) {
        asm volatile("vperm %0,%0,%1,%2\n\tvperm %0,%0,%1,%2\n\t"
                     "vperm %0,%0,%1,%2\n\tvperm %0,%0,%1,%2"
                     : "+v"(a.v) : "v"(b.v), "v"(c.v));
    }
    res[0x200] = tb() - t0;

    t0 = tb();
    for (i = 0; i < BENCH_ITERS; i++) {
        asm volatile("vmrghb %0,%0,%1\n\tvmrglh %0,%0,%1\n\t"
                     "vsldoi %0,%0,%1,3\n\tvpkuhum %0,%0,%1"
                     : "+v"(a.v) : "v"(b.v));
    }
    res[0x201] = tb() - t0;

    t0 = tb();
    for (i = 0; i < BENCH_ITERS; i++) {
        asm volatile("vmuleub %0,%0,%1\n\tvmsumubm %0,%0,%1,%2\n\t"
                     "vpkshus %0,%0,%1\n\tvupkhsb %0,%0"
                     : "+v"(a.v) : "v"(b.v), "v"(c.v));
    }
    res[0x202] = tb() - t0;

    t0 = tb();
    for (i = 0; i < BENCH_ITERS; i++) {
        asm volatile("vaddubm %0,%0,%1\n\tvaddubm %0,%0,%1\n\t"
                     "vaddubm %0,%0,%1\n\tvaddubm %0,%0,%1"
                     : "+v"(a.v) : "v"(b.v));
    }
    res[0x203] = tb() - t0;

    /* 0.5f, 1.0f, 0.25f: stays finite and normal. */
    a.w[0] = a.w[1] = a.w[2] = a.w[3] = 0x3f000000;
    b.w[0] = b.w[1] = b.w[2] = b.w[3] = 0x3f800000;
    c.w[0] = c.w[1] = c.w[2] = c.w[3] = 0x3e800000;
    t0 = tb();
    for (i = 0; i < BENCH_ITERS; i++) {
        asm volatile("vmaddfp %0,%0,%2,%1\n\tvmaddfp %0,%0,%2,%1\n\t"
                     "vmaddfp %0,%0,%2,%1\n\tvmaddfp %0,%0,%2,%1"
                     : "+v"(a.v) : "v"(b.v), "v"(c.v));
    }
    res[0x204] = tb() - t0;

    /* lfs; fmadds; stfs on an array of normal singles, x4 per iteration */
    {
        float buf[64];
        float k = 1.0f;
        for (i = 0; i < 64; i++) {
            buf[i] = 1.0f + i * 0.125f;
        }
        t0 = tb();
        for (i = 0; i < BENCH_ITERS; i++) {
            float *p = &buf[(i * 4) & 63];
            asm volatile("lfs 1,0(%0)\n\tfmadds 1,1,%1,%1\n\tstfs 1,0(%0)\n\t"
                         "lfs 2,4(%0)\n\tfmadds 2,2,%1,%1\n\tstfs 2,4(%0)\n\t"
                         "lfs 3,8(%0)\n\tfmadds 3,3,%1,%1\n\tstfs 3,8(%0)\n\t"
                         "lfs 4,12(%0)\n\tfmadds 4,4,%1,%1\n\tstfs 4,12(%0)"
                         : : "b"(p), "f"(k) : "fr1", "fr2", "fr3", "fr4", "memory");
            k = k * 0.5f + 0.25f;
        }
        res[0x205] = tb() - t0;
    }

    /* scalar FP latency: dependent chains vs independent vs integer */
    {
        double x = 1.0, y = 1.0, z = 1.0, w = 1.0, h = 0.5, q = 0.25;
        uint32_t r = 1, k = 3;

        t0 = tb();
        for (i = 0; i < BENCH_ITERS; i++) {
            asm volatile("fmadd %0,%0,%1,%2\n\tfmadd %0,%0,%1,%2\n\t"
                         "fmadd %0,%0,%1,%2\n\tfmadd %0,%0,%1,%2"
                         : "+f"(x) : "f"(h), "f"(q));
        }
        res[0x206] = tb() - t0;

        t0 = tb();
        for (i = 0; i < BENCH_ITERS; i++) {
            asm volatile("fmadd %0,%0,%4,%5\n\tfmadd %1,%1,%4,%5\n\t"
                         "fmadd %2,%2,%4,%5\n\tfmadd %3,%3,%4,%5"
                         : "+f"(x), "+f"(y), "+f"(z), "+f"(w) : "f"(h), "f"(q));
        }
        res[0x207] = tb() - t0;

        t0 = tb();
        for (i = 0; i < BENCH_ITERS; i++) {
            asm volatile("fadd %0,%0,%1\n\tfadd %0,%0,%1\n\t"
                         "fadd %0,%0,%1\n\tfadd %0,%0,%1"
                         : "+f"(x) : "f"(q));
        }
        res[0x208] = tb() - t0;

        t0 = tb();
        for (i = 0; i < BENCH_ITERS; i++) {
            asm volatile("add %0,%0,%1\n\tadd %0,%0,%1\n\t"
                         "add %0,%0,%1\n\tadd %0,%0,%1"
                         : "+r"(r) : "r"(k));
        }
        res[0x209] = tb() - t0;
        res[0x20a] = r ^ (uint32_t)(x + y + z + w);
    }

    /* Mandelbrot escape loop, as compiled C: fcmpu decides every exit */
    {
        uint32_t total = 0;
        int px, py, n;

        t0 = tb();
        for (py = 0; py < 48; py++) {
            for (px = 0; px < 64; px++) {
                double cr = -2.0 + px * (2.5 / 64), ci = -1.25 + py * (2.5 / 48);
                double zr = 0, zi = 0, zr2 = 0, zi2 = 0;
                for (n = 0; n < 256 && zr2 + zi2 <= 4.0; n++) {
                    zi = 2.0 * zr * zi + ci;
                    zr = zr2 - zi2 + cr;
                    zr2 = zr * zr;
                    zi2 = zi * zi;
                }
                total += n;
            }
        }
        res[0x20b] = tb() - t0;
        res[0x20c] = total;
    }

    /* Geekbench 2.2's Mandelbrot inner loop (GCC 4.0), instruction for
       instruction, points that never escape: 255 iterations each */
    {
        double c[4] = { 0.25, 0.0, 4.0, 0.0 };  /* cr, ci, escape radius^2 */
        uint32_t n = 0;

        t0 = tb();
        for (i = 0; i < 1000; i++) {
            if (i == 500) {
                res[0x20f] = tb() - t0;     /* first half: never escapes */
                c[0] = 1.5;                 /* then: escapes after 2 */
                c[1] = 0.5;
            }
            if (i >= 500) {
                int j;
                for (j = 0; j < 100; j++) {
                    asm volatile(
                        "lfd 8,0(%1)\n\tlfd 6,8(%1)\n\tlfd 7,16(%1)\n\t"
                        "fmr 9,6\n\tfmr 11,8\n\tfmr 10,9\n\t"
                        "li 0,255\n\tmtctr 0\n"
                        "1:\tfmul 0,10,10\n\taddi %0,%0,1\n\t"
                        "fmsub 0,11,11,0\n\tfadd 11,8,0\n\tfadd 12,11,11\n\t"
                        "fsub 13,11,8\n\tfmadd 10,10,12,9\n\tfsub 0,10,9\n\t"
                        "fmul 0,0,0\n\tfmadd 13,13,13,0\n\tfcmpu 7,13,7\n\t"
                        "cror 30,29,30\n\tbeq 7,2f\n\tbdnz 1b\n2:"
                        : "+r"(n) : "b"(c)
                        : "fr0", "fr6", "fr7", "fr8", "fr9", "fr10", "fr11", "fr12",
                          "fr13", "ctr", "cr7", "r0");
                }
                continue;
            }
            asm volatile(
                "lfd 8,0(%1)\n\tlfd 6,8(%1)\n\tlfd 7,16(%1)\n\t"
                "fmr 9,6\n\tfmr 11,8\n\tfmr 10,9\n\t"
                "li 0,255\n\tmtctr 0\n"
                "1:\tfmul 0,10,10\n\taddi %0,%0,1\n\t"
                "fmsub 0,11,11,0\n\tfadd 11,8,0\n\tfadd 12,11,11\n\t"
                "fsub 13,11,8\n\tfmadd 10,10,12,9\n\tfsub 0,10,9\n\t"
                "fmul 0,0,0\n\tfmadd 13,13,13,0\n\tfcmpu 7,13,7\n\t"
                "cror 30,29,30\n\tbeq 7,2f\n\tbdnz 1b\n2:"
                : "+r"(n) : "b"(c)
                : "fr0", "fr6", "fr7", "fr8", "fr9", "fr10", "fr11", "fr12",
                  "fr13", "ctr", "cr7", "r0");
        }
        res[0x20d] = tb() - t0;
        res[0x20e] = n;
    }

    /* Geekbench 2.2's whole Mandelbrot routine (0x26ac0-0x26b98): rows and
       columns converted to double through memory (stw, stw, lfd), the
       loop, and the flop count kept in r27:r28 with addc/adde */
    {
        double k[5] = {
            0.0, -2.0, 4.0, 1.25, 0.0,  /* magic (set below), cr0, esc, ci0 */
        };
        uint32_t buf[4];
        uint32_t lo = 0, hi = 0;
        union { uint64_t u; double d; } magic = { 0x4330000080000000ull };

        k[0] = magic.d;
        k[4] = 0.046875;                    /* 3/64: rows and columns */
        t0 = tb();
        for (i = 0; i < 4; i++) {
            asm volatile(
                "lfd 5,32(%[k])\n\t"            /* row scale */
                "lfd 6,32(%[k])\n\t"            /* column scale */
                "li 10,0\n\tli 26,48\n\tli 30,64\n\t"
                "cmpwi 6,30,0\n"
                "3:\txoris 2,10,0x8000\n\tlis 0,0x4330\n\tlfd 13,0(%[k])\n\t"
                "li 9,0\n\tstw 2,12(%[b])\n\tstw 0,8(%[b])\n\t"
                "lfd 0,8(%[b])\n\tfsub 0,0,13\n\tlfd 13,24(%[k])\n\t"
                "fmsub 8,5,0,13\n"
                "4:\txoris 2,9,0x8000\n\tlis 0,0x4330\n\tlfd 13,0(%[k])\n\t"
                "lfd 12,8(%[k])\n\tfmr 11,8\n\tstw 0,0(%[b])\n\tstw 2,4(%[b])\n\t"
                "li 0,0xff\n\tmtctr 0\n\tlfd 7,16(%[k])\n\tli 2,0\n\t"
                "lfd 0,0(%[b])\n\tfsub 0,0,13\n\tfmadd 9,6,0,12\n\tfmr 10,9\n"
                "1:\tfmul 0,10,10\n\taddi 2,2,1\n\t"
                "fmsub 0,11,11,0\n\tfadd 11,8,0\n\tfadd 12,11,11\n\t"
                "fsub 13,11,8\n\tfmadd 10,10,12,9\n\tfsub 0,10,9\n\t"
                "fmul 0,0,0\n\tfmadd 13,13,13,0\n\tfcmpu 7,13,7\n\t"
                "cror 30,29,30\n\tbeq 7,2f\n\tbdnz 1b\n"
                "2:\taddi 9,9,1\n\tmulli 2,2,14\n\tcmpw 7,30,9\n\tmr 3,2\n\t"
                "li 2,0\n\taddc %[lo],%[lo],3\n\tadde %[hi],%[hi],2\n\t"
                "bne 7,4b\n\t"
                "addi 10,10,1\n\tcmpw 7,26,10\n\tbeq 7,5f\n\t"
                "ble 6,3b\n\tb 3b\n"
                "5:"
                : [lo] "+r"(lo), [hi] "+r"(hi)
                : [k] "b"(k), [b] "b"(buf)
                : "r0", "r2", "r3", "r9", "r10", "r26", "r30",
                  "fr0", "fr5", "fr6", "fr7", "fr8", "fr9", "fr10", "fr11",
                  "fr12", "fr13", "ctr", "cr6", "cr7", "xer", "memory");
            k[4] = 0.046875 + i * 0.001;    /* a slightly different image each time */
        }
        res[0x210] = tb() - t0;
        res[0x211] = lo;
    }
}

/*
 * Scalar lfs/stfs: single <-> double conversion, which the translator
 * inlines for normal numbers (target/ppc/translate/fp-impl.c.inc).  The
 * references are helper_todouble/helper_tosingle's logic in integer code.
 * lfd/stfd move the 64-bit register image in and out unconverted.
 */
static uint64_t ref_todouble(uint32_t a)
{
    uint32_t abs = a & 0x7fffffff, e = (a >> 23) & 0xff;
    uint64_t r = (uint64_t)(a >> 31) << 63;

    if (abs >= 0x00800000) {
        if (e == 0xff) {
            return r | (0x7ffULL << 52) | (uint64_t)(a & 0x7fffff) << 29;
        }
        return r | (uint64_t)(e + 896) << 52 | (uint64_t)(a & 0x7fffff) << 29;
    }
    if (abs) {
        int shift = __builtin_clz(abs) - 8;
        r |= (uint64_t)(-126 - shift + 1023 - 1) << 52;
        r += (uint64_t)abs << (52 - 23 + shift);
    }
    return r;
}

static uint32_t ref_tosingle(uint64_t a)
{
    int e = (a >> 52) & 0x7ff;
    uint32_t r;

    if (e > 896) {
        return (uint32_t)(a >> 62) << 30 | (uint32_t)((a >> 29) & 0x3fffffff);
    }
    r = (uint32_t)(a >> 63) << 31;
    if (e >= 874) {
        r |= ((1ULL << 52) | (a & 0xfffffffffffffULL)) >> (896 + 30 - e);
    }
    return r;
}

static uint64_t hw_lfs(uint32_t s)
{
    volatile uint32_t in = s;
    volatile uint64_t out;
    double f;
    asm volatile("lfs %0,0(%1)" : "=f"(f) : "b"(&in) : "memory");
    asm volatile("stfd %0,0(%1)" : : "f"(f), "b"(&out) : "memory");
    return out;
}

static uint32_t hw_stfs(uint64_t d)
{
    volatile uint64_t in = d;
    volatile uint32_t out;
    double f;
    asm volatile("lfd %0,0(%1)" : "=f"(f) : "b"(&in) : "memory");
    asm volatile("stfs %0,0(%1)" : : "f"(f), "b"(&out) : "memory");
    return out;
}

static void fp_log(uint32_t kind, uint64_t in, uint64_t got, uint64_t exp)
{
    volatile uint32_t *p = res + 0x2a0 + (res[0x281] - 1) * 8;
    if (res[0x281] > 4) {
        return;
    }
    p[0] = kind;
    p[1] = in >> 32;  p[2] = in;
    p[3] = got >> 32; p[4] = got;
    p[5] = exp >> 32; p[6] = exp;
}

static void check_lfs(uint32_t s)
{
    uint64_t got = hw_lfs(s), exp = ref_todouble(s);
    res[0x280]++;
    if (got != exp) {
        res[0x281]++;
        fp_log(0, s, got, exp);
    }
}

static void check_stfs(uint64_t d)
{
    uint32_t got = hw_stfs(d), exp = ref_tosingle(d);
    res[0x280]++;
    if (got != exp) {
        res[0x281]++;
        fp_log(1, d, got, exp);
    }
}

static void scalar_ldst(uint32_t *seed)
{
    static const uint32_t s_edge[] = {
        0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00400000,
        0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001,
        0xffbfffff, 0x40000000, 0x3fffffff, 0x7f000000, 0x00ffffff,
    };
    int i, e;

    for (i = 0; i < (int)(sizeof(s_edge) / sizeof(s_edge[0])); i++) {
        check_lfs(s_edge[i]);
        check_stfs(ref_todouble(s_edge[i]));
    }
    /* stfs around its exponent boundaries (874, 896/897, 0x7ff) */
    for (e = 860; e < 910; e++) {
        for (i = 0; i < 20; i++) {
            uint64_t m = (uint64_t)rnd(seed) << 32 | rnd(seed);
            check_stfs((m & 0x800fffffffffffffULL) | (uint64_t)e << 52);
        }
    }
    for (i = 0; i < 20; i++) {
        uint64_t m = (uint64_t)rnd(seed) << 32 | rnd(seed);
        check_stfs(m | 0x7ff0000000000000ULL);
        check_stfs(m & 0x800fffffffffffffULL);
    }
    for (i = 0; i < 20000; i++) {
        uint32_t s = rnd(seed);
        check_lfs(s);
        check_stfs((uint64_t)rnd(seed) << 32 | rnd(seed));
    }
}


/*
 * Differential test for the inline scalar FP ops (fp-impl.c.inc
 * native_fp_op()): every A-form arithmetic op, with the target distinct
 * from or equal to each input, on operands from all the classes that
 * decide between the inline path and the helper (zeros, denormals,
 * infinities, quiet and signalling NaNs, exact and inexact singles,
 * cancellation, overflow).  Each case stores the result and FPSCR at
 * FPD_BASE; run.py --fpdiff runs this with PPC_NATIVE_FP=0 and =1 and
 * compares the two memory images.
 */
#define FPD_BASE    0x01000000u
#define FPD_BUF     0x00100000u
#define FPD_REC     6           /* a, c, b, result, fpscr, case id (u64) */
/* the case id: FPSCR state << 16 | op << 8 | form, follow-up << 20, CR << 32 */

static uint64_t fpd_pool(uint32_t *seed)
{
    static const uint64_t special[] = {
        0x0000000000000000ull, 0x8000000000000000ull,  /* +-0 */
        0x0000000000000001ull, 0x800fffffffffffffull,  /* denormals */
        0x0010000000000000ull, 0x8010000000000000ull,  /* min normal */
        0x3ff0000000000000ull, 0xbff0000000000000ull,  /* +-1 */
        0x7fefffffffffffffull, 0xffefffffffffffffull,  /* max */
        0x7ff0000000000000ull, 0xfff0000000000000ull,  /* +-inf */
        0x7ff8000000000000ull, 0xfff8000000000123ull,  /* qNaN */
        0x7ff0000000000001ull, 0xfff4000000000000ull,  /* sNaN */
        0x3810000000000000ull, 0x36a0000000000000ull,  /* single min normal, single denormal */
        0x47efffffe0000000ull, 0x4800000000000000ull,  /* single max, just above */
        0x3ff0000000000001ull, 0x4000000010000000ull,  /* inexact singles */
        0x4059000000000000ull, 0xc059000000000000ull,  /* +-100 */
        0x8000000000000001ull, 0x000fffffffffffffull,  /* -min, +max denormal */
        0x380fffffffffffffull, 0xb80ffffff0000000ull,  /* just under 2^-126 */
        0x47efffffefffffffull, 0xc7effffff0000000ull,  /* near single max */
        0x3ff0000010000000ull, 0x3ff0000030000000ull,  /* single ties */
    };
    uint32_t r = rnd(seed);
    uint64_t m = (uint64_t)rnd(seed) << 32 | rnd(seed);

    switch (r & 7) {
    case 0: case 1:
        return special[(r >> 3) % (sizeof(special) / sizeof(special[0]))];
    case 2: case 3:     /* exact single, moderate exponent */
        return (m & 0x800fffffe0000000ull) | (uint64_t)(1000 + (r >> 8) % 46) << 52;
    case 4:             /* double, moderate exponent */
        return (m & 0x800fffffffffffffull) | (uint64_t)(990 + (r >> 8) % 70) << 52;
    case 5:             /* exact single, full single range */
        return (m & 0x800fffffe0000000ull) | (uint64_t)(897 + (r >> 8) % 254) << 52;
    default:            /* anything */
        return m;
    }
}

#define FPD_ASM(insn) \
    asm volatile("lfd 2,0(%0)\n\tlfd 3,8(%0)\n\tlfd 4,16(%0)\n\tlfd 1,40(%0)\n\t" \
                 "lfd 5,48(%0)\n\tmtfsf 0xff,5\n\t" insn \
                 : : "b"(buf) : "fr1", "fr2", "fr3", "fr4", "fr5", "memory")
#define FPD_OUT(t) asm volatile("stfd " #t ",24(%0)" : : "b"(buf) : "memory")

/* one op, four target choices: f1 (distinct), f2 (=A), f3 (=C), f4 (=B) */
#define FPD_OP3(m) \
    case 0: FPD_ASM(m " 1,2,3,4"); FPD_OUT(1); break; \
    case 1: FPD_ASM(m " 2,2,3,4"); FPD_OUT(2); break; \
    case 2: FPD_ASM(m " 3,2,3,4"); FPD_OUT(3); break; \
    case 3: FPD_ASM(m " 4,2,3,4"); FPD_OUT(4); break;
#define FPD_OPB(m) \
    case 0: FPD_ASM(m " 1,2,4"); FPD_OUT(1); break; \
    case 1: FPD_ASM(m " 2,2,4"); FPD_OUT(2); break; \
    case 2: FPD_ASM(m " 4,4,2"); FPD_OUT(4); break; \
    case 3: FPD_ASM(m " 4,2,4"); FPD_OUT(4); break;
#define FPD_OPR(m) \
    case 0: FPD_ASM(m " 1,4"); FPD_OUT(1); break; \
    case 1: FPD_ASM(m " 4,4"); FPD_OUT(4); break; \
    case 2: FPD_ASM(m " 1,2"); FPD_OUT(1); break; \
    case 3: FPD_ASM(m " 2,2"); FPD_OUT(2); break;
#define FPD_OPC(m) \
    case 0: FPD_ASM(m " 1,2,3"); FPD_OUT(1); break; \
    case 1: FPD_ASM(m " 2,2,3"); FPD_OUT(2); break; \
    case 2: FPD_ASM(m " 3,2,3"); FPD_OUT(3); break; \
    case 3: FPD_ASM(m " 3,3,2"); FPD_OUT(3); break;

static void fpd_one(volatile uint64_t *buf, int op, int form)
{
    switch (op) {
    case 0:  switch (form) { FPD_OPB("fadd") }    break;
    case 1:  switch (form) { FPD_OPB("fadds") }   break;
    case 2:  switch (form) { FPD_OPB("fsub") }    break;
    case 3:  switch (form) { FPD_OPB("fsubs") }   break;
    case 4:  switch (form) { FPD_OPC("fmul") }    break;
    case 5:  switch (form) { FPD_OPC("fmuls") }   break;
    case 6:  switch (form) { FPD_OPB("fdiv") }    break;
    case 7:  switch (form) { FPD_OPB("fdivs") }   break;
    case 8:  switch (form) { FPD_OP3("fmadd") }   break;
    case 9:  switch (form) { FPD_OP3("fmadds") }  break;
    case 10: switch (form) { FPD_OP3("fmsub") }   break;
    case 11: switch (form) { FPD_OP3("fmsubs") }  break;
    case 12: switch (form) { FPD_OP3("fnmadd") }  break;
    case 13: switch (form) { FPD_OP3("fnmadds") } break;
    case 14: switch (form) { FPD_OP3("fnmsub") }  break;
    case 15: switch (form) { FPD_OP3("fnmsubs") } break;
    case 16: switch (form) { FPD_OPR("frsp") }    break;
    }
}

/*
 * What reads FPSCR after the op, in a separate asm (maybe a separate TB):
 * mffs alone, or after an instruction that changes or reads part of FPRF.
 * The CR is returned for mcrfs.
 */
static uint32_t fpd_tail(volatile uint64_t *buf, int follow)
{
    uint32_t cr = 0;

    switch (follow) {
    case 0:
        asm volatile("mffs 5\n\tstfd 5,32(%0)"
                     : : "b"(buf) : "fr5", "memory");
        break;
    case 1:     /* FPCC only, C kept */
        asm volatile("fcmpu 1,2,4\n\tmffs 5\n\tstfd 5,32(%1)\n\tmfcr %0"
                     : "=r"(cr) : "b"(buf) : "fr5", "cr1", "memory");
        break;
    case 2:     /* one FPCC bit (FU) */
        asm volatile("mtfsb1 19\n\tmffs 5\n\tstfd 5,32(%0)"
                     : : "b"(buf) : "fr5", "memory");
        break;
    default:    /* FPCC to CR1 */
        asm volatile("mcrfs 1,4\n\tmffs 5\n\tstfd 5,32(%1)\n\tmfcr %0"
                     : "=r"(cr) : "b"(buf) : "fr5", "cr1", "memory");
        break;
    }
    return cr;
}

static void fp_diff(uint32_t *seed)
{
    volatile uint64_t *buf = (volatile uint64_t *)FPD_BUF;
    volatile uint64_t *out = (volatile uint64_t *)FPD_BASE;
    uint32_t n = 0;
    /* FPSCR states: fast mode, then round-toward-zero (helper only) */
    static const uint64_t fpscr_in[2] = { 0, 1 };

    for (int st = 0; st < 2; st++) {
        for (int it = 0; it < (st ? 300 : 3000); it++) {
            uint64_t a = fpd_pool(seed), c = fpd_pool(seed), b = fpd_pool(seed);
            uint32_t k = rnd(seed);
            if ((k & 15) == 0) {
                b = a;                        /* x - x, x + -x shapes */
            } else if ((k & 15) == 1) {
                b = a ^ 0x8000000000000000ull;
            }
            for (int op = 0; op < 17; op++) {
                for (int form = 0; form < 4; form++) {
                    buf[0] = a; buf[1] = c; buf[2] = b;
                    buf[5] = 0x7ff8dead0000beefull;   /* old f1 */
                    buf[6] = fpscr_in[st];
                    int follow = it & 3;
                    uint32_t cr;

                    fpd_one(buf, op, form);
                    cr = fpd_tail(buf, follow);
                    out[0] = a; out[1] = c; out[2] = b;
                    out[3] = buf[3]; out[4] = buf[4];
                    out[5] = (uint64_t)cr << 32 | follow << 20 |
                             st << 16 | op << 8 | form;
                    out += FPD_REC;
                    n++;
                }
            }
        }
    }
    res[0x283] = n;
    /* back to FPSCR 0: round to nearest, as the benchmarks expect */
    asm volatile("mtfsfi 7,0");
}

int main(void)
{
    uint32_t seed = 0x2545f491;
    uint32_t checks = 0, fails = 0, logged = 0;
    int op, it, i;

    for (i = 0; i < 0x300; i++) {
        res[i] = 0;
    }
    for (op = 0; op < NOPS; op++) {
        for (it = 0; it < ITERS; it++) {
            V a = rand_vec(&seed), b = rand_vec(&seed), c = rand_vec(&seed);
            V got, exp;
            int gs, es, bad = 0;

            if (op >= FIRST_FP_OP) {
                a = rand_fvec(&seed);
                b = rand_fvec(&seed);
                c = rand_fvec(&seed);
            }
            run_op(op, a, b, c, it & 15, &got, &exp, &gs, &es);
            for (i = 0; i < 4; i++) {
                if (op >= FIRST_FP_OP && is_nan(exp.w[i])) {
                    /* NaN payloads are softfloat's either way. */
                    bad |= !is_nan(got.w[i]);
                } else {
                    bad |= got.w[i] != exp.w[i];
                }
            }
            bad |= gs != es;
            checks++;
            if (bad) {
                fails++;
                res[3 + op]++;
                if (logged < MAXLOG) {
                    volatile uint32_t *p = res + 0x40 + logged * 24;
                    p[0] = op;
                    p[1] = it | gs << 16 | es << 24;
                    log_vec(p + 4, a);
                    log_vec(p + 8, b);
                    log_vec(p + 12, c);
                    log_vec(p + 16, got);
                    log_vec(p + 20, exp);
                    logged++;
                }
            }
        }
    }
    res[1] = checks;
    res[2] = fails;
    scalar_ldst(&seed);
    fp_diff(&seed);
    bench();
    res[0] = 0x564d5854; /* 'VMXT' */
    return 0;
}
