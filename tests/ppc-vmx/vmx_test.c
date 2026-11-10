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
    bench();
    res[0] = 0x564d5854; /* 'VMXT' */
    return 0;
}
