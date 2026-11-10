/* Reset entry: MSR[VEC] and MSR[FP] on, stack in RAM, call main, spin. */
    .section .text.start, "ax"
    .globl _start
_start:
    lis     %r3, 0x0200         /* MSR_VEC */
    ori     %r3, %r3, 0x2000    /* MSR_FP */
    mtmsr   %r3
    isync
    lis     %r1, 0x0080         /* stack at 8 MB, real mode */
    li      %r0, 0
    stwu    %r0, -64(%r1)
    bl      main
1:  b       1b
