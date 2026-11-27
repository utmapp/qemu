/*
 * R300-family Programmable Vertex Shader (PVS).
 *
 * The PVS executes 128-bit instructions (one opcode/destination dword and
 * three source dwords).  Encodings and opcode semantics follow the AMD
 * "R5xx Acceleration" guide, section 7.5, restricted to what R3xx/R4xx
 * implement.
 *
 * This is a reference interpreter: the device runs vertex programs on the
 * host CPU and hands Metal post-transform vertices.  Pure C, no QEMU
 * dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_PVS_H
#define HW_DISPLAY_R300_PVS_H

#include <stdbool.h>
#include <stdint.h>

#define R300_PVS_MAX_INSTS      256     /* R3xx/R4xx code memory, in insts */
#define R300_PVS_NUM_INPUTS     32
#define R300_PVS_NUM_OUTPUTS    32
#define R300_PVS_NUM_TEMPS      128
#define R300_PVS_NUM_ALT_TEMPS  20

typedef struct R300PVSProgram {
    const uint32_t *code;   /* PVS code memory, 4 dwords per inst */
    unsigned first_inst;    /* VAP_PVS_CODE_CNTL_0.PVS_FIRST_INST */
    unsigned last_inst;     /* VAP_PVS_CODE_CNTL_0.PVS_LAST_INST */
    const float (*consts)[4];   /* constant memory, from CONST_BASE_OFFSET */
    int max_const;          /* reads beyond this return 0 */
    /* Flow control (AMD R5xx guide 7.5.5): VAP_PVS_FLOW_CNTL_OPC (2 bits
     * per instruction: 0 none, 1 JUMP, 2 LOOP, 3 JSR), _ADDRS and
     * _LOOP_INDEX.  Zero fc_opc for straight-line programs. */
    uint32_t fc_opc;
    uint32_t fc_addrs[16];
    uint32_t fc_loop[16];
} R300PVSProgram;

/*
 * Run the program for one vertex.  in[] is the input vertex memory (IVM),
 * out[] receives the output vertex memory (OVM); outputs the program does
 * not write keep their previous contents.  Returns a bitmask of flags for
 * things the interpreter does not implement (0 when fully handled).
 */
#define R300_PVS_UNSUP_FLOW     (1u << 0)   /* flow control ran past its step limit */
#define R300_PVS_UNSUP_PRED     (1u << 1)   /* R5xx predication */
#define R300_PVS_UNSUP_OPCODE   (1u << 2)   /* unknown opcode */
#define R300_PVS_UNSUP_RELDST   (1u << 3)   /* relative destination address */

uint32_t r300_pvs_run(const R300PVSProgram *prog,
                      const float in[R300_PVS_NUM_INPUTS][4],
                      float out[R300_PVS_NUM_OUTPUTS][4]);

/* One line of disassembly per instruction into buf; for traces. */
void r300_pvs_disasm_inst(const uint32_t d[4], char *buf, unsigned len);

/*
 * The program as GLSL statements for the body of a vertex shader's main()
 * (declarations first): inputs are read from locals iN (the caller loads
 * those *in_used names), constants from uniform vs.c[] with a function
 * vec4 pvs_c(int) for A0-relative reads (the caller defines it), and the
 * outputs land in locals o0..o31.  r300_pvs_glsl_helpers goes at file
 * scope.  False when the program needs the interpreter (flow control,
 * predication, relative temporaries or outputs, unknown opcodes).
 */
struct R300Sb;
extern const char r300_pvs_glsl_helpers[];
bool r300_pvs_to_glsl(const R300PVSProgram *prog, struct R300Sb *sb,
                      uint32_t *in_used);

#endif
