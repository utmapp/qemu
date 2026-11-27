/*
 * R200 vertex programs - see hw/display/ppc_mac_gpu_vp.c.
 *
 * Copyright (c) 2026 Spartan0285
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef PPC_MAC_GPU_VP_H
#define PPC_MAC_GPU_VP_H

/* Where a result goes (instruction word 0, bits 11:8). */
enum {
    VP_DST_TEMP      = 0,
    VP_DST_ADDR      = 3,
    VP_DST_POS       = 4,
    VP_DST_COLOR     = 5,
    VP_DST_TEXCOORD  = 6,
    VP_DST_FOG       = 7,
    VP_DST_POINTSIZE = 8,
};

/* What a source reads (source word, bits 3:0). */
enum {
    VP_SRC_TEMP  = 0,
    VP_SRC_ATTR  = 1,
    VP_SRC_CONST = 2,
    VP_SRC_NONE  = 9,
};

/* Component choices (three bits per destination component). */
enum {
    VP_SEL_X = 0, VP_SEL_Y = 1, VP_SEL_Z = 2, VP_SEL_W = 3,
    VP_SEL_ZERO = 4, VP_SEL_ONE = 5,
};

typedef struct PPCMacGPUVPSrc {
    uint8_t class;
    uint8_t index;
    uint8_t relative;
    uint8_t select[4];
    uint8_t negate[4];
} PPCMacGPUVPSrc;

typedef struct PPCMacGPUVPInst {
    uint32_t op;                /* opcode, engine and macro bits included */
    uint8_t dst_class;
    uint8_t dst_index;
    uint8_t write;              /* which of x, y, z, w to write */
    uint8_t sources;            /* how many source operands it reads */
    PPCMacGPUVPSrc src[3];
} PPCMacGPUVPInst;

#define PPC_MAC_GPU_VP_CONSTS 192
#define PPC_MAC_GPU_VP_MAX_INST 128

/* One vertex going through a program: what it reads, and what it leaves. */
typedef struct PPCMacGPUVPState {
    float in[16][4];            /* the vertex's attributes */
    /* The program's constants belong to the draw, not the vertex: they are
     * pointed at, not copied, because copying 3 KB per vertex cost more
     * than running the program did. */
    const float (*constant)[4];
    float temp[16][4];
    float out_pos[4];
    float out_color0[4];
    float out_color1[4];
    float out_tex[6][4];
    float out_fog[4];
    float out_psize[4];
    bool wrote_pos;
    uint32_t unknown_op;        /* nonzero: an opcode not modelled yet */
} PPCMacGPUVPState;

void ppc_mac_gpu_vp_decode(const uint32_t w[4], PPCMacGPUVPInst *out);
void ppc_mac_gpu_vp_run(PPCMacGPUVPState *st, const PPCMacGPUVPInst *prog,
                        uint32_t count);
void ppc_mac_gpu_vp_disasm(const PPCMacGPUVPInst *in, char *buf, size_t len);
uint32_t ppc_mac_gpu_vp_inst_addr(uint32_t i);
uint32_t ppc_mac_gpu_vp_const_addr(uint32_t i);

#endif
