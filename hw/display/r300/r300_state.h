/*
 * R300-family 3D engine state: the register file the command processor
 * writes, plus the memories behind the indexed upload ports.
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_STATE_H
#define HW_DISPLAY_R300_STATE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* 3D register window mirrored here (VAP 0x2000.. through ZB 0x4Fxx). */
#define R300_REG_BASE           0x1C00
#define R300_REG_END            0x5000
#define R300_REG_COUNT          ((R300_REG_END - R300_REG_BASE) / 4)

/* PVS vector memory: code at 0, constants at 512, clip planes at 1024. */
#define R300_PVS_MEM_VECS       1536
#define R300_PVS_CODE_START     0
#define R300_PVS_CONST_START    512
#define R300_PVS_UCP_START      1024

#define R300_VAP_PVS_VECTOR_INDX_REG    0x2200
#define R300_VAP_PVS_UPLOAD_DATA        0x2208

typedef struct R300State {
    uint32_t regs[R300_REG_COUNT];
    uint32_t pvs_mem[R300_PVS_MEM_VECS * 4];   /* 4 dwords per vector */
    uint32_t pvs_upload_dw;                    /* next dword to write */
    uint64_t pvs_gen;       /* bumps when PVS code/constants change */
    uint64_t draws;
    uint32_t glsl_flags;    /* r300_us_to_glsl() flags the renderer wants (not a
                               register; the device sets it before each draw) */
} R300State;

void r300_state_reset(R300State *st);

/* A register write from MMIO or a PM4 type-0/1 packet.  addr is a byte
 * offset in the MMIO BAR; writes outside the 3D window are ignored. */
void r300_state_write(R300State *st, uint32_t addr, uint32_t val);

static inline uint32_t r300_reg(const R300State *st, uint32_t addr)
{
    return st->regs[(addr - R300_REG_BASE) / 4];
}

static inline bool r300_state_owns(uint32_t addr)
{
    return addr >= R300_REG_BASE && addr < R300_REG_END;
}

/* Print every non-zero register (with names where known) and the live
 * part of PVS memory, for reverse-engineering what a driver programs. */
void r300_state_dump(const R300State *st, FILE *f);

/* Register name for traces, or NULL. */
const char *r300_reg_name(uint32_t addr);

#endif
