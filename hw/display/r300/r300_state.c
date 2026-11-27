/*
 * R300-family 3D engine state.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_state.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "r300_regnames.h"

#define R300_VAP_PVS_CODE_CNTL_0    0x22D0

void r300_state_reset(R300State *st)
{
    memset(st, 0, sizeof(*st));
}

void r300_state_write(R300State *st, uint32_t addr, uint32_t val)
{
    if (!r300_state_owns(addr)) {
        return;
    }
    st->regs[(addr - R300_REG_BASE) / 4] = val;

    switch (addr) {
    case R300_VAP_PVS_VECTOR_INDX_REG:
        /* Vector index; data writes then fill 4 dwords per vector. */
        st->pvs_upload_dw = (val & 0x7FF) * 4;
        break;
    case R300_VAP_PVS_UPLOAD_DATA:
        if (st->pvs_upload_dw < R300_PVS_MEM_VECS * 4) {
            st->pvs_mem[st->pvs_upload_dw] = val;
        }
        st->pvs_upload_dw++;
        st->pvs_gen++;
        break;
    }
}

const char *r300_reg_name(uint32_t addr)
{
    size_t lo = 0, hi = sizeof(r300_reg_names) / sizeof(r300_reg_names[0]);

    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (r300_reg_names[mid].addr == addr) {
            return r300_reg_names[mid].name;
        }
        if (r300_reg_names[mid].addr < addr) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return NULL;
}

void r300_state_dump(const R300State *st, FILE *f)
{
    fprintf(f, "=== R300 state at draw %" PRIu64 " ===\n", st->draws);
    for (uint32_t i = 0; i < R300_REG_COUNT; i++) {
        uint32_t addr = R300_REG_BASE + i * 4;
        uint32_t v = st->regs[i];
        const char *n;

        if (!v || addr == R300_VAP_PVS_UPLOAD_DATA) {
            continue;
        }
        n = r300_reg_name(addr);
        fprintf(f, "  %04X %08X  %s\n", addr, v, n ? n : "");
    }

    uint32_t cntl = r300_reg(st, R300_VAP_PVS_CODE_CNTL_0);
    unsigned first = cntl & 0x3FF, last = (cntl >> 20) & 0x3FF;
    fprintf(f, "  PVS code %u..%u (xyzw valid after %u):\n",
            first, last, (cntl >> 10) & 0x3FF);
    for (unsigned i = first; i <= last && i < 1024; i++) {
        const uint32_t *d = &st->pvs_mem[(R300_PVS_CODE_START + i) * 4];
        fprintf(f, "    %3u: %08X %08X %08X %08X\n", i, d[0], d[1], d[2], d[3]);
    }
    fprintf(f, "  PVS constants (non-zero):\n");
    for (unsigned i = 0; i < 256 + 6; i++) {
        const uint32_t *d = &st->pvs_mem[(R300_PVS_CONST_START + i) * 4];
        if (d[0] | d[1] | d[2] | d[3]) {
            float fv[4];
            memcpy(fv, d, sizeof(fv));
            fprintf(f, "    c%-3u %g %g %g %g\n", i, fv[0], fv[1], fv[2], fv[3]);
        }
    }
    fflush(f);
}
