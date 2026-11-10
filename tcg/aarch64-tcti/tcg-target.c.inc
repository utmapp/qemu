/*
 * Tiny Code Threaded Intepreter for QEMU
 *
 * Copyright (c) 2021 Kate Temkin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


// Rich disassembly is nice in theory, but it's -slow-.
//#define TCTI_GADGET_RICH_DISASSEMBLY

#define TCTI_GADGET_IMMEDIATE_ARRAY_LEN 64

// Specify the shape of the stack our runtime will use.
#define TCG_TARGET_CALL_STACK_OFFSET    0
#define TCG_TARGET_STACK_ALIGN          16
#define TCG_TARGET_CALL_ARG_I32         TCG_CALL_ARG_NORMAL
#define TCG_TARGET_CALL_ARG_I64         TCG_CALL_ARG_NORMAL
#define TCG_TARGET_CALL_ARG_I128        TCG_CALL_ARG_NORMAL
#define TCG_TARGET_CALL_RET_I128        TCG_CALL_RET_NORMAL

#include "tcg/tcg-ldst.h"

// Grab our gadget headers.
#include "tcti_gadgets.h"

/* Marker for missing code. */
#define TODO() \
    do { \
        fprintf(stderr, "TODO %s:%u: %s()\n", \
                __FILE__, __LINE__, __func__); \
        g_assert_not_reached(); \
    } while (0)


/* Enable TCTI assertions only when debugging TCG (and without NDEBUG defined).
 * Without assertions, the interpreter runs much faster. */
#if defined(CONFIG_DEBUG_TCG)
# define tcti_assert(cond) assert(cond)
#else
# define tcti_assert(cond) ((void)0)
#endif


/********************************
 *  TCG Constraints Definitions *
 ********************************/

static TCGConstraintSetIndex
tcg_target_op_def(TCGOpcode op, TCGType type, unsigned flags)
{
    switch (op) {

    case INDEX_op_ld8u_i32:
    case INDEX_op_ld8s_i32:
    case INDEX_op_ld16u_i32:
    case INDEX_op_ld16s_i32:
    case INDEX_op_ld_i32:
    case INDEX_op_ld8u_i64:
    case INDEX_op_ld8s_i64:
    case INDEX_op_ld16u_i64:
    case INDEX_op_ld16s_i64:
    case INDEX_op_ld32u_i64:
    case INDEX_op_ld32s_i64:
    case INDEX_op_ld_i64:
    case INDEX_op_not_i32:
    case INDEX_op_not_i64:
    case INDEX_op_neg_i32:
    case INDEX_op_neg_i64:
    case INDEX_op_ext8s_i32:
    case INDEX_op_ext8s_i64:
    case INDEX_op_ext16s_i32:
    case INDEX_op_ext16s_i64:
    case INDEX_op_ext8u_i32:
    case INDEX_op_ext8u_i64:
    case INDEX_op_ext16u_i32:
    case INDEX_op_ext16u_i64:
    case INDEX_op_ext32s_i64:
    case INDEX_op_ext32u_i64:
    case INDEX_op_ext_i32_i64:
    case INDEX_op_extu_i32_i64:
    case INDEX_op_bswap16_i32:
    case INDEX_op_bswap16_i64:
    case INDEX_op_bswap32_i32:
    case INDEX_op_bswap32_i64:
    case INDEX_op_bswap64_i64:
    case INDEX_op_extrl_i64_i32:
    case INDEX_op_extrh_i64_i32:
        return C_O1_I1(r, r);

    case INDEX_op_st8_i32:
    case INDEX_op_st16_i32:
    case INDEX_op_st_i32:
    case INDEX_op_st8_i64:
    case INDEX_op_st16_i64:
    case INDEX_op_st32_i64:
    case INDEX_op_st_i64:
        return C_O0_I2(r, r);

    case INDEX_op_div_i32:
    case INDEX_op_div_i64:
    case INDEX_op_divu_i32:
    case INDEX_op_divu_i64:
    case INDEX_op_rem_i32:
    case INDEX_op_rem_i64:
    case INDEX_op_remu_i32:
    case INDEX_op_remu_i64:
    case INDEX_op_add_i32:
    case INDEX_op_add_i64:
    case INDEX_op_sub_i32:
    case INDEX_op_sub_i64:
    case INDEX_op_mul_i32:
    case INDEX_op_mul_i64:
    case INDEX_op_and_i32:
    case INDEX_op_and_i64:
    case INDEX_op_andc_i32:
    case INDEX_op_andc_i64:
    case INDEX_op_eqv_i32:
    case INDEX_op_eqv_i64:
    case INDEX_op_nand_i32:
    case INDEX_op_nand_i64:
    case INDEX_op_nor_i32:
    case INDEX_op_nor_i64:
    case INDEX_op_or_i32:
    case INDEX_op_or_i64:
    case INDEX_op_orc_i32:
    case INDEX_op_orc_i64:
    case INDEX_op_xor_i32:
    case INDEX_op_xor_i64:
    case INDEX_op_shl_i32:
    case INDEX_op_shl_i64:
    case INDEX_op_shr_i32:
    case INDEX_op_shr_i64:
    case INDEX_op_sar_i32:
    case INDEX_op_sar_i64:
    case INDEX_op_rotl_i32:
    case INDEX_op_rotl_i64:
    case INDEX_op_rotr_i32:
    case INDEX_op_rotr_i64:
    case INDEX_op_setcond_i32:
    case INDEX_op_setcond_i64:
    case INDEX_op_clz_i32:
    case INDEX_op_clz_i64:
    case INDEX_op_ctz_i32:
    case INDEX_op_ctz_i64:
        return C_O1_I2(r, r, r);

    case INDEX_op_brcond_i32:
    case INDEX_op_brcond_i64:
        return C_O0_I2(r, r);

    case INDEX_op_qemu_ld_i32:
    case INDEX_op_qemu_ld_i64:
        return C_O1_I2(r, r, r);
    case INDEX_op_qemu_st_i32:
    case INDEX_op_qemu_st_i64:
        return C_O0_I3(r, r, r);

    //
    // Vector ops.
    //
    case INDEX_op_add_vec:
    case INDEX_op_sub_vec:
    case INDEX_op_mul_vec:
    case INDEX_op_xor_vec:
    case INDEX_op_ssadd_vec:
    case INDEX_op_sssub_vec:
    case INDEX_op_usadd_vec:
    case INDEX_op_ussub_vec:
    case INDEX_op_smax_vec:
    case INDEX_op_smin_vec:
    case INDEX_op_umax_vec:
    case INDEX_op_umin_vec:
    case INDEX_op_shlv_vec:
    case INDEX_op_shrv_vec:
    case INDEX_op_sarv_vec:
    case INDEX_op_aa64_sshl_vec:
        return C_O1_I2(w, w, w);
    case INDEX_op_not_vec:
    case INDEX_op_neg_vec:
    case INDEX_op_abs_vec:
    case INDEX_op_shli_vec:
    case INDEX_op_shri_vec:
    case INDEX_op_sari_vec:
        return C_O1_I1(w, w);
    case INDEX_op_ld_vec:
    case INDEX_op_dupm_vec:
        return C_O1_I1(w, r);
    case INDEX_op_st_vec:
        return C_O0_I2(w, r);
    case INDEX_op_dup_vec:
        return C_O1_I1(w, wr);
    case INDEX_op_or_vec:
    case INDEX_op_andc_vec:
        return C_O1_I2(w, w, w);
    case INDEX_op_and_vec:
    case INDEX_op_orc_vec:
        return C_O1_I2(w, w, w);
    case INDEX_op_cmp_vec:
        return C_O1_I2(w, w, w);
    case INDEX_op_bitsel_vec:
        return C_O1_I3(w, w, w, w);
    case INDEX_op_aa64_sli_vec:
        return C_O1_I2(w, 0, w);

    default:
        return C_NotImplemented;
    }
}

static const int tcg_target_reg_alloc_order[] = {

    // General purpose registers, in preference-of-allocation order.
    TCG_REG_R8,
    TCG_REG_R9,
    TCG_REG_R10,
    TCG_REG_R11,
    TCG_REG_R12,
    TCG_REG_R13,
    TCG_REG_R0,
    TCG_REG_R1,
    TCG_REG_R2,
    TCG_REG_R3,
    TCG_REG_R4,
    TCG_REG_R5,
    TCG_REG_R6,
    TCG_REG_R7,

    // Note: we do not allocate R14 or R15, as they're used for our
    // special-purpose values.

    // We'll use the high 16 vector register; avoiding the call-saved lower ones.
    TCG_REG_V16, TCG_REG_V17, TCG_REG_V18, TCG_REG_V19,
    TCG_REG_V20, TCG_REG_V21, TCG_REG_V22, TCG_REG_V23,
    TCG_REG_V24, TCG_REG_V25, TCG_REG_V26, TCG_REG_V27,
    TCG_REG_V28, TCG_REG_V29, TCG_REG_V30, TCG_REG_V31,
};

static const int tcg_target_call_iarg_regs[] = {
    TCG_REG_R0,
    TCG_REG_R1,
    TCG_REG_R2,
    TCG_REG_R3,
    TCG_REG_R4,
    TCG_REG_R5,
};

static TCGReg tcg_target_call_oarg_reg(TCGCallReturnKind kind, int slot)
{
    tcg_debug_assert(kind == TCG_CALL_RET_NORMAL);
    tcg_debug_assert(slot >= 0 && slot < 128 / TCG_TARGET_REG_BITS);
    return TCG_REG_R0 + slot;
}

#ifdef CONFIG_DEBUG_TCG
static const char *const tcg_target_reg_names[TCG_TARGET_GP_REGS] = {
    "r00",
    "r01",
    "r02",
    "r03",
    "r04",
    "r05",
    "r06",
    "r07",
    "r08",
    "r09",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
};
#endif

/*************************
 *  TCG Emitter Helpers  *
 *************************/

/* Bitfield n...m (in 32 bit value). */
#define BITS(n, m) (((0xffffffffU << (31 - n)) >> (31 - n + m)) << m)

/**
 * Macro that defines a look-up tree for named QEMU_LD gadgets.
 */
#define LD_MEMOP_LOOKUP(variable, arg, suffix) \
    switch (get_memop(arg) & MO_SSIZE) { \
        case MO_UB:   variable = gadget_qemu_ld_ub_   ## suffix; break; \
        case MO_SB:   variable = gadget_qemu_ld_sb_   ## suffix; break; \
        case MO_UW: variable = gadget_qemu_ld_leuw_ ## suffix; break; \
        case MO_SW: variable = gadget_qemu_ld_lesw_ ## suffix; break; \
        case MO_UL: variable = gadget_qemu_ld_leul_ ## suffix; break; \
        case MO_SL: variable = gadget_qemu_ld_lesl_ ## suffix; break; \
        case MO_UQ:  variable = gadget_qemu_ld_leq_  ## suffix; break; \
        default: \
            g_assert_not_reached(); \
    }
#define LD_MEMOP_HANDLER(variable, arg, suffix, a_bits, s_bits) \
        if (a_bits >= s_bits) { \
            LD_MEMOP_LOOKUP(variable, arg, aligned_ ## suffix ); \
        } else { \
            LD_MEMOP_LOOKUP(gadget, arg, unaligned_ ## suffix); \
        }



/**
 * Macro that defines a look-up tree for named QEMU_ST gadgets.
 */
#define ST_MEMOP_LOOKUP(variable, arg, suffix) \
    switch (get_memop(arg) & MO_SSIZE) { \
        case MO_UB:   variable = gadget_qemu_st_ub_   ## suffix; break; \
        case MO_UW: variable = gadget_qemu_st_leuw_ ## suffix; break; \
        case MO_UL: variable = gadget_qemu_st_leul_ ## suffix; break; \
        case MO_UQ:  variable = gadget_qemu_st_leq_  ## suffix; break; \
        default: \
            g_assert_not_reached(); \
    }
#define ST_MEMOP_HANDLER(variable, arg, suffix, a_bits, s_bits) \
        if (a_bits >= s_bits) { \
            ST_MEMOP_LOOKUP(variable, arg, aligned_ ## suffix ); \
        } else { \
            ST_MEMOP_LOOKUP(gadget, arg, unaligned_ ## suffix); \
        }


#define LOOKUP_SPECIAL_CASE_LDST_GADGET(arg, name, mode) \
    switch(tlb_mask_table_ofs(s, get_mmuidx(arg))) { \
        case -32:  \
            gadget = (a_bits >= s_bits) ?  \
                gadget_qemu_ ## name ## _aligned_ ## mode ## _off32_i64 : \
                gadget_qemu_ ## name ## _unaligned_ ## mode ## _off32_i64; \
            break; \
        case -48:  \
            gadget = (a_bits >= s_bits) ?  \
                gadget_qemu_ ## name ## _aligned_ ## mode ## _off48_i64 : \
                gadget_qemu_ ## name ## _unaligned_ ## mode ## _off48_i64; \
            break; \
        case -64: \
            gadget = (a_bits >= s_bits) ? \
                gadget_qemu_ ## name ## _aligned_ ## mode ## _off64_i64 : \
                gadget_qemu_ ## name ## _unaligned_ ## mode ## _off64_i64; \
            break; \
        case -96: \
            gadget = (a_bits >= s_bits) ? \
                gadget_qemu_ ## name ## _aligned_ ## mode ## _off96_i64 : \
                gadget_qemu_ ## name ## _unaligned_ ## mode ## _off96_i64; \
            break; \
        case -128: \
            gadget = (a_bits >= s_bits) ? \
                gadget_qemu_ ## name ## _aligned_ ## mode ## _off128_i64 : \
                gadget_qemu_ ## name ## _unaligned_ ## mode ## _off128_i64; \
            break;\
        default: \
            gadget = gadget_qemu_ ## name ## _slowpath_ ## mode ## _off0_i64; \
            break; \
        }


static bool patch_reloc(tcg_insn_unit *code_ptr, int type,
                        intptr_t value, intptr_t addend)
{
    /* tcg_out_reloc always uses the same type, addend. */
    tcg_debug_assert(type == sizeof(tcg_target_long));
    tcg_debug_assert(addend == 0);
    tcg_debug_assert(value != 0);
    if (TCG_TARGET_REG_BITS == 32) {
        tcg_patch32(code_ptr, value);
    } else {
        tcg_patch64(code_ptr, value);
    }
    return true;
}

#if defined(CONFIG_DEBUG_TCG_INTERPRETER)
/* Show current bytecode. Used by tcg interpreter. */
void tci_disas(uint8_t opc)
{
    const TCGOpDef *def = &tcg_op_defs[opc];
    fprintf(stderr, "TCG %s %u, %u, %u\n",
            def->name, def->nb_oargs, def->nb_iargs, def->nb_cargs);
}
#endif

/* Write value (native size). */
static void tcg_out_immediate(TCGContext *s, tcg_target_ulong v)
{
    if (TCG_TARGET_REG_BITS == 32) {
        //tcg_out32(s, v);
        tcg_out64(s, v);
    } else {
        tcg_out64(s, v);
    }
}

void tb_target_set_jmp_target(const TranslationBlock *tb, int n,
                              uintptr_t jmp_rx, uintptr_t jmp_rw)
{
    /* Get a pointer to our immediate, which exists after a single pointer. */
    uintptr_t immediate_addr = jmp_rw;
    uintptr_t addr = tb->jmp_target_addr[n];

    /* Patch it to be match our target address. */
    qatomic_set((uint64_t *)immediate_addr, addr);
}


/**
 * TCTI Thunk Helpers
 */

#ifdef CONFIG_SOFTMMU

// TODO: relocate these prototypes?
tcg_target_ulong helper_ldub_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr);
tcg_target_ulong helper_lduw_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr);
tcg_target_ulong helper_ldul_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr);

tcg_target_ulong helper_ldub_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr)
{
    return (int8_t)helper_ldub_mmu(env, addr, oi, retaddr);
}

tcg_target_ulong helper_lduw_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr)
{
    return (int16_t)helper_lduw_mmu(env, addr, oi, retaddr);
}

tcg_target_ulong helper_ldul_mmu_signed(CPUArchState *env, uint64_t addr, MemOpIdx oi, uintptr_t retaddr)
{
    return (int32_t)helper_ldul_mmu(env, addr, oi, retaddr);
}

#else
#error TCTI currently only supports use of the soft MMU.
#endif


/**
 * TCTI Emmiter Helpers
 */


/* Write gadget pointer. */
static void tcg_out_gadget(TCGContext *s, const void *gadget)
{
    tcg_out_immediate(s, (tcg_target_ulong)gadget);
}

/* Write gadget pointer, plus 64b immediate. */
static void tcg_out_imm64_gadget(TCGContext *s, const void *gadget, tcg_target_ulong immediate)
{
    tcg_out_gadget(s, gadget);
    tcg_out64(s, immediate);
}


/* Write gadget pointer (one register). */
static void tcg_out_unary_gadget(TCGContext *s, const void *gadget_base[TCG_TARGET_GP_REGS], unsigned reg0)
{
    tcg_out_gadget(s, gadget_base[reg0]);
}


/* Write gadget pointer (two registers). */
static void tcg_out_binary_gadget(TCGContext *s, const void *gadget_base[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS], unsigned reg0, unsigned reg1)
{
    tcg_out_gadget(s, gadget_base[reg0][reg1]);
}


/* Write gadget pointer (three registers). */
static void tcg_out_ternary_gadget(TCGContext *s, const void *gadget_base[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS], unsigned reg0, unsigned reg1, unsigned reg2)
{
    tcg_out_gadget(s, gadget_base[reg0][reg1][reg2]);
}


/* Write gadget pointer (three registers, last is immediate value). */
static void tcg_out_ternary_immediate_gadget(TCGContext *s, const void *gadget_base[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS][TCTI_GADGET_IMMEDIATE_ARRAY_LEN], unsigned reg0, unsigned reg1, unsigned reg2)
{
    tcg_out_gadget(s, gadget_base[reg0][reg1][reg2]);
}

/***************************
 *  TCG Scalar Operations  *
 ***************************/

/**
 * Version of our LDST generator that defers to more optimized gadgets selectively.
 */
static void tcg_out_ldst_gadget_inner(TCGContext *s,
    const void *gadget_base[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS],
    const void *gadget_pos_imm[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS][TCTI_GADGET_IMMEDIATE_ARRAY_LEN],
    const void *gadget_shifted_imm[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS][TCTI_GADGET_IMMEDIATE_ARRAY_LEN],
    const void *gadget_neg_imm[TCG_TARGET_GP_REGS][TCG_TARGET_GP_REGS][TCTI_GADGET_IMMEDIATE_ARRAY_LEN],
    unsigned reg0, unsigned reg1, uint32_t offset)
{
    int64_t extended_offset = (int32_t)offset;
    bool is_negative = (extended_offset < 0);

    // Optimal case: we have a gadget that handles our specific offset, so we don't need to encode
    // an immediate. This saves us a bunch of speed. :)

    // We handle positive and negative gadgets separately, in order to allow for asymmetrical
    // collections of pre-made gadgets.
    if (!is_negative)
    {
        uint64_t shifted_offset = (extended_offset >> 3);
        bool aligned_to_8B = ((extended_offset & 0b111) == 0);

        bool have_optimized_gadget = (extended_offset < TCTI_GADGET_IMMEDIATE_ARRAY_LEN);
        bool have_shifted_gadget   = (shifted_offset  < TCTI_GADGET_IMMEDIATE_ARRAY_LEN);

        // More optimal case: we have a gadget that directly encodes the argument.
        if (have_optimized_gadget) {
            tcg_out_gadget(s, gadget_pos_imm[reg0][reg1][extended_offset]);
            return;
        }

        // Special case: it's frequent to have low-numbered positive offsets that are aligned
        // to 16B boundaries
        else if(aligned_to_8B && have_shifted_gadget) {
            tcg_out_gadget(s, gadget_shifted_imm[reg0][reg1][shifted_offset]);
            return;
        }
    }
    else {
        uint64_t negated_offset = -(extended_offset);

        // More optimal case: we have a gadget that directly encodes the argument.
        if (negated_offset < TCTI_GADGET_IMMEDIATE_ARRAY_LEN) {
            tcg_out_gadget(s, gadget_neg_imm[reg0][reg1][negated_offset]);
            return;
        }
    }

    // Less optimal case: we don't have a gadget specifically for this. Emit the general case immediate.
    tcg_out_binary_gadget(s, gadget_base, reg0, reg1);
    tcg_out64(s, extended_offset); //tcg_out32(s, offset);
}

/* Shorthand for the above, that prevents us from having to specify the name three times. */
#define tcg_out_ldst_gadget(s, name, a, b, c) \
    tcg_out_ldst_gadget_inner(s, name, \
        name ## _imm,  \
        name ## _sh8_imm,  \
        name ## _neg_imm, \
    a, b, c)



/* Write label. */
static void tcti_out_label(TCGContext *s, TCGLabel *label)
{
    if (label->has_value) {
        tcg_out64(s, label->u.value);
        tcg_debug_assert(label->u.value);
    } else {
        tcg_out_reloc(s, s->code_ptr, sizeof(tcg_target_ulong), label, 0);
        s->code_ptr += sizeof(tcg_target_ulong);
    }
}


/* Register to register move using ORR (shifted register with no shift). */
static void tcg_out_movr(TCGContext *s, TCGType ext, TCGReg rd, TCGReg rm)
{
    switch(ext) {
        case TCG_TYPE_I32:
            tcg_out_binary_gadget(s, gadget_mov_i32, rd, rm);
            break;

        case TCG_TYPE_I64:
            tcg_out_binary_gadget(s, gadget_mov_i64, rd, rm);
            break;

        default:
            g_assert_not_reached();

    }
}


static bool tcg_out_mov(TCGContext *s, TCGType type, TCGReg ret, TCGReg arg)
{
    TCGReg w_ret = (ret - TCG_REG_V16);
    TCGReg w_arg = (arg - TCG_REG_V16);

    if (ret == arg) {
        return true;
    }

    switch (type) {
    case TCG_TYPE_I32:
    case TCG_TYPE_I64:

        // If this is a GP to GP register mov, issue our standard MOV.
        if (ret < 32 && arg < 32) {
            tcg_out_movr(s, type, ret, arg);
            break;
        } 
        // If this is a vector register to GP, issue a UMOV.
        else if (ret < 32) {
            void *gadget = (type == TCG_TYPE_I32) ? gadget_umov_s0 : gadget_umov_d0;
            tcg_out_binary_gadget(s, gadget, ret, w_arg);
            break;
        } 
        
        // If this is a GP to vector move, insert the vealue using INS.
        else if (arg < 32) {
            void *gadget = (type == TCG_TYPE_I32) ? gadget_ins_s0 : gadget_ins_d0;
            tcg_out_binary_gadget(s, gadget, w_ret, arg);
            break;
        }
        /* FALLTHRU */

    case TCG_TYPE_V64:
        tcg_debug_assert(ret >= 32 && arg >= 32);
        tcg_out_ternary_gadget(s, gadget_or_d, w_ret, w_arg, w_arg);
        break;

    case TCG_TYPE_V128:
        tcg_debug_assert(ret >= 32 && arg >= 32);
        tcg_out_ternary_gadget(s, gadget_or_q, w_ret, w_arg, w_arg);
        break;

    default:
        g_assert_not_reached();
    }
    return true;
}



static void tcg_out_movi_i32(TCGContext *s, TCGReg t0, tcg_target_long arg)
{
    bool is_negative = (arg < 0);

    // We handle positive and negative gadgets separately, in order to allow for asymmetrical
    // collections of pre-made gadgets.
    if (!is_negative)
    {
        // More optimal case: we have a gadget that directly encodes the argument.
        if (arg < ARRAY_SIZE(gadget_movi_imm_i32[t0])) {
            tcg_out_gadget(s, gadget_movi_imm_i32[t0][arg]);
            return;
        }
    }

    // Emit the mov and its immediate.
    tcg_out_unary_gadget(s, gadget_movi_i32, t0);
    tcg_out64(s, arg); // TODO: make 32b?
}


static void tcg_out_movi_i64(TCGContext *s, TCGReg t0, tcg_target_long arg)
{
    uint8_t is_negative = arg < 0;

    // We handle positive and negative gadgets separately, in order to allow for asymmetrical
    // collections of pre-made gadgets.
    if (!is_negative)
    {
        // More optimal case: we have a gadget that directly encodes the argument.
        if (arg < ARRAY_SIZE(gadget_movi_imm_i64[t0])) {
            tcg_out_gadget(s, gadget_movi_imm_i64[t0][arg]);
            return;
        }
    }

    // TODO: optimize the negative case, too?

    // Less optimal case: emit the mov and its immediate.
    tcg_out_unary_gadget(s, gadget_movi_i64, t0);
    tcg_out64(s, arg);
}


/**
 * Generate an immediate-to-register MOV.
 */
static void tcg_out_movi(TCGContext *s, TCGType type, TCGReg t0, tcg_target_long arg)
{
    if (type == TCG_TYPE_I32) {
        tcg_out_movi_i32(s, t0, arg);
    } else {
        tcg_out_movi_i64(s, t0, arg);
    }
}

static void tcg_out_ext8s(TCGContext *s, TCGType type, TCGReg rd, TCGReg rs)
{
    switch (type) {
    case TCG_TYPE_I32:
        tcg_debug_assert(TCG_TARGET_HAS_ext8s_i32);
        tcg_out_binary_gadget(s, gadget_ext8s_i32, rd, rs);
        break;
#if TCG_TARGET_REG_BITS == 64
    case TCG_TYPE_I64:
        tcg_debug_assert(TCG_TARGET_HAS_ext8s_i64);
        tcg_out_binary_gadget(s, gadget_ext8s_i64, rd, rs);
        break;
#endif
    default:
        g_assert_not_reached();
    }
}

static void tcg_out_ext8u(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_out_binary_gadget(s, gadget_ext8u, rd, rs);
}

static void tcg_out_ext16s(TCGContext *s, TCGType type, TCGReg rd, TCGReg rs)
{
    switch (type) {
    case TCG_TYPE_I32:
        tcg_debug_assert(TCG_TARGET_HAS_ext16s_i32);
        tcg_out_binary_gadget(s, gadget_ext16s_i32, rd, rs);
        break;
#if TCG_TARGET_REG_BITS == 64
    case TCG_TYPE_I64:
        tcg_debug_assert(TCG_TARGET_HAS_ext16s_i64);
        tcg_out_binary_gadget(s, gadget_ext16s_i64, rd, rs);
        break;
#endif
    default:
        g_assert_not_reached();
    }
}

static void tcg_out_ext16u(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_out_binary_gadget(s, gadget_ext16u, rd, rs);
}

static void tcg_out_ext32s(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_debug_assert(TCG_TARGET_REG_BITS == 64);
    tcg_debug_assert(TCG_TARGET_HAS_ext32s_i64);
    tcg_out_binary_gadget(s, gadget_ext32s_i64, rd, rs);
}

static void tcg_out_ext32u(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_debug_assert(TCG_TARGET_REG_BITS == 64);
    tcg_debug_assert(TCG_TARGET_HAS_ext32u_i64);
    tcg_out_binary_gadget(s, gadget_ext32u_i64, rd, rs);
}

static void tcg_out_exts_i32_i64(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_out_ext32s(s, rd, rs);
}

static void tcg_out_extu_i32_i64(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_out_ext32u(s, rd, rs);
}

static void tcg_out_extrl_i64_i32(TCGContext *s, TCGReg rd, TCGReg rs)
{
    tcg_debug_assert(TCG_TARGET_REG_BITS == 64);
    tcg_out_binary_gadget(s, gadget_extrl, rd, rs);
}

static bool tcg_out_xchg(TCGContext *s, TCGType type, TCGReg r1, TCGReg r2)
{
    return false;
}

static void tcg_out_addi_ptr(TCGContext *s, TCGReg rd, TCGReg rs,
                              tcg_target_long imm)
{
    /* This function is only used for passing structs by reference. */
    g_assert_not_reached();
}

/**
 * Generate a CALL.
 */
static void tcg_out_call(TCGContext *s, const tcg_insn_unit *func,
                         const TCGHelperInfo *info)
{
    tcg_out_gadget(s, gadget_call);
    tcg_out64(s, (uintptr_t)func);
}

/**
 * Generates LD instructions.
 */
static void tcg_out_ld(TCGContext *s, TCGType type, TCGReg ret, TCGReg arg1,
                       intptr_t arg2)
{

    if (type == TCG_TYPE_I32) {
        tcg_out_ldst_gadget(s, gadget_ld32u, ret, arg1, arg2);
    } else {
        tcg_out_ldst_gadget(s, gadget_ld_i64, ret, arg1, arg2);
    }
}

static void tcg_out_exit_tb(TCGContext *s, uintptr_t arg)
{
    // Emit a simple gadget with a known return code.
    tcg_out_imm64_gadget(s, gadget_exit_tb, arg);
}

static void tcg_out_goto_tb(TCGContext *s, int which)
{
    // If we're using a direct jump, we'll emit a "relocation" that can be usd
    // to patch our gadget stream with the target address, later.

    // Emit our gadget.
    tcg_out_gadget(s, gadget_br);

    // Place our current instruction into our "relocation table", so it can
    // be patched once we know where the branch will target...
    s->gen_tb->jmp_insn_offset[which] = tcg_current_code_size(s);

    // ... and emit our relocation.
    tcg_out64(s, which);

    set_jmp_reset_offset(s, which);
}

/* We expect to use a 7-bit scaled negative offset from ENV.  */
#define MIN_TLB_MASK_TABLE_OFS  -512

/**
 * Generate every other operation.
 */
static void tcg_out_op(TCGContext *s, TCGOpcode opc, TCGType type,
                       const TCGArg args[TCG_MAX_OP_ARGS],
                       const int const_args[TCG_MAX_OP_ARGS])
{
    switch (opc) {

    // Simple branch.
    case INDEX_op_br:
        tcg_out_gadget(s, gadget_br);
        tcti_out_label(s, arg_label(args[0]));
        break;


    // Set condition flag.
    // a0 = Rd, a1 = Rn, a2 = Rm
    case INDEX_op_setcond_i32:
    {
        void *gadget;

        // We have to emit a different gadget per condition; we'll select which.
        switch(args[3]) {
            case TCG_COND_EQ:  gadget = gadget_setcond_i32_eq; break;
            case TCG_COND_NE:  gadget = gadget_setcond_i32_ne; break;
            case TCG_COND_LT:  gadget = gadget_setcond_i32_lt; break;
            case TCG_COND_GE:  gadget = gadget_setcond_i32_ge; break;
            case TCG_COND_LE:  gadget = gadget_setcond_i32_le; break;
            case TCG_COND_GT:  gadget = gadget_setcond_i32_gt; break;
            case TCG_COND_LTU: gadget = gadget_setcond_i32_lo; break;
            case TCG_COND_GEU: gadget = gadget_setcond_i32_hs; break;
            case TCG_COND_LEU: gadget = gadget_setcond_i32_ls; break;
            case TCG_COND_GTU: gadget = gadget_setcond_i32_hi; break;
            default:
                g_assert_not_reached();
        }

        tcg_out_ternary_gadget(s, gadget, args[0], args[1], args[2]);
        break;
    }

    case INDEX_op_setcond_i64:
    {
        void *gadget;

        // We have to emit a different gadget per condition; we'll select which.
        switch(args[3]) {
            case TCG_COND_EQ:  gadget = gadget_setcond_i64_eq; break;
            case TCG_COND_NE:  gadget = gadget_setcond_i64_ne; break;
            case TCG_COND_LT:  gadget = gadget_setcond_i64_lt; break;
            case TCG_COND_GE:  gadget = gadget_setcond_i64_ge; break;
            case TCG_COND_LE:  gadget = gadget_setcond_i64_le; break;
            case TCG_COND_GT:  gadget = gadget_setcond_i64_gt; break;
            case TCG_COND_LTU: gadget = gadget_setcond_i64_lo; break;
            case TCG_COND_GEU: gadget = gadget_setcond_i64_hs; break;
            case TCG_COND_LEU: gadget = gadget_setcond_i64_ls; break;
            case TCG_COND_GTU: gadget = gadget_setcond_i64_hi; break;
            default:
                g_assert_not_reached();
        }

        tcg_out_ternary_gadget(s, gadget, args[0], args[1], args[2]);
        break;
    }

    /**
     * Load instructions.
     */

    case INDEX_op_ld8u_i32:
    case INDEX_op_ld8u_i64:
        tcg_out_ldst_gadget(s, gadget_ld8u, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld8s_i32:
        tcg_out_ldst_gadget(s, gadget_ld8s_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld8s_i64:
        tcg_out_ldst_gadget(s, gadget_ld8s_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld16u_i32:
    case INDEX_op_ld16u_i64:
        tcg_out_ldst_gadget(s, gadget_ld16u, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld16s_i32:
        tcg_out_ldst_gadget(s, gadget_ld16s_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld16s_i64:
        tcg_out_ldst_gadget(s, gadget_ld16s_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld_i32:
    case INDEX_op_ld32u_i64:
        tcg_out_ldst_gadget(s, gadget_ld32u, args[0], args[1], args[2]);
        break;

    case INDEX_op_ld_i64:
        tcg_out_ldst_gadget(s, gadget_ld_i64, args[0], args[1], args[2]);
        break;
   
    case INDEX_op_ld32s_i64:
        tcg_out_ldst_gadget(s, gadget_ld32s_i64, args[0], args[1], args[2]);
        break;


    /**
     * Store instructions.
     */
    case INDEX_op_st8_i32:
    case INDEX_op_st8_i64:
        tcg_out_ldst_gadget(s, gadget_st8, args[0], args[1], args[2]);
        break;

    case INDEX_op_st16_i32:
    case INDEX_op_st16_i64:
        tcg_out_ldst_gadget(s, gadget_st16, args[0], args[1], args[2]);
        break;

    case INDEX_op_st_i32:
    case INDEX_op_st32_i64:
        tcg_out_ldst_gadget(s, gadget_st_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_st_i64:
        tcg_out_ldst_gadget(s, gadget_st_i64, args[0], args[1], args[2]);
        break;

    /**
     * Arithmetic instructions.
     */

    case INDEX_op_add_i32:
        tcg_out_ternary_gadget(s, gadget_add_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_sub_i32:
        tcg_out_ternary_gadget(s, gadget_sub_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_mul_i32:
        tcg_out_ternary_gadget(s, gadget_mul_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_nand_i32:     /* Optional (TCG_TARGET_HAS_nand_i32). */
        tcg_out_ternary_gadget(s, gadget_nand_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_nor_i32:     /* Optional (TCG_TARGET_HAS_nor_i32). */
        tcg_out_ternary_gadget(s, gadget_nor_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_and_i32:
        tcg_out_ternary_gadget(s, gadget_and_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_andc_i32:     /* Optional (TCG_TARGET_HAS_andc_i32). */
        tcg_out_ternary_gadget(s, gadget_andc_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_orc_i32:      /* Optional (TCG_TARGET_HAS_orc_i64). */
        tcg_out_ternary_gadget(s, gadget_orc_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_eqv_i32:      /* Optional (TCG_TARGET_HAS_orc_i64). */
        tcg_out_ternary_gadget(s, gadget_eqv_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_or_i32:
        tcg_out_ternary_gadget(s, gadget_or_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_xor_i32:
        tcg_out_ternary_gadget(s, gadget_xor_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_shl_i32:
        tcg_out_ternary_gadget(s, gadget_shl_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_shr_i32:
        tcg_out_ternary_gadget(s, gadget_shr_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_sar_i32:
        tcg_out_ternary_gadget(s, gadget_sar_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_rotr_i32:     /* Optional (TCG_TARGET_HAS_rot_i32). */
        tcg_out_ternary_gadget(s, gadget_rotr_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_rotl_i32:     /* Optional (TCG_TARGET_HAS_rot_i32). */
        tcg_out_ternary_gadget(s, gadget_rotl_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_add_i64:
        tcg_out_ternary_gadget(s, gadget_add_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_sub_i64:
        tcg_out_ternary_gadget(s, gadget_sub_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_mul_i64:
        tcg_out_ternary_gadget(s, gadget_mul_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_and_i64:
        tcg_out_ternary_gadget(s, gadget_and_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_andc_i64:     /* Optional (TCG_TARGET_HAS_andc_i64). */
        tcg_out_ternary_gadget(s, gadget_andc_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_orc_i64:      /* Optional (TCG_TARGET_HAS_orc_i64). */
        tcg_out_ternary_gadget(s, gadget_orc_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_eqv_i64:      /* Optional (TCG_TARGET_HAS_eqv_i64). */
        tcg_out_ternary_gadget(s, gadget_eqv_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_nand_i64:     /* Optional (TCG_TARGET_HAS_nand_i64). */
        tcg_out_ternary_gadget(s, gadget_nand_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_nor_i64:      /* Optional (TCG_TARGET_HAS_nor_i64). */
        tcg_out_ternary_gadget(s, gadget_nor_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_or_i64:
        tcg_out_ternary_gadget(s, gadget_or_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_xor_i64:
        tcg_out_ternary_gadget(s, gadget_xor_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_shl_i64:
        tcg_out_ternary_gadget(s, gadget_shl_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_shr_i64:
        tcg_out_ternary_gadget(s, gadget_shr_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_sar_i64:
        tcg_out_ternary_gadget(s, gadget_sar_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_rotl_i64:     /* Optional (TCG_TARGET_HAS_rot_i64). */
        tcg_out_ternary_gadget(s, gadget_rotl_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_rotr_i64:     /* Optional (TCG_TARGET_HAS_rot_i64). */
        tcg_out_ternary_gadget(s, gadget_rotr_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_div_i64:      /* Optional (TCG_TARGET_HAS_div_i64). */
        tcg_out_ternary_gadget(s, gadget_div_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_divu_i64:     /* Optional (TCG_TARGET_HAS_div_i64). */
        tcg_out_ternary_gadget(s, gadget_divu_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_rem_i64:      /* Optional (TCG_TARGET_HAS_div_i64). */
        tcg_out_ternary_gadget(s, gadget_rem_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_remu_i64:     /* Optional (TCG_TARGET_HAS_div_i64). */
        tcg_out_ternary_gadget(s, gadget_remu_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_brcond_i64:
    {
        static uint8_t last_brcond_i64 = 0;
        void *gadget;

        // We have to emit a different gadget per condition; we'll select which.
        switch(args[2]) {
            case TCG_COND_EQ:  gadget = gadget_brcond_i64_eq; break;
            case TCG_COND_NE:  gadget = gadget_brcond_i64_ne; break;
            case TCG_COND_LT:  gadget = gadget_brcond_i64_lt; break;
            case TCG_COND_GE:  gadget = gadget_brcond_i64_ge; break;
            case TCG_COND_LE:  gadget = gadget_brcond_i64_le; break;
            case TCG_COND_GT:  gadget = gadget_brcond_i64_gt; break;
            case TCG_COND_LTU: gadget = gadget_brcond_i64_lo; break;
            case TCG_COND_GEU: gadget = gadget_brcond_i64_hs; break;
            case TCG_COND_LEU: gadget = gadget_brcond_i64_ls; break;
            case TCG_COND_GTU: gadget = gadget_brcond_i64_hi; break;
            default:
                g_assert_not_reached();
        }

        // We'll select the which branch to used based on a cycling counter.
        // This means we'll pick one of 16 identical brconds. Spreading this out
        // helps the processor's branch prediction be less "squished", as not every
        // branch is going throuh the same instruction.
        tcg_out_ternary_gadget(s, gadget, last_brcond_i64, args[0], args[1]);
        last_brcond_i64 = (last_brcond_i64 + 1) % TCG_TARGET_GP_REGS;

        // Branch target immediate.
        tcti_out_label(s, arg_label(args[3]));
        break;
    }


    case INDEX_op_bswap16_i32:  /* Optional (TCG_TARGET_HAS_bswap16_i32). */
    case INDEX_op_bswap16_i64:  /* Optional (TCG_TARGET_HAS_bswap16_i64). */
        tcg_out_binary_gadget(s, gadget_bswap16, args[0], args[1]);
        break;

    case INDEX_op_bswap32_i32:  /* Optional (TCG_TARGET_HAS_bswap32_i32). */
    case INDEX_op_bswap32_i64:  /* Optional (TCG_TARGET_HAS_bswap32_i64). */
        tcg_out_binary_gadget(s, gadget_bswap32, args[0], args[1]);
        break;

    case INDEX_op_bswap64_i64:  /* Optional (TCG_TARGET_HAS_bswap64_i64). */
        tcg_out_binary_gadget(s, gadget_bswap64, args[0], args[1]);
        break;

    case INDEX_op_not_i64:      /* Optional (TCG_TARGET_HAS_not_i64). */
        tcg_out_binary_gadget(s, gadget_not_i64, args[0], args[1]);
        break;

    case INDEX_op_neg_i64:      /* Optional (TCG_TARGET_HAS_neg_i64). */
        tcg_out_binary_gadget(s, gadget_neg_i64, args[0], args[1]);
        break;

    case INDEX_op_clz_i64:      /* Optional (TCG_TARGET_HAS_clz_i64). */
        tcg_out_ternary_gadget(s, gadget_clz_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_ctz_i64:      /* Optional (TCG_TARGET_HAS_ctz_i64). */
        tcg_out_ternary_gadget(s, gadget_ctz_i64, args[0], args[1], args[2]);
        break;

    case INDEX_op_extrh_i64_i32:
        tcg_out_binary_gadget(s, gadget_extrh, args[0], args[1]);
        break;

    case INDEX_op_neg_i32:      /* Optional (TCG_TARGET_HAS_neg_i32). */
        tcg_out_binary_gadget(s, gadget_neg_i32, args[0], args[1]);
        break;

    case INDEX_op_clz_i32:      /* Optional (TCG_TARGET_HAS_clz_i32). */
        tcg_out_ternary_gadget(s, gadget_clz_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_ctz_i32:      /* Optional (TCG_TARGET_HAS_ctz_i32). */
        tcg_out_ternary_gadget(s, gadget_ctz_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_not_i32:      /* Optional (TCG_TARGET_HAS_not_i32). */
        tcg_out_binary_gadget(s, gadget_not_i32, args[0], args[1]);
        break;

    case INDEX_op_div_i32:      /* Optional (TCG_TARGET_HAS_div_i32). */
        tcg_out_ternary_gadget(s, gadget_div_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_divu_i32:     /* Optional (TCG_TARGET_HAS_div_i32). */
        tcg_out_ternary_gadget(s, gadget_divu_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_rem_i32:      /* Optional (TCG_TARGET_HAS_div_i32). */
        tcg_out_ternary_gadget(s, gadget_rem_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_remu_i32:     /* Optional (TCG_TARGET_HAS_div_i32). */
        tcg_out_ternary_gadget(s, gadget_remu_i32, args[0], args[1], args[2]);
        break;

    case INDEX_op_brcond_i32:
    {
        static uint8_t last_brcond_i32 = 0;
        void *gadget;

        // We have to emit a different gadget per condition; we'll select which.
        switch(args[2]) {
            case TCG_COND_EQ:  gadget = gadget_brcond_i32_eq; break;
            case TCG_COND_NE:  gadget = gadget_brcond_i32_ne; break;
            case TCG_COND_LT:  gadget = gadget_brcond_i32_lt; break;
            case TCG_COND_GE:  gadget = gadget_brcond_i32_ge; break;
            case TCG_COND_LE:  gadget = gadget_brcond_i32_le; break;
            case TCG_COND_GT:  gadget = gadget_brcond_i32_gt; break;
            case TCG_COND_LTU: gadget = gadget_brcond_i32_lo; break;
            case TCG_COND_GEU: gadget = gadget_brcond_i32_hs; break;
            case TCG_COND_LEU: gadget = gadget_brcond_i32_ls; break;
            case TCG_COND_GTU: gadget = gadget_brcond_i32_hi; break;
            default:
                g_assert_not_reached();
        }

        // We'll select the which branch to used based on a cycling counter.
        // This means we'll pick one of 16 identical brconds. Spreading this out
        // helps the processor's branch prediction be less "squished", as not every
        // branch is going throuh the same instruction.
        tcg_out_ternary_gadget(s, gadget, last_brcond_i32, args[0], args[1]);
        last_brcond_i32 = (last_brcond_i32 + 1) % TCG_TARGET_GP_REGS;

        // Branch target immediate.
        tcti_out_label(s, arg_label(args[3]));

        break;
    }

    case INDEX_op_qemu_ld_i32:
    {
        MemOp opc = get_memop(args[2]);
        unsigned a_bits = memop_alignment_bits(opc);
        unsigned s_bits = opc & MO_SIZE;

        void *gadget;

        switch(tlb_mask_table_ofs(s, get_mmuidx(args[2]))) {
            case -32:  LD_MEMOP_HANDLER(gadget, args[2],  off32_i32, a_bits, s_bits); break;
            case -48:  LD_MEMOP_HANDLER(gadget, args[2],  off48_i32, a_bits, s_bits); break;
            case -64:  LD_MEMOP_HANDLER(gadget, args[2],  off64_i32, a_bits, s_bits); break;
            case -96:  LD_MEMOP_HANDLER(gadget, args[2],  off96_i32, a_bits, s_bits); break;
            case -128: LD_MEMOP_HANDLER(gadget, args[2], off128_i32, a_bits, s_bits); break;
            default:   LD_MEMOP_LOOKUP(gadget, args[2], slowpath_off0_i32); break;
        }

        // Args:
        // - an immediate32 encodes our operation index
        tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        tcg_out64(s, args[2]); // TODO: fix encoding to be 4b
        break;
    }

    case INDEX_op_qemu_ld_i64:
    {
        MemOp opc = get_memop(args[2]);
        unsigned a_bits = memop_alignment_bits(opc);
        unsigned s_bits = opc & MO_SIZE;

        void *gadget;

        // Special optimization case: if we have an common case.
        // Delegate to our special-case handler.
        if (args[2] == 0x02) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], ld_ub, mode02)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        } else if (args[2] == 0x32) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], ld_leq, mode32)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        } else if(args[2] == 0x3a) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], ld_leq, mode3a)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        }
        // Otherwise, handle the generic case.
        else {
            switch(tlb_mask_table_ofs(s, get_mmuidx(args[2]))) {
                case -32:  LD_MEMOP_HANDLER(gadget, args[2],  off32_i64, a_bits, s_bits); break;
                case -48:  LD_MEMOP_HANDLER(gadget, args[2],  off48_i64, a_bits, s_bits); break;
                case -64:  LD_MEMOP_HANDLER(gadget, args[2],  off64_i64, a_bits, s_bits); break;
                case -96:  LD_MEMOP_HANDLER(gadget, args[2],  off96_i64, a_bits, s_bits); break;
                case -128: LD_MEMOP_HANDLER(gadget, args[2], off128_i64, a_bits, s_bits); break;
                default:   LD_MEMOP_LOOKUP(gadget, args[2], slowpath_off0_i64); break;
            }

            // Args:
            // - an immediate32 encodes our operation index
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
            tcg_out64(s, args[2]); // TODO: fix encoding to be 4b
        }

        break;
    }

    case INDEX_op_qemu_st_i32:
    {
        MemOp opc = get_memop(args[2]);
        unsigned a_bits = memop_alignment_bits(opc);
        unsigned s_bits = opc & MO_SIZE;

        void *gadget;

        switch(tlb_mask_table_ofs(s, get_mmuidx(args[2]))) {
            case -32:  ST_MEMOP_HANDLER(gadget, args[2],  off32_i32, a_bits, s_bits); break;
            case -48:  ST_MEMOP_HANDLER(gadget, args[2],  off48_i32, a_bits, s_bits); break;
            case -64:  ST_MEMOP_HANDLER(gadget, args[2],  off64_i32, a_bits, s_bits); break;
            case -96:  ST_MEMOP_HANDLER(gadget, args[2],  off96_i32, a_bits, s_bits); break;
            case -128: ST_MEMOP_HANDLER(gadget, args[2], off128_i32, a_bits, s_bits); break;
            default:   ST_MEMOP_LOOKUP(gadget, args[2], slowpath_off0_i32); break;
        }

        // Args:
        // - our gadget encodes the target and address registers
        // - an immediate32 encodes our operation index
        tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        tcg_out64(s, args[2]); // FIXME: double encoded
        break;
    }

    case INDEX_op_qemu_st_i64:
    {
        MemOp opc = get_memop(args[2]);
        unsigned a_bits = memop_alignment_bits(opc);
        unsigned s_bits = opc & MO_SIZE;

        void *gadget;

        // Special optimization case: if we have an common case.
        // Delegate to our special-case handler.
        if (args[2] == 0x02) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], st_ub, mode02)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        } else if (args[2] == 0x32) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], st_leq, mode32)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        } else if(args[2] == 0x3a) {
            LOOKUP_SPECIAL_CASE_LDST_GADGET(args[2], st_leq, mode3a)
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
        }
        // Otherwise, handle the generic case.
        else {
            switch(tlb_mask_table_ofs(s, get_mmuidx(args[2]))) {
                case -32:  ST_MEMOP_HANDLER(gadget, args[2],  off32_i64, a_bits, s_bits); break;
                case -48:  ST_MEMOP_HANDLER(gadget, args[2],  off48_i64, a_bits, s_bits); break;
                case -64:  ST_MEMOP_HANDLER(gadget, args[2],  off64_i64, a_bits, s_bits); break;
                case -96:  ST_MEMOP_HANDLER(gadget, args[2],  off96_i64, a_bits, s_bits); break;
                case -128: ST_MEMOP_HANDLER(gadget, args[2], off128_i64, a_bits, s_bits); break;
                default:   ST_MEMOP_LOOKUP(gadget, args[2], slowpath_off0_i64); break;
            }

            // Args:
            // - our gadget encodes the target and address registers
            // - an immediate32 encodes our operation index
            tcg_out_binary_gadget(s, gadget, args[0], args[1]);
            tcg_out64(s, args[2]); // FIXME: double encoded
        }

        break;
    }

    // Memory barriers.
    case INDEX_op_mb:
    {
        static void* sync[] = {
            [0 ... TCG_MO_ALL]            = gadget_mb_all,
            [TCG_MO_ST_ST]                = gadget_mb_st,
            [TCG_MO_LD_LD]                = gadget_mb_ld,
            [TCG_MO_LD_ST]                = gadget_mb_ld,
            [TCG_MO_LD_ST | TCG_MO_LD_LD] = gadget_mb_ld,
        };
        tcg_out_gadget(s, sync[args[0] & TCG_MO_ALL]);

        break;
    }

    case INDEX_op_mov_i32:  /* Always emitted via tcg_out_mov.  */
    case INDEX_op_mov_i64:
    case INDEX_op_call:     /* Always emitted via tcg_out_call.  */
    case INDEX_op_exit_tb:  /* Always emitted via tcg_out_exit_tb.  */
    case INDEX_op_goto_tb:  /* Always emitted via tcg_out_goto_tb.  */
    case INDEX_op_ext8s_i32:    /* Always emitted via tcg_reg_alloc_op. */
    case INDEX_op_ext8s_i64:
    case INDEX_op_ext8u_i32:
    case INDEX_op_ext8u_i64:
    case INDEX_op_ext16s_i32:
    case INDEX_op_ext16s_i64:
    case INDEX_op_ext16u_i32:
    case INDEX_op_ext16u_i64:
    case INDEX_op_ext32s_i64:
    case INDEX_op_ext32u_i64:
    case INDEX_op_ext_i32_i64:
    case INDEX_op_extu_i32_i64:
    case INDEX_op_extrl_i64_i32:
    default:
        g_assert_not_reached();
    }
}

/**
 * Generate immediate stores.
 */
static void tcg_out_st(TCGContext *s, TCGType type, TCGReg arg, TCGReg arg1,
                       intptr_t arg2)
{
    if (type == TCG_TYPE_I32) {
        tcg_out_ldst_gadget(s, gadget_st_i32, arg, arg1, arg2);
    } else {
        tcg_out_ldst_gadget(s, gadget_st_i64, arg, arg1, arg2);
    }
}

static inline bool tcg_out_sti(TCGContext *s, TCGType type, TCGArg val,
                               TCGReg base, intptr_t ofs)
{
    return false;
}

/* Test if a constant matches the constraint. */
static bool tcg_target_const_match(int64_t val, int ct,
                                   TCGType type, TCGCond cond, int vece)
{
    return ct & TCG_CT_CONST;
}

static void tcg_out_nop_fill(tcg_insn_unit *p, int count)
{
    memset(p, 0, sizeof(*p) * count);
}

/***************************
 *  TCG Vector Operations  *
 ***************************/

//
// Helper for emitting DUPI (immediate DUP) instructions.
//
#define tcg_out_dupi_gadget(s, name, q, rd, op, cmode, arg) \
    if (q) { \
        tcg_out_gadget(s, gadget_ ## name ## _cmode_ ## cmode ## _op ## op ## _q1[rd][arg]); \
    } else { \
        tcg_out_gadget(s, gadget_ ## name ## _cmode_ ## cmode ## _op ## op ## _q0[rd][arg]); \
    }


//
// Helpers for emitting D/Q variant instructions.
//
#define tcg_out_dq_gadget(s, name, arity, is_q, args...) \
    if (is_q) { \
        tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _q, args); \
    } else { \
        tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _d, args); \
    }

#define tcg_out_unary_dq_gadget(s, name, is_q, a) \
    tcg_out_dq_gadget(s, name, unary, is_q, a) 
#define tcg_out_binary_dq_gadget(s, name, is_q, a, b) \
    tcg_out_dq_gadget(s, name, binary, is_q, a, b)
#define tcg_out_ternary_dq_gadget(s, name, is_q, a, b, c) \
    tcg_out_dq_gadget(s, name, ternary, is_q, a, b, c)


//
// Helper for emitting the gadget appropriate for a vector's size.
//
#define tcg_out_sized_vector_gadget(s, name, arity, vece, args...) \
    switch(vece) { \
        case MO_8: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _8b, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _16b, args); \
            } \
            break; \
        case MO_16: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _4h, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _8h, args); \
            } \
            break; \
        case MO_32: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _2s, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _4s, args); \
            } \
            break; \
        case MO_64: \
            if (type == TCG_TYPE_V128) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _2d, args); \
            } \
            else { \
                g_assert_not_reached(); \
            } \
            break;  \
        default: \
            g_assert_not_reached(); \
    } 
#define tcg_out_sized_vector_gadget_no64(s, name, arity, vece, args...) \
    switch(vece) { \
        case MO_8: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _8b, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _16b, args); \
            } \
            break; \
        case MO_16: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _4h, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _8h, args); \
            } \
            break; \
        case MO_32: \
            if (type == TCG_TYPE_V64) { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _2s, args); \
            } else { \
                tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _4s, args); \
            } \
            break; \
        default: \
            g_assert_not_reached(); \
    } 


#define tcg_out_unary_vector_gadget(s, name, vece, a) \
    tcg_out_sized_vector_gadget(s, name, unary, vece, a)
#define tcg_out_binary_vector_gadget(s, name, vece, a, b) \
    tcg_out_sized_vector_gadget(s, name, binary, vece, a, b)
#define tcg_out_ternary_vector_gadget(s, name, vece, a, b, c) \
    tcg_out_sized_vector_gadget(s, name, ternary, vece, a, b, c)

#define tcg_out_ternary_vector_gadget_no64(s, name, vece, a, b, c) \
    tcg_out_sized_vector_gadget_no64(s, name, ternary, vece, a, b, c)


#define tcg_out_sized_gadget_with_scalar(s, name, arity, is_scalar, vece, args...) \
    if (is_scalar) { \
        tcg_out_ ## arity ## _gadget(s, gadget_ ## name ## _scalar, args); \
    } else { \
        tcg_out_sized_vector_gadget(s, name, arity, vece, args); \
    }

#define tcg_out_ternary_vector_gadget_with_scalar(s, name, is_scalar, vece, a, b, c) \
    tcg_out_sized_gadget_with_scalar(s, name, ternary, is_scalar, vece, a, b, c)

#define tcg_out_ternary_immediate_vector_gadget_with_scalar(s, name, is_scalar, vece, a, b, c) \
    tcg_out_sized_gadget_with_scalar(s, name, ternary_immediate, is_scalar, vece, a, b, c)

/* Return true if v16 is a valid 16-bit shifted immediate.  */
static bool is_shimm16(uint16_t v16, int *cmode, int *imm8)
{
    if (v16 == (v16 & 0xff)) {
        *cmode = 0x8;
        *imm8 = v16 & 0xff;
        return true;
    } else if (v16 == (v16 & 0xff00)) {
        *cmode = 0xa;
        *imm8 = v16 >> 8;
        return true;
    }
    return false;
}


/** Core vector operation emission. */
static void tcg_out_vec_op(TCGContext *s, TCGOpcode opc, unsigned vecl, unsigned vece,
    const TCGArg args[TCG_MAX_OP_ARGS], const int const_args[TCG_MAX_OP_ARGS])
{
    TCGType type = vecl + TCG_TYPE_V64;
    TCGArg r0, r1, r2, r3, w0, w1, w2, w3;

    // Typing flags for vector operations.
    bool is_v128 = (type == TCG_TYPE_V128);
    bool is_scalar = !is_v128 && (vece == MO_64);

    // Argument shortcuts.
    r0 = args[0];
    r1 = args[1];
    r2 = args[2];
    r3 = args[3];

    // Offset argument shortcuts; offset to convert register numbers to gadget numberes.
    w0 = args[0] - TCG_REG_V16;
    w1 = args[1] - TCG_REG_V16;
    w2 = args[2] - TCG_REG_V16;
    w3 = args[3] - TCG_REG_V16;

    // Argument shortcuts, as signed.
    int64_t signed_offset_arg = (int32_t)args[2];

    switch (opc) {

    // Load memory -> vector: followed by a 64-bit offset immediate
    case INDEX_op_ld_vec:
        tcg_out_binary_dq_gadget(s, ldr, is_v128, w0, r1);
        tcg_out64(s, signed_offset_arg);
        break;
    
    // Store memory -> vector: followed by a 64-bit offset immediate
    case INDEX_op_st_vec:
        tcg_out_binary_dq_gadget(s, str, is_v128, w0, r1);
        tcg_out64(s, signed_offset_arg);
        break;

    // Duplciate memory to all vector elements.
    case INDEX_op_dupm_vec:
        // DUPM handles normalization itself; pass arguments raw.
        tcg_out_dupm_vec(s, type, vece, r0, r1, r2);
        break;

    case INDEX_op_add_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, add, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_sub_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, sub, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_mul_vec: // optional
        tcg_out_ternary_vector_gadget_no64(s, mul, vece, w0, w1, w2);
        break;

    case INDEX_op_neg_vec: // optional
        tcg_out_binary_vector_gadget(s, neg, vece, w0, w1);
        break;

    case INDEX_op_abs_vec: // optional
        tcg_out_binary_vector_gadget(s, abs, vece, w0, w1);
        break;

    case INDEX_op_and_vec: // optional
        tcg_out_ternary_dq_gadget(s, and, is_v128, w0, w1, w2);
        break;

    case INDEX_op_or_vec:
        tcg_out_ternary_dq_gadget(s, or, is_v128, w0, w1, w2);
        break;

    case INDEX_op_andc_vec:
        tcg_out_ternary_dq_gadget(s, andc, is_v128, w0, w1, w2);
        break;

    case INDEX_op_orc_vec: // optional
        tcg_out_ternary_dq_gadget(s, orc, is_v128, w0, w1, w2);
        break;

    case INDEX_op_xor_vec:
        tcg_out_ternary_dq_gadget(s, xor, is_v128, w0, w1, w2);
        break;

    case INDEX_op_ssadd_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, ssadd, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_sssub_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, sssub, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_usadd_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, usadd, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_ussub_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, ussub, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_smax_vec:
        tcg_out_ternary_vector_gadget_no64(s, smax, vece, w0, w1, w2);
        break;

    case INDEX_op_smin_vec:
        tcg_out_ternary_vector_gadget_no64(s, smin, vece, w0, w1, w2);
        break;

    case INDEX_op_umax_vec:
        tcg_out_ternary_vector_gadget_no64(s, umax, vece, w0, w1, w2);
        break;

    case INDEX_op_umin_vec:
        tcg_out_ternary_vector_gadget_no64(s, umin, vece, w0, w1, w2);
        break;

    case INDEX_op_not_vec: // optional
        tcg_out_binary_dq_gadget(s, not, is_v128, w0, w1);
        break;

    case INDEX_op_shlv_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, shlv, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_aa64_sshl_vec:
        tcg_out_ternary_vector_gadget_with_scalar(s, sshl, is_scalar, vece, w0, w1, w2);
        break;

    case INDEX_op_cmp_vec:
        switch (args[3]) {
            case TCG_COND_EQ:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmeq, is_scalar, vece, w0, w1, w2);
                break;
            case TCG_COND_NE:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmeq, is_scalar, vece, w0, w1, w2);
                tcg_out_binary_dq_gadget(s, not, is_v128, w0, w0);
                break;
            case TCG_COND_GT:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmgt, is_scalar, vece, w0, w1, w2);
                break;
            case TCG_COND_LE:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmge, is_scalar, vece, w0, w2, w1);
                break;
            case TCG_COND_GE:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmge, is_scalar, vece, w0, w1, w2);
                break;
            case TCG_COND_LT:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmgt, is_scalar, vece, w0, w2, w1);
                break;
            case TCG_COND_GTU:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmhi, is_scalar, vece, w0, w1, w2);
                break;
            case TCG_COND_LEU:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmhs, is_scalar, vece, w0, w2, w1);
                break;
            case TCG_COND_GEU:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmhs, is_scalar, vece, w0, w1, w2);
                break;
            case TCG_COND_LTU:
                tcg_out_ternary_vector_gadget_with_scalar(s, cmhi, is_scalar, vece, w0, w2, w1);
                break;
            default:
                g_assert_not_reached();
        }
        break;

    case INDEX_op_bitsel_vec: // optional
    {
        if (r0 == r3) {
            tcg_out_ternary_dq_gadget(s, bit, is_v128, w0, w2, w1);
        } else if (r0 == r2) {
            tcg_out_ternary_dq_gadget(s, bif, is_v128, w0, w3, w1);
        } else {
            if (r0 != r1) {
                tcg_out_mov(s, type, r0, r1);
            }
            tcg_out_ternary_dq_gadget(s, bsl, is_v128, w0, w2, w3);
        }
        break;
    }

    /* inhibit compiler warning because we use imm as a register */
    case INDEX_op_shli_vec:
        tcg_out_ternary_immediate_vector_gadget_with_scalar(s, shl, is_scalar, vece, w0, w1, r2);
        break;
    case INDEX_op_shri_vec:
        tcg_out_ternary_immediate_vector_gadget_with_scalar(s, ushr, is_scalar, vece, w0, w1, r2 - 1);
        break;
    case INDEX_op_sari_vec:
        tcg_out_ternary_immediate_vector_gadget_with_scalar(s, sshr, is_scalar, vece, w0, w1, r2 - 1);
        break;
    case INDEX_op_aa64_sli_vec:
        tcg_out_ternary_immediate_vector_gadget_with_scalar(s, sli, is_scalar, vece, w0, w2, r3);
        break;

    case INDEX_op_mov_vec:  /* Always emitted via tcg_out_mov.  */
    case INDEX_op_dup_vec:  /* Always emitted via tcg_out_dup_vec.  */
    default:
        g_assert_not_reached();
    }
}


int tcg_can_emit_vec_op(TCGOpcode opc, TCGType type, unsigned vece)
{
    switch (opc) {
    case INDEX_op_add_vec:
    case INDEX_op_sub_vec:
    case INDEX_op_and_vec:
    case INDEX_op_or_vec:
    case INDEX_op_xor_vec:
    case INDEX_op_andc_vec:
    case INDEX_op_orc_vec:
    case INDEX_op_neg_vec:
    case INDEX_op_abs_vec:
    case INDEX_op_not_vec:
    case INDEX_op_cmp_vec:
    case INDEX_op_shli_vec:
    case INDEX_op_shri_vec:
    case INDEX_op_sari_vec:
    case INDEX_op_ssadd_vec:
    case INDEX_op_sssub_vec:
    case INDEX_op_usadd_vec:
    case INDEX_op_ussub_vec:
    case INDEX_op_shlv_vec:
    case INDEX_op_bitsel_vec:
        return 1;
    case INDEX_op_rotli_vec:
    case INDEX_op_shrv_vec:
    case INDEX_op_sarv_vec:
    case INDEX_op_rotlv_vec:
    case INDEX_op_rotrv_vec:
        return -1;
    case INDEX_op_mul_vec:
    case INDEX_op_smax_vec:
    case INDEX_op_smin_vec:
    case INDEX_op_umax_vec:
    case INDEX_op_umin_vec:
        return vece < MO_64;

    default:
        return 0;
    }
}

void tcg_expand_vec_op(TCGOpcode opc, TCGType type, unsigned vece,
                       TCGArg a0, ...)
{
    va_list va;
    TCGv_vec v0, v1, v2, t1, t2, c1;
    TCGArg a2;


    va_start(va, a0);
    v0 = temp_tcgv_vec(arg_temp(a0));
    v1 = temp_tcgv_vec(arg_temp(va_arg(va, TCGArg)));
    a2 = va_arg(va, TCGArg);
    va_end(va);

    switch (opc) {
    case INDEX_op_rotli_vec:
        t1 = tcg_temp_new_vec(type);
        tcg_gen_shri_vec(vece, t1, v1, -a2 & ((8 << vece) - 1));
        vec_gen_4(INDEX_op_aa64_sli_vec, type, vece,
                  tcgv_vec_arg(v0), tcgv_vec_arg(t1), tcgv_vec_arg(v1), a2);
        tcg_temp_free_vec(t1);
        break;

    case INDEX_op_shrv_vec:
    case INDEX_op_sarv_vec:
        /* Right shifts are negative left shifts for AArch64.  */
        v2 = temp_tcgv_vec(arg_temp(a2));
        t1 = tcg_temp_new_vec(type);
        tcg_gen_neg_vec(vece, t1, v2);
        opc = (opc == INDEX_op_shrv_vec
               ? INDEX_op_shlv_vec : INDEX_op_aa64_sshl_vec);
        vec_gen_3(opc, type, vece, tcgv_vec_arg(v0),
                  tcgv_vec_arg(v1), tcgv_vec_arg(t1));
        tcg_temp_free_vec(t1);
        break;

    case INDEX_op_rotlv_vec:
        v2 = temp_tcgv_vec(arg_temp(a2));
        t1 = tcg_temp_new_vec(type);
        c1 = tcg_constant_vec(type, vece, 8 << vece);
        tcg_gen_sub_vec(vece, t1, v2, c1);
        /* Right shifts are negative left shifts for AArch64.  */
        vec_gen_3(INDEX_op_shlv_vec, type, vece, tcgv_vec_arg(t1),
                  tcgv_vec_arg(v1), tcgv_vec_arg(t1));
        vec_gen_3(INDEX_op_shlv_vec, type, vece, tcgv_vec_arg(v0),
                  tcgv_vec_arg(v1), tcgv_vec_arg(v2));
        tcg_gen_or_vec(vece, v0, v0, t1);
        tcg_temp_free_vec(t1);
        break;

    case INDEX_op_rotrv_vec:
        v2 = temp_tcgv_vec(arg_temp(a2));
        t1 = tcg_temp_new_vec(type);
        t2 = tcg_temp_new_vec(type);
        c1 = tcg_constant_vec(type, vece, 8 << vece);
        tcg_gen_neg_vec(vece, t1, v2);
        tcg_gen_sub_vec(vece, t2, c1, v2);
        /* Right shifts are negative left shifts for AArch64.  */
        vec_gen_3(INDEX_op_shlv_vec, type, vece, tcgv_vec_arg(t1),
                  tcgv_vec_arg(v1), tcgv_vec_arg(t1));
        vec_gen_3(INDEX_op_shlv_vec, type, vece, tcgv_vec_arg(t2),
                  tcgv_vec_arg(v1), tcgv_vec_arg(t2));
        tcg_gen_or_vec(vece, v0, t1, t2);
        tcg_temp_free_vec(t1);
        tcg_temp_free_vec(t2);
        break;

    default:
        g_assert_not_reached();
    }
}


/* Generate DUPI (move immediate) vector ops. */
static bool tcg_out_optimized_dupi_vec(TCGContext *s, TCGType type, unsigned vece, TCGReg rd, int64_t v64)
{
    bool q = (type == TCG_TYPE_V128);
    int cmode, imm8, i;

    // If we're copying an 8b immediate, we implicitly have a simple gadget for this,
    // since there are only 256 possible values * 16 registers. Emit a MOVI gadget implicitly.
    if (vece == MO_8) {
        imm8 = (uint8_t)v64;
        tcg_out_dupi_gadget(s, movi, q, rd, 0, e, imm8);
        return true;
    }

    // Otherwise, if we have a value that's all 0x00 and 0xFF bytes,
    // we can use the scalar variant of MOVI (op=1, cmode=e), which handles
    // that case directly.
    for (i = imm8 = 0; i < 8; i++) {
        uint8_t byte = v64 >> (i * 8);
        if (byte == 0xff) {
            imm8 |= 1 << i;
        } else if (byte != 0) {
            goto fail_bytes;
        }
    }
    tcg_out_dupi_gadget(s, movi, q, rd, 1, e, imm8);
    return true;
 fail_bytes:

    // Handle 16B moves.
    if (vece == MO_16) {
        uint16_t v16 = v64;

        // Check to see if we have a value representable in as a MOV imm8, possibly via a shift.
        if (is_shimm16(v16, &cmode, &imm8)) {
            // Output the corret instruction CMode for either a regular MOVI (8) or a LSL8 MOVI (a).
            if (cmode == 0x8) {
                tcg_out_dupi_gadget(s, movi, q, rd, 0, 8, imm8);
            } else {
                tcg_out_dupi_gadget(s, movi, q, rd, 0, a, imm8);
            }
            return true;
        }

        // Check to see if we have a value representable in as an inverted MOV imm8, possibly via a shift.
        if (is_shimm16(~v16, &cmode, &imm8)) {
            // Output the corret instruction CMode for either a regular MOVI (8) or a LSL8 MOVI (a).
            if (cmode == 0x8) {
                tcg_out_dupi_gadget(s, mvni, q, rd, 0, 8, imm8);
            } else {
                tcg_out_dupi_gadget(s, mvni, q, rd, 0, a, imm8);
            }
            return true;
        }

        // If we can't perform either of the optimizations, we'll need to do this in two steps.
        // Normally, we'd emit a gadget for both steps, but in this case that'd result in needing -way-
        // too many gadgets. We'll emit two, instead.
        tcg_out_dupi_gadget(s, movi, q, rd, 0, 8, v16 & 0xff);
        tcg_out_dupi_gadget(s, orr,  q, rd, 0, a, v16 >> 8);
        return true;
    }

    // FIXME: implement 32B move optimizations

     
    // Try to create optimized 32B moves.
    //else if (vece == MO_32) {
    //    uint32_t v32 = v64;
    //    uint32_t n32 = ~v32;

    //    if (is_shimm32(v32, &cmode, &imm8) ||
    //        is_soimm32(v32, &cmode, &imm8) ||
    //        is_fimm32(v32, &cmode, &imm8)) {
    //        tcg_out_insn(s, 3606, MOVI, q, rd, 0, cmode, imm8);
    //        return;
    //    }
    //    if (is_shimm32(n32, &cmode, &imm8) ||
    //        is_soimm32(n32, &cmode, &imm8)) {
    //        tcg_out_insn(s, 3606, MVNI, q, rd, 0, cmode, imm8);
    //        return;
    //    }

    //    //
    //    // Restrict the set of constants to those we can load with
    //    // two instructions.  Others we load from the pool.
    //    //
    //    i = is_shimm32_pair(v32, &cmode, &imm8);
    //    if (i) {
    //        tcg_out_insn(s, 3606, MOVI, q, rd, 0, cmode, imm8);
    //        tcg_out_insn(s, 3606, ORR, q, rd, 0, i, extract32(v32, i * 4, 8));
    //        return;
    //    }
    //    i = is_shimm32_pair(n32, &cmode, &imm8);
    //    if (i) {
    //        tcg_out_insn(s, 3606, MVNI, q, rd, 0, cmode, imm8);
    //        tcg_out_insn(s, 3606, BIC, q, rd, 0, i, extract32(n32, i * 4, 8));
    //        return;
    //    }
    //} 

    return false;
}


/* Emits instructions that can load an immediate into a vector. */
static void tcg_out_dupi_vec(TCGContext *s, TCGType type, unsigned vece, TCGReg rd, int64_t v64)
{
    // Convert Rd into a simple gadget number.
    rd = rd - (TCG_REG_V16);

    // First, try to create an optimized implementation, if possible.
    if (tcg_out_optimized_dupi_vec(s, type, vece, rd, v64)) {
        return;
    }

    // If we didn't, we'll need to load the full vector from memory.
    // Emit it into our bytecode stream as an immediate; which we'll then
    // load inside the gadget.
    if (type == TCG_TYPE_V128) {
        tcg_out_unary_gadget(s, gadget_ldi_q, rd);
        tcg_out64(s, v64);
        tcg_out64(s, v64);
    } else {
        tcg_out_unary_gadget(s, gadget_ldi_d, rd);
        tcg_out64(s, v64);
    }
}


/* Emits instructions that can load a register into a vector. */
static bool tcg_out_dup_vec(TCGContext *s, TCGType type, unsigned vece, TCGReg rd, TCGReg rs)
{
    // Compute the gadget index for the relevant vector register.
    TCGReg wd = rd - (TCG_REG_V16);

    // Emit a DUP gadget to handles the operation.
    tcg_out_binary_vector_gadget(s, dup, vece, wd, rs);
    return true;
}

static bool tcg_out_dupm_vec(TCGContext *s, TCGType type, unsigned vece, TCGReg r, TCGReg base, intptr_t offset)
{
    int64_t extended_offset = (int32_t)offset;

    // Convert the register into a simple register number for our gadgets.
    r = r - TCG_REG_V16;

    // Emit a DUPM gadget...
    tcg_out_binary_vector_gadget(s, dupm, vece, r, base);

    // ... and emit its int64 immediate offset.
    tcg_out64(s, extended_offset);

    return true;
}


/********************************
 *  TCG Runtime & Platform Def  *
 *******************************/

static void tcg_target_init(TCGContext *s)
{
    /* The current code uses uint8_t for tcg operations. */
    tcg_debug_assert(tcg_op_defs_max <= UINT8_MAX);

    // Registers available for each type of operation.
    tcg_target_available_regs[TCG_TYPE_I32]  = TCG_MASK_GP_REGISTERS;
    tcg_target_available_regs[TCG_TYPE_I64]  = TCG_MASK_GP_REGISTERS;
    tcg_target_available_regs[TCG_TYPE_V64]  = TCG_MASK_VECTOR_REGISTERS;
    tcg_target_available_regs[TCG_TYPE_V128] = TCG_MASK_VECTOR_REGISTERS;

    TCGReg unclobbered_registers[] = {
        // We don't use registers R16+ in our runtime, so we'll not bother protecting them.
        TCG_REG_R16, TCG_REG_R17, TCG_REG_R18, TCG_REG_R19,
        TCG_REG_R20, TCG_REG_R21, TCG_REG_R22, TCG_REG_R23,
        TCG_REG_R24, TCG_REG_R25, TCG_REG_R26, TCG_REG_R27,
        TCG_REG_R28, TCG_REG_R29, TCG_REG_R30, TCG_REG_R31,

        // Per our calling convention.
        TCG_REG_V8,  TCG_REG_V9,  TCG_REG_V10, TCG_REG_V11,
        TCG_REG_V12, TCG_REG_V13, TCG_REG_V14, TCG_REG_V15,
   };

    // Specify which registers are clobbered during call.
    tcg_target_call_clobber_regs = -1ull;
    for (unsigned i = 0; i < ARRAY_SIZE(unclobbered_registers); ++i) {
        tcg_regset_reset_reg(tcg_target_call_clobber_regs, unclobbered_registers[i]);
    }

    // Specify which local registers we're reserving.
    //
    // Note that we only have to specify registers that are used in the runtime,
    // and so not e.g. the register that contains AREG0, which can never be allocated.
    s->reserved_regs = 0;
    tcg_regset_set_reg(s->reserved_regs, TCG_REG_CALL_STACK);

    /* We use negative offsets from "sp" so that we can distinguish
       stores that might pretend to be call arguments.  */
    tcg_set_frame(s, TCG_REG_CALL_STACK, -CPU_TEMP_BUF_NLONGS * sizeof(long), CPU_TEMP_BUF_NLONGS * sizeof(long));
}

/* Generate global QEMU prologue and epilogue code. */
static inline void tcg_target_qemu_prologue(TCGContext *s)
{
    // No prologue; as we're interpreted.
}

static void tcg_out_tb_start(TCGContext *s)
{
    /* nothing to do */
}

bool tcg_target_has_memory_bswap(MemOp memop)
{
    return true;
}

/**
 * TCTI 'interpreter' bootstrap.
 */

// Store the current return address during helper calls.
__thread uintptr_t tcti_call_return_address;

/* Dispatch the bytecode stream contained in our translation buffer. */
uintptr_t QEMU_DISABLE_CFI tcg_qemu_tb_exec(CPUArchState *env, const void *v_tb_ptr)
{
    // Create our per-CPU temporary storage.
    long tcg_temps[CPU_TEMP_BUF_NLONGS];

    uint64_t return_value = 0;
    uintptr_t sp_value    = (uintptr_t)(tcg_temps + CPU_TEMP_BUF_NLONGS);
    uintptr_t pc_mirror   = (uintptr_t)&tcti_call_return_address;

    // Ensure our target configuration hasn't changed.
    tcti_assert(TCG_AREG0 == TCG_REG_R14);
    tcti_assert(TCG_REG_CALL_STACK == TCG_REG_R15);

    asm(
        // Our threaded-dispatch prologue needs to set up things for our machine to run.
        // This means:
        //   - Set up TCG_AREG0 (R14) to point to our architectural state.
        //   - Set up TCG_REG_CALL_STACK (R15) to point to our temporary buffer.
        //   - Point x28 (our bytecode "instruction pointer") to the relevant stream address.
        "ldr x14, %[areg0]\n"
        "ldr x15, %[sp_value]\n"
        "ldr x25, %[pc_mirror]\n"
        "ldr x28, %[start_tb_ptr]\n"

        // To start our code, we'll -call- the gadget at the first bytecode pointer.
        // Note that we call/branch-with-link, here; so our TB_EXIT gadget can RET in order
        // to return to this point when things are complete.
        "ldr x27, [x28], #8\n"
        "blr x27\n"

        // Finally, we'll copy out our final return value.
        "str x0, %[return_value]\n"

        : [return_value] "=m" (return_value)

        : [areg0]        "m"  (env),
          [sp_value]     "m"  (sp_value),
          [start_tb_ptr] "m"  (v_tb_ptr),
          [pc_mirror]    "m"  (pc_mirror)

        // We touch _every_ one of the lower registers, as we use these to execute directly.
        : "x0", "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
          "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",

        // We also use x26/x27 for temporary values, and x28 as our bytecode poitner.
        "x25", "x26", "x27", "x28", "cc", "memory"
    );

    return return_value;
}


/**
 *  Disassembly output support.
 */
#include <dlfcn.h>


/* Disassemble TCI bytecode. */
int print_insn_tcti(bfd_vma addr, disassemble_info *info)
{

#ifdef TCTI_GADGET_RICH_DISASSEMBLY
    Dl_info symbol_info = {};
    char symbol_name[48] ;
#endif

    int status;
    uint64_t block;

    // Read the relevant pointer.
    status = info->read_memory_func(addr, (void *)&block, sizeof(block), info);
    if (status != 0) {
        info->memory_error_func(status, addr, info);
        return -1;
    }

#ifdef TCTI_GADGET_RICH_DISASSEMBLY
    // Most of our disassembly stream will be gadgets. Try to get their names, for nice output.
    dladdr((void *)block, &symbol_info);

    if(symbol_info.dli_sname != 0) {
        strncpy(symbol_name, symbol_info.dli_sname, sizeof(symbol_name));
        symbol_name[sizeof(symbol_name) - 1] = 0;
        info->fprintf_func(info->stream, "%s", symbol_name);
    } else {
        info->fprintf_func(info->stream, "%016lx", block);
    }

#else
    info->fprintf_func(info->stream, "%016lx", block);
#endif

    return sizeof(block);
}

static bool tcg_out_qemu_ld_slow_path(TCGContext *s, TCGLabelQemuLdst *l)
{
    g_assert_not_reached();
}

static bool tcg_out_qemu_st_slow_path(TCGContext *s, TCGLabelQemuLdst *l)
{
    g_assert_not_reached();
}
