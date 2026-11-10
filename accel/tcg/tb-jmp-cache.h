/*
 * The per-CPU TranslationBlock jump cache.
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ACCEL_TCG_TB_JMP_CACHE_H
#define ACCEL_TCG_TB_JMP_CACHE_H

#include "qemu/rcu.h"
#include "exec/cpu-common.h"

/*
 * 4096 entries (the upstream default) thrash badly under a Mac OS X guest:
 * a game's hot code covers 100k+ translation blocks, so most indirect
 * branches -- and PowerPC returns through blr are indirect -- miss the
 * cache and fall back to the (much slower) hash table.  64K entries cost
 * 1 MB per vCPU and turn most of those misses into hits.
 */
#define TB_JMP_CACHE_BITS 16
#define TB_JMP_CACHE_SIZE (1 << TB_JMP_CACHE_BITS)

/*
 * Invalidated in parallel; all accesses to 'tb' must be atomic.
 * A valid entry is read/written by a single CPU, therefore there is
 * no need for qatomic_rcu_read() and pc is always consistent with a
 * non-NULL value of 'tb'.  Strictly speaking pc is only needed for
 * CF_PCREL, but it's used always for simplicity.
 */
typedef struct CPUJumpCache {
    struct rcu_head rcu;
    /*
     * Emptying the cache means bumping this counter: an entry counts only
     * while its own gen matches.  Clearing 64K entries outright on every
     * TLB flush would cost far more than the cache saves, and a 32-bit
     * hash MMU guest flushes constantly.
     */
    uint32_t gen;
    struct {
        TranslationBlock *tb;
        vaddr pc;
        uint32_t gen;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
