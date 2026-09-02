/*
 * QEMU Hypervisor.framework (HVF) support -- ARM specifics
 *
 * Copyright (c) 2021 Alexander Graf
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 */

#ifndef QEMU_HVF_ARM_H
#define QEMU_HVF_ARM_H

#include "cpu.h"

/**
 * hvf_arm_init_debug() - initialize guest debug capabilities
 *
 * Should be called only once before using guest debug capabilities.
 */
void hvf_arm_init_debug(void);

void hvf_arm_set_cpu_features_from_host(ARMCPU *cpu);

#if defined(CONFIG_HVF)

uint32_t hvf_arm_get_default_ipa_bit_size(void);
uint32_t hvf_arm_get_max_ipa_bit_size(void);

/*
 * Hypervisor.framework in-kernel GICv3 (hv_gic_*).  hvf_arm_gic_create()
 * must run after the VM exists and before any vCPU is created, i.e. before
 * the board realizes its CPUs; the "hvf-arm-gicv3" device then only wires
 * interrupt lines to it.
 */
void hvf_arm_gic_create(hwaddr dist_base, hwaddr redist_base,
                        hwaddr redist_size, unsigned max_cpus,
                        unsigned num_spi, Error **errp);
bool hvf_arm_gic_created(void);
void hvf_arm_gic_set_spi(uint32_t intid, int level);
void hvf_arm_gic_set_ppi(CPUState *cpu, uint32_t intid, int level);
void hvf_arm_gic_reset(void);

#else

static inline void hvf_arm_gic_create(hwaddr dist_base, hwaddr redist_base,
                                      hwaddr redist_size, unsigned max_cpus,
                                      unsigned num_spi, Error **errp)
{
}

static inline bool hvf_arm_gic_created(void)
{
    return false;
}

static inline void hvf_arm_gic_set_spi(uint32_t intid, int level)
{
}

static inline void hvf_arm_gic_set_ppi(CPUState *cpu, uint32_t intid,
                                       int level)
{
}

static inline void hvf_arm_gic_reset(void)
{
}

static inline uint32_t hvf_arm_get_default_ipa_bit_size(void)
{
    return 0;
}

static inline uint32_t hvf_arm_get_max_ipa_bit_size(void)
{
    return 0;
}

#endif

#endif
