/*
 * QEMU Hypervisor.framework (HVF) support
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 */

/* header to be included in HVF-specific code */

#ifndef HVF_INT_H
#define HVF_INT_H

#ifdef __aarch64__
#include <Hypervisor/Hypervisor.h>
typedef hv_vcpu_t hvf_vcpuid;
#else
#include <Hypervisor/hv.h>
typedef hv_vcpuid_t hvf_vcpuid;
#endif

#if defined(CONFIG_HVF_PRIVATE)
extern hv_return_t _hv_vm_config_set_isa(hv_vm_config_t config, uint32_t isa);
extern hv_return_t _hv_vcpu_get_actlr(hv_vcpu_t vcpu, uint64_t* value);
extern hv_return_t _hv_vcpu_set_actlr(hv_vcpu_t vcpu, uint64_t value);
#endif

#if defined(__aarch64__)
#if defined(CONFIG_HVF_PRIVATE)
#define HV_VM_CONFIG_ISA_PRIVATE (3)
#define ACTLR_EL1_TSO_ENABLE_MASK ((1 << 1) | (1 << 9))
#else
#define ACTLR_EL1_TSO_ENABLE_MASK ((1 << 1))
#endif
#endif

/* hvf_slot flags */
#define HVF_SLOT_LOG (1 << 0)

/* Represent memory logically mapped by QEMU */
typedef struct hvf_slot {
    uint64_t start;
    uint64_t size;
    uint8_t *mem;
    int slot_id;
    uint32_t flags;
    MemoryRegion *region;
} hvf_slot;

/* Represent memory currently mapped in HVF */
typedef struct hvf_mac_slot {
    int present;
    uint64_t size;
    uint64_t gpa_start;
    uint64_t gva;
} hvf_mac_slot;

typedef struct hvf_vcpu_caps {
    uint64_t vmx_cap_pinbased;
    uint64_t vmx_cap_procbased;
    uint64_t vmx_cap_procbased2;
    uint64_t vmx_cap_entry;
    uint64_t vmx_cap_exit;
    uint64_t vmx_cap_preemption_timer;
} hvf_vcpu_caps;

struct HVFState {
    AccelState parent;
    hvf_slot *slots;
    hvf_mac_slot *mac_slots;
    int num_slots;

    hvf_vcpu_caps *hvf_caps;
    uint64_t vtimer_offset;
    QTAILQ_HEAD(, hvf_sw_breakpoint) hvf_sw_breakpoints;
};
extern HVFState *hvf_state;
extern bool hvf_tso_mode;
extern uint32_t hvf_ipa_granule_size;

struct AccelCPUState {
    hvf_vcpuid fd;
    void *exit;
    bool vtimer_masked;
    sigset_t unblock_ipi_mask;
    bool guest_debug_enabled;
    bool dirty;
};

void assert_hvf_ok_impl(hv_return_t ret, const char *file, unsigned int line,
                        const char *exp);
#define assert_hvf_ok(EX) assert_hvf_ok_impl((EX), __FILE__, __LINE__, #EX)
const char *hvf_return_string(hv_return_t ret);
int hvf_arch_init(void);
hv_return_t hvf_arch_vm_create(MachineState *ms, uint32_t pa_range,
                               uint32_t ipa_granule_size);
int hvf_arch_init_vcpu(CPUState *cpu);
void hvf_arch_vcpu_destroy(CPUState *cpu);
int hvf_vcpu_exec(CPUState *);
hvf_slot *hvf_find_overlap_slot(uint64_t, uint64_t);
int hvf_put_registers(CPUState *);
int hvf_get_registers(CPUState *);
void hvf_kick_vcpu_thread(CPUState *cpu);

#endif
