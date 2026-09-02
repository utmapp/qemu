/*
 * ARM Generic Interrupt Controller v3 backed by the Hypervisor.framework
 * in-kernel GIC (hv_gic_*, macOS 15+).
 *
 * The distributor, the redistributors and the CPU interfaces live inside
 * the hypervisor: SGIs, timer PPIs and register accesses never leave the
 * kernel.  This device owns only the QEMU-side wiring -- SPIs raised by
 * emulated devices go to hv_gic_set_spi(), and the PPIs QEMU itself raises
 * (the emulated PMU) are pended into the owning vCPU's redistributor from
 * that vCPU's thread.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "hw/intc/arm_gicv3_common.h"
#include "hw/core/cpu.h"
#include "qemu/error-report.h"
#include "qemu/module.h"
#include "system/hvf.h"
#include "hvf_arm.h"
#include "migration/blocker.h"
#include "qom/object.h"

#define TYPE_HVF_ARM_GICV3 "hvf-arm-gicv3"
typedef struct HVFARMGICv3Class HVFARMGICv3Class;
/* This is reusing the GICv3State typedef from ARM_GICV3_ITS_COMMON */
DECLARE_OBJ_CHECKERS(GICv3State, HVFARMGICv3Class,
                     HVF_ARM_GICV3, TYPE_HVF_ARM_GICV3)

struct HVFARMGICv3Class {
    ARMGICv3CommonClass parent_class;
    DeviceRealize parent_realize;
    ResettablePhases parent_phases;
};

/*
 * GPIO layout (gicv3_init_irqs_and_mmio): [0..N-1] SPIs, then 32 PPI
 * lines per CPU indexed by (intid - 16).
 */
static void hvf_arm_gicv3_set_irq(void *opaque, int irq, int level)
{
    GICv3State *s = opaque;
    int num_spi = s->num_irq - GIC_INTERNAL;

    if (irq < num_spi) {
        hvf_arm_gic_set_spi(irq + GIC_INTERNAL, level);
    } else {
        int cpuidx = (irq - num_spi) / GIC_INTERNAL;
        int intid = (irq - num_spi) % GIC_INTERNAL + GIC_NR_SGIS;

        hvf_arm_gic_set_ppi(qemu_get_cpu(cpuidx), intid, level);
    }
}

static void hvf_arm_gicv3_realize(DeviceState *dev, Error **errp)
{
    GICv3State *s = HVF_ARM_GICV3(dev);
    HVFARMGICv3Class *hgc = HVF_ARM_GICV3_GET_CLASS(s);
    Error *local_err = NULL;

    hgc->parent_realize(dev, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        return;
    }

    if (s->revision != 3) {
        error_setg(errp, "unsupported GIC revision %d for the HVF in-kernel GIC",
                   s->revision);
        return;
    }
    if (s->security_extn) {
        error_setg(errp, "the HVF in-kernel GIC does not implement the "
                   "security extensions");
        return;
    }
    if (s->nmi_support) {
        error_setg(errp, "NMI is not supported with the HVF in-kernel GIC");
        return;
    }
    if (s->nb_redist_regions != 1) {
        error_setg(errp, "the HVF in-kernel GIC supports a single "
                   "redistributor region");
        return;
    }
    if (!hvf_arm_gic_created()) {
        error_setg(errp, "the HVF in-kernel GIC must be created before the "
                   "vCPUs (the board did not call hvf_arm_gic_create())");
        return;
    }

    /*
     * The MMIO regions are placeholders: guest accesses to the distributor
     * and redistributor frames are handled by the hypervisor and never
     * reach QEMU.
     */
    gicv3_init_irqs_and_mmio(s, hvf_arm_gicv3_set_irq, NULL);

    error_setg(&s->migration_blocker,
               "migration is not supported with the HVF in-kernel GIC");
    if (migrate_add_blocker(&s->migration_blocker, errp) < 0) {
        return;
    }
}

static void hvf_arm_gicv3_reset_hold(Object *obj, ResetType type)
{
    GICv3State *s = HVF_ARM_GICV3(obj);
    HVFARMGICv3Class *hgc = HVF_ARM_GICV3_GET_CLASS(s);

    if (hgc->parent_phases.hold) {
        hgc->parent_phases.hold(obj, type);
    }
    hvf_arm_gic_reset();
}

static void hvf_arm_gicv3_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);
    ARMGICv3CommonClass *agcc = ARM_GICV3_COMMON_CLASS(klass);
    HVFARMGICv3Class *hgc = HVF_ARM_GICV3_CLASS(klass);

    agcc->pre_save = NULL;
    agcc->post_load = NULL;
    device_class_set_parent_realize(dc, hvf_arm_gicv3_realize,
                                    &hgc->parent_realize);
    resettable_class_set_parent_phases(rc, NULL, hvf_arm_gicv3_reset_hold,
                                       NULL, &hgc->parent_phases);
}

static const TypeInfo hvf_arm_gicv3_info = {
    .name = TYPE_HVF_ARM_GICV3,
    .parent = TYPE_ARM_GICV3_COMMON,
    .instance_size = sizeof(GICv3State),
    .class_init = hvf_arm_gicv3_class_init,
    .class_size = sizeof(HVFARMGICv3Class),
};

static void hvf_arm_gicv3_register_types(void)
{
    type_register_static(&hvf_arm_gicv3_info);
}

type_init(hvf_arm_gicv3_register_types)
