#include "qemu/osdep.h"
#include "system/system.h"
#include "hw/pci/pci.h"
#include "ui/console.h"
#include "hw/qdev-properties.h"
#include "virtio-ramfb.h"
#include "qapi/error.h"
#include "qom/object.h"

static int virtio_ramfb_get_flags(void *opaque)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->hw_ops->get_flags) {
        return g->hw_ops->get_flags(g);
    } else {
        return 0;
    }
}

static void virtio_ramfb_invalidate_display(void *opaque)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->enable) {
        g->hw_ops->invalidate(g);
    }
}

static void virtio_ramfb_text_update(void *opaque, console_ch_t *chardata)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->hw_ops->text_update) {
        g->hw_ops->text_update(g, chardata);
    }
}

static void virtio_ramfb_update_display(void *opaque)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->enable) {
        g->hw_ops->gfx_update(g);
    } else {
        ramfb_display_update(g->scanout[0].con, vramfb->ramfb);
    }
}

static void virtio_ramfb_ui_info(void *opaque, uint32_t idx, QemuUIInfo *info)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->hw_ops->ui_info) {
        g->hw_ops->ui_info(g, idx, info);
    }
}

static void virtio_ramfb_gl_block(void *opaque, bool block)
{
    VirtIORAMFBBase *vramfb = opaque;
    VirtIOGPUBase *g = vramfb->vgpu;

    if (g->hw_ops->gl_block) {
        g->hw_ops->gl_block(g, block);
    }
}

static const GraphicHwOps virtio_ramfb_ops = {
    .get_flags = virtio_ramfb_get_flags,
    .invalidate = virtio_ramfb_invalidate_display,
    .gfx_update = virtio_ramfb_update_display,
    .text_update = virtio_ramfb_text_update,
    .ui_info = virtio_ramfb_ui_info,
    .gl_block = virtio_ramfb_gl_block,
};

static const VMStateDescription vmstate_virtio_ramfb = {
    .name = "virtio-ramfb",
    .version_id = 2,
    .minimum_version_id = 2,
    .fields = (VMStateField[]) {
        /* no pci stuff here, saving the virtio device will handle that */
        /* FIXME */
        VMSTATE_END_OF_LIST()
    }
};

/* RAMFB device wrapper around PCI device around virtio GPU */
static void virtio_ramfb_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    VirtIORAMFBBase *vramfb = VIRTIO_RAMFB_BASE(vpci_dev);
    VirtIOGPUBase *g = vramfb->vgpu;
    int i;

    if (virtio_gpu_hostmem_enabled(g->conf)) {
        vpci_dev->msix_bar_idx = 1;
        vpci_dev->modern_mem_bar_idx = 2;
        memory_region_init(&g->hostmem, OBJECT(g), "virtio-gpu-hostmem",
                           g->conf.hostmem);
        pci_register_bar(&vpci_dev->pci_dev, 4,
                         PCI_BASE_ADDRESS_SPACE_MEMORY |
                         PCI_BASE_ADDRESS_MEM_PREFETCH |
                         PCI_BASE_ADDRESS_MEM_TYPE_64,
                         &g->hostmem);
        virtio_pci_add_shm_cap(vpci_dev, 4, 0, g->conf.hostmem,
                               VIRTIO_GPU_SHM_ID_HOST_VISIBLE);
    }

    /* init virtio bits */
    virtio_pci_force_virtio_1(vpci_dev);
    if (!qdev_realize(DEVICE(g), BUS(&vpci_dev->bus), errp)) {
        return;
    }

    /* init ramfb */
    vramfb->ramfb = ramfb_setup(errp);
    graphic_console_set_hwops(g->scanout[0].con, &virtio_ramfb_ops, vramfb);

    for (i = 0; i < g->conf.max_outputs; i++) {
        object_property_set_link(OBJECT(g->scanout[i].con), "device",
                                 OBJECT(vpci_dev), &error_abort);
    }
}

static void virtio_ramfb_reset_hold(Object *obj, ResetType type)
{
    VirtIORAMFBBase *dev = VIRTIO_RAMFB_BASE(obj);
    VirtIORAMFBBaseClass *klass = VIRTIO_RAMFB_BASE_GET_CLASS(dev);

    /* reset virtio-gpu */
    if (klass->parent_phases.hold) {
        klass->parent_phases.hold(obj, type);
    }
}

static const Property virtio_ramfb_base_properties[] = {
    DEFINE_VIRTIO_GPU_PCI_PROPERTIES(VirtIOPCIProxy),
};

static void virtio_ramfb_base_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);
    VirtIORAMFBBaseClass *v = VIRTIO_RAMFB_BASE_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    device_class_set_props(dc, virtio_ramfb_base_properties);
    dc->vmsd = &vmstate_virtio_ramfb;
    dc->hotpluggable = false;
    resettable_class_set_parent_phases(rc, NULL, virtio_ramfb_reset_hold, NULL,
                                       &v->parent_phases);

    k->realize = virtio_ramfb_realize;
    pcidev_k->class_id = PCI_CLASS_DISPLAY_OTHER;
}

static const TypeInfo virtio_ramfb_base_info = {
    .name          = TYPE_VIRTIO_RAMFB_BASE,
    .parent        = TYPE_VIRTIO_PCI,
    .instance_size = sizeof(VirtIORAMFBBase),
    .class_size    = sizeof(VirtIORAMFBBaseClass),
    .class_init    = virtio_ramfb_base_class_init,
    .abstract      = true,
};

#define TYPE_VIRTIO_RAMFB "virtio-ramfb"

typedef struct VirtIORAMFB VirtIORAMFB;
DECLARE_INSTANCE_CHECKER(VirtIORAMFB, VIRTIO_RAMFB,
                         TYPE_VIRTIO_RAMFB)

struct VirtIORAMFB {
    VirtIORAMFBBase parent_obj;

    VirtIOGPU     vdev;
};

static void virtio_ramfb_inst_initfn(Object *obj)
{
    VirtIORAMFB *dev = VIRTIO_RAMFB(obj);

    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                                TYPE_VIRTIO_GPU);
    VIRTIO_RAMFB_BASE(dev)->vgpu = VIRTIO_GPU_BASE(&dev->vdev);
}

static VirtioPCIDeviceTypeInfo virtio_ramfb_info = {
    .generic_name  = TYPE_VIRTIO_RAMFB,
    .parent        = TYPE_VIRTIO_RAMFB_BASE,
    .instance_size = sizeof(VirtIORAMFB),
    .instance_init = virtio_ramfb_inst_initfn,
};

static void virtio_ramfb_register_types(void)
{
    type_register_static(&virtio_ramfb_base_info);
    virtio_pci_types_register(&virtio_ramfb_info);
}

type_init(virtio_ramfb_register_types)
