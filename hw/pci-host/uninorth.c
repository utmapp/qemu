/*
 * QEMU Uninorth PCI host (for all Mac99 and newer machines)
 *
 * Copyright (c) 2006 Fabrice Bellard
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

#include "qemu/osdep.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "qemu/module.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/pci_host.h"
#include "hw/pci-host/uninorth.h"
#include "qemu/range.h"
#include "trace.h"

/*
 * UniNorth AGP GART table base.
 *
 * Mac OS X's AGP driver programs the physical address of the GART page
 * table into the AGP bridge's PCI config space at UNI_N_CFG_GART_BASE.
 * The ppc-mac-gpu device needs it to translate AGP aperture addresses,
 * because the bridge's GART is often programmed to a different table than
 * the GPU's own AIC_PT_BASE.
 */
#define UNI_N_CFG_GART_BASE 0x8c

/*
 * Real Uni-North AGP bridge layout (see Linux drivers/char/agp/uninorth-agp.c):
 * the AGP capability sits at 0x80 and fills 0x80-0x8b, and the GART
 * registers follow immediately after it.
 */
#define UNI_N_AGP_CAP_OFFSET   0x80
#define UNI_N_CFG_AGP_BASE     0x90
#define UNI_N_CFG_GART_CTRL    0x94
#define UNI_N_CFG_INTERNAL_STATUS 0x98

/*
 * Advertise an AGP capability on a host bridge and make its GART registers
 * writable.  Apple's AppleMacRiscAGP::configure() looks for capability 2 on
 * the host bridge itself and refuses the whole bus without it; only then
 * will it create IOAGPDevice nubs for AGP-capable cards behind it.
 *
 * QEMU drops config writes outside the write mask, so the command and GART
 * registers must be marked writable or the guest's programming is lost.
 */
static void unin_add_agp_capability(PCIDevice *d, Error **errp)
{
    int cap = pci_add_capability(d, PCI_CAP_ID_AGP, UNI_N_AGP_CAP_OFFSET,
                                 12, errp);
    if (cap < 0) {
        return;
    }
    d->config[cap + 2] = 0x20;                      /* AGP revision 2.0 */
    /* Status: RQ depth 0x1f, sideband addressing, 4x/2x/1x rates. */
    pci_set_long(d->config + cap + 4, 0x1f000207);
    pci_set_long(d->wmask + cap + 8, 0xffffffff);   /* AGP command */
    pci_set_long(d->wmask + UNI_N_CFG_GART_BASE, 0xffffffff);
    pci_set_long(d->wmask + UNI_N_CFG_AGP_BASE, 0xffffffff);
    pci_set_long(d->wmask + UNI_N_CFG_GART_CTRL, 0xffffffff);

    /*
     * Internal status, bit 0: AGP state machine idle/ready.  After enabling
     * AGP, AppleMacRiscAGP::setAGPEnable() spins on this bit with no
     * timeout; left at 0 it hangs the boot in a tight config-read loop.
     * An emulated link is always ready, so report it permanently set.
     */
    pci_set_long(d->config + UNI_N_CFG_INTERNAL_STATUS, 0x00000001);
}

static hwaddr unin_agp_gart_base;
static uint32_t unin_agp_gart_gen;     /* bumped on any GART register write */

uint32_t uninorth_get_agp_gart_gen(void)
{
    return qatomic_read(&unin_agp_gart_gen);
}

hwaddr uninorth_get_agp_gart_base(void)
{
    return unin_agp_gart_base;
}

static int pci_unin_map_irq(PCIDevice *pci_dev, int irq_num)
{
    return (irq_num + (pci_dev->devfn >> 3)) & 3;
}

static void pci_unin_set_irq(void *opaque, int irq_num, int level)
{
    UNINHostState *s = opaque;

    trace_unin_set_irq(irq_num, level);
    qemu_set_irq(s->irqs[irq_num], level);
}

static uint32_t unin_get_config_reg(uint32_t reg, uint32_t addr)
{
    uint32_t retval;

    if (reg & (1u << 31)) {
        /* XXX OpenBIOS compatibility hack */
        retval = reg | (addr & 3);
    } else if (reg & 1) {
        /* CFA1 style */
        retval = (reg & ~7u) | (addr & 7);
    } else {
        uint32_t slot, func;

        /* Grab CFA0 style values */
        slot = ctz32(reg & 0xfffff800);
        if (slot == 32) {
            slot = -1; /* XXX: should this be 0? */
        }
        func = PCI_FUNC(reg >> 8);

        /* ... and then convert them to x86 format */
        /* config pointer */
        retval = (reg & (0xff - 7)) | (addr & 7);
        /* slot, fn */
        retval |= PCI_DEVFN(slot, func) << 8;
    }

    trace_unin_get_config_reg(reg, addr, retval);

    return retval;
}

static void unin_data_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned len)
{
    UNINHostState *s = opaque;
    PCIHostState *phb = PCI_HOST_BRIDGE(s);
    trace_unin_data_write(addr, len, val);
    pci_data_write(phb->bus,
                   unin_get_config_reg(phb->config_reg, addr),
                   val, len);
}

static uint64_t unin_data_read(void *opaque, hwaddr addr,
                               unsigned len)
{
    UNINHostState *s = opaque;
    PCIHostState *phb = PCI_HOST_BRIDGE(s);
    uint32_t val;

    val = pci_data_read(phb->bus,
                        unin_get_config_reg(phb->config_reg, addr),
                        len);
    trace_unin_data_read(addr, len, val);
    return val;
}

static const MemoryRegionOps unin_data_ops = {
    .read = unin_data_read,
    .write = unin_data_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static char *pci_unin_main_ofw_unit_address(const SysBusDevice *dev)
{
    UNINHostState *s = UNI_NORTH_PCI_HOST_BRIDGE(dev);

    return g_strdup_printf("%x", s->ofw_addr);
}

static void pci_unin_main_realize(DeviceState *dev, Error **errp)
{
    UNINHostState *s = UNI_NORTH_PCI_HOST_BRIDGE(dev);
    PCIHostState *h = PCI_HOST_BRIDGE(dev);

    h->bus = pci_register_root_bus(dev, NULL,
                                   pci_unin_set_irq, pci_unin_map_irq,
                                   s,
                                   &s->pci_mmio,
                                   &s->pci_io,
                                   PCI_DEVFN(11, 0), 4, TYPE_PCI_BUS);

    pci_create_simple(h->bus, PCI_DEVFN(11, 0), "uni-north-pci");

    /*
     * DEC 21154 bridge was unused for many years, this comment is
     * a placeholder for whoever wishes to resurrect it
     */
}

static void pci_unin_main_init(Object *obj)
{
    UNINHostState *s = UNI_NORTH_PCI_HOST_BRIDGE(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    /* Use values found on a real PowerMac */
    /* Uninorth main bus */
    memory_region_init_io(&h->conf_mem, OBJECT(h), &pci_host_conf_le_ops,
                          obj, "unin-pci-conf-idx", 0x1000);
    memory_region_init_io(&h->data_mem, OBJECT(h), &unin_data_ops, obj,
                          "unin-pci-conf-data", 0x1000);

    memory_region_init(&s->pci_mmio, OBJECT(s), "unin-pci-mmio",
                       0x100000000ULL);
    memory_region_init_io(&s->pci_io, OBJECT(s), &unassigned_io_ops, obj,
                          "unin-pci-isa-mmio", 0x00800000);

    memory_region_init_alias(&s->pci_hole, OBJECT(s),
                             "unin-pci-hole", &s->pci_mmio,
                             0x80000000ULL, 0x10000000ULL);

    sysbus_init_mmio(sbd, &h->conf_mem);
    sysbus_init_mmio(sbd, &h->data_mem);
    sysbus_init_mmio(sbd, &s->pci_hole);
    sysbus_init_mmio(sbd, &s->pci_io);

    qdev_init_gpio_out(DEVICE(obj), s->irqs, ARRAY_SIZE(s->irqs));
}

static void pci_u3_agp_realize(DeviceState *dev, Error **errp)
{
    UNINHostState *s = U3_AGP_HOST_BRIDGE(dev);
    PCIHostState *h = PCI_HOST_BRIDGE(dev);

    h->bus = pci_register_root_bus(dev, NULL,
                                   pci_unin_set_irq, pci_unin_map_irq,
                                   s,
                                   &s->pci_mmio,
                                   &s->pci_io,
                                   PCI_DEVFN(11, 0), 4, TYPE_PCI_BUS);

    pci_create_simple(h->bus, PCI_DEVFN(11, 0), "u3-agp");
}

static void pci_u3_agp_init(Object *obj)
{
    UNINHostState *s = U3_AGP_HOST_BRIDGE(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    /* Uninorth U3 AGP bus */
    memory_region_init_io(&h->conf_mem, OBJECT(h), &pci_host_conf_le_ops,
                          obj, "unin-pci-conf-idx", 0x1000);
    memory_region_init_io(&h->data_mem, OBJECT(h), &unin_data_ops, obj,
                          "unin-pci-conf-data", 0x1000);

    memory_region_init(&s->pci_mmio, OBJECT(s), "unin-pci-mmio",
                       0x100000000ULL);
    memory_region_init_io(&s->pci_io, OBJECT(s), &unassigned_io_ops, obj,
                          "unin-pci-isa-mmio", 0x00800000);

    memory_region_init_alias(&s->pci_hole, OBJECT(s),
                             "unin-pci-hole", &s->pci_mmio,
                             0x80000000ULL, 0x70000000ULL);

    sysbus_init_mmio(sbd, &h->conf_mem);
    sysbus_init_mmio(sbd, &h->data_mem);
    sysbus_init_mmio(sbd, &s->pci_hole);
    sysbus_init_mmio(sbd, &s->pci_io);

    qdev_init_gpio_out(DEVICE(obj), s->irqs, ARRAY_SIZE(s->irqs));
}

static void pci_unin_agp_realize(DeviceState *dev, Error **errp)
{
    UNINHostState *s = UNI_NORTH_AGP_HOST_BRIDGE(dev);
    PCIHostState *h = PCI_HOST_BRIDGE(dev);

    h->bus = pci_register_root_bus(dev, NULL,
                                   pci_unin_set_irq, pci_unin_map_irq,
                                   s,
                                   &s->pci_mmio,
                                   &s->pci_io,
                                   PCI_DEVFN(11, 0), 4, TYPE_PCI_BUS);

    pci_create_simple(h->bus, PCI_DEVFN(11, 0), "uni-north-agp");
}

static void pci_unin_agp_init(Object *obj)
{
    UNINHostState *s = UNI_NORTH_AGP_HOST_BRIDGE(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    /* Uninorth AGP bus */
    memory_region_init_io(&h->conf_mem, OBJECT(h), &pci_host_conf_le_ops,
                          obj, "unin-agp-conf-idx", 0x1000);
    memory_region_init_io(&h->data_mem, OBJECT(h), &pci_host_data_le_ops,
                          obj, "unin-agp-conf-data", 0x1000);

    sysbus_init_mmio(sbd, &h->conf_mem);
    sysbus_init_mmio(sbd, &h->data_mem);

    qdev_init_gpio_out(DEVICE(obj), s->irqs, ARRAY_SIZE(s->irqs));
}

static void pci_unin_internal_realize(DeviceState *dev, Error **errp)
{
    UNINHostState *s = UNI_NORTH_INTERNAL_PCI_HOST_BRIDGE(dev);
    PCIHostState *h = PCI_HOST_BRIDGE(dev);

    h->bus = pci_register_root_bus(dev, NULL,
                                   pci_unin_set_irq, pci_unin_map_irq,
                                   s,
                                   &s->pci_mmio,
                                   &s->pci_io,
                                   PCI_DEVFN(14, 0), 4, TYPE_PCI_BUS);

    pci_create_simple(h->bus, PCI_DEVFN(14, 0), "uni-north-internal-pci");
}

static void pci_unin_internal_init(Object *obj)
{
    UNINHostState *s = UNI_NORTH_INTERNAL_PCI_HOST_BRIDGE(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    /* Uninorth internal bus */
    memory_region_init_io(&h->conf_mem, OBJECT(h), &pci_host_conf_le_ops,
                          obj, "unin-pci-conf-idx", 0x1000);
    memory_region_init_io(&h->data_mem, OBJECT(h), &pci_host_data_le_ops,
                          obj, "unin-pci-conf-data", 0x1000);

    sysbus_init_mmio(sbd, &h->conf_mem);
    sysbus_init_mmio(sbd, &h->data_mem);

    qdev_init_gpio_out(DEVICE(obj), s->irqs, ARRAY_SIZE(s->irqs));
}

/*
 * The main PCI host bridge (0xf2000000) carries the boot disk, USB and
 * mac-io, so presenting it as AGP-capable is opt-in:
 *   -global uni-north-pci.agp-capable=on
 * The same switch widens the PCI memory window (see realize below); with
 * it off the bridge is exactly the stock one.
 */
typedef struct UNINMainPCIHost {
    PCIDevice parent_obj;
    bool agp_capable;
} UNINMainPCIHost;

static void unin_main_pci_host_realize(PCIDevice *d, Error **errp)
{
    d->config[PCI_CACHE_LINE_SIZE] = 0x08;
    d->config[PCI_LATENCY_TIMER] = 0x10;
    d->config[PCI_CAPABILITY_LIST] = 0x00;

    /*
     * Set kMacRISCPCIAddressSelect (0x48) register to indicate PCI
     * memory space with base 0x80000000, size 0x10000000 for Apple's
     * AppleMacRiscPCI driver
     */
    d->config[0x48] = 0x0;
    d->config[0x49] = 0x0;
    d->config[0x4a] = 0x0;
    d->config[0x4b] = 0x1;

    if (((UNINMainPCIHost *)d)->agp_capable) {
        BusState *bus = BUS(pci_get_bus(d));
        UNINHostState *s = UNI_NORTH_PCI_HOST_BRIDGE(bus->parent);

        /*
         * 1.75 GB (0x80000000-0xefffffff) rather than the 256 MB of the
         * stock machine: a graphics card's VRAM BAR is aligned to its size,
         * and the R300's BAR holds two apertures (2 x VRAM).  With 128 MB
         * of VRAM the 256 MB BAR lands at 0x90000000 and the register BAR
         * at 0xa0000000; with 256 MB the 512 MB BAR lands at 0xa0000000 and
         * the register BAR at 0xc0000000, where the CPU could not reach it
         * through a 1 GB hole.  Nothing else lives at 0x9xxxxxxx-0xexxxxxxx
         * on mac99.  The alias is not mapped yet, so it can be resized.
         *
         * kMacRISCPCIAddressSelect: the upper 16 bits select 256 MB
         * regions, bit n for n * 256 MB; bits 8-14 are 0x80000000-0xefffffff.
         */
        memory_region_set_size(&s->pci_hole, 0x70000000ULL);
        d->config[0x4b] = 0x7f;
        unin_add_agp_capability(d, errp);
    }
}

static void unin_agp_pci_host_realize(PCIDevice *d, Error **errp)
{
    d->config[PCI_CACHE_LINE_SIZE] = 0x08;
    d->config[PCI_LATENCY_TIMER] = 0x10;
    /* d->config[PCI_CAPABILITY_LIST] = 0x80; */
}

static void unin_agp_pci_host_config_write(PCIDevice *d, uint32_t addr,
                                           uint32_t val, int len)
{
    pci_default_write_config(d, addr, val, len);

    /* GART base / aperture / control (invalidate) - drop cached translations */
    if (ranges_overlap(addr, len, UNI_N_CFG_GART_BASE, 12)) {
        qatomic_inc(&unin_agp_gart_gen);
    }
    if (ranges_overlap(addr, len, UNI_N_CFG_GART_BASE, 4)) {
        uint32_t base = pci_get_long(d->config + UNI_N_CFG_GART_BASE);
        /*
         * The low bits carry control flags (bit 0 enables the GART); the
         * table itself is page aligned.
         */
        unin_agp_gart_base = (hwaddr)(base & ~0xFFFU);
    }
}

static void u3_agp_pci_host_realize(PCIDevice *d, Error **errp)
{
    d->config[PCI_CACHE_LINE_SIZE] = 0x08;
    d->config[PCI_LATENCY_TIMER] = 0x10;
}

static void unin_internal_pci_host_realize(PCIDevice *d, Error **errp)
{
    d->config[PCI_CACHE_LINE_SIZE] = 0x08;
    d->config[PCI_LATENCY_TIMER] = 0x10;
    d->config[PCI_CAPABILITY_LIST] = 0x00;
}

static const Property unin_main_pci_host_props[] = {
    DEFINE_PROP_BOOL("agp-capable", UNINMainPCIHost, agp_capable, false),
};

static void unin_main_pci_host_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    DeviceClass *dc = DEVICE_CLASS(klass);

    k->realize   = unin_main_pci_host_realize;
    k->config_write = unin_agp_pci_host_config_write;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    k->device_id = PCI_DEVICE_ID_APPLE_UNI_N_PCI;
    device_class_set_props(dc, unin_main_pci_host_props);
    k->revision  = 0x00;
    k->class_id  = PCI_CLASS_BRIDGE_HOST;
    /*
     * PCI-facing part of the host bridge, not usable without the
     * host-facing part, which can't be device_add'ed, yet.
     */
    dc->user_creatable = false;
}

static const TypeInfo unin_main_pci_host_info = {
    .name = "uni-north-pci",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(UNINMainPCIHost),
    .class_init = unin_main_pci_host_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void u3_agp_pci_host_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    DeviceClass *dc = DEVICE_CLASS(klass);

    k->realize   = u3_agp_pci_host_realize;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    k->device_id = PCI_DEVICE_ID_APPLE_U3_AGP;
    k->revision  = 0x00;
    k->class_id  = PCI_CLASS_BRIDGE_HOST;
    /*
     * PCI-facing part of the host bridge, not usable without the
     * host-facing part, which can't be device_add'ed, yet.
     */
    dc->user_creatable = false;
}

static const TypeInfo u3_agp_pci_host_info = {
    .name = "u3-agp",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PCIDevice),
    .class_init = u3_agp_pci_host_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void unin_agp_pci_host_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    DeviceClass *dc = DEVICE_CLASS(klass);

    k->realize   = unin_agp_pci_host_realize;
    k->config_write = unin_agp_pci_host_config_write;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    k->device_id = PCI_DEVICE_ID_APPLE_UNI_N_AGP;
    k->revision  = 0x00;
    k->class_id  = PCI_CLASS_BRIDGE_HOST;
    /*
     * PCI-facing part of the host bridge, not usable without the
     * host-facing part, which can't be device_add'ed, yet.
     */
    dc->user_creatable = false;
}

static const TypeInfo unin_agp_pci_host_info = {
    .name = "uni-north-agp",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PCIDevice),
    .class_init = unin_agp_pci_host_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void unin_internal_pci_host_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    DeviceClass *dc = DEVICE_CLASS(klass);

    k->realize   = unin_internal_pci_host_realize;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    k->device_id = PCI_DEVICE_ID_APPLE_UNI_N_I_PCI;
    k->revision  = 0x00;
    k->class_id  = PCI_CLASS_BRIDGE_HOST;
    /*
     * PCI-facing part of the host bridge, not usable without the
     * host-facing part, which can't be device_add'ed, yet.
     */
    dc->user_creatable = false;
}

static const TypeInfo unin_internal_pci_host_info = {
    .name = "uni-north-internal-pci",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PCIDevice),
    .class_init = unin_internal_pci_host_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static const Property pci_unin_main_pci_host_props[] = {
    DEFINE_PROP_UINT32("ofw-addr", UNINHostState, ofw_addr, -1),
};

static void pci_unin_main_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    SysBusDeviceClass *sbc = SYS_BUS_DEVICE_CLASS(klass);

    dc->realize = pci_unin_main_realize;
    device_class_set_props(dc, pci_unin_main_pci_host_props);
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
    dc->fw_name = "pci";
    sbc->explicit_ofw_unit_address = pci_unin_main_ofw_unit_address;
}

static const TypeInfo pci_unin_main_info = {
    .name          = TYPE_UNI_NORTH_PCI_HOST_BRIDGE,
    .parent        = TYPE_PCI_HOST_BRIDGE,
    .instance_size = sizeof(UNINHostState),
    .instance_init = pci_unin_main_init,
    .class_init    = pci_unin_main_class_init,
};

static void pci_u3_agp_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = pci_u3_agp_realize;
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo pci_u3_agp_info = {
    .name          = TYPE_U3_AGP_HOST_BRIDGE,
    .parent        = TYPE_PCI_HOST_BRIDGE,
    .instance_size = sizeof(UNINHostState),
    .instance_init = pci_u3_agp_init,
    .class_init    = pci_u3_agp_class_init,
};

static void pci_unin_agp_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = pci_unin_agp_realize;
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo pci_unin_agp_info = {
    .name          = TYPE_UNI_NORTH_AGP_HOST_BRIDGE,
    .parent        = TYPE_PCI_HOST_BRIDGE,
    .instance_size = sizeof(UNINHostState),
    .instance_init = pci_unin_agp_init,
    .class_init    = pci_unin_agp_class_init,
};

static void pci_unin_internal_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = pci_unin_internal_realize;
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo pci_unin_internal_info = {
    .name          = TYPE_UNI_NORTH_INTERNAL_PCI_HOST_BRIDGE,
    .parent        = TYPE_PCI_HOST_BRIDGE,
    .instance_size = sizeof(UNINHostState),
    .instance_init = pci_unin_internal_init,
    .class_init    = pci_unin_internal_class_init,
};

/* UniN device */
static void unin_write(void *opaque, hwaddr addr, uint64_t value,
                       unsigned size)
{
    trace_unin_write(addr, value);
}

static uint64_t unin_read(void *opaque, hwaddr addr, unsigned size)
{
    uint32_t value;

    switch (addr) {
    case 0:
        value = UNINORTH_VERSION_10A;
        break;
    default:
        value = 0;
    }

    trace_unin_read(addr, value);

    return value;
}

static const MemoryRegionOps unin_ops = {
    .read = unin_read,
    .write = unin_write,
    .endianness = DEVICE_BIG_ENDIAN,
};

static void unin_init(Object *obj)
{
    UNINState *s = UNI_NORTH(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->mem, obj, &unin_ops, s, "unin", 0x1000);

    sysbus_init_mmio(sbd, &s->mem);
}

static void unin_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo unin_info = {
    .name          = TYPE_UNI_NORTH,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(UNINState),
    .instance_init = unin_init,
    .class_init    = unin_class_init,
};

static void unin_register_types(void)
{
    type_register_static(&unin_main_pci_host_info);
    type_register_static(&u3_agp_pci_host_info);
    type_register_static(&unin_agp_pci_host_info);
    type_register_static(&unin_internal_pci_host_info);

    type_register_static(&pci_unin_main_info);
    type_register_static(&pci_u3_agp_info);
    type_register_static(&pci_unin_agp_info);
    type_register_static(&pci_unin_internal_info);

    type_register_static(&unin_info);
}

type_init(unin_register_types)
