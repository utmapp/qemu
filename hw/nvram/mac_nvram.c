/*
 * PowerMac NVRAM emulation
 *
 * Copyright (c) 2005-2007 Fabrice Bellard
 * Copyright (c) 2007 Jocelyn Mayer
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
#include "qapi/error.h"
#include "hw/nvram/chrp_nvram.h"
#include "hw/nvram/mac_nvram.h"
#include "hw/qdev-properties.h"
#include "hw/qdev-properties-system.h"
#include "system/block-backend.h"
#include "migration/vmstate.h"
#include "qemu/cutils.h"
#include "qemu/module.h"
#include "qemu/error-report.h"
#include "trace.h"
#include <zlib.h> /* for adler32 */

#define DEF_SYSTEM_SIZE 0xc10

/* macio style NVRAM device */
static void macio_nvram_writeb(void *opaque, hwaddr addr,
                               uint64_t value, unsigned size)
{
    MacIONVRAMState *s = opaque;

    addr = (addr >> s->it_shift) & (s->size - 1);
    trace_macio_nvram_write(addr, value);
    s->data[addr] = value;
    if (s->blk) {
        if (blk_pwrite(s->blk, addr, 1, &s->data[addr], 0) < 0) {
            error_report("%s: write of NVRAM data to backing store failed",
                         blk_name(s->blk));
        }
    }
}

static uint64_t macio_nvram_readb(void *opaque, hwaddr addr,
                                  unsigned size)
{
    MacIONVRAMState *s = opaque;
    uint32_t value;

    addr = (addr >> s->it_shift) & (s->size - 1);
    value = s->data[addr];
    trace_macio_nvram_read(addr, value);

    return value;
}

static const MemoryRegionOps macio_nvram_ops = {
    .read = macio_nvram_readb,
    .write = macio_nvram_writeb,
    .valid.min_access_size = 1,
    .valid.max_access_size = 4,
    .impl.min_access_size = 1,
    .impl.max_access_size = 1,
    .endianness = DEVICE_BIG_ENDIAN,
};

static const VMStateDescription vmstate_macio_nvram = {
    .name = "macio_nvram",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_VBUFFER_UINT32(data, MacIONVRAMState, 0, NULL, size),
        VMSTATE_END_OF_LIST()
    }
};


static void macio_nvram_reset(DeviceState *dev)
{
}

static void macio_nvram_realizefn(DeviceState *dev, Error **errp)
{
    SysBusDevice *d = SYS_BUS_DEVICE(dev);
    MacIONVRAMState *s = MACIO_NVRAM(dev);

    s->data = g_malloc0(s->size);

    if (s->blk) {
        int64_t len = blk_getlength(s->blk);
        if (len < 0) {
            error_setg_errno(errp, -len,
                             "could not get length of nvram backing image");
            return;
        } else if (len != s->size) {
            error_setg_errno(errp, -len,
                             "invalid size nvram backing image");
            return;
        }
        if (blk_set_perm(s->blk, BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,
                         BLK_PERM_ALL, errp) < 0) {
            return;
        }
        if (blk_pread(s->blk, 0, s->size, s->data, 0) < 0) {
            error_setg(errp, "can't read-nvram contents");
            return;
        }
    }

    memory_region_init_io(&s->mem, OBJECT(s), &macio_nvram_ops, s,
                          "macio-nvram", s->size << s->it_shift);
    sysbus_init_mmio(d, &s->mem);
}

static void macio_nvram_unrealizefn(DeviceState *dev)
{
    MacIONVRAMState *s = MACIO_NVRAM(dev);

    g_free(s->data);
}

static const Property macio_nvram_properties[] = {
    DEFINE_PROP_UINT32("size", MacIONVRAMState, size, 0),
    DEFINE_PROP_UINT32("it_shift", MacIONVRAMState, it_shift, 0),
    DEFINE_PROP_DRIVE("drive", MacIONVRAMState, blk),
};

static void macio_nvram_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);

    dc->realize = macio_nvram_realizefn;
    dc->unrealize = macio_nvram_unrealizefn;
    device_class_set_legacy_reset(dc, macio_nvram_reset);
    dc->vmsd = &vmstate_macio_nvram;
    device_class_set_props(dc, macio_nvram_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo macio_nvram_type_info = {
    .name = TYPE_MACIO_NVRAM,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(MacIONVRAMState),
    .class_init = macio_nvram_class_init,
};

static void macio_nvram_register_types(void)
{
    type_register_static(&macio_nvram_type_info);
}

/* Set up a system OpenBIOS NVRAM partition */
static void pmac_format_nvram_partition_of(uint8_t *data, int len)
{
    int sysp_end;

    /* OpenBIOS nvram variables partition */
    sysp_end = chrp_nvram_create_system_partition(data, DEF_SYSTEM_SIZE, len);

    /*
     * Free space partition.  It is made to reach the end of the bank so
     * that whoever walks these partitions stops here: on a NewWorld Mac
     * the bytes past it are read by somebody else (see
     * pmac_format_nvram_partition below).
     */
    chrp_nvram_create_free_partition(&data[sysp_end], len - sysp_end);
}

#define OSX_NVRAM_SIGNATURE     (0x5A)

/*
 * Mac OS X finds its Open Firmware variables by walking the bank for a
 * partition called "common" -- it matches on the name, not the signature.
 * Without one, IODTNVRAM comes up with no variables at all: the "options"
 * node exists but has no properties, /usr/sbin/nvram fails, and everything
 * that reads a variable through it fails with it.  bless --setBoot dies on
 * that error, and bless --setBoot is the last step of every Mac OS X
 * install, so the install ends in "could not make the computer start up
 * from the volume" over a system that is complete and correct.
 *
 * These are only the values the bank starts out with; the guest writes
 * whatever it likes over them, and nothing here reads them back.  Open
 * Firmware keeps its own variables in its own bank.
 */
static const char * const osx_of_variables[] = {
    "little-endian?=false",
    "real-mode?=false",
    "auto-boot?=true",
    "diag-switch?=false",
    "use-nvramrc?=false",
    "boot-command=mac-boot",
    "boot-args=",
    "boot-device=hd:,\\\\:tbxi",
    "boot-file=",
    "boot-screen=",
    "console-screen=",
    "diag-device=enet",
    "diag-file=",
    "input-device=keyboard",
    "output-device=screen",
};

#define OSX_COMMON_SIZE     0x800
#define OSX_XPRAM_SIZE      0x100

/* Set up the Mac OS X bank: "common", parameter RAM, and free space */
static void pmac_format_nvram_partition_osx(uint8_t *bank, int len)
{
    ChrpNvramPartHdr *part_header;
    unsigned char *data;
    unsigned int i;
    int at = 0, end, free_len;

    /* The variables partition, which is the one Mac OS X is looking for. */
    data = &bank[at];
    part_header = (ChrpNvramPartHdr *)data;
    part_header->signature = CHRP_NVPART_SYSTEM;
    pstrcpy(part_header->name, sizeof(part_header->name), "common");
    end = sizeof(*part_header);
    for (i = 0; i < ARRAY_SIZE(osx_of_variables); i++) {
        int l = strlen(osx_of_variables[i]) + 1;
        if (end + l + 1 > OSX_COMMON_SIZE) {
            break;
        }
        memcpy(&data[end], osx_of_variables[i], l);
        end += l;
    }
    data[end] = '\0';                          /* end of the variable list */
    chrp_nvram_finish_partition(part_header, OSX_COMMON_SIZE);
    at += OSX_COMMON_SIZE;

    /* Where a real Mac keeps its parameter RAM.  Empty is fine. */
    data = &bank[at];
    part_header = (ChrpNvramPartHdr *)data;
    part_header->signature = OSX_NVRAM_SIGNATURE;
    pstrcpy(part_header->name, sizeof(part_header->name), "APL,MacOS75");
    chrp_nvram_finish_partition(part_header, OSX_XPRAM_SIZE);
    at += OSX_XPRAM_SIZE;

    /* The rest is free, and carries this bank's generation and checksum. */
    free_len = len - at;
    data = &bank[at];
    part_header = (ChrpNvramPartHdr *)data;
    part_header->signature = OSX_NVRAM_SIGNATURE;
    pstrcpy(part_header->name, sizeof(part_header->name), "wwwwwwwwwwww");

    chrp_nvram_finish_partition(part_header, free_len);

    /* Generation */
    stl_be_p(&data[20], 2);

    /* Adler32 checksum */
    stl_be_p(&data[16], adler32(0, &data[20], free_len - 20));
}

/*
 * Set up NVRAM with an Open Firmware bank and a Mac OS X bank.
 *
 * The two readers do not agree about this chip, and both are hard to
 * argue with:
 *
 *   - OpenBIOS reads it the way a NewWorld Mac's device tree describes
 *     it, 8 KB of storage in 16 KB of address space, one byte every
 *     other address.  Its variables have to be there or it ignores them
 *     -- including the boot-command PowerEmu uses to set up the AGP
 *     bridge.  Measured: with the chip mapped one byte per address,
 *     auto-boot?=false is not obeyed.
 *
 *   - Mac OS X reads the second 8 KB of that address space byte by byte
 *     and takes it for a whole bank.  Measured: two sweeps, both one
 *     byte per address, 0x0000-0x3fff and then 0x2000-0x3fff.  Give it
 *     every other byte and it finds no partitions at all.
 *
 * So the chip is 16 KB of plain, byte-addressed storage, and Open
 * Firmware's bank is written into the first half twice over -- each byte
 * at both of the addresses OpenBIOS's stride would read it from -- while
 * Mac OS X's bank sits in the second half where Mac OS X looks for it.
 * Each reader then finds what it expects at the address it expects.
 *
 * The one place they overlap is Open Firmware's free space, which in its
 * view covers the bytes Mac OS X's bank occupies.  OpenBIOS does not
 * write there: it cannot grow a partition, which is why the system
 * partition is created with room to spare.
 */
void pmac_format_nvram_banks(MacIONVRAMState *nvr, int bank)
{
    g_autofree uint8_t *of = g_malloc0(bank);
    int i;

    /*
     * Open Firmware's partitions are laid out over the whole of its
     * storage, so that its free partition reaches the end and it never
     * looks at what is beyond -- but only the first half of them is
     * written out, because the second half of its storage is the other
     * bank, and it is only ever free space to Open Firmware.
     */
    pmac_format_nvram_partition_of(of, bank);
    for (i = 0; i < bank / 2; i++) {
        nvr->data[i * 2] = of[i];
        nvr->data[i * 2 + 1] = of[i];
    }

    pmac_format_nvram_partition_osx(&nvr->data[bank], bank);
}

/*
 * The original layout, byte for byte: Open Firmware's partitions in the
 * first half, one empty Mac OS X partition in the second. OldWorld Macs use
 * it, and so does mac99 unless osx-nvram is set, so existing VMs and their
 * saved states are unchanged.
 */
void pmac_format_nvram_partition(MacIONVRAMState *nvr, int len)
{
    ChrpNvramPartHdr *part_header;
    unsigned char *data = &nvr->data[len / 2];
    int half = len / 2;

    pmac_format_nvram_partition_of(nvr->data, half);

    part_header = (ChrpNvramPartHdr *)data;
    part_header->signature = OSX_NVRAM_SIGNATURE;
    pstrcpy(part_header->name, sizeof(part_header->name), "wwwwwwwwwwww");
    chrp_nvram_finish_partition(part_header, half);
    stl_be_p(&data[20], 2);                                 /* generation */
    stl_be_p(&data[16], adler32(0, &data[20], half - 20));  /* checksum */
}
type_init(macio_nvram_register_types)
