/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: memory controller view and
 * command processor (CP, "PM4"), emulated at the packet level.
 *
 * Copyright (c) 2026 RetroBoot contributors
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 *
 * Register offsets and packet layouts follow AMD's R100 documentation and
 * the register definitions in the X.org/Linux radeon drivers.  The CP
 * microcode the guest uploads is kept (so it reads back) but not executed:
 * packets are interpreted directly.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "trace.h"

#define CP_PACKET_MAX_DWORDS 0x4000

/* ---------------------------------------------------------------------
 * Memory controller address space
 *
 * The CP, the bus-master engines and the 3D unit address memory with
 * card ("MC") addresses: MC_FB_LOCATION maps VRAM, MC_AGP_LOCATION the AGP
 * aperture (AGP_BASE = bus address), and a PCI card's GART (AIC_*) maps
 * pages of system memory through a table in system memory.  Anything else is
 * treated as a bus address (PCI cards bus-master straight to RAM).
 */

static uint64_t ati_mc_fb_start(ATIVGAState *s)
{
    return (uint64_t)(s->regfile[MC_FB_LOCATION / 4] & 0xffff) << 16;
}

static uint64_t ati_mc_fb_end(ATIVGAState *s)
{
    return ((uint64_t)(s->regfile[MC_FB_LOCATION / 4] >> 16) << 16) | 0xffff;
}

/*
 * Translate an MC address. Returns ATI_MC_VRAM with *off set to the VRAM
 * offset, ATI_MC_BUS with *off = bus address, or ATI_MC_NONE.
 * *contig is the number of bytes that are contiguous from there.
 */
int ati_mc_translate(ATIVGAState *s, uint64_t mc, uint64_t *off,
                     uint64_t *contig)
{
    uint64_t fb_start = ati_mc_fb_start(s), fb_end = ati_mc_fb_end(s);
    uint32_t aic = s->regfile[ATI_AIC_CNTL / 4];

    if (fb_end > fb_start && mc >= fb_start && mc <= fb_end) {
        uint64_t o = mc - fb_start;
        if (o >= s->vga.vram_size) {
            return ATI_MC_NONE;
        }
        *off = o;
        *contig = s->vga.vram_size - o;
        return ATI_MC_VRAM;
    }
    if (aic & 1) {      /* PCIGART_TRANSLATE_EN */
        uint64_t lo = s->regfile[ATI_AIC_LO_ADDR / 4];
        uint64_t hi = s->regfile[ATI_AIC_HI_ADDR / 4];
        if (mc >= lo && mc <= hi) {
            uint64_t page = (mc - lo) >> 12;
            uint64_t pte_addr = s->regfile[ATI_AIC_PT_BASE / 4] + page * 4;
            uint32_t pte = 0;
            pci_dma_read(&s->dev, pte_addr, &pte, 4);
            /* the GART table is written by the CPU in its own byte order */
            pte = ldl_le_p(&pte);
            *off = (pte & ~0xfffULL) | (mc & 0xfff);
            *contig = 0x1000 - (mc & 0xfff);
            return ATI_MC_BUS;
        }
    }
    {
        uint32_t agp = s->regfile[MC_AGP_LOCATION / 4];
        uint64_t a_start = (uint64_t)(agp & 0xffff) << 16;
        uint64_t a_end = ((uint64_t)(agp >> 16) << 16) | 0xffff;
        if (a_end > a_start && mc >= a_start && mc <= a_end) {
            *off = s->regfile[AGP_BASE / 4] + (mc - a_start);
            *contig = a_end + 1 - mc;
            return ATI_MC_BUS;
        }
    }
    *off = mc;
    *contig = UINT64_MAX;
    return ATI_MC_BUS;
}

bool ati_mc_rw(ATIVGAState *s, uint64_t mc, void *buf, uint64_t len,
               bool write)
{
    uint8_t *p = buf;

    while (len) {
        uint64_t off, contig, n;
        int kind = ati_mc_translate(s, mc, &off, &contig);

        n = MIN(len, contig);
        if (kind == ATI_MC_VRAM) {
            if (write) {
                memcpy(s->vga.vram_ptr + off, p, n);
                memory_region_set_dirty(&s->vga.vram, off, n);
            } else {
                memcpy(p, s->vga.vram_ptr + off, n);
            }
        } else if (kind == ATI_MC_BUS) {
            if (write) {
                pci_dma_write(&s->dev, off, p, n);
            } else {
                pci_dma_read(&s->dev, off, p, n);
            }
        } else {
            qemu_log_mask(LOG_GUEST_ERROR, "ati: MC access to unmapped 0x%"
                          PRIx64 "\n", mc);
            if (!write) {
                memset(p, 0, n);
            }
            return false;
        }
        p += n;
        mc += n;
        len -= n;
    }
    return true;
}

/* ---------------------------------------------------------------------
 * Packet execution
 */

static void ati_cp_exec_ib(ATIVGAState *s, uint32_t base, uint32_t ndw);

/*
 * Execute one complete packet starting at pkt[0]; avail dwords are present.
 * Returns the packet length in dwords, or 0 if it is incomplete.
 */
static unsigned ati_cp_packet(ATIVGAState *s, const uint32_t *pkt,
                              unsigned avail)
{
    uint32_t h = pkt[0];
    unsigned type = h >> 30, count, i;

    switch (type) {
    case 0:
    {
        uint32_t reg = (h & 0x1fff) << 2;
        bool one = h & BIT(15);
        count = ((h >> 16) & 0x3fff) + 1;
        if (avail < count + 1) {
            return 0;
        }
        trace_ati_cp_packet0(reg, count, one);
        for (i = 0; i < count; i++) {
            ati_reg_write(s, one ? reg : reg + 4 * i, pkt[1 + i]);
        }
        return count + 1;
    }
    case 1:
        if (avail < 3) {
            return 0;
        }
        ati_reg_write(s, (h & 0x7ff) << 2, pkt[1]);
        ati_reg_write(s, ((h >> 11) & 0x7ff) << 2, pkt[2]);
        return 3;
    case 2:
        return 1;
    default:
    {
        uint8_t op = (h >> 8) & 0xff;
        count = ((h >> 16) & 0x3fff) + 1;
        if (avail < count + 1) {
            return 0;
        }
        trace_ati_cp_packet3(op, count);
        s->cp.packets3++;
        switch (op) {
        case 0x10:  /* NOP */
        case 0x24:  /* LOAD_MICROCODE */
        case 0x26:  /* WAIT_FOR_IDLE: always idle here */
            break;
        case 0x33:  /* INDX_BUFFER: indirect buffer of 3D indices handled
                       by the 3D draw that uses it */
            ati_3d_packet(s, op, &pkt[1], count);
            break;
        case 0x91 ... 0x9f:   /* 2D CNTL packets */
            ati_2d_packet(s, op, &pkt[1], count);
            break;
        case 0x20 ... 0x23:
        case 0x25:
        case 0x28 ... 0x2f:
        case 0x32:
        case 0x34 ... 0x3f:
            ati_3d_packet(s, op, &pkt[1], count);
            break;
        default:
            {
                static uint32_t seen[8];

                if (!(seen[op >> 5] & (1u << (op & 31)))) {
                    seen[op >> 5] |= 1u << (op & 31);
                    trace_ati_cp_unhandled(op, count);
                    qemu_log_mask(LOG_UNIMP,
                                  "ati: CP packet3 op 0x%02x (%u dwords)\n",
                                  op, count);
                }
            }
            break;
        }
        return count + 1;
    }
    }
}

/* run complete packets in buf[0..n); returns dwords consumed */
static unsigned ati_cp_run(ATIVGAState *s, const uint32_t *buf, unsigned n)
{
    unsigned pos = 0;

    while (pos < n) {
        unsigned len = ati_cp_packet(s, buf + pos, n - pos);
        if (!len) {
            break;
        }
        pos += len;
    }
    return pos;
}

static void ati_cp_fetch(ATIVGAState *s, uint64_t mc, uint32_t *dst,
                         unsigned ndw, bool swap32)
{
    ati_mc_rw(s, mc, dst, (uint64_t)ndw * 4, false);
    for (unsigned i = 0; i < ndw; i++) {
        dst[i] = swap32 ? ldl_be_p(&dst[i]) : ldl_le_p(&dst[i]);
    }
}

static void ati_cp_exec_ib(ATIVGAState *s, uint32_t base, uint32_t ndw)
{
    uint32_t *buf;
    unsigned done;

    if (!ndw || ndw > 1 << 20) {
        return;
    }
    if (s->cp.depth > 4) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: indirect buffers nested too deep\n");
        return;
    }
    s->cp.depth++;
    buf = g_new(uint32_t, ndw);
    /* indirect buffers use the ring's swap setting */
    ati_cp_fetch(s, base, buf, ndw, (s->regfile[CP_RB_CNTL / 4] >> 16) & 3);
    trace_ati_cp_ib(base, ndw);
    s->accel.ib_dw += ndw;
    done = ati_cp_run(s, buf, ndw);
    if (done != ndw) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: indirect buffer ends inside a "
                      "packet (%u of %u dwords)\n", done, ndw);
    }
    g_free(buf);
    s->cp.depth--;
}

/* bus-master ring: run everything between RPTR and WPTR */
static void ati_cp_ring_kick(ATIVGAState *s)
{
    uint32_t cntl = s->regfile[CP_RB_CNTL / 4];
    unsigned size = 2u << (cntl & 0x3f);           /* dwords */
    uint32_t wptr = s->regfile[CP_RB_WPTR / 4] & (size - 1);
    uint32_t rptr = s->cp.rptr & (size - 1);
    uint64_t base = s->regfile[CP_RB_BASE / 4];
    bool swap = (cntl >> 16) & 3;
    unsigned n, done;
    uint32_t *buf;

    if (size > (1u << 22) || rptr == wptr) {
        return;
    }
    n = (wptr - rptr) & (size - 1);
    buf = g_new(uint32_t, n);
    if (rptr + n <= size) {
        ati_cp_fetch(s, base + rptr * 4, buf, n, swap);
    } else {
        unsigned first = size - rptr;
        ati_cp_fetch(s, base + rptr * 4, buf, first, swap);
        ati_cp_fetch(s, base, buf + first, n - first, swap);
    }
    trace_ati_cp_ring(rptr, wptr, n);
    s->accel.ring_dw += n;
    done = ati_cp_run(s, buf, n);
    g_free(buf);
    s->cp.rptr = (rptr + done) & (size - 1);
    if (!(cntl & BIT(27))) {            /* RB_NO_UPDATE */
        uint32_t addr = s->regfile[CP_RB_RPTR_ADDR / 4];
        uint32_t v = s->cp.rptr;
        uint32_t out = (addr & 3) == 2 ? cpu_to_be32(v) : cpu_to_le32(v);
        ati_mc_rw(s, addr & ~3u, &out, 4, true);
    }
}

/*
 * PIO queues (CP_CSQ_APER_PRIMARY / INDIRECT).  The aperture is a FIFO port:
 * the address within it does not order anything (Mac OS's ATI queue server
 * writes every indirect-buffer packet at offset 0 and every timestamp packet
 * at offset 12).  Packets run as soon as they are complete.
 */
static void ati_cp_pio_write(ATIVGAState *s, int q, unsigned slot, uint32_t dw)
{
    ATICPQueue *pq = &s->cp.pio[q];
    unsigned done;

    (void)slot;
    s->accel.pio_dw++;
    if (pq->n >= ARRAY_SIZE(pq->buf)) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: PIO packet too long, dropped\n");
        pq->n = 0;
    }
    pq->buf[pq->n++] = dw;
    done = ati_cp_run(s, pq->buf, pq->n);
    if (done) {
        memmove(pq->buf, pq->buf + done, (pq->n - done) * 4);
        pq->n -= done;
    }
}

/* ---------------------------------------------------------------------
 * Register interface
 */

bool ati_cp_write(ATIVGAState *s, hwaddr addr, uint32_t data)
{
    switch (addr) {
    case CP_RB_BASE:
        s->cp.rptr = 0;
        return true;
    case CP_RB_RPTR_WR:
        s->cp.rptr = data;
        return true;
    case CP_RB_WPTR:
        ati_cp_ring_kick(s);
        return true;
    case CP_IB_BUFSZ:
        ati_cp_exec_ib(s, s->regfile[CP_IB_BASE / 4], data & 0x7fffff);
        return true;
    case CP_ME_RAM_ADDR:
        s->cp.me_addr = data & 0xff;
        return true;
    case CP_ME_RAM_RADDR:
        s->cp.me_addr = data & 0xff;
        return true;
    case CP_ME_RAM_DATAH:
        s->cp.me_ram[s->cp.me_addr].hi = data;
        return true;
    case CP_ME_RAM_DATAL:
        s->cp.me_ram[s->cp.me_addr].lo = data;
        s->cp.me_addr = (s->cp.me_addr + 1) & 0xff;
        return true;
    case CP_CSQ_APER_PRIMARY ... CP_CSQ_APER_PRIMARY + 0x1fc:
        ati_cp_pio_write(s, 0, (addr - CP_CSQ_APER_PRIMARY) / 4, data);
        return true;
    case CP_CSQ_APER_INDIRECT ... CP_CSQ_APER_INDIRECT + 0xfc:
        ati_cp_pio_write(s, 1, (addr - CP_CSQ_APER_INDIRECT) / 4, data);
        return true;
    case SCRATCH_REG0 ... SCRATCH_REG5:
    {
        int i = (addr - SCRATCH_REG0) / 4;
        if (s->regfile[SCRATCH_UMSK / 4] & BIT(i)) {
            uint32_t out = cpu_to_le32(data);
            ati_mc_rw(s, s->regfile[SCRATCH_ADDR / 4] + 4 * i, &out, 4, true);
        }
        return true;
    }
    default:
        return false;
    }
}

bool ati_cp_read(ATIVGAState *s, hwaddr addr, uint32_t *val)
{
    switch (addr) {
    case CP_RB_RPTR:
        *val = s->cp.rptr;
        return true;
    case CP_ME_RAM_DATAH:
        *val = s->cp.me_ram[s->cp.me_addr].hi;
        return true;
    case CP_ME_RAM_DATAL:
        *val = s->cp.me_ram[s->cp.me_addr].lo;
        s->cp.me_addr = (s->cp.me_addr + 1) & 0xff;
        return true;
    case CP_CSQ_CNTL:
        /*
         * CSQ_CNT_PRIMARY (bits 7:0) counts free primary-queue entries; the
         * PIO queues drain immediately here, so they are always empty.
         */
        *val = (s->regfile[CP_CSQ_CNTL / 4] & 0xf0000000) | 0xc0;
        return true;
    case CP_CSQ_STAT:
        /* queues always drained: read pointers equal write pointers */
        *val = 0;
        return true;
    case CP_STAT:
        *val = 0;
        return true;
    default:
        return false;
    }
}

void ati_cp_reset(ATIVGAState *s)
{
    memset(&s->cp, 0, sizeof(s->cp));
}
