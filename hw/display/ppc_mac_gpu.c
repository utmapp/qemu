/*
 * PPC Mac GPU - ATI Radeon 9200 (RV280) compatible display device for QEMU
 *
 * Emulates enough of the RV280 register interface for Mac OS X Tiger/Leopard
 * ATI kext attachment and accelerated framebuffer operation on PowerMac G4.
 *
 * Strategy: Radeon compatibility path. We present PCI IDs that match the
 * Radeon 9200 PRO (RV280), which Tiger's ATIRadeon9700.kext will recognize
 * and attach to. We emulate only the registers the driver actually touches.
 *
 * Host targets: Intel macOS + Apple Silicon macOS (both via Metal or SW).
 * Guest targets: Mac OS X Tiger 10.4.11 PPC, Leopard 10.5 PPC.
 *
 * This file contains the PCI device model, MMIO register handlers,
 * VRAM management, CRTC mode tracking, interrupt generation, and
 * display surface hookup. All code is pure C with no platform dependencies.
 *
 * Copyright (c) 2024
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "r300/r300_state.h"
#include "r300/r300_draw.h"
#include "ui/poweremu-harmony.h"
#include <math.h>
#include <sched.h>
#include "qemu/log.h"
#include "qemu/host-utils.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "qemu/timer.h"
#include "qemu/datadir.h"
#include "qemu/rcu.h"
#include "qemu/main-loop.h"
#include "hw/pci/pci_device.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "exec/address-spaces.h"
#include "hw/display/ppc_mac_gpu.h"
#include "ppc_mac_gpu_3d_regs.h"
#include "ppc_mac_gpu_renderer.h"
#include "ppc_mac_gpu_surface.h"
#include "ui/console.h"
#include "ui/qemu-pixman.h"
#include "system/system.h"
#include "hw/display/ppc_mac_gpu_vp.h"
#include "trace.h"

/* Device writes to VRAM: dirty for the display, and the end of any
 * card-order colour buffer record they overlap (see r300_e0_note). */
static void r300_e0_forget(uint64_t base, uint64_t len);
#define ppc_vram_dirty_e0(s, off, len) do {                     \
        uint64_t off_ = (off), len_ = (len);                    \
        memory_region_set_dirty(&(s)->vram, off_, len_);        \
        r300_e0_forget(off_, len_);                             \
    } while (0)

/* UniNorth AGP bridge GART base — defined in hw/pci-host/uninorth.c */
hwaddr uninorth_get_agp_gart_base(void);
uint32_t uninorth_get_agp_gart_gen(void);

/* ========================================================================
 * Blit path instrumentation — tracks which engine fires and when
 * ======================================================================== */

typedef struct BlitPathStats {
    uint64_t sep_count;        /* DST_HEIGHT_WIDTH (0x143C) trigger */
    uint64_t mmio_count;       /* DST_WIDTH_HEIGHT (0x1598) trigger */
    uint64_t bbm_count;        /* PM4 BITBLT_MULTI (0x99) */
    uint64_t paint_count;      /* PM4 PAINT_MULTI (0x9a) */
    uint64_t hostdata_count;   /* PM4 HOSTDATA_BLT (0x9c) */
    uint64_t last_dump;        /* last count at which we dumped */
} BlitPathStats;

static BlitPathStats g_blit_stats;

/*
 * Bring-up diagnostics.  These were written while the GPU model was being
 * reverse-engineered and several of them do real per-draw work (CRC scans,
 * VRAM probes, register audits), so they stay off unless asked for.
 */
static bool gpu_diag_on(void)
{
    static int on = -1;

    if (on < 0) {
        on = getenv("PPCGPU_DIAG") != NULL;
    }
    return on;
}

/* ========================================================================
 * Framebuffer write watch — Phase A drag body audit
 * Logs every non-SRT write to the visible framebuffer.
 * ======================================================================== */
static int g_fb_watch_count = 0;
static void fb_write_watch(const char *path, const char *src_kind,
                           uint32_t src_off, uint32_t dst_off,
                           uint32_t dst_x, uint32_t dst_y,
                           uint32_t w, uint32_t h,
                           bool from_srt, bool nonzero_sample)
{
    if (!gpu_diag_on()) return;
    /* Only log the first 500 events to keep it manageable */
    if (g_fb_watch_count >= 500) return;
    /* Only log framebuffer writes (dst near offset 0) */
    if (dst_off > 0x10000) return;
    /* Skip tiny writes (< 4x4) */
    if (w < 4 || h < 4) return;

    g_fb_watch_count++;
    fprintf(stderr,
        "[FB_WRITE_WATCH] #%d path=%s src_off=0x%x src_kind=%s "
        "dst_xy=(%u,%u) size=(%ux%u) from_srt=%d nz=%d\n",
        g_fb_watch_count, path, src_off, src_kind,
        dst_x, dst_y, w, h, from_srt, nonzero_sample);
}

/* Set by the Metal backend when it tracks frames; NULL otherwise. */
FrameTracker *g_frame_tracker;

static void frame_tracker_record_2d_event(PassEventType type,
                                          uint32_t surface_offset,
                                          uint32_t pitch_bytes,
                                          uint32_t src_offset,
                                          uint32_t dst_offset,
                                          uint32_t src_x,
                                          uint32_t src_y,
                                          uint32_t dst_x,
                                          uint32_t dst_y,
                                          uint32_t w,
                                          uint32_t h,
                                          bool from_srt)
{
    if (!g_frame_tracker) {
        return;
    }

    frame_tracker_record_detail(g_frame_tracker, type,
                                surface_offset, pitch_bytes / 4,
                                src_offset, dst_offset,
                                (uint16_t)w, (uint16_t)h,
                                (uint16_t)src_x, (uint16_t)src_y,
                                (uint16_t)dst_x, (uint16_t)dst_y,
                                0, 0, from_srt);

    if ((type == PASS_EVENT_PRESENT || type == PASS_EVENT_FALLBACK_2D) &&
        dst_offset == 0 && w >= 800 && h >= 400) {
        frame_tracker_dump(g_frame_tracker);
        frame_tracker_new_frame(g_frame_tracker);
    }
}

/* Global BLIT region table — maps RT regions to screen positions.
 * Populated by BITBLT_MULTI, consumed by sw_draw_3d for coordinate mapping. */
BlitRegionTable g_blit_region_table;

/*
 * Record a BLIT region into the global table.
 * Called from BITBLT_MULTI handler for each sub-blit.
 */
static void blit_region_record(uint32_t src_offset, uint32_t src_pitch,
                                uint32_t src_x, uint32_t src_y,
                                uint32_t dst_x, uint32_t dst_y,
                                uint32_t w, uint32_t h)
{
    BlitRegionTable *t = &g_blit_region_table;

    /* Find or create the RT entry */
    BlitRegionRT *rt_entry = NULL;
    for (int i = 0; i < t->rt_count; i++) {
        if (t->rts[i].rt_offset == src_offset &&
            t->rts[i].rt_pitch == src_pitch) {
            rt_entry = &t->rts[i];
            break;
        }
    }
    if (!rt_entry) {
        if (t->rt_count >= BLIT_REGION_RT_MAX) {
            /* Table full — evict oldest (slot 0) */
            memmove(&t->rts[0], &t->rts[1],
                    (BLIT_REGION_RT_MAX - 1) * sizeof(BlitRegionRT));
            t->rt_count = BLIT_REGION_RT_MAX - 1;
        }
        rt_entry = &t->rts[t->rt_count++];
        rt_entry->rt_offset = src_offset;
        rt_entry->rt_pitch = src_pitch;
        rt_entry->region_count = 0;
    }

    /* Check if this region already exists (same src/dst/size) — update it */
    for (int i = 0; i < rt_entry->region_count; i++) {
        BlitRegion *r = &rt_entry->regions[i];
        if (r->src_x == src_x && r->src_y == src_y &&
            r->dst_x == dst_x && r->dst_y == dst_y &&
            r->width == w && r->height == h) {
            r->sequence = t->sequence++;
            return;  /* duplicate — just update sequence */
        }
    }

    /* Add new region */
    if (rt_entry->region_count >= BLIT_REGION_MAX_PER_RT) {
        /* Evict oldest region */
        uint32_t oldest_seq = UINT32_MAX;
        int oldest_idx = 0;
        for (int i = 0; i < rt_entry->region_count; i++) {
            if (rt_entry->regions[i].sequence < oldest_seq) {
                oldest_seq = rt_entry->regions[i].sequence;
                oldest_idx = i;
            }
        }
        rt_entry->regions[oldest_idx] = (BlitRegion){
            .src_x = src_x, .src_y = src_y,
            .dst_x = dst_x, .dst_y = dst_y,
            .width = w, .height = h,
            .sequence = t->sequence++,
        };
    } else {
        rt_entry->regions[rt_entry->region_count++] = (BlitRegion){
            .src_x = src_x, .src_y = src_y,
            .dst_x = dst_x, .dst_y = dst_y,
            .width = w, .height = h,
            .sequence = t->sequence++,
        };
    }
}

/*
 * Unified, ordered log of every rectangle written to VRAM by any 2D path.
 *
 * The per-path logs (gpu_blit_paths.log, gpu_blit_diag.txt) carry separate
 * counters and different fields, so a save and the restore that consumes it
 * cannot be paired.  Here every executed rectangle gets one line and one
 * global sequence number, tagged with its path and whether it arrived
 * through the PM4 ring or a direct CPU register write.
 *
 * Opened O_APPEND and line-buffered, so the host can truncate it while the
 * VM runs (`: > /tmp/gpu_seq.log`) to start a clean capture.
 */
static FILE *g_seq_log;
static uint64_t g_seq;
static int g_in_pm4;            /* > 0 while executing PM4 packets */

static bool r200_direct_enabled(void);

/* Finish batched direct-R200 rendering before VRAM is read on the CPU side. */
static void r200_fence_drain(PPCMacGPUState *s, uint32_t done);
/* Why batches get flushed: histogram logged with the renderer stats. */
static uint64_t r200_flush_why[0x10000 / 4 + 8];
static uint64_t r200_flush_total;
static struct {
    uint64_t presents, draws, ops2d, flushes, flush_us, flips;
    uint64_t gpu_vs;         /* R300 draws whose vertex program ran on the GPU */
    uint64_t co_copies, co_partials, co_skips, co_defers;  /* scanout path */
    int64_t since;
} r200_rate;

/*
 * Scanout-path switches, all default-on, each settable to 0 to get the
 * pre-optimisation behaviour back for A/B testing:
 *   PPCGPU_DIRTY_SCANOUT  consume the VRAM dirty log to copy and publish
 *                         only the rows that changed
 *   PPCGPU_FLIP_KICK      refresh the UI on a page flip instead of waiting
 *                         for the polling tick
 *   PPCGPU_ASYNC_PRESENT  hand in-flight GPU work its completion callback
 *                         instead of draining it on the UI thread
 *   PPCGPU_DIAG           per-frame diagnostic VRAM scans (default off)
 */
static bool ppc_gpu_env_on(const char *name)
{
    const char *v = getenv(name);

    return !v || strcmp(v, "0") != 0;
}

static bool ppc_gpu_env_diag(void)
{
    static int cached = -1;

    if (cached < 0) {
        cached = getenv("PPCGPU_DIAG") != NULL;
    }
    return cached;
}

/*
 * Running totals for PowerEmu's performance overlay (the "perf" property):
 * a frame is a flip or a write into the scanout that follows 3D drawing;
 * textures are counted per draw by where they are read from; vram_high is
 * the highest VRAM byte any render target, depth buffer or texture used.
 */
static struct {
    uint64_t frames, draws, tex_vram, tex_agp, agp_bytes, vram_high;
    bool drew;
} r200_perf;

static void r200_perf_present(void)
{
    if (r200_perf.drew) {
        r200_perf.frames++;
        r200_perf.drew = false;
    }
}

static void r200_perf_high(uint64_t end)
{
    if (end > r200_perf.vram_high) {
        r200_perf.vram_high = end;
    }
}

/*
 * Draws on a worker thread.
 * ------------------------
 * A host sample put 16.7% of the vCPU thread in this device model, with the
 * draw path the largest named function in the whole profile -- while the
 * machine around it sat with most of its cores idle. The emulator is one
 * thread per guest CPU and the guest has one CPU, so the only way to use
 * those cores is to take work out of that thread.
 *
 * What makes it safe is a contract the guest already keeps. Real hardware
 * reads vertex memory asynchronously, so a driver must not reuse a buffer
 * until a fence says the GPU is done with it -- and this device already
 * implements those fences. Reading guest memory from a worker is therefore
 * not a new hazard; it is the hazard the interface was designed around.
 *
 * What is *not* safe is reading registers, because the guest keeps writing
 * them while the worker runs. Each job therefore carries a snapshot. The
 * copy is the whole device state by value: the fields a draw reads besides
 * the registers are three pointers and a size, and copying them along with
 * the registers is cheaper than auditing which ones a future change might
 * add.
 *
 * Ordering rides on r200_flush_at(), which everything that needs to see
 * finished drawing already calls. Draining there keeps 2D blits, scanout,
 * fences and register reads in the order the guest asked for, without any
 * new ordering rules to get wrong.
 *
 * Off by default: PPCGPU_ASYNC_DRAW=1 turns it on. A threading fault in a
 * device model is the kind that corrupts rarely rather than failing
 * cleanly, so this does not become the default until it has been measured
 * and lived with.
 */
typedef struct R200DrawJob {
    PPCMacGPUState state;       /* snapshot: registers by value */
    uint32_t *body;             /* the PM4 body, copied */
    uint32_t body_dw;
    int src;
    struct R200DrawJob *next;
} R200DrawJob;

static struct {
    QemuThread thread;
    QemuMutex lock;
    QemuCond wake;              /* work arrived, or stop */
    QemuCond idle;              /* queue drained */
    R200DrawJob *head, *tail;
    unsigned inflight;          /* queued + currently running */
    bool started;
    bool stop;
} r200_async;

/* The command processor thread (see "The command processor on its own
 * thread" below).  Metal encoding runs on the renderer's render queue
 * thread, off the BQL, so a slice is pure ring parsing (register decode
 * and vertex fetch); the yield only exists so a catch-up can take the
 * ring over, and rarely fires. */
#define CP_SLICE_US 2000

static struct {
    PPCMacGPUState *s;
    QemuThread thread;
    QemuEvent kick;
    bool started;
    bool stop;
    unsigned epoch;             /* bumped whenever the ring changes hands */
    bool aborted;               /* the replay was taken over mid-segment */
    uint32_t seg_rptr;          /* where the segment being replayed began */
    uint32_t ring_mask;
    unsigned pkts;
    int64_t yield_at;
    /* Statistics (BQL): catch-ups the guest forced, by register. */
    uint32_t syncs, kicks;
    uint16_t sync_reg[0x10000 / 4];
} cp;

static bool r200_async_enabled(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("PPCGPU_ASYNC_DRAW");
        on = e && e[0] == '1';
    }
    return on;
}

/*
 * Submit to the renderer.
 *
 * A queue-async renderer (the Metal render queue) only enqueues the draw
 * and encodes it on its own thread, so no lock is needed here: the
 * vertices and textures are copied into the job, and every flush, fence
 * and CPU-side VRAM access drains the queue first.
 *
 * Every other caller of the renderer runs on the vCPU thread or the main
 * loop, and those are already serialised against each other by the BQL --
 * which is exactly why this device could call Metal freely before draws
 * moved off-thread, and why Metal asserted the moment they did.  For a
 * synchronous backend, taking the same lock here restores that guarantee
 * without touching the thirty other call sites, and without serialising
 * the part worth parallelising: by the time a draw reaches this point its
 * vertices are already fetched and transformed, which is the expensive
 * half.
 */
static int r200_render_draw(PPCMacGPUState *s, uint8_t *vram,
                            const R200DrawPacket *pkt)
{
    bool need_bql = !s->renderer->draw_queue_async && !bql_locked();
    int r;

    if (need_bql) {
        bql_lock();
    }
    r = s->renderer->draw_r200(s->renderer_opaque, vram, s->vram_size, pkt);
    if (need_bql) {
        bql_unlock();
    }
    return r;
}

/* Forward declaration: the worker runs the ordinary draw path. */
static bool ppc_mac_gpu_r200_draw_now(PPCMacGPUState *s, const uint32_t *d,
                                      uint32_t body_dw, int src);

static void *r200_async_worker(void *opaque)
{
    qemu_mutex_lock(&r200_async.lock);
    for (;;) {
        R200DrawJob *job;

        while (!r200_async.head && !r200_async.stop) {
            qemu_cond_wait(&r200_async.wake, &r200_async.lock);
        }
        if (!r200_async.head && r200_async.stop) {
            break;
        }
        job = r200_async.head;
        r200_async.head = job->next;
        if (!r200_async.head) {
            r200_async.tail = NULL;
        }
        /*
         * Unlocked while drawing: this is the whole point, and the job owns
         * everything it touches. inflight is not decremented until the draw
         * is finished, so a drain waits for the work and not merely for the
         * queue to empty.
         */
        qemu_mutex_unlock(&r200_async.lock);
        ppc_mac_gpu_r200_draw_now(&job->state, job->body, job->body_dw,
                                  job->src);
        g_free(job->body);
        g_free(job->state.draw_verts);
        g_free(job->state.draw_idx);
        g_free(job);
        qemu_mutex_lock(&r200_async.lock);
        if (--r200_async.inflight == 0) {
            qemu_cond_broadcast(&r200_async.idle);
        }
    }
    qemu_mutex_unlock(&r200_async.lock);
    return NULL;
}

/* Wait until every queued draw has finished. */
static void r200_async_drain(void)
{
    if (!r200_async.started) {
        return;
    }
    qemu_mutex_lock(&r200_async.lock);
    while (r200_async.inflight) {
        qemu_cond_wait(&r200_async.idle, &r200_async.lock);
    }
    qemu_mutex_unlock(&r200_async.lock);
}

static bool r200_async_submit(PPCMacGPUState *s, const uint32_t *d,
                              uint32_t body_dw, int src)
{
    R200DrawJob *job;

    if (!r200_async_enabled()) {
        return false;
    }
    if (!r200_async.started) {
        qemu_mutex_init(&r200_async.lock);
        qemu_cond_init(&r200_async.wake);
        qemu_cond_init(&r200_async.idle);
        r200_async.started = true;
        qemu_thread_create(&r200_async.thread, "r200-draw",
                           r200_async_worker, NULL, QEMU_THREAD_JOINABLE);
    }

    job = g_new(R200DrawJob, 1);
    job->state = *s;                    /* registers and pointers, by value */
    /*
     * The scratch buffers are the one thing that must not be shared. A
     * snapshot copies their pointers, and the worker growing one would
     * g_renew memory the device still points at -- which surfaces much
     * later, as a heap abort inside an unrelated allocation. Give the job
     * its own, allocated on first use and freed with the job.
     */
    job->state.draw_verts = NULL;
    job->state.draw_verts_cap = 0;
    job->state.draw_idx = NULL;
    job->state.draw_idx_cap = 0;
    job->body_dw = body_dw;
    job->src = src;
    job->next = NULL;
    job->body = g_memdup2(d, (size_t)body_dw * 4);

    qemu_mutex_lock(&r200_async.lock);
    r200_async.inflight++;
    if (r200_async.tail) {
        r200_async.tail->next = job;
    } else {
        r200_async.head = job;
    }
    r200_async.tail = job;
    qemu_cond_signal(&r200_async.wake);
    qemu_mutex_unlock(&r200_async.lock);
    return true;
}

/* Register-access flushes are tagged by register; others by a tag below. */
#define R200_WHY_REG(a)   ((uint32_t)(a) / 4)
#define R200_WHY_TAG(t)   (0x4000u + (t))
enum { R200_WHY_T3 = 1, R200_WHY_DISPLAY, R200_WHY_FENCE, R200_WHY_OTHER };

static void ppc_mac_gpu_frame_ready(PPCMacGPUState *s);
static void ppc_mac_gpu_hwc_define(PPCMacGPUState *s);
static void ppc_mac_gpu_hwc_move(PPCMacGPUState *s);

static void r200_flush_at(PPCMacGPUState *s, uint32_t why)
{
    /*
     * Everything that must see finished drawing comes through here, so this
     * is the only place that has to wait for the workers -- no new ordering
     * rules, and none to forget at a new call site.
     */
    r200_async_drain();
    if (s->renderer && s->renderer->flush_r200) {
        int64_t t0 = g_get_monotonic_time();
        bool did = s->renderer->flush_r200(s->renderer_opaque);
        if (did) {
            r200_rate.flushes++;
            r200_rate.flush_us += g_get_monotonic_time() - t0;
        }
        /* Drawing finished: get it on screen soon.  The display's own
         * flush (below the refresh) must not re-arm the kick, or an idle
         * desktop would refresh at 60 Hz forever. */
        if (did && why != R200_WHY_TAG(R200_WHY_DISPLAY)) {
            ppc_mac_gpu_frame_ready(s);
        }
        if (did) {
            r200_flush_why[MIN(why, (uint32_t)ARRAY_SIZE(r200_flush_why) - 1)]++;
            if (++r200_flush_total % 1000 == 0) {
                GString *g = g_string_new("ppc-mac-gpu r200: flush causes:");
                for (int k = 0; k < 5; k++) {
                    uint32_t best = 0;
                    for (uint32_t i = 1; i < ARRAY_SIZE(r200_flush_why); i++) {
                        if (r200_flush_why[i] > r200_flush_why[best]) {
                            best = i;
                        }
                    }
                    if (!r200_flush_why[best]) {
                        break;
                    }
                    g_string_append_printf(g, " %s0x%x=%llu",
                        best >= 0x4000 ? "tag" : "reg", best >= 0x4000 ?
                        best - 0x4000 : best * 4,
                        (unsigned long long)r200_flush_why[best]);
                    r200_flush_why[best] = 0;
                }
                qemu_log("%s\n", g->str);
                g_string_free(g, TRUE);
                memset(r200_flush_why, 0, sizeof(r200_flush_why));
            }
        }
        if (s->regs.r200_fence_n) {
            r200_fence_drain(s, s->regs.r200_fence_last_seq);
        }
    }
}
#define r200_flush(s) r200_flush_at((s), R200_WHY_TAG(R200_WHY_OTHER))
enum { R200_WHY_2D = 0x40 };

/*
 * A CPU-side 2D operation is about to read (write_access=false) or write
 * VRAM rows [lo, hi).  Flush batched 3D work only if it overlaps: work in
 * flight that neither reads nor writes these bytes can keep running.
 */
static void r200_vram_access(PPCMacGPUState *s, uint64_t lo, uint64_t hi,
                             bool write_access, uint32_t tag)
{
    if (write_access && tag <= 5) {
        r200_rate.ops2d++;
        if (lo < (uint64_t)s->regs.crtc_offset + (uint64_t)s->disp.stride * s->disp.height &&
            hi > s->regs.crtc_offset) {
            r200_rate.presents++;          /* 2D write into the scanout */
            r200_perf_present();
        }
    }
    if (!s->renderer || !s->renderer->range_busy_r200 || hi <= lo) {
        return;
    }
    if (s->renderer->range_busy_r200(s->renderer_opaque, lo, hi, write_access)) {
        r200_flush_at(s, R200_WHY_TAG(R200_WHY_2D + tag));
    }
}
#define R200_ROWS(off, pitch, y, h) \
    (uint64_t)(off) + (uint64_t)(y) * (pitch), \
    (uint64_t)(off) + ((uint64_t)(y) + (h)) * (pitch)

/*
 * Fill len bytes at dst with a repeating 2/4-byte little-endian value:
 * write the pattern once and double the filled span with memcpy, which
 * for a megabyte-scale clear beats a per-pixel store loop by an order
 * of magnitude.
 */
static void r200_pattern_fill(uint8_t *dst, uint64_t len, uint32_t v,
                              uint32_t bpp)
{
    uint8_t pat[4];
    uint64_t done = bpp;

    if (!len) {
        return;
    }
    if (bpp == 4) {
        stl_le_p(pat, v);
    } else {
        stw_le_p(pat, v);
    }
    memcpy(dst, pat, bpp);
    while (done < len) {
        uint64_t n = MIN(done, len - done);
        memcpy(dst + done, dst, n);
        done += n;
    }
}

/* -1 until asked; reset to -1 when the trace property changes it, so a log
 * can be turned on in a machine that is already running. */
static int g_seq_log_enabled = -1;


/* ========================================================================
 * Seeing the guest's windows
 *
 * Harmony mode -- showing a virtual Mac's windows on this Mac's desktop
 * rather than its whole screen -- needs to know what the windows are.  The
 * compositor never says; what it does is copy each window's contents to the
 * screen, piece by piece, every time anything changes.  A window therefore
 * shows up as a run of copies that share a source surface, and its frame is
 * the rectangle those copies cover.
 *
 * This watches those copies and keeps a list.  It is only a reader: nothing
 * here changes what is drawn.  PPCGPU_WINDOWS=1 turns it on, and the list
 * is printed whenever it settles.
 *
 * A window is redrawn in bursts: a run of copies close together, then
 * nothing until something changes again.  The frame is the rectangle one
 * burst covers, and each new burst starts a new rectangle -- otherwise a
 * window that moves reports the union of everywhere it has ever been,
 * which is no use to anything that wants to draw it.
 *
 * Known limits, measured on 10.4 and 10.5 alike: a surface is sometimes
 * reused for more than one window (the same address turns up with two
 * different scanline lengths), so a surface is not an identity.  Grouping
 * by surface *and* scanline length is closer, and the guest agent will have
 * to supply the real identity in the end.
 * ======================================================================== */
#define PE_WINDOW_MAX 32
/*
 * Long enough that one redraw of a window stays one burst even when the
 * guest is slow, short enough that dragging a window reports where it is
 * rather than where it has been.  A drag redraws far faster than this.
 */
#define PE_WINDOW_BURST_US (200 * 1000)
typedef struct {
    uint32_t surface, pitch;
    uint32_t x0, y0, x1, y1;        /* the rectangle this burst covers */
    uint32_t fx0, fy0, fx1, fy1;    /* the last burst that finished */
    uint32_t scr_w, scr_h;          /* the screen it was copied onto */
    uint64_t pieces;
    int64_t last_us;
    bool live, settled;
} PEWindow;

static PEWindow pe_windows[PE_WINDOW_MAX];
static int64_t pe_windows_printed_us;

/*
 * What a run of copies is, told from the copies alone.
 *
 * Nothing in the command stream says "this is the desktop" -- but the
 * desktop is the only thing drawn the full width of the screen and down
 * to the bottom of it, and the menu bar is the only thing drawn the full
 * width and a couple of dozen pixels tall.  Everything else that reaches
 * the screen is a window, or part of one.
 *
 * This replaces asking where a surface lives in video memory, which was
 * how the old surface classifier worked.  Addresses are no use: they
 * differ between Tiger and Leopard, and between one start and the next --
 * the same Finder window came back at 0x9b6000, 0x9da000 and 0x16ac000 on
 * three runs of the same machine.  What the guest *does* with a surface
 * is the same on both systems.
 */
static PEHarmonyArea pe_area_of(uint32_t x0, uint32_t y0,
                                  uint32_t x1, uint32_t y1,
                                  uint32_t scr_w, uint32_t scr_h)
{
    uint32_t w = x1 - x0, h = y1 - y0;

    if (!scr_w || !scr_h) {
        return PE_AREA_UNKNOWN;
    }
    if (x0 == 0 && x1 >= scr_w) {
        if (y1 >= scr_h) {
            return PE_AREA_DESKTOP;    /* full width, down to the bottom */
        }
        /*
         * The menu bar is 22 points on both systems, but the copy that
         * reaches the screen carries its shadow: 36 pixels, measured on
         * 10.5.  Anything full-width and this shallow at the top of the
         * screen is the menu bar and nothing else.
         */
        if (y0 == 0 && h <= 48) {
            return PE_AREA_MENUBAR;
        }
    }
    if (w >= scr_w && h >= scr_h) {
        return PE_AREA_DESKTOP;
    }
    return PE_AREA_WINDOW;
}

static const char *pe_area_name(PEHarmonyArea a)
{
    switch (a) {
    case PE_AREA_DESKTOP: return "desktop";
    case PE_AREA_MENUBAR: return "menubar";
    case PE_AREA_WINDOW:  return "window";
    default:              return "unknown";
    }
}

static uint32_t pe_desktop_surface;       /* whose copy last covered the screen */
static int64_t pe_desktop_surface_us;     /* and when */

/*
 * The fill-ins arrive with the desktop, not minutes later.  Two seconds is
 * long enough to cover one repaint and short enough that the surface being
 * handed to a window afterwards -- which happens; the same address drew the
 * wallpaper in one run and a Finder window in the next -- does not make that
 * window vanish.
 */
#define PE_DESKTOP_SURFACE_US (2 * G_TIME_SPAN_SECOND)

/* What a copy out of `surface` covering this rectangle is. */
static PEHarmonyArea pe_area_of_copy(uint32_t surface,
                                       uint32_t x0, uint32_t y0,
                                       uint32_t x1, uint32_t y1,
                                       uint32_t scr_w, uint32_t scr_h)
{
    PEHarmonyArea a = pe_area_of(x0, y0, x1, y1, scr_w, scr_h);
    int64_t now = g_get_monotonic_time();

    if (a == PE_AREA_DESKTOP) {
        pe_desktop_surface = surface;
        pe_desktop_surface_us = now;
    } else if (a == PE_AREA_WINDOW && surface == pe_desktop_surface &&
               now - pe_desktop_surface_us < PE_DESKTOP_SURFACE_US) {
        /*
         * Part of the desktop repaint that is going on right now.  When the
         * screen changes size the compositor fills the newly uncovered
         * strips from the wallpaper's own surface -- (1024,0) 656x768 and
         * (0,768) 1680x282, going from 1024x768 to 1680x1050 -- and by
         * shape alone those are windows.  Harmony mode would leave two
         * slabs of wallpaper lying on this Mac's desktop.
         */
        a = PE_AREA_DESKTOP;
    }
    return a;
}

static const char *pe_window_kind(const PEWindow *e)
{
    return pe_area_name(pe_area_of_copy(e->surface, e->x0, e->y0, e->x1, e->y1,
                                        e->scr_w, e->scr_h));
}

/* ---- the tile grid ---------------------------------------------------- */

static bool pe_harm_on;
static uint8_t *pe_harm_tiles;
static int pe_harm_cols, pe_harm_rows;
static uint32_t pe_harm_gen;
static int64_t pe_harm_resized_us;

/*
 * The grid is kept whether harmony mode is on or not, because it has to
 * be right the instant it is switched on: it is a shadow of the screen,
 * saying what last covered each part of it, and there is no way to work
 * that out after the fact.  A window nobody has touched for ten minutes is
 * still on the screen, and a menu that was closed an hour ago is not.
 * Keeping it costs a few bytes written per copy.
 */
void ppc_mac_gpu_harmony_enable(bool on)
{
    pe_harm_on = on;
    pe_harm_gen++;
}

bool ppc_mac_gpu_harmony_tiles(const uint8_t **tiles, int *cols, int *rows,
                                 uint32_t *generation)
{
    if (!pe_harm_on || !pe_harm_tiles) {
        return false;
    }
    *tiles = pe_harm_tiles;
    *cols = pe_harm_cols;
    *rows = pe_harm_rows;
    *generation = pe_harm_gen;
    return true;
}

/*
 * Remember what covered each tile.  A window marks every tile it touches,
 * a desktop or menu bar only the tiles it covers completely: the bias is
 * towards showing a little too much of the guest rather than cutting the
 * edge off one of its windows.
 */
static void pe_harmony_mark(uint32_t surface,
                              uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                              uint32_t scr_w, uint32_t scr_h)
{
    if (!scr_w || !scr_h) {
        return;
    }
    int cols = (scr_w + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
    int rows = (scr_h + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
    if (cols != pe_harm_cols || rows != pe_harm_rows) {
        g_free(pe_harm_tiles);
        pe_harm_tiles = g_malloc0((size_t)cols * rows);
        pe_harm_cols = cols;
        pe_harm_rows = rows;
        pe_harm_resized_us = g_get_monotonic_time();
    }

    /*
     * Changing resolution, the compositor fills the newly uncovered strips
     * from the wallpaper's own surface -- (1024,0) 656x768 and (0,768)
     * 1680x282, going from 1024x768 to 1680x1050 -- before it puts up a
     * whole screen.  By shape those are windows, and taking them for
     * windows leaves a slab of wallpaper lying on this Mac's desktop for
     * the rest of the session.  Nothing drawn in the moment after a resize
     * is worth believing; whatever is really there is drawn again directly
     * afterwards.
     */
    int64_t now = g_get_monotonic_time();
    bool settling = now - pe_harm_resized_us < G_TIME_SPAN_SECOND * 3 / 2;
    if (settling && ((uint64_t)w * h * 10 < (uint64_t)scr_w * scr_h * 9)) {
        return;
    }

    PEHarmonyArea a = pe_area_of_copy(surface, x, y, x + w, y + h,
                                        scr_w, scr_h);
    int tx0, ty0, tx1, ty1;
    if (a == PE_AREA_WINDOW) {
        tx0 = x / PE_HARMONY_TILE;
        ty0 = y / PE_HARMONY_TILE;
        tx1 = (x + w + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
        ty1 = (y + h + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
    } else {
        tx0 = (x + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
        ty0 = (y + PE_HARMONY_TILE - 1) / PE_HARMONY_TILE;
        tx1 = (x + w) / PE_HARMONY_TILE;
        ty1 = (y + h) / PE_HARMONY_TILE;
    }
    tx1 = MIN(tx1, cols);
    ty1 = MIN(ty1, rows);

    /*
     * A copy that covers nearly the whole screen is the compositor putting
     * the finished picture up -- wallpaper and windows together -- not the
     * wallpaper alone.  It says nothing about which part is which, so it
     * only fills in ground nothing has been seen on yet; taking it for
     * desktop everywhere wipes out every window that has been found so far,
     * and they come back only as they happen to be redrawn.  Measured: the
     * menu bar's clock reappeared within the minute and the rest of the
     * screen stayed empty.
     */
    bool whole_screen = (uint64_t)w * h * 10 >= (uint64_t)scr_w * scr_h * 9;

    for (int ty = ty0; ty < ty1; ty++) {
        uint8_t *row = pe_harm_tiles + (size_t)ty * cols;
        for (int tx = tx0; tx < tx1; tx++) {
            if (whole_screen && row[tx] != PE_AREA_UNKNOWN) {
                continue;
            }
            if (row[tx] != (uint8_t)a) {
                row[tx] = a;
                pe_harm_gen++;
            }
        }
    }
}

static void pe_window_saw_blit(uint32_t surface, uint32_t pitch,
                               uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                               uint32_t scr_w, uint32_t scr_h)
{
    const char *on = getenv("PPCGPU_WINDOWS");
    if (!w || !h) {
        return;
    }
    pe_harmony_mark(surface, x, y, w, h, scr_w, scr_h);
    int64_t now = g_get_monotonic_time();
    if (on && on[0] >= '2') {
        /* Every copy, for working out what the compositor is doing. */
        fprintf(stderr, "PEBLIT surface=%06x pitch=%u at=(%u,%u) %ux%u "
                "screen=%ux%u\n", surface, pitch, x, y, w, h, scr_w, scr_h);
    }
    PEWindow *slot = NULL;
    for (int i = 0; i < PE_WINDOW_MAX; i++) {
        PEWindow *e = &pe_windows[i];
        if (e->live && e->surface == surface && e->pitch == pitch) {
            slot = e;
            break;
        }
    }
    if (!slot) {
        /* A free slot, or the one nobody has drawn to for longest. */
        int64_t oldest = now;
        for (int i = 0; i < PE_WINDOW_MAX; i++) {
            if (!pe_windows[i].live) { slot = &pe_windows[i]; break; }
            if (pe_windows[i].last_us <= oldest) {
                oldest = pe_windows[i].last_us;
                slot = &pe_windows[i];
            }
        }
        memset(slot, 0, sizeof(*slot));
        slot->live = true;
        slot->surface = surface;
        slot->pitch = pitch;
        slot->last_us = now - PE_WINDOW_BURST_US - 1;   /* starts a burst */
    }
    if (now - slot->last_us > PE_WINDOW_BURST_US) {
        /* A new burst: keep what the last one worked out, and start over. */
        if (slot->settled) {
            slot->fx0 = slot->x0; slot->fy0 = slot->y0;
            slot->fx1 = slot->x1; slot->fy1 = slot->y1;
        }
        slot->x0 = x; slot->y0 = y; slot->x1 = x + w; slot->y1 = y + h;
        slot->settled = true;
    } else {
        slot->x0 = MIN(slot->x0, x);
        slot->y0 = MIN(slot->y0, y);
        slot->x1 = MAX(slot->x1, x + w);
        slot->y1 = MAX(slot->y1, y + h);
    }
    slot->pieces++;
    slot->last_us = now;
    slot->scr_w = scr_w;
    slot->scr_h = scr_h;

    /*
     * Print every couple of seconds.  Printing "once the copies settle"
     * cannot work from here: this is only called *by* a copy, so the
     * moment things go quiet is the moment nothing calls it again.
     */
    if (!on || now - pe_windows_printed_us < 2 * G_TIME_SPAN_SECOND) {
        return;
    }
    pe_windows_printed_us = now;
    fprintf(stderr, "PEWINDOWS ----\n");
    for (int i = 0; i < PE_WINDOW_MAX; i++) {
        PEWindow *e = &pe_windows[i];
        if (!e->live || now - e->last_us > 5 * G_TIME_SPAN_SECOND) {
            continue;
        }
        fprintf(stderr, "PEWINDOW surface=%06x pitch=%u frame=(%u,%u)-(%u,%u) "
                "%ux%u kind=%s pieces=%" PRIu64 "\n",
                e->surface, e->pitch, e->x0, e->y0, e->x1, e->y1,
                e->x1 - e->x0, e->y1 - e->y0, pe_window_kind(e), e->pieces);
    }
}

static void G_GNUC_PRINTF(1, 2) seq_log(const char *fmt, ...)
{
    if (g_seq_log_enabled < 0) {
        const char *e = getenv("PPCGPU_SEQ_LOG");
        g_seq_log_enabled = e && e[0] != '0';
    }
    if (!g_seq_log_enabled) {
        return;
    }
    if (!g_seq_log) {
        g_seq_log = fopen("/tmp/gpu_seq.log", "a");
        if (!g_seq_log) {
            return;
        }
        setvbuf(g_seq_log, NULL, _IOLBF, 0);
    }
    va_list ap;
    va_start(ap, fmt);
    fprintf(g_seq_log, "%llu %s ", (unsigned long long)++g_seq,
            g_in_pm4 ? "RING" : "CPU ");
    vfprintf(g_seq_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_seq_log);
}

static void blit_path_log(const char *path, const char *fmt, ...)
{
    static FILE *f = NULL;
    if (!f) {
        f = fopen("/tmp/gpu_blit_paths.log", "w");
        if (!f) return;
        fprintf(f, "=== Blit Path Instrumentation ===\n");
    }
    uint64_t total = g_blit_stats.sep_count + g_blit_stats.mmio_count +
                     g_blit_stats.bbm_count + g_blit_stats.paint_count +
                     g_blit_stats.hostdata_count;
    fprintf(f, "[%s #%llu] ", path, (unsigned long long)total);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fprintf(f, "\n");

    /* Periodically dump summary counts */
    if (total - g_blit_stats.last_dump >= 500) {
        fprintf(f, "--- COUNTS: SEP=%llu MMIO=%llu BBM=%llu PAINT=%llu HD=%llu ---\n",
                (unsigned long long)g_blit_stats.sep_count,
                (unsigned long long)g_blit_stats.mmio_count,
                (unsigned long long)g_blit_stats.bbm_count,
                (unsigned long long)g_blit_stats.paint_count,
                (unsigned long long)g_blit_stats.hostdata_count);
        g_blit_stats.last_dump = total;
        fflush(f);
    }
}

/* ========================================================================
 * Post-probe MMIO read audit — Phase A-D instrumentation
 *
 * Goal: Capture ALL MMIO reads occurring after probe draws begin,
 * to find the gate that prevents QE promotion from probe to compositor.
 * The audit runs in a narrow window to keep logs manageable.
 * ======================================================================== */

/*
 * Global draw counter — incremented by ppc_mac_gpu_dispatch_3d_draw,
 * read by ppc_mac_gpu_mmio_read to define the audit window.
 */
static volatile int g_post_probe_draw_count = 0;

/*
 * Post-probe read audit state.
 * Tracks unique registers read and groups them for summary.
 */
#define POST_PROBE_MAX_REGS 256
typedef struct PostProbeReadEntry {
    uint32_t reg;
    uint32_t last_value;
    uint32_t count;
    uint32_t first_at_draw;   /* draw count when first seen */
    uint32_t last_at_draw;    /* draw count when last seen */
} PostProbeReadEntry;

static struct {
    PostProbeReadEntry entries[POST_PROBE_MAX_REGS];
    int num_entries;
    int total_reads;
    int audit_active;         /* 1 = currently in audit window */
    int window_logged;        /* 1 = PROBE_WINDOW already emitted */
    int summary_logged;       /* 1 = summary already emitted */
    int first_draw_seen;      /* draw count when first 3D draw arrived */
    int window_start_draw;    /* draw count at window start */
    int window_end_draw;      /* draw count at window end */
} g_post_probe_audit;

static int post_probe_find_or_add_reg(uint32_t reg)
{
    for (int i = 0; i < g_post_probe_audit.num_entries; i++) {
        if (g_post_probe_audit.entries[i].reg == reg) {
            return i;
        }
    }
    if (g_post_probe_audit.num_entries >= POST_PROBE_MAX_REGS) {
        return -1;
    }
    int idx = g_post_probe_audit.num_entries++;
    g_post_probe_audit.entries[idx].reg = reg;
    g_post_probe_audit.entries[idx].count = 0;
    g_post_probe_audit.entries[idx].first_at_draw = g_post_probe_draw_count;
    return idx;
}

/* ========================================================================
 * Register name lookup for tracing
 * ======================================================================== */

static const char *ppc_mac_gpu_reg_name(hwaddr addr)
{
    switch (addr & ~3ULL) {
    case R200_MM_INDEX:            return "MM_INDEX";
    case R200_MM_DATA:             return "MM_DATA";
    case R200_BUS_CNTL:            return "BUS_CNTL";
    case R200_GEN_INT_CNTL:        return "GEN_INT_CNTL";
    case R200_GEN_INT_STATUS:      return "GEN_INT_STATUS";
    case R200_CLOCK_CNTL_INDEX:    return "CLOCK_CNTL_INDEX";
    case R200_CLOCK_CNTL_DATA:     return "CLOCK_CNTL_DATA";
    case R200_MC_FB_LOCATION:      return "MC_FB_LOCATION";
    case R200_MC_AGP_LOCATION:     return "MC_AGP_LOCATION";
    case R200_MC_STATUS:           return "MC_STATUS";
    case R200_CONFIG_APER_0_BASE:  return "CONFIG_APER_0_BASE";
    case R200_CONFIG_APER_SIZE:    return "CONFIG_APER_SIZE";
    case R200_CONFIG_MEMSIZE:      return "CONFIG_MEMSIZE";
    case R200_CONFIG_CNTL:         return "CONFIG_CNTL";
    case R200_RBBM_STATUS:         return "RBBM_STATUS";
    case R200_RBBM_SOFT_RESET:     return "RBBM_SOFT_RESET";
    case R200_GUI_STAT:            return "GUI_STAT";
    case R200_HOST_PATH_CNTL:      return "HOST_PATH_CNTL";
    case R200_SURFACE_CNTL:        return "SURFACE_CNTL";
    case R200_CRTC_GEN_CNTL:       return "CRTC_GEN_CNTL";
    case R200_CRTC_EXT_CNTL:       return "CRTC_EXT_CNTL";
    case R200_CRTC_STATUS:         return "CRTC_STATUS";
    case R200_CRTC_VLINE_CRNT_VLINE: return "CRTC_VLINE_CRNT_VLINE";
    case R200_CRTC_CRNT_FRAME:     return "CRTC_CRNT_FRAME";
    case R200_CRTC_H_TOTAL_DISP:   return "CRTC_H_TOTAL_DISP";
    case R200_CRTC_V_TOTAL_DISP:   return "CRTC_V_TOTAL_DISP";
    case R200_CRTC_OFFSET:         return "CRTC_OFFSET";
    case R200_CRTC_OFFSET_CNTL:    return "CRTC_OFFSET_CNTL";
    case R200_CRTC_PITCH:          return "CRTC_PITCH";
    case R200_DAC_CNTL:            return "DAC_CNTL";
    case R200_DAC_CNTL2:           return "DAC_CNTL2";
    case R200_PALETTE_INDEX:       return "PALETTE_INDEX";
    case R200_PALETTE_DATA:        return "PALETTE_DATA";
    case R200_CUR_OFFSET:          return "CUR_OFFSET";
    case R200_CUR_HORZ_VERT_POSN:  return "CUR_HORZ_VERT_POSN";
    case R200_CUR_HORZ_VERT_OFF:   return "CUR_HORZ_VERT_OFF";
    case R200_CUR_CLR0:            return "CUR_CLR0";
    case R200_CUR_CLR1:            return "CUR_CLR1";
    case R200_DISP_MISC_CNTL:      return "DISP_MISC_CNTL";
    case R200_DISP_OUTPUT_CNTL:    return "DISP_OUTPUT_CNTL";
    case R200_DISP_MERGE_CNTL:     return "DISP_MERGE_CNTL";
    case R200_FP_GEN_CNTL:         return "FP_GEN_CNTL";
    case R200_DST_OFFSET:          return "DST_OFFSET";
    case R200_DST_PITCH:           return "DST_PITCH";
    case R200_DST_WIDTH:           return "DST_WIDTH";
    case R200_DST_HEIGHT:          return "DST_HEIGHT";
    case R200_DST_HEIGHT_WIDTH:    return "DST_HEIGHT_WIDTH";
    case R200_SRC_OFFSET:          return "SRC_OFFSET";
    case R200_SRC_PITCH:           return "SRC_PITCH";
    case R200_DP_GUI_MASTER_CNTL:  return "DP_GUI_MASTER_CNTL";
    case R200_DP_CNTL:             return "DP_CNTL";
    case R200_DP_DATATYPE:         return "DP_DATATYPE";
    case R200_DP_MIX:              return "DP_MIX";
    case R200_CP_RB_BASE:          return "CP_RB_BASE";
    case R200_CP_RB_CNTL:          return "CP_RB_CNTL";
    case R200_CP_RB_RPTR_WR:       return "CP_RB_RPTR_WR";
    case R200_CP_RB_RPTR:          return "CP_RB_RPTR";
    case R200_CP_RB_WPTR:          return "CP_RB_WPTR";
    case R200_CP_ME_CNTL:          return "CP_ME_CNTL";
    case R200_CP_IB_BASE:          return "CP_IB_BASE";
    case R200_CP_IB_BUFSZ:         return "CP_IB_BUFSZ";
    case R200_CP_CSQ_CNTL:         return "CP_CSQ_CNTL";
    case R200_CP_STAT:             return "CP_STAT";
    case R200_SCRATCH_UMSK:         return "SCRATCH_UMSK";
    case R200_SCRATCH_ADDR:         return "SCRATCH_ADDR";
    case R200_SCRATCH_REG0:        return "SCRATCH_REG0";
    case R200_SCRATCH_REG1:        return "SCRATCH_REG1";
    case R200_SCRATCH_REG2:        return "SCRATCH_REG2";
    case R200_SCRATCH_REG3:        return "SCRATCH_REG3";
    case R200_SCRATCH_REG4:        return "SCRATCH_REG4";
    case R200_SCRATCH_REG5:        return "SCRATCH_REG5";
    case R200_CP_ME_RAM_ADDR:      return "CP_ME_RAM_ADDR";
    case R200_CP_ME_RAM_DATAH:     return "CP_ME_RAM_DATAH";
    case R200_CP_ME_RAM_DATAL:     return "CP_ME_RAM_DATAL";
    case R200_CP_CSQ_IND_ADDR:     return "CP_CSQ_IND_ADDR";
    case R200_CP_CSQ_IND_DATA:     return "CP_CSQ_IND_DATA";
    case R200_CP_CSQ_STAT:         return "CP_CSQ_STAT";
    case R200_ISYNC_CNTL:          return "ISYNC_CNTL";
    case R200_WAIT_UNTIL:          return "WAIT_UNTIL";
    case R200_AGP_BASE:            return "AGP_BASE";
    case R200_AGP_CNTL:            return "AGP_CNTL";
    case R200_AGP_COMMAND:         return "AGP_COMMAND";
    case R200_AGP_STATUS:          return "AGP_STATUS";
    case R200_GPIO_VGA_DDC:        return "GPIO_VGA_DDC";
    case R200_GPIO_DVI_DDC:        return "GPIO_DVI_DDC";
    case R200_GPIO_MONID:          return "GPIO_MONID";
    default:
        if (addr >= R200_BIOS_0_SCRATCH && addr <= R200_BIOS_7_SCRATCH) {
            return "BIOS_SCRATCH";
        }
        if (addr >= R200_HOST_DATA0 && addr <= R200_HOST_DATA_LAST) {
            return "HOST_DATA";
        }
        if (addr >= R200_PCI_MIRROR_BASE && addr < R200_PCI_MIRROR_END) {
            return "PCI_MIRROR";
        }
        return "UNKNOWN";
    }
}

/* ========================================================================
 * Debug logging - shared sequence counter for all access types
 * ======================================================================== */

static FILE *gpu_debug_fp = NULL;
static uint64_t gpu_debug_seq = 0;

static bool gpu_verify_fetch(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("PPCGPU_VERIFY_FETCH");
        on = e && e[0] == '1';
    }
    return on;
}

static bool gpu_debug_enabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *e = getenv("PPCGPU_DEBUG_LOG");
        enabled = e && e[0] == '1';
    }
    return enabled;
}

/* PPCGPU_VP=0 puts vertex programs aside, for comparing against the
 * fixed-function path when a program draws something unexpected. */
static bool gpu_vp_disabled(void)
{
    static int off = -1;
    if (off < 0) {
        const char *e = getenv("PPCGPU_VP");
        off = e && e[0] == '0';
    }
    return off;
}

static void gpu_debug_log(const char *fmt, ...)
{
    /* Per-access debug log: opt-in (PPCGPU_DEBUG_LOG=1).  It writes and
     * flushes a line for every register access, which on its own costs a
     * large share of guest graphics performance. */
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("PPCGPU_DEBUG_LOG");
        enabled = e && e[0] == '1';
    }
    if (!enabled) {
        return;
    }
    if (!gpu_debug_fp) {
        gpu_debug_fp = fopen("/tmp/gpu_all_access.log", "w");
        if (!gpu_debug_fp) return;
    }
    va_list ap;
    va_start(ap, fmt);
    fprintf(gpu_debug_fp, "[%06"PRIu64"] ", gpu_debug_seq++);
    vfprintf(gpu_debug_fp, fmt, ap);
    fputc('\n', gpu_debug_fp);
    va_end(ap);
    fflush(gpu_debug_fp);
}

/* ========================================================================
 * I2C / DDC helpers for EDID
 * ======================================================================== */

/*
 * ATI GPIO DDC register bit layout (same for VGA_DDC, DVI_DDC):
 *   Bit 0:  SDA output value
 *   Bit 1:  SCL output value
 *   Bit 8:  SDA input value (read-only)
 *   Bit 9:  SCL input value (read-only)
 *   Bit 16: SDA output enable (1 = drive, 0 = tristate)
 *   Bit 17: SCL output enable (1 = drive, 0 = tristate)
 */
static uint64_t ppc_mac_gpu_i2c(bitbang_i2c_interface *i2c,
                                 uint64_t data, int base)
{
    bool c = (data & BIT(base + 17) ? !!(data & BIT(base + 1)) : 1);
    bool d = (data & BIT(base + 16) ? !!(data & BIT(base)) : 1);

    bitbang_i2c_set(i2c, BITBANG_I2C_SCL, c);
    d = bitbang_i2c_set(i2c, BITBANG_I2C_SDA, d);

    data &= ~0xf00ULL;
    if (c) {
        data |= BIT(base + 9);
    }
    if (d) {
        data |= BIT(base + 8);
    }
    return data;
}

static inline uint64_t ppc_mac_gpu_reg_read_offs(uint32_t reg, int offs,
                                                  unsigned int size)
{
    if (offs == 0 && size == 4) {
        return reg;
    } else {
        return extract32(reg, offs * BITS_PER_BYTE, size * BITS_PER_BYTE);
    }
}

static inline void ppc_mac_gpu_reg_write_offs(uint32_t *reg, int offs,
                                               uint64_t data, unsigned int size)
{
    if (offs == 0 && size == 4) {
        *reg = data;
    } else {
        *reg = deposit32(*reg, offs * BITS_PER_BYTE, size * BITS_PER_BYTE,
                         data);
    }
}

/* ========================================================================
 * Interrupt handling
 * ======================================================================== */

static void ppc_mac_gpu_update_irq(PPCMacGPUState *s)
{
    bool level = !!(s->regs.gen_int_status & s->regs.gen_int_cntl);
    pci_set_irq(&s->pci, level);
    trace_ppc_mac_gpu_irq(s->regs.gen_int_status, s->regs.gen_int_cntl,
                          level);
}

static void ppc_mac_gpu_vblank(void *opaque)
{
    PPCMacGPUState *s = opaque;

    /* Schedule next VBlank at 60 Hz */
    timer_mod(&s->vblank_timer,
              qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NANOSECONDS_PER_SECOND / 60);

    /* Set VBlank interrupt status */
    s->regs.gen_int_status |= R200_CRTC_VBLANK_INT;
    s->regs.stall_irqs++;

    /* Toggle VBLANK_SAVE in CRTC_STATUS */
    s->regs.crtc_status ^= R200_CRTC_VBLANK_SAVE;

    ppc_mac_gpu_update_irq(s);
}

/* ========================================================================
 * Display mode tracking
 * ======================================================================== */

/* [STRIDE_CHANGE] traces fire on every desktop update; opt in with PPCGPU_STRIDE_DIAG. */
static bool ppcgpu_stride_diag(void)
{
    static int on = -1;

    if (on < 0) {
        on = getenv("PPCGPU_STRIDE_DIAG") != NULL;
    }
    return on;
}

/*
 * Derive the current display mode from CRTC registers.
 * Returns true if the mode is valid and has changed.
 */
/*
 * Adopt a presentation pitch seen in a compositor BLT.  A pitch narrower
 * than a scanline of the current mode cannot describe the framebuffer, and
 * taking it would size the shadow buffer (and the surface handed to the UI)
 * shorter than the rows read out of it.
 */
static void r200_set_present_pitch(PPCMacGPUState *s, uint32_t dst_pitch)
{
    uint32_t min = s->disp.width * ((s->disp.bpp + 7) / 8);

    if (dst_pitch < min) {
        static int pitch_reject_log;
        if (pitch_reject_log < 10) {
            if (ppcgpu_stride_diag()) fprintf(stderr, "[STRIDE_CHANGE] ignoring BLT pitch %u for "
                    "%ux%u@%u (needs %u)\n", dst_pitch, s->disp.width,
                    s->disp.height, s->disp.bpp, min);
            pitch_reject_log++;
        }
        return;
    }
    s->disp_stride_override_active = true;
    s->disp_stride_override_value = dst_pitch;
    s->disp.stride = dst_pitch;
    s->display_invalid = true;
}

static bool ppc_mac_gpu_update_display_mode(PPCMacGPUState *s)
{
    PPCMacGPUDisplayMode old = s->disp;
    PPCMacGPUDisplayMode *m = &s->disp;
    uint32_t crtc_gen = s->regs.crtc_gen_cntl;
    uint32_t h_total_disp = s->regs.crtc_h_total_disp;
    uint32_t v_total_disp = s->regs.crtc_v_total_disp;
    uint32_t pitch = s->regs.crtc_pitch;

    /* Check if CRTC is enabled in extended mode */
    if (!(crtc_gen & R200_CRTC_EN) && !(crtc_gen & R200_CRTC_EXT_DISP_EN)) {
        /* Not in extended mode, keep last valid mode or use default */
        if (m->width == 0) {
            m->width = 640;
            m->height = 480;
            m->bpp = 32;
            m->stride = 640 * 4;
            m->offset = 0;
            m->format = PIXMAN_BE_x8r8g8b8;
        }
        m->crtc_ext = false;
        return false;
    }
    m->crtc_ext = true;

    /* Extract display dimensions from CRTC timing registers */
    /* H_TOTAL_DISP: bits [7:0] = h_total/8-1, bits [23:16] = h_disp/8-1 */
    m->width = (((h_total_disp >> 16) & 0xFF) + 1) * 8;

    /* V_TOTAL_DISP: bits [10:0] = v_total-1, bits [26:16] = v_disp-1 */
    m->height = ((v_total_disp >> 16) & 0x7FF) + 1;

    /* Sanity check dimensions */
    if (m->width < 64 || m->width > 4096 ||
        m->height < 64 || m->height > 4096) {
        m->width = 640;
        m->height = 480;
    }

    /* Pixel width from CRTC_GEN_CNTL */
    switch (crtc_gen & R200_CRTC_PIX_WIDTH_MASK) {
    case R200_CRTC_PIX_WIDTH_8BPP:
        m->bpp = 8;
        /* 8bpp uses palette; for surface we expand to 32bpp */
        m->format = PIXMAN_BE_x8r8g8b8;
        break;
    case R200_CRTC_PIX_WIDTH_15BPP:
        m->bpp = 15;
        m->format = PIXMAN_x1r5g5b5;
        break;
    case R200_CRTC_PIX_WIDTH_16BPP:
        m->bpp = 16;
        m->format = PIXMAN_r5g6b5;
        break;
    case R200_CRTC_PIX_WIDTH_24BPP:
        m->bpp = 24;
        m->format = PIXMAN_BE_r8g8b8;
        break;
    case R200_CRTC_PIX_WIDTH_32BPP:
        m->bpp = 32;
        /*
         * PPC guest writes big-endian VRAM. ATI Radeon hardware on PPC Macs
         * has a byte-swap engine controlled by SURFACE_CNTL. When the
         * surface is configured for byte-swapping, the hardware presents
         * the data in the correct order.
         *
         * For now, we assume big-endian VRAM from the PPC guest.
         */
        m->format = PIXMAN_BE_x8r8g8b8;
        break;
    default:
        m->bpp = 32;
        m->format = PIXMAN_BE_x8r8g8b8;
        break;
    }

    /* Pitch from CRTC_PITCH register (in units of pixels / 8 for pre-R300) */
    /* Actually for R200: bits [10:0] of CRTC_PITCH * 8 = pitch in pixels */
    if (pitch & 0x7FF) {
        m->stride = (pitch & 0x7FF) * 8 * ((m->bpp + 7) / 8);
    } else {
        m->stride = m->width * ((m->bpp + 7) / 8);
    }

    /*
     * Clear the stride override on a resolution change: it belongs to the
     * mode it was captured in.  This has to happen before the override is
     * applied below, or the frame that changes resolution gets the old
     * mode's pitch with the new mode's width -- a stride shorter than a
     * scanline, which walks off the end of the framebuffer.
     *
     * Coming from no mode at all is not a resolution change.  A machine
     * woken from sleep is a fresh device that has just had the sleeping
     * one's registers and learned scanline length poured into it, and its
     * first display update therefore reads as 0x0 -> 1680x1050 -- which
     * threw away the very override the snapshot had just restored, and the
     * desktop came back sheared into bands.  There is nothing to protect
     * here: with no previous mode there is no stale pitch to carry over.
     */
    if (s->disp_stride_override_active &&
        old.width != 0 && old.height != 0 &&
        (old.width != m->width || old.height != m->height ||
         old.bpp != m->bpp)) {
        if (ppcgpu_stride_diag()) fprintf(stderr, "[STRIDE_CHANGE] mode change %ux%u->%ux%u: "
                "clearing stride override (was %u)\n",
                old.width, old.height, m->width, m->height,
                s->disp_stride_override_value);
        s->disp_stride_override_active = false;
        s->disp_stride_override_value = 0;
    }

    /*
     * Presentation stride override (Phase C):
     * When the QE compositor BLTs its back buffer to the visible
     * framebuffer, the BLT dst_pitch may differ from CRTC_PITCH.
     * If the override is active, use the BLT pitch instead of the
     * CRTC-derived stride. This prevents every-frame reset from
     * undoing the pitch correction.
     */
    if (s->disp_stride_override_active && s->disp_stride_override_value > 0) {
        if (m->stride != s->disp_stride_override_value) {
            static int stride_override_log = 0;
            if (stride_override_log < 20) {
                if (ppcgpu_stride_diag()) fprintf(stderr, "[STRIDE_CHANGE] update_display_mode: "
                        "CRTC gives %u, override forces %u\n",
                        m->stride, s->disp_stride_override_value);
                stride_override_log++;
            }
        }
        m->stride = s->disp_stride_override_value;
    }

    /*
     * A stride narrower than one scanline cannot be right whatever the
     * guest wrote, and anything reading the framebuffer row by row would
     * run past its end.  Keep the display consistent instead.
     */
    if (m->stride < m->width * ((m->bpp + 7) / 8)) {
        if (ppcgpu_stride_diag()) fprintf(stderr, "[STRIDE_CHANGE] stride %u too small for %ux%u@%u, "
                "using %u\n", m->stride, m->width, m->height, m->bpp,
                m->width * ((m->bpp + 7) / 8));
        m->stride = m->width * ((m->bpp + 7) / 8);
    }

    /* Scanout offset */
    /*
     * The rv250 reports its CRTC_OFFSET R200_CRTC_OFFSET_BIAS bytes in (see
     * the register read), and Mac OS X's accelerator then draws the screen
     * from there (it starts at x = 8): scan out from the same place.
     */
    m->offset = s->regs.crtc_offset + (s->rv250 ? R200_CRTC_OFFSET_BIAS : 0);

    /* Validate offset + frame fits in VRAM */
    if (m->offset + (uint64_t)m->stride * m->height > s->vram_size) {
        m->offset = 0;
    }

    /* Check if mode changed */
    if (memcmp(&old, m, sizeof(old)) != 0) {
        /* not for a page flip (only the offset moves): Doom 3 flips every frame */
        if (old.width != m->width || old.height != m->height || old.bpp != m->bpp ||
            old.stride != m->stride) {
            fprintf(stderr, "GPU MODE: %ux%u bpp=%u stride=%u offset=0x%x pitch_reg=0x%x crtc_gen=0x%x\n",
                    m->width, m->height, m->bpp, m->stride, m->offset, pitch, crtc_gen);
        }
        if (old.stride != m->stride) {
            if (ppcgpu_stride_diag()) fprintf(stderr, "[STRIDE_CHANGE] update_display_mode: "
                    "%u -> %u (override=%s val=%u)\n",
                    old.stride, m->stride,
                    s->disp_stride_override_active ? "yes" : "no",
                    s->disp_stride_override_value);
        }
        return true;
    }
    return false;
}

/* ========================================================================
 * Display update callback
 * ======================================================================== */

/*
 * Byte-swap a scanline of 32bpp pixels from PPC big-endian VRAM to
 * host-native order for the Cocoa display backend.
 *
 * PPC guest writes XRGB as big-endian: bytes [X, R, G, B] in memory.
 * Cocoa's CGImage uses kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst,
 * expecting LE XRGB: bytes [B, G, R, X] in memory.
 * A bswap32 per pixel converts between the two.
 */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

static void ppc_mac_gpu_bswap_line32(uint32_t *restrict dst,
                                     const uint32_t *restrict src,
                                     int width)
{
    int i = 0;

#if defined(__ARM_NEON)
    /* reverse the bytes of each 32-bit pixel; 16 bytes = 4 pixels a round,
     * four vectors a pass so the loads and stores overlap. */
    for (; i + 16 <= width; i += 16) {
        uint8x16_t a = vrev32q_u8(vld1q_u8((const uint8_t *)(src + i)));
        uint8x16_t b = vrev32q_u8(vld1q_u8((const uint8_t *)(src + i + 4)));
        uint8x16_t c = vrev32q_u8(vld1q_u8((const uint8_t *)(src + i + 8)));
        uint8x16_t d = vrev32q_u8(vld1q_u8((const uint8_t *)(src + i + 12)));
        vst1q_u8((uint8_t *)(dst + i), a);
        vst1q_u8((uint8_t *)(dst + i + 4), b);
        vst1q_u8((uint8_t *)(dst + i + 8), c);
        vst1q_u8((uint8_t *)(dst + i + 12), d);
    }
#elif defined(__SSE2__)
    const __m128i m = _mm_set1_epi32(0x00FF00FF);
    for (; i + 4 <= width; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(src + i));
        v = _mm_or_si128(_mm_srli_epi32(v, 16), _mm_slli_epi32(v, 16));
        v = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(v, m), 8),
                         _mm_srli_epi16(_mm_and_si128(v, m), 8));
        _mm_storeu_si128((__m128i *)(dst + i), v);
    }
#endif
    for (; i < width; i++) {
        dst[i] = bswap32(src[i]);
    }
}

/*
 * CP thread statistics: how often the guest submitted, and how often it
 * made the ring catch up before touching the card (and through which
 * register) -- each of those is parallelism given back.
 */
static void ppc_mac_gpu_cp_rate_log(double sec)
{
    static const char *names[4];
    unsigned top[3] = { 0 }, n = 0;

    if (!cp.started || (!cp.kicks && !cp.syncs)) {
        return;
    }
    for (unsigned r = 0; r < ARRAY_SIZE(cp.sync_reg); r++) {
        for (unsigned k = 0; k < 3; k++) {
            if (cp.sync_reg[r] > cp.sync_reg[top[k]]) {
                memmove(&top[k + 1], &top[k], (2 - k) * sizeof(top[0]));
                top[k] = r;
                break;
            }
        }
    }
    for (unsigned k = 0; k < 3; k++) {
        names[k] = cp.sync_reg[top[k]] ? ppc_mac_gpu_reg_name(top[k] * 4) : NULL;
        n += names[k] != NULL;
    }
    qemu_log("ppc-mac-gpu ring: %.0f submits/s, %.0f catch-ups/s%s%s%s%s%s%s%s\n",
             cp.kicks / sec, cp.syncs / sec, n ? " (" : "",
             names[0] ? names[0] : "", names[1] ? ", " : "",
             names[1] ? names[1] : "", names[2] ? ", " : "",
             names[2] ? names[2] : "", n ? ")" : "");
    qatomic_set(&cp.kicks, 0);
    cp.syncs = 0;
    memset(cp.sync_reg, 0, sizeof(cp.sync_reg));
}

static bool ppc_gpu_dirty_scanout = true;   /* read from env in realize */
static bool ppc_gpu_flip_kick = true;
static bool ppc_gpu_frame_kick = true;
/*
 * Deferred ("async") present is off by default.  While the guest composites
 * continuously (Quartz Extreme window drag at ~47 frames/s) the scanout range
 * is busy on almost every refresh, so each tick was deferred and the next
 * tick found new work again: only ~12 of 33 refreshes per second reached the
 * screen.  The synchronous flush it avoids costs ~0.1% GPU wait.
 * PPCGPU_ASYNC_PRESENT=1 turns it back on.
 */
static bool ppc_gpu_async_present;

/*
 * Render-queue completion for a deferred present (see the head of
 * ppc_mac_gpu_display_update): runs on a backend completion thread, so it
 * touches nothing but the console, asking the main loop to run the refresh
 * we skipped -- with the batch now finished, that pass copies for real.
 */
static void ppc_mac_gpu_present_done(void *arg, uint32_t seq)
{
    PPCMacGPUState *s = arg;

    (void)seq;
    dpy_refresh_soon(s->con);
}

/*
 * Frame kick.  The UI polls the display every 30 ms (SPICE's default), so a
 * guest compositing ~50 frames/s (Tiger's Quartz Extreme during a window
 * drag) was shown at most ~33 times a second.  When the guest finishes a
 * batch of drawing, ask for a refresh as soon as 16 ms have passed since the
 * last one: up to ~60 frames/s while it draws, and nothing extra when idle
 * (the 30 ms poll continues as before).  PPCGPU_FRAME_KICK=0 turns it off.
 */
#define PPC_GPU_FRAME_KICK_MS 16

static void ppc_mac_gpu_frame_kick_cb(void *opaque)
{
    PPCMacGPUState *s = opaque;

    dpy_refresh_soon(s->con);
}

static void ppc_mac_gpu_frame_ready(PPCMacGPUState *s)
{
    int64_t now, due;

    if (!ppc_gpu_frame_kick || !s->con ||
        timer_pending(&s->frame_kick_timer)) {
        return;
    }
    now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    due = qatomic_read(&s->last_scanout_ms) + PPC_GPU_FRAME_KICK_MS;
    timer_mod(&s->frame_kick_timer, MAX(now, due));
}

static void ppc_mac_gpu_display_update(void *opaque)
{
    PPCMacGPUState *s = opaque;

    qatomic_set(&s->last_scanout_ms, qemu_clock_get_ms(QEMU_CLOCK_REALTIME));

    /* A machine that has just been woken: give the window the pointer the
     * sleeping one had (see ppc_mac_gpu_post_load). */
    if (s->hwc_announce && s->con) {
        s->hwc_announce = false;
        ppc_mac_gpu_hwc_define(s);
        ppc_mac_gpu_hwc_move(s);
    }

    {
        int64_t now = g_get_monotonic_time();
        if (!r200_rate.since) {
            r200_rate.since = now;
        } else if (now - r200_rate.since >= 1000000) {
            if (r200_rate.draws || r200_rate.ops2d) {
                double sec = (now - r200_rate.since) / 1e6;
                qemu_log("ppc-mac-gpu rate: %.1f flips/s, %.1f present-ops/s, "
                         "%.0f draws/s (%.0f%% GPU vertex programs), "
                         "%.0f 2D ops/s, %.0f flushes/s, GPU wait %.1f%%\n",
                         r200_rate.flips / sec,
                         r200_rate.presents / sec, r200_rate.draws / sec,
                         100.0 * r200_rate.gpu_vs / r200_rate.draws,
                         r200_rate.ops2d / sec, r200_rate.flushes / sec,
                         r200_rate.flush_us / (sec * 1e4));
                if (r200_rate.co_copies || r200_rate.co_skips ||
                    r200_rate.co_defers) {
                    qemu_log("ppc-mac-gpu scanout: %.0f copies/s "
                             "(%.0f partial, %.0f unchanged/s skipped), "
                             "%.0f presents deferred/s\n",
                             r200_rate.co_copies / sec,
                             r200_rate.co_partials / sec,
                             r200_rate.co_skips / sec,
                             r200_rate.co_defers / sec);
                }
            }
            ppc_mac_gpu_cp_rate_log((now - r200_rate.since) / 1e6);
            memset(&r200_rate, 0, sizeof(r200_rate));
            r200_rate.since = now;
        }
    }
    if (s->disp.stride && s->disp.height) {
        uint64_t lo = s->regs.crtc_offset;
        uint64_t hi = lo + (uint64_t)s->disp.stride * s->disp.height;
        if (s->renderer && s->renderer->range_busy_r200 &&
            s->renderer->range_busy_r200(s->renderer_opaque, lo, hi, false)) {
            /*
             * Unfinished 3D work may still write the scanout.  With an
             * asynchronous render queue there is a cheaper answer than
             * blocking this thread until it drains: commit the batch, take
             * the completion callback, and come back when it fires.  A
             * handful of deferrals is a frame of latency; after that fall
             * back to the synchronous flush, so a busy queue can never
             * starve the display.
             */
            if (s->renderer->draw_queue_async && s->scanout_log_owner &&
                ppc_gpu_async_present && s->renderer->submit_r200 &&
                s->scanout_defers < 4) {
                s->scanout_defers++;
                r200_rate.co_defers++;
                s->renderer->submit_r200(s->renderer_opaque,
                                         ppc_mac_gpu_present_done, s);
                return;
            }
            r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
        }
    } else {
        r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
    }
    bool mode_changed;
    uint8_t *vram_ptr;

    mode_changed = ppc_mac_gpu_update_display_mode(s);

    /* Phase D: Reset Metal RT tracking for new frame */
    s->metal_rt_count = 0;
    s->metal_rt_frame_id++;

    if (s->disp.width == 0 || s->disp.height == 0) {
        static int zero_count = 0;
        if (zero_count < 3) {
            fprintf(stderr, "GPU DIAG: display_update called but w=%u h=%u, skipping\n",
                    s->disp.width, s->disp.height);
            zero_count++;
        }
        return;
    }

    vram_ptr = memory_region_get_ram_ptr(&s->vram);

    /* Phase D — Post-present tile range check: sample the window texture
     * range every 200 frames (after boot) to detect late population by
     * direct guest CPU writes. */
    if (ppc_gpu_env_diag()) {
        static int phase_d_frame = 0;
        phase_d_frame++;
        if (phase_d_frame == 500 || phase_d_frame == 700 ||
            phase_d_frame == 1000 || phase_d_frame == 1500) {
            int nonzero_count = 0;
            uint32_t first_nz_off = 0, first_nz_val = 0;
            for (uint64_t a = 0x353000; a < 0x413000; a += 64) {
                if (a + 4 <= s->vram_size) {
                    uint32_t v = *(uint32_t *)(vram_ptr + a);
                    if (v != 0) {
                        if (nonzero_count == 0) {
                            first_nz_off = (uint32_t)a;
                            first_nz_val = v;
                        }
                        nonzero_count++;
                    }
                }
            }
            fprintf(stderr,
                "[TILE_CHECK_FRAME] frame=%d range=0x353000-0x413000 "
                "nonzero_samples=%d/%d first_nz=0x%x(0x%08x)\n",
                phase_d_frame, nonzero_count,
                (int)((0x413000 - 0x353000) / 64),
                first_nz_off, first_nz_val);
        }
    }

    /*
     * UNIFIED display path — always use shadow buffer + bswap32.
     *
     * QEMU's Cocoa backend hardcodes kCGBitmapByteOrder32Little in
     * CGImageCreate regardless of the pixman format.  This means
     * PIXMAN_BE_x8r8g8b8 surfaces get misinterpreted by Core Graphics,
     * causing visual corruption / banding.
     *
     * Solution: always bswap32 the VRAM (BE) into a shadow buffer and
     * present it as PIXMAN_x8r8g8b8 (native LE), which Cocoa handles
     * correctly.  Works for both CRTC extended mode and VGA/boot mode
     * since PPC always writes BE to memory_region_init_ram.
     */
    uint32_t width = s->disp.width;
    uint32_t height = s->disp.height;
    uint32_t stride = s->disp.stride;
    uint32_t bpp_bytes = (s->disp.bpp + 7) / 8;

    /*
     * 15/16bpp (games switch the display to "thousands of colours"): the
     * Cocoa front end only understands 32bpp little-endian surfaces, so
     * expand the big-endian ARGB1555 / RGB565 scanout into the shadow buffer.
     */
    if (bpp_bytes == 2) {
        uint64_t need = (uint64_t)width * 4 * height;
        if (!s->shadow_buf || s->shadow_buf_size < need) {
            g_free(s->shadow_buf);
            s->shadow_buf = g_malloc(need);
            s->shadow_buf_size = need;
            mode_changed = true;
        }
        if (mode_changed || s->display_invalid) {
            s->display_invalid = false;
            DisplaySurface *ds = qemu_create_displaysurface_from(
                width, height, PIXMAN_x8r8g8b8, width * 4, s->shadow_buf);
            dpy_gfx_replace_surface(s->con, ds);
        }
        bool is565 = s->disp.bpp == 16;
        if ((uint64_t)s->disp.offset + (uint64_t)stride * height <= s->vram_size) {
            for (uint32_t y = 0; y < height; y++) {
                const uint8_t *sp = vram_ptr + s->disp.offset + (uint64_t)y * stride;
                uint32_t *dp = (uint32_t *)s->shadow_buf + (uint64_t)y * width;
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t v = (sp[2 * x] << 8) | sp[2 * x + 1];
                    uint32_t r, g, b;
                    if (is565) {
                        r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31;
                        r = (r << 3) | (r >> 2); g = (g << 2) | (g >> 4);
                    } else {
                        r = (v >> 10) & 31; g = (v >> 5) & 31; b = v & 31;
                        r = (r << 3) | (r >> 2); g = (g << 3) | (g >> 2);
                    }
                    b = (b << 3) | (b >> 2);
                    dp[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
                }
            }
        }
        dpy_gfx_update_full(s->con);
        return;
    }

    /* 8bpp and other non-32bpp modes */
    if (bpp_bytes != 4) {
        static int non32_count = 0;
        if (non32_count < 3) {
            fprintf(stderr, "GPU DIAG: non-32bpp path, bpp=%u w=%u h=%u\n",
                    s->disp.bpp, width, height);
            non32_count++;
        }
        /* Non-32bpp: display directly (rare, only during early boot) */
        if (mode_changed || s->display_invalid) {
            s->display_invalid = false;
            DisplaySurface *ds = qemu_create_displaysurface_from(
                width, height, s->disp.format, stride,
                vram_ptr + s->disp.offset);
            dpy_gfx_replace_surface(s->con, ds);
        }
        dpy_gfx_update_full(s->con);
        return;
    }

    /*
     * VRAM WRITE TRAP: snapshot-diff the compositor RT region to
     * detect CPU writes during drag.
     *
     * Mechanism: save a CRC of a sample row in the compositor RT.
     * On the next display update, compare.  If the CRC changed but
     * our SRT PRESENT_BLIT didn't write there, something else did
     * (likely CPU memcpy from the WindowServer).
     *
     * We scan MULTIPLE potential RT offsets since the compositor
     * offset varies per session (0x300000, 0x900000, 0x940000, etc.)
     */
    if (ppc_gpu_env_diag()) {
        static uint32_t trap_frame = 0;
        static uint32_t prev_fb_crc = 0;
        static uint32_t prev_rt_crcs[4] = {0};
        static uint32_t rt_offsets[4] = {0};
        static int rt_count = 0;
        static int trap_log = 0;
        trap_frame++;

        /* Watch all common compositor RT offsets unconditionally */
        if (trap_frame == 300) {
            rt_offsets[0] = 0x300000;
            rt_offsets[1] = 0x900000;
            rt_offsets[2] = 0x940000;
            rt_offsets[3] = 0xa02000;
            rt_count = 4;
            fprintf(stderr, "[VRAM_WRITE_TRAP] watching 4 RT offsets + framebuffer\n");
        }

        /* Every 10 frames, snapshot and diff */
        if (trap_frame > 300 && trap_frame % 10 == 0 && trap_log < 300) {
            /* Sample framebuffer at y=300 */
            uint32_t fb_crc = 0;
            {
                int xi;
                for (xi = 0; xi < 256; xi++) {
                    uint64_t addr = (uint64_t)s->disp.offset + 300 * stride + xi * 4;
                    if (addr + 4 <= s->vram_size) {
                        fb_crc ^= *(uint32_t *)(vram_ptr + addr);
                        fb_crc = (fb_crc << 7) | (fb_crc >> 25);
                    }
                }
            }

            /* Sample each RT at y=300 */
            uint32_t rt_crcs[4] = {0};
            {
                int ri;
                for (ri = 0; ri < rt_count; ri++) {
                    int xi;
                    for (xi = 0; xi < 256; xi++) {
                        /* Use stride 4096 for probing (matches CRTC pitch) */
                        uint64_t addr = (uint64_t)rt_offsets[ri] + 300 * 4096 + xi * 4;
                        if (addr + 4 <= s->vram_size) {
                            rt_crcs[ri] ^= *(uint32_t *)(vram_ptr + addr);
                            rt_crcs[ri] = (rt_crcs[ri] << 7) | (rt_crcs[ri] >> 25);
                        }
                    }
                }
            }

            /* Diff against previous */
            bool fb_changed = (fb_crc != prev_fb_crc && prev_fb_crc != 0);
            bool any_rt_changed = false;
            {
                int ri;
                for (ri = 0; ri < rt_count; ri++) {
                    if (rt_crcs[ri] != prev_rt_crcs[ri] && prev_rt_crcs[ri] != 0) {
                        any_rt_changed = true;
                    }
                }
            }

            if (fb_changed || any_rt_changed) {
                fprintf(stderr, "[VRAM_WRITE_TRAP] frame=%u fb_changed=%d",
                        trap_frame, fb_changed);
                {
                    int ri;
                    for (ri = 0; ri < rt_count; ri++) {
                        fprintf(stderr, " rt[0x%x]_changed=%d",
                                rt_offsets[ri],
                                (rt_crcs[ri] != prev_rt_crcs[ri]) ? 1 : 0);
                    }
                }
                fprintf(stderr, "\n");
                trap_log++;
            }

            prev_fb_crc = fb_crc;
            {
                int ri;
                for (ri = 0; ri < rt_count; ri++)
                    prev_rt_crcs[ri] = rt_crcs[ri];
            }
        }
    }

    /* 32bpp path: shadow buffer with bswap32 */
    uint64_t frame_size = (uint64_t)stride * height;

    /* (Re)allocate shadow buffer if needed */
    if (!s->shadow_buf || s->shadow_buf_size < frame_size) {
        g_free(s->shadow_buf);
        s->shadow_buf = g_malloc(frame_size);
        s->shadow_buf_size = frame_size;
        mode_changed = true;  /* force surface recreation */
    }

    /*
     * With a GL display (UTM) show VRAM itself, in the guest's byte order:
     * the texture upload swizzles it (ui/console-gl.c), so no CPU copy.
     * gl-zero-copy=off keeps the byte-swapped shadow copy.
     */
    bool zero_copy = s->gl_zero_copy && console_has_gl(s->con) &&
                     s->disp.offset + frame_size <= s->vram_size;
    uint8_t *surface_src = zero_copy ? vram_ptr + s->disp.offset
                                     : (uint8_t *)s->shadow_buf;

    /*
     * Create/replace the surface only when it would actually differ.
     *
     * display_invalid means "the contents changed", which a page flip or a
     * blit sets constantly -- but replacing the surface is not how contents
     * are published.  Each replacement allocates a DisplaySurface and a
     * pixman image, runs every listener's gfx_switch (which copies the whole
     * frame), and frees the old one.  At 1280x1024 that was several
     * megabytes of pointless copying per flip.  Contents reach the UI
     * through dpy_gfx_update below.
     */
    if (mode_changed || width != s->surface_width ||
        height != s->surface_height || stride != s->surface_stride ||
        (void *)surface_src != s->surface_data) {
        s->display_invalid = false;
        s->surface_width = width;
        s->surface_height = height;
        s->surface_stride = stride;
        s->surface_data = surface_src;
        trace_ppc_mac_gpu_mode_change(width, height, s->disp.bpp, stride);
        DisplaySurface *ds = qemu_create_displaysurface_from(
            width, height,
            /* the guest's A,R,G,B bytes, or the LE shadow copy */
            zero_copy ? PIXMAN_BE_x8r8g8b8 : PIXMAN_x8r8g8b8,
            stride,
            surface_src);
        dpy_gfx_replace_surface(s->con, ds);
    } else {
        s->display_invalid = false;
    }

    /*
     * PHASE 1 DIAGNOSTIC: Check if IOFBBlitSurfaceCopy wrote body
     * pixels to VRAM at the NEW drag position.
     *
     * Sample VRAM at the current drag origin (from Metal state).
     * If body-like pixels appear at the NEW position (not the old),
     * IOFBBlitSurfaceCopy IS running. If only zeros/wallpaper,
     * it's NOT running.
     *
     * Also sample at the OLD position for comparison.
     */
    if (ppc_gpu_env_diag() && s->renderer && s->renderer->get_drag_state) {
        static uint32_t prev_diag_ox = 0, prev_diag_oy = 0;
        static uint32_t prev_diag_gen = 0;
        static bool prev_diag_valid = false;

        uint32_t cur_ox, cur_oy, cur_w, cur_h, cur_gen;
        if (s->renderer->get_drag_state(s->renderer_opaque,
                                         &cur_ox, &cur_oy,
                                         &cur_w, &cur_h, &cur_gen)) {
            if (prev_diag_valid &&
                cur_gen != prev_diag_gen &&
                (cur_ox != prev_diag_ox || cur_oy != prev_diag_oy)) {
                /* Drag detected — sample VRAM at both positions */
                uint32_t margin = 50;
                uint32_t sample_y_rel = cur_h / 2;
                uint32_t sample_w = 100;

                /* Sample NEW position body area */
                uint32_t new_y = cur_oy + sample_y_rel;
                uint32_t new_x_start = cur_ox + margin;
                uint32_t new_nz = 0, new_bright = 0, new_zero = 0;
                if (new_y < height && new_x_start + sample_w <= width) {
                    const uint32_t *row_new = (const uint32_t *)(vram_ptr +
                        s->disp.offset + new_y * stride);
                    uint32_t x;
                    for (x = new_x_start; x < new_x_start + sample_w; x++) {
                        uint32_t px = __builtin_bswap32(row_new[x]);
                        if (px == 0) { new_zero++; continue; }
                        new_nz++;
                        uint32_t r = (px >> 16) & 0xFF;
                        uint32_t g = (px >> 8) & 0xFF;
                        uint32_t b = px & 0xFF;
                        if (r > 200 && g > 200 && b > 200) new_bright++;
                    }
                }

                /* Sample OLD position body area */
                uint32_t old_y = prev_diag_oy + sample_y_rel;
                uint32_t old_x_start = prev_diag_ox + margin;
                uint32_t old_nz = 0, old_bright = 0, old_zero = 0;
                if (old_y < height && old_x_start + sample_w <= width) {
                    const uint32_t *row_old = (const uint32_t *)(vram_ptr +
                        s->disp.offset + old_y * stride);
                    uint32_t x;
                    for (x = old_x_start; x < old_x_start + sample_w; x++) {
                        uint32_t px = __builtin_bswap32(row_old[x]);
                        if (px == 0) { old_zero++; continue; }
                        old_nz++;
                        uint32_t r = (px >> 16) & 0xFF;
                        uint32_t g = (px >> 8) & 0xFF;
                        uint32_t b = px & 0xFF;
                        if (r > 200 && g > 200 && b > 200) old_bright++;
                    }
                }

                static int diag_log = 0;
                if (diag_log < 40) {
                    fprintf(stderr, "[DRAG_BODY_CHECK] gen=%u "
                            "old=(%u,%u) new=(%u,%u) "
                            "OLD_body: nz=%u bright=%u zero=%u | "
                            "NEW_body: nz=%u bright=%u zero=%u | "
                            "IOFBBlit=%s\n",
                            cur_gen,
                            prev_diag_ox, prev_diag_oy,
                            cur_ox, cur_oy,
                            old_nz, old_bright, old_zero,
                            new_nz, new_bright, new_zero,
                            (new_bright > 50) ? "LIKELY_YES" : "NO");
                    diag_log++;
                }
            }

            prev_diag_ox = cur_ox;
            prev_diag_oy = cur_oy;
            prev_diag_gen = cur_gen;
            prev_diag_valid = true;
        }
    }

    /* Flush any pending drag body paste — pastes saved body pixels
     * to the new position AFTER all compositor rendering is done. */
    if (s->renderer && s->renderer->flush_drag_paste) {
        s->renderer->flush_drag_paste(s->renderer_opaque, vram_ptr);
    }

    /*
     * Which scanout rows to copy.
     *
     * The VRAM dirty log (kept on for renderers that do not consume it,
     * and fed by every guest CPU store and every device and render-queue
     * write) knows which rows of the framebuffer were written since the
     * last copy.  A page flip invalidates the partial view -- writes to
     * the buffer being flipped *to* may have been consumed by an earlier
     * snapshot while it was still the back buffer -- so the frame after
     * the offset moves is copied whole, as is the frame after any mode
     * change.  Skipping the copy entirely when no row is dirty also
     * skips the UI's full-frame CGImage refresh.
     */
    int y0 = 0, y1 = (int)height;
    DirtyBitmapSnapshot *snap = NULL;

    if (s->scanout_log_owner && ppc_gpu_dirty_scanout) {
        snap = memory_region_snapshot_and_clear_dirty(
                   &s->vram, s->disp.offset, frame_size, DIRTY_MEMORY_VGA);
    }
    if (snap && s->scanout_dirty_valid && !mode_changed &&
        s->scanout_offset == s->disp.offset) {
        int lo = -1, hi = -1, y;
        for (y = 0; y < (int)height; y++) {
            if (memory_region_snapshot_get_dirty(&s->vram, snap,
                    s->disp.offset + (uint64_t)y * stride, width * 4)) {
                if (lo < 0) {
                    lo = y;
                }
                hi = y + 1;
            }
        }
        g_free(snap);
        snap = NULL;
        if (lo < 0) {
            r200_rate.co_skips++;
            return;                 /* nothing was written: nothing to show */
        }
        y0 = lo;
        y1 = hi;
    } else if (snap) {
        g_free(snap);
        snap = NULL;
    }

    /* Bswap copy: VRAM (BE) → shadow buffer (LE) */
    const uint32_t *src = (const uint32_t *)(vram_ptr + s->disp.offset);
    uint32_t *dst = (uint32_t *)s->shadow_buf;
    uint32_t stride_u32 = stride / 4;
    int y;
    r200_rate.co_copies++;
    if (y0 != 0 || y1 != (int)height) {
        r200_rate.co_partials++;
    }
    for (y = y0; y < y1 && !zero_copy; y++) {
        ppc_mac_gpu_bswap_line32(dst + y * stride_u32,
                                  src + y * stride_u32,
                                  width);
    }
    s->scanout_offset = s->disp.offset;
    s->scanout_dirty_valid = s->scanout_log_owner;
    s->scanout_defers = 0;

    if (y0 == 0 && y1 == (int)height) {
        dpy_gfx_update_full(s->con);
    } else {
        dpy_gfx_update(s->con, 0, y0, width, y1 - y0);
    }
}

static const GraphicHwOps ppc_mac_gpu_gfx_ops = {
    .gfx_update = ppc_mac_gpu_display_update,
};

/* ========================================================================
 * Scratch register writeback to VRAM
 *
 * When SCRATCH_UMSK bit N is set, writes to SCRATCH_REG<N> are also
 * written to VRAM at SCRATCH_ADDR + N*4. The kext uses this to check
 * CP command completion by polling a VRAM location.
 * ======================================================================== */

/* Forward declarations for GART translation (defined later in file) */
static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                        hwaddr *phys_out);
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                       hwaddr *phys_out);

static void ppc_mac_gpu_scratch_writeback_val(PPCMacGPUState *s, int reg_idx,
                                             uint32_t wb_val)
{
    if (reg_idx < 0 || reg_idx > 5) {
        return;
    }
    if (!(s->regs.scratch_umsk & (1 << reg_idx))) {
        return;
    }

    uint32_t wb_addr = s->regs.scratch_addr + reg_idx * 4;

    /*
     * The scratch writeback address may be in GART space (system RAM
     * mapped through the AGP/GART translation table) or in VRAM.
     * GART addresses must be translated to physical system RAM addresses.
     */
    hwaddr phys;
    /*
     * Where this lands, and in which byte order, decides whether a program
     * waiting on the card ever wakes up.
     *
     * The address names memory the card reaches over the bus, never a spot
     * in its own memory: the driver builds it from the ring's read-pointer
     * address, which is system RAM.  When address translation is switched
     * off the address passes through untranslated, so a plain guest
     * address is the normal case and must be written as such -- there was
     * no path for it here, and the value went into VRAM or nowhere.
     *
     * The card writes these little-endian whatever the processor's order,
     * which is why the drivers that read them swap.  This wrote them
     * big-endian, so a number like 2200 arrived as 0x98080000: read as a
     * signed difference, permanently "not yet".  Either fault alone leaves
     * a program waiting for ever while everything else looks healthy.
     * POWEREMU_WB_BE=1 restores the old order for comparison.
     */
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t wire_val = getenv("POWEREMU_WB_BE") ? cpu_to_be32(wb_val)
                                                 : cpu_to_le32(wb_val);

    if (ppc_mac_gpu_gart_translate(s, wb_addr, &phys) ||
        ppc_mac_gpu_agp_translate(s, wb_addr, &phys)) {
        address_space_write(&address_space_memory, phys,
                            MEMTXATTRS_UNSPECIFIED, &wire_val, 4);
        gpu_debug_log("SCRATCH_WB reg%d=0x%x -> GART[0x%x] phys=0x%"PRIx64,
                      reg_idx, wb_val, wb_addr, (uint64_t)phys);
    } else if (wb_addr >= fb_base && wb_addr + 4 <= fb_base + s->vram_size) {
        uint32_t off = wb_addr - fb_base;
        r200_vram_access(s, off, off + 4, true, 6);
        uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
        memcpy(vram + off, &wire_val, 4);
        ppc_vram_dirty_e0(s, off, 4);
        gpu_debug_log("SCRATCH_WB reg%d=0x%x -> VRAM[0x%x]",
                      reg_idx, wb_val, off);
    } else {
        /* Straight out to memory, as a card whose translation is off does. */
        address_space_write(&address_space_memory, wb_addr,
                            MEMTXATTRS_UNSPECIFIED, &wire_val, 4);
        gpu_debug_log("SCRATCH_WB reg%d=0x%x -> memory[0x%x]", reg_idx, wb_val,
                      wb_addr);
    }
}

/*
 * Keep the copy of the ring's read pointer that lives in memory.
 *
 * Mac OS X's ATI driver can read the read pointer either from the card or
 * from a copy the card keeps in memory, and it chooses at start-up.  When
 * it reads the copy and nothing ever writes it, the driver finds the ring
 * permanently full: it then waits a millisecond and looks again, a
 * thousand times, for every batch of commands it wants to send.  Halo
 * drew its first frames and then crawled, with its sound looping, because
 * of this.
 */
static void ppc_mac_gpu_rptr_writeback(PPCMacGPUState *s)
{
    uint32_t addr = s->regs.cp_rb_rptr_addr & ~3u;
    hwaddr phys;

    if (!addr || (s->regs.cp_rb_cntl & (1u << 27))) {
        return;                       /* nowhere to write, or the guest said not to */
    }
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t wire_val = getenv("POWEREMU_WB_BE")
        ? cpu_to_be32(s->regs.cp_rb_rptr) : cpu_to_le32(s->regs.cp_rb_rptr);

    if (!ppc_mac_gpu_gart_translate(s, addr, &phys) &&
        !ppc_mac_gpu_agp_translate(s, addr, &phys)) {
        if (addr >= fb_base && addr + 4 <= fb_base + s->vram_size) {
            uint32_t off = addr - fb_base;
            memcpy((uint8_t *)memory_region_get_ram_ptr(&s->vram) + off,
                   &wire_val, 4);
            ppc_vram_dirty_e0(s, off, 4);
        } else {
            address_space_write(&address_space_memory, addr,
                                MEMTXATTRS_UNSPECIFIED, &wire_val, 4);
        }
        return;
    }
    address_space_write(&address_space_memory, phys, MEMTXATTRS_UNSPECIFIED,
                        &wire_val, 4);
}

static void ppc_mac_gpu_scratch_writeback(PPCMacGPUState *s, int reg_idx)
{
    if (reg_idx >= 0 && reg_idx <= 5) {
        ppc_mac_gpu_scratch_writeback_val(s, reg_idx, s->regs.scratch_reg[reg_idx]);
    }
}

/* Tell the renderer about a 2D fill (mirrors depth-buffer clears). */
static void r200_fill_notify(PPCMacGPUState *s, uint32_t offset, uint32_t pitch,
                             uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                             uint32_t bpp, uint32_t value)
{
    if (r200_direct_enabled() && s->renderer && s->renderer->fill_notify_r200) {
        s->renderer->fill_notify_r200(s->renderer_opaque, offset, pitch,
                                      x, y, w, h, bpp, value);
    }
}

/* ---- Direct-R200 asynchronous fences ---- */

static QEMUBH *r200_fence_bh;

/* Perform queued scratch writebacks whose GPU batch has completed. */
static void r200_fence_drain(PPCMacGPUState *s, uint32_t done)
{
    /*
     * Take each fence off the queue before writing it back: the write-back
     * is a guest memory write that can land on the card again, run the ring
     * and drain (or queue) fences re-entrantly.  Counting first and
     * trimming afterwards then trimmed more than was left -- a memmove of
     * ~50 GB and a crashed QEMU in Quake III.
     */
    while (s->regs.r200_fence_n &&
           (int32_t)(done - s->regs.r200_fence_q[0].seq) >= 0) {
        int reg = s->regs.r200_fence_q[0].reg;
        uint32_t val = s->regs.r200_fence_q[0].val;
        s->regs.r200_fence_n--;
        memmove(s->regs.r200_fence_q, s->regs.r200_fence_q + 1,
                s->regs.r200_fence_n * sizeof(s->regs.r200_fence_q[0]));
        ppc_mac_gpu_scratch_writeback_val(s, reg, val);
    }
}

static void r200_fence_bh_cb(void *opaque)
{
    PPCMacGPUState *s = opaque;
    r200_fence_drain(s, qatomic_read(&s->regs.r200_fence_done));
}

/* Called from a Metal completion thread. */
static void r200_fence_done_cb(void *arg, uint32_t seq)
{
    PPCMacGPUState *s = arg;
    qatomic_set(&s->regs.r200_fence_done, seq);
    qemu_bh_schedule(r200_fence_bh);
}

/*
 * The driver reads a scratch register to see how far the GPU has got.
 * Flushing everything for it (as other status reads do) made each fence
 * check wait for all the work queued after that fence too - with deep
 * GPU-side batch chains that was a fifth of the vCPU's time in Warcraft
 * III.  Wait only for the newest fence queued on this register (its work
 * is already committed), then report it.  Returns false if the generic
 * flush is still needed.
 */
static bool r200_scratch_read_wait(PPCMacGPUState *s, int idx)
{
    uint32_t want = 0;
    bool pending = false;

    for (uint32_t i = 0; i < s->regs.r200_fence_n; i++) {
        if (s->regs.r200_fence_q[i].reg == idx) {
            want = s->regs.r200_fence_q[i].seq;
            pending = true;
        }
    }
    if (pending) {
        int64_t deadline = g_get_monotonic_time() + 100000;
        while ((int32_t)(qatomic_read(&s->regs.r200_fence_done) - want) < 0) {
            if (g_get_monotonic_time() >= deadline) {
                return false;                /* something is stuck: flush */
            }
            g_usleep(20);
        }
        r200_fence_drain(s, qatomic_read(&s->regs.r200_fence_done));
    }
    return true;
}

/*
 * A scratch-register write is the driver's fence ("work up to here is
 * done").  If rendering is still in flight, defer the writeback until the
 * GPU finishes it, as the hardware does, instead of stalling for it now.
 */
static void r200_agp_tc_flush(void);
static void r200_scratch_write(PPCMacGPUState *s, int idx, uint32_t val)
{
    uint32_t seq = 0;

    if (s->regs.aic_ctrl & 1) {
        r200_agp_tc_flush();       /* only the card's own GART can remap */
    }
    /*
     * Fence modes (PPCGPU_ASYNC_FENCE): 0 synchronous, 1 deferred, 2 (default)
     * hybrid.  The guest driver polls a fence once and otherwise sleeps with
     * ~10 ms granularity, so the desktop compositor - which fences small
     * batches and waits on them at once - wants the fence done immediately
     * (synchronous: 34 window moves/s vs 22 deferred).  A game fences big
     * frames it does not wait on right away, and a synchronous flush then
     * stalls the vCPU for the whole GPU frame (Warcraft III menu 6-7 fps
     * synchronous, 13-14 deferred).  Hybrid submits, waits up to ~1.5 ms for
     * completion, and defers if the GPU is not done by then; after a timeout
     * it stops waiting for the next 16 fences.
     */
    static int fence_mode = -1;
    static int nowait;
    if (fence_mode < 0) {
        const char *e = getenv("PPCGPU_ASYNC_FENCE");
        fence_mode = e ? (e[0] == '1' ? 1 : e[0] == '0' ? 0 : 2) : 2;
    }
    bool sync_fences = fence_mode == 0;
    if (sync_fences) {
        r200_flush_at(s, R200_WHY_TAG(R200_WHY_FENCE));
        ppc_mac_gpu_scratch_writeback_val(s, idx, val);
        return;
    }
    if (r200_direct_enabled() && s->renderer && s->renderer->submit_r200) {
        if (!r200_fence_bh) {
            r200_fence_bh = qemu_bh_new(r200_fence_bh_cb, s);
        }
        seq = s->renderer->submit_r200(s->renderer_opaque,
                                       r200_fence_done_cb, s);
    }
    if (!seq && s->regs.r200_fence_n) {
        seq = s->regs.r200_fence_last_seq;      /* keep fences in order */
    }
    if (!seq || s->regs.r200_fence_n == ARRAY_SIZE(s->regs.r200_fence_q)) {
        r200_flush_at(s, R200_WHY_TAG(R200_WHY_FENCE));
        ppc_mac_gpu_scratch_writeback_val(s, idx, val);
        return;
    }
    s->regs.r200_fence_q[s->regs.r200_fence_n].seq = seq;
    s->regs.r200_fence_q[s->regs.r200_fence_n].reg = idx;
    s->regs.r200_fence_q[s->regs.r200_fence_n].val = val;
    s->regs.r200_fence_n++;
    s->regs.r200_fence_last_seq = seq;
    if (fence_mode == 2) {
        if (nowait > 0) {
            nowait--;
        } else {
            /*
             * How long the vCPU will spin for the renderer before deferring.
             * Tunable because the right value moves with emulator speed, and
             * it moved: 1500 us was a small part of a 6-14 fps frame and is a
             * large part of a 55 fps one, where this spin was 15% of the vCPU
             * thread.
             *
             * Measured at the Warcraft menu: 1500 us gives 53.6 fps, 400 us
             * 60.8, 150 us 62.0, and never waiting 61.5. The knee is far
             * below the old value, and for a reason worth stating -- a
             * compositor batch is small and completes in tens of
             * microseconds, so a short budget still catches it, while a game
             * frame takes milliseconds and was never going to complete inside
             * 1500 us. The long budget only ever caught what a short one
             * catches, and only ever wasted time on what neither could.
             *
             * 250 us sits inside the flat region with margin for slower
             * machines, where the GPU takes longer to finish the small
             * batches this is here to catch.
             */
            static int wait_us = -1;
            if (wait_us < 0) {
                const char *w = getenv("PPCGPU_FENCE_WAIT_US");
                wait_us = w ? atoi(w) : 250;
            }
            int64_t deadline = g_get_monotonic_time() + wait_us;
            while ((int32_t)(qatomic_read(&s->regs.r200_fence_done) - seq) < 0) {
                if (g_get_monotonic_time() >= deadline) {
                    nowait = 16;
                    break;
                }
                sched_yield();
            }
        }
    }
    r200_fence_drain(s, qatomic_read(&s->regs.r200_fence_done));
}

/* ========================================================================
 * PM4 command processing (PIO FIFO + Indirect Buffer)
 *
 * When the CP is in PIO mode, the kext writes PM4 packets to the
 * FIFO data registers at 0x1000-0x101C. We parse type 0 packets
 * (register writes) and execute them, which triggers scratch
 * writeback if applicable.
 *
 * The kext also submits Indirect Buffers (IB) by writing CP_IB_BASE
 * and CP_IB_BUFSZ via PM4. Writing CP_IB_BUFSZ triggers the CP to
 * read and execute PM4 commands from VRAM at the IB address.
 * ======================================================================== */

/*
 * R300 (ati-radeon-9700) type-3 packets: 3D_LOAD_VBPNTR and the three draw
 * packets go to the R300 pipeline (hw/display/r300/); everything else takes
 * the common path.  With $R300_DUMP set, the 3D state at the first draws
 * (and every 1000th) is written out for reverse engineering.
 */
static uint8_t *r200_agp_page(PPCMacGPUState *s, uint32_t gpu_addr);
static uint32_t g_r300_arm_rt;      /* debug: drag-frame capture target */
static void r300_surfwatch_update(PPCMacGPUState *s);
static void r300_surface_changed(PPCMacGPUState *s);
static void r300_zconv(PPCMacGPUState *s, bool to_card);

/* Raw guest GPU memory (VRAM or AGP), bytes as they lie. */
static bool r300_read_raw(void *opaque, uint32_t gpu_addr, void *dst,
                          uint32_t len)
{
    PPCMacGPUState *s = opaque;
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint8_t *out = dst;

    if (gpu_addr >= fb_base && (uint64_t)gpu_addr - fb_base + len <= s->vram_size) {
        memcpy(dst, (uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                    (gpu_addr - fb_base), len);
        return true;
    }
    while (len) {
        uint8_t *page = r200_agp_page(s, gpu_addr);
        uint32_t n = MIN(len, 0x1000 - (gpu_addr & 0xFFF));
        if (!page) {
            return false;
        }
        memcpy(out, page + (gpu_addr & 0xFFF), n);
        out += n;
        gpu_addr += n;
        len -= n;
    }
    return true;
}

static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                        hwaddr *phys_addr);
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                      hwaddr *phys_addr);

/* A dword the card writes back to guest GPU memory (VRAM or AGP/GART),
 * bytes as given (lowest address first). */
static bool r300_write_raw(PPCMacGPUState *s, uint32_t gpu_addr,
                           const uint8_t bytes[4])
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    hwaddr phys;

    if (gpu_addr >= fb_base && (uint64_t)gpu_addr - fb_base + 4 <= s->vram_size) {
        uint64_t off = gpu_addr - fb_base;
        /* Only batched work that touches these four bytes has to finish
         * first; a write-back next to rendering keeps the GPU running. */
        r200_vram_access(s, off, off + 4, true, 6);
        memcpy((uint8_t *)memory_region_get_ram_ptr(&s->vram) + off, bytes, 4);
        ppc_vram_dirty_e0(s, off, 4);
        return true;
    }
    if (ppc_mac_gpu_agp_translate(s, gpu_addr, &phys) ||
        ppc_mac_gpu_gart_translate(s, gpu_addr, &phys)) {
        cpu_physical_memory_write(phys, bytes, 4);
        return true;
    }
    return false;
}

/* The card's endian swap modes (VC_SWAP, DEPTHENDIAN, ...). */
static uint32_t r300_swap_mode(uint32_t v, unsigned mode)
{
    switch (mode & 3) {
    case 1:  return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    case 2:  return bswap32(v);
    case 3:  return (v << 16) | (v >> 16);
    default: return v;
    }
}

/*
 * ZB_ZPASS_DATA / ZB_ZPASS_ADDR (occlusion queries).  Writing DATA sets
 * the Z-pass counter; writing ADDR stores it at that GPU address, in the
 * depth buffer's endian mode, for the pipes SU_REG_DEST selects.  The
 * renderer counts every sample on one counter, which is reported as
 * pipe 0's (the driver sums the pipes).
 */
static void r300_zpass_write(PPCMacGPUState *s, uint32_t addr, uint32_t val)
{
    PPCMacGPURenderer *r = s->renderer;

    if (addr == 0x4F58) {
        if (r && r->zpass_r300) {
            r->zpass_r300(s->renderer_opaque, true, val);
        }
        s->r3_zpass_active = true;
        return;
    }
    uint32_t dest = r300_reg(s->r3, 0x42C8);            /* SU_REG_DEST */
    uint32_t n = (r && r->zpass_r300) ?
                 r->zpass_r300(s->renderer_opaque, false, 0) : 0;
    uint32_t w = r300_swap_mode((dest & 1) || !dest ? n : 0,
                                (r300_reg(s->r3, 0x4F24) >> 19) & 3);
    uint8_t b[4] = { w, w >> 8, w >> 16, w >> 24 };     /* card stores LE */

    s->r3_zpass_active = false;
    if (!r300_write_raw(s, val & ~3u, b)) {
        qemu_log("ppc-mac-gpu r300: ZPASS_ADDR %08x not mapped\n", val);
    }
}

/*
 * Samples per pixel of the colour or depth buffer at VRAM offset off:
 * GB_AA_CONFIG's, or when that is off now, what the buffer was last drawn
 * with.  A multisampled buffer holds one row per sample for each row
 * (R300DrawPacket.aa_samples), so a clear or resolve covers ns rows.
 */
static uint32_t r300_cb_samples(PPCMacGPUState *s, uint32_t off)
{
    uint32_t ns = r300_aa_samples(s->r3);

    return ns == 1 && off == s->r3_cb_aa_off ? s->r3_cb_aa_ns : ns;
}

static uint32_t r300_zb_samples(PPCMacGPUState *s, uint32_t gpu_off)
{
    uint32_t ns = r300_aa_samples(s->r3);

    return ns == 1 && gpu_off == s->r3_zb_offset ? MAX(s->r3_zb_ns, 1u) : ns;
}

/*
 * 3D_CLEAR_ZMASK (0x32) marks every Z tile as holding ZB_DEPTHCLEARVALUE.
 * Depth is kept uncompressed in memory here, so write the clear value
 * over the buffer (the extent seen at earlier draws, else the scissor),
 * every sample of it.
 */
static void r300_zmask_clear(PPCMacGPUState *s)
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t off = r300_reg(s->r3, 0x4F20) & ~0x1Fu;
    uint32_t pitch = r300_reg(s->r3, 0x4F24);
    uint32_t bpp = (r300_reg(s->r3, 0x4F10) & 0xF) == 2 ? 4 : 2;
    uint32_t rows = ((r300_reg(s->r3, 0x43E4) >> 13) & 0x1FFF);
    uint32_t v = r300_swap_mode(r300_reg(s->r3, 0x4F28), (pitch >> 19) & 3);
    uint64_t bpr = (uint64_t)(pitch & 0x3FFC) * bpp;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

    rows = rows > 1440 ? rows - 1440 + 1 : 0;
    if (s->r3_zb_offset == off && s->r3_zb_height > rows) {
        rows = s->r3_zb_height;
    }
    rows *= r300_zb_samples(s, off);
    if (off < fb_base || !bpr || !rows) {
        return;
    }
    off -= fb_base;
    if (off + bpr * rows > s->vram_size) {
        rows = (s->vram_size - off) / bpr;
    }
    r200_vram_access(s, off, off + bpr * rows, true, 7);
    r200_pattern_fill(vram + off, bpr * rows, v, bpp);
    ppc_vram_dirty_e0(s, off, bpr * rows);
    static int logged;
    if (logged++ < 4) {
        qemu_log("ppc-mac-gpu r300: 3D_CLEAR_ZMASK: depth at 0x%x, %u rows of "
                 "%llu bytes <- %08x\n", off, rows, (unsigned long long)bpr,
                 r300_reg(s->r3, 0x4F28));
    }
}

/*
 * 3D_CLEAR_CMASK (type-3 0x38): a fast colour clear.  The card marks the
 * colour buffer's CMASK tiles cleared and fills them with
 * RB3D_COLOR_CLEAR_VALUE (ARGB) as it next reads them.  No CMASK is kept
 * here, so the clear goes straight into colour buffer 0, over the scissor
 * height (all of its samples), in the buffer's COLOR_ENDIAN order.  Apple's GL driver clears a
 * window's back buffer this way each frame (Chess).
 */
static void r300_cmask_clear(PPCMacGPUState *s)
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t off = r300_reg(s->r3, 0x4E28) & ~0x1Fu;
    uint32_t pitch = r300_reg(s->r3, 0x4E38);
    uint32_t cf = (pitch >> 21) & 0xF, endian = (pitch >> 19) & 3;
    uint32_t rows = (r300_reg(s->r3, 0x43E4) >> 13) & 0x1FFF;
    uint32_t v = r300_reg(s->r3, 0x4E14);
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t bpp;
    uint64_t bpr;

    switch (cf) {
    case 6:                     /* ARGB8888 */
        bpp = 4;
        break;
    case 4:                     /* RGB565 */
        bpp = 2;
        v = ((v >> 8) & 0xF800) | ((v >> 5) & 0x07E0) | ((v >> 3) & 0x001F);
        break;
    default:
        qemu_log_mask(LOG_UNIMP, "ppc-mac-gpu r300: 3D_CLEAR_CMASK on colour format %u\n", cf);
        return;
    }
    bpr = (uint64_t)(pitch & 0x3FFE) * bpp;
    rows = rows > 1440 ? rows - 1440 + 1 : 0;
    if (off < fb_base || !bpr || !rows) {
        return;
    }
    off -= fb_base;
    rows *= r300_cb_samples(s, off);     /* every sample's rows */
    if (off + bpr * rows > s->vram_size) {
        rows = (s->vram_size - off) / bpr;
    }
    r200_vram_access(s, off, off + bpr * rows, true, 8);
    if (bpp == 4) {
        r200_pattern_fill(vram + off, bpr * rows, r300_swap_mode(v, endian), 4);
    } else {
        r200_pattern_fill(vram + off, bpr * rows,
                          endian == 1 || endian == 2 ? bswap16(v) : v, 2);
    }
    ppc_vram_dirty_e0(s, off, bpr * rows);
    static int logged;
    if (logged++ < 4) {
        qemu_log("ppc-mac-gpu r300: 3D_CLEAR_CMASK: colour at 0x%x, %u rows of "
                 "%llu bytes <- %08x\n", off, rows, (unsigned long long)bpr, v);
    }
}

static bool r300_to_vram(PPCMacGPUState *s, uint32_t *addr, uint64_t len)
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;

    if (*addr < fb_base || (uint64_t)*addr - fb_base + len > s->vram_size) {
        return false;
    }
    *addr -= fb_base;
    return true;
}

static void r300_warn_once(const char *what, const char *detail)
{
    static GHashTable *seen;

    if (!seen) {
        seen = g_hash_table_new(g_str_hash, g_str_equal);
    }
    if (!g_hash_table_contains(seen, what)) {
        g_hash_table_add(seen, (gpointer)what);
        qemu_log("ppc-mac-gpu r300: %s%s%s\n", what, detail ? ": " : "",
                 detail ? detail : "");
    }
}

/* $R300_DRAWLOG: one line per draw (and per other type-3 packet) with the
 * state that decides depth, culling, tiling and texturing; capped. */
static FILE *r300_drawlog(void)
{
    static FILE *f;
    static int init;
    static unsigned lines;

    if (!init) {
        const char *p = getenv("R300_DRAWLOG");
        init = 1;
        f = p ? fopen(p, "w") : NULL;
    }
    if (f && ++lines > 200000) {
        fclose(f);
        f = NULL;
    }
    return f;
}

static void r300_drawlog_draw(PPCMacGPUState *s, uint32_t opcode,
                              const uint32_t *d, uint32_t body_dw)
{
    FILE *f = r300_drawlog();
    const R300State *r = s->r3;

    if (!f) {
        return;
    }
    fprintf(f, "D%llu op%02x vf=%08x n=%u rt=%08x/%08x zb=%x zs=%08x rm=%08x "
            "zf=%x zo=%08x zp=%08x bw=%x cull=%x pm=%x vte=%x vc=%x cb=%08x "
            "ab=%08x cm=%x af=%x ten=%x aa=%x ms=%08x/%08x res=%x",
            (unsigned long long)r->draws, opcode, body_dw ? d[0] : 0, body_dw,
            r300_reg(r, 0x4E28), r300_reg(r, 0x4E38), r300_reg(r, 0x4F00),
            r300_reg(r, 0x4F04), r300_reg(r, 0x4F08), r300_reg(r, 0x4F10),
            r300_reg(r, 0x4F20), r300_reg(r, 0x4F24), r300_reg(r, 0x4F1C),
            r300_reg(r, 0x42B8), r300_reg(r, 0x4288), r300_reg(r, 0x20B0),
            r300_reg(r, 0x2140), r300_reg(r, 0x4E04), r300_reg(r, 0x4E08),
            r300_reg(r, 0x4E0C), r300_reg(r, 0x4BD4), r300_reg(r, 0x4104),
            r300_reg(r, 0x4020), r300_reg(r, 0x4010), r300_reg(r, 0x4014),
            r300_reg(r, 0x4E88));
    for (int t = 0; t < 16; t++) {
        if (r300_reg(r, 0x4104) & (1u << t)) {
            fprintf(f, " t%d=%08x/%08x/%08x/%08x/%08x", t,
                    r300_reg(r, 0x4540 + 4 * t), r300_reg(r, 0x4480 + 4 * t),
                    r300_reg(r, 0x44C0 + 4 * t), r300_reg(r, 0x4500 + 4 * t),
                    r300_reg(r, 0x4400 + 4 * t));
        }
    }
    fprintf(f, "\n");
    fflush(f);
}

/* Log each texture and colour-buffer format combination once: which
 * formats a guest workload actually uses (and with which swaps). */
static void r300_note_formats(PPCMacGPUState *s, const R300DrawPacket *pkt)
{
    static uint32_t seen[128];
    static unsigned nseen;
    uint32_t keys[1 + R300_NUM_TEX_UNITS];
    unsigned nk = 0;

    keys[nk++] = 0x80000000u | (pkt->rt_format << 24) |
                 ((r300_reg(s->r3, 0x4E38) >> 19) & 3) << 20 |
                 (r300_reg(s->r3, 0x46A4) & 0xF001F);
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &pkt->tex[t];
        uint32_t f1 = r300_reg(s->r3, 0x44C0 + 4 * t);
        if (td->bound) {
            keys[nk++] = ((f1 & 0x1F) << 24) | (f1 & 0x1E0) |
                         (r300_reg(s->r3, 0x4540 + 4 * t) & 3) << 20 |
                         ((f1 >> 21) & 1) << 19 | ((f1 >> 22) & 1) << 18 |
                         (td->host_data ? 1u << 17 : 0) |
                         ((r300_reg(s->r3, 0x4400 + 4 * t) >> 9) & 0xF);
        }
    }
    for (unsigned i = 0; i < nk && nseen < ARRAY_SIZE(seen); i++) {
        unsigned j;
        for (j = 0; j < nseen && seen[j] != keys[i]; j++) {
        }
        if (j < nseen) {
            continue;
        }
        seen[nseen++] = keys[i];
        if (keys[i] & 0x80000000u) {
            qemu_log("ppc-mac-gpu r300: colour buffer format %u endian %u "
                     "US_OUT_FMT %05x (%ux%u at %08x)\n", pkt->rt_format,
                     (keys[i] >> 20) & 3, keys[i] & 0xF001F, pkt->rt_pitch,
                     pkt->rt_height, pkt->rt_gpu_addr);
        } else {
            qemu_log("ppc-mac-gpu r300: texture format 0x%02x swap %u "
                     "sign %x gamma %u yuv %u %s filter %x\n", keys[i] >> 24,
                     (keys[i] >> 20) & 3, (keys[i] >> 5) & 0xF,
                     (keys[i] >> 19) & 1, (keys[i] >> 18) & 1,
                     (keys[i] >> 17) & 1 ? "GART" : "VRAM", keys[i] & 0xF);
        }
    }
}

/* Log the first draw using each rendering feature (vm/gpu-trace.log), so
 * guest workloads show which of them they exercise. */
static void r300_note_features(PPCMacGPUState *s, const R300DrawPacket *pkt)
{
    const R300State *r = s->r3;
    static uint32_t seen;
    uint32_t f = 0;
    static const char *const names[] = {
        "mipmapped texture", "3D texture", "cube map", "border colour wrap",
        "anisotropic filter", "LOD bias", "fog", "polygon mode", "polygon offset",
        "wide lines", "line stipple", "flat/solid shading", "two-sided colours",
        "user clip planes", "multiple render targets", "logic op",
        "depth from the fragment program", "discard src pixels",
        "vertex program flow control", "per-vertex point size", "signed texture",
        "sRGB texture",
    };

    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &pkt->tex[t];
        uint32_t f0 = td->filter0, f1 = r300_reg(r, 0x44C0 + 4 * t);
        if (!td->bound) {
            continue;
        }
        f |= (td->levels > 1) << 0 | (td->dim == R300_TEXDIM_3D) << 1 |
             (td->dim == R300_TEXDIM_CUBE) << 2;
        for (int a = 0; a < 3; a++) {
            unsigned m = (f0 >> (3 * a)) & 7;
            f |= (m == 6 || m == 7) << 3;
        }
        f |= (((f0 >> 9) & 3) == 3 || ((f0 >> 11) & 3) == 3) << 4;
        f |= (((td->filter1 >> 3) & 0x3FF) != 0) << 5;
        f |= (((f1 >> 5) & 0xF) != 0) << 20 | ((f1 >> 21) & 1) << 21;
    }
    f |= (r300_reg(r, 0x4BC0) & 1) << 6;
    f |= ((r300_reg(r, 0x4288) & 3) == 1) << 7;
    f |= ((r300_reg(r, 0x42B4) & 3) != 0) << 8;
    f |= ((r300_reg(r, 0x4234) & 0xFFFF) > 9) << 9;
    f |= ((r300_reg(r, 0x4238) & ~3u) != 0) << 10;
    for (int c = 0; c < 4; c++) {
        uint32_t cc = r300_reg(r, 0x4278);
        f |= (((cc >> (4 * c)) & 3) != 2 || ((cc >> (4 * c + 2)) & 3) != 2) << 11;
    }
    f |= ((r300_reg(r, 0x2090) & (3u << 3)) != 0) << 12;
    f |= ((r300_reg(r, 0x221C) & 0x3F) != 0) << 13;
    f |= (pkt->num_cb > 1) << 14;
    f |= ((r300_reg(r, 0x4E18) >> 2) & 1) << 15;
    f |= ((r300_reg(r, 0x4BD8) & 1) && pkt->glsl && strstr(pkt->glsl, "ow = ar")) << 16;
    f |= ((r300_reg(r, 0x4E04) & 1) && ((r300_reg(r, 0x4E04) >> 3) & 7)) << 17;
    f |= (r300_reg(r, 0x22DC) != 0) << 18;
    f |= ((r300_reg(r, 0x2090) >> 16) & 1) << 19;
    f &= ~seen;
    for (unsigned i = 0; f; i++, f >>= 1) {
        if (f & 1) {
            seen |= 1u << i;
            qemu_log("ppc-mac-gpu r300: first use of %s (draw %llu)\n", names[i],
                     (unsigned long long)r->draws);
        }
    }
}

/*
 * RB3D_AARESOLVE_CTL.AARESOLVE_MODE: the colour buffer is in resolve mode,
 * and the pixels a draw covers are filtered from it into the resolve
 * buffer (RB3D_AARESOLVE_OFFSET/PITCH) instead of being rendered.  Apple's
 * GL driver swaps a window's back buffer this way, with one point sprite
 * over the drawable.  The filter is the mean of the pixel's samples, with
 * AARESOLVE_GAMMA 2.2 averaging linear light and AARESOLVE_ALPHA taking
 * sample 0's alpha or the mean.  Without multisampling it is a copy.
 */
static void r300_aa_resolve(PPCMacGPUState *s, const R300DrawPacket *pkt)
{
    uint32_t ctl = r300_reg(s->r3, 0x4E88);
    uint32_t dst = r300_reg(s->r3, 0x4E80) & ~0x1Fu;
    uint32_t dpitch = (r300_reg(s->r3, 0x4E84) & 0x3FFE) * 4;
    uint32_t spitch = pkt->rt_pitch * pkt->rt_bpp;
    uint32_t ns = r300_cb_samples(s, pkt->rt_gpu_addr);
    uint64_t srow = (uint64_t)spitch * ns;          /* a row of every sample */
    float x0 = pkt->rt_width, y0 = pkt->rt_height, x1 = 0, y1 = 0;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

    for (uint32_t i = 0; i < pkt->num_verts; i++) {
        const float *p = pkt->verts[i].pos;
        float w = p[3] != 0.0f ? p[3] : 1.0f;
        float x = (p[0] / w + 1.0f) * pkt->rt_width * 0.5f;
        float y = (1.0f - p[1] / w) * pkt->rt_height * 0.5f;
        x0 = MIN(x0, x); x1 = MAX(x1, x);
        y0 = MIN(y0, y); y1 = MAX(y1, y);
    }
    int ix0 = MAX((int)floorf(x0 + 0.5f), (int)pkt->scissor[0]);
    int iy0 = MAX((int)floorf(y0 + 0.5f), (int)pkt->scissor[1]);
    int ix1 = MIN((int)floorf(x1 + 0.5f), (int)MIN(pkt->scissor[2], pkt->rt_width));
    int iy1 = MIN((int)floorf(y1 + 0.5f), (int)MIN(pkt->scissor[3], pkt->rt_height));
    uint32_t w = ix1 > ix0 ? ix1 - ix0 : 0, h = iy1 > iy0 ? iy1 - iy0 : 0;

    if (!w || !h || !dpitch || (uint32_t)ix1 * 4 > dpitch) {
        return;
    }
    if (!r300_to_vram(s, &dst, (uint64_t)dpitch * iy1)) {
        r300_warn_once("AA resolve buffer outside VRAM", NULL);
        return;
    }
    if ((uint64_t)pkt->rt_gpu_addr + srow * iy1 > s->vram_size) {
        r300_warn_once("multisampled colour buffer outside VRAM", NULL);
        return;
    }
    {
        bool need_bql = !bql_locked();
        if (need_bql) {
            bql_lock();
        }
        /* Wait only for batched work that races the read of the samples
         * or the write of the resolve buffer. */
        r200_vram_access(s, pkt->rt_gpu_addr,
                         pkt->rt_gpu_addr + srow * iy1, false, 9);
        r200_vram_access(s, dst, dst + (uint64_t)dpitch * iy1, true, 10);
        if (need_bql) {
            bql_unlock();
        }
    }
    /* 8 bits a channel averages; other formats keep sample 0. */
    bool avg = ns > 1 && pkt->rt_bpp == 4 &&
               pkt->rt_format == 6;                    /* ARGB8888 */
    if (ns > 1 && !avg) {
        r300_warn_once("AA resolve of a colour format other than 8888: "
                       "sample 0 kept", NULL);
    }
    if (!avg) {
        for (int y = iy0; y < iy1; y++) {
            memmove(vram + dst + (uint64_t)y * dpitch + ix0 * 4,
                    vram + pkt->rt_gpu_addr + (uint64_t)y * srow + ix0 * 4,
                    (size_t)w * 4);
        }
    } else {
        static float degamma[256];
        static uint8_t regamma[4096];
        bool gamma = ctl & 2, alpha_avg = ctl & 4;
        /* The byte alpha lands in: the ARGB dword in COLOR_ENDIAN order,
         * stored little-endian (as 3D_CLEAR_CMASK writes it). */
        uint32_t amask = r300_swap_mode(0xFF000000u,
                                        (r300_reg(s->r3, 0x4E38) >> 19) & 3);
        int ab = ctz32(amask) / 8;

        if (gamma && degamma[255] == 0.0f) {
            for (int i = 0; i < 256; i++) {
                degamma[i] = powf(i / 255.0f, 2.2f);
            }
            for (int i = 0; i < 4096; i++) {
                regamma[i] = (uint8_t)lrintf(255.0f * powf(i / 4095.0f, 1.0f / 2.2f));
            }
        }
        for (int y = iy0; y < iy1; y++) {
            const uint8_t *sp = vram + pkt->rt_gpu_addr + (uint64_t)y * srow + ix0 * 4;
            uint8_t *dp = vram + dst + (uint64_t)y * dpitch + ix0 * 4;
            for (uint32_t x = 0; x < w; x++, sp += 4, dp += 4) {
                for (int c = 0; c < 4; c++) {
                    if (c == ab && !alpha_avg) {
                        dp[c] = sp[c];
                    } else if (gamma && c != ab) {
                        float sum = 0;
                        for (uint32_t k = 0; k < ns; k++) {
                            sum += degamma[sp[k * spitch + c]];
                        }
                        dp[c] = regamma[(int)lrintf(sum / ns * 4095.0f)];
                    } else {
                        uint32_t sum = ns / 2;
                        for (uint32_t k = 0; k < ns; k++) {
                            sum += sp[k * spitch + c];
                        }
                        dp[c] = sum / ns;
                    }
                }
            }
        }
    }
    ppc_vram_dirty_e0(s, dst + (uint64_t)iy0 * dpitch,
                            (uint64_t)dpitch * h);
    static int logged;
    if (logged++ < 3) {
        qemu_log("ppc-mac-gpu r300: AA resolve %06x/%u -> %06x/%u (%d,%d %ux%u), "
                 "%u samples%s\n", pkt->rt_gpu_addr, spitch, dst, dpitch, ix0,
                 iy0, w, h, ns, avg && (ctl & 2) ? ", gamma 2.2" : "");
    }
}

/*
 * Colour buffers written with COLOR_ENDIAN 0 hold the card's own byte
 * order, while RGBA8 texture decode assumes the CPU's (reversed for no
 * TXO swap, see r300_draw.c).  Leopard's Dock renders its stack and
 * icon-flash layers that way (tiled, COLOR_ENDIAN 0, sampled back with
 * an R=X swizzle): read with the CPU's order, alpha landed in red and
 * the fades flashed red/pink.  Remember the VRAM ranges last rendered
 * like that and decode RGBA8 textures that start at one of them the
 * other way round.
 *
 * A record ends when anything else writes over its range: a CPU-order
 * render or a device write (ppc_vram_dirty_e0), or the guest CPU, which
 * the VGA dirty log shows (cleared for the range when the record is
 * made).  Without that, System Preferences' Aqua controls, uploaded by
 * the CPU where a Core Image layer had been, came out pink and faded.
 * Only textures at a record's base count: the Dock samples its layers
 * from there, and "anywhere inside" caught unrelated textures.
 */
#define R300_E0_RANGES 64
static struct { uint32_t base, size; } r300_e0[R300_E0_RANGES];

static void r300_e0_forget(uint64_t base, uint64_t len)
{
    for (int i = 0; i < R300_E0_RANGES; i++) {
        if (r300_e0[i].size && base < (uint64_t)r300_e0[i].base + r300_e0[i].size &&
            r300_e0[i].base < base + len) {
            r300_e0[i].size = 0;
        }
    }
}

/* A colour buffer about to be written: COLOR_ENDIAN 0 (e0) or not. */
static void r300_e0_note(uint32_t base, uint32_t size, bool e0)
{
    r300_e0_forget(base, size);
    if (e0) {
        static unsigned next;
        int slot = -1;
        for (int i = 0; i < R300_E0_RANGES && slot < 0; i++) {
            if (!r300_e0[i].size) {
                slot = i;
            }
        }
        if (slot < 0) {
            slot = (int)(next++ % R300_E0_RANGES);
        }
        r300_e0[slot].base = base;
        r300_e0[slot].size = size;
    }
}

/* After the draw marked its colour buffer dirty: start watching the
 * record's range for CPU writes (off-screen only; the scanout keeps its
 * own dirty bits). */
static bool r300_e0_on_screen(PPCMacGPUState *s, uint32_t base, uint32_t size)
{
    uint64_t fb0 = s->disp.offset;
    uint64_t fb1 = fb0 + (uint64_t)s->disp.stride * s->disp.height;

    return base < fb1 && fb0 < (uint64_t)base + size;
}

static void r300_e0_arm(PPCMacGPUState *s, uint32_t base, uint32_t size)
{
    if (r300_e0_on_screen(s, base, size)) {
        return;
    }
    for (int i = 0; i < R300_E0_RANGES; i++) {
        if (r300_e0[i].size && r300_e0[i].base == base) {
            memory_region_reset_dirty(&s->vram, base, size, DIRTY_MEMORY_VGA);
            return;
        }
    }
}

static bool r300_e0_has(PPCMacGPUState *s, uint32_t addr)
{
    for (int i = 0; i < R300_E0_RANGES; i++) {
        if (r300_e0[i].size && addr == r300_e0[i].base) {
            uint32_t len = r300_e0[i].size;
            /* Watched ranges only (see r300_e0_arm); reading the log
             * clears it, which nothing else uses off-screen. */
            if (s->renderer && !s->renderer->set_dirty_source &&
                !r300_e0_on_screen(s, addr, len)) {
                DirtyBitmapSnapshot *snap =
                    memory_region_snapshot_and_clear_dirty(&s->vram, addr, len,
                                                           DIRTY_MEMORY_VGA);
                bool written = memory_region_snapshot_get_dirty(&s->vram, snap,
                                                                addr, len);
                g_free(snap);
                if (written) {
                    r300_e0[i].size = 0;    /* written since: CPU data now */
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

static void r300_render(PPCMacGPUState *s, uint32_t opcode, const uint32_t *d,
                        uint32_t body_dw, const R300Indices *idx)
{
    R300DrawPacket pkt;
    const char *err;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

    r300_drawlog_draw(s, opcode, d, body_dw);

    if (s->r3_dump) {
        /* Each distinct texture setup once, for format/layout work. */
        static GHashTable *seen_tex;
        uint32_t en = r300_reg(s->r3, 0x4104);

        if (!seen_tex) {
            seen_tex = g_hash_table_new(g_str_hash, g_str_equal);
        }
        for (int t = 0; t < 16; t++) {
            if (!(en & (1u << t))) {
                continue;
            }
            uint32_t f0 = r300_reg(s->r3, 0x4480 + 4 * t);
            uint32_t f1 = r300_reg(s->r3, 0x44C0 + 4 * t);
            uint32_t f2 = r300_reg(s->r3, 0x4500 + 4 * t);
            uint32_t off = r300_reg(s->r3, 0x4540 + 4 * t);
            uint32_t fl0 = r300_reg(s->r3, 0x4400 + 4 * t);
            char *key = g_strdup_printf("%08x %08x %08x %x %08x", f0, f1, f2,
                                        off & 0x1F, fl0);
            if (g_hash_table_contains(seen_tex, key)) {
                g_free(key);
                continue;
            }
            g_hash_table_add(seen_tex, key);
            fprintf(s->r3_dump, "TEX unit%d fmt %02x %ux%u pitch %u%s off %08x "
                    "flags %x f0 %08x f1 %08x f2 %08x filter %08x rt %08x/%u\n",
                    t, f1 & 0x1F, (f0 & 0x7FF) + 1, ((f0 >> 11) & 0x7FF) + 1,
                    (f2 & 0x3FFF) + 1, (f0 >> 31) ? "" : " (pow2)", off & ~0x1Fu,
                    off & 0x1F, f0, f1, f2, fl0,
                    r300_reg(s->r3, 0x4E28), r300_reg(s->r3, 0x4E38) & 0x3FFE);
        }
    }
    {
        uint32_t f0 = r300_reg(s->r3, 0x4480), f2 = r300_reg(s->r3, 0x4500);
        if ((r300_reg(s->r3, 0x4E28) & ~0x1Fu) == 0) {
            seq_log("R3DCLIP rule=%04x r0=%08x/%08x r1=%08x/%08x r2=%08x/%08x "
                    "r3=%08x/%08x sc=%08x/%08x zb=%08x cb=%08x mask=%08x",
                    r300_reg(s->r3, 0x43D0), r300_reg(s->r3, 0x43B0),
                    r300_reg(s->r3, 0x43B4), r300_reg(s->r3, 0x43B8),
                    r300_reg(s->r3, 0x43BC), r300_reg(s->r3, 0x43C0),
                    r300_reg(s->r3, 0x43C4), r300_reg(s->r3, 0x43C8),
                    r300_reg(s->r3, 0x43CC), r300_reg(s->r3, 0x43E0),
                    r300_reg(s->r3, 0x43E4), r300_reg(s->r3, 0x4F00),
                    r300_reg(s->r3, 0x4E04), r300_reg(s->r3, 0x4E0C));
        }
        seq_log("R3D  rt=%06x/%u tex0=%s%06x %ux%u/%u vf=%08x v0=%08x %08x",
                r300_reg(s->r3, 0x4E28) & ~0x1Fu,
                (r300_reg(s->r3, 0x4E38) & 0x3FFE) * 4,
                (r300_reg(s->r3, 0x4104) & 1) ? "" : "(off)",
                r300_reg(s->r3, 0x4540) & ~0x1Fu, (f0 & 0x7FF) + 1,
                ((f0 >> 11) & 0x7FF) + 1, ((f2 & 0x3FFF) + 1) * 4,
                body_dw ? d[0] : 0, body_dw > 1 ? d[1] : 0,
                body_dw > 2 ? d[2] : 0);
    }
    if (opcode != 0x35) {
        uint32_t aos[1 + 3 * 8];
        for (int k = 0; k < 1 + 3 * 8; k++) {
            aos[k] = r300_reg(s->r3, 0x20C0 + 4 * k);
        }
        r300_load_vbpntr(&s->r3_arrays, aos, 1 + 3 * 8);
    }
    if (s->r3_dump) {
        static GHashTable *seen_rt;
        char *key = g_strdup_printf("%08x %08x", r300_reg(s->r3, 0x4E28),
                                    r300_reg(s->r3, 0x4E38));
        if (!seen_rt) {
            seen_rt = g_hash_table_new(g_str_hash, g_str_equal);
        }
        if (!g_hash_table_contains(seen_rt, key)) {
            g_hash_table_add(seen_rt, key);
            fprintf(s->r3_dump, "RT %s\n", key);
        } else {
            g_free(key);
        }
    }
    s->r3->glsl_flags = s->renderer ? s->renderer->r300_glsl_flags : 0;
    {
        /* R300_CPU_VS=1: run every vertex program on the interpreter */
        static int cpu_vs = -1;
        if (cpu_vs < 0) {
            cpu_vs = getenv("R300_CPU_VS") != NULL;
        }
        if (cpu_vs) {
            s->r3->glsl_flags &= ~R300_GLSL_GPU_VS;
        }
    }
    bool build_ok = idx ?
        r300_draw_build_indexed(s->r3, &s->r3_arrays, d[0], idx, r300_read_raw,
                                s, &pkt, &err) :
        r300_draw_build(s->r3, &s->r3_arrays, opcode, d, body_dw,
                        r300_read_raw, s, &pkt, &err);
    if (s->r3_dump && g_r300_arm_rt &&
        (r300_reg(s->r3, 0x4E28) & ~0x1Fu) == g_r300_arm_rt) {
        fprintf(s->r3_dump, "TRY op %02x vf %08x ok %d err %s tex0 en %u off %06x "
                "fmt %08x/%08x/%08x scissor %u,%u-%u,%u verts %u\n", opcode,
                body_dw ? d[0] : 0, build_ok, build_ok ? "-" : err,
                r300_reg(s->r3, 0x4104) & 1, r300_reg(s->r3, 0x4540),
                r300_reg(s->r3, 0x4480), r300_reg(s->r3, 0x44C0),
                r300_reg(s->r3, 0x4500), pkt.scissor[0], pkt.scissor[1],
                pkt.scissor[2], pkt.scissor[3], build_ok ? pkt.num_verts : 0);
        for (uint32_t v = 0; build_ok && v < pkt.num_verts && v < 48; v++) {
            const R300Vertex *x = &pkt.verts[v];
            fprintf(s->r3_dump, "   v%u ndc %.3f %.3f w %.3f  t %.4f %.4f %.4f %.4f"
                    " | %.4f %.4f %.4f %.4f\n", v, x->pos[0] / x->pos[3],
                    x->pos[1] / x->pos[3], x->pos[3], x->v[0][0], x->v[0][1],
                    x->v[0][2], x->v[0][3], x->v[1][0], x->v[1][1], x->v[1][2],
                    x->v[1][3]);
        }
    }
    if (!build_ok) {
        static int fail_dumps;
        r300_warn_once(err, NULL);
        if (s->r3_dump && fail_dumps++ < 3) {
            fprintf(s->r3_dump, "--- draw failed (%s): op %02x vf %08x vtx_size %u "
                    "psc %08x %08x %08x %08x ext %08x %08x, %u arrays:",
                    err, opcode, body_dw ? d[0] : 0, r300_reg(s->r3, 0x20B4),
                    r300_reg(s->r3, 0x2150), r300_reg(s->r3, 0x2154),
                    r300_reg(s->r3, 0x2158), r300_reg(s->r3, 0x215C),
                    r300_reg(s->r3, 0x21E0), r300_reg(s->r3, 0x21E4),
                    s->r3_arrays.count);
            for (unsigned i = 0; i < s->r3_arrays.count; i++) {
                fprintf(s->r3_dump, " [%08x size %u stride %u]",
                        s->r3_arrays.a[i].addr, s->r3_arrays.a[i].size_dw,
                        s->r3_arrays.a[i].stride_dw);
            }
            fprintf(s->r3_dump, "\n");
        }
        return;
    }
    if (pkt.warn & R300_WARN_PVS)      r300_warn_once("vertex program uses unimplemented features", NULL);
    if (pkt.warn & R300_WARN_DEPTH)    r300_warn_once("depth format 13E3 or reserved: treated as integer Z", NULL);
    if (pkt.warn & R300_WARN_TEXFMT)   r300_warn_once("texture format not implemented yet", NULL);
    if (pkt.warn & R300_WARN_RTFMT)    r300_warn_once("colour buffer endian swap across pixels ignored", NULL);
    if (pkt.warn & R300_WARN_VTXFMT)   r300_warn_once("vertex format not implemented", NULL);
    if (pkt.warn & R300_WARN_VARYINGS) r300_warn_once("too many interpolants", NULL);
    if (pkt.warn & R300_WARN_FLOW)     r300_warn_once("vertex program flow control ran away", NULL);

    uint32_t ns = pkt.aa_samples;
    if (!r300_to_vram(s, &pkt.rt_gpu_addr,
                      (uint64_t)pkt.rt_pitch * pkt.rt_bpp * pkt.rt_height * ns)) {
        r300_warn_once("colour buffer outside VRAM", NULL);
        r300_draw_free(&pkt);
        return;
    }
    /* Render targets B-D (multiple render targets / multiwrites) */
    for (uint32_t k = 1; k < pkt.num_cb; k++) {
        R300ColorDesc *c = &pkt.cb[k];
        if (!r300_to_vram(s, &c->gpu_addr,
                          (uint64_t)c->pitch * c->bpp * pkt.rt_height * ns)) {
            r300_warn_once("render target B-D outside VRAM", NULL);
            pkt.num_cb = k;
            break;
        }
    }
    /* Only layers written red first (US_OUT_FMT_0 C0_SEL = R, as the Dock
     * and Core Image do): System Preferences also renders COLOR_ENDIAN 0
     * layers with C0 = B, and those read right as they are. */
    r300_e0_note(pkt.rt_gpu_addr, pkt.rt_pitch * pkt.rt_bpp * pkt.rt_height * ns,
                 pkt.rt_bpp == 4 && pkt.uniforms.rt_endian == 0 &&
                 ((r300_reg(s->r3, 0x46A4) >> 8) & 3) == 1);
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        R300TexDesc *td = &pkt.tex[t];
        uint64_t tlen = td->size_bytes;     /* the whole mip chain */
        if (!td->bound) {
            continue;
        }
        uint32_t ga = td->gpu_addr;
        if (r300_to_vram(s, &td->gpu_addr, tlen)) {
            /* Only as the guest samples its own card-order layers: red
             * from X (RGBA order).  CPU uploads read as ARGB (red from Z)
             * that land where a layer was stay as they are, whatever
             * write path we failed to see. */
            if (td->kind == R300_TEXK_RGBA8 && !(td->endian & 1) &&
                pkt.uniforms.tex_swz[t][0] == 0 &&
                r300_e0_has(s, td->gpu_addr)) {
                pkt.uniforms.tex_info[t][1] ^= 1;
            }
            continue;
        }
        /* AGP/GART texture: copy the texels out of guest memory (freed
         * by r300_draw_free with free()). */
        td->host_data = tlen && tlen <= 64 * 1024 * 1024 ? malloc(tlen) : NULL;
        if (!td->host_data || !r300_read_raw(s, ga, td->host_data, tlen)) {
            r300_warn_once("texture outside VRAM and the GART", NULL);
            free(td->host_data);
            td->host_data = NULL;
            td->bound = false;
            pkt.uniforms.tex_info[t][0] = 0;
            continue;
        }
        /*
         * 32-bit texels decode in the shader (tex_info.y): reversed for
         * no TXO swap, as is for a 32-bit swap, which is right for VRAM.
         * GART textures with a 32-bit swap read reversed as well (Leopard
         * draws list selections from one: as is, the blue highlight came
         * out purple), while unswapped ones (Aqua buttons) decode as in
         * VRAM.  Measured on Leopard; 16-bit swaps unchanged.
         */
        if (td->kind == R300_TEXK_RGBA8 && (td->endian & 3) == 2) {
            pkt.uniforms.tex_info[t][1] = 1;
        }
        static bool said;
        if (!said) {
            said = true;
            qemu_log("ppc-mac-gpu r300: texture in AGP/GART memory at %08x "
                     "(%ux%u), read through the GART\n", ga, td->width, td->height);
        }
    }
    r300_note_formats(s, &pkt);
    r300_note_features(s, &pkt);
    {
        /* $R300_GLSLDUMP=prefix: each distinct fragment shader's GLSL */
        static const char *gpre;
        static int ginit, gn;
        static GHashTable *gseen;

        if (!ginit) {
            ginit = 1;
            gpre = getenv("R300_GLSLDUMP");
        }
        if (gpre && pkt.glsl && gn < 80) {
            if (!gseen) {
                gseen = g_hash_table_new(g_str_hash, g_str_equal);
            }
            if (!g_hash_table_contains(gseen, pkt.glsl)) {
                g_hash_table_add(gseen, g_strdup(pkt.glsl));
                char *fn = g_strdup_printf("%s.%02d.frag", gpre, gn++);
                FILE *f = fopen(fn, "w");
                if (f) {
                    fprintf(f, "// draw %llu blend %08x/%08x/%08x alpha-test %08x\n%s",
                            (unsigned long long)s->r3->draws,
                            r300_reg(s->r3, 0x4E04), r300_reg(s->r3, 0x4E08),
                            r300_reg(s->r3, 0x4E0C), r300_reg(s->r3, 0x4BD4),
                            pkt.glsl);
                    fclose(f);
                }
                g_free(fn);
            }
        }
    }
    {
        /* $R300_DXTDUMP=prefix: raw bytes of each distinct DXT texture */
        static const char *pre;
        static int init, nfiles;
        static GHashTable *done;

        if (!init) {
            init = 1;
            pre = getenv("R300_DXTDUMP");
        }
        for (int t = 0; pre && nfiles < 60 && t < R300_NUM_TEX_UNITS; t++) {
            const R300TexDesc *td = &pkt.tex[t];
            if (!td->bound || td->kind < R300_TEXK_DXT1 || !td->size_bytes) {
                continue;
            }
            const uint8_t *src = td->host_data ? td->host_data :
                (uint8_t *)memory_region_get_ram_ptr(&s->vram) + td->gpu_addr;
            if (!done) {
                done = g_hash_table_new(g_direct_hash, g_direct_equal);
            }
            gpointer key = GUINT_TO_POINTER(td->gpu_addr ^ (td->width << 20) ^ td->height);
            if (g_hash_table_contains(done, key)) {
                continue;
            }
            g_hash_table_add(done, key);
            char *fn = g_strdup_printf("%s.%02d_dxt%u_%ux%u_l%u_p%u%s.bin", pre,
                                       nfiles++, td->kind - R300_TEXK_DXT1,
                                       td->width, td->height, td->levels,
                                       td->pitch_bytes, td->host_data ? "_gart" : "");
            FILE *f = fopen(fn, "wb");
            if (f) {
                fwrite(src, 1, td->size_bytes, f);
                fclose(f);
            }
            g_free(fn);
        }
    }
    if (s->r3_zconv && (pkt.depth.attach || (r300_reg(s->r3, 0x4F00) & 7))) {
        r300_zconv(s, false);           /* drawing again: back to linear */
        s->r3_zconv = false;
    }
    if (pkt.depth.attach) {
        uint32_t zo = pkt.depth.gpu_addr;
        if (!r300_to_vram(s, &pkt.depth.gpu_addr,
                          (uint64_t)pkt.depth.pitch * pkt.depth.bpp * pkt.rt_height * ns)) {
            r300_warn_once("depth buffer outside VRAM", NULL);
            pkt.depth.attach = false;
        } else {
            if (s->r3_zb_offset != zo) {
                s->r3_zb_offset = zo;
                s->r3_zb_height = 0;
            }
            s->r3_zb_height = MAX(s->r3_zb_height, pkt.rt_height);
            s->r3_zb_ns = ns;
        }
    }
    pkt.uniforms.zpass_count = s->r3_zpass_active;
    if (r300_reg(s->r3, 0x4E88) & 1) {         /* RB3D_AARESOLVE_CTL */
        r300_aa_resolve(s, &pkt);
        r300_draw_free(&pkt);
        s->regs.stall_draws++;
        return;
    }
    if (ns > 1) {
        static bool said;
        if (!said) {
            said = true;
            qemu_log("ppc-mac-gpu r300: first multisampled draw: %u samples "
                     "(GB_MSPOS %08x %08x) into %06x\n", ns,
                     r300_reg(s->r3, 0x4010), r300_reg(s->r3, 0x4014),
                     pkt.rt_gpu_addr);
        }
        s->r3_cb_aa_off = pkt.rt_gpu_addr;
        s->r3_cb_aa_ns = ns;
    } else if (s->r3_cb_aa_off == pkt.rt_gpu_addr) {
        s->r3_cb_aa_ns = 1;
    }
    if (s->renderer && s->renderer->draw_r300) {
        /* A queue-async renderer enqueues the draw and encodes it on its
         * own thread, so it needs no big lock (the Metal render queue).
         * Everything else below this is QEMU state and still runs under
         * the BQL the caller (or this block) holds. */
        bool need_bql = !s->renderer->draw_queue_async && !bql_locked();

        if (need_bql) {
            bql_lock();
        }
        static int sync_each = -1;
        if (sync_each < 0) {
            sync_each = getenv("R300_SYNC") != NULL;   /* debug: no batching */
        }
        r200_rate.gpu_vs += pkt.vs_glsl != NULL;
        int rr = s->renderer->draw_r300(s->renderer_opaque, vram, s->vram_size, &pkt);
        if (s->r3_dump && g_r300_arm_rt && pkt.rt_gpu_addr == g_r300_arm_rt) {
            fprintf(s->r3_dump, "   renderer -> %d\n", rr);
        }
        if (rr < 0) {
            r300_warn_once("renderer rejected a draw", NULL);
        } else {
            if (sync_each && s->renderer->flush_r200) {
                s->renderer->flush_r200(s->renderer_opaque);
            }
            /*
             * Debug ($R300_DUMP): after a present that samples an
             * off-screen buffer X (a drag frame), save X after each of the
             * next 60 draws into it, with the draw's state, to find the
             * draw that damages it.
             */
            {
                static uint32_t arm_rt, armed_left, step;
                if (s->r3_dump && !armed_left && s->r3->draws > 5000 &&
                    pkt.rt_gpu_addr == 0 && pkt.tex[0].bound &&
                    pkt.tex[0].gpu_addr != 0 && pkt.tex[0].height > 350 &&
                    pkt.tex[0].width > 500 &&
                    pkt.tex[0].pitch_bytes != pkt.rt_pitch * pkt.rt_bpp && step == 0) {
                    arm_rt = pkt.tex[0].gpu_addr;
                    g_r300_arm_rt = arm_rt;
                    armed_left = 60;
                    fprintf(s->r3_dump, "ARMED on %06x\n", arm_rt);
                } else if (armed_left && pkt.rt_gpu_addr == arm_rt) {
                    char path[256];
                    FILE *f;
                    s->renderer->flush_r200(s->renderer_opaque);
                    snprintf(path, sizeof(path), "%s.step%02u_p%u_h%u.bin",
                             getenv("R300_DUMP"), step, pkt.rt_pitch * pkt.rt_bpp,
                             pkt.rt_height);
                    if ((f = fopen(path, "wb"))) {
                        fwrite(vram + pkt.rt_gpu_addr, 1,
                               (size_t)pkt.rt_pitch * pkt.rt_bpp * pkt.rt_height, f);
                        fclose(f);
                    }
                    fprintf(s->r3_dump, "STEP %02u tex %06x %ux%u/%u kind %u "
                            "verts %u cls %u cblend %08x ablend %08x rule %04x "
                            "clip0 %d,%d-%d,%d scissor %u,%u-%u,%u\n", step,
                            pkt.tex[0].gpu_addr, pkt.tex[0].width,
                            pkt.tex[0].height, pkt.tex[0].pitch_bytes,
                            pkt.tex[0].kind, pkt.num_verts, pkt.prim_class,
                            pkt.uniforms.cblend, pkt.uniforms.ablend,
                            pkt.uniforms.clip_rule, pkt.uniforms.cliprect[0][0],
                            pkt.uniforms.cliprect[0][1], pkt.uniforms.cliprect[0][2],
                            pkt.uniforms.cliprect[0][3], pkt.scissor[0],
                            pkt.scissor[1], pkt.scissor[2], pkt.scissor[3]);
                    for (uint32_t v = 0; pkt.verts && v < pkt.num_verts && v < 6; v++) {
                        const R300Vertex *x = &pkt.verts[v];
                        fprintf(s->r3_dump, "   v%u pos %.3f %.3f %.3f %.3f  v0 %.4f %.4f %.4f %.4f"
                                "  v1 %.4f %.4f %.4f %.4f\n", v, x->pos[0], x->pos[1],
                                x->pos[2], x->pos[3], x->v[0][0], x->v[0][1], x->v[0][2],
                                x->v[0][3], x->v[1][0], x->v[1][1], x->v[1][2], x->v[1][3]);
                    }
                    step++;
                    if (--armed_left == 0) {
                        g_r300_arm_rt = 0;
                    }
                }
            }
            /* Debug: snapshot a drag frame (scratch texture + screen). */
            {
                static int reads;
                uint32_t en = r300_reg(s->r3, 0x4104);
                if (s->r3_dump && pkt.rt_gpu_addr == 0 && (en & 1) &&
                    pkt.tex[0].bound && pkt.tex[0].gpu_addr != 0 &&
                    pkt.tex[0].width < 1000 && pkt.tex[0].height > 100 &&
                    s->r3->draws > 3000 && ++reads % 150 == 0 &&
                    reads <= 1500) {
                    char path[256];
                    FILE *f;
                    s->renderer->flush_r200(s->renderer_opaque);
                    snprintf(path, sizeof(path), "%s.%02d.tex_%06x_%ux%u_p%u.bin",
                             getenv("R300_DUMP"), reads / 150, pkt.tex[0].gpu_addr,
                             pkt.tex[0].width, pkt.tex[0].height,
                             pkt.tex[0].pitch_bytes);
                    if ((f = fopen(path, "wb"))) {
                        fwrite(vram + pkt.tex[0].gpu_addr, 1,
                               (size_t)pkt.tex[0].pitch_bytes * pkt.tex[0].height, f);
                        fclose(f);
                    }
                    snprintf(path, sizeof(path), "%s.%02d.screen_p%u_h%u.bin",
                             getenv("R300_DUMP"), reads / 150, pkt.rt_pitch * pkt.rt_bpp,
                             pkt.rt_height);
                    if ((f = fopen(path, "wb"))) {
                        fwrite(vram, 1, (size_t)pkt.rt_pitch * pkt.rt_bpp * pkt.rt_height, f);
                        fclose(f);
                    }
                    fprintf(s->r3_dump, "SNAP taken at draw %llu\n",
                            (unsigned long long)s->r3->draws);
                }
            }
            /* As the R200 path does: the display and the dirty-tracking
             * scanout only refresh what they are told changed.  Not for
             * renderers that read the dirty log as CPU writes (Vulkan):
             * to them this would say the CPU overwrote what they drew. */
            uint64_t len = (uint64_t)pkt.rt_pitch * pkt.rt_bpp * pkt.rt_height;
            if (!s->renderer->set_dirty_source) {
                memory_region_set_dirty(&s->vram, pkt.rt_gpu_addr, len);
                for (uint32_t k = 1; k < pkt.num_cb; k++) {
                    memory_region_set_dirty(&s->vram, pkt.cb[k].gpu_addr,
                                            (uint64_t)pkt.cb[k].pitch * pkt.cb[k].bpp *
                                            pkt.rt_height);
                }
            }
            if (!s->renderer->set_dirty_source && pkt.rt_bpp == 4 &&
                pkt.uniforms.rt_endian == 0) {
                r300_e0_arm(s, pkt.rt_gpu_addr, (uint32_t)len);
            }
            r200_rate.draws++;
            r200_perf.draws++;
            r200_perf.drew = true;
            r200_perf_high(pkt.rt_gpu_addr + len);
            /* The compositor presents by drawing the visible frame buffer
             * at the card's 3D pitch (a multiple of 32 pixels): scan out
             * at that pitch, as the R200 path does for its present BLT. */
            if (pkt.rt_gpu_addr == s->disp.offset &&
                pkt.rt_pitch * pkt.rt_bpp != s->disp.stride) {
                r200_set_present_pitch(s, pkt.rt_pitch * pkt.rt_bpp);
            }
        }
        if (need_bql) {
            bql_unlock();
        }
    }
    r300_draw_free(&pkt);
    s->regs.stall_draws++;
}

/* Returns true when the packet was consumed by the R300 path. */
static bool ppc_mac_gpu_r300_packet3(PPCMacGPUState *s, uint32_t opcode,
                                     const uint32_t *d, uint32_t body_dw)
{
    static uint32_t op_seen[256];
    bool draw = opcode == 0x34 || opcode == 0x35 || opcode == 0x36;

    if (s->r3_dump && op_seen[opcode & 0xFF]++ == 0) {
        fprintf(s->r3_dump, "--- first type-3 opcode 0x%02X (%u dwords):",
                opcode, body_dw);
        for (uint32_t k = 0; k < body_dw && k < 16; k++) {
            fprintf(s->r3_dump, " %08X", d[k]);
        }
        fprintf(s->r3_dump, "\n");
        fflush(s->r3_dump);
    }
    if (s->r3_dump && s->r3->draws && s->r3_trace_lines < 40000) {
        s->r3_trace_lines++;
        fprintf(s->r3_dump, "T3 %02x n=%u %08x %08x\n", opcode,
                body_dw, body_dw ? d[0] : 0, body_dw > 1 ? d[1] : 0);
    }
    if (!draw) {
        /* one call: the log may close itself on reaching its cap */
        FILE *dl = r300_drawlog();
        if (dl) {
            fprintf(dl, "P op%02x n=%u %08x %08x %08x %08x\n", opcode,
                    body_dw, body_dw ? d[0] : 0, body_dw > 1 ? d[1] : 0,
                    body_dw > 2 ? d[2] : 0, body_dw > 3 ? d[3] : 0);
        }
    }
    if (opcode == 0x2F) {
        /* 3D_LOAD_VBPNTR loads the VAP_VTX_NUM_ARRAYS / AOS registers
         * (0x20C0 on); Apple's driver often writes them directly. */
        for (uint32_t k = 0; k < body_dw && k < 1 + 3 * 8; k++) {
            r300_state_write(s->r3, 0x20C0 + 4 * k, d[k]);
        }
        return true;
    }
    if (opcode == 0x32) {                   /* 3D_CLEAR_ZMASK */
        r300_zmask_clear(s);
        return true;
    }
    if (opcode == 0x38) {                   /* 3D_CLEAR_CMASK */
        r300_cmask_clear(s);
        return true;
    }
    if (opcode == 0x37) {                   /* 3D_CLEAR_HIZ: no HiZ kept */
        return true;
    }
    /*
     * Indices from memory: DRAW_INDX_2 carries only VAP_VF_CNTL (PRIM_WALK
     * = indices) and INDX_BUFFER says where they are (Mesa emits the draw
     * first; take either order).
     */
    bool indx_mem = opcode == 0x36 && body_dw == 1 && ((d[0] >> 4) & 3) == 1 &&
                    (d[0] >> 16) != 0;
    if (opcode == 0x33 || (indx_mem && s->r3_indx_buf_pending)) {
        uint32_t vf, ib[3];
        if (opcode == 0x33) {
            if (body_dw < 3) {
                return true;
            }
            memcpy(ib, d, sizeof(ib));
            if (!s->r3_indx_vf) {
                memcpy(s->r3_indx_buf, ib, sizeof(ib));
                s->r3_indx_buf_pending = true;
                return true;
            }
            vf = s->r3_indx_vf;
        } else {
            memcpy(ib, s->r3_indx_buf, sizeof(ib));
            vf = d[0];
        }
        s->r3_indx_vf = 0;
        s->r3_indx_buf_pending = false;

        /* ib: ONE_REG_WR | dst (VAP_PORT_IDX0 >> 2), GPU address, dwords */
        uint32_t ndw = MIN(ib[2] & 0xFFFFFF, 4u * 1024 * 1024);
        uint32_t *idx_dw = g_try_malloc(MAX(ndw, 1u) * 4);
        if (!idx_dw || !r300_read_raw(s, ib[1] & ~3u, idx_dw, ndw * 4)) {
            r300_warn_once("INDX_BUFFER outside VRAM and the GART", NULL);
            g_free(idx_dw);
            return true;
        }
        R300Indices idx = { idx_dw, ndw };
        s->r3->draws++;
        r300_render(s, 0x36, &vf, 1, &idx);
        g_free(idx_dw);
        return true;
    }
    if (indx_mem) {
        s->r3_indx_vf = d[0];               /* wait for INDX_BUFFER */
        return true;
    }
    if (!draw) {
        return false;
    }
    s->r3->draws++;
    if (s->r3_dump && s->r3->draws == 1) {
        hwaddr ph = 0;
        bool g = ppc_mac_gpu_gart_translate(s, s->regs.cp_rb_rptr_addr & ~3u, &ph);
        bool a = !g && ppc_mac_gpu_agp_translate(s, s->regs.cp_rb_rptr_addr & ~3u, &ph);
        fprintf(s->r3_dump, "MC: fb_location %08x agp_location %08x agp_base %08x "
                "config_memsize %08x\nCP: rb_base %08x rb_cntl %08x rptr_addr %08x "
                "(gart %d agp %d -> %08" PRIx64 ") scratch_addr %08x umsk %x\n",
                s->regs.mc_fb_location, s->regs.mc_agp_location, s->regs.agp_base,
                s->regs.config_memsize, s->regs.cp_rb_base, s->regs.cp_rb_cntl,
                s->regs.cp_rb_rptr_addr, g, a, (uint64_t)ph,
                s->regs.scratch_addr, s->regs.scratch_umsk);
    }
    static int screen_dumps, scratch_w, scratch_r;
    static uint32_t scratch_rt;
    uint32_t rt_off = r300_reg(s->r3, 0x4E28) & ~0x1Fu;
    uint32_t rt_pitch = r300_reg(s->r3, 0x4E38) & 0x3FFE;
    bool scratch_write = s->r3->draws > 3000 && rt_off && rt_pitch != 0x340 &&
                         rt_pitch >= 0x100;
    if (scratch_write) {
        scratch_rt = rt_off;
    }
    bool scratch_read = scratch_rt && (r300_reg(s->r3, 0x4104) & 1) &&
                        (r300_reg(s->r3, 0x4540) & ~0x1Fu) == scratch_rt;
    if (s->r3_dump && (scratch_write || scratch_read)) {
        static int seq_lines;
        if (seq_lines++ < 3000) {
            uint32_t f0 = r300_reg(s->r3, 0x4480), f2 = r300_reg(s->r3, 0x4500);
            fprintf(s->r3_dump, "SEQ %s rt %06x/%u tex %06x %ux%u/%u draw %llu\n",
                    scratch_write ? "W" : "R", rt_off, rt_pitch,
                    r300_reg(s->r3, 0x4540) & ~0x1Fu, (f0 & 0x7FF) + 1,
                    ((f0 >> 11) & 0x7FF) + 1, (f2 & 0x3FFF) + 1,
                    (unsigned long long)s->r3->draws);
        }
    }
    if (s->r3_dump && (s->r3->draws <= 4 ||
                       (r300_reg(s->r3, 0x4E28) == 0 && screen_dumps++ < 3) ||
                       (scratch_write && scratch_w++ < 0) ||
                       (scratch_read && scratch_r++ < 0))) {
        fprintf(s->r3_dump, "%s", scratch_write ? "### SCRATCH WRITE\n" :
                                  scratch_read ? "### SCRATCH READ\n" : "");
        fprintf(s->r3_dump, "--- draw op 0x%02X vf_cntl %08X, %u dwords:",
                opcode, body_dw ? d[0] : 0, body_dw);
        for (uint32_t k = 1; k < body_dw && k < 100; k++) {
            fprintf(s->r3_dump, " %08X", d[k]);
        }
        fprintf(s->r3_dump, "\n");
        r300_state_dump(s->r3, s->r3_dump);
    }
    r300_render(s, opcode, d, body_dw, NULL);
    return true;
}

/* Forward declarations */
static void ppc_mac_gpu_execute_ib(PPCMacGPUState *s,
                                    uint32_t ib_base,
                                    uint32_t ib_size_dw);
static void ppc_mac_gpu_2d_blit(PPCMacGPUState *s);
static void ppc_mac_gpu_2d_blit_sep(PPCMacGPUState *s);
static unsigned g_r300_dst_swap;
static void ppc_mac_gpu_mmio_write(void *opaque, hwaddr addr,
                                    uint64_t val, unsigned int size);

/*
 * GART (AGP Intelligent Controller) address translation.
 *
 * The kext configures GART to map GPU addresses in the AGP aperture
 * (AIC_LO_ADDR..AIC_HI_ADDR) to system RAM via a page table at
 * AIC_PT_BASE.  IB command buffers are placed in system RAM and
 * accessed by the GPU through GART.
 *
 * On PPC Macs, the PCI bridge byte-swaps DMA data.  The GART page
 * table entries are written in CPU (big-endian) byte order, but
 * represent little-endian physical addresses for the GPU.  We read
 * the PTE in big-endian and byte-swap to get the physical address.
 *
 * Returns true if translation succeeded, with *phys_addr set.
 */
static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s,
                                        uint32_t gpu_addr,
                                        hwaddr *phys_addr)
{
    if (!(s->regs.aic_ctrl & 1)) {
        return false;  /* GART disabled */
    }
    if (gpu_addr < s->regs.aic_lo_addr || gpu_addr > s->regs.aic_hi_addr) {
        return false;  /* Outside GART aperture */
    }

    uint32_t offset = gpu_addr - s->regs.aic_lo_addr;
    uint32_t page_idx = offset >> 12;       /* 4KB pages */
    uint32_t page_off = offset & 0xFFF;

    /* Read page table entry from guest physical memory.
     * PTE is stored at aic_pt_base + page_idx * 4.
     * On PPC Mac, the AGP GART driver writes PTEs as physical page addresses.
     * We use ldl_be_p semantics: read as big-endian 32-bit from system RAM. */
    /*
     * The R300 driver (ATIRadeon9700) programs the table base at 0x0AB0
     * (low word; 0x0AB4 high) and leaves AIC_PT_BASE (0x01D8) at zero.
     */
    uint32_t pt_base = (s->r300 && s->r300_aic_pt_base) ? s->r300_aic_pt_base
                                                        : s->regs.aic_pt_base;
    hwaddr pte_addr = (hwaddr)pt_base + page_idx * 4;
    uint32_t pte_raw = 0;
    MemTxResult r = address_space_read(
        &address_space_memory, pte_addr,
        MEMTXATTRS_UNSPECIFIED, &pte_raw, 4);
    if (r != MEMTX_OK) {
        return false;
    }

    /* The Mac OS X AGP GART driver writes PTEs as little-endian physical
     * page addresses (GPU-native LE format). The PCI bridge performs byte
     * swapping when the PPC CPU writes to system RAM, so the bytes in RAM
     * are in LE order. address_space_read on a LE host gives us the value
     * directly without needing byte swap. On a BE host we'd need le32_to_cpu. */
    uint32_t pte_val = le32_to_cpu(pte_raw);
    hwaddr phys_page = (hwaddr)(pte_val & ~0xFFFU);

    /* PTE must have a non-zero physical page to be valid */
    if (phys_page == 0) {
        return false;
    }

    static int gart_log_count = 0;
    if (gart_log_count < 20) {
        gpu_debug_log("GART_XLAT gpu=0x%x page=%u pte_addr=0x%"PRIx64
                      " pte_raw=0x%x pte_val=0x%x phys=0x%"PRIx64,
                      gpu_addr, page_idx, (uint64_t)pte_addr,
                      pte_raw, pte_val, phys_page | page_off);
        gart_log_count++;
    }

    *phys_addr = phys_page | page_off;
    return true;
}

/*
 * Translate a GPU address through the AGP bridge (uni-north) GART.
 *
 * The ATI kext places ring buffers and indirect buffers in AGP memory
 * (MC_AGP_LOCATION range, typically 0x10000000+). These addresses are
 * NOT covered by the GPU's internal GART (AIC); instead they go through
 * the uni-north AGP bridge's GART page table.
 *
 * On Mac OS X, the bridge GART and the GPU's AIC GART share the same
 * page table (both point to AIC_PT_BASE). But the aperture offsets differ:
 *   - AIC: covers AIC_LO_ADDR..AIC_HI_ADDR
 *   - AGP: covers MC_AGP_LOCATION range (start..end)
 * GPU addresses in the AGP range are offset from MC_AGP_LOCATION start.
 *
 * GART PTE format: bits[31:12] = physical page address, bit 0 = valid.
 * PTEs are written by PPC CPU in big-endian byte order. However, the
 * uni-north bridge reads PTEs in little-endian (GPU-native) format,
 * matching how the AIC GART PTEs are stored.
 */
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s,
                                       uint32_t gpu_addr,
                                       hwaddr *phys_addr)
{
    /* Check if address falls within MC_AGP_LOCATION range.
     *   bits [15:0] = AGP start >> 16
     *   bits [31:16] = AGP end >> 16 */
    uint32_t agp_loc = s->regs.mc_agp_location;
    if (agp_loc == 0) {
        return false;
    }
    uint32_t agp_start = (agp_loc & 0xFFFF) << 16;
    uint32_t agp_end = (((agp_loc >> 16) & 0xFFFF) << 16) | 0xFFFF;

    if (gpu_addr < agp_start || gpu_addr > agp_end) {
        return false;
    }

    /* Use the uni-north AGP bridge's GART page table base.
     * This is programmed by the kext via PCI config space (offset 0x8C)
     * and is often different from the GPU's AIC_PT_BASE. */
    hwaddr gart_table = uninorth_get_agp_gart_base();
    if (gart_table == 0) {
        /* Fall back to AIC_PT_BASE if bridge GART not yet programmed */
        gart_table = (hwaddr)s->regs.aic_pt_base;
        if (gart_table == 0) {
            return false;
        }
    }

    /* Compute offset from AGP aperture start */
    uint32_t offset = gpu_addr - agp_start;
    uint32_t page_idx = offset >> 12;
    uint32_t page_off = offset & 0xFFF;

    /* Read GART PTE from system RAM */
    hwaddr pte_addr = gart_table + page_idx * 4;
    uint32_t pte_raw = 0;
    MemTxResult r = address_space_read(&address_space_memory, pte_addr,
                                        MEMTXATTRS_UNSPECIFIED, &pte_raw, 4);
    if (r != MEMTX_OK) {
        return false;
    }

    /*
     * PTE endianness: GART PTEs are stored in little-endian format.
     * Both the Apple and Linux AGP drivers use cpu_to_le32() when writing
     * PTEs, because the GPU/bridge hardware reads them in LE natively.
     * PTE format: bits [31:12] = physical page, bit 0 = valid.
     */
    uint32_t pte_val = le32_to_cpu(pte_raw);
    hwaddr phys_page = (hwaddr)(pte_val & 0xFFFFF000U);

    if (phys_page == 0) {
        return false;
    }

    static int agp_log_count = 0;
    if (agp_log_count < 50) {
        gpu_debug_log("AGP_XLAT gpu=0x%x offset=0x%x page=%u "
                      "pte_addr=0x%"PRIx64" pte_raw=0x%x pte_val=0x%x "
                      "gart_base=0x%"PRIx64" phys=0x%"PRIx64,
                      gpu_addr, offset, page_idx, (uint64_t)pte_addr,
                      pte_raw, pte_val, (uint64_t)gart_table,
                      phys_page | page_off);
        agp_log_count++;
    }

    *phys_addr = phys_page | page_off;
    return true;
}

/*
 * Read an Indirect Buffer from system RAM via GART translation.
 * Tries GPU internal GART (AIC) first, then AGP bridge GART.
 * Returns a g_malloc'd buffer of ib_size_dw DWORDs (in CPU byte order),
 * or NULL if translation fails.  Caller must g_free.
 */
static uint32_t *ppc_mac_gpu_read_ib_via_gart(PPCMacGPUState *s,
                                                uint32_t ib_base,
                                                uint32_t ib_size_dw)
{
    uint32_t *buf = g_malloc(ib_size_dw * 4);
    AddressSpace *as = pci_get_address_space(&s->pci);

    /* One translation and one read per 4 KB page, not per dword. */
    for (uint32_t i = 0; i < ib_size_dw;) {
        uint32_t gpu_addr = ib_base + i * 4;
        hwaddr phys;

        if (!ppc_mac_gpu_gart_translate(s, gpu_addr, &phys) &&
            !ppc_mac_gpu_agp_translate(s, gpu_addr, &phys)) {
            g_free(buf);
            return NULL;
        }

        uint32_t n = MAX((0x1000 - (phys & 0xFFF)) / 4, 1u);
        n = MIN(n, ib_size_dw - i);
        MemTxResult r = address_space_read(as, phys, MEMTXATTRS_UNSPECIFIED,
                                           &buf[i], n * 4);
        if (r != MEMTX_OK) {
            g_free(buf);
            return NULL;
        }

        /* Data in guest memory is big-endian (PPC native).
         * PM4 commands are written in CPU byte order.
         * Convert to host byte order for parsing. */
        for (uint32_t k = i; k < i + n; k++) {
            buf[k] = be32_to_cpu(buf[k]);
        }
        i += n;
    }
    return buf;
}

/*
 * How the guest actually talks to us.  A paravirtual device would replace
 * trapping register accesses with commands in shared memory, so measuring
 * the split says what such a rewrite could win.  PPCGPU_TRAFFIC=1 prints it
 * once a second.
 */
static struct {
    uint64_t mmio_writes, mmio_reads, ring_dwords, type0_regs, draws;
    int64_t t0;
} r200_traffic;

static bool r200_in_pm4;          /* replaying the ring, not a guest trap */

static bool r200_traffic_on(void)
{
    static int on = -1;

    if (on < 0) {
        on = getenv("PPCGPU_TRAFFIC") != NULL;
    }
    return on;
}

static void r200_traffic_tick(void)
{
    int64_t now;

    if (!r200_traffic_on()) {
        return;
    }
    now = g_get_monotonic_time();
    if (!r200_traffic.t0) {
        r200_traffic.t0 = now;
        return;
    }
    if (now - r200_traffic.t0 >= 1000000) {
        double secs = (now - r200_traffic.t0) / 1e6;
        fprintf(stderr, "[TRAFFIC] %.0f mmio-writes/s %.0f mmio-reads/s "
                "%.0f ring-dwords/s %.0f type0-regs/s %.0f draws/s\n",
                r200_traffic.mmio_writes / secs, r200_traffic.mmio_reads / secs,
                r200_traffic.ring_dwords / secs, r200_traffic.type0_regs / secs,
                r200_traffic.draws / secs);
        memset(&r200_traffic, 0, sizeof(r200_traffic));
        r200_traffic.t0 = now;
    }
}

static void ppc_mac_gpu_pm4_process_type0(PPCMacGPUState *s,
                                           uint32_t reg_base,
                                           bool one_reg_wr,
                                           uint32_t count,
                                           uint32_t *data)
{
    /*
     * PM4 Type 0 packets write to GPU registers — the same registers
     * accessible via MMIO.  Rather than duplicating handling for each
     * register, forward every write to the MMIO write handler so that
     * ALL registers (CRTC, 2D engine, DP, 3D, etc.) are handled
     * uniformly.  Values in PM4 data[] are already in native byte order
     * (be32_to_cpu from ring buffer), which matches what the MMIO
     * handler expects (DEVICE_LITTLE_ENDIAN auto-swaps for PPC guests).
     *
     * When one_reg_wr is set (bit 15 in PM4 Type 0 header), all data
     * words write to the SAME register address.  This is used for
     * FIFO-style indirect registers like SE_TCL_VECTOR_DATA_REG (0x2204)
     * where each write pushes data into the TCL engine's internal
     * matrix/constant storage.
     */
    uint32_t i;
    for (i = 0; i < count; i++) {
        uint32_t reg_addr = one_reg_wr ? (reg_base * 4)
                                       : ((reg_base + i) * 4);
        uint32_t val = data[i];

        gpu_debug_log("PM4_EXEC type0 reg=0x%04x val=0x%x%s",
                      reg_addr, val, one_reg_wr ? " [ONE_REG]" : "");

        if (unlikely(r200_traffic_on())) {
            r200_in_pm4 = true;
            ppc_mac_gpu_mmio_write(s, reg_addr, val, 4);
            r200_in_pm4 = false;
        } else {
            ppc_mac_gpu_mmio_write(s, reg_addr, val, 4);
        }
    }
}

/*
 * Execute a 2D blit operation triggered by a DST_WIDTH_HEIGHT write.
 *
 * SRC_PITCH_OFFSET / DST_PITCH_OFFSET format (R200):
 *   bits [21:0]  = offset / 1024 (byte offset in VRAM, 1024-byte granularity)
 *   bits [31:22] = pitch / 64 (stride in bytes, 64-byte granularity)
 *
 * DST_Y_X:      bits [15:0] = X, bits [31:16] = Y
 * SRC_Y_X:      bits [15:0] = X, bits [31:16] = Y
 * DST_WIDTH_HEIGHT: bits [15:0] = height, bits [31:16] = width (or vice versa)
 *
 * DP_GUI_MASTER_CNTL bits [27:24] = GMC_ROP3 (0xCC = SRC copy)
 */

/* Forward declarations — defined after r200_macro_tile_addr */
static inline uint64_t r200_macro_tile_addr(uint32_t base_offset,
                                             uint32_t pitch,
                                             uint32_t x, uint32_t y,
                                             uint32_t bpp);
static inline void mc_vram_write32(PPCMacGPUState *s, uint8_t *vram,
                                    uint64_t linear_addr, uint32_t value);
static inline uint32_t mc_vram_read32(PPCMacGPUState *s, uint8_t *vram,
                                       uint64_t linear_addr);

/*
 * HOST_DATA write handler: stores CPU-written pixel data to VRAM.
 *
 * When the kext sets up a HOST data blit (DP_GUI_MASTER_CNTL src_type=3),
 * ppc_mac_gpu_2d_blit() sets host_data_active=true with the destination rect.
 * Then the kext writes pixel data to HOST_DATA0..HOST_DATA_LAST registers.
 * Each 32-bit write = one pixel (at 32bpp). We store it to the destination
 * position in VRAM and advance the cursor left-to-right, top-to-bottom.
 */
/*
 * Bytes per pixel of a 2D-engine datatype (GMC bits 11:8, DP_DATATYPE bits
 * 3:0).  Every 2D path used to assume 32 bpp, so 16-bit surfaces - Warcraft
 * III's Z16 depth buffer and its 1555/4444/565 textures - were cleared and
 * uploaded two pixels at a time (a depth clear to 0xFFFF left alternate
 * columns at 0: vertical stripes wherever the depth test ran).
 */
static uint32_t r200_datatype_bpp(uint32_t dt)
{
    switch (dt & 0xF) {
    case 2: case 7: case 8: case 9:           /* CI8, RGB332, Y8, RGB8 */
        return 1;
    case 3: case 4: case 0xB: case 0xC: case 0xF:  /* 1555, 565, YUV, 4444 */
        return 2;
    default:                                   /* 32bpp, aYUV; 24bpp unsupported */
        return 4;
    }
}

static inline uint32_t gmc_dst_bpp(uint32_t gmc)
{
    /* POWEREMU_TEX_TRACE: each 2D pixel format the guest asks for, once,
     * to find where a program's video frames really go (Halo's logos come
     * out green and magenta, and they are not 3D textures). */
    if (getenv("POWEREMU_TEX_TRACE")) {
        static uint32_t seen_2d;
        uint32_t dt = (gmc >> 8) & 0xF;
        if (!(seen_2d & (1u << dt))) {
            seen_2d |= 1u << dt;
            fprintf(stderr, "ppc-mac-gpu 2d: datatype %u (gmc=0x%08x)\n", dt, gmc);
        }
    }
    return r200_datatype_bpp(gmc >> 8);
}

/*
 * Host data at fewer than 32 bpp: each dword carries 4/bpp pixels in memory
 * byte order (the same bytes the 32bpp path stores whole), and every row
 * starts on a fresh dword.  Returns false once the rectangle is complete.
 */
static bool host_data_put_narrow(PPCMacGPUState *s, uint8_t *vram, uint32_t val,
                                 uint32_t bpp, uint64_t base, uint32_t pitch,
                                 uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                                 uint32_t *cx, uint32_t *cy)
{
    /*
     * 8-bit on the R300: the card takes the dword byte-reversed from how
     * it arrives here (the 32bpp path stores it host-endian, which is
     * VRAM's aperture-0 view), so its first pixel is the top byte, and
     * the card's byte c sits at c ^ 3 in VRAM.
     */
    bool lanes = s->r300 && bpp == 1;
    for (int i = 0; i < 4; i += bpp) {
        if (*cy >= h) {
            return false;
        }
        uint64_t a = base + (uint64_t)(y0 + *cy) * pitch +
                     (uint64_t)(x0 + *cx) * bpp;
        if (lanes) {
            if ((a | 3) < s->vram_size) {
                vram[a ^ 3] = val >> (8 * (3 - i));
            }
        } else if (a + bpp <= s->vram_size) {
            for (uint32_t b = 0; b < bpp; b++) {
                vram[a + b] = val >> (8 * (i + b));
            }
        }
        if (++*cx >= w) {
            *cx = 0;
            ++*cy;
            /* rows are dword-padded, except 8-bit rows on the R300,
             * which run on in the same dword (QuartzGL's glyphs: an
             * atlas dump re-read packed showed every glyph intact) */
            if (!lanes) {
                break;
            }
        }
    }
    return *cy < h;
}

static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                       hwaddr *phys_out);
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s, uint32_t gpu_addr,
                                      hwaddr *phys_out);
static unsigned g_r300_dst_swap;

/*
 * 32bpp host data into system memory (GART or AGP): ATI's OpenGL driver
 * uploads vertex buffers in AGP memory this way (Doom 3's menu).  The card
 * writes each dword little-endian after the destination swap (0x15D0), so
 * with a swap of 2 the CPU's big-endian floats land as it wrote them; the
 * VRAM path's byte order put them byte-reversed, and every vertex decoded
 * as a denormal near zero.
 */
static void ppc_mac_gpu_host_data_write_sys(PPCMacGPUState *s, uint32_t val)
{
    uint32_t bpp = s->host_data_bpp ? s->host_data_bpp : 4;
    uint32_t gpu = s->host_data_offset +
                   (s->host_data_dst_y + s->host_data_cur_y) * s->host_data_pitch +
                   (s->host_data_dst_x + s->host_data_cur_x) * bpp;
    uint8_t b[4];
    hwaddr phys;
    static bool logged;

    if (!logged) {
        logged = true;
        qemu_log("ppc-mac-gpu: host data into system memory at %08x/%u %ux%u bpp %u "
                 "dst swap %u\n", s->host_data_offset, s->host_data_pitch,
                 s->host_data_w, s->host_data_h, bpp, g_r300_dst_swap);
    }
    stl_le_p(b, r300_swap_mode(val, g_r300_dst_swap));
    if (bpp == 4 &&
        (ppc_mac_gpu_agp_translate(s, gpu, &phys) ||
         ppc_mac_gpu_gart_translate(s, gpu, &phys))) {
        address_space_write(pci_get_address_space(&s->pci), phys,
                            MEMTXATTRS_UNSPECIFIED, b, 4);
    }
    if (++s->host_data_cur_x >= s->host_data_w) {
        s->host_data_cur_x = 0;
        if (++s->host_data_cur_y >= s->host_data_h) {
            s->host_data_active = false;
        }
    }
}

static void ppc_mac_gpu_host_data_write(PPCMacGPUState *s, uint32_t val)
{
    if (!s->host_data_active) {
        return;
    }

    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t bpp = s->host_data_bpp ? s->host_data_bpp : 4;

    if (bpp < 4) {
        if (!host_data_put_narrow(s, vram, val, bpp, s->host_data_offset,
                                  s->host_data_pitch, s->host_data_dst_x,
                                  s->host_data_dst_y, s->host_data_w,
                                  s->host_data_h, &s->host_data_cur_x,
                                  &s->host_data_cur_y)) {
            uint64_t start = (uint64_t)s->host_data_offset +
                             (uint64_t)s->host_data_dst_y * s->host_data_pitch;
            uint64_t len = (uint64_t)s->host_data_h * s->host_data_pitch;
            if (start + len <= s->vram_size) {
                ppc_vram_dirty_e0(s, start, len);
            }
            s->host_data_active = false;
            s->display_invalid = true;
        }
        return;
    }

    if (s->host_data_offset >= s->vram_size) {
        ppc_mac_gpu_host_data_write_sys(s, val);
        return;
    }

    /* Compute VRAM byte offset for current pixel — MC handles tiling */
    uint32_t x = s->host_data_dst_x + s->host_data_cur_x;
    uint32_t y = s->host_data_dst_y + s->host_data_cur_y;
    uint64_t linear_addr = (uint64_t)s->host_data_offset +
                           (uint64_t)y * s->host_data_pitch +
                           (uint64_t)x * bpp;

    /* FB write watch: log first pixel of each HOST_DATA sequence to FB */
    if (s->host_data_offset <= 0x10000 && s->host_data_cur_x == 0 &&
        s->host_data_cur_y == 0) {
        fb_write_watch("host_data", "cpu",
                       0, s->host_data_offset,
                       s->host_data_dst_x, s->host_data_dst_y,
                       s->host_data_w, s->host_data_h,
                       false, val != 0);
    }

    /* Phase A — VRAM write watch: window texture tile range */
    if (linear_addr >= 0x353000 && linear_addr < 0x413000) {
        static int hd_watch_log = 0;
        if (hd_watch_log < 50) {
            fprintf(stderr, "[VRAM_WATCH] host_data_write linear=0x%06llx "
                    "val=0x%08x x=%u y=%u off=0x%x pitch=%u\n",
                    (unsigned long long)linear_addr, val,
                    x, y, s->host_data_offset, s->host_data_pitch);
            hd_watch_log++;
        }
    }

    mc_vram_write32(s, vram, linear_addr, val);

    /* Advance cursor: left-to-right, then next row */
    s->host_data_cur_x++;
    if (s->host_data_cur_x >= s->host_data_w) {
        s->host_data_cur_x = 0;
        s->host_data_cur_y++;
        if (s->host_data_cur_y >= s->host_data_h) {
            /* Transfer complete — mark dirty and deactivate */
            uint64_t dirty_start = (uint64_t)s->host_data_offset +
                                   (uint64_t)s->host_data_dst_y *
                                       s->host_data_pitch +
                                   (uint64_t)s->host_data_dst_x * bpp;
            uint64_t dirty_len = (s->host_data_h > 1)
                ? (uint64_t)(s->host_data_h - 1) * s->host_data_pitch +
                  (uint64_t)s->host_data_w * bpp
                : (uint64_t)s->host_data_w * bpp;
            if (dirty_start + dirty_len <= s->vram_size) {
                ppc_vram_dirty_e0(s, dirty_start, dirty_len);
            }
            if (g_frame_tracker) {
                frame_tracker_record_detail(g_frame_tracker,
                                            PASS_EVENT_HOST_UPLOAD,
                                            s->host_data_offset,
                                            s->host_data_pitch / 4,
                                            0,
                                            s->host_data_offset,
                                            (uint16_t)s->host_data_w,
                                            (uint16_t)s->host_data_h,
                                            0, 0,
                                            (uint16_t)s->host_data_dst_x,
                                            (uint16_t)s->host_data_dst_y,
                                            0, 0, false);
            }
            s->host_data_active = false;
            s->display_invalid = true;

            static int hd_log_count = 0;
            if (hd_log_count < 50) {
                gpu_debug_log("HOST_DATA complete dst=0x%x+%u xy=%u,%u %ux%u",
                              s->host_data_offset, s->host_data_pitch,
                              s->host_data_dst_x, s->host_data_dst_y,
                              s->host_data_w, s->host_data_h);
                hd_log_count++;
            }
        }
    }
}

/*
 * R200 macro-tile address translation.
 *
 * Macro tiles are 256 bytes wide × 16 rows tall (for 32bpp: 64 pixels × 16).
 * Each macro tile is 4096 bytes.  Within a macro tile, pixels are stored in
 * row-major order (row 0..15 of the 256-byte-wide strip).
 *
 * Linear address:  y * pitch + x * bpp
 * Tiled address:   tile_index * 4096 + (y%16) * 256 + (x%64) * bpp
 *   where tile_index = (y/16) * (pitch/256) + (x/64)
 */
static inline uint64_t r200_macro_tile_addr(uint32_t base_offset,
                                             uint32_t pitch,
                                             uint32_t x, uint32_t y,
                                             uint32_t bpp)
{
    uint32_t tile_w_pixels = 256 / bpp;       /* 64 for 32bpp */
    uint32_t tile_h = 16;
    uint32_t tiles_per_row = pitch / 256;
    uint32_t tile_x = x / tile_w_pixels;
    uint32_t tile_y = y / tile_h;
    uint32_t in_x   = x % tile_w_pixels;
    uint32_t in_y   = y % tile_h;
    uint64_t tile_idx = (uint64_t)tile_y * tiles_per_row + tile_x;
    return (uint64_t)base_offset +
           tile_idx * (256 * tile_h) +
           (uint64_t)in_y * 256 +
           (uint64_t)in_x * bpp;
}

/* ========================================================================
 * MC-style tiled surface registry
 *
 * Tiling is a memory-controller property: ALL accesses to a tiled surface
 * go through the tiling swizzle, regardless of which engine initiated them.
 *
 * The registry is populated from PITCH_OFFSET register writes where
 * macro_tile=1.  The MC helpers below transparently remap linear addresses
 * into tiled physical addresses when they fall within a registered surface.
 * ======================================================================== */

/* Enable/disable the MC tiling layer.  Set to 0 to bypass (all linear). */
#define MC_TILED_SURFACES_ENABLED 0  /* Disabled — see tiling audit notes */

/*
 * Register a tiled surface.  Deduplicates by base address — if a surface
 * with the same base already exists, update its pitch (in case the driver
 * reconfigures it).  Returns the slot index, or -1 if full.
 */
static int mc_register_tiled_surface(PPCMacGPUState *s,
                                      uint32_t base, uint32_t pitch,
                                      const char *label)
{
    /*
     * Debug toggle: PPCGPU_NO_TILING=1 disables the MC tiling swizzle
     * entirely, so every VRAM access is treated as linear.  Used to test
     * whether the PITCH_OFFSET bit-30 "macro tile" interpretation is
     * actually correct — the guest addresses one base (0x300000) with a
     * dozen different pitches, which a genuinely tiled surface could not
     * survive on real hardware either.
     */
    static int no_tiling = -1;
    if (no_tiling < 0) {
        const char *e = getenv("PPCGPU_NO_TILING");
        no_tiling = (e && *e == '1') ? 1 : 0;
        if (no_tiling) {
            fprintf(stderr, "[MC_TILING] disabled by PPCGPU_NO_TILING=1 — "
                            "all VRAM treated as linear\n");
        }
    }
    /*
     * The R300 path renders and samples every surface linearly (tiling is
     * a memory-layout detail no host consumer needs), so 2D blits must
     * store linearly too, or a window uploaded by blit and composited by
     * the 3D engine comes out scrambled.
     */
    if (no_tiling || s->r300) {
        return -1;
    }

    /* Skip framebuffer (offset 0) — CRTC scanout is always linear */
    if (base == 0) {
        return -1;
    }
    if (pitch == 0) {
        return -1;
    }

    /* Check for existing entry with same base */
    for (int i = 0; i < MC_TILED_SURFACE_MAX; i++) {
        if (s->tiled_surfaces[i].active && s->tiled_surfaces[i].base == base) {
            s->tiled_surfaces[i].pitch = pitch;
            /* Estimate surface size: pitch * 1024 rows as a generous upper bound */
            s->tiled_surfaces[i].size = pitch * 1024;
            if (s->tiled_surfaces[i].size > s->vram_size - base) {
                s->tiled_surfaces[i].size = (uint32_t)(s->vram_size - base);
            }
            return i;
        }
    }

    /* Find a free slot */
    for (int i = 0; i < MC_TILED_SURFACE_MAX; i++) {
        if (!s->tiled_surfaces[i].active) {
            s->tiled_surfaces[i].base = base;
            s->tiled_surfaces[i].pitch = pitch;
            s->tiled_surfaces[i].size = pitch * 1024;
            if (s->tiled_surfaces[i].size > s->vram_size - base) {
                s->tiled_surfaces[i].size = (uint32_t)(s->vram_size - base);
            }
            s->tiled_surfaces[i].active = true;
            snprintf(s->tiled_surfaces[i].label,
                     sizeof(s->tiled_surfaces[i].label), "%s", label);

            static int reg_log_count = 0;
            if (reg_log_count < 50) {
                blit_path_log("SURFACE_REG",
                    "slot=%d base=0x%06x pitch=%u size=0x%x label=%s",
                    i, base, pitch, s->tiled_surfaces[i].size, label);
                reg_log_count++;
            }
            return i;
        }
    }

    return -1;  /* full */
}

/*
 * Unregister a tiled surface by base address.
 */
static void mc_unregister_tiled_surface(PPCMacGPUState *s, uint32_t base)
{
    for (int i = 0; i < MC_TILED_SURFACE_MAX; i++) {
        if (s->tiled_surfaces[i].active && s->tiled_surfaces[i].base == base) {
            s->tiled_surfaces[i].active = false;
            static int unreg_log_count = 0;
            if (unreg_log_count < 20) {
                blit_path_log("SURFACE_UNREG",
                    "slot=%d base=0x%06x was=%s",
                    i, base, s->tiled_surfaces[i].label);
                unreg_log_count++;
            }
        }
    }
}

/*
 * Look up whether a linear VRAM address falls within a registered tiled surface.
 * If so, returns the surface slot index and sets *surf_base and *surf_pitch.
 * Returns -1 if the address is not in any tiled surface (→ linear access).
 */
static int mc_find_tiled_surface(PPCMacGPUState *s, uint64_t linear_addr,
                                  uint32_t *surf_base, uint32_t *surf_pitch)
{
    if (!MC_TILED_SURFACES_ENABLED) {
        return -1;
    }
    /*
     * Find the tiled surface whose base is closest to (but not exceeding)
     * linear_addr.  This avoids matching a wrong surface when generous
     * size estimates cause ranges to overlap.
     *
     * Example: surface A at base=0x1ee000, size=0x340000 would otherwise
     * swallow addresses meant for surface B at base=0x300000.  By choosing
     * the closest base, B wins for addresses >= 0x300000.
     */
    int best = -1;
    uint64_t best_off = UINT64_MAX;
    for (int i = 0; i < MC_TILED_SURFACE_MAX; i++) {
        if (!s->tiled_surfaces[i].active) continue;
        uint32_t base = s->tiled_surfaces[i].base;
        uint32_t end  = base + s->tiled_surfaces[i].size;
        if (linear_addr >= base && linear_addr < end) {
            uint64_t off = linear_addr - base;
            if (off < best_off) {
                best_off = off;
                best = i;
            }
        }
    }
    if (best >= 0) {
        *surf_base = s->tiled_surfaces[best].base;
        *surf_pitch = s->tiled_surfaces[best].pitch;
    }
    return best;
}

/*
 * Do the source and destination rectangles of a blit share pixels?
 *
 * Only meaningful for copies within one surface: if the two rectangles sit
 * in different surfaces (different base offset or pitch) they cannot alias,
 * and a plain forward copy is always safe.  Callers use this to decide
 * which direction to walk the rectangle, the way memmove does.
 */
static bool blit_overlaps(uint32_t src_offset, uint32_t src_pitch,
                          uint32_t src_x, uint32_t src_y,
                          uint32_t dst_offset, uint32_t dst_pitch,
                          uint32_t dst_x, uint32_t dst_y,
                          uint32_t w, uint32_t h, uint32_t bpp)
{
    if (src_offset != dst_offset || src_pitch != dst_pitch) {
        return false;
    }
    if (src_x == dst_x && src_y == dst_y) {
        return false;   /* copy onto itself — nothing can be corrupted */
    }
    return !(src_x + w <= dst_x || dst_x + w <= src_x ||
             src_y + h <= dst_y || dst_y + h <= src_y);
}

/* Is the surface holding this rectangle macro-tiled? */
static bool mc_rect_is_tiled(PPCMacGPUState *s, uint32_t offset,
                             uint32_t pitch, uint32_t x, uint32_t y,
                             uint32_t bpp)
{
    uint32_t surf_base, surf_pitch;
    uint64_t addr = (uint64_t)offset + (uint64_t)y * pitch +
                    (uint64_t)x * bpp;
    return mc_find_tiled_surface(s, addr, &surf_base, &surf_pitch) >= 0;
}

/*
 * Fast path for rectangle copies between untiled VRAM surfaces.
 *
 * The per-pixel path calls mc_vram_read32/mc_vram_write32 for every pixel,
 * and each of those re-runs mc_find_tiled_surface — a linear scan of the
 * surface table plus a division and a modulo.  For a fullscreen blit that
 * is well over a million redundant lookups.
 *
 * Tiling cannot change part-way through a blit, so resolve it once here.
 * When neither surface is tiled the copy is a plain row-by-row move, which
 * is what the overwhelming majority of blits are: mc_register_tiled_surface
 * never registers the framebuffer (offset 0), so anything touching the
 * visible screen lands on this path.
 *
 * memmove handles overlap within a row; overlap between rows is handled by
 * choosing the row direction, exactly as the per-pixel path does.
 *
 * Returns false if either surface is tiled, and the caller must then fall
 * back to the per-pixel loop.
 */
static bool blit_rect_untiled(PPCMacGPUState *s, uint8_t *vram,
                              uint32_t src_offset, uint32_t src_pitch,
                              uint32_t src_x, uint32_t src_y,
                              uint32_t dst_offset, uint32_t dst_pitch,
                              uint32_t dst_x, uint32_t dst_y,
                              uint32_t w, uint32_t h, uint32_t bpp)
{
    if (mc_rect_is_tiled(s, src_offset, src_pitch, src_x, src_y, bpp) ||
        mc_rect_is_tiled(s, dst_offset, dst_pitch, dst_x, dst_y, bpp)) {
        return false;
    }

    bool rev_rows = blit_overlaps(src_offset, src_pitch, src_x, src_y,
                                  dst_offset, dst_pitch, dst_x, dst_y,
                                  w, h, bpp) && dst_y > src_y;
    size_t row_bytes = (size_t)w * bpp;
    /*
     * 8-bit pixels on the R300: VRAM holds aperture 0's 32-bit-swapped
     * view, so the card's byte c sits at c ^ 3.  Copy by the card's
     * positions; a plain move is only right where source and
     * destination share their place in a dword, and only for whole
     * dwords.  (QuartzGL copies glyphs into an X8 atlas this way:
     * glyphs that landed at another dword offset came out scrambled.)
     */
    uint8_t *tmp = s->r300 && bpp == 1 ? g_malloc(row_bytes + 6) : NULL;

    for (uint32_t r = 0; r < h; r++) {
        uint32_t row = rev_rows ? (h - 1 - r) : r;
        uint64_t s_row = (uint64_t)src_offset +
                         (uint64_t)(src_y + row) * src_pitch +
                         (uint64_t)src_x * bpp;
        uint64_t d_row = (uint64_t)dst_offset +
                         (uint64_t)(dst_y + row) * dst_pitch +
                         (uint64_t)dst_x * bpp;

        /* Drop rows that fall outside VRAM, matching the per-pixel path,
         * which silently discards out-of-range pixels. */
        /* the ^ 3 reach is the rest of the last dword */
        uint64_t s_end = s_row + row_bytes, d_end = d_row + row_bytes;
        if (tmp) {
            s_end = (s_end + 3) & ~3ull;
            d_end = (d_end + 3) & ~3ull;
        }
        if (s_end > s->vram_size || d_end > s->vram_size) {
            continue;
        }
        if (tmp) {
            for (size_t i = 0; i < row_bytes; i++) {
                tmp[i] = vram[(s_row + i) ^ 3];
            }
            for (size_t i = 0; i < row_bytes; i++) {
                vram[(d_row + i) ^ 3] = tmp[i];
            }
            continue;
        }
        memmove(vram + d_row, vram + s_row, row_bytes);
    }
    g_free(tmp);
    return true;
}

/*
 * MC VRAM write: write a 32-bit pixel to VRAM at a linear address.
 * If the address falls within a tiled surface, the tiling swizzle is
 * applied transparently.
 */
/*
 * Copy a rectangle from GART/AGP system RAM into untiled VRAM.
 *
 * Video frames and textures arrive this way every frame.  Translating and
 * reading each pixel separately (a page-table read plus a memory lookup
 * per pixel) took about 8% of the emulated CPU's time during video
 * playback; this translates once per 4 KB page and moves whole runs.
 * The byte swaps match the per-pixel path in ppc_mac_gpu_2d_blit_sep.
 */
static void gart_blit_rows(PPCMacGPUState *s, AddressSpace *as, bool gart,
                           uint32_t src_offset, uint32_t src_pitch,
                           uint32_t src_x, uint32_t src_y, uint8_t *vram,
                           uint32_t dst_offset, uint32_t dst_pitch,
                           uint32_t dst_x, uint32_t dst_y,
                           uint32_t w, uint32_t h, uint32_t bpp)
{
    size_t row_bytes = (size_t)w * bpp;
    /*
     * 8-bit texels: VRAM holds aperture 0's 32-bit-swapped view of the
     * card's bytes (see r300_ap_swap), and the DMA engine stores the
     * source bytes through SRC swap.  So each byte lands in another lane
     * of its dword: reversed for no swap (QuartzGL's glyph atlases),
     * where it lies for a 32-bit swap.  Rows go through a scratch copy so
     * the bytes of a partly covered dword outside the rect keep theirs.
     */
    unsigned lane_xor = 0;
    bool narrow = s->r300 && bpp == 1 && s->r300_src_swap != 2;
    uint8_t *tmp = NULL;
    if (narrow) {
        static const unsigned x[4] = { 3, 2, 0, 1 };   /* by SRC swap mode */
        lane_xor = x[s->r300_src_swap & 3];
        tmp = g_malloc(row_bytes);
    }

    for (uint32_t row = 0; row < h; row++) {
        uint32_t gpu = src_offset + (src_y + row) * src_pitch + src_x * bpp;
        uint64_t d = (uint64_t)dst_offset + (uint64_t)(dst_y + row) * dst_pitch +
                     (uint64_t)dst_x * bpp;
        if (d + row_bytes > s->vram_size || (narrow && ((d + row_bytes + 3) & ~3ull) > s->vram_size)) {
            continue;
        }
        uint8_t *out = narrow ? tmp : vram + d;

        for (size_t done = 0; done < row_bytes;) {
            hwaddr phys;
            bool ok = gart ? ppc_mac_gpu_gart_translate(s, gpu + done, &phys)
                           : ppc_mac_gpu_agp_translate(s, gpu + done, &phys);
            size_t n = ok ? 0x1000 - (phys & 0xFFF) : 1;
            n = MIN(n, row_bytes - done);
            if (!ok || address_space_read(as, phys, MEMTXATTRS_UNSPECIFIED,
                                          out + done, n) != MEMTX_OK) {
                memset(out + done, 0, n);
            }
            done += n;
        }

        if (narrow) {
            for (size_t i = 0; i < row_bytes; i++) {
                vram[(d + i) ^ lane_xor] = tmp[i];
            }
        } else if (s->r300 && bpp == 4 && s->r300_src_swap != 2) {
            for (size_t i = 0; i + 4 <= row_bytes; i += 4) {
                uint32_t px = ldl_he_p(out + i);
                stl_he_p(out + i, bswap32(r300_swap_mode(px, s->r300_src_swap)));
            }
        } else if (s->r300 && bpp == 2 && s->r300_src_swap == 0) {
            for (size_t i = 0; i + 2 <= row_bytes; i += 2) {
                stw_he_p(out + i, bswap16(lduw_he_p(out + i)));
            }
        }
    }
    g_free(tmp);
}

static inline void mc_vram_write32(PPCMacGPUState *s, uint8_t *vram,
                                    uint64_t linear_addr, uint32_t value)
{
    uint32_t surf_base, surf_pitch;
    int slot = mc_find_tiled_surface(s, linear_addr, &surf_base, &surf_pitch);

    uint64_t phys_addr;
    if (slot >= 0) {
        /* Tiled surface: decompose linear offset → (x, y) → tiled address */
        uint64_t off = linear_addr - surf_base;
        uint32_t y = (uint32_t)(off / surf_pitch);
        uint32_t x = (uint32_t)((off % surf_pitch) / 4);  /* 4 = bpp for 32bpp */
        phys_addr = r200_macro_tile_addr(surf_base, surf_pitch, x, y, 4);
    } else {
        phys_addr = linear_addr;
    }

    if (phys_addr + 4 <= s->vram_size) {
        /* Phase A — VRAM write watch: window texture tile range */
        if (phys_addr >= 0x353000 && phys_addr < 0x413000) {
            static int vram_watch_log = 0;
            static int zero_write_log = 0;
            /* Log first 50 writes, plus up to 20 zero-value writes */
            if (vram_watch_log < 50 || (value == 0 && zero_write_log < 20)) {
                fprintf(stderr, "[VRAM_WATCH] mc_vram_write32 phys=0x%06llx "
                        "linear=0x%06llx val=0x%08x tiled=%d\n",
                        (unsigned long long)phys_addr,
                        (unsigned long long)linear_addr,
                        value, (slot >= 0));
                vram_watch_log++;
                if (value == 0) zero_write_log++;
            }
        }
        *(uint32_t *)(vram + phys_addr) = value;
    }
}

/*
 * Store one solid-fill pixel.  VRAM holds pixels as the CPU sees them
 * (big-endian, see qemu-ppc-gpu-endianness-and-tiling-are-correct), while a
 * fill colour arrives as a register value, so it is stored big-endian at the
 * surface's pixel size.  (Copies move bytes and need no conversion.)
 */
static inline void fill_vram_px(PPCMacGPUState *s, uint8_t *vram,
                                uint64_t addr, uint32_t color, uint32_t bpp)
{
    if (bpp == 4) {
        mc_vram_write32(s, vram, addr, be32_to_cpu(color));
    } else if (addr + bpp <= s->vram_size) {
        if (bpp == 2) {
            stw_be_p(vram + addr, color);
        } else {
            vram[addr] = color;
        }
    }
}

/*
 * MC VRAM read: read a 32-bit pixel from VRAM at a linear address.
 * If the address falls within a tiled surface, the tiling swizzle is
 * applied transparently.
 */
static inline uint32_t mc_vram_read32(PPCMacGPUState *s, uint8_t *vram,
                                       uint64_t linear_addr)
{
    uint32_t surf_base, surf_pitch;
    int slot = mc_find_tiled_surface(s, linear_addr, &surf_base, &surf_pitch);

    uint64_t phys_addr;
    if (slot >= 0) {
        uint64_t off = linear_addr - surf_base;
        uint32_t y = (uint32_t)(off / surf_pitch);
        uint32_t x = (uint32_t)((off % surf_pitch) / 4);
        phys_addr = r200_macro_tile_addr(surf_base, surf_pitch, x, y, 4);

        /*
         * Phase A — Tiling Consistency Audit: log sample reads from
         * tiled surfaces to prove MC read path + tiling formula used.
         * Log for same sample points as Metal writeback, once per base.
         */
        {
            static uint32_t audit_base = 0;
            static int audit_count = 0;
            if (audit_count < 6 && surf_base >= 0x100000) {
                if (audit_base == 0) audit_base = surf_base;
                if (surf_base == audit_base &&
                    ((x == 0 && y == 0) || (x == 63 && y == 0) ||
                     (x == 64 && y == 0) || (x == 0 && y == 15) ||
                     (x == 0 && y == 16) || (x == 100 && y == 20))) {
                    fprintf(stderr,
                        "[TILE_PATH] rt_off=0x%x path=mc_read "
                        "tiling_applied=yes formula=mc_macro "
                        "x=%u y=%u pitch=%u(bytes) "
                        "linear_addr=0x%llx resolved_addr=0x%llx "
                        "note=surf_slot=%d\n",
                        surf_base,
                        x, y, surf_pitch,
                        (unsigned long long)linear_addr,
                        (unsigned long long)phys_addr,
                        slot);
                    audit_count++;
                }
            }
        }
    } else {
        phys_addr = linear_addr;
    }

    if (phys_addr + 4 <= s->vram_size) {
        return *(uint32_t *)(vram + phys_addr);
    }
    return 0;
}

/*
 * 2D features the blit paths do not implement.  Say once when a blit asks
 * for one, instead of drawing it silently without (GMC bits as in the
 * Radeon X driver: 28 CLR_CMP_CNTL_DIS, 29 AUX_CLIP_DIS, 30 WR_MSK_DIS).
 */
static void r200_note_2d_gaps(PPCMacGPUState *s, uint32_t gmc, const char *path)
{
    static bool said_cmp, said_aux, said_msk;

    if (!(gmc & (1u << 28)) && !said_cmp) {
        said_cmp = true;
        qemu_log("ppc-mac-gpu 2d: %s blit with colour compare enabled "
                 "(gmc %08x) - not implemented, drawn without it\n", path, gmc);
    }
    if (!(gmc & (1u << 29)) && !said_aux) {
        said_aux = true;
        qemu_log("ppc-mac-gpu 2d: %s blit with aux clipping enabled "
                 "(gmc %08x) - not implemented, drawn without it\n", path, gmc);
    }
    if (!(gmc & (1u << 30)) && s->regs.dp_write_msk != 0xFFFFFFFFu && !said_msk) {
        said_msk = true;
        qemu_log("ppc-mac-gpu 2d: %s blit with write mask %08x (gmc %08x) "
                 "- not implemented, all bits written\n", path,
                 s->regs.dp_write_msk, gmc);
    }
}

static void ppc_mac_gpu_2d_blit(PPCMacGPUState *s)
{
    /*
     * DP_GUI_MASTER_CNTL bits 0/1 select, per side, between the explicit
     * SRC_/DST_PITCH_OFFSET register (set) and DEFAULT_PITCH_OFFSET
     * (clear).  Apple's window restores set both bits, so for them this
     * matches the old behaviour exactly.
     */
    uint32_t gmc_po = s->regs.dp_gui_master_cntl;
    uint32_t src_po = (gmc_po & R200_GMC_SRC_PITCH_OFFSET_CNTL)
                      ? s->regs.src_pitch_offset
                      : s->regs.default_pitch_offset;
    uint32_t dst_po = (gmc_po & R200_GMC_DST_PITCH_OFFSET_CNTL)
                      ? s->regs.dst_pitch_offset
                      : s->regs.default_pitch_offset;

    /* R200 PITCH_OFFSET format (from QEMU ati.c):
     *   bits [21:0]  = offset >> 10 (byte offset, 1024-byte granularity)
     *   bits [29:22] = pitch >> 6 (stride in bytes, 64-byte granularity)
     *   bit  30      = macro tile
     *   bit  31      = micro tile
     * Decode: offset = (val & 0x3FFFFF) << 10
     *         pitch  = (val & 0x3FC00000) >> 16   (= bits[29:22] << 6) */
    uint32_t src_offset = (src_po & 0x3FFFFF) << 10;
    uint32_t src_pitch  = (src_po & 0x3FC00000) >> 16;
    uint32_t dst_offset = (dst_po & 0x3FFFFF) << 10;
    uint32_t dst_pitch  = (dst_po & 0x3FC00000) >> 16;
    /*
     * Tiling policy — MC-style surface registry.
     *
     * We no longer use per-blit macro_tile flags.  Instead, mc_vram_read32
     * and mc_vram_write32 transparently apply the tiling swizzle for any
     * address that falls within a registered tiled surface.
     *
     * When SRC_PITCH_OFFSET or DST_PITCH_OFFSET is written with macro_tile=1,
     * the surface is registered.  The MC helpers handle the rest.
     *
     * Also register tiled surfaces from the current PO values at blit time,
     * in case they were set via PM4 packets rather than direct MMIO writes.
     */
    if ((src_po >> 30) & 1) {
        mc_register_tiled_surface(s, src_offset, src_pitch, "SRC_PO_blit");
    }
    if ((dst_po >> 30) & 1) {
        mc_register_tiled_surface(s, dst_offset, dst_pitch, "DST_PO_blit");
    }

    uint32_t dwh = s->regs.dst_width_height;
    uint32_t blit_w = (dwh >> 16) & 0x3FFF;
    uint32_t blit_h = dwh & 0x3FFF;

    if (s->r3_dump) {
        static GHashTable *seen_blit;
        char *key = g_strdup_printf("%08x %08x %08x", src_po, dst_po,
                                    s->regs.dp_gui_master_cntl);
        if (!seen_blit) {
            seen_blit = g_hash_table_new(g_str_hash, g_str_equal);
        }
        if (!g_hash_table_contains(seen_blit, key)) {
            g_hash_table_add(seen_blit, key);
            fprintf(s->r3_dump, "BLIT src %06x/%u dst %06x/%u gmc %08x "
                    "src_xy %08x dst_xy %08x wh %08x\n", src_offset, src_pitch,
                    dst_offset, dst_pitch, s->regs.dp_gui_master_cntl,
                    s->regs.src_y_x, s->regs.dst_y_x, dwh);
        } else {
            g_free(key);
        }
    }

    uint32_t src_yx = s->regs.src_y_x;
    uint32_t dst_yx = s->regs.dst_y_x;
    uint32_t src_x = src_yx & 0xFFFF;
    uint32_t src_y = (src_yx >> 16) & 0xFFFF;
    uint32_t dst_x = dst_yx & 0xFFFF;
    uint32_t dst_y = (dst_yx >> 16) & 0xFFFF;

    /* Extract ROP3 and source type from DP_GUI_MASTER_CNTL */
    uint32_t gmc = s->regs.dp_gui_master_cntl;
    uint32_t rop3 = (gmc >> 16) & 0xFF;
    uint32_t src_type = (gmc >> 24) & 0x7; /* bits [26:24] = source type */
    r200_note_2d_gaps(s, gmc, "MMIO");

    if (blit_w == 0 || blit_h == 0) {
        return;
    }

    /* Path instrumentation: log MMIO combined blit with its packed state */
    g_blit_stats.mmio_count++;
    {
        static uint64_t mmio_logged = 0;
        if (mmio_logged < 500) {
            bool raw_src_mt = (src_po >> 30) & 1;
            bool raw_dst_mt = (dst_po >> 30) & 1;
            const char *dst_region = "offscreen";
            if (dst_offset == 0 && (dst_pitch == 4096 || dst_pitch == 3200))
                dst_region = "FRAMEBUFFER";
            else if (dst_offset >= 0x200000)
                dst_region = "backbuf";

            blit_path_log("MMIO",
                "rop=0x%02x src_type=%u  "
                "src_po=0x%08x[off=0x%06x pitch=%u mt=%d]  "
                "dst_po=0x%08x[off=0x%06x pitch=%u mt=%d]  "
                "src_xy=(%u,%u) dst_xy=(%u,%u) %ux%u  dst=%s",
                rop3, src_type,
                src_po, src_offset, src_pitch, raw_src_mt,
                dst_po, dst_offset, dst_pitch, raw_dst_mt,
                src_x, src_y, dst_x, dst_y,
                blit_w, blit_h, dst_region);
            mmio_logged++;
        }
    }

    r200_vram_access(s, R200_ROWS(dst_offset, dst_pitch, dst_y, blit_h), true, 1);
    if (src_offset < s->vram_size) {
        r200_vram_access(s, R200_ROWS(src_offset, src_pitch ? src_pitch : dst_pitch,
                                      src_y, blit_h), false, 1);
    }
    seq_log("REG  rop=%02x src=%06x/%u (%u,%u) dst=%06x/%u (%u,%u) %ux%u",
            rop3, src_offset, src_pitch, src_x, src_y,
            dst_offset, dst_pitch, dst_x, dst_y, blit_w, blit_h);

    /* Record BLIT region for 3D coordinate mapping (MMIO path).
     * Same logic as in BITBLT_MULTI: record when source is off-screen RT. */
    if (rop3 == 0xCC && src_type == 2 && src_offset >= 0x100000) {
        blit_region_record(src_offset, src_pitch,
                           src_x, src_y,
                           dst_x, dst_y,
                           blit_w, blit_h);
    }

    /* Source type 3 = HOST data: set up host data transfer state.
     * The actual pixel data arrives via HOST_DATA register writes. */
    if (src_type == 3) {
        s->host_data_offset = dst_offset;
        s->host_data_pitch = dst_pitch;
        /* MC tiling is now handled transparently by mc_vram_write32 */
        s->host_data_macro_tile = false;
        s->host_data_dst_x = dst_x;
        s->host_data_dst_y = dst_y;
        s->host_data_w = blit_w;
        s->host_data_h = blit_h;
        s->host_data_cur_x = 0;
        s->host_data_cur_y = 0;
        s->host_data_bpp = gmc_dst_bpp(s->regs.dp_gui_master_cntl);
        s->host_data_active = true;

        gpu_debug_log("2D_HOST_BLT dst=0x%x+%u dxy=%u,%u %ux%u",
                      dst_offset, dst_pitch, dst_x, dst_y, blit_w, blit_h);
        return;
    }
    if (src_pitch == 0) {
        src_pitch = dst_pitch;
    }
    if (dst_pitch == 0) {
        return;
    }

    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t bpp = gmc_dst_bpp(s->regs.dp_gui_master_cntl);

    gpu_debug_log("2D_BLIT rop=0x%02x src=0x%x+%u dst=0x%x+%u "
                  "sxy=%u,%u dxy=%u,%u %ux%u",
                  rop3, src_offset, src_pitch, dst_offset, dst_pitch,
                  src_x, src_y, dst_x, dst_y, blit_w, blit_h);

    /* Debug: check if source data has non-zero pixels */
    {
        static int blit_check = 0;
        if (blit_check < 30) {
            uint64_t check_off = (uint64_t)src_offset +
                                 (uint64_t)src_y * src_pitch +
                                 (uint64_t)src_x * bpp;
            uint32_t nonzero = 0;
            uint32_t first_nz = 0;
            for (uint32_t p = 0; p < blit_w && p < 32; p++) {
                uint64_t po = check_off + (uint64_t)p * bpp;
                if (po + 4 <= s->vram_size) {
                    uint32_t pix = *(uint32_t *)(vram + po);
                    if (pix != 0) {
                        if (nonzero == 0) first_nz = pix;
                        nonzero++;
                    }
                }
            }
            gpu_debug_log("  BLIT_SRC_CHECK: %u/32 nonzero (first=0x%08x) "
                          "vram_ptr=%p off=0x%lx",
                          nonzero, first_nz, (void *)vram,
                          (unsigned long)check_off);
            blit_check++;
        }
    }

    /* Check if source is in GART range (system RAM mapped into GPU space) */
    bool src_is_gart = (s->regs.aic_ctrl & 1) &&
                       src_offset >= s->regs.aic_lo_addr &&
                       src_offset <= s->regs.aic_hi_addr;

    if (rop3 == 0xCC) {
        /* SRC copy blit — MC helpers handle tiling transparently */

        /* Detect intra-buffer moves AND cross-buffer copies to the
         * compositor RT — both are used during window drag. */
        if (src_offset == dst_offset && src_pitch == dst_pitch &&
            (src_x != dst_x || src_y != dst_y) &&
            src_offset >= 0x100000 && blit_w >= 4 && blit_h >= 4) {
            static int intra_log = 0;
            if (intra_log < 100) {
                fprintf(stderr,
                    "[INTRA_RT_MOVE] off=0x%x pitch=%u "
                    "src=(%u,%u) dst=(%u,%u) %ux%u\n",
                    src_offset, src_pitch / bpp,
                    src_x, src_y, dst_x, dst_y,
                    blit_w, blit_h);
                intra_log++;
            }
        }

        /* Try shadow RT redirect first: if the 3D renderer has a
         * cached shadow RT covering this BLIT source, read pixels
         * from the shadow buffer instead of VRAM. */
        if (!r200_direct_enabled() && s->renderer && s->renderer->blit_2d) {
            PPCMacGPUBlit blit_desc = {
                .src_x = src_x, .src_y = src_y,
                .src_offset = src_offset,
                .src_pitch = src_pitch,
                .dst_x = dst_x, .dst_y = dst_y,
                .dst_offset = dst_offset,
                .dst_pitch = dst_pitch,
                .width = blit_w, .height = blit_h,
                .rop3 = rop3,
                .bpp = bpp * 8,
            };
            int blit_ret = s->renderer->blit_2d(
                s->renderer_opaque, vram, &blit_desc);
            if (blit_ret == 0) {
                /* Shadow RT handled the BLIT — mark dirty */
                if (dst_offset <= 0x10000) {
                    fb_write_watch("mmio_srt_present", "srt",
                                   src_offset, dst_offset,
                                   dst_x, dst_y, blit_w, blit_h,
                                   true, true);
                }
                uint64_t dirty_start = (uint64_t)dst_offset +
                                       (uint64_t)dst_y * dst_pitch +
                                       (uint64_t)dst_x * bpp;
                uint64_t dirty_len = (blit_h > 1)
                    ? (uint64_t)(blit_h - 1) * dst_pitch + (uint64_t)blit_w * bpp
                    : (uint64_t)blit_w * bpp;
                ppc_vram_dirty_e0(s, dirty_start, dirty_len);

                /* Phase A/C: stride override for shadow RT path */
                if (dst_offset == s->regs.crtc_offset &&
                    s->disp.bpp == 32 && dst_pitch > 0 &&
                    blit_w >= 256 && blit_h >= 256) {
                    static int srt_present_log = 0;
                    if (srt_present_log < 50) {
                        fprintf(stderr,
                            "[PRESENT_BLIT] MMIO_SRT: "
                            "src=0x%x+%u dst=0x%x+%u "
                            "sxy=%u,%u dxy=%u,%u %ux%u "
                            "crtc_stride=%u\n",
                            src_offset, src_pitch,
                            dst_offset, dst_pitch,
                            src_x, src_y, dst_x, dst_y,
                            blit_w, blit_h,
                            s->disp.stride);
                        srt_present_log++;
                    }
                    if (dst_pitch != s->disp.stride ||
                        !s->disp_stride_override_active) {
                        static int srt_so_log = 0;
                        if (srt_so_log < 20) {
                            if (ppcgpu_stride_diag()) fprintf(stderr,
                                "[STRIDE_CHANGE] MMIO_SRT: "
                                "override %s %u -> %u\n",
                                s->disp_stride_override_active
                                    ? "update" : "activate",
                                s->disp.stride, dst_pitch);
                            srt_so_log++;
                        }
                    }
                    r200_set_present_pitch(s, dst_pitch);
                }

                /* SRT write-through: metal_blit_2d read from source SRT
                 * and wrote to VRAM, but if the DESTINATION also has an
                 * SRT (intra-RT moves during window drag), update it. */
                if (!r200_direct_enabled() && s->renderer->srt_write_through &&
                    dst_offset != s->regs.crtc_offset) {
                    s->renderer->srt_write_through(
                        s->renderer_opaque, vram,
                        dst_offset, dst_pitch,
                        dst_x, dst_y, blit_w, blit_h,
                        bpp * 8);
                }
                return;
            }
        }

        if (dst_offset <= 0x10000) {
            fb_write_watch(src_is_gart ? "mmio_gart_blit" : "mmio_vram_blit",
                           src_is_gart ? "gart" : "vram",
                           src_offset, dst_offset,
                           dst_x, dst_y, blit_w, blit_h,
                           false, true);
        }

        if (src_is_gart) {
            /* Source is in GART-mapped system RAM (always linear).
             * Write to VRAM destination via MC helper (handles tiling). */
            for (uint32_t row = 0; row < blit_h; row++) {
                for (uint32_t col = 0; col < blit_w; col++) {
                    uint32_t gpu_addr = src_offset +
                                        (src_y + row) * src_pitch +
                                        (src_x + col) * bpp;
                    hwaddr phys;
                    uint32_t pixel = 0;

                    if (ppc_mac_gpu_gart_translate(s, gpu_addr, &phys)) {
                        address_space_read(&address_space_memory, phys,
                                           MEMTXATTRS_UNSPECIFIED,
                                           &pixel, 4);
                    }

                    uint64_t dst_linear = (uint64_t)dst_offset +
                                          (uint64_t)(dst_y + row) * dst_pitch +
                                          (uint64_t)(dst_x + col) * bpp;
                    mc_vram_write32(s, vram, dst_linear, pixel);
                }
            }
        } else {
            /*
             * VRAM → VRAM copy.
             * MC helpers handle tiling for both src and dst transparently.
             *
             * `skip` samples two source pixels to spot a blit coming from an
             * all-zero off-screen buffer.  It is DIAGNOSTIC ONLY and must not
             * gate the copy: a two-pixel sample cannot tell an unpopulated
             * buffer from legitimately black content, and suppressing those
             * blits silently drops real drawing.
             */
            uint64_t chk_linear = (uint64_t)src_offset +
                                  (uint64_t)src_y * src_pitch +
                                  (uint64_t)src_x * bpp;
            bool skip = false;
            if (src_offset >= 0x100000) {
                uint32_t fp = mc_vram_read32(s, vram, chk_linear);
                uint64_t mp_linear = chk_linear + (uint64_t)(blit_w / 2) * bpp;
                uint32_t mp_v = mc_vram_read32(s, vram, mp_linear);
                if (fp == 0 && mp_v == 0) skip = true;
            }

            /*
             * Phase A — PRESENT_BLIT tiling audit.
             * For the first non-skipped PRESENT_BLIT from an offscreen RT,
             * check whether mc_find_tiled_surface finds the source, and
             * log sample pixel reads to prove tiling is applied correctly.
             */
            if (dst_offset == s->regs.crtc_offset &&
                src_offset >= 0x100000 && blit_w >= 256 && blit_h >= 256) {
                static int present_audit_count = 0;
                if (present_audit_count < 3) {
                    present_audit_count++;
                    uint32_t sb, sp;
                    int sl = mc_find_tiled_surface(s, (uint64_t)src_offset,
                                                    &sb, &sp);
                    fprintf(stderr,
                        "[TILE_PATH] rt_off=0x%x path=present_blit_src_check "
                        "tiling_applied=%s "
                        "src_offset=0x%x src_pitch=%u "
                        "mc_surface_found=%s mc_slot=%d "
                        "mc_surf_base=0x%x mc_surf_pitch=%u "
                        "skip=%s "
                        "src_po=0x%08x src_po_macro=%d "
                        "note=blit_%ux%u\n",
                        src_offset,
                        (sl >= 0) ? "yes" : "NO_TILED_SURFACE",
                        src_offset, src_pitch,
                        (sl >= 0) ? "yes" : "no", sl,
                        (sl >= 0) ? sb : 0, (sl >= 0) ? sp : 0,
                        skip ? "yes" : "no",
                        src_po, (int)((src_po >> 30) & 1),
                        blit_w, blit_h);
                    /* If not skipped, log first 3 sample pixel reads */
                    if (!skip && sl >= 0) {
                        for (int si = 0; si < 3; si++) {
                            uint32_t sx = (si == 0) ? 0
                                        : (si == 1) ? 64
                                        : 100;
                            uint32_t sy = (si == 0) ? 0
                                        : (si == 1) ? 0
                                        : 20;
                            uint64_t sl_addr = (uint64_t)src_offset +
                                (uint64_t)(src_y + sy) * src_pitch +
                                (uint64_t)(src_x + sx) * bpp;
                            uint32_t pix = mc_vram_read32(s, vram, sl_addr);
                            /* Also read raw (no tiling) for comparison */
                            uint32_t raw_pix = 0;
                            if (sl_addr + 4 <= s->vram_size) {
                                raw_pix = *(uint32_t *)(vram + sl_addr);
                            }
                            fprintf(stderr,
                                "[TILE_PATH] rt_off=0x%x path=present_blit_pixel "
                                "x=%u y=%u linear_addr=0x%llx "
                                "mc_read=0x%08x raw_read=0x%08x "
                                "tiled_vs_raw=%s\n",
                                src_offset,
                                src_x + sx, src_y + sy,
                                (unsigned long long)sl_addr,
                                pix, raw_pix,
                                (pix == raw_pix) ? "same" : "DIFFERENT");
                        }
                    }
                }
            }

            /* Overlapping copies must walk away from the overlap —
             * see blit_overlaps(). */
            bool rev_rows = false, rev_cols = false;
            if (blit_overlaps(src_offset, src_pitch, src_x, src_y,
                              dst_offset, dst_pitch, dst_x, dst_y,
                              blit_w, blit_h, bpp)) {
                rev_rows = dst_y > src_y;
                rev_cols = (dst_y == src_y) && (dst_x > src_x);
            }

            if (!blit_rect_untiled(s, vram, src_offset, src_pitch,
                                   src_x, src_y, dst_offset, dst_pitch,
                                   dst_x, dst_y, blit_w, blit_h, bpp))
            for (uint32_t r = 0; r < blit_h; r++) {
                uint32_t row = rev_rows ? (blit_h - 1 - r) : r;
                for (uint32_t c = 0; c < blit_w; c++) {
                    uint32_t col = rev_cols ? (blit_w - 1 - c) : c;
                    uint64_t src_linear = (uint64_t)src_offset +
                                          (uint64_t)(src_y + row) * src_pitch +
                                          (uint64_t)(src_x + col) * bpp;
                    uint64_t dst_linear = (uint64_t)dst_offset +
                                          (uint64_t)(dst_y + row) * dst_pitch +
                                          (uint64_t)(dst_x + col) * bpp;
                    uint32_t pixel = mc_vram_read32(s, vram, src_linear);
                    mc_vram_write32(s, vram, dst_linear, pixel);
                }
            }
        }
        /* Mark display dirty */
        uint64_t dirty_start = (uint64_t)dst_offset +
                               (uint64_t)dst_y * dst_pitch +
                               (uint64_t)dst_x * bpp;
        uint64_t dirty_len = (blit_h > 1)
            ? (uint64_t)(blit_h - 1) * dst_pitch + (uint64_t)blit_w * bpp
            : (uint64_t)blit_w * bpp;
        ppc_vram_dirty_e0(s, dirty_start, dirty_len);

        /*
         * SRT write-through: if this 2D BLIT wrote to a VRAM offset
         * that has an active SRT, update the SRT with the written pixels.
         * This is critical for window drag: the compositor repositions
         * window bodies via 2D BLITs, and the SRT must reflect this
         * content so that PRESENT_BLITs show the moved window.
         */
        if (!r200_direct_enabled() && s->renderer && s->renderer->srt_write_through &&
            dst_offset != s->regs.crtc_offset) {
            s->renderer->srt_write_through(
                s->renderer_opaque, vram,
                dst_offset, dst_pitch,
                dst_x, dst_y, blit_w, blit_h,
                bpp * 8);
        }
        if (dst_offset == s->regs.crtc_offset) {
            /*
             * The screen's size comes from the CRTC rather than from
             * s->disp, which is only filled in when the console refreshes
             * -- and with no display attached that is only when somebody
             * asks for a screendump.
             */
            pe_window_saw_blit(src_offset, src_pitch, dst_x, dst_y,
                               blit_w, blit_h,
                               ((((s->regs.crtc_h_total_disp) >> 16) & 0xFF) + 1) * 8,
                               (((s->regs.crtc_v_total_disp) >> 16) & 0x7FF) + 1);
            frame_tracker_record_2d_event(PASS_EVENT_FALLBACK_2D,
                                          src_offset, src_pitch,
                                          src_offset, dst_offset,
                                          src_x, src_y,
                                          dst_x, dst_y,
                                          blit_w, blit_h,
                                          false);
        } else {
            frame_tracker_record_2d_event(PASS_EVENT_BLIT_UPLOAD,
                                          dst_offset, dst_pitch,
                                          src_offset, dst_offset,
                                          src_x, src_y,
                                          dst_x, dst_y,
                                          blit_w, blit_h,
                                          false);
        }

        /* Phase A/C: stride override for non-shadow MMIO path */
        if (dst_offset == s->regs.crtc_offset &&
            s->disp.bpp == 32 && dst_pitch > 0 &&
            blit_w >= 256 && blit_h >= 256) {
            static int mmio_pb_log = 0;
            if (mmio_pb_log < 50) {
                fprintf(stderr,
                    "[PRESENT_BLIT] MMIO_NOSRT: "
                    "src=0x%x+%u dst=0x%x+%u "
                    "sxy=%u,%u dxy=%u,%u %ux%u "
                    "crtc_stride=%u\n",
                    src_offset, src_pitch,
                    dst_offset, dst_pitch,
                    src_x, src_y, dst_x, dst_y,
                    blit_w, blit_h,
                    s->disp.stride);
                mmio_pb_log++;
            }
            if (dst_pitch != s->disp.stride ||
                !s->disp_stride_override_active) {
                static int mmio_nosrt_so = 0;
                if (mmio_nosrt_so < 20) {
                    if (ppcgpu_stride_diag()) fprintf(stderr,
                        "[STRIDE_CHANGE] MMIO_NOSRT: "
                        "override %s %u -> %u\n",
                        s->disp_stride_override_active
                            ? "update" : "activate",
                        s->disp.stride, dst_pitch);
                    mmio_nosrt_so++;
                }
            }
            r200_set_present_pitch(s, dst_pitch);

            /* Content probe: check if source VRAM has non-zero data */
            {
                static int content_probe_log = 0;
                if (gpu_diag_on() && content_probe_log < 30) {
                    int nonzero_count = 0;
                    int total_probed = 0;
                    uint32_t sample_px = 0;
                    /* Sample every 16th pixel in a grid */
                    for (uint32_t py = src_y; py < src_y + blit_h; py += 16) {
                        for (uint32_t px = src_x; px < src_x + blit_w; px += 16) {
                            uint64_t addr = (uint64_t)src_offset +
                                (uint64_t)py * src_pitch + (uint64_t)px * bpp;
                            if (addr + 4 <= s->vram_size) {
                                uint32_t v = mc_vram_read32(s, vram, addr);
                                if (v != 0) {
                                    nonzero_count++;
                                    if (!sample_px) sample_px = v;
                                }
                                total_probed++;
                            }
                        }
                    }
                    fprintf(stderr,
                        "[VRAM_PROBE] PRESENT src=0x%x+%u "
                        "sxy=%u,%u %ux%u dst_xy=%u,%u "
                        "nonzero=%d/%d sample=0x%08x\n",
                        src_offset, src_pitch,
                        src_x, src_y, blit_w, blit_h,
                        dst_x, dst_y,
                        nonzero_count, total_probed, sample_px);
                    content_probe_log++;
                }
            }
        }
    } else if (rop3 == 0xF0) {
        /* Pattern fill (solid color from DP_BRUSH_FRGD_CLR) */
        uint32_t color = s->regs.dp_brush_frgd_clr;
        for (uint32_t row = 0; row < blit_h; row++) {
            for (uint32_t col = 0; col < blit_w; col++) {
                uint64_t dst_linear = (uint64_t)dst_offset +
                                      (uint64_t)(dst_y + row) * dst_pitch +
                                      (uint64_t)(dst_x + col) * bpp;
                fill_vram_px(s, vram, dst_linear, color, bpp);
            }
        }
        uint64_t fill_dirty_start = (uint64_t)dst_offset +
                                    (uint64_t)dst_y * dst_pitch +
                                    (uint64_t)dst_x * bpp;
        uint64_t fill_dirty_len = (blit_h > 1)
            ? (uint64_t)(blit_h - 1) * dst_pitch + (uint64_t)blit_w * bpp
            : (uint64_t)blit_w * bpp;
        ppc_vram_dirty_e0(s, fill_dirty_start, fill_dirty_len);
        r200_fill_notify(s, dst_offset, dst_pitch, dst_x, dst_y,
                         blit_w, blit_h, bpp, color);
    }
    /* Other ROP3 values: silently ignore for now */
}

/*
 * Execute a 2D blit using separate SRC_OFFSET/SRC_PITCH and
 * DST_OFFSET/DST_PITCH registers.  Triggered by DST_HEIGHT_WIDTH (0x143C).
 *
 * The kext uses this path to blit from GART-mapped system RAM to VRAM
 * offscreen surfaces.  The source offset may be an AGP/GART GPU address
 * (0x10000000+ range) pointing to system RAM.
 */
/*
 * A copy into system memory through the GART or the AGP aperture: how
 * ATIRadeon9700 fills a surface's backing store (from VRAM) and how the
 * OpenGL driver fills vertex buffers (from a staging buffer that is itself
 * in AGP memory: Doom 3's menu vertices).  VRAM holds the CPU's view
 * through the 32-bit swapping aperture (see the uploads in gart_blit_rows),
 * as system memory does; the card stores its own bytes through the
 * destination swap (0x15D0), so a swap of 2 is a plain copy.
 */
static unsigned g_r300_dst_swap = 2;
static void ppc_mac_gpu_2d_blit_to_gart(PPCMacGPUState *s, uint32_t src_offset,
                                        uint32_t src_pitch, uint32_t src_x,
                                        uint32_t src_y, uint32_t dst_offset,
                                        uint32_t dst_pitch, uint32_t dst_x,
                                        uint32_t dst_y, uint32_t w, uint32_t h,
                                        uint32_t bpp)
{
    bool gart = (s->regs.aic_ctrl & 1) && dst_offset >= s->regs.aic_lo_addr &&
                dst_offset <= s->regs.aic_hi_addr;
    uint32_t agp_loc = s->regs.mc_agp_location;
    bool agp = !gart && agp_loc &&
               dst_offset >= ((agp_loc & 0xFFFF) << 16) &&
               dst_offset <= ((((agp_loc >> 16) & 0xFFFF) << 16) | 0xFFFF);
    /* a source past VRAM is system memory too (GART or AGP) */
    bool src_sys = src_offset >= s->vram_size;
    static uint32_t seen[16];
    static unsigned nseen;
    uint32_t key = (src_sys << 31) ^ (g_r300_dst_swap << 28) ^ (s->r300_src_swap << 26) ^
                   (bpp << 20) ^ (gart << 19) ^ (agp << 18) ^ (dst_pitch & 0x3FFFF);
    unsigned k;

    for (k = 0; k < nseen && seen[k] != key; k++) {
    }
    if (k == nseen && nseen < ARRAY_SIZE(seen)) {
        seen[nseen++] = key;
        qemu_log("ppc-mac-gpu: 2D copy %s %06x/%u -> %s %08x/%u %ux%u bpp %u "
                 "swap src %u dst %u\n", src_sys ? "sysmem" : "VRAM", src_offset, src_pitch,
                 gart ? "GART" : agp ? "AGP" : "unmapped", dst_offset, dst_pitch,
                 w, h, bpp, s->r300_src_swap, g_r300_dst_swap);
    }
    if ((!gart && !agp) || (bpp != 4 && bpp != 2 && bpp != 1) ||
        (!src_sys &&
         (uint64_t)src_offset + (uint64_t)(src_y + h) * src_pitch > s->vram_size)) {
        return;
    }
    /* The GPU may still be drawing the source: finish that first. */
    if (!src_sys) {
        r200_vram_access(s, R200_ROWS(src_offset, src_pitch, src_y, h), false, 2);
    }

    AddressSpace *as = pci_get_address_space(&s->pci);
    const uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    size_t row_bytes = (size_t)w * bpp;
    uint8_t *row = g_malloc(row_bytes);

    for (uint32_t y = 0; y < h; y++) {
        if (src_sys) {
            uint32_t sg = src_offset + (src_y + y) * src_pitch + src_x * bpp;
            for (size_t done = 0; done < row_bytes;) {
                hwaddr phys;
                bool ok = ppc_mac_gpu_agp_translate(s, sg + done, &phys) ||
                          ppc_mac_gpu_gart_translate(s, sg + done, &phys);
                size_t n = MIN(0x1000 - ((sg + done) & 0xFFF), row_bytes - done);
                if (ok) {
                    address_space_read(as, phys, MEMTXATTRS_UNSPECIFIED, row + done, n);
                } else {
                    memset(row + done, 0, n);
                }
                done += n;
            }
        } else {
            memcpy(row, vram + src_offset + (uint64_t)(src_y + y) * src_pitch +
                   (uint64_t)src_x * bpp, row_bytes);
        }
        if (bpp == 4 && g_r300_dst_swap != 2) {
            for (size_t i = 0; i + 4 <= row_bytes; i += 4) {
                stl_he_p(row + i, r300_swap_mode(bswap32(ldl_he_p(row + i)),
                                                 g_r300_dst_swap));
            }
        }
        uint32_t gpu = dst_offset + (dst_y + y) * dst_pitch + dst_x * bpp;
        for (size_t done = 0; done < row_bytes;) {
            hwaddr phys;
            bool ok = gart ? ppc_mac_gpu_gart_translate(s, gpu + done, &phys)
                           : ppc_mac_gpu_agp_translate(s, gpu + done, &phys);
            size_t n = ok ? 0x1000 - (phys & 0xFFF) : 0x1000 - ((gpu + done) & 0xFFF);
            n = MIN(n, row_bytes - done);
            if (ok) {
                address_space_write(as, phys, MEMTXATTRS_UNSPECIFIED, row + done, n);
            }
            done += n;
        }
    }
    g_free(row);
}

static void ppc_mac_gpu_2d_blit_sep(PPCMacGPUState *s)
{
    uint32_t src_offset = s->regs.src_offset;
    uint32_t src_pitch  = s->regs.src_pitch;  /* R200 separate PITCH regs are in bytes */
    uint32_t dst_offset = s->regs.dst_offset;
    uint32_t dst_pitch  = s->regs.dst_pitch;  /* R200 separate PITCH regs are in bytes */

    /*
     * MC-style tiling: the DST_PITCH_OFFSET register may describe a tiled
     * parent surface.  Ensure it's registered so mc_vram_write32 can handle
     * tiling transparently for all writes to that surface's VRAM range.
     */
    uint32_t parent_po    = s->regs.dst_pitch_offset;
    uint32_t parent_base  = (parent_po & 0x3FFFFF) << 10;
    uint32_t parent_pitch = (parent_po & 0x3FC00000) >> 16;
    bool     parent_macro = (parent_po >> 30) & 1;
    if (parent_macro && parent_base > 0 && parent_pitch > 0) {
        mc_register_tiled_surface(s, parent_base, parent_pitch, "SEP_parent");
    }

    uint32_t dwh = s->regs.dst_width_height;
    uint32_t blit_w = (dwh >> 16) & 0x3FFF;
    uint32_t blit_h = dwh & 0x3FFF;

    uint32_t src_yx = s->regs.src_y_x;
    uint32_t dst_yx = s->regs.dst_y_x;
    uint32_t src_x = src_yx & 0xFFFF;
    uint32_t src_y = (src_yx >> 16) & 0xFFFF;
    uint32_t dst_x = dst_yx & 0xFFFF;
    uint32_t dst_y = (dst_yx >> 16) & 0xFFFF;

    uint32_t gmc = s->regs.dp_gui_master_cntl;
    uint32_t rop3 = (gmc >> 16) & 0xFF;
    uint32_t src_type = (gmc >> 24) & 0x7;
    r200_note_2d_gaps(s, gmc, "MMIO separate-pitch");

    if (blit_w == 0 || blit_h == 0 || dst_pitch == 0) {
        return;
    }
    /* Into system memory: a copy, or host data (src_type 3, the OpenGL
     * driver's vertex buffer uploads) that ppc_mac_gpu_host_data_write
     * stores through the GART or the AGP aperture. */
    if (dst_offset >= s->vram_size && src_type != 3) {
        ppc_mac_gpu_2d_blit_to_gart(s, src_offset, src_pitch ? src_pitch : dst_pitch,
                                    src_x, src_y, dst_offset, dst_pitch,
                                    dst_x, dst_y, blit_w, blit_h, gmc_dst_bpp(gmc));
        return;
    }

    if (dst_offset < s->vram_size) {
        r200_vram_access(s, R200_ROWS(dst_offset, dst_pitch, dst_y, blit_h), true, 2);
    }
    if (src_offset < s->vram_size) {
        r200_vram_access(s, R200_ROWS(src_offset, src_pitch ? src_pitch : dst_pitch,
                                      src_y, blit_h), false, 2);
    }
    seq_log("SEP  rop=%02x srctype=%u src=%06x/%u (%u,%u) "
            "dst=%06x/%u (%u,%u) %ux%u",
            rop3, src_type, src_offset, src_pitch, src_x, src_y,
            dst_offset, dst_pitch, dst_x, dst_y, blit_w, blit_h);

    /* Source type 3 = HOST data */
    if (src_type == 3) {
        s->host_data_offset = dst_offset;
        s->host_data_pitch = dst_pitch;
        s->host_data_macro_tile = false;  /* MC handles tiling */
        s->host_data_dst_x = dst_x;
        s->host_data_dst_y = dst_y;
        s->host_data_w = blit_w;
        s->host_data_h = blit_h;
        s->host_data_cur_x = 0;
        s->host_data_cur_y = 0;
        s->host_data_bpp = gmc_dst_bpp(gmc);
        s->host_data_active = true;
        return;
    }

    if (src_pitch == 0) {
        src_pitch = dst_pitch;
    }

    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t bpp = gmc_dst_bpp(gmc);

    /* Check if source is in GART/AGP range (system RAM mapped into GPU) */
    bool src_is_gart = (s->regs.aic_ctrl & 1) &&
                       src_offset >= s->regs.aic_lo_addr &&
                       src_offset <= s->regs.aic_hi_addr;
    bool src_is_agp = false;
    if (!src_is_gart) {
        uint32_t agp_loc = s->regs.mc_agp_location;
        if (agp_loc != 0) {
            uint32_t agp_start = (agp_loc & 0xFFFF) << 16;
            uint32_t agp_end = (((agp_loc >> 16) & 0xFFFF) << 16) | 0xFFFF;
            src_is_agp = (src_offset >= agp_start && src_offset <= agp_end);
        }
    }

    /* Path instrumentation */
    g_blit_stats.sep_count++;
    {
        static uint64_t sep_logged = 0;
        if (sep_logged < 500) {
            uint32_t mc_base, mc_pitch;
            uint64_t test_addr = (uint64_t)dst_offset +
                                 (uint64_t)dst_y * dst_pitch +
                                 (uint64_t)dst_x * bpp;
            int mc_slot = mc_find_tiled_surface(s, test_addr, &mc_base, &mc_pitch);
            blit_path_log("SEP_MC",
                "src_off=0x%06x src_pitch=%u "
                "dst_off=0x%06x dst_pitch=%u  "
                "%ux%u rop=0x%02x src_type=%u "
                "src_is_gart=%d src_is_agp=%d "
                "mc_tiled=%s(slot=%d)",
                src_offset, src_pitch,
                dst_offset, dst_pitch,
                blit_w, blit_h, rop3, src_type,
                src_is_gart, src_is_agp,
                mc_slot >= 0 ? "YES" : "no", mc_slot);
            sep_logged++;
        }
    }

    if (rop3 == 0xCC) {
        /* SRC copy blit — MC helpers handle tiling transparently */
        if (src_is_gart || src_is_agp) {
            /* Source is in GART/AGP-mapped system RAM (always linear).
             * Write to VRAM destination via MC helper. */
            AddressSpace *as = pci_get_address_space(&s->pci);

            /* ---- Phase A+B: Wallpaper upload verification ---- */
            static int wup_log_count = 0;
            static int tex_upload_log = 0;
            bool is_tex_range = (dst_offset >= 0x353000 &&
                                 dst_offset < 0x420000 &&
                                 blit_w == 128 && blit_h >= 16);
            bool wup_log = ((wup_log_count < 30 || (is_tex_range && tex_upload_log < 20)) &&
                            dst_offset >= 0x100000 &&
                            blit_w >= 32 && blit_h >= 32);

            /* Phase B: capture VRAM before */
            uint32_t vram_before[4] = {0};
            if (wup_log) {
                uint64_t check_off = (uint64_t)dst_offset +
                                     (uint64_t)dst_y * dst_pitch +
                                     (uint64_t)dst_x * bpp;
                for (int i = 0; i < 4 && check_off + (i+1)*4 <= s->vram_size; i++) {
                    memcpy(&vram_before[i], vram + check_off + i*4, 4);
                }
            }

            uint32_t xlate_ok = 0, xlate_fail = 0;
            uint32_t nonzero_pixels = 0;
            uint32_t first_pixel = 0, pixel_8_8 = 0;
            uint32_t distinct_pixels = 0;
            uint32_t prev_pixel_val = 0;
            bool prev_set = false;

            /* Phase A deep: log AGP translate details for first upload */
            static int agp_detail_log = 0;

            bool fast = !wup_log &&
                        !mc_rect_is_tiled(s, dst_offset, dst_pitch,
                                          dst_x, dst_y, bpp);
            if (fast) {
                gart_blit_rows(s, as, src_is_gart, src_offset, src_pitch,
                               src_x, src_y, vram, dst_offset, dst_pitch,
                               dst_x, dst_y, blit_w, blit_h, bpp);
            }

            for (uint32_t row = 0; !fast && row < blit_h; row++) {
                uint32_t row_gpu_addr = src_offset +
                                        (src_y + row) * src_pitch +
                                        src_x * bpp;

                for (uint32_t col = 0; col < blit_w; col++) {
                    uint32_t gpu_addr = row_gpu_addr + col * bpp;
                    hwaddr phys = 0;
                    uint32_t pixel = 0;
                    bool xlated = false;

                    if (src_is_gart) {
                        xlated = ppc_mac_gpu_gart_translate(s, gpu_addr, &phys);
                    } else {
                        xlated = ppc_mac_gpu_agp_translate(s, gpu_addr, &phys);
                    }
                    if (xlated) {
                        address_space_read(as, phys,
                                           MEMTXATTRS_UNSPECIFIED,
                                           &pixel, 4);
                        /*
                         * VRAM holds what the CPU sees through its
                         * byte-swapping aperture (32-bit swap for 32bpp,
                         * 16-bit for 16bpp).  The copy lands in the card's
                         * view after the 0x15D4 swap; store the CPU view.
                         * A swap equal to the aperture's is a plain copy.
                         */
                        if (s->r300 && bpp == 4 && s->r300_src_swap != 2) {
                            pixel = bswap32(r300_swap_mode(pixel, s->r300_src_swap));
                        } else if (s->r300 && bpp == 2 && s->r300_src_swap == 0) {
                            pixel = r300_swap_mode(pixel, 1);
                        }
                        xlate_ok++;
                    } else {
                        xlate_fail++;
                    }

                    /* AGP detail logging for first upload, first few pixels */
                    if (agp_detail_log == 0 && wup_log &&
                        ((row == 0 && col < 4) ||
                         (row == 1 && col == 0) ||
                         (row == 128 && col == 128))) {
                        blit_path_log("AGP_DETAIL",
                            "row=%u col=%u gpu_addr=0x%08x phys=0x%llx "
                            "xlated=%d pixel=0x%08x",
                            row, col, gpu_addr,
                            (unsigned long long)phys, xlated, pixel);
                    }

                    if (pixel != 0) nonzero_pixels++;
                    if (row == 0 && col == 0) first_pixel = pixel;
                    if (row == 8 && col == 8) pixel_8_8 = pixel;
                    if (!prev_set) {
                        prev_pixel_val = pixel;
                        prev_set = true;
                        distinct_pixels = 1;
                    } else if (pixel != prev_pixel_val) {
                        distinct_pixels++;
                        prev_pixel_val = pixel;
                    }

                    /* Write via MC — tiling applied transparently */
                    uint64_t dst_linear = (uint64_t)dst_offset +
                        (uint64_t)(dst_y + row) * dst_pitch +
                        (uint64_t)(dst_x + col) * bpp;
                    mc_vram_write32(s, vram, dst_linear, pixel);
                }
            }

            /* Phase A+B logging */
            if (wup_log) {
                uint64_t check_off = (uint64_t)dst_offset +
                                     (uint64_t)dst_y * dst_pitch +
                                     (uint64_t)dst_x * bpp;
                uint32_t vram_after[4] = {0};
                for (int i = 0; i < 4 && check_off + (i+1)*4 <= s->vram_size; i++) {
                    memcpy(&vram_after[i], vram + check_off + i*4, 4);
                }
                bool changed = memcmp(vram_before, vram_after, 16) != 0;
                blit_path_log("WALLPAPER_UPLOAD",
                    "src_off=0x%x src_pitch=%u dst_off=0x%x dst_pitch=%u "
                    "size=(%u,%u) src_type=%s "
                    "xlate_ok=%u xlate_fail=%u nonzero=%u/%u distinct=%u "
                    "first_pixel=0x%08x pixel_8_8=0x%08x "
                    "vram_changed=%s "
                    "before=[0x%08x,0x%08x,0x%08x,0x%08x] "
                    "after=[0x%08x,0x%08x,0x%08x,0x%08x]",
                    src_offset, src_pitch, dst_offset, dst_pitch,
                    blit_w, blit_h,
                    src_is_gart ? "GART" : "AGP",
                    xlate_ok, xlate_fail, nonzero_pixels, blit_w * blit_h,
                    distinct_pixels,
                    first_pixel, pixel_8_8,
                    changed ? "YES" : "NO",
                    vram_before[0], vram_before[1], vram_before[2], vram_before[3],
                    vram_after[0], vram_after[1], vram_after[2], vram_after[3]);
                wup_log_count++;
                if (is_tex_range) tex_upload_log++;
                agp_detail_log++;
            }
        } else {
            /*
             * Source is in VRAM.
             *
             * This used to skip blits whose first and middle source pixels
             * were both zero, on the theory that they came from an
             * unpopulated back buffer.  A two-pixel sample cannot tell that
             * apart from legitimately black content, so the test dropped
             * real drawing; the copy now always runs.
             *
             * Source and destination may overlap — see blit_overlaps().
             */
            bool rev_rows = false, rev_cols = false;
            if (blit_overlaps(src_offset, src_pitch, src_x, src_y,
                              dst_offset, dst_pitch, dst_x, dst_y,
                              blit_w, blit_h, bpp)) {
                rev_rows = dst_y > src_y;
                rev_cols = (dst_y == src_y) && (dst_x > src_x);
            }
            if (!blit_rect_untiled(s, vram, src_offset, src_pitch,
                                   src_x, src_y, dst_offset, dst_pitch,
                                   dst_x, dst_y, blit_w, blit_h, bpp)) {
                for (uint32_t r = 0; r < blit_h; r++) {
                    uint32_t row = rev_rows ? (blit_h - 1 - r) : r;
                    for (uint32_t c = 0; c < blit_w; c++) {
                        uint32_t col = rev_cols ? (blit_w - 1 - c) : c;
                        uint64_t src_linear = (uint64_t)src_offset +
                            (uint64_t)(src_y + row) * src_pitch +
                            (uint64_t)(src_x + col) * bpp;
                        uint64_t dst_linear = (uint64_t)dst_offset +
                            (uint64_t)(dst_y + row) * dst_pitch +
                            (uint64_t)(dst_x + col) * bpp;
                        /* MC handles tiling for both src and dst */
                        uint32_t pixel = mc_vram_read32(s, vram, src_linear);
                        mc_vram_write32(s, vram, dst_linear, pixel);
                    }
                }
            }
        }

        /* Mark dirty */
        uint64_t dirty_start = (uint64_t)dst_offset +
                               (uint64_t)dst_y * dst_pitch +
                               (uint64_t)dst_x * bpp;
        uint64_t dirty_len = (blit_h > 1)
            ? (uint64_t)(blit_h - 1) * dst_pitch + (uint64_t)blit_w * bpp
            : (uint64_t)blit_w * bpp;
        if (dirty_start + dirty_len <= s->vram_size) {
            ppc_vram_dirty_e0(s, dirty_start, dirty_len);
        }
        s->display_invalid = true;

        /*
         * Phase A: [PRESENT_BLIT] logging for MMIO path.
         * Phase C: Stride override for MMIO BLT path.
         */
        if (dst_offset == s->regs.crtc_offset && s->disp.bpp == 32 &&
            dst_pitch > 0 && blit_w >= 256 && blit_h >= 256) {
            static int mmio_present_log = 0;
            if (mmio_present_log < 50) {
                fprintf(stderr,
                    "[PRESENT_BLIT] MMIO_2D: "
                    "src=0x%x+%u dst=0x%x+%u "
                    "sxy=%u,%u dxy=%u,%u %ux%u "
                    "crtc_stride=%u\n",
                    src_offset, src_pitch,
                    dst_offset, dst_pitch,
                    src_x, src_y, dst_x, dst_y,
                    blit_w, blit_h,
                    s->disp.stride);
                mmio_present_log++;
            }
            if (!s->disp_stride_override_active ||
                s->disp_stride_override_value != dst_pitch) {
                static int mmio_so_log = 0;
                if (mmio_so_log < 20) {
                    if (ppcgpu_stride_diag()) fprintf(stderr,
                        "[STRIDE_CHANGE] MMIO_2D: "
                        "override %s %u -> %u\n",
                        s->disp_stride_override_active
                            ? "update" : "activate",
                        s->disp_stride_override_active
                            ? s->disp_stride_override_value
                            : s->disp.stride,
                        dst_pitch);
                    mmio_so_log++;
                }
            }
            r200_set_present_pitch(s, dst_pitch);
        }
    } else if (rop3 == 0xF0) {
        /* Pattern fill — MC helper handles tiling */
        uint32_t color = s->regs.dp_brush_frgd_clr;
        for (uint32_t row = 0; row < blit_h; row++) {
            for (uint32_t col = 0; col < blit_w; col++) {
                uint64_t dst_linear = (uint64_t)dst_offset +
                    (uint64_t)(dst_y + row) * dst_pitch +
                    (uint64_t)(dst_x + col) * bpp;
                fill_vram_px(s, vram, dst_linear, color, bpp);
            }
        }
        uint64_t dirty_start = (uint64_t)dst_offset +
                               (uint64_t)dst_y * dst_pitch +
                               (uint64_t)dst_x * bpp;
        uint64_t dirty_len = (blit_h > 1)
            ? (uint64_t)(blit_h - 1) * dst_pitch + (uint64_t)blit_w * bpp
            : (uint64_t)blit_w * bpp;
        if (dirty_start + dirty_len <= s->vram_size) {
            ppc_vram_dirty_e0(s, dirty_start, dirty_len);
        }
        r200_fill_notify(s, dst_offset, dst_pitch, dst_x, dst_y,
                         blit_w, blit_h, bpp, color);
        s->display_invalid = true;
    }
}

/*
 * Snapshot 3D register state from the shadow array for a draw call.
 */
static void ppc_mac_gpu_snapshot_3d_state(PPCMacGPUState *s,
                                           PPCMacGPU3DState *state)
{
    /* Helper to read from the 3D shadow array */
    #define RD3D(addr) s->regs.regs_3d[R200_3D_IDX(addr)]

    state->se_vtx_fmt_0     = RD3D(R200_SE_VTX_FMT_0);
    state->se_vtx_fmt_1     = RD3D(R200_SE_VTX_FMT_1);
    state->se_vte_cntl      = RD3D(R200_SE_VTE_CNTL);
    state->pp_cntl          = RD3D(R200_PP_CNTL);
    state->pp_misc          = RD3D(R200_PP_MISC);

    /* Texture unit 0 — R200 register bank */
    state->pp_txfilter_0    = RD3D(R200_PP_TXFILTER_0);
    state->pp_txformat_0    = RD3D(R200_PP_TXFORMAT_0);
    state->pp_txformat_x_0  = RD3D(R200_PP_TXFORMAT_X_0);
    state->pp_txsize_0      = RD3D(R200_PP_TXSIZE_0);
    state->pp_txpitch_0     = RD3D(R200_PP_TXPITCH_0);
    state->pp_txoffset_0    = RD3D(R200_PP_TXOFFSET_0);

    /* Texture unit 0 — R100 register bank (Apple kext uses these) */
    state->r100_pp_txfilter_0  = RD3D(R100_PP_TXFILTER_0);
    state->r100_pp_txformat_0  = RD3D(R100_PP_TXFORMAT_0);
    state->r100_pp_txoffset_0  = RD3D(R100_PP_TXOFFSET_0);
    state->r100_pp_tex_size_0  = RD3D(R100_PP_TEX_SIZE_0);
    state->r100_pp_tex_pitch_0 = RD3D(R100_PP_TEX_PITCH_0);

    /* Texture combiners */
    state->pp_txcblend_0    = RD3D(R200_PP_TXCBLEND_0);
    state->pp_txcblend2_0   = RD3D(R200_PP_TXCBLEND2_0);
    state->pp_txablend_0    = RD3D(R200_PP_TXABLEND_0);
    state->pp_txablend2_0   = RD3D(R200_PP_TXABLEND2_0);

    /* Render backend */
    state->rb3d_cntl        = RD3D(R200_RB3D_CNTL);
    state->rb3d_coloroffset = RD3D(R200_RB3D_COLOROFFSET);
    state->rb3d_colorpitch  = RD3D(R200_RB3D_COLORPITCH);
    state->rb3d_blendcntl   = RD3D(R200_RB3D_BLENDCNTL);
    state->rb3d_ablendcntl  = RD3D(R200_RB3D_ABLENDCNTL);
    state->rb3d_cblendcntl  = RD3D(R200_RB3D_CBLENDCNTL);
    state->rb3d_zstencilcntl = RD3D(R200_RB3D_ZSTENCILCNTL);

    /* Setup engine */
    state->se_cntl          = RD3D(R200_SE_CNTL);
    state->re_cntl          = RD3D(R200_RE_CNTL);

    /* Viewport */
    state->se_vport_xscale  = RD3D(R200_SE_VPORT_XSCALE);
    state->se_vport_xoffset = RD3D(R200_SE_VPORT_XOFFSET);
    state->se_vport_yscale  = RD3D(R200_SE_VPORT_YSCALE);
    state->se_vport_yoffset = RD3D(R200_SE_VPORT_YOFFSET);
    state->se_vport_zscale  = RD3D(R200_SE_VPORT_ZSCALE);
    state->se_vport_zoffset = RD3D(R200_SE_VPORT_ZOFFSET);

    /* 3D scissor */
    state->re_top_left      = RD3D(R200_RE_TOP_LEFT);
    state->re_width_height  = RD3D(R200_RE_WIDTH_HEIGHT);

    /* Screen dimensions from display mode */
    state->screen_width     = s->disp.width;
    state->screen_height    = s->disp.height;

    #undef RD3D
}

/*
 * Log 3D draw state for compositor analysis.
 * Writes to /tmp/gpu_3d_draws.log with all relevant state per draw.
 */
static void ppc_mac_gpu_log_3d_draw(const PPCMacGPU3DState *st,
                                     const PPCMacGPU3DDrawCmd *cmd)
{
    static FILE *f3d = NULL;
    static uint64_t draw_count = 0;

    if (!f3d) {
        f3d = fopen("/tmp/gpu_3d_draws.log", "w");
        if (!f3d) return;
        fprintf(f3d, "=== 3D Draw Command Log ===\n");
    }

    if (draw_count >= 2000) {
        /* Limit log size */
        if (draw_count == 2000) {
            fprintf(f3d, "--- LOG LIMIT REACHED (2000 draws) ---\n");
            fflush(f3d);
        }
        draw_count++;
        return;
    }

    const char *opname = "UNKNOWN";
    switch (cmd->opcode) {
    case R200_3D_DRAW_VBUF: opname = "DRAW_VBUF"; break;
    case R200_3D_DRAW_IMMD: opname = "DRAW_IMMD"; break;
    case R200_3D_DRAW_INDX: opname = "DRAW_INDX"; break;
    }

    const char *primname = "?";
    switch (cmd->prim_type) {
    case 0: primname = "NONE"; break;
    case 1: primname = "POINTS"; break;
    case 2: primname = "LINES"; break;
    case 3: primname = "LINE_STRIP"; break;
    case 4: primname = "TRI_LIST"; break;
    case 5: primname = "TRI_FAN"; break;
    case 6: primname = "TRI_STRIP"; break;
    case 7: primname = "TRI_TYPE2"; break;
    case 8: primname = "RECT_LIST"; break;
    case 9: primname = "3VRT_POINTS"; break;
    case 10: primname = "3VRT_LINES"; break;
    case 11: primname = "POINT_SPRITES"; break;
    case 12: primname = "LINE_LOOP"; break;
    case 13: primname = "QUAD_LIST"; break;
    case 14: primname = "QUAD_STRIP"; break;
    case 15: primname = "POLYGON"; break;
    }

    /* Decode texture format */
    uint32_t txfmt_raw = st->pp_txformat_0 & 0x1F;
    const char *txfmt_name = "?";
    switch (txfmt_raw) {
    case 0x06: txfmt_name = "ARGB8888"; break;
    case 0x07: txfmt_name = "RGBA8888"; break;
    case 0x15: txfmt_name = "XRGB8888"; break;
    case 0x05: txfmt_name = "RGB565"; break;
    case 0x04: txfmt_name = "ARGB1555"; break;
    case 0x11: txfmt_name = "ARGB4444"; break;
    case 0x01: txfmt_name = "AI88"; break;
    case 0x00: txfmt_name = "I8"; break;
    default: txfmt_name = "OTHER"; break;
    }

    /* Decode texture size */
    uint32_t tex_w = (st->pp_txsize_0 & 0x7FF) + 1;
    uint32_t tex_h = ((st->pp_txsize_0 >> 16) & 0x7FF) + 1;
    uint32_t tex_pitch = (st->pp_txpitch_0 & 0x3FFF) + 32;
    uint32_t tex_offset = st->pp_txoffset_0;
    /*
     * R200 PP_CNTL texture enable bits are at positions 4, 8, 12, 16, 20, 24
     * (NOT at bits 0-5 like R100). Bit 4 = TEX_0_ENABLE.
     */
    bool tex0_enabled = (st->pp_cntl >> 4) & 1;

    /* Decode color target */
    uint32_t color_off = st->rb3d_coloroffset;
    uint32_t color_pitch_raw = st->rb3d_colorpitch;
    uint32_t color_pitch_val = (color_pitch_raw & 0x3FFF) * 8;  /* in bytes */
    bool micro_tile = (color_pitch_raw >> 15) & 1;
    bool macro_tile = (color_pitch_raw >> 16) & 1;
    bool color_tile = micro_tile || macro_tile;

    /* Decode blend */
    uint32_t blend = st->rb3d_blendcntl;
    bool blend_enable = (blend >> 24) & 1;
    uint32_t src_blend = blend & 0xF;
    uint32_t dst_blend = (blend >> 16) & 0xF;

    /* Decode viewport (floats stored as uint32_t) */
    float vp_xscale, vp_xoff, vp_yscale, vp_yoff;
    memcpy(&vp_xscale, &st->se_vport_xscale, 4);
    memcpy(&vp_xoff, &st->se_vport_xoffset, 4);
    memcpy(&vp_yscale, &st->se_vport_yscale, 4);
    memcpy(&vp_yoff, &st->se_vport_yoffset, 4);

    /* Decode vertex format */
    bool vtx_xy = st->se_vtx_fmt_0 & 1;
    bool vtx_z = (st->se_vtx_fmt_0 >> 1) & 1;
    bool vtx_w = (st->se_vtx_fmt_0 >> 2) & 1;
    bool vtx_color_packed = (st->se_vtx_fmt_0 >> 4) & 1;
    uint32_t vtx_color_cnt = (st->se_vtx_fmt_0 >> 8) & 7;
    uint32_t tex0_comp_cnt = st->se_vtx_fmt_1 & 7;  /* 0=S,1=ST,2=STR,3=STRQ */
    bool vte_xy_fmt = (st->se_vte_cntl >> 8) & 1;  /* 1=screen-space coords */

    /* Decode scissor */
    uint32_t sc_x = st->re_top_left & 0x3FFF;
    uint32_t sc_y = (st->re_top_left >> 16) & 0x3FFF;
    uint32_t sc_w = st->re_width_height & 0x3FFF;
    uint32_t sc_h = (st->re_width_height >> 16) & 0x3FFF;

    fprintf(f3d,
        "\n[3D_DRAW #%llu]\n"
        "  opcode=%s  prim=%s  verts=%u\n"
        "  color_off=0x%06x  color_pitch=%u  color_tile=%d\n"
        "  tex0_en=%d  tex0_off=0x%06x  tex0_pitch=%u  tex0_size=%ux%u  tex0_fmt=%s(0x%x)\n"
        "  blend_en=%d  src_blend=0x%x  dst_blend=0x%x\n"
        "  vtx_fmt0=0x%08x  vtx_fmt1=0x%08x  vte_cntl=0x%08x\n"
        "  vtx: xy=%d z=%d w=%d color_packed=%d color_cnt=%u tex0_comp=%u screen_space=%d\n"
        "  viewport: xscale=%.1f xoff=%.1f yscale=%.1f yoff=%.1f\n"
        "  scissor: (%u,%u) %ux%u\n"
        "  se_cntl=0x%08x  re_cntl=0x%08x  pp_cntl=0x%08x\n"
        "  rb3d_cntl=0x%08x  zstencil=0x%08x\n",
        (unsigned long long)draw_count,
        opname, primname, cmd->num_vertices,
        color_off, color_pitch_val, color_tile,
        tex0_enabled, tex_offset, tex_pitch, tex_w, tex_h, txfmt_name, txfmt_raw,
        blend_enable, src_blend, dst_blend,
        st->se_vtx_fmt_0, st->se_vtx_fmt_1, st->se_vte_cntl,
        vtx_xy, vtx_z, vtx_w, vtx_color_packed, vtx_color_cnt, tex0_comp_cnt, vte_xy_fmt,
        vp_xscale, vp_xoff, vp_yscale, vp_yoff,
        sc_x, sc_y, sc_w, sc_h,
        st->se_cntl, st->re_cntl, st->pp_cntl,
        st->rb3d_cntl, st->rb3d_zstencilcntl);

    /* For DRAW_IMMD, dump the first few vertex data words */
    if (cmd->opcode == R200_3D_DRAW_IMMD && cmd->vertex_data && cmd->vertex_data_dwords > 0) {
        uint32_t dump_count = cmd->vertex_data_dwords < 32 ? cmd->vertex_data_dwords : 32;
        fprintf(f3d, "  immd_data[%u]:", cmd->vertex_data_dwords);
        for (uint32_t i = 0; i < dump_count; i++) {
            if (i % 8 == 0 && i > 0) fprintf(f3d, "\n               ");
            float fval;
            memcpy(&fval, &cmd->vertex_data[i], 4);
            fprintf(f3d, " %08x(%.2f)", cmd->vertex_data[i], fval);
        }
        fprintf(f3d, "\n");
    }

    /* For DRAW_VBUF, log VB and AOS info */
    if (cmd->opcode == R200_3D_DRAW_VBUF) {
        fprintf(f3d, "  vb_addr=0x%08x  vb_stride=%u (bytes)\n",
                cmd->vb_addr, cmd->vb_stride);
    }

    /* For DRAW_INDX, log index data */
    if (cmd->opcode == R200_3D_DRAW_INDX && cmd->index_data) {
        uint32_t idx_dump = cmd->num_indices < 16 ? cmd->num_indices : 16;
        fprintf(f3d, "  indices[%u]:", cmd->num_indices);
        for (uint32_t i = 0; i < idx_dump; i++) {
            fprintf(f3d, " %u", cmd->index_data[i]);
        }
        fprintf(f3d, "\n");
    }

    draw_count++;

    if (draw_count % 50 == 0) {
        fflush(f3d);
    }
}

/*
 * Dispatch a 3D draw command to the renderer backend.
 *
 * For DRAW_VBUF with GART vertex buffers, we resolve the GART address,
 * read vertex data from system RAM, and convert to DRAW_IMMD so the
 * renderer backend doesn't need GART access.
 */
/* ========================================================================
 * Direct R200 draw path
 *
 * Decodes one R200 vertex-buffer draw into an R200DrawPacket: vertices are
 * fetched from their AOS arrays (VRAM or AGP), transformed by the TCL MVP
 * matrix and the viewport into window pixels, assembled into a triangle
 * list, and handed to the renderer, which draws straight into VRAM.
 *
 * Register usage follows what Apple's R200 driver actually programs (from a
 * Quartz Extreme trace): texture units 0-2 are written at the R100 register
 * addresses (PP_TXFILTER_0 0x1C54, PP_TXFORMAT_0 0x1C58, PP_TXOFFSET_0
 * 0x1C5C, PP_TEX_SIZE_0 0x1D04, PP_TEX_PITCH_0 0x1D08, PP_TFACTOR_0 0x1C68,
 * stride 0x18 / 8), units 3-5 at the R200 ones.  Everything else is R200.
 * ======================================================================== */

static bool r200_direct_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("PPCGPU_R200_DIRECT");
        enabled = !(e && e[0] == '0');
    }
    return enabled;
}

#define R3D(a) (s->regs.regs_3d[R200_3D_IDX(a)])

static inline float r200_f32(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/*
 * Read guest-GPU memory (VRAM via MC_FB_LOCATION, else AGP/PCI GART) into
 * dst as host-order dwords.  Guest data is big-endian in both places.
 */
/*
 * AGP/GART page -> host pointer cache.  Vertex fetch reads a few dwords per
 * vertex per array from AGP memory, and each read used to walk the GART PTE
 * and the flat view twice (~25% of the vCPU in Warcraft III).  Entries are
 * dropped when the host bridge's GART registers are written (the driver
 * invalidates the GART after changing it, as real hardware requires) and at
 * every fence.
 */
static struct { uint32_t tag; uint8_t *host; } r200_agp_tc[1024];
static uint32_t r200_agp_tc_gen;

static void r200_agp_tc_flush(void)
{
    memset(r200_agp_tc, 0, sizeof(r200_agp_tc));
}

static uint8_t *r200_agp_page(PPCMacGPUState *s, uint32_t gpu_addr)
{
    uint32_t g = uninorth_get_agp_gart_gen();
    if (g != r200_agp_tc_gen) {
        r200_agp_tc_gen = g;
        r200_agp_tc_flush();
    }
    uint32_t page = gpu_addr >> 12;
    uint32_t i = page & (ARRAY_SIZE(r200_agp_tc) - 1);
    if (r200_agp_tc[i].tag == page + 1) {
        return r200_agp_tc[i].host;
    }
    hwaddr phys;
    if (!ppc_mac_gpu_agp_translate(s, page << 12, &phys) &&
        !ppc_mac_gpu_gart_translate(s, page << 12, &phys)) {
        return NULL;
    }
    RCU_READ_LOCK_GUARD();
    hwaddr xlat, plen = 0x1000;
    MemoryRegion *mr = address_space_translate(&address_space_memory,
                                               phys & ~(hwaddr)0xFFF, &xlat,
                                               &plen, false,
                                               MEMTXATTRS_UNSPECIFIED);
    if (!memory_region_is_ram(mr) || plen < 0x1000) {
        return NULL;
    }
    uint8_t *h = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
    r200_agp_tc[i].tag = page + 1;
    r200_agp_tc[i].host = h;
    return h;
}

static bool r200_read_dwords(PPCMacGPUState *s, uint32_t gpu_addr,
                             uint32_t *dst, uint32_t count)
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t len = count * 4;
    uint8_t buf[256];

    if (len > sizeof(buf)) {
        return false;
    }
    if (gpu_addr >= fb_base && gpu_addr - fb_base + len <= s->vram_size) {
        /*
         * Byte-swap straight out of VRAM.  Going via the bounce buffer made
         * a second copy of every vertex fetched, and this is the path
         * almost every fetch takes; the copy was a third of all the time
         * the emulator spent in memmove.
         */
        const uint8_t *src = (const uint8_t *)
            memory_region_get_ram_ptr(&s->vram) + (gpu_addr - fb_base);
        for (uint32_t i = 0; i < count; i++) {
            dst[i] = ldl_be_p(src + i * 4);
        }
        return true;
    } else {
        uint32_t done = 0;
        while (done < len) {
            hwaddr phys;
            uint32_t a = gpu_addr + done;
            uint32_t chunk = MIN(0x1000 - (a & 0xFFF), len - done);
            uint8_t *hp = r200_agp_page(s, a);
            if (hp) {
                memcpy(buf + done, hp + (a & 0xFFF), chunk);
                done += chunk;
                continue;
            }
            if (!ppc_mac_gpu_agp_translate(s, a, &phys) &&
                !ppc_mac_gpu_gart_translate(s, a, &phys)) {
                return false;
            }
            if (address_space_read(&address_space_memory, phys,
                                   MEMTXATTRS_UNSPECIFIED, buf + done,
                                   chunk) != MEMTX_OK) {
                return false;
            }
            done += chunk;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = ldl_be_p(buf + i * 4);
    }
    return true;
}

static void r200_warn_once(uint32_t *mask, uint32_t bit, const char *fmt, ...)
    G_GNUC_PRINTF(3, 4);
static void r200_warn_once(uint32_t *mask, uint32_t bit, const char *fmt, ...)
{
    if (*mask & bit) {
        return;
    }
    *mask |= bit;
    va_list ap;
    va_start(ap, fmt);
    char *msg = g_strdup_vprintf(fmt, ap);
    va_end(ap);
    qemu_log("ppc-mac-gpu r200: %s\n", msg);
    g_free(msg);
}

static uint32_t r200_warned;

/* Texture unit n: R100 register addresses for units 0-2, R200 for 3-5. */
/*
 * Fetch `have` words into dst and zero it out to `fill`.
 *
 * Both counts are tiny -- at most 16, and usually two or three -- so this
 * replaces a memcpy and a memset whose lengths are only known at run time.
 * They were called once per attribute per vertex, a couple of million times
 * a second, and the call overhead dwarfed the handful of words each one
 * moved: together they were the emulator's hottest memmove by a wide
 * margin.  A plain word loop the compiler can see through is faster.
 */
static inline void r200_fetch_words(uint32_t *dst, const uint32_t *src,
                                    uint32_t have, uint32_t fill)
{
    uint32_t i = 0;
    for (; i < have; i++) {
        dst[i] = src[i];
    }
    for (; i < fill; i++) {
        dst[i] = 0;
    }
}

/*
 * Grow the shared vertex scratch to hold at least n vertices and return it.
 *
 * Every growth must go through here.  The scratch is reachable from the
 * device, so reallocating it and storing the result only in a local leaves
 * the device pointing at freed memory -- which is not noticed here but in
 * the *next* draw, as a heap abort inside an unrelated allocation.
 */
static R200Vertex *r200_draw_verts(PPCMacGPUState *s, uint32_t n)
{
    if (n > s->draw_verts_cap) {
        s->draw_verts = g_renew(R200Vertex, s->draw_verts, n);
        s->draw_verts_cap = n;
    }
    return s->draw_verts;
}

static void r200_decode_tex_unit(PPCMacGPUState *s, int n, R200TexUnit *t)
{
    uint32_t filter, format, offset, size, pitch;
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;

    if (n < 3) {
        filter = R3D(0x1C54 + n * 0x18);
        format = R3D(0x1C58 + n * 0x18);
        offset = R3D(0x1C5C + n * 0x18);
        size   = R3D(0x1D04 + n * 8);
        pitch  = R3D(0x1D08 + n * 8);
    } else {
        filter = R3D(0x2C00 + n * 0x20);
        format = R3D(0x2C04 + n * 0x20);
        size   = R3D(0x2C0C + n * 0x20);
        pitch  = R3D(0x2C10 + n * 0x20);
        offset = R3D(0x2D00 + n * 0x18);
    }

    memset(t, 0, sizeof(*t));
    t->enabled = 1;
    t->format = format & 0x1F;
    t->alpha_in_map = (format >> 6) & 1;
    t->filter = filter;
    uint32_t bpp = (t->format == 0 || t->format == 8) ? 1 :      /* I8, Y8 */
                   (t->format == 1 || t->format == 3 ||          /* AI88,1555 */
                    t->format == 4 || t->format == 5 ||          /* 565,4444 */
                    t->format == 10 || t->format == 11) ? 2 : 4; /* YUV 4:2:2 */
    if (format & (1 << 7)) {              /* NON_POWER2: explicit size/pitch */
        t->width  = (size & 0x7FF) + 1;
        t->height = ((size >> 16) & 0x7FF) + 1;
        t->pitch  = (pitch & 0x3FE0) + 32;
        t->denorm = 1;                    /* rectangle textures use texels */
    } else {
        t->width  = 1u << ((format >> 8) & 0xF);
        t->height = 1u << ((format >> 12) & 0xF);
        t->pitch  = t->width * bpp;
    }
    /* The swap only matters where byte order within a texel group does:
      * YUV 4:2:2 (the backend applies it).  VRAM already holds the CPU's
      * byte order, which is what 16/32-bit texel decoding expects. */
    t->swap = offset & 3;
    static int tex_trace = -1;                   /* getenv is an environ scan */
    if (tex_trace < 0) {
        tex_trace = getenv("POWEREMU_TEX_TRACE") != NULL;
    }
    if (tex_trace) {                             /* every texture format, once */
        static uint32_t seen_tex;
        if (t->format < 32 && !(seen_tex & (1u << t->format))) {
            seen_tex |= 1u << t->format;
            fprintf(stderr, "ppc-mac-gpu tex: format %u (0x%08x) %ux%u offset=0x%08x\n",
                    t->format, format, t->width, t->height, offset);
        }
    }
    /* POWEREMU_YUV_TRACE: what a YUV texture really holds, to tell the
      * 4:2:2 orderings apart (Tiger's welcome movie vs Halo's logos). */
    static int yuv_trace = -1;
    if (yuv_trace < 0) {
        yuv_trace = getenv("POWEREMU_YUV_TRACE") != NULL;
    }
    if ((t->format == 10 || t->format == 11) && yuv_trace) {
        static int yuv_raw_logged;
        if (yuv_raw_logged++ < 12) {
            uint32_t o = offset & ~0x1Fu;
            const uint8_t *b = (o >= fb_base && o - fb_base < s->vram_size)
                               ? (const uint8_t *)memory_region_get_ram_ptr(&s->vram) + (o - fb_base) : NULL;
            fprintf(stderr, "ppc-mac-gpu yuv raw: unit %d format=0x%08x fmt=%d swap=%d offset=0x%08x "
                     "filter=0x%08x in_vram=%d size=%dx%d bytes=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                     n, format, t->format, t->swap, offset, filter, b != NULL, t->width, t->height,
                     b ? b[0] : 0, b ? b[1] : 0, b ? b[2] : 0, b ? b[3] : 0,
                     b ? b[4] : 0, b ? b[5] : 0, b ? b[6] : 0, b ? b[7] : 0);
        }
    }
    offset &= ~0x1Fu;
    /*
     * The AGP aperture can start inside the VRAM the card has but does not
     * report (the top megabytes are the command processor's): Jaguar's
     * driver puts it at 124 MB, and texture offsets there are AGP.
     */
    uint32_t agp_loc = s->regs.mc_agp_location;
    bool in_agp = agp_loc != 0 &&
                  offset >= ((agp_loc & 0xFFFF) << 16) &&
                  offset <= ((((agp_loc >> 16) & 0xFFFF) << 16) | 0xFFFF);
    if (in_agp || offset < fb_base || offset - fb_base >= s->vram_size) {
        /*
         * AGP/GART texture (client storage / texture range): copy the texels
         * out of guest memory.  Level 0 only; the caller frees host_data.
         */
        uint64_t len = (uint64_t)t->pitch * t->height;
        if (t->format == 12 || t->format == 14 || t->format == 15) {
            len = (uint64_t)t->pitch * ((t->height + 3) / 4);
        }
        if (len == 0 || len > 16 * 1024 * 1024) {
            t->enabled = 0;
            return;
        }
        uint8_t *buf = g_malloc(len);
        for (uint64_t done = 0; done < len;) {
            hwaddr phys;
            uint32_t a = offset + done;
            uint32_t chunk = MIN(0x1000 - (a & 0xFFF), len - done);
            uint8_t *hp = r200_agp_page(s, a);
            if (hp) {
                memcpy(buf + done, hp + (a & 0xFFF), chunk);
                done += chunk;
                continue;
            }
            if ((!ppc_mac_gpu_agp_translate(s, a, &phys) &&
                 !ppc_mac_gpu_gart_translate(s, a, &phys)) ||
                address_space_read(&address_space_memory, phys,
                                   MEMTXATTRS_UNSPECIFIED, buf + done,
                                   chunk) != MEMTX_OK) {
                r200_warn_once(&r200_warned, 2, "AGP texture at 0x%08x not "
                               "readable", offset);
                g_free(buf);
                t->enabled = 0;
                return;
            }
            done += chunk;
        }
        static int agp_logged;
        if (agp_logged++ < 3) {
            qemu_log("ppc-mac-gpu r200: AGP texture %ux%u fmt %u at 0x%08x\n",
                     t->width, t->height, t->format, offset);
        }
        /* POWEREMU_YUV_TRACE: the frame's own bytes, from the middle row,
          * which say which 4:2:2 order a program really uses. */
        if ((t->format == 10 || t->format == 11) && yuv_trace) {
            /* Log each new kind of frame, not just the first ones: a game
             * plays several videos and only some come out wrong. */
            static uint64_t last_kind;
            static int agp_yuv_logged;
            uint64_t kind = ((uint64_t)t->format << 40) | ((uint64_t)t->swap << 36)
                          | ((uint64_t)t->width << 20) | ((uint64_t)t->height << 4);
            if (kind != last_kind && agp_yuv_logged++ < 40) {
                last_kind = kind;
                uint64_t mid = (uint64_t)t->pitch * (t->height / 2);
                const uint8_t *m = (mid + 16 <= len) ? buf + mid : buf;
                fprintf(stderr, "ppc-mac-gpu yuv agp: fmt %u swap %u %ux%u pitch %u "
                        "mid=%02x %02x %02x %02x %02x %02x %02x %02x  "
                        "first=%02x %02x %02x %02x\n",
                        t->format, t->swap, t->width, t->height, t->pitch,
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                        buf[0], buf[1], buf[2], buf[3]);
            }
        }
        t->host_data = buf;
        t->offset = 0;
        return;
    }
    t->offset = offset - fb_base;
}

/* TCL constant memory accessors (vector address, component). */
static inline float r200_vf(PPCMacGPUState *s, uint32_t addr, int c)
{
    return r200_f32(s->regs.tcl_vec[addr & 0x7ff][c]);
}

/* 4x4 matrix N lives at vector address 0x80 + 4N, one row per vector. */
static void r200_mat(PPCMacGPUState *s, uint32_t n, float m[4][4])
{
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            m[r][c] = r200_vf(s, 0x80 + 4 * (n & 0x1f) + r, c);
        }
    }
}

static inline void r200_xform(const float m[4][4], const float v[4], float o[4])
{
    for (int r = 0; r < 4; r++) {
        o[r] = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3] * v[3];
    }
}

static inline void r200_norm3(float v[3])
{
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-20f) {
        v[0] /= l; v[1] /= l; v[2] /= l;
    }
}

/*
 * R200 TCL fixed-function lighting for one vertex (GL semantics).
 * p_eye/n_eye are in lighting space; col0/col1 are the vertex colours.
 * Writes the lit primary colour to out0 and secondary (specular) to out1.
 */
static void r200_light_vertex(PPCMacGPUState *s, const float p_eye[4],
                              const float n_in[3], const float col0[4],
                              const float col1[4], float out0[4], float out1[4])
{
    uint32_t lm0 = R3D(0x2268), lm1 = R3D(0x226C);
    float n[3] = { n_in[0], n_in[1], n_in[2] };
    r200_norm3(n);

    /* material factor sources: 0 light-premultiplied, 1 material 0,
     * 2.. vertex colour k-2, 0xf material 1 */
    const float one[4] = { 1, 1, 1, 1 };
    float fac[4][4];                      /* emissive, ambient, diffuse, specular */
    for (int k = 0; k < 4; k++) {
        uint32_t src = (lm1 >> (4 * k)) & 0xF;
        uint32_t mat_base = src == 0xF ? 0xB4 : 0xB0;
        if (src == 0) {
            memcpy(fac[k], k == 0 ? (float[4]){ 0, 0, 0, 0 } : one, sizeof(one));
            if (k == 0) {                 /* emissive has no premultiplied form */
                for (int c = 0; c < 4; c++) fac[k][c] = r200_vf(s, 0xB0, c);
            }
        } else if (src == 1 || src == 0xF) {
            for (int c = 0; c < 4; c++) fac[k][c] = r200_vf(s, mat_base + k, c);
        } else {
            const float *vc = src == 2 ? col0 : col1;
            memcpy(fac[k], vc, sizeof(float) * 4);
        }
    }

    float acc[3], spec[3] = { 0, 0, 0 };
    for (int c = 0; c < 3; c++) {
        acc[c] = fac[0][c] + r200_vf(s, 0x5C, c) * fac[1][c];   /* emissive + scene ambient */
    }
    float v[3] = { 0, 0, 1 };
    if (lm0 & (1u << 2)) {                                    /* LOCAL_VIEWER */
        v[0] = -p_eye[0]; v[1] = -p_eye[1]; v[2] = -p_eye[2];
        r200_norm3(v);
    }
    float shin = r200_f32(s->regs.tcl_scalar[0x100]);

    for (int i = 0; i < 8; i++) {
        uint32_t ctl = (R3D(0x2270 + 4 * (i / 2)) >> (16 * (i & 1))) & 0xFFFF;
        if (!(ctl & 1)) {
            continue;
        }
        float l[3], att = 1.0f;
        for (int c = 0; c < 3; c++) {
            l[c] = r200_vf(s, 0x40 + i, c);
        }
        if (ctl & (1u << 3)) {                                /* local light */
            for (int c = 0; c < 3; c++) l[c] -= p_eye[c];
            float d = sqrtf(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
            r200_norm3(l);
            /* ATTENUATION (0x50+i) is (quadratic, linear, constant, -),
             * the order Mesa's r200 driver uses. */
            float aq = r200_vf(s, 0x50 + i, 0), al = r200_vf(s, 0x50 + i, 1),
                  ac = r200_vf(s, 0x50 + i, 2);
            float den = ac + al * d + aq * d * d;
            if (den > 1e-20f) att = 1.0f / den;
            if (ctl & (1u << 4)) {                            /* spot light */
                /* HWVSPOT (0x48+i) holds the direction back towards the
                 * light, and scalar memory the exponent (0x10+i) and
                 * cos(cutoff) (0x18+i): the hardware lights a vertex when
                 * dot(L, dir) >= cutoff.  Signs confirmed against Apple's
                 * driver (Chess selection spot: 0.9962, exponent 100). */
                float sd[3] = { r200_vf(s, 0x48 + i, 0), r200_vf(s, 0x48 + i, 1),
                                r200_vf(s, 0x48 + i, 2) };
                r200_norm3(sd);
                float sexp = r200_f32(s->regs.tcl_scalar[0x10 + i]);
                float scut = r200_f32(s->regs.tcl_scalar[0x18 + i]);
                float cosang = l[0] * sd[0] + l[1] * sd[1] + l[2] * sd[2];
                static int spot_logged;
                static int spotlog = -1;
                if (spotlog < 0) {
                    spotlog = getenv("PPCGPU_SPOTLOG") != NULL;
                }
                if (spotlog && spot_logged < 8) {
                    spot_logged++;
                    qemu_log("r200 spot: light %d ctl=%04x dir=(%.3f,%.3f,%.3f) "
                             "exp=%.3f coscut=%.4f cosang=%.4f d=%.2f "
                             "atten=(%g,%g,%g,%g) att=%g\n", i, ctl,
                             sd[0], sd[1], sd[2], sexp, scut, cosang, d,
                             r200_vf(s, 0x50 + i, 0), r200_vf(s, 0x50 + i, 1),
                             r200_vf(s, 0x50 + i, 2), r200_vf(s, 0x50 + i, 3),
                             att);
                }
                if (cosang < scut) {
                    continue;                         /* outside the cone */
                }
                if (sexp > 0.0f) {
                    att *= powf(MAX(cosang, 0.0f), sexp);
                }
            }
        } else {
            r200_norm3(l);
        }
        float ndl = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
        for (int c = 0; c < 3; c++) {
            float t = 0;
            if (ctl & (1u << 1)) {
                t += r200_vf(s, 0x28 + i, c) * fac[1][c];      /* ambient */
            }
            if (ndl > 0) {
                t += r200_vf(s, 0x30 + i, c) * fac[2][c] * ndl;
            }
            acc[c] += att * t;
        }
        if ((ctl & (1u << 2)) && (lm0 & (1u << 5)) && ndl > 0) {  /* specular */
            float h[3] = { l[0] + v[0], l[1] + v[1], l[2] + v[2] };
            r200_norm3(h);
            float ndh = n[0] * h[0] + n[1] * h[1] + n[2] * h[2];
            if (ndh > 0) {
                float sp = powf(ndh, shin);
                for (int c = 0; c < 3; c++) {
                    spec[c] += att * r200_vf(s, 0x38 + i, c) * fac[3][c] * sp;
                }
            }
        }
    }
    bool combine = lm0 & (1u << 6);                           /* DIFFUSE_SPECULAR_COMBINE */
    for (int c = 0; c < 3; c++) {
        out0[c] = MIN(MAX(acc[c] + (combine ? spec[c] : 0), 0.0f), 1.0f);
        out1[c] = combine ? 0.0f : MIN(MAX(spec[c], 0.0f), 1.0f);
    }
    out0[3] = MIN(MAX(fac[2][3], 0.0f), 1.0f);                /* diffuse alpha */
    out1[3] = 0.0f;
}


/* Fill the current 3D depth buffer with RB3D_DEPTHCLEARVALUE (see the
 * 3D_CLEAR_ZMASK handler). */
static void r200_clear_depth_buffer(PPCMacGPUState *s)
{
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t base = R3D(0x1C24) & ~0xFu;
    uint32_t pitch_px = R3D(0x1C28) & 0x1FF8;
    uint32_t zfmt = R3D(0x1C2C) & 0xF;
    uint32_t bpp = zfmt == 0 ? 2 : 4;
    uint32_t h = ((R3D(0x1C44) >> 16) & 0x7FF) + 1;
    /* Apple's driver never programs RB3D_DEPTHCLEARVALUE; it resets to all
     * ones (reset handler), i.e. far depth - what cleared tiles must read as
     * for Warcraft III's LEQUAL test to pass on real hardware.  Keep the
     * stencil byte clear for Z24S8. */
    uint32_t value = R3D(0x3230);
    if (value == 0xFFFFFFFFu && zfmt != 0) {
        value = 0x00FFFFFFu;
    }
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

    if (base < fb_base || pitch_px == 0) {
        return;
    }
    uint64_t off = base - fb_base, pitch = (uint64_t)pitch_px * bpp;
    if (off + pitch * h > s->vram_size) {
        h = (s->vram_size - off) / pitch;
    }
    r200_vram_access(s, R200_ROWS(off, pitch, 0, h), true, 5);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *row = vram + off + y * pitch;
        for (uint32_t x = 0; x < pitch_px; x++) {
            if (bpp == 2) {
                stw_be_p(row + 2 * x, value);
            } else {
                stl_be_p(row + 4 * x, value);
            }
        }
    }
}

/* Vertex sources for ppc_mac_gpu_r200_draw(). */
enum { R200_SRC_VBUF, R200_SRC_INDX, R200_SRC_IMMD };

/*
 * Handle one R200 draw packet: 3D_DRAW_VBUF_2 (vertices from the AOS
 * arrays, in order), 3D_DRAW_INDX_2 (AOS arrays, 16-bit indices inline
 * after vf_cntl) or 3D_DRAW_IMMD_2 (vertices inline after vf_cntl, packed
 * per SE_VTX_FMT_0/1).  d[0] is vf_cntl; ndw counts d[].  Returns true if
 * handled — including draws deliberately skipped — so the legacy path never
 * runs in direct mode.
 */
/*
 * The draw itself. Runs on the vCPU thread, or on the draw worker against a
 * snapshot -- it cannot tell which, and must not care: everything it reads
 * comes through `s`, which is the snapshot when a worker is running it.
 */
static bool ppc_mac_gpu_r200_draw_now(PPCMacGPUState *s, const uint32_t *d,
                                  uint32_t ndw, int src)
{
    if (!r200_direct_enabled() || !s->renderer || !s->renderer->draw_r200) {
        return false;
    }

    uint32_t vf_cntl = d[0];
    uint32_t prim = vf_cntl & 0xF;
    uint32_t nverts = vf_cntl >> 16;
    if (nverts == 0) {
        return true;
    }
    if (nverts > 16384) {
        r200_warn_once(&r200_warned, 8, "draw with %u vertices skipped", nverts);
        return true;
    }

    /* ---- Vertex attribute layout (SE_VTX_FMT_0/1) ---- */
    uint32_t fmt0 = R3D(0x2088), fmt1 = R3D(0x208C);
    enum { A_POS, A_NORMAL, A_COLOR0, A_COLOR1, A_SKIP, A_TEX0 };
    int attr_kind[20], attr_extra[20], attr_comps[20], nattr = 0;
    /*
     * The chip's own slot number for each attribute (0 position, 1 blend
     * weights, 2 normal, 3 fog, 4-7 colours, 8-13 texture coordinates).
     * A vertex program reads its inputs by slot, including the ones the
     * fixed-function path throws away, so every attribute is kept.
     */
    int attr_slot[20];
#define ADD_ATTR(k, e, c, sl) do { attr_kind[nattr] = (k); attr_extra[nattr] = (e); \
                                   attr_slot[nattr] = (sl); \
                                   attr_comps[nattr++] = (c); } while (0)
    ADD_ATTR(A_POS, 0, 2 + (fmt0 & 1) + ((fmt0 >> 1) & 1), 0);
    if ((fmt0 >> 2) & 7) {
        ADD_ATTR(A_SKIP, 0, (fmt0 >> 2) & 7, 1);       /* blend weights */
    }
    if (fmt0 & (1u << 6)) {
        ADD_ATTR(A_NORMAL, 0, 3, 2);
    }
    if (fmt0 & (1u << 7)) {
        ADD_ATTR(A_SKIP, 0, 1, 15);                    /* point size */
    }
    if (fmt0 & (1u << 8)) {
        /* Fog, one float, in the stream between the normal and the
         * colours. It was missing here, and a vertex that carries it
         * shifted every attribute after it by one word. */
        ADD_ATTR(A_SKIP, 0, 1, 3);
    }
    /*
     * Colours.  The first two are the primary and the secondary; the rest
     * are consumed but unused, while still being handed to a vertex
     * program by their own slot number.
     */
    int colours_seen = 0;
    for (int c = 0; c < 8; c++) {
        uint32_t cf = (fmt0 >> (11 + 2 * c)) & 3;
        if (cf) {
            int kind = colours_seen == 0 ? A_COLOR0 : colours_seen == 1 ? A_COLOR1 : A_SKIP;
            colours_seen++;
            ADD_ATTR(kind, cf, cf == 1 ? 1 : cf == 2 ? 3 : 4, 4 + c);
        }
    }
    for (int t = 0; t < R200_MAX_TEX; t++) {
        uint32_t n = (fmt1 >> (3 * t)) & 7;
        if (n) {
            ADD_ATTR(A_TEX0 + t, 0, n, 8 + t);
        }
    }
#undef ADD_ATTR
    uint32_t vtx_dw = 0;
    for (int a = 0; a < nattr; a++) {
        vtx_dw += attr_comps[a];
    }

    /* ---- Where each attribute comes from ---- */
    uint32_t arr_addr[16] = { 0 }, arr_stride[16] = { 0 }, arr_count[16] = { 0 };
    uint32_t narrays = 0;
    const uint32_t *inl = d + 1;          /* IMMD vertices / INDX indices */
    uint32_t inl_dw = ndw - 1;
    if (src == R200_SRC_IMMD) {
        static int immd_dumps;
        if (immd_dumps++ < 6) {
            GString *g = g_string_new(NULL);
            for (uint32_t i = 0; i < ndw && i < 40; i++) {
                g_string_append_printf(g, " %08x", d[i]);
            }
            qemu_log("r200 IMMD_2 fmt0=%08x fmt1=%08x vte=%08x vap=%08x vtx_dw=%u "
                     "raw:%s\n", fmt0, fmt1, R3D(0x20B0), R3D(0x2080), vtx_dw, g->str);
            g_string_free(g, TRUE);
        }
        if (inl_dw < nverts * vtx_dw) {
            r200_warn_once(&r200_warned, 32, "IMMD_2 short: %u dwords for %u "
                           "vertices of %u", inl_dw, nverts, vtx_dw);
            return true;
        }
    } else {
        /*
         * AOS arrays: (count | stride<<8) halves at 0x20C4 + 12*k.  The
         * arrays are one stream: array k supplies `count` dwords per vertex,
         * in order, and SE_VTX_FMT consumes the stream attribute by
         * attribute - so one interleaved array can carry position, colour
         * and texcoords (Warcraft III's text does).  Mapping array k to
         * attribute k dropped every attribute past the arrays' number.
         */
        narrays = MIN(R3D(0x20C0) & 0x1F, 16u);
        uint32_t stream_dw = 0;
        for (uint32_t a = 0; a < narrays; a++) {
            uint32_t attr = R3D(0x20C4 + (a / 2) * 12);
            uint32_t half = (a & 1) ? attr >> 16 : attr & 0xFFFF;
            arr_count[a] = half & 0xFF;
            arr_stride[a] = (half >> 8) & 0xFF;
            arr_addr[a] = R3D(0x20C8 + (a / 2) * 12 + (a & 1) * 4);
            stream_dw += arr_count[a];
        }
        if (stream_dw != vtx_dw) {
            r200_warn_once(&r200_warned, 32, "AOS arrays give %u dwords per "
                           "vertex, format wants %u (fmt0=0x%x fmt1=0x%x)",
                           stream_dw, vtx_dw, fmt0, fmt1);
        }
        if (src == R200_SRC_INDX && inl_dw < (nverts + 1) / 2) {
            r200_warn_once(&r200_warned, 32, "INDX_2 short: %u index dwords "
                           "for %u indices", inl_dw, nverts);
            return true;
        }
    }

    /* ---- Transform state ---- */
    uint32_t vap = R3D(0x2080);                     /* SE_VAP_CNTL */
    bool tcl = vap & 1;
    /*
     * Bit 2 puts the chip in vertex-program mode: each vertex is
     * transformed by a program the guest uploaded, not by the matrices and
     * lights below -- which share the same memory, so they hold program
     * instructions now and must not be read.  The programs themselves are
     * not run yet, so such a draw is skipped rather than drawn wrongly.
     */
    bool vertex_program = (vap & 4) != 0;
    if (vertex_program) {
        static uint32_t last_cntl[2];
        uint32_t c1 = R3D(0x22D0), c2 = R3D(0x22D4);
        if ((c1 != last_cntl[0] || c2 != last_cntl[1]) && getenv("POWEREMU_VP_TRACE")) {
            uint32_t first = c1 & 0x3FF, last = (c1 >> 20) & 0x3FF;
            last_cntl[0] = c1; last_cntl[1] = c2;
            fprintf(stderr, "ppc-mac-gpu vp inputs: fmt0=0x%08x fmt1=0x%08x arrays=%u "
                    "route=%08x %08x %08x %08x vte=0x%08x\n",
                    R3D(0x2088), R3D(0x208C), narrays,
                    R3D(0x2254), R3D(0x2258), R3D(0x225C), R3D(0x2260), R3D(0x20B0));
            for (int a = 0; a < nattr; a++) {
                fprintf(stderr, "    attribute %d: slot %2d, %d words\n",
                        a, attr_slot[a], attr_comps[a]);
            }
            fprintf(stderr, "ppc-mac-gpu vp program: vap=0x%08x instructions %u..%u "
                    "(position settled at %u), constants from %u, up to %u\n", vap,
                    first, last, (c1 >> 10) & 0x3FF, c2 & 0xFF, (c2 >> 16) & 0xFF);
            for (uint32_t i = first; i <= last && i < 128; i++) {
                const uint32_t *w = s->regs.tcl_vec[ppc_mac_gpu_vp_inst_addr(i)];
                PPCMacGPUVPInst in;
                char text[128];
                ppc_mac_gpu_vp_decode(w, &in);
                ppc_mac_gpu_vp_disasm(&in, text, sizeof(text));
                fprintf(stderr, "  %3u: %08x %08x %08x %08x  %s\n",
                        i, w[0], w[1], w[2], w[3], text);
            }
        }
    }
    uint32_t mvp_n = R3D(0x2238) & 0x1F;            /* MODELPROJECT_0 */
    float mvp[4][4];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            mvp[r][c] = vertex_program ? (r == c ? 1.0f : 0.0f)
                : r200_f32(s->regs.tcl_vec[(0x80 + 4 * mvp_n + r) & 0x7ff][c]);
        }
    }
    bool lighting = tcl && !vertex_program && (R3D(0x2268) & 1);
    uint32_t tcl_fog = tcl ? (R3D(0x22C0) >> 8) & 3 : 0;
    uint32_t texproc0 = tcl ? R3D(0x22B0) : 0, texproc1 = R3D(0x22B4);
    float mv[4][4], itmv[4][4], texmat[R200_MAX_TEX][4][4];
    r200_mat(s, R3D(0x2230) & 0x1F, mv);                /* MODELVIEW_0 */
    r200_mat(s, R3D(0x2234) & 0x1F, itmv);              /* IT_MODELVIEW_0 */
    for (int t = 0; t < R200_MAX_TEX; t++) {
        uint32_t sel = t < 4 ? R3D(0x223C) >> (8 * t) : R3D(0x2240) >> (8 * (t - 4));
        r200_mat(s, sel & 0x1F, texmat[t]);
    }
    if (lighting && (R3D(0x2270) | R3D(0x2274) | R3D(0x2278) | R3D(0x227C)) &
        0x00300030u) {
        r200_warn_once(&r200_warned, 128, "spot lights not modelled");
    }
    uint32_t vte = R3D(0x20B0);
    float vp[6];
    for (int i = 0; i < 6; i++) {
        vp[i] = r200_f32(R3D(0x1D98 + 4 * i));
    }
    float pix_bias = (R3D(0x1C4C) & (1u << 27)) ? 0.0f : 0.5f;  /* D3D centres */

    /*
     * Resolve each vertex array to a host pointer once for the whole draw.
     *
     * r200_read_dwords() was called per vertex per array, and each call
     * re-derived the framebuffer base from a register, bounds-checked, then
     * copied into a stack buffer only to byte-swap out of it again -- two
     * passes over every vertex. Between them the fetch and that copying were
     * most of a 13.8% cluster in the profile.
     *
     * Arrays that live in VRAM, which is nearly all of them, are read
     * straight from the host pointer with the swap done in place. Anything
     * else -- AGP, GART, an array that runs off the end -- still goes through
     * r200_read_dwords, which stays the one place that knows how to find
     * memory that is not simply there.
     */
    uint8_t *const vram_host = memory_region_get_ram_ptr(&s->vram);
    const uint32_t arr_fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t arr_off[16];
    bool arr_fast[16];
    for (uint32_t a = 0; a < narrays && a < 16; a++) {
        arr_fast[a] = vram_host && arr_addr[a] >= arr_fb_base;
        arr_off[a] = arr_fast[a] ? arr_addr[a] - arr_fb_base : 0;
    }

    /*
     * Where each attribute starts in the gathered stream, when there is one
     * array per attribute. This is a property of the draw, not of the
     * vertex, but it used to be recomputed inside the vertex loop with an
     * inner loop over the preceding attributes -- O(attributes squared) per
     * vertex, for an answer that never changed.
     */
    /*
     * An attribute the vertex declares but does not stream must not eat an
     * array.
     *
     * With one array per attribute, the arrays line up with the attributes
     * in order.  Mac OS X 10.5's compositor declares fog in SE_VTX_FMT_0 --
     * a single float between the position and the colours -- but supplies
     * only two arrays, position and colour: the fog comes from a register,
     * not from the vertex.  Lining those up naively gave the fog the
     * colour's array and left the colour with none, so every vertex kept
     * whatever colour the previous draw had left.  For the Dock, whose
     * colour comes straight from the vertex, that was a red slab.
     *
     * Only the attributes that contribute nothing to the result are
     * dropped, last first, and only enough of them to make the counts
     * agree -- so a vertex that really does stream its fog is untouched.
     */
    if (src != R200_SRC_IMMD && narrays > 0 && narrays < (uint32_t)nattr) {
        for (int a = nattr - 1; a >= 0 && narrays < (uint32_t)nattr; a--) {
            if (attr_kind[a] != A_SKIP) {
                continue;
            }
            for (int k = a; k < nattr - 1; k++) {
                attr_kind[k] = attr_kind[k + 1];
                attr_extra[k] = attr_extra[k + 1];
                attr_comps[k] = attr_comps[k + 1];
                attr_slot[k] = attr_slot[k + 1];
            }
            nattr--;
        }
    }

    /*
     * The arrays say how wide each attribute really is.
     *
     * SE_VTX_FMT_0 says what kind of thing each attribute is; the array's
     * own count says how many dwords of it the vertex carries.  They can
     * disagree, and when they do the array is right -- it is what was
     * actually written.  Mac OS X 10.5's compositor declares its colour as
     * four floats and then streams it as one dword of packed bytes; read
     * as four floats that gave an impossible colour and swallowed the two
     * texture coordinates that followed it, which is why the Dock arrived
     * as a red slab with 0xffffffff and 0xcccccccc -- opaque white and
     * translucent grey, perfectly ordinary colours -- in the red channel.
     *
     * Only narrowing is honoured, and only per attribute: an array that
     * carries more than the format asks for is left alone, since the extra
     * words belong to whatever comes next.
     */
    if (narrays == (uint32_t)nattr) {
        for (int a = 0; a < nattr; a++) {
            uint32_t have = arr_count[a];
            if (!have || (int)have >= attr_comps[a]) {
                continue;
            }
            if ((attr_kind[a] == A_COLOR0 || attr_kind[a] == A_COLOR1 ||
                 attr_kind[a] == A_SKIP) && have == 1) {
                attr_extra[a] = 1;            /* packed, 0xAARRGGBB */
            }
            attr_comps[a] = have;
        }
        vtx_dw = 0;
        for (int a = 0; a < nattr; a++) {
            vtx_dw += attr_comps[a];
        }
    }

    uint32_t attr_start[20];          /* matches attr_comps[] above */
    if (narrays == (uint32_t)nattr) {
        /*
         * Only in this case, and only over the arrays that exist: arr_count
         * is filled per array, so summing it across all attributes would
         * read past what was initialised whenever the two counts differ.
         * The original inner loop was safe for the same reason -- it ran
         * only on this branch -- and hoisting it has to keep that.
         */
        uint32_t acc = 0;
        for (int a = 0; a < nattr; a++) {
            attr_start[a] = acc;
            acc += arr_count[a];
        }
    }

    /*
     * A vertex program, if one is in charge: decoded once for the draw,
     * with the constants it reads.  Which input register sees which
     * attribute slot is fixed in the hardware.
     */
    /* Decoded once and kept: the same program runs for draw after draw. */
    static PPCMacGPUVPInst vp_prog[PPC_MAC_GPU_VP_MAX_INST];
    static uint32_t vp_decoded_for[2];
    static uint32_t vp_decoded_head[4];
    static float vp_consts[PPC_MAC_GPU_VP_CONSTS][4];
    uint32_t vp_count = 0;
    /*
     * Which program input each vertex slot feeds.  Fixed in the hardware
     * for the named attributes; the colour slots carry a program's own
     * attributes (Halo keeps bone indices and weights there) and which
     * input each lands in is the driver's business, not the format's.
     * PPCGPU_VP_MAP picks between the orders while that is settled:
     * 0 (default) 2,3,4,5 -- what the Linux driver documents
     * 1           4,5,2,3      2  5,4,2,3      3  3,2,5,4
     */
    int vp_slot_to_input[16] = {
        /* 0 pos */ 0, /* 1 weights */ 12, /* 2 normal */ 1, /* 3 fog */ 15,
        /* 4-7 colours */ 2, 3, 4, 5,
        /* 8-13 texture coordinates */ 6, 7, 8, 9, 10, 11,
        /* 14 second position */ 13, /* 15 point size */ 14,
    };
    /*
     * The colour slots carry a program's own attributes, and which input
     * each lands in depends on how many the draw declares -- the driver
     * does not use one fixed arrangement.  Measured in Halo: the last two
     * colour slots always feed inputs 2 and 3 (its bone indices and
     * weights), and any earlier slots feed 5 and 4, highest first.  With
     * two slots that is the plain 2, 3 its interface draws want.
     * PPCGPU_VP_MAP=0 goes back to the plain order throughout.
     */
    {
        int colour_slot[4], ncolour = 0;
        for (int c = 0; c < 4; c++) {
            if ((fmt0 >> (11 + 2 * c)) & 3) {
                colour_slot[ncolour++] = 4 + c;
            }
        }
        const char *e = getenv("PPCGPU_VP_MAP");
        /* 0 plain, 1 highest slot first, 2 lowest first -- 2 is what makes
         * Halo's sky come out right, and is the default. */
        int mode = e ? atoi(e) : 2;
        if (mode && ncolour >= 2) {
            for (int i = 0; i < ncolour - 2; i++) {
                /* The slots left over after the last two carry the normal
                 * and the texture coordinates; which is which is the one
                 * thing the draw doesn't say, so it can be swapped here. */
                vp_slot_to_input[colour_slot[i]] = mode == 1 ? 5 - i : 4 + i;
            }
            vp_slot_to_input[colour_slot[ncolour - 2]] = 2;
            vp_slot_to_input[colour_slot[ncolour - 1]] = 3;
        }
    }
    if (vertex_program && !gpu_vp_disabled()) {
        uint32_t c1 = R3D(0x22D0), c2 = R3D(0x22D4);
        uint32_t first = c1 & 0x3FF, last = (c1 >> 20) & 0x3FF;
        uint32_t cbase = c2 & 0xFF;
        if (last >= first && last < PPC_MAC_GPU_VP_MAX_INST) {
            const uint32_t *head = s->regs.tcl_vec[ppc_mac_gpu_vp_inst_addr(first)];
            vp_count = last - first + 1;
            if (vp_decoded_for[0] != c1 || vp_decoded_for[1] != first ||
                memcmp(vp_decoded_head, head, sizeof(vp_decoded_head)) != 0) {
                for (uint32_t i = 0; i < vp_count; i++) {
                    ppc_mac_gpu_vp_decode(s->regs.tcl_vec[ppc_mac_gpu_vp_inst_addr(first + i)],
                                          &vp_prog[i]);
                }
                vp_decoded_for[0] = c1;
                vp_decoded_for[1] = first;
                memcpy(vp_decoded_head, head, sizeof(vp_decoded_head));
            }
        }
        for (uint32_t i = 0; i < PPC_MAC_GPU_VP_CONSTS; i++) {
            const uint32_t *v = s->regs.tcl_vec[ppc_mac_gpu_vp_const_addr(cbase + i) & 0x7ff];
            for (int c = 0; c < 4; c++) {
                vp_consts[i][c] = r200_f32(v[c]);
            }
        }
    }

    s->regs.stall_draws++;

    /* ---- Fetch and transform vertices ---- */
    R200Vertex *verts = r200_draw_verts(s, nverts);
    memset(verts, 0, sizeof(R200Vertex) * nverts);
    for (uint32_t v = 0; v < nverts; v++) {
        R200Vertex *o = &verts[v];
        float pos[4] = { 0, 0, 0, 1 };
        float nrm[3] = { 0, 0, 1 };
        /* What a vertex program would read: every attribute, by slot. */
        float slot[16][4];
        if (vp_count) {
            for (int i = 0; i < 16; i++) {
                slot[i][0] = slot[i][1] = slot[i][2] = 0.0f;
                slot[i][3] = 1.0f;
            }
        }
        o->color[0] = o->color[1] = o->color[2] = o->color[3] = 1.0f;
        for (int t = 0; t < R200_MAX_TEX; t++) {
            o->tex[t][3] = 1.0f;
        }
        /* element index: sequential, or from the inline 16-bit index list */
        uint32_t e = v;
        if (src == R200_SRC_INDX) {
            uint32_t w = inl[v / 2];
            e = (v & 1) ? w >> 16 : w & 0xFFFF;
        }
        const uint32_t *iv = inl + (size_t)v * vtx_dw;   /* IMMD cursor */
        uint32_t stream[64], sn = 0;
        if (src != R200_SRC_IMMD) {
            /* gather this vertex's dwords from every array, in order */
            for (uint32_t a = 0; a < narrays; a++) {
                uint32_t n = MIN(arr_count[a], 64u - sn);
                uint32_t off;

                if (!n) {
                    continue;
                }
                /*
                 * The bounds test is per access rather than per draw because
                 * an indexed draw picks its element from the index list and
                 * the range it will touch is not known in advance.
                 */
                off = arr_off[a] + e * arr_stride[a] * 4;
                if (a < 16 && arr_fast[a] &&
                    (uint64_t)off + (uint64_t)n * 4 <= s->vram_size) {
                    const uint8_t *p = vram_host + off;
                    for (uint32_t i = 0; i < n; i++) {
                        stream[sn + i] = ldl_be_p(p + i * 4);
                    }
                    /*
                     * PPCGPU_VERIFY_FETCH=1 runs the old path as well and
                     * reports any disagreement. Wrong vertex data does not
                     * crash; it draws something slightly wrong, which is far
                     * harder to attribute later than a loud complaint now.
                     */
                    if (unlikely(gpu_verify_fetch())) {
                        uint32_t ref[64];
                        if (r200_read_dwords(s, arr_addr[a] +
                                             e * arr_stride[a] * 4, ref, n) &&
                            memcmp(ref, stream + sn, n * 4) != 0) {
                            r200_warn_once(&r200_warned, 512,
                                           "fast vertex fetch differs at "
                                           "0x%08x", arr_addr[a]);
                        }
                    }
                } else if (!r200_read_dwords(s,
                                             arr_addr[a] + e * arr_stride[a] * 4,
                                             stream + sn, n)) {
                    r200_warn_once(&r200_warned, 256,
                                   "vertex fetch failed at 0x%08x", arr_addr[a]);
                    return true;
                }
                sn += n;
            }
            iv = stream;
        }
        for (int a = 0; a < nattr; a++) {
            /*
             * Deliberately not zero-initialised: that zeroed all 64 bytes
             * for every attribute of every vertex, which is the hottest
             * memset in the device model. Only what can be read is cleared.
             *
             * "What can be read" is not simply [0, n): the packed-colour
             * path below reads raw[0] unconditionally, so the first four
             * words must be defined even when n is smaller. Getting that
             * wrong gives colours built from stack garbage.
             */
            uint32_t raw[16];
            const uint32_t raw_used = 4;       /* raw[0..3]: see above */
            uint32_t n = MIN((uint32_t)attr_comps[a], 16u);
            if (src == R200_SRC_IMMD) {
                r200_fetch_words(raw, iv, n, MAX(n, raw_used));
                iv += n;
            } else {
                uint32_t have;
                if (narrays == (uint32_t)nattr) {
                    /* one array per attribute: the array's count wins */
                    iv = stream + MIN(attr_start[a], sn);
                    have = MIN(n, arr_count[a]);
                } else {
                    uint32_t used = iv - stream;
                    have = used < sn ? MIN(n, sn - used) : 0;
                }
                r200_fetch_words(raw, iv, have, MAX(n, raw_used));
                if (have < n && attr_kind[a] == A_POS && n == 4 && have == 3) {
                    raw[3] = 0x3F800000;          /* short array: w = 1 */
                }
                iv += have;
            }
            if (vp_count) {
                int sl = attr_slot[a] & 15;
                uint32_t have = MIN(n, 4u);
                for (uint32_t i = 0; i < have; i++) {
                    slot[sl][i] = r200_f32(raw[i]);
                }
                /* A packed colour is four bytes, not four floats. */
                if ((attr_kind[a] == A_COLOR0 || attr_kind[a] == A_COLOR1 ||
                     attr_kind[a] == A_SKIP) && attr_extra[a] == 1 && sl >= 4 && sl <= 7) {
                    uint32_t c = raw[0];
                    slot[sl][0] = ((c >> 16) & 0xFF) / 255.0f;
                    slot[sl][1] = ((c >> 8) & 0xFF) / 255.0f;
                    slot[sl][2] = (c & 0xFF) / 255.0f;
                    slot[sl][3] = (c >> 24) / 255.0f;
                }
            }
            switch (attr_kind[a]) {
            case A_POS:
                for (uint32_t i = 0; i < n && i < 4; i++) {
                    pos[i] = r200_f32(raw[i]);
                }
                break;
            case A_NORMAL:
                for (uint32_t i = 0; i < n && i < 3; i++) {
                    nrm[i] = r200_f32(raw[i]);
                }
                break;
            case A_SKIP:
                break;
            case A_COLOR0:
            case A_COLOR1: {
                float *dstc = attr_kind[a] == A_COLOR0 ? o->color : o->spec;
                if (attr_extra[a] == 1) {           /* packed, 0xAARRGGBB */
                    dstc[0] = ((raw[0] >> 16) & 0xFF) / 255.0f;
                    dstc[1] = ((raw[0] >> 8) & 0xFF) / 255.0f;
                    dstc[2] = (raw[0] & 0xFF) / 255.0f;
                    dstc[3] = (raw[0] >> 24) / 255.0f;
                } else {
                    for (uint32_t i = 0; i < n && i < 4; i++) {
                        dstc[i] = r200_f32(raw[i]);
                    }
                    /*
                     * Floating-point colours arrive blue first: Mac OS X
                     * 10.2's WindowServer fills the margin around its
                     * windows with the Aqua blue (0.678, 0.420, 0.259) and
                     * means 0x426BAD, which shows as brown unless the red
                     * and blue slots are exchanged.
                     */
                    if (n >= 3) {
                        float t = dstc[0];
                        dstc[0] = dstc[2];
                        dstc[2] = t;
                    }
                }
                break;
            }
            default: {
                int t = attr_kind[a] - A_TEX0;
                for (uint32_t i = 0; i < n && i < 4; i++) {
                    o->tex[t][i] = r200_f32(raw[i]);
                }
                if (n == 3) {            /* s,t,q */
                    o->tex[t][3] = o->tex[t][2];
                    o->tex[t][2] = 0;
                }
                break;
            }
            }
        }

        if (fmt1 == 0 && v < 4) {
            static int dump_draws;
            if (dump_draws < 6) {
                uint32_t raw[16];
                qemu_log("r200 vtxdump fmt0=0x%x v%u:", fmt0, v);
                for (int a = 0; a < nattr && a < 4 && src != R200_SRC_IMMD; a++) {
                    if (r200_read_dwords(s, arr_addr[a] + v * arr_stride[a] * 4,
                                         raw, MIN((uint32_t)attr_comps[a], 16u))) {
                        for (int i = 0; i < attr_comps[a] && i < 8; i++) {
                            qemu_log(" %08x", raw[i]);
                        }
                        qemu_log(" |");
                    }
                }
                qemu_log("\n");
                if (v == 3) {
                    dump_draws++;
                }
            }
        }

        float clip[4];
        if (vp_count) {
            PPCMacGPUVPState st;
            memset(st.temp, 0, sizeof(st.temp));
            memset(st.out_pos, 0, sizeof(st.out_pos));
            memset(st.out_color0, 0, sizeof(st.out_color0));
            memset(st.out_color1, 0, sizeof(st.out_color1));
            memset(st.out_tex, 0, sizeof(st.out_tex));
            memset(st.out_fog, 0, sizeof(st.out_fog));
            memset(st.out_psize, 0, sizeof(st.out_psize));
            st.constant = (const float (*)[4])vp_consts;
            st.wrote_pos = false;
            st.unknown_op = 0;
            for (int i = 0; i < 16; i++) {
                memcpy(st.in[vp_slot_to_input[i]], slot[i], sizeof(slot[i]));
            }
            st.out_color0[3] = st.out_color1[3] = 1.0f;
            for (int t = 0; t < 6; t++) {
                st.out_tex[t][3] = 1.0f;
            }
            ppc_mac_gpu_vp_run(&st, vp_prog, vp_count);
            if (getenv("POWEREMU_VP_TRACE")) {
                static int shown; static uint32_t shown_for;
                if (vp_count != shown_for) { shown_for = vp_count; shown = 0; }
                if (shown++ < 3) {
                    fprintf(stderr, "ppc-mac-gpu vp vertex (%u instructions):\n"
                            "   v0=%.3f %.3f %.3f %.3f\n"
                            "   v2=%.3f %.3f %.3f %.3f\n"
                            "   v3=%.3f %.3f %.3f %.3f\n"
                            "   v4=%.3f %.3f %.3f %.3f\n"
                            "   v5=%.3f %.3f %.3f %.3f\n"
                            "   c5=%.3f %.3f %.3f %.3f  c9=%.3f %.3f %.3f %.3f\n"
                            "   -> pos=%.3f %.3f %.3f %.3f\n",
                            vp_count,
                            st.in[0][0], st.in[0][1], st.in[0][2], st.in[0][3],
                            st.in[2][0], st.in[2][1], st.in[2][2], st.in[2][3],
                            st.in[3][0], st.in[3][1], st.in[3][2], st.in[3][3],
                            st.in[4][0], st.in[4][1], st.in[4][2], st.in[4][3],
                            st.in[5][0], st.in[5][1], st.in[5][2], st.in[5][3],
                            vp_consts[5][0], vp_consts[5][1], vp_consts[5][2], vp_consts[5][3],
                            vp_consts[9][0], vp_consts[9][1], vp_consts[9][2], vp_consts[9][3],
                            st.out_pos[0], st.out_pos[1], st.out_pos[2], st.out_pos[3]);
                }
            }
            memcpy(clip, st.out_pos, sizeof(clip));
            memcpy(o->color, st.out_color0, sizeof(o->color));
            memcpy(o->spec, st.out_color1, sizeof(o->spec));
            for (int t = 0; t < R200_MAX_TEX && t < 6; t++) {
                memcpy(o->tex[t], st.out_tex[t], sizeof(o->tex[t]));
            }
            if (st.unknown_op) {
                r200_warn_once(&r200_warned, 1024,
                               "vertex program opcode %u not modelled",
                               st.unknown_op);
            }
        } else if (tcl) {
            for (int r = 0; r < 4; r++) {
                clip[r] = mvp[r][0] * pos[0] + mvp[r][1] * pos[1] +
                          mvp[r][2] * pos[2] + mvp[r][3] * pos[3];
            }
            if (lighting || texproc0 || tcl_fog) {
                float p_eye[4], n_eye[4], n4[4] = { nrm[0], nrm[1], nrm[2], 0 };
                if (R3D(0x2268) & (1u << 1)) {         /* LIGHT_IN_MODELSPACE */
                    memcpy(p_eye, pos, sizeof(p_eye));
                    memcpy(n_eye, n4, sizeof(n_eye));
                } else {
                    r200_xform(mv, pos, p_eye);
                    r200_xform(itmv, n4, n_eye);
                }
                if (lighting) {
                    float c0[4], c1[4];
                    memcpy(c0, o->color, sizeof(c0));
                    memcpy(c1, o->spec, sizeof(c1));
                    r200_light_vertex(s, p_eye, n_eye, c0, c1, o->color, o->spec);
                }
                if (tcl_fog) {
                    /* Fog parameters, vector 0x5D = (-, c, d, -) as Mesa's
                     * r200 lays them out (FOG_C/FOG_D follow a pad word):
                     * linear f = c + d*z with c = end/(end-start),
                     * d = -1/(end-start); exp f = e^(d*z); exp2 f = e^(d*z^2).
                     * Components 0/1 used to be read, which fully fogged
                     * anything near z = 0 - Warcraft III's UI came out black. */
                    float z = (R3D(0x22C0) & (1u << 10))
                        ? sqrtf(p_eye[0] * p_eye[0] + p_eye[1] * p_eye[1] +
                                p_eye[2] * p_eye[2])
                        : fabsf(p_eye[2]);
                    float fc = r200_vf(s, 0x5D, 1), fd = r200_vf(s, 0x5D, 2);
                    float f = tcl_fog == 3 ? fc + fd * z
                            : tcl_fog == 1 ? expf(fd * z) : expf(fd * z * z);
                    o->spec[3] = MIN(MAX(f, 0.0f), 1.0f);
                }
                for (int t = 0; t < R200_MAX_TEX; t++) {
                    if (!(texproc0 & ((1u << t) | (0x100u << t)))) {
                        continue;
                    }
                    float in[4];
                    uint32_t src_sel = (texproc1 >> (4 * t)) & 0xF;
                    switch (src_sel) {
                    case 8:  memcpy(in, pos, sizeof(in)); break;     /* object */
                    case 9:  memcpy(in, p_eye, sizeof(in)); break;   /* eye */
                    case 10: memcpy(in, n_eye, sizeof(in)); in[3] = 1; break;
                    default:
                        memcpy(in, o->tex[src_sel < R200_MAX_TEX ? src_sel : t],
                               sizeof(in));
                        break;
                    }
                    if (texproc0 & (0x100u << t)) {          /* TEXMAT_t */
                        r200_xform(texmat[t], in, o->tex[t]);
                    } else {
                        memcpy(o->tex[t], in, sizeof(in));
                    }
                }
            }
        } else {
            memcpy(clip, pos, sizeof(clip));
        }
        float w = clip[3] != 0.0f ? clip[3] : 1.0f;
        float x = clip[0], y = clip[1], z = clip[2];
        if (!(vte & (1u << 8))) {        /* VTX_XY_FMT clear: divide by w */
            x /= w;
            y /= w;
        }
        if (!(vte & (1u << 9))) {
            z /= w;
        }
        if (vte & 0x01) x *= vp[0];
        if (vte & 0x02) x += vp[1];
        if (vte & 0x04) y *= vp[2];
        if (vte & 0x08) y += vp[3];
        if (vte & 0x10) z *= vp[4];
        if (vte & 0x20) z += vp[5];
        o->pos[0] = x + pix_bias;
        o->pos[1] = y + pix_bias;
        o->pos[2] = z;
        o->pos[3] = (vte & (1u << 10)) ? w : 1.0f;
    }

    /* ---- Primitive assembly into a triangle list ---- */
    size_t idx_need = (size_t)nverts * 6 + 6;
    if (idx_need > s->draw_idx_cap) {
        s->draw_idx = g_renew(uint32_t, s->draw_idx, idx_need);
        s->draw_idx_cap = idx_need;
    }
    uint32_t *idx = s->draw_idx;
    uint32_t ni = 0, prim_class = 0;
    switch (prim) {
    case 4:                                     /* TRIANGLES */
        for (uint32_t i = 0; i + 2 < nverts; i += 3) {
            idx[ni++] = i; idx[ni++] = i + 1; idx[ni++] = i + 2;
        }
        break;
    case 5:                                     /* TRIANGLE_FAN */
    case 15:                                    /* POLYGON */
        for (uint32_t i = 1; i + 1 < nverts; i++) {
            idx[ni++] = 0; idx[ni++] = i; idx[ni++] = i + 1;
        }
        break;
    case 6:                                     /* TRIANGLE_STRIP */
        for (uint32_t i = 0; i + 2 < nverts; i++) {
            idx[ni++] = i; idx[ni++] = i + 1; idx[ni++] = i + 2;
        }
        break;
    case 13:                                    /* QUADS */
        for (uint32_t i = 0; i + 3 < nverts; i += 4) {
            idx[ni++] = i; idx[ni++] = i + 1; idx[ni++] = i + 2;
            idx[ni++] = i; idx[ni++] = i + 2; idx[ni++] = i + 3;
        }
        break;
    case 14:                                    /* QUAD_STRIP */
        for (uint32_t i = 0; i + 3 < nverts; i += 2) {
            idx[ni++] = i; idx[ni++] = i + 1; idx[ni++] = i + 3;
            idx[ni++] = i; idx[ni++] = i + 3; idx[ni++] = i + 2;
        }
        break;
    case 8: {                                   /* RECT_LIST: 3 corners each */
        /*
         * Three corners of an axis-aligned rectangle; the fourth completes
         * it opposite the right-angle corner c (the vertex sharing x with
         * one of the others and y with the other): v3 = va + vb - vc.
         * Drivers differ in which corner comes first (X.org: TL,BL,BR;
         * Apple: TL,BL,TR), so find c rather than assume it.
         */
        uint32_t nrect = nverts / 3;
        verts = r200_draw_verts(s, nverts + nrect);
        for (uint32_t r = 0; r < nrect; r++) {
            R200Vertex *v = &verts[r * 3];
            int c = 1;
            for (int k = 0; k < 3; k++) {
                const R200Vertex *p = &v[k], *a = &v[(k + 1) % 3], *b = &v[(k + 2) % 3];
                if ((p->pos[0] == a->pos[0] && p->pos[1] == b->pos[1]) ||
                    (p->pos[0] == b->pos[0] && p->pos[1] == a->pos[1])) {
                    c = k;
                    break;
                }
            }
            int ia = (c + 1) % 3, ib = (c + 2) % 3;
            R200Vertex *v3 = &verts[nverts + r];
            *v3 = v[ia];
            for (int k = 0; k < 2; k++) {
                v3->pos[k] = v[ia].pos[k] + v[ib].pos[k] - v[c].pos[k];
            }
            for (int t = 0; t < R200_MAX_TEX; t++) {
                for (int k = 0; k < 4; k++) {
                    v3->tex[t][k] = v[ia].tex[t][k] + v[ib].tex[t][k] - v[c].tex[t][k];
                }
            }
            idx[ni++] = r * 3 + c;  idx[ni++] = r * 3 + ia; idx[ni++] = r * 3 + ib;
            idx[ni++] = r * 3 + ia; idx[ni++] = nverts + r; idx[ni++] = r * 3 + ib;
        }
        nverts += nrect;
        break;
    }
    case 1:                                     /* POINTS */
        for (uint32_t i = 0; i < nverts; i++) {
            idx[ni++] = i;
        }
        prim_class = 2;
        break;
    case 2:                                     /* LINES */
        for (uint32_t i = 0; i + 1 < nverts; i += 2) {
            idx[ni++] = i; idx[ni++] = i + 1;
        }
        prim_class = 1;
        break;
    case 3:                                     /* LINE_STRIP */
    case 12:                                    /* LINE_LOOP */
        for (uint32_t i = 0; i + 1 < nverts; i++) {
            idx[ni++] = i; idx[ni++] = i + 1;
        }
        if (prim == 12 && nverts > 2) {
            idx[ni++] = nverts - 1; idx[ni++] = 0;
        }
        prim_class = 1;
        break;
    default:
        r200_warn_once(&r200_warned, 512, "primitive type %u not supported",
                       prim);
        break;
    }
    if (ni == 0) {
        return true;
    }

    /* ---- Render target, scissor, pixel state ---- */
    R200DrawPacket pkt = { 0 };
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;
    uint32_t tl = R3D(0x26C0), br = R3D(0x1C44);  /* RE_TOP_LEFT, RE_WIDTH_HEIGHT */
    pkt.rt_offset = (R3D(0x1C40) & ~0xFu) - fb_base;
    pkt.rt_pitch = R3D(0x1C48) & 0x7FF;
    pkt.rb3d_cntl = R3D(0x1C3C);
    pkt.rt_format = (pkt.rb3d_cntl >> 10) & 0xF;
    pkt.scissor[0] = tl & 0x7FF;
    pkt.scissor[1] = (tl >> 16) & 0x7FF;
    pkt.scissor[2] = (br & 0x7FF) + 1;
    pkt.scissor[3] = ((br >> 16) & 0x7FF) + 1;
    /* Auxiliary scissors (RE_AUX_SCISSOR_CNTL): an enabled inclusive rect
     * clips like the main scissor; an exclusive one masks its inside. */
    uint32_t aux = R3D(0x26F0);
    for (int k = 0; k < 3; k++) {
        if (!(aux & (0x10000000u << k))) {
            continue;
        }
        uint32_t atl = R3D(0x1CD8 + 8 * k), abr = R3D(0x1CDC + 8 * k);
        uint32_t r[4] = { atl & 0x7FF, (atl >> 16) & 0x7FF,
                          (abr & 0x7FF) + 1, ((abr >> 16) & 0x7FF) + 1 };
        if (aux & (0x01000000u << k)) {
            memcpy(pkt.excl[pkt.num_excl++], r, sizeof(r));
        } else {
            pkt.scissor[0] = MAX(pkt.scissor[0], r[0]);
            pkt.scissor[1] = MAX(pkt.scissor[1], r[1]);
            pkt.scissor[2] = MIN(pkt.scissor[2], r[2]);
            pkt.scissor[3] = MIN(pkt.scissor[3], r[3]);
        }
    }
    pkt.rt_width = pkt.rt_pitch;
    pkt.rt_height = ((br >> 16) & 0x7FF) + 1;  /* stable across aux clips */
    pkt.pp_cntl = R3D(0x1C38);
    pkt.pp_misc = R3D(0x1C14);
    pkt.depth_enable = (pkt.rb3d_cntl >> 8) & 1;
    pkt.stencil_enable = (pkt.rb3d_cntl >> 7) & 1;
    if (pkt.depth_enable || pkt.stencil_enable) {
        uint32_t zfmt;
        pkt.zstencil = R3D(0x1C2C);
        pkt.stencil_refmask = R3D(0x1D7C);
        pkt.depth_offset = (R3D(0x1C24) & ~0xFu) - fb_base;
        pkt.depth_pitch = R3D(0x1C28) & 0x1FF8;
        zfmt = pkt.zstencil & 0xF;
        pkt.depth_bpp = zfmt == 0 ? 2 : 4;
        if (zfmt != 0 && zfmt != 2 && zfmt != 4) {
            r200_warn_once(&r200_warned, 2048, "depth format %u treated as "
                           "integer Z", zfmt);
        }
    }
    pkt.plane_mask = (pkt.rb3d_cntl & (1u << 1)) ? R3D(0x1D84) : 0xFFFFFFFFu;
    pkt.fog_color = R3D(0x1C18);
    pkt.prim_class = prim_class;
    pkt.cblend = R3D(0x3220);
    pkt.ablend = R3D(0x321C);
    pkt.blend_color = R3D(0x3218);
    for (int i = 0; i < R200_MAX_STAGES; i++) {
        pkt.txcblend[i]  = R3D(0x2F00 + i * 0x10);
        pkt.txcblend2[i] = R3D(0x2F04 + i * 0x10);
        pkt.txablend[i]  = R3D(0x2F08 + i * 0x10);
        pkt.txablend2[i] = R3D(0x2F0C + i * 0x10);
    }
    static const uint16_t tf_reg[8] = { 0x1C68, 0x1C80, 0x1C98, 0x2EEC,
                                        0x2EF0, 0x2EF4, 0x2EF8, 0x2EFC };
    for (int i = 0; i < 8; i++) {
        pkt.tfactor[i] = R3D(tf_reg[i]);
    }
    for (int t = 0; t < R200_MAX_TEX; t++) {
        if (pkt.pp_cntl & (0x10u << t)) {
            r200_decode_tex_unit(s, t, &pkt.tex[t]);
            /* SE_VTE_CNTL VTX_ST_DENORMALIZED: s,t arrive in texels.  The
             * driver's AGP->VRAM texture copies (Warcraft III's glyph cache)
             * use it on power-of-two textures. */
            if (R3D(0x20B0) & (1u << 12)) {
                pkt.tex[t].denorm = 1;
            }
        }
    }
    pkt.verts = verts;
    pkt.num_verts = nverts;
    pkt.indices = idx;
    pkt.num_indices = ni;

    {
        const R200Vertex *v0 = &verts[idx[0]], *v2 = &verts[idx[MIN(2u, ni - 1)]];
        seq_log("3D   rt=%06x/%u sc=(%u,%u)-(%u,%u) tex0=%06x/%u %ux%u f%u "
                "blend=%u cb=%08x txc=%08x txa=%08x nv=%u "
                "v0=(%.1f,%.1f) v2=(%.1f,%.1f) t0=(%.1f,%.1f) t2=(%.1f,%.1f) "
                "col=(%.2f,%.2f,%.2f,%.2f)",
                pkt.rt_offset, pkt.rt_pitch, pkt.scissor[0], pkt.scissor[1],
                pkt.scissor[2], pkt.scissor[3], pkt.tex[0].offset,
                pkt.tex[0].pitch, pkt.tex[0].width, pkt.tex[0].height,
                pkt.tex[0].format, pkt.rb3d_cntl & 1, pkt.cblend,
                pkt.txcblend[0], pkt.txablend[0], nverts,
                v0->pos[0], v0->pos[1], v2->pos[0], v2->pos[1],
                v0->tex[0][0], v0->tex[0][1], v2->tex[0][0], v2->tex[0][1],
                v0->color[0], v0->color[1], v0->color[2], v0->color[3]);
    }

    static int texlog = -1;
    if (texlog < 0) {
        texlog = getenv("PPCGPU_TEXLOG") != NULL;
    }
    if (texlog) {
        /* Debug: log each texture unit whose parameters or leading texels
         * changed since it was last seen at that offset. */
        static struct { uint32_t off, key, crc; } tl[256];
        static int ntl;
        static uint64_t ndraw;
        ndraw++;
        uint8_t *vr = memory_region_get_ram_ptr(&s->vram);
        for (int t = 0; t < R200_MAX_TEX; t++) {
            const R200TexUnit *u = &pkt.tex[t];
            if (!u->enabled) {
                continue;
            }
            uint32_t key = u->format | u->width << 5 | u->height << 18 |
                           (u->host_data ? 1u << 31 : 0);
            uint32_t crc = 0;
            const uint8_t *src = u->host_data ? u->host_data : vr + u->offset;
            uint64_t len = MIN((uint64_t)u->pitch * u->height, 65536);
            if (u->host_data || u->offset + len <= s->vram_size) {
                for (uint64_t i = 0; i < len; i += 4) {
                    crc = (crc ^ ldl_he_p(src + i)) * 16777619u;
                }
            }
            int j;
            for (j = 0; j < ntl && tl[j].off != u->offset; j++) {
            }
            if (j == ntl || tl[j].key != key || tl[j].crc != crc) {
                if (j == ntl && ntl < 256) {
                    ntl++;
                }
                if (j < 256) {
                    tl[j].off = u->offset; tl[j].key = key; tl[j].crc = crc;
                }
                static int ndump, texdump = -1;
                if (texdump < 0) {
                    texdump = getenv("PPCGPU_TEXDUMP") != NULL;
                }
                if (texdump && ndump < 600 &&
                    (u->format != 6 || u->host_data) &&
                    u->pitch * u->height <= 4 * 1024 * 1024 &&
                    (u->host_data || u->offset + (uint64_t)u->pitch * u->height
                                     <= s->vram_size)) {
                    char fn[128];
                    g_mkdir_with_parents("/tmp/texdump", 0755);
                    snprintf(fn, sizeof(fn), "/tmp/texdump/%03d_u%d_%s%08x_f%u_%ux%u_p%u_tc%08x_ta%08x.bin",
                             ndump++, t, u->host_data ? "agp" : "vram",
                             u->host_data ? R3D(0x1C5C + t * 0x18) : u->offset,
                             u->format, u->width, u->height, u->pitch,
                             pkt.txcblend[0], pkt.txablend[0]);
                    g_file_set_contents(fn, (const char *)src,
                                        (gssize)u->pitch * u->height, NULL);
                }
                qemu_log("r200 tex d=%llu u%d off=%08x fmt=%u %ux%u p=%u "
                         "filt=%08x agp=%d crc=%08x first=%08x cb=%08x\n",
                         (unsigned long long)ndraw, t, u->offset, u->format,
                         u->width, u->height, u->pitch, u->filter,
                         u->host_data != NULL, crc,
                         (u->host_data || u->offset + 4 <= s->vram_size)
                         ? ldl_be_p(src) : 0, pkt.txcblend[0]);
            }
        }
    }

    {
        /* PPCGPU_TRACETEX=<hex offset>: full state of draws sampling it. */
        static int tf = -2, tw, th, tagp;  /* "fmt:w:h[:agp]" */
        static int traced;
        if (tf == -2) {
            const char *e = getenv("PPCGPU_TRACETEX");
            tf = -1;
            if (e && sscanf(e, "%d:%d:%d", &tf, &tw, &th) != 3) {
                tf = -1;
            }
            tagp = e && strstr(e, ":agp") != NULL;
        }
        if (tf >= 0 && traced < 6 && pkt.tex[0].enabled &&
            (!tagp || pkt.tex[0].host_data) &&
            pkt.tex[0].format == (uint32_t)tf && pkt.tex[0].width == (uint32_t)tw &&
            pkt.tex[0].height == (uint32_t)th) {
            traced++;
            const R200Vertex *v0 = &verts[idx[0]];
            qemu_log("r200 tracetex: prim=%u n=%u pp=%08x rb3d=%08x cb=%08x ab=%08x "
                     "misc=%08x z=%08x txc0=%08x/%08x txa0=%08x/%08x tf0=%08x "
                     "fmt=%08x/%08x vte=%08x vap=%08x light=%08x tex0filt=%08x "
                     "v0 pos=(%.1f,%.1f,%.3f,%.3f) col=(%.2f,%.2f,%.2f,%.2f) "
                     "spec=(%.2f,%.2f,%.2f,%.2f) t0=(%.3f,%.3f,%.3f,%.3f) rt=%x\n",
                     prim, nverts, pkt.pp_cntl, pkt.rb3d_cntl, pkt.cblend,
                     pkt.ablend, pkt.pp_misc, pkt.zstencil, pkt.txcblend[0],
                     pkt.txcblend2[0], pkt.txablend[0], pkt.txablend2[0],
                     pkt.tfactor[0], fmt0, fmt1, R3D(0x20B0), R3D(0x2080),
                     R3D(0x2268), pkt.tex[0].filter,
                     v0->pos[0], v0->pos[1], v0->pos[2], v0->pos[3],
                     v0->color[0], v0->color[1], v0->color[2], v0->color[3],
                     v0->spec[0], v0->spec[1], v0->spec[2], v0->spec[3],
                     v0->tex[0][0], v0->tex[0][1], v0->tex[0][2], v0->tex[0][3],
                     pkt.rt_offset);
            for (uint32_t k = 0; k < nverts && k < 8; k++) {
                const R200Vertex *vk = &verts[k];
                qemu_log("r200 tracetex  v%u pos=(%.2f,%.2f,%.3f,%.3f) t0=(%.4f,%.4f) "
                         "col=(%.2f,%.2f,%.2f,%.2f)\n", k, vk->pos[0], vk->pos[1],
                         vk->pos[2], vk->pos[3], vk->tex[0][0], vk->tex[0][1],
                         vk->color[0], vk->color[1], vk->color[2], vk->color[3]);
            }
            qemu_log("r200 tracetex  ni=%u idx:", ni);
            for (uint32_t k = 0; k < ni && k < 12; k++) {
                qemu_log(" %u", idx[k]);
            }
            qemu_log(" sc=(%u,%u)-(%u,%u) rtpitch=%u rth=%u\n", pkt.scissor[0],
                     pkt.scissor[1], pkt.scissor[2], pkt.scissor[3], pkt.rt_pitch,
                     pkt.rt_height);
            qemu_log("r200 tracetex fog: 22c0=%08x fogcol=%08x vec5d=(%g,%g,%g,%g) "
                     "mv=%u mvrow2=(%g,%g,%g,%g) mvp=%u vtx_raw_pos?\n",
                     R3D(0x22C0), R3D(0x1C18),
                     r200_vf(s, 0x5D, 0), r200_vf(s, 0x5D, 1), r200_vf(s, 0x5D, 2),
                     r200_vf(s, 0x5D, 3), R3D(0x2230) & 0x1F,
                     r200_vf(s, 0x80 + 4 * (R3D(0x2230) & 0x1F) + 2, 0),
                     r200_vf(s, 0x80 + 4 * (R3D(0x2230) & 0x1F) + 2, 1),
                     r200_vf(s, 0x80 + 4 * (R3D(0x2230) & 0x1F) + 2, 2),
                     r200_vf(s, 0x80 + 4 * (R3D(0x2230) & 0x1F) + 2, 3),
                     R3D(0x2238) & 0x1F);
        }
    }

    if (gpu_diag_on()) {
        /* Log each new combination of pipeline state once: a cheap map of
         * which R200 features an application actually exercises. */
        static uint64_t seen[512];
        static int nseen;
        uint32_t k[] = {
            prim | (src << 8), R3D(0x2080), fmt0, fmt1, R3D(0x20B0),
            pkt.pp_cntl, pkt.rb3d_cntl, pkt.zstencil, pkt.cblend, pkt.ablend,
            R3D(0x2268), R3D(0x22B0), R3D(0x22B4), pkt.pp_misc,
            pkt.tex[0].format | pkt.tex[1].format << 8 | pkt.tex[2].format << 16,
            pkt.txcblend[0], pkt.txablend[0], pkt.txcblend[1], pkt.txablend[1],
        };
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < ARRAY_SIZE(k); i++) {
            h = (h ^ k[i]) * 1099511628211ull;
        }
        bool found = false;
        for (int i = 0; i < nseen; i++) {
            found |= seen[i] == h;
        }
        if (!found && nseen < (int)ARRAY_SIZE(seen)) {
            seen[nseen++] = h;
            qemu_log("r200 combo #%d: prim=%u src=%d vap=%08x fmt=%08x/%08x "
                     "vte=%08x pp=%08x rb3d=%08x z=%08x cb=%08x ab=%08x "
                     "light=%08x texproc=%08x/%08x misc=%08x txfmt=%u/%u/%u "
                     "txc0=%08x txa0=%08x txc1=%08x txa1=%08x rt=%u depth@%x "
                     "vport=%.1f,%.1f,%.1f,%.1f re_tl=%08x re_wh=%08x\n",
                     nseen, prim, src, R3D(0x2080), fmt0, fmt1, R3D(0x20B0),
                     pkt.pp_cntl, pkt.rb3d_cntl, pkt.zstencil, pkt.cblend,
                     pkt.ablend, R3D(0x2268), R3D(0x22B0), R3D(0x22B4),
                     pkt.pp_misc, pkt.tex[0].format, pkt.tex[1].format,
                     pkt.tex[2].format, pkt.txcblend[0], pkt.txablend[0],
                     pkt.txcblend[1], pkt.txablend[1], pkt.rt_pitch,
                     pkt.depth_offset, vp[0], vp[1], vp[2], vp[3],
                     R3D(0x26C0), R3D(0x1C44));
        }
    }

    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t rt_cfmt = (pkt.rb3d_cntl >> 10) & 0xF;
    uint32_t rt_bpp = (rt_cfmt == 3 || rt_cfmt == 4 || rt_cfmt == 15) ? 2 : 4;
    /* A queue-async renderer takes the AGP texture copies over with the
     * draw; note which units had one before handing it over. */
    bool tex_agp[R200_MAX_TEX];
    for (int t = 0; t < R200_MAX_TEX; t++) {
        tex_agp[t] = pkt.tex[t].host_data != NULL;
    }
    if (pkt.rt_pitch == 0 ||
        (uint64_t)pkt.rt_offset + (uint64_t)pkt.rt_height * pkt.rt_pitch * rt_bpp >
        s->vram_size) {
        r200_warn_once(&r200_warned, 1024,
                       "render target 0x%x pitch %u height %u outside VRAM",
                       pkt.rt_offset, pkt.rt_pitch, pkt.rt_height);
    } else if (r200_rate.draws++,
               r200_render_draw(s, vram, &pkt) == 0) {
        ppc_vram_dirty_e0(s, pkt.rt_offset,
                                (uint64_t)pkt.rt_height * pkt.rt_pitch * rt_bpp);
        r200_perf.draws++;
        r200_traffic.draws++;
        r200_perf.drew = true;
        r200_perf_high((uint64_t)pkt.rt_offset + (uint64_t)pkt.rt_height * pkt.rt_pitch * rt_bpp);
        if (pkt.depth_enable && pkt.depth_pitch) {
            r200_perf_high((uint64_t)pkt.depth_offset +
                           (uint64_t)pkt.rt_height * pkt.depth_pitch * pkt.depth_bpp);
        }
        for (int t = 0; t < R200_MAX_TEX; t++) {
            if (!pkt.tex[t].enabled) {
                continue;
            }
            uint64_t bytes = (uint64_t)pkt.tex[t].pitch * pkt.tex[t].height;
            if (tex_agp[t]) {
                r200_perf.tex_agp++;
                r200_perf.agp_bytes += bytes;
            } else {
                r200_perf.tex_vram++;
                r200_perf_high((uint64_t)pkt.tex[t].offset + bytes);
            }
        }
    }
    for (int t = 0; t < R200_MAX_TEX; t++) {
        g_free((void *)pkt.tex[t].host_data);
    }
    return true;
}

/*
 * Hand the draw to a worker when asynchronous drawing is on, and otherwise
 * run it here. Returning the same value either way is deliberate: a queued
 * draw has not failed, and its result cannot be known yet.
 */
static bool ppc_mac_gpu_r200_draw(PPCMacGPUState *s, const uint32_t *d,
                                  uint32_t body_dw, int src)
{
    if (r200_async_submit(s, d, body_dw, src)) {
        return true;
    }
    return ppc_mac_gpu_r200_draw_now(s, d, body_dw, src);
}

static void ppc_mac_gpu_dispatch_3d_draw(PPCMacGPUState *s,
                                          const PPCMacGPU3DDrawCmd *cmd)
{
    /* Phase A — pre-dispatch VRAM check for draws 77-78 */
    {
        static int dispatch_count = 0;
        static bool track_vram_clr = false;
        if (dispatch_count == 77) {
            track_vram_clr = true;
            fprintf(stderr, "[DISPATCH_CHECK] Starting VRAM 0x353000 tracking "
                    "after dispatch 77\n");
        }
        if (dispatch_count == 77 || dispatch_count == 78) {
            uint8_t *vp = memory_region_get_ram_ptr(&s->vram);
            uint32_t v = 0;
            if (0x353000 + 4 <= s->vram_size) {
                v = *(uint32_t *)(vp + 0x353000);
            }
            fprintf(stderr, "[DISPATCH_CHECK] dispatch=%d PRE-DISPATCH "
                    "vram[0x353000]=0x%08x\n", dispatch_count, v);
        }
        if (dispatch_count == 79) track_vram_clr = false;
        dispatch_count++;
        /* Export tracking flag for PM4 processor */
        s->vram_watch_active = track_vram_clr;
    }

    /* Post-probe audit: increment global draw counter */
    g_post_probe_draw_count++;

    PPCMacGPU3DState state;
    ppc_mac_gpu_snapshot_3d_state(s, &state);

    /* Phase 1: Log every 3D draw for compositor analysis */
    if (gpu_diag_on()) {
        ppc_mac_gpu_log_3d_draw(&state, cmd);
    }

    /*
     * Phase 3 — Next-gate detection.
     *
     * Track blend mode and texture state across ALL draws.
     * If we see 1000+ draws and never get a non-NOP blend or
     * real texture (offset != 0, size != 1×1), emit [NEXT_GATE].
     */
    {
        static int gate_draw_count = 0;
        static int nop_count = 0;
        static int real_blend_count = 0;
        static int real_tex_count = 0;
        static int gate_logged = 0;
        static uint64_t first_real_blend_at = 0;
        static uint64_t first_real_tex_at = 0;
        static int unique_targets = 0;
        static uint32_t seen_targets[32];
        static int seen_target_count = 0;

        gate_draw_count++;

        uint32_t blend = state.rb3d_blendcntl;
        uint32_t src_b = blend & 0x1F;
        uint32_t dst_b = (blend >> 16) & 0x1F;
        bool is_nop = (src_b == 0 && dst_b == 1);

        uint32_t tex_off = state.pp_txoffset_0;
        uint32_t tex_w = (state.pp_txsize_0 & 0x7FF) + 1;
        uint32_t tex_h = ((state.pp_txsize_0 >> 16) & 0x7FF) + 1;
        bool real_tex = (tex_off != 0 || tex_w > 1 || tex_h > 1);
        bool tex_en = (state.pp_cntl >> 4) & 1;

        if (is_nop) {
            nop_count++;
        } else {
            real_blend_count++;
            if (real_blend_count == 1) {
                first_real_blend_at = gate_draw_count;
                qemu_log("[NEXT_GATE] event=first_real_blend "
                         "draw_id=%d blend=0x%08x src=0x%x dst=0x%x "
                         "tex_off=0x%x tex_size=%ux%u tex_en=%d "
                         "color_off=0x%x\n",
                         gate_draw_count, blend, src_b, dst_b,
                         tex_off, tex_w, tex_h, tex_en,
                         state.rb3d_coloroffset);
            }
        }
        if (real_tex && tex_en) {
            real_tex_count++;
            if (real_tex_count == 1) {
                first_real_tex_at = gate_draw_count;
                qemu_log("[NEXT_GATE] event=first_real_texture "
                         "draw_id=%d tex_off=0x%x tex_size=%ux%u "
                         "tex_fmt=0x%x blend=0x%08x\n",
                         gate_draw_count, tex_off, tex_w, tex_h,
                         state.pp_txformat_0, blend);
            }
        }

        /* Track unique render targets */
        {
            uint32_t co = state.rb3d_coloroffset;
            bool found = false;
            for (int t = 0; t < seen_target_count; t++) {
                if (seen_targets[t] == co) { found = true; break; }
            }
            if (!found && seen_target_count < 32) {
                seen_targets[seen_target_count++] = co;
            }
        }

        /* Emit summary at milestones */
        if (!gate_logged &&
            (gate_draw_count == 200 || gate_draw_count == 500 ||
             gate_draw_count == 1000 || gate_draw_count == 2000)) {
            qemu_log("[NEXT_GATE] event=draw_milestone "
                     "sequence_id=%d total_draws=%d "
                     "nop_draws=%d real_blend_draws=%d "
                     "real_tex_draws=%d unique_targets=%d "
                     "first_real_blend_at=%llu first_real_tex_at=%llu "
                     "reason=%s confidence=%s\n",
                     gate_draw_count, gate_draw_count,
                     nop_count, real_blend_count,
                     real_tex_count, seen_target_count,
                     (unsigned long long)first_real_blend_at,
                     (unsigned long long)first_real_tex_at,
                     (real_blend_count == 0 && real_tex_count == 0)
                         ? "all_draws_nop_blend_dummy_texture_QE_not_compositing"
                         : (real_blend_count > 0 && real_tex_count > 0)
                             ? "QE_compositing_active"
                             : "partial_QE_activity",
                     (gate_draw_count >= 1000)
                         ? "high" : "medium");

            /* At 1000 draws, if still all NOP, emit definitive gate */
            if (gate_draw_count >= 1000 && real_blend_count == 0) {
                gate_logged = 1;
                qemu_log("[NEXT_GATE] event=definitive_gate "
                         "sequence_id=%d "
                         "reg_or_addr=PP_TXOFFSET_0/RB3D_BLENDCNTL "
                         "value=never_written_with_real_values "
                         "reason=QE_compositor_never_sends_real_textured_draws_"
                         "all_%d_draws_are_NOP_blend_with_1x1_dummy_texture_"
                         "WindowServer_has_not_activated_Quartz_Extreme "
                         "confidence=high\n",
                         gate_draw_count, gate_draw_count);
            }
        }
    }

    if (!s->renderer || !s->renderer->draw_3d) {
        return;
    }

    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    PPCMacGPU3DDrawCmd resolved_cmd = *cmd;
    uint32_t *gart_vb_data = NULL;

    /*
     * If DRAW_VBUF and vertex buffer is outside VRAM (i.e. in GART space),
     * read vertex data from system RAM via GART translation and convert
     * to DRAW_IMMD with inline vertex data.
     */
    if (cmd->opcode == R200_3D_DRAW_VBUF && cmd->vb_addr >= s->vram_size &&
        cmd->vb_stride > 0 && cmd->num_vertices > 0 &&
        cmd->num_vertices <= 65536) {
        uint32_t total_bytes = cmd->num_vertices * cmd->vb_stride;
        uint32_t total_dw = (total_bytes + 3) / 4;
        gart_vb_data = g_malloc(total_dw * 4);

        bool ok = true;
        uint32_t bytes_read = 0;
        uint32_t gpu_addr = cmd->vb_addr;

        while (bytes_read < total_bytes) {
            hwaddr phys;
            /* Try AGP (MC_AGP_LOCATION) first, then AIC GART */
            if (!ppc_mac_gpu_agp_translate(s, gpu_addr, &phys) &&
                !ppc_mac_gpu_gart_translate(s, gpu_addr, &phys)) {
                gpu_debug_log("3D_DRAW: AGP/GART translate failed at 0x%x",
                              gpu_addr);
                ok = false;
                break;
            }
            /* Read one page at a time (handle page boundaries) */
            uint32_t page_off = gpu_addr & 0xFFF;
            uint32_t chunk = 0x1000 - page_off;
            if (chunk > total_bytes - bytes_read) {
                chunk = total_bytes - bytes_read;
            }
            MemTxResult r = address_space_read(
                &address_space_memory, phys,
                MEMTXATTRS_UNSPECIFIED,
                ((uint8_t *)gart_vb_data) + bytes_read, chunk);
            if (r != MEMTX_OK) {
                gpu_debug_log("3D_DRAW: GART read failed at phys 0x%"PRIx64,
                              (uint64_t)phys);
                ok = false;
                break;
            }
            bytes_read += chunk;
            gpu_addr += chunk;
        }

        if (ok) {
            /*
             * Byte-swap vertex data: PPC guest writes BE to system RAM,
             * but GPU (and our renderer) expects LE floats.
             * The AGP bridge normally does this swap in hardware.
             */
            for (uint32_t dw = 0; dw < total_dw; dw++) {
                gart_vb_data[dw] = bswap32(gart_vb_data[dw]);
            }

            resolved_cmd.opcode = R200_3D_DRAW_IMMD;
            resolved_cmd.vertex_data = gart_vb_data;
            resolved_cmd.vertex_data_dwords = total_dw;
            /* vb_stride stays set so Metal knows bytes-per-vertex */

            /* Log vertex data for first 10 draws to understand format */
            static int vb_dump_count = 0;
            if (vb_dump_count < 10) {
                uint32_t stride_dw = cmd->vb_stride / 4;
                gpu_debug_log("3D_VB_DATA draw#%d: %u verts stride=%u "
                              "aos_desc0=0x%08x",
                              vb_dump_count, cmd->num_vertices, cmd->vb_stride,
                              s->regs.regs_3d[R200_3D_IDX(R200_AOS_DESC_0)]);
                for (uint32_t v = 0; v < cmd->num_vertices && v < 8; v++) {
                    const uint32_t *vd = gart_vb_data + v * stride_dw;
                    float f[4];
                    for (int k = 0; k < 4 && k < (int)stride_dw; k++) {
                        memcpy(&f[k], &vd[k], 4);
                    }
                    gpu_debug_log("  v[%u]: (%.3f, %.3f, %.3f, %.3f)",
                                  v, f[0], f[1], f[2], f[3]);
                }
                /* Also dump raw 48-byte region (3 arrays worth) for vertex 0 */
                if (total_dw >= 12) {
                    gpu_debug_log("  raw 48B for v0:");
                    for (int row = 0; row < 3; row++) {
                        float rf[4];
                        for (int k = 0; k < 4; k++) {
                            memcpy(&rf[k], &gart_vb_data[row * 4 + k], 4);
                        }
                        gpu_debug_log("    [%d]: (%.3f, %.3f, %.3f, %.3f)",
                                      row, rf[0], rf[1], rf[2], rf[3]);
                    }
                }
                vb_dump_count++;
            }
        } else {
            g_free(gart_vb_data);
            gart_vb_data = NULL;
            return;
        }
    }

    {
        static int disp3d = 0;
        if (disp3d < 5) {
            gpu_debug_log("3D_DISPATCH_TO_SW: renderer=%p draw_3d=%p opaque=%p cmd_op=0x%x",
                          (void*)s->renderer, (void*)s->renderer->draw_3d,
                          (void*)s->renderer_opaque, resolved_cmd.opcode);
            disp3d++;
        }
    }
    int ret = s->renderer->draw_3d(s->renderer_opaque,
                                    vram, s->vram_size,
                                    &state, &resolved_cmd);
    g_free(gart_vb_data);

    if (ret == 0) {
        /* 3D draw succeeded — mark the color buffer region dirty */
        uint32_t color_offset = state.rb3d_coloroffset;
        uint64_t dirty_len = (uint64_t)state.screen_height *
                             (uint64_t)state.screen_width * 4;
        if (color_offset + dirty_len <= s->vram_size) {
            ppc_vram_dirty_e0(s, color_offset, dirty_len);
        }
        s->display_invalid = true;

        /* Phase D: Record this RT offset for present BLT source tracking */
        bool found = false;
        for (int t = 0; t < s->metal_rt_count; t++) {
            if (s->metal_rt_offsets[t] == color_offset) {
                found = true;
                break;
            }
        }
        if (!found && s->metal_rt_count < 16) {
            s->metal_rt_offsets[s->metal_rt_count++] = color_offset;
        }
    }
}

/*
 * Process a stream of PM4 packets from a pre-read buffer.
 * Used by both IB execution and ring buffer processing.
 */
/* $R300_RINGDUMP=path: every PM4 packet (header and body) and IB, capped. */
static FILE *r300_ringdump(void)
{
    static FILE *f;
    static int init;
    static uint64_t bytes;

    if (!init) {
        const char *p = getenv("R300_RINGDUMP");
        init = 1;
        f = p ? fopen(p, "w") : NULL;
    }
    if (f && ftell(f) > (long)2000 * 1024 * 1024) {
        fclose(f);
        f = NULL;
    }
    (void)bytes;
    return f;
}

static bool ppc_mac_gpu_cp_yield(PPCMacGPUState *s, uint32_t pos_dw);
static bool cp_in_ring;             /* the CP thread is replaying the ring */

static void ppc_mac_gpu_process_pm4(PPCMacGPUState *s,
                                     uint32_t *pm4_data,
                                     uint32_t size_dw)
{
    g_in_pm4++;
    uint32_t i = 0;
    while (i < size_dw) {
        /* Between two ring packets the CP thread may let others in. */
        if (unlikely(cp_in_ring) && g_in_pm4 == 1 &&
            ppc_mac_gpu_cp_yield(s, i)) {
            break;
        }
        uint32_t hdr = pm4_data[i];
        uint32_t type = (hdr >> 30) & 3;
        FILE *rd = r300_ringdump();
        if (rd) {
            uint32_t n = type == 2 ? 0 : ((hdr >> 16) & 0x3FFF) + 1;
            fprintf(rd, "%c%u %08x:", "0123"[type], g_in_pm4, hdr);
            for (uint32_t k = 0; k < n && i + 1 + k < size_dw && k < 512; k++) {
                fprintf(rd, " %08x", pm4_data[i + 1 + k]);
            }
            fputc('\n', rd);
        }
        i++;

        if (type == 0) {
            /* Type 0: register write(s)
             * Bits [14:0]  = register index
             * Bit  [15]    = ONE_REG_WR (all writes to same register)
             * Bits [29:16] = count - 1
             */
            uint32_t pkt_count = ((hdr >> 16) & 0x3FFF) + 1;
            uint32_t reg_base = hdr & 0x7FFF;
            bool one_reg_wr = (hdr >> 15) & 1;

            /* Phase A — VRAM watch: check before & after Type 0 processing */
            uint32_t vw_before = 0;
            if (s->vram_watch_active && 0x353000 + 4 <= s->vram_size) {
                uint8_t *vp = memory_region_get_ram_ptr(&s->vram);
                vw_before = *(uint32_t *)(vp + 0x353000);
            }

            if (i + pkt_count > size_dw) {
                pkt_count = size_dw - i;
            }

            ppc_mac_gpu_pm4_process_type0(s, reg_base, one_reg_wr, pkt_count,
                                           &pm4_data[i]);

            /* Phase A — VRAM watch: check after Type 0 processing */
            if (s->vram_watch_active && 0x353000 + 4 <= s->vram_size) {
                uint8_t *vp = memory_region_get_ram_ptr(&s->vram);
                uint32_t vw_after = *(uint32_t *)(vp + 0x353000);
                if (vw_after != vw_before) {
                    fprintf(stderr, "[VRAM_CLR_FOUND] Type0 reg=0x%04x "
                            "count=%u CHANGED vram[0x353000] from "
                            "0x%08x to 0x%08x\n",
                            reg_base * 4, pkt_count,
                            vw_before, vw_after);
                }
            }

            i += pkt_count;
        } else if (type == 2) {
            /* Type 2: NOP */
        } else if (type == 3) {
            /* Type 3: GPU engine command */
            uint32_t pkt_count = ((hdr >> 16) & 0x3FFF) + 2;
            uint32_t opcode = (hdr >> 8) & 0xFF;
            uint32_t body_dw = pkt_count - 1;
            uint32_t body_adv = body_dw;   /* 0x1B may widen body_dw */
            uint32_t *d = &pm4_data[i];
            if (s->r3 && ppc_mac_gpu_r300_packet3(s, opcode, d, body_dw)) {
                opcode = 0x10;      /* consumed: let the common path see a NOP */
            }
            if (opcode == 0x29) {
                static int immd1_dumps;
                if (immd1_dumps++ < 4) {
                    GString *g = g_string_new(NULL);
                    for (uint32_t k = 0; k < body_dw && k < 48; k++) {
                        g_string_append_printf(g, " %08x", d[k]);
                    }
                    qemu_log("r200 DRAW_IMMD(0x29) n=%u fmt0=%08x fmt1=%08x vte=%08x "
                             "vap=%08x pp=%08x rt=%08x/%u tex0=%08x raw:%s\n", body_dw,
                             s->regs.regs_3d[R200_3D_IDX(0x2088)],
                             s->regs.regs_3d[R200_3D_IDX(0x208C)],
                             s->regs.regs_3d[R200_3D_IDX(0x20B0)],
                             s->regs.regs_3d[R200_3D_IDX(0x2080)],
                             s->regs.regs_3d[R200_3D_IDX(0x1C38)],
                             s->regs.regs_3d[R200_3D_IDX(0x1C40)],
                             s->regs.regs_3d[R200_3D_IDX(0x1C48)] & 0x7ff,
                             s->regs.regs_3d[R200_3D_IDX(0x1C5C)], g->str);
                    g_string_free(g, TRUE);
                }
            }
            if (opcode != 0x34 && opcode != 0x35 && opcode != 0x36) {
                seq_log("T3   op=%02x n=%u d0=%08x d1=%08x d2=%08x", opcode, body_dw,
                        body_dw > 0 ? d[0] : 0, body_dw > 1 ? d[1] : 0,
                        body_dw > 2 ? d[2] : 0);
            }


            /* Phase A — VRAM watch: check before Type 3 processing */
            uint32_t t3_vw_before = 0;
            if (s->vram_watch_active && 0x353000 + 4 <= s->vram_size) {
                uint8_t *vp = memory_region_get_ram_ptr(&s->vram);
                t3_vw_before = *(uint32_t *)(vp + 0x353000);
            }

            /*
             * BITBLT_MULTI (0x1B): the header-less form.  Its body is just
             * (src X_Y, dst X_Y, W_H) triplets; the control word, pitch/
             * offsets and clip come from the last CNTL_BITBLT_MULTI.  The
             * window server moves a dragged window's body with it (one
             * 0x9B for a strip, then 0x1B for the body and the rest), so
             * without it the body never reaches the compositor's buffer.
             */
            static uint32_t bbm_hdr[6], bbm_hdr_n;
            uint32_t bbm_cont[256];
            if (opcode == 0x1b && bbm_hdr_n && body_dw >= 3 &&
                bbm_hdr_n + body_dw <= ARRAY_SIZE(bbm_cont)) {
                memcpy(bbm_cont, bbm_hdr, bbm_hdr_n * 4);
                memcpy(bbm_cont + bbm_hdr_n, d, body_dw * 4);
                d = bbm_cont;
                body_dw += bbm_hdr_n;
                opcode = 0x9b;
            }
            /*
             * CNTL_BITBLT (0x92): one rectangle, laid out like a
             * CNTL_BITBLT_MULTI with a single (src X_Y, dst X_Y, W_H)
             * triplet.  QuartzGL scrolls a view with it (System
             * Preferences' Desktop picture grid: the window surface moved
             * up 26 rows in place); dropped, only the newly exposed strip
             * was drawn and the grid filled with slivers of earlier frames.
             * It does not set the header that 0x1B continues from.
             */
            bool single_blt = opcode == 0x92;
            if (single_blt) {
                opcode = 0x9b;
            }
            /*
             * CNTL_HOSTDATA_BLT (0x94): ATI's OpenGL driver builds one with
             * GMC 0x03CC30FA | datatype (host data, SRCCOPY, destination
             * pitch/offset and clip fields, no brush).  The header is the
             * GMC-selected fields of the 2D packets, then DST_Y_X,
             * DST_HEIGHT_WIDTH and the data: rewritten into the layout the
             * 0x9C handler below takes (GMC, DST_PITCH_OFFSET, DST_Y_X,
             * DST_HEIGHT_WIDTH, data).  Not yet seen from a game: the first
             * one is logged with its decoded geometry.
             */
            static uint32_t *hd_buf;
            static uint32_t hd_cap;
            if (opcode == 0x94 && body_dw >= 3) {
                uint32_t gmc = d[0], h = 1, po;
                uint32_t brush = (gmc >> 4) & 0xF, sdt = (gmc >> 12) & 3;
                if (gmc & R200_GMC_SRC_PITCH_OFFSET_CNTL) {
                    h++;
                }
                po = (gmc & R200_GMC_DST_PITCH_OFFSET_CNTL) ? d[h++]
                                                            : s->regs.default_pitch_offset;
                if (gmc & R200_GMC_SRC_CLIPPING) {
                    h++;
                }
                if (gmc & R200_GMC_DST_CLIPPING) {
                    h += 2;
                }
                h += brush == 13 ? 1 : brush < 2 ? 4 : 0;   /* solid / 8x8 mono */
                h += sdt < 2 ? 2 : 0;                       /* mono source: FRGD, BKGD */
                if (body_dw >= h + 2) {
                    uint32_t n = 4 + body_dw - h - 2;
                    static bool hd_logged;
                    if (!hd_logged) {
                        hd_logged = true;
                        qemu_log("ppc-mac-gpu: HOSTDATA_BLT gmc %08x po %08x yx %08x hw %08x, "
                                 "%u data dwords after a %u-dword header\n", gmc, po, d[h],
                                 d[h + 1], body_dw - h - 2, h);
                    }
                    if (n > hd_cap) {
                        hd_cap = n;
                        hd_buf = g_renew(uint32_t, hd_buf, hd_cap);
                    }
                    hd_buf[0] = gmc;
                    hd_buf[1] = po;
                    memcpy(hd_buf + 2, d + h, (body_dw - h) * 4);
                    d = hd_buf;
                    body_dw = n;
                    opcode = 0x9c;
                }
            }
            /* CNTL_PAINT (0x91) is likewise a one-rectangle
             * CNTL_PAINT_MULTI. */
            if (opcode == 0x91) {
                opcode = 0x9a;
            }
            if (opcode == 0x9b && body_dw >= 4) {
                /*
                 * CNTL_BITBLT_MULTI — screen-to-screen blt
                 *
                 * R200 PM4 packet format:
                 *   d[0]  = DP_GUI_MASTER_CNTL
                 *   [SRC_PITCH_OFFSET]  — only if GMC bit 0 is set
                 *   [DST_PITCH_OFFSET]  — only if GMC bit 1 is set
                 *   then groups of 3: SRC_Y_X, DST_Y_X, DST_HEIGHT_WIDTH
                 *
                 * The two pitch-offset words are optional and are emitted in
                 * bit order — source first, then destination.  When a bit is
                 * clear the word is absent and the engine uses the value the
                 * guest last wrote to the matching MMIO register, so the
                 * header is anywhere from 1 to 3 dwords long.  Assuming a
                 * fixed 3-dword header misreads the first blit coordinate as
                 * a pitch-offset and shifts every triplet that follows.
                 */
                uint32_t gmc = d[0];
                uint32_t rop3 = (gmc >> 16) & 0xFF;

                uint32_t hdr_dw = 1;   /* d[0] = GMC */
                r200_note_2d_gaps(s, gmc, "BITBLT_MULTI");
                uint32_t src_po, dst_po;

                /*
                 * A clear control bit does NOT mean "use the SRC_/DST_
                 * PITCH_OFFSET register": it selects DEFAULT_PITCH_OFFSET.
                 * Getting this wrong only shows on the first frame of a
                 * window drag, because from frame 2 the guest happens to
                 * leave DST_PITCH_OFFSET equal to the default — but that
                 * first save lands at a stale pitch and the drag then
                 * copies the damage forward every frame.
                 */
                if (gmc & R200_GMC_SRC_PITCH_OFFSET_CNTL) {
                    src_po = d[hdr_dw++];
                } else {
                    src_po = s->regs.default_pitch_offset;
                }
                if (gmc & R200_GMC_DST_PITCH_OFFSET_CNTL) {
                    dst_po = d[hdr_dw++];
                } else {
                    dst_po = s->regs.default_pitch_offset;
                }
                /* Packet clip fields (GMC bits 2/3) precede the triplets. */
                uint32_t bbm_clip_tl = 0, bbm_clip_br = 0x3FFF3FFF;
                if (gmc & R200_GMC_SRC_CLIPPING) {
                    hdr_dw++;                   /* SRC_SC_BOTTOM_RIGHT */
                }
                if (gmc & R200_GMC_DST_CLIPPING) {
                    bbm_clip_tl = d[hdr_dw++];
                    bbm_clip_br = d[hdr_dw++];
                }

                if (!single_blt && hdr_dw <= ARRAY_SIZE(bbm_hdr)) {
                    memcpy(bbm_hdr, d, hdr_dw * 4);     /* for 0x1B */
                    bbm_hdr_n = hdr_dw;
                }

                /* Need at least one complete (src, dst, size) triplet. */
                if (body_dw < hdr_dw + 3) {
                    i += body_adv;
                    continue;
                }

                uint32_t dst_offset = (dst_po & 0x3FFFFF) << 10;
                uint32_t dst_pitch  = (dst_po & 0x3FC00000) >> 16;
                uint32_t src_offset = (src_po & 0x3FFFFF) << 10;
                uint32_t src_pitch  = (src_po & 0x3FC00000) >> 16;
                /* MC-style tiling: register surfaces from PO if macro_tile=1 */
                if ((dst_po >> 30) & 1) {
                    mc_register_tiled_surface(s, dst_offset, dst_pitch, "BBM_dst");
                }
                if ((src_po >> 30) & 1) {
                    mc_register_tiled_surface(s, src_offset, src_pitch, "BBM_src");
                }
                /* dst_macro_tile/src_macro_tile no longer used — MC handles tiling */
                uint32_t bpp = gmc_dst_bpp(gmc);

                if (dst_pitch == 0 || src_pitch == 0) {
                    i += body_adv;
                    continue;
                }

                uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
                uint32_t num_blits = (body_dw - hdr_dw) / 3;

                seq_log("BBMRAW gmc=%08x clip_tl=%08x clip_br=%08x n=%u "
                        "t0=%08x %08x %08x", gmc, bbm_clip_tl, bbm_clip_br,
                        num_blits, d[hdr_dw], d[hdr_dw + 1], d[hdr_dw + 2]);
                /* Destination clip rectangle from the packet (GMC bit 3). */
                bool use_scissors = (gmc & R200_GMC_DST_CLIPPING) != 0;
                uint32_t sc_left = 0, sc_top = 0;
                uint32_t sc_right = 0x3FFF, sc_bottom = 0x3FFF;
                if (use_scissors) {
                    sc_left   = bbm_clip_tl & 0x3FFF;
                    sc_top    = (bbm_clip_tl >> 16) & 0x3FFF;
                    sc_right  = bbm_clip_br & 0x3FFF;
                    sc_bottom = (bbm_clip_br >> 16) & 0x3FFF;
                }

                /* Path instrumentation for BBM */
                g_blit_stats.bbm_count++;
                {
                    bool raw_src_mt = (src_po >> 30) & 1;
                    bool raw_dst_mt = (dst_po >> 30) & 1;
                    static uint64_t bbm_logged = 0;
                    if (bbm_logged < 500) {
                        const char *dst_region = "offscreen";
                        if (dst_offset == 0 && (dst_pitch == 4096 || dst_pitch == 3200))
                            dst_region = "FRAMEBUFFER";
                        else if (dst_offset >= 0x200000)
                            dst_region = "backbuf";
                        blit_path_log("BBM",
                            "rop=0x%02x n=%u  "
                            "src_po=0x%08x[off=0x%06x pitch=%u mt=%d]  "
                            "dst_po=0x%08x[off=0x%06x pitch=%u mt=%d]  "
                            "dst=%s",
                            rop3, num_blits,
                            src_po, src_offset, src_pitch, raw_src_mt,
                            dst_po, dst_offset, dst_pitch, raw_dst_mt,
                            dst_region);
                        bbm_logged++;
                    }
                }

                for (uint32_t b = 0; b < num_blits; b++) {
                    uint32_t src_xy = d[hdr_dw + b * 3 + 0];
                    uint32_t dst_xy = d[hdr_dw + b * 3 + 1];
                    uint32_t dwh    = d[hdr_dw + b * 3 + 2];

                    /*
                     * CNTL_BITBLT_MULTI triplets use X_Y / WIDTH_HEIGHT:
                     *   X in bits[31:16], Y in bits[15:0]
                     *   W in bits[31:16], H in bits[15:0]
                     *
                     * Note this is the OPPOSITE of the MMIO registers
                     * SRC_Y_X / DST_Y_X / DST_HEIGHT_WIDTH.  The packet
                     * writes the contiguous 0x1590 block (SRC_X_Y,
                     * DST_X_Y, DST_WIDTH_HEIGHT), not the 0x1434 block.
                     *
                     * Verified against captured guest packets: under this
                     * decode all 10 sampled blits fit both the scissor
                     * rectangle and the 1024x768 source framebuffer, while
                     * the Y_X reading put every one of them out of bounds
                     * (e.g. a 617-row read from a 768-row buffer starting
                     * at y=266).
                     */
                    uint32_t src_x = (src_xy >> 16) & 0xFFFF;
                    uint32_t src_y = src_xy & 0xFFFF;
                    uint32_t dst_x = (dst_xy >> 16) & 0xFFFF;
                    uint32_t dst_y = dst_xy & 0xFFFF;
                    uint32_t blit_w = (dwh >> 16) & 0x3FFF;
                    uint32_t blit_h = dwh & 0x3FFF;

                    if (blit_w == 0 || blit_h == 0) {
                        continue;
                    }

                    /* Apply scissor clipping to destination */
                    if (use_scissors) {
                        uint32_t dst_x2 = dst_x + blit_w;
                        uint32_t dst_y2 = dst_y + blit_h;

                        if (dst_x2 <= sc_left || dst_x >= sc_right ||
                            dst_y2 <= sc_top  || dst_y >= sc_bottom) {
                            continue; /* Entirely clipped */
                        }

                        if (dst_x < sc_left) {
                            uint32_t clip = sc_left - dst_x;
                            src_x += clip;
                            dst_x = sc_left;
                            blit_w -= clip;
                        }
                        if (dst_y < sc_top) {
                            uint32_t clip = sc_top - dst_y;
                            src_y += clip;
                            dst_y = sc_top;
                            blit_h -= clip;
                        }
                        if (dst_x + blit_w > sc_right) {
                            blit_w = sc_right - dst_x;
                        }
                        if (dst_y + blit_h > sc_bottom) {
                            blit_h = sc_bottom - dst_y;
                        }

                        if (blit_w == 0 || blit_h == 0) {
                            continue;
                        }
                    }

                    r200_vram_access(s, R200_ROWS(dst_offset, dst_pitch, dst_y,
                                                  blit_h), true, 3);
                    r200_vram_access(s, R200_ROWS(src_offset, src_pitch, src_y,
                                                  blit_h), false, 3);
                    seq_log("BBM  rop=%02x src=%06x/%u (%u,%u) "
                            "dst=%06x/%u (%u,%u) %ux%u",
                            rop3, src_offset, src_pitch, src_x, src_y,
                            dst_offset, dst_pitch, dst_x, dst_y,
                            blit_w, blit_h);

                    /* Record BLIT region for 3D coordinate mapping.
                     * Record all SRCCOPY BLITs from off-screen RTs.
                     * This captures the RT→screen mapping that the 3D
                     * renderer needs for coordinate transforms. */
                    if (rop3 == 0xCC && src_offset >= 0x100000) {
                        blit_region_record(src_offset, src_pitch,
                                           src_x, src_y,
                                           dst_x, dst_y,
                                           blit_w, blit_h);
                    }

                    if (rop3 == 0xCC) {
                        /* SRCCOPY — separate src/dst offsets and pitches */
                        bool handled_by_renderer = false;

                        /* Try shadow RT redirect first: if the 3D renderer
                         * has an active shadow RT covering this BLIT source,
                         * read pixels from the shadow buffer instead of VRAM.
                         * This is critical for Quartz Extreme compositing where
                         * 3D writes go to the shadow (keeping VRAM clean for
                         * texture reads) and the 2D BLIT engine picks up
                         * composited content from the shadow. */
                        if (!r200_direct_enabled() && s->renderer && s->renderer->blit_2d) {
                            PPCMacGPUBlit blit_desc = {
                                .src_x = src_x, .src_y = src_y,
                                .src_offset = src_offset,
                                .src_pitch = src_pitch,
                                .dst_x = dst_x, .dst_y = dst_y,
                                .dst_offset = dst_offset,
                                .dst_pitch = dst_pitch,
                                .width = blit_w, .height = blit_h,
                                .rop3 = rop3,
                                .bpp = bpp * 8,
                            };
                            int blit_ret = s->renderer->blit_2d(
                                s->renderer_opaque, vram, &blit_desc);
                            if (blit_ret == 0) {
                                /* Shadow RT handled the BLIT */
                                handled_by_renderer = true;
                                if (dst_offset <= 0x10000) {
                                    fb_write_watch("bbm_srt_present", "srt",
                                                   src_offset, dst_offset,
                                                   dst_x, dst_y,
                                                   blit_w, blit_h,
                                                   true, true);
                                }
                                goto blit_mark_dirty;
                            }
                        }

                        /* Log non-SRT BBM writes to framebuffer */
                        if (dst_offset <= 0x10000 && blit_w >= 4 && blit_h >= 4) {
                            bool is_gart_src = (s->regs.aic_ctrl & 1) &&
                                src_offset >= s->regs.aic_lo_addr &&
                                src_offset <= s->regs.aic_hi_addr;
                            hwaddr test_phys;
                            bool is_agp_src = !is_gart_src &&
                                ppc_mac_gpu_agp_translate(s, src_offset,
                                                          &test_phys);
                            fb_write_watch("bbm_direct",
                                           is_gart_src ? "gart" :
                                           is_agp_src ? "agp" : "vram",
                                           src_offset, dst_offset,
                                           dst_x, dst_y,
                                           blit_w, blit_h,
                                           false, true);
                        }

                        bool bbm_src_gart = (s->regs.aic_ctrl & 1) &&
                                            src_offset >= s->regs.aic_lo_addr &&
                                            src_offset <= s->regs.aic_hi_addr;
                        /* Also check AGP aperture (MC_AGP_LOCATION) for
                         * window backing stores that live in system RAM.
                         * During window drag, the compositor BLITs window
                         * body from AGP-mapped system RAM to framebuffer. */
                        bool bbm_src_agp = false;
                        if (!bbm_src_gart) {
                            hwaddr test_phys;
                            bbm_src_agp = ppc_mac_gpu_agp_translate(
                                s, src_offset, &test_phys);
                        }

                        if (bbm_src_gart || bbm_src_agp) {
                            /* Source in GART/AGP: read from system RAM,
                             * write to VRAM via MC helper. */
                            for (uint32_t row = 0; row < blit_h; row++) {
                                for (uint32_t col = 0; col < blit_w; col++) {
                                    uint32_t gpu_addr = src_offset +
                                        (src_y + row) * src_pitch +
                                        (src_x + col) * bpp;
                                    hwaddr phys;
                                    uint32_t pixel = 0;
                                    if (ppc_mac_gpu_agp_translate(s, gpu_addr, &phys) ||
                                        ppc_mac_gpu_gart_translate(s, gpu_addr, &phys)) {
                                        address_space_read(
                                            &address_space_memory, phys,
                                            MEMTXATTRS_UNSPECIFIED,
                                            &pixel, 4);
                                    }
                                    uint64_t d_linear = (uint64_t)dst_offset +
                                        (uint64_t)(dst_y + row) * dst_pitch +
                                        (uint64_t)(dst_x + col) * bpp;
                                    mc_vram_write32(s, vram, d_linear, pixel);
                                }
                            }
                        } else {
                            /*
                             * VRAM → VRAM via MC helpers.
                             *
                             * Source and destination may overlap — a window
                             * drag is precisely an overlapping screen-to-
                             * screen copy — so walk the rectangle away from
                             * the overlap, the way memmove picks a direction.
                             * Copying forward when the destination is below
                             * or to the right of the source would re-read
                             * pixels this same blit has already overwritten,
                             * smearing the dragged content.
                             */
                            bool rev_rows = false, rev_cols = false;
                            if (blit_overlaps(src_offset, src_pitch,
                                              src_x, src_y,
                                              dst_offset, dst_pitch,
                                              dst_x, dst_y,
                                              blit_w, blit_h, bpp)) {
                                rev_rows = dst_y > src_y;
                                rev_cols = (dst_y == src_y) && (dst_x > src_x);
                            }

                            if (blit_rect_untiled(s, vram,
                                                  src_offset, src_pitch,
                                                  src_x, src_y,
                                                  dst_offset, dst_pitch,
                                                  dst_x, dst_y,
                                                  blit_w, blit_h, bpp)) {
                                goto blit_mark_dirty;
                            }

                            for (uint32_t r = 0; r < blit_h; r++) {
                                uint32_t row = rev_rows ? (blit_h - 1 - r) : r;
                                for (uint32_t c = 0; c < blit_w; c++) {
                                    uint32_t col = rev_cols ? (blit_w - 1 - c) : c;
                                    uint64_t s_linear = (uint64_t)src_offset +
                                        (uint64_t)(src_y + row) * src_pitch +
                                        (uint64_t)(src_x + col) * bpp;
                                    uint64_t d_linear = (uint64_t)dst_offset +
                                        (uint64_t)(dst_y + row) * dst_pitch +
                                        (uint64_t)(dst_x + col) * bpp;
                                    uint32_t pixel = mc_vram_read32(s, vram, s_linear);
                                    mc_vram_write32(s, vram, d_linear, pixel);
                                }
                            }
                        }
                    blit_mark_dirty:
                        /* Mark dirty */
                        {
                        uint64_t dirty_start = (uint64_t)dst_offset +
                            (uint64_t)dst_y * dst_pitch +
                            (uint64_t)dst_x * bpp;
                        uint64_t dirty_len = (blit_h > 1)
                            ? (uint64_t)(blit_h - 1) * dst_pitch +
                              (uint64_t)blit_w * bpp
                            : (uint64_t)blit_w * bpp;
                        if (dirty_start + dirty_len <= s->vram_size) {
                            ppc_vram_dirty_e0(s,
                                                    dirty_start, dirty_len);
                        }
                        }

                        /* SRT write-through for BBM path */
                        if (!r200_direct_enabled() && s->renderer && s->renderer->srt_write_through &&
                            dst_offset != s->regs.crtc_offset) {
                            s->renderer->srt_write_through(
                                s->renderer_opaque, vram,
                                dst_offset, dst_pitch,
                                dst_x, dst_y, blit_w, blit_h,
                                bpp * 8);
                        }
                        if (!handled_by_renderer) {
                            if (dst_offset == s->regs.crtc_offset) {
                                frame_tracker_record_2d_event(
                                    PASS_EVENT_FALLBACK_2D,
                                    src_offset, src_pitch,
                                    src_offset, dst_offset,
                                    src_x, src_y,
                                    dst_x, dst_y,
                                    blit_w, blit_h,
                                    false);
                            } else {
                                frame_tracker_record_2d_event(
                                    PASS_EVENT_BLIT_UPLOAD,
                                    dst_offset, dst_pitch,
                                    src_offset, dst_offset,
                                    src_x, src_y,
                                    dst_x, dst_y,
                                    blit_w, blit_h,
                                    false);
                            }
                        }

                        /*
                         * Phase A: [PRESENT_BLIT] logging for visible
                         * framebuffer writes.
                         * Phase C: Stride override activation.
                         */
                        if (dst_offset == s->regs.crtc_offset &&
                            rop3 == 0xCC &&
                            blit_w >= 256 && blit_h >= 256) {
                            /* Phase D: Check if BLT source was a recent
                             * Metal render target */
                            bool src_rendered = false;
                            for (int rt = 0; rt < s->metal_rt_count; rt++) {
                                if (s->metal_rt_offsets[rt] == src_offset) {
                                    src_rendered = true;
                                    break;
                                }
                            }
                            static int present_blit_log = 0;
                            if (present_blit_log < 50) {
                                fprintf(stderr,
                                    "[PRESENT_BLIT] PM4_BBM: "
                                    "src=0x%x+%u dst=0x%x+%u "
                                    "sxy=%u,%u dxy=%u,%u %ux%u "
                                    "crtc_stride=%u\n",
                                    src_offset, src_pitch,
                                    dst_offset, dst_pitch,
                                    src_x, src_y, dst_x, dst_y,
                                    blit_w, blit_h,
                                    s->disp.stride);
                                fprintf(stderr,
                                    "[PRESENT_SOURCE] src_off=0x%x "
                                    "dst_off=0x%x size=(%u,%u) "
                                    "rendered_this_frame=%s "
                                    "source_kind=%s "
                                    "matches_latest_rt=%s "
                                    "note=PM4_BBM_visible_blit\n",
                                    src_offset, dst_offset,
                                    blit_w, blit_h,
                                    src_rendered ? "yes" : "no",
                                    src_rendered ? "metal_result" : "raw_vram",
                                    src_rendered ? "yes" : "no");
                                present_blit_log++;
                            }
                            /* Activate stride override if BLT pitch
                             * differs from current display stride */
                            if (dst_pitch > 0 && s->disp.bpp == 32) {
                                if (!s->disp_stride_override_active ||
                                    s->disp_stride_override_value != dst_pitch) {
                                    static int so_log = 0;
                                    if (so_log < 20) {
                                        if (ppcgpu_stride_diag()) fprintf(stderr,
                                            "[STRIDE_CHANGE] PM4_BBM: "
                                            "override %s %u -> %u\n",
                                            s->disp_stride_override_active
                                                ? "update" : "activate",
                                            s->disp_stride_override_active
                                                ? s->disp_stride_override_value
                                                : s->disp.stride,
                                            dst_pitch);
                                        so_log++;
                                    }
                                }
                                r200_set_present_pitch(s, dst_pitch);
                            }
                        }
                    }
                    /* Other ROP3 values: ignore for now */
                }
            } else if (opcode == 0x9a && body_dw >= 3) {
                /*
                 * CNTL_PAINT_MULTI — solid fill
                 *
                 * Packet layout depends on GMC flags:
                 * - GMC bit 1 (DST_PITCH_OFFSET_CNTL):
                 *   0 = DST pitch/offset from MMIO regs
                 *   1 = DST_PITCH_OFFSET in packet
                 * - GMC bits 7:4 (brush type):
                 *   0xD = solid color (color word in packet)
                 *
                 * Format when bit1=0, solid brush:
                 *   d[0]=GMC, d[1]=COLOR, d[2..]=fill pairs
                 * Format when bit1=1, solid brush:
                 *   d[0]=GMC, d[1]=PITCH_OFFSET, d[2]=COLOR,
                 *   d[3..]=fill pairs
                 */
                uint32_t gmc = d[0];
                uint32_t rop3 = (gmc >> 16) & 0xFF;
                bool has_po = (gmc >> 1) & 1;
                r200_note_2d_gaps(s, gmc, "PAINT_MULTI");
                uint32_t bpp = gmc_dst_bpp(gmc);
                uint32_t idx = 1;

                uint32_t offset, pitch;
                if (has_po) {
                    uint32_t po = d[idx++];
                    offset = (po & 0x3FFFFF) << 10;
                    pitch  = (po & 0x3FC00000) >> 16;
                    /* MC-style: register tiled surface if macro_tile=1 */
                    if ((po >> 30) & 1) {
                        mc_register_tiled_surface(s, offset, pitch, "PAINT_PO");
                    }
                } else {
                    /* GMC_DST_PITCH_OFFSET_CNTL clear: DEFAULT_PITCH_OFFSET,
                     * not the separate DST_OFFSET/DST_PITCH registers. */
                    uint32_t po = s->regs.default_pitch_offset;
                    offset = (po & 0x3FFFFF) << 10;
                    pitch  = (po & 0x3FC00000) >> 16;
                }

                /* Packet clip fields (GMC bits 2/3) come before the brush. */
                uint32_t pm_clip_x0 = 0, pm_clip_y0 = 0;
                uint32_t pm_clip_x1 = 0x3FFF, pm_clip_y1 = 0x3FFF;
                if (gmc & R200_GMC_SRC_CLIPPING) {
                    idx++;                      /* SRC_SC_BOTTOM_RIGHT */
                }
                if ((gmc & R200_GMC_DST_CLIPPING) && idx + 2 <= body_dw) {
                    uint32_t tl = d[idx++], br = d[idx++];
                    pm_clip_x0 = tl & 0x3FFF;
                    pm_clip_y0 = (tl >> 16) & 0x3FFF;
                    pm_clip_x1 = br & 0x3FFF;          /* exclusive */
                    pm_clip_y1 = (br >> 16) & 0x3FFF;
                }

                /* Brush type in bits 7:4 — 0xD = solid color */
                uint32_t brush_type = (gmc >> 4) & 0xF;
                uint32_t color;
                if (brush_type == 0xD) {
                    color = d[idx++];
                } else {
                    color = s->regs.dp_brush_frgd_clr;
                }

                if (pitch == 0) {
                    i += body_adv;
                    continue;
                }

                uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
                uint32_t num_fills = (body_dw - idx) / 2;

                /* Path instrumentation for PAINT */
                g_blit_stats.paint_count++;
                {
                    static uint64_t paint_logged = 0;
                    if (paint_logged < 500) {
                        const char *dst_region = "offscreen";
                        if (offset == 0 && (pitch == 4096 || pitch == 3200))
                            dst_region = "FRAMEBUFFER";
                        else if (offset >= 0x200000)
                            dst_region = "backbuf";
                        blit_path_log("PAINT",
                            "rop=0x%02x color=0x%08x  "
                            "off=0x%06x pitch=%u has_po=%d  "
                            "fills=%u  dst=%s",
                            rop3, color, offset, pitch, has_po,
                            num_fills, dst_region);
                        paint_logged++;
                    }
                }

                for (uint32_t f = 0; f < num_fills; f++) {
                    uint32_t dst_yx = d[idx + f * 2 + 0];
                    uint32_t dwh    = d[idx + f * 2 + 1];

                    /* X_Y format: X in bits[31:16], Y in bits[15:0] */
                    uint32_t dst_x = (dst_yx >> 16) & 0xFFFF;
                    uint32_t dst_y = dst_yx & 0xFFFF;
                    /* W_H format: W in bits[31:16], H in bits[15:0] */
                    uint32_t blit_w = (dwh >> 16) & 0x3FFF;
                    uint32_t blit_h = dwh & 0x3FFF;

                    if (blit_w == 0 || blit_h == 0) {
                        continue;
                    }
                    /* Clip to the packet clip rectangle and the surface. */
                    uint32_t cx0 = MAX(dst_x, pm_clip_x0);
                    uint32_t cy0 = MAX(dst_y, pm_clip_y0);
                    uint32_t cx1 = MIN(dst_x + blit_w, MIN(pm_clip_x1, pitch / bpp));
                    uint32_t cy1 = MIN(dst_y + blit_h, pm_clip_y1);
                    if (cx0 >= cx1 || cy0 >= cy1) {
                        continue;
                    }
                    dst_x = cx0;
                    dst_y = cy0;
                    blit_w = cx1 - cx0;
                    blit_h = cy1 - cy0;
                    if ((uint64_t)offset + (uint64_t)(dst_y + blit_h) * pitch >
                        s->vram_size) {
                        continue;
                    }

                    r200_vram_access(s, R200_ROWS(offset, pitch, dst_y, blit_h),
                                     true, 4);
                    seq_log("FILL rop=%02x dst=%06x/%u (%u,%u) %ux%u "
                            "color=%08x",
                            rop3, offset, pitch, dst_x, dst_y,
                            blit_w, blit_h, color);

                    if (rop3 == 0xF0 || rop3 == 0xCC) {
                        if (offset <= 0x10000) {
                            fb_write_watch("paint_fill", "solid",
                                           0, offset,
                                           dst_x, dst_y, blit_w, blit_h,
                                           false, color != 0);
                        }
                        /* MC helper handles tiling transparently */
                        for (uint32_t row = 0; row < blit_h; row++) {
                            for (uint32_t col = 0; col < blit_w; col++) {
                                uint64_t d_linear = (uint64_t)offset +
                                    (uint64_t)(dst_y + row) * pitch +
                                    (uint64_t)(dst_x + col) * bpp;
                                fill_vram_px(s, vram, d_linear, color, bpp);
                            }
                        }
                        uint64_t dirty_start = (uint64_t)offset +
                            (uint64_t)dst_y * pitch +
                            (uint64_t)dst_x * bpp;
                        uint64_t dirty_len = (blit_h > 1)
                            ? (uint64_t)(blit_h - 1) * pitch +
                              (uint64_t)blit_w * bpp
                            : (uint64_t)blit_w * bpp;
                        if (dirty_start + dirty_len <= s->vram_size) {
                            ppc_vram_dirty_e0(s,
                                                    dirty_start, dirty_len);
                        }
                        r200_fill_notify(s, offset, pitch, dst_x, dst_y,
                                         blit_w, blit_h, bpp, color);
                    }
                }
            } else if (opcode == 0x9c && body_dw >= 4) {
                /*
                 * CNTL_HOSTDATA_BLT — CPU-to-VRAM blit via PM4
                 * d[0] = DP_GUI_MASTER_CNTL (with src_type=3 for HOST)
                 * d[1] = DST PITCH_OFFSET
                 * d[2] = DST_Y_X
                 * d[3] = DST_WIDTH_HEIGHT
                 * d[4..] = inline pixel data (host data words)
                 */
                uint32_t gmc = d[0];
                uint32_t po = d[1];
                uint32_t dst_offset = (po & 0x3FFFFF) << 10;
                uint32_t dst_pitch  = (po & 0x3FC00000) >> 16;
                /* MC-style: register tiled surface if macro_tile=1 */
                if ((po >> 30) & 1) {
                    mc_register_tiled_surface(s, dst_offset, dst_pitch, "HD_PO");
                }
                uint32_t dst_yx = d[2];
                /* DST_Y_X format: Y in bits[31:16], X in bits[15:0] */
                uint32_t dst_x = dst_yx & 0xFFFF;
                uint32_t dst_y = (dst_yx >> 16) & 0xFFFF;
                uint32_t dwh = d[3];
                /* DST_HEIGHT_WIDTH: H in bits[31:16], W in bits[15:0] */
                uint32_t blit_w = dwh & 0x3FFF;
                uint32_t blit_h = (dwh >> 16) & 0x3FFF;

                {   /* geometry order is ambiguous here: cover both */
                    uint32_t ymax = MAX(dst_yx & 0xFFFF, (dst_yx >> 16) & 0xFFFF);
                    uint32_t hmax = MAX(dwh & 0x3FFF, (dwh >> 16) & 0x3FFF);
                    r200_vram_access(s, R200_ROWS(dst_offset, dst_pitch, 0,
                                                  ymax + hmax), true, 5);
                }
                seq_log("HOST dst=%06x/%u raw_yx=%08x raw_wh=%08x "
                        "asYX_HW=(%u,%u) %ux%u  asXY_WH=(%u,%u) %ux%u "
                        "data=%u dw",
                        dst_offset, dst_pitch, dst_yx, dwh,
                        dst_yx & 0xFFFF, (dst_yx >> 16) & 0xFFFF,
                        dwh & 0x3FFF, (dwh >> 16) & 0x3FFF,
                        (dst_yx >> 16) & 0xFFFF, dst_yx & 0xFFFF,
                        (dwh >> 16) & 0x3FFF, dwh & 0x3FFF,
                        body_dw - 4);

                /* Path instrumentation for HOSTDATA */
                g_blit_stats.hostdata_count++;
                {
                    static uint64_t hd_logged = 0;
                    if (hd_logged < 500) {
                        const char *dst_region = "offscreen";
                        if (dst_offset == 0 && (dst_pitch == 4096 || dst_pitch == 3200))
                            dst_region = "FRAMEBUFFER";
                        else if (dst_offset >= 0x200000)
                            dst_region = "backbuf";
                        blit_path_log("HD",
                            "off=0x%06x pitch=%u  "
                            "xy=(%u,%u) %ux%u  data=%u dw  dst=%s",
                            dst_offset, dst_pitch,
                            dst_x, dst_y, blit_w, blit_h,
                            body_dw - 4, dst_region);
                        hd_logged++;
                    }
                }

                if (blit_w > 0 && blit_h > 0 && dst_pitch > 0) {
                    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
                    uint32_t bpp = gmc_dst_bpp(gmc);
                    uint32_t pix_idx = 0;
                    uint32_t data_dw = body_dw - 4;

                    if (bpp < 4) {
                        uint32_t cx = 0, cy = 0;
                        for (uint32_t dw = 0; dw < data_dw; dw++) {
                            if (!host_data_put_narrow(s, vram, d[4 + dw], bpp,
                                                      dst_offset, dst_pitch,
                                                      dst_x, dst_y, blit_w,
                                                      blit_h, &cx, &cy)) {
                                break;
                            }
                        }
                        /* continue in HOST_DATA writes if the rect is open */
                        pix_idx = cy >= blit_h ? blit_w * blit_h
                                               : cy * blit_w + cx;
                        data_dw = 0;
                    }
                    for (uint32_t dw = 0; dw < data_dw; dw++) {
                        uint32_t cx = pix_idx % blit_w;
                        uint32_t cy = pix_idx / blit_w;
                        if (cy >= blit_h) break;

                        /* MC helper handles tiling transparently */
                        uint64_t d_linear = (uint64_t)dst_offset +
                            (uint64_t)(dst_y + cy) * dst_pitch +
                            (uint64_t)(dst_x + cx) * bpp;
                        mc_vram_write32(s, vram, d_linear, d[4 + dw]);
                        pix_idx++;
                    }

                    /* If inline data didn't fill the rect, set up host_data
                     * state so subsequent HOST_DATA writes continue */
                    if (pix_idx < (uint32_t)blit_w * blit_h) {
                        s->host_data_offset = dst_offset;
                        s->host_data_pitch = dst_pitch;
                        s->host_data_macro_tile = false;  /* MC handles tiling */
                        s->host_data_dst_x = dst_x;
                        s->host_data_dst_y = dst_y;
                        s->host_data_w = blit_w;
                        s->host_data_h = blit_h;
                        s->host_data_cur_x = pix_idx % blit_w;
                        s->host_data_cur_y = pix_idx / blit_w;
                        s->host_data_bpp = bpp;
                        s->host_data_active = true;
                    } else {
                        /* All inline data written, mark dirty */
                        uint64_t dirty_start = (uint64_t)dst_offset +
                            (uint64_t)dst_y * dst_pitch +
                            (uint64_t)dst_x * bpp;
                        uint64_t dirty_len = (blit_h > 1)
                            ? (uint64_t)(blit_h - 1) * dst_pitch +
                              (uint64_t)blit_w * bpp
                            : (uint64_t)blit_w * bpp;
                        if (dirty_start + dirty_len <= s->vram_size) {
                            ppc_vram_dirty_e0(s,
                                                    dirty_start, dirty_len);
                        }
                        s->display_invalid = true;
                    }
                }
            } else if (opcode == R200_3D_LOAD_VBPNTR && body_dw >= 1) {
                /*
                 * 3D_LOAD_VBPNTR — Load vertex buffer pointers
                 * d[0] = number of vertex arrays
                 * Then pairs: (size_dword, address_dword) for each array
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t num_arrays = d[0];
                if (num_arrays > 4) num_arrays = 4;
                s->vb_count = num_arrays;
                gpu_debug_log("3D_LOAD_VBPNTR: %u arrays", num_arrays);
                for (uint32_t a = 0; a < num_arrays && (1 + a * 2 + 1) < body_dw; a++) {
                    s->vb_stride[a] = d[1 + a * 2] & 0xFF;
                    s->vb_addr[a] = d[1 + a * 2 + 1];
                    gpu_debug_log("  VB[%u]: stride=%u addr=0x%08x",
                                  a, s->vb_stride[a], s->vb_addr[a]);
                }

            } else if (opcode == R200_3D_DRAW_VBUF && body_dw >= 1) {
                /*
                 * 3D_DRAW_VBUF — Draw from vertex buffer
                 * d[0]: bits[3:0]=prim_type, bits[15:4]=num_verts
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t prim_type = d[0] & 0xF;
                uint32_t num_verts = (d[0] >> 4) & 0xFFF;
                gpu_debug_log("3D_DRAW_VBUF: prim=%u verts=%u",
                              prim_type, num_verts);

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_VBUF,
                    .prim_type = prim_type,
                    .num_vertices = num_verts,
                    .vb_addr = (s->vb_count > 0) ? s->vb_addr[0] : 0,
                    .vb_stride = (s->vb_count > 0) ? s->vb_stride[0] : 0,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_IMMD && body_dw >= 1) {
                /*
                 * 3D_DRAW_IMMD — Draw with inline vertex data
                 * d[0]: bits[3:0]=prim_type, bits[15:4]=num_verts
                 * d[1..N]: inline vertex data
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t prim_type = d[0] & 0xF;
                uint32_t num_verts = (d[0] >> 4) & 0xFFF;
                gpu_debug_log("3D_DRAW_IMMD: prim=%u verts=%u data_dw=%u",
                              prim_type, num_verts, body_dw - 1);

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_IMMD,
                    .prim_type = prim_type,
                    .num_vertices = num_verts,
                    .vertex_data = (body_dw > 1) ? &d[1] : NULL,
                    .vertex_data_dwords = (body_dw > 1) ? body_dw - 1 : 0,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_INDX && body_dw >= 1) {
                /*
                 * 3D_DRAW_INDX — Draw with index buffer
                 * d[0]: bits[3:0]=prim_type, bits[15:4]=num_indices
                 * d[1..N]: index data (packed u16 pairs)
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t prim_type = d[0] & 0xF;
                uint32_t num_idx = (d[0] >> 4) & 0xFFF;
                gpu_debug_log("3D_DRAW_INDX: prim=%u indices=%u",
                              prim_type, num_idx);

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_INDX,
                    .prim_type = prim_type,
                    .num_vertices = num_idx,
                    .num_indices = num_idx,
                    .index_data = (body_dw > 1) ? (const uint16_t *)&d[1] : NULL,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_VBUF_2_ALT && body_dw >= 1 &&
                       ppc_mac_gpu_r200_draw(s, d, body_dw, R200_SRC_VBUF)) {
                /* Rendered by the direct R200 path. */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
            } else if (opcode == 0x29 && body_dw >= 2 &&
                       ppc_mac_gpu_r200_draw(s, d + 1, body_dw - 1,
                                             R200_SRC_IMMD)) {
                /*
                 * 3D_DRAW_IMMD (R100-style): d[0] is a vertex format word the
                 * R200 ignores (layout comes from SE_VTX_FMT_0/1), d[1] is
                 * VF_CNTL, vertices follow.  Apple's GL driver uses it for
                 * the buffer swap: a RECT_LIST that resolves the 2x-tall
                 * supersampled back buffer into the window surface.
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
            } else if ((opcode == 0x35 || opcode == 0x36) && body_dw >= 1 &&
                       ppc_mac_gpu_r200_draw(s, d, body_dw, opcode == 0x35 ?
                                             R200_SRC_IMMD : R200_SRC_INDX)) {
                /* 3D_DRAW_IMMD_2 / 3D_DRAW_INDX_2, direct R200 path. */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
            } else if (opcode == R200_3D_DRAW_VBUF_2_ALT && body_dw >= 1) {
                /*
                 * R200 3D_DRAW_VBUF_2 (opcode 0x34)
                 * Body dword = SE_VF_CNTL value:
                 *   bits[3:0]   = VF_PRIM_TYPE
                 *   bits[5:4]   = VF_PRIM_WALK (2=list for VBUF)
                 *   bits[15:6]  = flags (EN_MAOS, etc.)
                 *   bits[31:16] = NUM_VERTICES
                 *
                 * Vertex buffer addresses come from AOS registers
                 * (0x20C4-0x20D8), not from LOAD_VBPNTR.
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t vf_cntl = d[0];
                uint32_t prim_type = vf_cntl & R200_VF_PRIM_TYPE_MASK;
                uint32_t prim_walk = (vf_cntl & R200_VF_PRIM_WALK_MASK)
                                     >> R200_VF_PRIM_WALK_SHIFT;
                uint32_t num_verts = (vf_cntl & R200_VF_NUM_VERTICES_MASK)
                                     >> R200_VF_NUM_VERTICES_SHIFT;

                /* Read AOS vertex buffer from 3D register shadow */
                uint32_t aos_desc0 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_DESC_0)];
                uint32_t aos_addr0 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_ADDR_0)];
                /* AOS descriptor: comp[7:0] | interleaved_stride[15:8] */
                uint32_t aos_stride0 = (aos_desc0 >> 8) & 0xFF;  /* in dwords */

                /* Read all AOS addresses */
                uint32_t aos_addr1 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_ADDR_1)];
                uint32_t aos_desc1 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_DESC_1)];
                uint32_t a0_size = aos_desc0 & 0xFF;
                uint32_t a1_size = (aos_desc0 >> 16) & 0xFF;
                uint32_t a1_stride = (aos_desc0 >> 24) & 0xFF;

                gpu_debug_log("3D_DRAW_VBUF_2: vf_cntl=0x%08x prim=%u walk=%u "
                              "verts=%u aos_addr0=0x%08x aos_addr1=0x%08x "
                              "aos_stride=%u a0_size=%u a1_size=%u",
                              vf_cntl, prim_type, prim_walk,
                              num_verts, aos_addr0, aos_addr1,
                              aos_stride0, a0_size, a1_size);

                /* Dump AOS layout for first few draws */
                {
                    static int aos_dump_count = 0;
                    if (aos_dump_count < 5) {
                        fprintf(stderr, "[AOS_DUMP] draw vf=0x%08x "
                                "desc0=0x%08x desc1=0x%08x "
                                "arr0: addr=0x%08x size=%u stride=%u "
                                "arr1: addr=0x%08x size=%u stride=%u "
                                "vtx_fmt0=0x%08x vtx_fmt1=0x%08x\n",
                                vf_cntl, aos_desc0, aos_desc1,
                                aos_addr0, a0_size, aos_stride0,
                                aos_addr1, a1_size, a1_stride,
                                s->regs.regs_3d[R200_3D_IDX(R200_SE_VTX_FMT_0)],
                                s->regs.regs_3d[R200_3D_IDX(R200_SE_VTX_FMT_1)]);
                        aos_dump_count++;
                    }
                }

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_VBUF,  /* treat as DRAW_VBUF */
                    .prim_type = prim_type,
                    .num_vertices = num_verts,
                    .vb_addr = aos_addr0,
                    .vb_stride = aos_stride0 * 4,  /* convert dwords to bytes */
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_VBUF_2 && body_dw >= 1) {
                /*
                 * R200 3D_DRAW_VBUF_2 (opcode 0x23) — same VF_CNTL format
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t vf_cntl = d[0];
                uint32_t prim_type = vf_cntl & R200_VF_PRIM_TYPE_MASK;
                uint32_t num_verts = (vf_cntl & R200_VF_NUM_VERTICES_MASK)
                                     >> R200_VF_NUM_VERTICES_SHIFT;

                uint32_t aos_desc0 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_DESC_0)];
                uint32_t aos_addr0 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_ADDR_0)];
                uint32_t aos_addr1 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_ADDR_1)];
                uint32_t aos_desc1 = s->regs.regs_3d[R200_3D_IDX(R200_AOS_DESC_1)];
                uint32_t aos_stride0 = (aos_desc0 >> 8) & 0xFF;  /* interleaved stride in dwords */

                /* Dump AOS layout for first few draws */
                {
                    static int aos_dump_count = 0;
                    if (aos_dump_count < 5) {
                        uint32_t a0_size = aos_desc0 & 0xFF;
                        uint32_t a0_stride = (aos_desc0 >> 8) & 0xFF;
                        uint32_t a1_size = (aos_desc0 >> 16) & 0xFF;
                        uint32_t a1_stride = (aos_desc0 >> 24) & 0xFF;
                        fprintf(stderr, "[AOS_DUMP] draw vf=0x%08x "
                                "desc0=0x%08x desc1=0x%08x "
                                "arr0: addr=0x%08x size=%u stride=%u "
                                "arr1: addr=0x%08x size=%u stride=%u "
                                "vtx_fmt0=0x%08x vtx_fmt1=0x%08x\n",
                                vf_cntl, aos_desc0, aos_desc1,
                                aos_addr0, a0_size, a0_stride,
                                aos_addr1, a1_size, a1_stride,
                                s->regs.regs_3d[R200_3D_IDX(R200_SE_VTX_FMT_0)],
                                s->regs.regs_3d[R200_3D_IDX(R200_SE_VTX_FMT_1)]);
                        aos_dump_count++;
                    }
                }

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_VBUF,
                    .prim_type = prim_type,
                    .num_vertices = num_verts,
                    .vb_addr = aos_addr0,
                    .vb_stride = aos_stride0 * 4,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_IMMD_2 && body_dw >= 1) {
                /*
                 * R200 3D_DRAW_IMMD_2 (opcode 0x24)
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t prim_type = d[0] & 0xF;
                uint32_t num_verts = (d[0] >> 4) & 0xFFF;
                gpu_debug_log("3D_DRAW_IMMD_2: prim=%u verts=%u data_dw=%u",
                              prim_type, num_verts, body_dw - 1);

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_IMMD,
                    .prim_type = prim_type,
                    .num_vertices = num_verts,
                    .vertex_data = (body_dw > 1) ? &d[1] : NULL,
                    .vertex_data_dwords = (body_dw > 1) ? body_dw - 1 : 0,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_3D_DRAW_INDX_2 && body_dw >= 1) {
                /*
                 * R200 3D_DRAW_INDX_2 (opcode 0x25)
                 */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t prim_type = d[0] & 0xF;
                uint32_t num_idx = (d[0] >> 4) & 0xFFF;
                gpu_debug_log("3D_DRAW_INDX_2: prim=%u indices=%u",
                              prim_type, num_idx);

                PPCMacGPU3DDrawCmd cmd = {
                    .opcode = R200_3D_DRAW_INDX,
                    .prim_type = prim_type,
                    .num_vertices = num_idx,
                    .num_indices = num_idx,
                    .index_data = (body_dw > 1) ? (const uint16_t *)&d[1] : NULL,
                };
                ppc_mac_gpu_dispatch_3d_draw(s, &cmd);

            } else if (opcode == R200_WAIT_FOR_IDLE) {
                /* Wait for idle — we are always idle */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);

            } else if (opcode == 0x32) {
                /*
                 * 3D_CLEAR_ZMASK (d[0] first tile, d[1] tile count, d[2]
                 * mask value).  With Z compression on (ZSTENCILCNTL bits
                 * 28/31, as Warcraft III runs) a depth clear only resets the
                 * tile-status "Z mask"; cleared tiles then read back as
                 * RB3D_DEPTHCLEARVALUE.  Compression is not modelled, so
                 * store the clear value into the depth buffer itself.  This
                 * opcode used to be run as an indirect buffer at GPU address
                 * d[0] - i.e. the front buffer's pixels executed as PM4.
                 */
                r200_clear_depth_buffer(s);

            } else if (opcode == 0x37) {
                /* 3D_CLEAR_HIZ: no hierarchical Z is modelled. */

            } else if (opcode == 0x33) {
                /*
                 * INDIRECT_BUFFER / INDIRECT_BUFFER_PFD
                 * d[0] = IB base address (GPU/AGP address)
                 * d[1] = IB size in dwords
                 */
                if (body_dw >= 2) {
                    uint32_t sub_ib_base = d[0];
                    uint32_t sub_ib_size = d[1] & 0xFFFFF;
                    gpu_debug_log("PM4_IB_DISPATCH base=0x%x size=%u dw",
                                  sub_ib_base, sub_ib_size);
                    ppc_mac_gpu_execute_ib(s, sub_ib_base, sub_ib_size);
                }

            } else {
                /*
                 * Say once which packets are dropped: CNTL_BITBLT went
                 * unnoticed for months because nothing here logged it.
                 * 0x10 is NOP.
                 */
                static uint64_t t3_unhandled[4];
                if (opcode != 0x10 &&
                    !(t3_unhandled[opcode >> 6] & (1ull << (opcode & 63)))) {
                    t3_unhandled[opcode >> 6] |= 1ull << (opcode & 63);
                    qemu_log("ppc-mac-gpu: type-3 packet 0x%02x (%u dwords) "
                             "not handled, dropped: %08x %08x %08x\n", opcode,
                             body_dw, body_dw > 0 ? d[0] : 0,
                             body_dw > 1 ? d[1] : 0, body_dw > 2 ? d[2] : 0);
                }
                /*
                 * PPCGPU_T3_DUMP: whole body of the first 4 packets of each
                 * unhandled opcode, to work out layouts (ATI VA driver 0x3A).
                 */
                static uint8_t t3_dumped[256];
                if (opcode != 0x10 && t3_dumped[opcode] < 4 &&
                    getenv("PPCGPU_T3_DUMP")) {
                    t3_dumped[opcode]++;
                    qemu_log("ppc-mac-gpu: T3DUMP op 0x%02x #%u hdr %08x "
                             "(%u dw)\n", opcode, t3_dumped[opcode], hdr,
                             body_dw);
                    for (uint32_t l = 0; l < body_dw; l += 8) {
                        GString *line = g_string_new(NULL);
                        for (uint32_t k = l; k < body_dw && k < l + 8; k++) {
                            g_string_append_printf(line, " %08x", d[k]);
                        }
                        qemu_log("  [%3u]%s\n", l, line->str);
                        g_string_free(line, true);
                    }
                }
                /* Log other Type 3 opcodes for analysis */
                trace_ppc_mac_gpu_3d_cmd(opcode, body_dw);
                uint32_t log_count = body_dw < 8 ? body_dw : 8;
                for (uint32_t l = 0; l < log_count; l++) {
                    gpu_debug_log("  3D_CMD op=0x%02x [%u]=0x%08x",
                                  opcode, l, d[l]);
                }
            }

            /* Phase A — VRAM watch: check after Type 3 processing */
            if (s->vram_watch_active && 0x353000 + 4 <= s->vram_size) {
                uint8_t *vp = memory_region_get_ram_ptr(&s->vram);
                uint32_t t3_vw_after = *(uint32_t *)(vp + 0x353000);
                if (t3_vw_after != t3_vw_before) {
                    fprintf(stderr, "[VRAM_CLR_FOUND] Type3 op=0x%02x "
                            "body=%u CHANGED vram[0x353000] from "
                            "0x%08x to 0x%08x\n",
                            opcode, body_dw,
                            t3_vw_before, t3_vw_after);
                }
            }

            i += body_adv;
        } else {
            /* Type 1: skip data words */
            uint32_t pkt_count = ((hdr >> 16) & 0x3FFF) + 1;
            i += pkt_count;
        }
    }
    g_in_pm4--;
}

/*
 * Execute an Indirect Buffer.
 *
 * The IB address may be in VRAM or in the GART aperture (system RAM).
 * We try GART first; if that fails, fall back to VRAM.
 */
static void ppc_mac_gpu_execute_ib(PPCMacGPUState *s,
                                    uint32_t ib_base,
                                    uint32_t ib_size_dw)
{
    if (r300_ringdump()) {
        fprintf(r300_ringdump(), "IB base=%08x size=%u\n", ib_base, ib_size_dw);
    }
    if (ib_size_dw == 0 || ib_size_dw > 0x100000) {
        s->regs.stall_ib_lost++;
        qemu_log("ppc-mac-gpu: IB lost: base=%08x size=%u\n", ib_base, ib_size_dw);
        return;
    }

    gpu_debug_log("IB_EXEC base=0x%x size=%u dwords", ib_base, ib_size_dw);

    /* Try GART translation first (IB is usually in system RAM) */
    uint32_t *ib_data = ppc_mac_gpu_read_ib_via_gart(s, ib_base, ib_size_dw);

    if (!ib_data) {
        /* GART failed - try reading from VRAM */
        uint64_t ib_end = (uint64_t)ib_base + (uint64_t)ib_size_dw * 4;
        if (ib_end > s->vram_size) {
            gpu_debug_log("IB_EXEC: neither GART nor VRAM (base=0x%x)", ib_base);
            qemu_log("ppc-mac-gpu: IB lost (not in GART or VRAM): base=%08x size=%u\n",
                     ib_base, ib_size_dw);
            s->regs.stall_ib_lost++;
            return;
        }
        uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
        ib_data = g_malloc(ib_size_dw * 4);
        for (uint32_t d = 0; d < ib_size_dw; d++) {
            ib_data[d] = ldl_be_p(vram + ib_base + d * 4);
        }
    }

    s->regs.stall_ib_done++;
    s->regs.stall_ib_dwords += ib_size_dw;
    ppc_mac_gpu_process_pm4(s, ib_data, ib_size_dw);
    g_free(ib_data);
}

/*
 * Process commands from the CP ring buffer.
 *
 * Called when CP_RB_WPTR advances. Reads PM4 packets from the ring buffer
 * (located in AGP/system RAM via GART) between old_rptr and new_wptr,
 * handling wrap-around based on the ring buffer size in CP_RB_CNTL.
 *
 * On real R200 hardware:
 *   - CP_RB_CNTL bits[5:0] = log2 of ring size in QWords (8 bytes)
 *   - Ring size in DWords = 2 << rb_bufsz
 *   - RPTR and WPTR are in DWord units
 *   - Ring wraps at (ring_size_dw - 1)
 */
/*
 * The command processor on its own thread.
 *
 * A real Radeon executes its ring while the CPU goes on running, and the
 * driver is written for that: it learns how far the card has got from
 * the read pointer and from fences, never from the timing of its own
 * register writes.  Here the ring used to run inside the guest's write to
 * CP_RB_WPTR, on the vCPU thread, so the guest stood still while every
 * packet was decoded, translated and handed to Metal.
 *
 * Now a WPTR write only records the pointer and wakes the "ppc-gpu-cp"
 * thread.  That thread replays the ring holding the BQL, exactly as the
 * vCPU used to, so every device path keeps the serialisation it was
 * written for (registers, Metal, interrupts, the display refresh).  The
 * gain is that TCG runs guest code without the BQL: the guest carries on
 * building its next batch while this one is carried out.
 *
 * Ordering with everything the guest does by other routes is kept with
 * one rule: any guest access to the card's registers or apertures first
 * catches the ring up, on the accessing thread, before it is served (see
 * ppc_mac_gpu_cp_sync).  That is precisely the state the guest saw when
 * the ring ran synchronously, so fences, read-pointer write-backs, 2D
 * blits through registers and CPU readbacks stay ordered as before.  The
 * exceptions are the accesses that have no ordering to keep: the WPTR
 * write itself, which goes through a small lock-free region so a submit
 * never waits for the BQL, reads of the read pointer (the card's
 * progress, which is what the driver wants), and the cursor.
 *
 * So that a long batch does not hold the BQL -- and with it other devices
 * and further submits -- for its whole length, the thread lets go between
 * packets every CP_SLICE_US.  It publishes its progress as the read
 * pointer first; whoever takes the lock can catch up from there, and the
 * thread notices (cp.epoch) and drops the rest of its copy.
 *
 * The Metal half of each draw is not under the BQL at all: with the
 * default renderer, the CP thread's draw calls enqueue the decoded packet
 * on the Metal render queue (see ppc_mac_gpu_metal.m) and a backend
 * thread encodes it, so a slice is parse work only and the yield almost
 * never fires.  PPCGPU_RENDER_QUEUE=0 puts the encoding back on the
 * caller's thread, as before.
 *
 * PPCGPU_CP_SYNC=1 restores the synchronous ring.
 */
static bool ppc_mac_gpu_cp_on_thread(void)
{
    return cp.started && qemu_thread_is_self(&cp.thread);
}

/*
 * Called by the CP thread between two ring packets, BQL held.  Returns
 * true if the ring was taken over while the lock was released, in which
 * case the rest of this segment has been (or will be) run by someone else.
 */
static bool ppc_mac_gpu_cp_yield(PPCMacGPUState *s, uint32_t pos_dw)
{
    int64_t now;
    unsigned epoch;
    int nest;

    if (++cp.pkts & 15) {
        return false;
    }
    now = g_get_monotonic_time();
    if (now < cp.yield_at) {
        return false;
    }
    /* Publish progress: a catch-up starts here, and the guest may reuse
     * the ring space behind it. */
    s->regs.cp_rb_rptr = (cp.seg_rptr + pos_dw) & cp.ring_mask;
    ppc_mac_gpu_rptr_writeback(s);
    epoch = cp.epoch;
    nest = g_in_pm4;
    g_in_pm4 = 0;
    cp_in_ring = false;
    bql_unlock();
    bql_lock();
    g_in_pm4 = nest;
    if (cp.epoch != epoch) {
        cp.aborted = true;
        return true;
    }
    cp_in_ring = true;
    cp.yield_at = g_get_monotonic_time() + CP_SLICE_US;
    return false;
}

/* Returns false if the ring was taken over part-way (CP thread only). */
static bool ppc_mac_gpu_process_ring_buffer(PPCMacGPUState *s,
                                             uint32_t old_rptr,
                                             uint32_t new_wptr)
{
    uint32_t rb_bufsz = s->regs.cp_rb_cntl & 0x3F;
    if (rb_bufsz == 0 || rb_bufsz > 25) {
        return true; /* Invalid or uninitialized */
    }
    uint32_t ring_size_dw = 2U << rb_bufsz;
    uint32_t ring_mask = ring_size_dw - 1;

    /* Calculate number of new DWords to process */
    uint32_t count = (new_wptr - old_rptr) & ring_mask;
    if (count == 0 || count > ring_size_dw) {
        return true;
    }

    /* Safety limit — don't process excessively large batches */
    if (count > 65536) {
        gpu_debug_log("RING: skipping oversized batch (%u dw)", count);
        return true;
    }

    /* Read ring buffer data via GART translation.
     * The ring buffer base (CP_RB_BASE) is an AGP address that needs
     * GART translation to access system RAM.  Fetch page-sized runs at
     * a time: one translation and one bulk read per run instead of one
     * per dword, which for a big submit was thousands of address-space
     * dispatches on the CP thread. */
    uint32_t *rb_data = g_malloc(count * 4);
    AddressSpace *as = pci_get_address_space(&s->pci);

    for (uint32_t i = 0; i < count;) {
        uint32_t offset_dw = (old_rptr + i) & ring_mask;
        uint32_t gpu_addr = s->regs.cp_rb_base + offset_dw * 4;
        /* Run length: to the end of the page, of the ring, of the batch. */
        uint32_t n = count - i;
        n = MIN(n, (0x1000 - (gpu_addr & 0xFFF)) / 4);
        n = MIN(n, ring_size_dw - offset_dw);
        uint8_t *page = r200_agp_page(s, gpu_addr);
        if (page) {
            memcpy(rb_data + i, page + (gpu_addr & 0xFFF), n * 4);
        } else {
            hwaddr phys;

            if (!ppc_mac_gpu_gart_translate(s, gpu_addr, &phys) &&
                !ppc_mac_gpu_agp_translate(s, gpu_addr, &phys)) {
                gpu_debug_log("RING: GART translate failed at gpu_addr=0x%x "
                              "(rb_base=0x%x offset_dw=%u)",
                              gpu_addr, s->regs.cp_rb_base, offset_dw);
                g_free(rb_data);
                return true;
            }
            if (address_space_read(as, phys, MEMTXATTRS_UNSPECIFIED,
                                   rb_data + i, n * 4) != MEMTX_OK) {
                gpu_debug_log("RING: read failed at phys=0x%"PRIx64,
                              (uint64_t)phys);
                g_free(rb_data);
                return true;
            }
        }
        /* Ring buffer data is written by PPC CPU in big-endian */
        for (uint32_t k = 0; k < n; k++) {
            rb_data[i + k] = be32_to_cpu(rb_data[i + k]);
        }
        i += n;
    }

    gpu_debug_log("RING: processing %u dwords (rptr=%u wptr=%u rb_bufsz=%u)",
                  count, old_rptr, new_wptr, rb_bufsz);

    bool on_thread = ppc_mac_gpu_cp_on_thread();
    if (on_thread) {
        cp.seg_rptr = old_rptr;
        cp.ring_mask = ring_mask;
        cp.aborted = false;
        cp_in_ring = true;
    }
    ppc_mac_gpu_process_pm4(s, rb_data, count);
    cp_in_ring = false;
    g_free(rb_data);
    return !(on_thread && cp.aborted);
}

/*
 * Carry out the ring up to the newest write pointer, BQL held.  Runs on
 * the CP thread, or on whichever thread must see the ring caught up.
 */
static void ppc_mac_gpu_cp_run(PPCMacGPUState *s)
{
    unsigned epoch = ++cp.epoch;    /* a yielded CP thread gives up its copy */

    for (;;) {
        uint32_t wptr = qatomic_read(&s->regs.cp_rb_wptr);
        uint32_t rptr = s->regs.cp_rb_rptr;

        if (wptr == rptr) {
            return;
        }
        /*
         * Process ring buffer commands if the microengine is started.
         * CP_ME_CNTL bit 28 = ME_HALT: 0 = running, 1 = halted.
         */
        if (!(s->regs.cp_me_cntl & (1 << 28))) {
            uint32_t n = (wptr >= rptr) ? (wptr - rptr) : wptr;

            s->regs.stall_ring_dwords += n;
            r200_traffic.ring_dwords += n;
            if (!ppc_mac_gpu_process_ring_buffer(s, rptr, wptr)) {
                return;             /* someone else caught the ring up */
            }
        }
        if (cp.epoch != epoch) {
            return;
        }
        /* Everything up to wptr is carried out: the ring is empty again --
         * in the card's own register and in the copy the driver may be
         * reading from memory instead. */
        s->regs.cp_rb_rptr = wptr;
        ppc_mac_gpu_rptr_writeback(s);
        /* Bump CSQ stat counter so the kext's "wait for CSQ change" poll
         * sees a different value after we process commands. */
        s->regs.cp_csq_stat_counter++;
    }
}

static bool ppc_mac_gpu_cp_pending(PPCMacGPUState *s)
{
    return qatomic_read(&s->regs.cp_rb_wptr) != s->regs.cp_rb_rptr;
}

/* Bring the ring up to date before the guest sees the card (BQL held). */
static void ppc_mac_gpu_cp_sync(PPCMacGPUState *s, hwaddr why)
{
    if (!ppc_mac_gpu_cp_pending(s) || ppc_mac_gpu_cp_on_thread()) {
        return;
    }
    cp.syncs++;
    if (why < 0x10000 && cp.sync_reg[why / 4] < UINT16_MAX) {
        cp.sync_reg[why / 4]++;
    }
    ppc_mac_gpu_cp_run(s);
}

/* The guest wrote CP_RB_WPTR.  May be called without the BQL. */
static void ppc_mac_gpu_cp_kick(PPCMacGPUState *s, uint32_t wptr)
{
    qatomic_set(&s->regs.cp_rb_wptr, wptr);
    if (cp.started) {
        qatomic_inc(&cp.kicks);
        qemu_event_set(&cp.kick);
    } else {
        ppc_mac_gpu_cp_run(s);
    }
}

static void *ppc_mac_gpu_cp_thread(void *opaque)
{
    PPCMacGPUState *s = opaque;

    rcu_register_thread();
    for (;;) {
        qemu_event_reset(&cp.kick);
        if (qatomic_read(&cp.stop)) {
            break;
        }
        if (qatomic_read(&s->regs.cp_rb_wptr) !=
            qatomic_read(&s->regs.cp_rb_rptr)) {
            bql_lock();
            cp.yield_at = g_get_monotonic_time() + CP_SLICE_US;
            ppc_mac_gpu_cp_run(s);
            bql_unlock();
            continue;
        }
        qemu_event_wait(&cp.kick);
    }
    rcu_unregister_thread();
    return NULL;
}

/* Lock-free CP_RB_WPTR, so a submit never waits for the BQL. */
static uint64_t ppc_mac_gpu_cp_wptr_read(void *opaque, hwaddr addr,
                                         unsigned size)
{
    PPCMacGPUState *s = opaque;
    return qatomic_read(&s->regs.cp_rb_wptr);
}

static void ppc_mac_gpu_cp_wptr_write(void *opaque, hwaddr addr,
                                      uint64_t val, unsigned size)
{
    ppc_mac_gpu_cp_kick(opaque, val);
}

static const MemoryRegionOps ppc_mac_gpu_cp_wptr_ops = {
    .read = ppc_mac_gpu_cp_wptr_read,
    .write = ppc_mac_gpu_cp_wptr_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void ppc_mac_gpu_cp_start(PPCMacGPUState *s)
{
    const char *e = getenv("PPCGPU_CP_SYNC");

    if (e && e[0] == '1') {
        return;
    }
    cp.s = s;
    qemu_event_init(&cp.kick, false);
    cp.started = true;
    qemu_thread_create(&cp.thread, "ppc-gpu-cp", ppc_mac_gpu_cp_thread, s,
                       QEMU_THREAD_JOINABLE);
    memory_region_init_io(&s->cp_wptr_mr, OBJECT(s), &ppc_mac_gpu_cp_wptr_ops,
                          s, "ppc-mac-gpu-cp-wptr", 4);
    memory_region_enable_lockless_io(&s->cp_wptr_mr);
    memory_region_add_subregion_overlap(&s->mmio, R200_CP_RB_WPTR,
                                        &s->cp_wptr_mr, 1);
}

static void ppc_mac_gpu_cp_stop(void)
{
    if (!cp.started) {
        return;
    }
    qatomic_set(&cp.stop, true);
    qemu_event_set(&cp.kick);
    bql_unlock();
    qemu_thread_join(&cp.thread);
    bql_lock();
    cp.started = false;
}

/* The last command packets pushed in by hand, for the stall report. */
static struct { uint32_t hdr, type, opcode, count; } pm4_recent[32];
static unsigned pm4_recent_n;

static void ppc_mac_gpu_pm4_fifo_push(PPCMacGPUState *s, uint32_t val)
{
    s->regs.stall_pio_dwords++;
    s->regs.csq_just_submitted = true;
    if (s->pm4_pkt_count == 0) {
        unsigned k = pm4_recent_n++ & 31;
        pm4_recent[k].hdr = val;
        pm4_recent[k].type = (val >> 30) & 3;
        pm4_recent[k].opcode = (val >> 8) & 0xFF;
        pm4_recent[k].count = ((val >> 16) & 0x3FFF) + 1;
    }
    if (s->pm4_pkt_count == 0) {
        /* New packet header */
        uint32_t type = (val >> 30) & 3;

        if (type == 0) {
            /* Type 0: Write N+1 registers
             * Bits [14:0]  = register index
             * Bit  [15]    = ONE_REG_WR (all writes to same register)
             * Bits [29:16] = count - 1
             */
            uint32_t count = ((val >> 16) & 0x3FFF) + 1;
            uint32_t reg_base = val & 0x7FFF;
            bool one_reg_wr = (val >> 15) & 1;

            s->pm4_pkt_reg = reg_base;
            s->pm4_pkt_one_reg = one_reg_wr;
            s->pm4_pkt_count = count;
            s->pm4_fifo_idx = 0;

            gpu_debug_log("PM4_HDR type0 reg=0x%04x count=%u%s",
                          reg_base * 4, count,
                          one_reg_wr ? " ONE_REG_WR" : "");
        } else if (type == 2) {
            /* Type 2: NOP/filler - no action needed.
             * The kext uses NOP as end-of-batch marker. We don't
             * auto-complete scratch registers here; the CP processes
             * commands synchronously through PIO, so completion is
             * immediate from the kext's perspective.
             */
            gpu_debug_log("PM4_NOP (batch complete)");
        } else if (type == 3) {
            /* Type 3: GPU engine command */
            uint32_t count = ((val >> 16) & 0x3FFF) + 1;
            s->pm4_pkt_count = count;
            s->pm4_pkt_reg = 0xFFFFFFFF; /* marker: not type 0 */
            s->pm4_pkt_opcode = (val >> 8) & 0xFF;
            s->pm4_fifo_idx = 0;

            gpu_debug_log("PM4_PIO type3 opcode=0x%02x count=%u",
                          s->pm4_pkt_opcode, count);
        } else {
            /* Type 1: reserved/unused */
            gpu_debug_log("PM4_PIO type%u (ignored)", type);
        }
    } else {
        /* Data word for current packet */
        if (s->pm4_fifo_idx < ARRAY_SIZE(s->pm4_fifo)) {
            s->pm4_fifo[s->pm4_fifo_idx++] = val;
        }
        s->pm4_pkt_count--;

        if (s->pm4_pkt_count == 0) {
            if (s->pm4_pkt_reg != 0xFFFFFFFF) {
                /* Type 0 complete - execute register writes */
                ppc_mac_gpu_pm4_process_type0(s, s->pm4_pkt_reg,
                                               s->pm4_pkt_one_reg,
                                               s->pm4_fifo_idx,
                                               s->pm4_fifo);
            } else {
                /* Type 3 complete - build a PM4 packet and process it.
                 * Reconstruct the header so process_pm4 can parse it. */
                uint32_t opcode = s->pm4_pkt_opcode;
                uint32_t ndw = s->pm4_fifo_idx;
                if (ndw > 0 && ndw <= 60) {
                    uint32_t pm4_buf[64];
                    /* Type 3 header: type=3, count=ndw-1, opcode */
                    pm4_buf[0] = (3U << 30) |
                                 ((ndw - 1) << 16) |
                                 (opcode << 8);
                    memcpy(&pm4_buf[1], s->pm4_fifo, ndw * 4);
                    ppc_mac_gpu_process_pm4(s, pm4_buf, ndw + 1);
                }
            }
            s->pm4_fifo_idx = 0;
        }
    }
}

/* ========================================================================
 * R300 register range detection for catch-all shadow storage.
 *
 * Returns true if addr is in a known R300-specific register range that
 * ATIRadeon9700.kext accesses but our RV280 doesn't natively implement.
 * These are handled via a shadow array so reads return written values.
 * ======================================================================== */

static bool is_r300_register(uint32_t addr)
{
    /* R300 VAP (Vertex Assembly Processor) registers */
    if (addr >= 0x2080 && addr <= 0x22D8) return true;
    /* R300 misc */
    if (addr >= 0x3428 && addr <= 0x342C) return true;
    /* R300 GB, TX, US, FG, RB3D, ZB registers (0x4000-0x4F78) */
    if (addr >= 0x4000 && addr <= 0x4F78) return true;
    /* R300 surface/tiling config */
    if (addr >= 0x0400 && addr <= 0x0514) return true;
    /* R300 misc lower registers */
    if (addr == 0x015C || addr == 0x01D0 || addr == 0x01DC ||
        addr == 0x01E0) return true;
    if (addr == 0x0218 || addr == 0x023C || addr == 0x0328) return true;
    if (addr >= 0x070C && addr <= 0x0744) return true;
    if (addr == 0x0AB0 || addr == 0x0AB4) return true;
    if (addr >= 0x142C && addr <= 0x1438) return true;
    if (addr >= 0x15D0 && addr <= 0x15FC) return true;
    if (addr >= 0x16CC && addr <= 0x1710) return true;
    if (addr >= 0x1D98 && addr <= 0x1DAC) return true;
    if (addr == 0x1FA8) return true;
    return false;
}

/* ========================================================================
 * MMIO read handler
 * ======================================================================== */

/*
 * Hardware cursor, driven by the patched qemu_vga.ndrv (see
 * QEMU Project/ndrv-hwcursor).  Mac OS X otherwise draws a software cursor
 * with the CPU and keeps a saved copy of the pixels beneath it, which goes
 * stale whenever the GPU redraws a GL surface under a stationary cursor.
 * Registers at MMIO BAR + 0xFF00, 32-bit little-endian:
 *   00 SIZE w|h<<16 (starts an upload)  04 DATA ARGB pixel  08 COMMIT
 *   0C X  10 Y (image top-left, signed)  14 SHOW bit 0  18 ID 'HWC1'
 *   1C HOT x|y<<16 (hotspot inside the image, before COMMIT; optional)
 */
#define PPC_MAC_GPU_HWC_BASE  0xFF00
#define PPC_MAC_GPU_HWC_END   0xFF20
#define PPC_MAC_GPU_HWC_MAGIC 0x48574331

/*
 * X/Y from the NDRV are the image's top-left. The hotspot (reg 0x1C) puts
 * the host pointer on the cursor's tip: SPICE client mode draws the image
 * at pointer - hot, and server mode wants the hotspot's screen position.
 */
static void ppc_mac_gpu_hwc_define(PPCMacGPUState *s)
{
    QEMUCursor *c;

    if (!s->hwc_w || !s->hwc_h || !s->con) {
        return;
    }
    c = cursor_alloc(s->hwc_w, s->hwc_h);
    c->hot_x = MIN(s->hwc_hot_x, s->hwc_w - 1);
    c->hot_y = MIN(s->hwc_hot_y, s->hwc_h - 1);
    memcpy(c->data, s->hwc_pix, s->hwc_w * s->hwc_h * 4);
    dpy_cursor_define(s->con, c);
    cursor_unref(c);
}

static void ppc_mac_gpu_hwc_move(PPCMacGPUState *s)
{
    if (!s->con) {
        return;
    }
    dpy_mouse_set(s->con,
                  s->hwc_x + (int32_t)MIN(s->hwc_hot_x, MAX(s->hwc_w, 1) - 1),
                  s->hwc_y + (int32_t)MIN(s->hwc_hot_y, MAX(s->hwc_h, 1) - 1),
                  s->hwc_visible);
}

static void ppc_mac_gpu_hwc_write(PPCMacGPUState *s, uint32_t reg, uint32_t val)
{
    switch (reg) {
    case 0x00:
        s->hwc_w = MIN(val & 0xFFFF, 64);
        s->hwc_h = MIN(val >> 16, 64);
        s->hwc_idx = 0;
        break;
    case 0x04:
        if (s->hwc_idx < s->hwc_w * s->hwc_h) {
            s->hwc_pix[s->hwc_idx++] = val;
        }
        break;
    case 0x08:
        ppc_mac_gpu_hwc_define(s);
        break;
    case 0x0C:
        s->hwc_x = (int32_t)val;
        break;
    case 0x10:
        s->hwc_y = (int32_t)val;
        break;
    case 0x14:
        s->hwc_visible = val & 1;
        ppc_mac_gpu_hwc_move(s);
        break;
    case 0x1C:
        /* hotspot x | y << 16, sent before COMMIT by newer NDRVs */
        s->hwc_hot_x = val & 0xFFFF;
        s->hwc_hot_y = val >> 16;
        break;
    }
}

static uint32_t ppc_mac_gpu_hwc_read(PPCMacGPUState *s, uint32_t reg)
{
    switch (reg) {
    case 0x00: return s->hwc_w | s->hwc_h << 16;
    case 0x0C: return s->hwc_x;
    case 0x10: return s->hwc_y;
    case 0x14: return s->hwc_visible;
    case 0x18: return PPC_MAC_GPU_HWC_MAGIC;
    case 0x1C: return s->hwc_hot_x | s->hwc_hot_y << 16;
    }
    return 0;
}

static uint64_t ppc_mac_gpu_mmio_read(void *opaque, hwaddr addr,
                                       unsigned int size)
{
    PPCMacGPUState *s = opaque;
    uint64_t val = 0;

    /*
     * POWEREMU_STALL_TRACE: the last register accesses before the guest
     * stops drawing.  A game that renders and then sits there is waiting
     * for something; this says what it touched on the way in, and what it
     * keeps touching while stuck.
     */
    if (getenv("POWEREMU_STALL_TRACE")) {
        static struct { uint32_t addr, val; bool wr; } ring[256];
        static unsigned n;
        static int64_t last_draw_seen;
        static bool dumped;
        int64_t now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

        ring[n & 255].addr = addr;
        ring[n & 255].val = 0;
        ring[n & 255].wr = false;
        n++;
        if (s->regs.stall_draws != s->regs.stall_draws_seen || !last_draw_seen) {
            s->regs.stall_draws_seen = s->regs.stall_draws;
            last_draw_seen = now;
            dumped = false;
        }
        if (!dumped && last_draw_seen && now - last_draw_seen > 4000) {
            dumped = true;
            fprintf(stderr, "ppc-mac-gpu stalled: no drawing for 4 s.\n"
                    "  fences: reg0=%u reg1=%u reg2=%u reg3=%u "
                    "(copy at 0x%08x, enabled 0x%x)\n"
                    "  ring: rptr=%u wptr=%u, copy kept at 0x%08x; "
                    "commands by hand: %llu dwords, through the ring: %llu\n"
                    "  buffers fetched from memory: %llu carried out (%llu dwords), "
                    "%llu not found\n"
                    "  interrupts: asked for 0x%08x, pending 0x%08x, "
                    "%llu raised\n"
                    "  the fence copy lands in %s\n"
                    "  the last %u registers the guest touched:\n",
                    s->regs.scratch_reg[0], s->regs.scratch_reg[1],
                    s->regs.scratch_reg[2], s->regs.scratch_reg[3],
                    s->regs.scratch_addr, s->regs.scratch_umsk,
                    s->regs.cp_rb_rptr, s->regs.cp_rb_wptr,
                    s->regs.cp_rb_rptr_addr,
                    (unsigned long long)s->regs.stall_pio_dwords,
                    (unsigned long long)s->regs.stall_ring_dwords,
                    (unsigned long long)s->regs.stall_ib_done,
                    (unsigned long long)s->regs.stall_ib_dwords,
                    (unsigned long long)s->regs.stall_ib_lost,
                    s->regs.gen_int_cntl, s->regs.gen_int_status,
                    (unsigned long long)s->regs.stall_irqs,
                    ({
                        hwaddr ph;
                        ppc_mac_gpu_gart_translate(s, s->regs.scratch_addr, &ph) ? "memory the guest shares (through GART)"
                            : ppc_mac_gpu_agp_translate(s, s->regs.scratch_addr, &ph) ? "memory the guest shares (through AGP)"
                            : s->regs.scratch_addr + 4 <= s->vram_size ? "the card's own memory -- the guest cannot see it there"
                            : "nowhere at all";
                    }), 256u);
            /* What the waiting program actually sees in that page. */
            {
                hwaddr ph;
                if (ppc_mac_gpu_gart_translate(s, s->regs.scratch_addr, &ph) ||
                    ppc_mac_gpu_agp_translate(s, s->regs.scratch_addr, &ph)) {
                    uint32_t page[16];
                    address_space_read(&address_space_memory, ph,
                                       MEMTXATTRS_UNSPECIFIED, page, sizeof(page));
                    fprintf(stderr, "  that page (guest address 0x%" PRIx64 "), "
                            "as words, each shown both ways round:\n", (uint64_t)ph);
                    for (int w = 0; w < 16; w += 4) {
                        fprintf(stderr, "    +%02x:", w * 4);
                        for (int k = 0; k < 4; k++) {
                            uint32_t raw = page[w + k];
                            fprintf(stderr, "  %10u / %10u", le32_to_cpu(raw),
                                    be32_to_cpu(raw));
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
            fprintf(stderr, "  the last commands it pushed:\n");
            for (unsigned i = 0; i < 32; i++) {
                unsigned k = (pm4_recent_n + i) & 31;
                if (!pm4_recent[k].hdr) {
                    continue;
                }
                if (pm4_recent[k].type == 0) {
                    fprintf(stderr, "    write %u register(s) from %s(0x%03x)\n",
                            pm4_recent[k].count,
                            ppc_mac_gpu_reg_name((pm4_recent[k].hdr & 0x7FFF) * 4),
                            (pm4_recent[k].hdr & 0x7FFF) * 4);
                } else {
                    fprintf(stderr, "    type %u opcode 0x%02x, %u word(s)\n",
                            pm4_recent[k].type, pm4_recent[k].opcode,
                            pm4_recent[k].count);
                }
            }
            fprintf(stderr, "  the last %u registers the guest touched:\n", 32u);
            for (unsigned i = 224; i < 256; i++) {
                unsigned k = (n + i) & 255;
                fprintf(stderr, "    %s %s(0x%03x)\n", ring[k].wr ? "wrote" : "read",
                        ppc_mac_gpu_reg_name(ring[k].addr), ring[k].addr);
            }
        }
    }

    /*
     * POWEREMU_POLL_TRACE: which registers the guest reads over and over.
     * A program waiting on the card sits in a loop reading one of them, and
     * this says which -- Halo's title screen draws and then stops, with its
     * sound looping, so something it waits for never arrives.
     */
    if (getenv("POWEREMU_POLL_TRACE")) {
        static uint32_t count[0x4000 / 4];
        static int64_t next_report;
        int64_t now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
        if (addr < 0x4000) {
            count[addr / 4]++;
        }
        if (now > next_report) {
            if (next_report) {
                uint32_t top[5] = { 0 }, topi[5] = { 0 };
                for (uint32_t i = 0; i < 0x4000 / 4; i++) {
                    for (int k = 0; k < 5; k++) {
                        if (count[i] > top[k]) {
                            for (int j = 4; j > k; j--) {
                                top[j] = top[j - 1]; topi[j] = topi[j - 1];
                            }
                            top[k] = count[i]; topi[k] = i;
                            break;
                        }
                    }
                }
                fprintf(stderr, "ppc-mac-gpu polled most:");
                for (int k = 0; k < 5 && top[k]; k++) {
                    fprintf(stderr, " %s(0x%03x)x%u", ppc_mac_gpu_reg_name(topi[k] * 4),
                            topi[k] * 4, top[k]);
                }
                fprintf(stderr, "\n");
                memset(count, 0, sizeof(count));
            }
            next_report = now + 5000;
        }
    }

    if (addr >= PPC_MAC_GPU_HWC_BASE && addr < PPC_MAC_GPU_HWC_END) {
        return size == 4 ? ppc_mac_gpu_hwc_read(s, addr - PPC_MAC_GPU_HWC_BASE) : 0;
    }

    if (addr >= 0x780 && addr < 0x7C0) {
        seq_log("DMA  rd reg=%03x", (unsigned)addr);
    }
    if (unlikely(r200_traffic_on())) {
        r200_traffic.mmio_reads++;
    }
    /* Idle/fence status reads precede CPU access to rendered VRAM; FIFO
     * space, scanline and interrupt polls do not. */
    if (addr >= R200_SCRATCH_REG0 && addr <= R200_SCRATCH_REG5 && !(addr & 3) &&
        r200_direct_enabled() && r200_scratch_read_wait(s, (addr - R200_SCRATCH_REG0) / 4)) {
        /* fence poll: answered without flushing later work */
    } else if (addr != R200_CP_CSQ_CNTL && addr != R200_CP_CSQ_STAT &&
        addr != R200_CRTC_VLINE_CRNT_VLINE && addr != 0x0044) {
        r200_flush_at(s, R200_WHY_REG(addr));
    }
    if (unlikely(gpu_debug_enabled())) {
        gpu_debug_log("MMIO_RD size=%u addr=0x%04"PRIx64" (%s)",
                      size, (uint64_t)addr, ppc_mac_gpu_reg_name(addr));
    }

    /* EDID compatibility for QEMU VGA NDRV:
     * The NDRV reads EDID data via byte-sized reads at MMIO offsets 0x00-0x7F
     * during initialization (before the first VBE mode set).
     * After EDID reading completes, byte reads return ATI register values.
     * edid_read_done is set when VBE_ENABLE is first written (mode set). */
    if (size == 1 && addr < 128 && !s->edid_read_done) {
        val = s->edid_blob[addr];
        gpu_debug_log("EDID_RD offset=0x%02x -> 0x%02x", (unsigned)addr, (unsigned)val);
        return val;
    }

    /* Handle indexed register access via MM_INDEX/MM_DATA */
    if (addr == R200_MM_DATA && s->regs.mm_index) {
        /* Recursive call with the indexed address */
        return ppc_mac_gpu_mmio_read(s, s->regs.mm_index, size);
    }

    switch (addr) {
    case R200_MM_INDEX:
        val = s->regs.mm_index;
        break;

    /* BIOS scratch registers */
    case R200_BIOS_0_SCRATCH ... R200_BIOS_7_SCRATCH:
        val = s->regs.bios_scratch[(addr - R200_BIOS_0_SCRATCH) / 4];
        break;

    /* Interrupt */
    case R200_GEN_INT_CNTL:
        val = s->regs.gen_int_cntl;
        break;
    case R200_GEN_INT_STATUS:
        val = s->regs.gen_int_status;
        break;

    /* Config / Memory controller */
    case R200_CONFIG_MEMSIZE:
        val = s->regs.config_memsize;
        break;
    case R200_CONFIG_APER_0_BASE:
        val = pci_default_read_config(&s->pci,
                                      PCI_BASE_ADDRESS_0, 4) & ~0xFU;
        break;
    case R200_CONFIG_APER_1_BASE:
        if (s->r300) {
            val = (pci_default_read_config(&s->pci, PCI_BASE_ADDRESS_0, 4) &
                   ~0xFU) + (uint32_t)s->vram_size;
            break;
        }
        val = pci_default_read_config(&s->pci,
                                      PCI_BASE_ADDRESS_1, 4) & ~0x3U;
        break;
    case R200_CONFIG_APER_SIZE:
        val = s->r300 ? (uint32_t)s->vram_size : s->regs.config_memsize;
        break;
    case R200_CONFIG_REG_1_BASE:
        val = pci_default_read_config(&s->pci,
                                      PCI_BASE_ADDRESS_2, 4) & ~0xFU;
        break;
    case R200_CONFIG_REG_APER_SIZE:
        val = PPC_MAC_GPU_MMIO_SIZE;
        break;
    case R200_CONFIG_CNTL:
        val = s->regs.config_cntl;
        break;
    case R200_MC_FB_LOCATION:
        val = s->regs.mc_fb_location;
        break;
    case R200_MC_AGP_LOCATION:
        val = s->regs.mc_agp_location;
        break;
    case R200_MC_STATUS:
        /* Report memory controller idle */
        val = R200_MC_IDLE;
        break;

    /* Bus */
    case R200_BUS_CNTL:
        val = s->regs.bus_cntl;
        break;
    case R200_HOST_PATH_CNTL:
        val = s->regs.host_path_cntl | (1 << 23); /* HDP_SOFT_RESET capable */
        break;

    /* RBBM - report idle with 64 free FIFO entries */
    case R200_RBBM_STATUS:
        /* bits [6:0] = CMDFIFO_AVAIL (free entries)
         * bit 31 = GUI_ACTIVE (1=busy, 0=idle)
         * bits [30:7] = various pipeline busy flags
         * ATI kext checks GUI_ACTIVE to wait for idle */
        val = 64; /* 64 free entries, GUI_ACTIVE=0 (idle) */
        gpu_debug_log("STATUS_RD RBBM_STATUS -> 0x%08x (idle, %u free)", val, val & 0x7f);
        {
            static int rbbm_read_count = 0;
            rbbm_read_count++;
            if (gpu_diag_on() &&
                (rbbm_read_count <= 20 || rbbm_read_count % 1000 == 0)) {
                qemu_log("[QE_GATE_READ] reg=0x%04x value=0x%08x "
                         "context=idle_poll sequence_id=%d "
                         "note=RBBM_STATUS_always_idle\n",
                         (uint32_t)addr, val, rbbm_read_count);
            }
        }
        break;

    /* GUI status - report idle */
    case R200_GUI_STAT:
        val = 64; /* Free FIFO entries */
        gpu_debug_log("STATUS_RD GUI_STAT -> 0x%08x", val);
        break;

    /*
     * R300/RV350 register stubs — needed for ATIRadeon9700.kext to init
     * without crashing.  These registers don't exist on R200/RV280 but
     * the 9700 kext reads them during startup.  Return safe defaults.
     */
    case 0x4018: /* R300_GB_TILE_CONFIG */
        /* Report 1 pipe, 1 Z pipe — minimum config */
        val = 0x00000001;
        break;
    case 0x4024: /* R300_GB_FIFO_SIZE */
        /* 32 entries per pipe */
        val = 0x00000020;
        break;
    case 0x4008: /* R300_GB_ENABLE */
        val = 0;
        break;
    case 0x4010: /* R300_GB_MSPOS0 */
    case 0x4014: /* R300_GB_MSPOS1 */
        val = 0x66666666; /* standard MS sample positions */
        break;
    case 0x401C: /* R300_GB_SELECT */
    case 0x4020: /* R300_GB_AA_CONFIG */
        val = 0;
        break;
    case 0x2140: /* R300_VAP_CNTL_STATUS */
        val = 0; /* no errors, not busy */
        break;
    case 0x2150: /* R300_VAP_PVS_STATE_FLUSH_REG */
        val = 0;
        break;
    case 0x170C: /* R300_RBBM_SOFT_RESET */
        val = 0;
        break;
    /* CRTC */
    case R200_CRTC_GEN_CNTL:
        val = s->regs.crtc_gen_cntl;
        break;
    case R200_CRTC_EXT_CNTL:
        val = s->regs.crtc_ext_cntl;
        break;
    case R200_CRTC_STATUS:
        val = s->regs.crtc_status;
        break;
    case R200_CRTC_H_TOTAL_DISP:
        val = s->regs.crtc_h_total_disp;
        break;
    case R200_CRTC_H_SYNC_STRT_WID:
        val = s->regs.crtc_h_sync_strt_wid;
        break;
    case R200_CRTC_V_TOTAL_DISP:
        val = s->regs.crtc_v_total_disp;
        break;
    case R200_CRTC_V_SYNC_STRT_WID:
        val = s->regs.crtc_v_sync_strt_wid;
        break;
    case R200_CRTC_OFFSET:
        /*
         * Jaguar's ATIRadeon8500.kext reserves the scanout buffer from
         * CRTC_OFFSET/CRTC_PITCH only when the offset differs from its
         * (zero) initial value.  A framebuffer at VRAM 0 therefore looks
         * "unchanged": nothing is reserved, and the first GL surface it
         * allocates lands on the screen and corrupts the allocator.
         * Report the offset 32 bytes in (the unreserved head is too small
         * to ever be handed out); the scanout itself still starts at 0.
         */
        val = s->regs.crtc_offset | (s->rv250 ? R200_CRTC_OFFSET_BIAS : 0);
        break;
    case R200_CRTC_OFFSET_CNTL:
        val = s->regs.crtc_offset_cntl;
        break;
    case R200_CRTC_PITCH:
        val = s->regs.crtc_pitch;
        break;
    case R200_CRTC_GUI_TRIG_VLINE:
        val = 0;
        break;
    case R200_CRTC_VLINE_CRNT_VLINE: {
        /*
         * A scanline that walks with time.  The line the beam is on is
         * reported in bits [26:16]; bits [10:0] are the line the guest
         * asked to be told about, which it wrote here.  The current line
         * used to be returned in the low bits, where nothing reads it, so
         * anything waiting for a scanline saw zero for ever.
         */
        uint32_t vtotal = (s->regs.crtc_v_total_disp & 0x7FF) + 1;
        if (vtotal == 0) vtotal = 628; /* safety */
        int64_t now_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        uint64_t frame_ns = NANOSECONDS_PER_SECOND / 60;
        uint32_t line = (uint32_t)((now_ns % frame_ns) * vtotal / frame_ns);
        val = (s->regs.crtc_vline & 0x7FF) | ((line & 0x7FF) << 16);
        break;
    }
    case R200_CRTC_CRNT_FRAME:
        /* Frame counter - increment with time */
        val = (uint32_t)(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) /
                         (NANOSECONDS_PER_SECOND / 60));
        break;

    /* DAC */
    case R200_DAC_CNTL:
        val = s->regs.dac_cntl;
        break;
    case R200_DAC_CNTL2:
        val = s->regs.dac_cntl2;
        break;

    /* Palette */
    case R200_PALETTE_INDEX:
        val = s->regs.palette_index;
        break;
    case R200_PALETTE_DATA:
        if (s->regs.palette_index < 256) {
            val = s->regs.palette[s->regs.palette_index];
        }
        break;

    /* Cursor */
    case R200_CUR_OFFSET:
        val = s->regs.cur_offset;
        break;
    case R200_CUR_HORZ_VERT_POSN:
        val = s->regs.cur_horz_vert_posn;
        break;
    case R200_CUR_HORZ_VERT_OFF:
        val = s->regs.cur_horz_vert_off;
        break;
    case R200_CUR_CLR0:
        val = s->regs.cur_clr0;
        break;
    case R200_CUR_CLR1:
        val = s->regs.cur_clr1;
        break;

    /* Surface */
    case R200_SURFACE_CNTL:
        val = s->regs.surface_cntl;
        break;

    /* Display */
    case R200_DISP_MISC_CNTL:
        val = s->regs.disp_misc_cntl;
        break;
    case R200_DISP_OUTPUT_CNTL:
        val = s->regs.disp_output_cntl;
        break;
    case R200_DISP_MERGE_CNTL:
        val = s->regs.disp_merge_cntl;
        break;
    case R200_FP_GEN_CNTL:
        val = s->regs.fp_gen_cntl;
        break;

    /* AGP - report AGP 2.0, 4x capable */
    case R200_AGP_STATUS:
        val = 0x1F000203; /* AGP 2.0, SBA, 4x/2x/1x capable */
        break;
    case R200_AGP_COMMAND:
        val = s->regs.agp_command;
        break;
    case R200_AGP_BASE:
        val = s->regs.agp_base;
        break;
    case R200_AGP_CNTL:
        val = s->regs.agp_cntl;
        break;

    /* Clock/PLL - stub with fixed values */
    case R200_CLOCK_CNTL_INDEX:
        val = s->regs.clock_cntl_index;
        break;
    case R200_CLOCK_CNTL_DATA:
        val = s->regs.clock_cntl_data;
        break;

    /* 2D engine - read back state */
    case R200_DST_OFFSET:
        val = s->regs.dst_offset;
        break;
    case R200_DST_PITCH:
        val = s->regs.dst_pitch;
        break;
    case R200_SRC_OFFSET:
        val = s->regs.src_offset;
        break;
    case R200_SRC_PITCH:
        val = s->regs.src_pitch;
        break;
    case R200_DP_GUI_MASTER_CNTL:
        val = s->regs.dp_gui_master_cntl;
        break;
    case R200_DP_CNTL:
        val = s->regs.dp_cntl;
        break;
    case R200_DP_DATATYPE:
        val = s->regs.dp_datatype;
        break;
    case R200_DP_MIX:
        val = s->regs.dp_mix;
        break;
    case R200_DP_WRITE_MSK:
        val = s->regs.dp_write_msk;
        break;
    case R200_SC_TOP_LEFT:
        val = s->regs.sc_top_left;
        break;
    case R200_SC_BOTTOM_RIGHT:
        val = s->regs.sc_bottom_right;
        break;
    case R200_DEFAULT_SC_BOTTOM_RIGHT:
        val = s->regs.default_sc_bottom_right;
        break;

    /* CP (command processor) */
    case R200_CP_RB_BASE:
        val = s->regs.cp_rb_base;
        break;
    case R200_CP_RB_CNTL:
        val = s->regs.cp_rb_cntl;
        break;
    case R200_CP_RB_RPTR:
        val = s->regs.cp_rb_rptr;
        break;
    case R200_CP_RB_WPTR:
        val = s->regs.cp_rb_wptr;
        break;
    case R200_CP_ME_CNTL:
        val = s->regs.cp_me_cntl;
        break;
    case R200_CP_STAT:
        /* CP status - report idle (all pipeline stages inactive)
         * bit 31 = CP_BUSY (1=processing commands)
         * bits [30:0] = various stage busy flags */
        val = 0;
        gpu_debug_log("STATUS_RD CP_STAT -> 0x%08x (idle)", val);
        break;
    case R200_CP_IB_BASE:
        val = s->regs.cp_ib_base;
        break;
    case R200_CP_IB_BUFSZ:
        val = s->regs.cp_ib_bufsz;
        break;
    case R200_CP_CSQ_CNTL:
        /*
         * Kext writes CSQ mode in bits[31:28] and polls until queue
         * count fields become non-zero.  bits[7:0]=primary count,
         * bits[15:8]=indirect count, bits[23:16]=indirect2 count.
         * When mode enables indirect queues (mode 7 = PRIBM+INDBM),
         * the kext checks ALL enabled queue counts.
         */
        /*
         * How much is still queued.  Everything the guest submits is
         * carried out before the write returns, so in truth nothing ever
         * is -- but reporting zero here stops Mac OS X booting: the
         * driver's engine bring-up waits for these counts to become
         * non-zero, and hangs at the grey Apple with no spinner if they
         * never do.  So the counts stay as they were, and PPCGPU_CSQ_IDLE=1
         * reports an empty queue for anyone chasing a driver that waits
         * for the queue to drain instead.
         */
        /*
         * How much is still queued.  Everything submitted is carried out
         * before the write returns, so the honest answer is "nothing" --
         * but a driver bringing the engine up wants to see the queue
         * non-empty at least once, and reporting zero from the start left
         * Mac OS X hanging at the grey Apple.  So: busy on the first read
         * after something is submitted, empty on the reads after that.
         * A driver waiting for the queue to drain now gets its answer,
         * which is what left Halo waiting for ever.
         * PPCGPU_CSQ=busy keeps the old always-busy reply, =idle always
         * reports empty.
         */
        {
            static int mode = -1;
            if (mode < 0) {
                const char *e = getenv("PPCGPU_CSQ");
                mode = (e && !strcmp(e, "busy")) ? 1
                     : (e && !strcmp(e, "idle")) ? 2 : 0;
            }
            /*
             * Bits [7:0] are the only count this register has, and the
             * driver reads them as room to submit into: a steady 0x40 says
             * "space for 64", which is what lets Mac OS X finish bringing
             * the engine up.  Reporting zero hangs the boot.  Halo's freeze
             * was never here -- it survived both extremes -- so this stays
             * steady rather than pretending to drain.
             */
            uint32_t top = s->regs.cp_csq_cntl & 0xFF000000;
            val = (mode == 2) ? top : (top | 0x40);
        }
        break;
    case R200_SCRATCH_UMSK:
        val = s->regs.scratch_umsk;
        break;
    case R200_SCRATCH_ADDR:
        val = s->regs.scratch_addr;
        break;
    case R200_CP_ME_RAM_ADDR:
        val = s->regs.cp_me_ram_addr;
        break;
    case R200_CP_CSQ_STAT:
        /* CSQ status register (0x07f8).
         * R200/ATIRadeon8500: bits[7:0]=pri_rptr, [15:8]=pri_wptr,
         * [23:16]=ind_rptr, [31:24]=ind_wptr.  All equal = idle.
         * R300/ATIRadeon9700: the kext polls this waiting for the value
         * to change, but we process commands instantly so it never changes
         * naturally.  Return 0 so both kexts see "queues empty/idle".
         * The 9700 kext's 99999-iteration timeout will fire and it will
         * continue initialization with an error (which is acceptable —
         * we're reverting to the 8500 kext anyway). */
        val = 0;
        break;
    case R200_CP_CSQ_IND_ADDR:
        val = 0;
        break;
    case R200_CP_CSQ_IND_DATA:
        val = 0;
        break;

    /* GPIO / I2C / DDC */
    case R200_GPIO_VGA_DDC ... R200_GPIO_VGA_DDC + 3:
        val = ppc_mac_gpu_reg_read_offs(s->regs.gpio_vga_ddc,
                                         addr - R200_GPIO_VGA_DDC, size);
        break;
    case R200_GPIO_DVI_DDC ... R200_GPIO_DVI_DDC + 3:
        val = ppc_mac_gpu_reg_read_offs(s->regs.gpio_dvi_ddc,
                                         addr - R200_GPIO_DVI_DDC, size);
        break;
    case R200_GPIO_MONID ... R200_GPIO_MONID + 3:
        val = ppc_mac_gpu_reg_read_offs(s->regs.gpio_monid,
                                         addr - R200_GPIO_MONID, size);
        break;

    /* Wait/sync */
    case R200_WAIT_UNTIL:
        /*
         * WAIT_UNTIL is write-to-trigger / read-for-completion.
         *
         * The kext writes a bitmask of conditions to wait for (e.g.
         * WAIT_2D_IDLECLEAN, WAIT_HOST_IDLECLEAN, WAIT_DMA_IDLECLEAN),
         * then polls the register until it reads back 0 (all conditions
         * satisfied).  On real R200 hardware the GPU clears the bits
         * once the corresponding pipeline stages become idle.
         *
         * In this emulation there is no asynchronous pipeline —
         * command processing completes synchronously, and RBBM_STATUS
         * already reports idle.  All wait conditions are therefore
         * trivially satisfied by the time the guest reads back, so
         * we always return 0.
         *
         * The written value is still latched in s->regs.wait_until
         * for tracing/debug purposes.
         */
        val = 0;
        break;
    case R200_ISYNC_CNTL:
        val = s->regs.isync_cntl;
        break;

    /* RBBM soft reset */
    case R200_RBBM_SOFT_RESET:
        val = 0;
        break;

    /* DISP_PWR_MAN */
    case R200_DISP_PWR_MAN:
        val = 0;
        break;

    /* Scratch registers — key fence-completion polling path */
    case R200_SCRATCH_REG0 ... R200_SCRATCH_REG5: {
        int scratch_idx = (addr - R200_SCRATCH_REG0) / 4;
        val = s->regs.scratch_reg[scratch_idx];
        if (getenv("POWEREMU_FENCE_TRACE")) {
            static uint32_t last[6] = { 0xFFFFFFFF }; static int n;
            if (val != last[scratch_idx] && n++ < 40) {
                last[scratch_idx] = val;
                fprintf(stderr, "ppc-mac-gpu fence: guest reads reg%d = %u\n",
                        scratch_idx, val);
            }
        }
        gpu_debug_log("STATUS_RD SCRATCH_REG%d -> 0x%08x",
                      scratch_idx, val);
        /*
         * Phase 3B: Log scratch register reads for QE gate analysis.
         * The guest polls scratch registers to check fence completion.
         * Log the first 50 reads and then periodically.
         */
        {
            static int scratch_read_count = 0;
            scratch_read_count++;
            if (gpu_diag_on() &&
                (scratch_read_count <= 50 || scratch_read_count % 500 == 0)) {
                qemu_log("[QE_GATE_READ] reg=0x%04x value=0x%08x "
                         "context=scratch_poll sequence_id=%d "
                         "note=SCRATCH_REG%d_fence_check\n",
                         (uint32_t)addr, val, scratch_read_count,
                         scratch_idx);
            }
        }
        break;
    }

    /* CRTC2 / misc registers (stubs) */
    case R200_CRTC2_OFFSET_CNTL:  /* 0x0328 */
    case R200_CRTC2_PITCH:        /* 0x023C */
    case R200_FP2_GEN_CNTL:       /* 0x033C */
    case R200_RBBM_CNTL:          /* 0x1710 */
        val = 0;
        break;

    /* 3D engine registers — return shadow values for known 3D regs */
    case R200_RB3D_CNTL:          /* 0x1C3C */
    case R200_SE_CNTL:            /* 0x1C4C */
    case R200_RB3D_DSTCACHE_MODE: /* 0x3258 */
    {
        uint32_t idx = ((uint32_t)addr - 0x1C00) / 4;
        if (idx < 2304) {
            val = s->regs.regs_3d[idx];
        }
        break;
    }
    case R200_RB3D_DSTCACHE_CTLSTAT: /* 0x325C — cache flush status */
    {
        /*
         * Kext writes bit 0 (DC_FLUSH) or bit 1 (DC_FREE) to request
         * a destination cache flush, then polls until those bits clear.
         * We're a software emulator with no cache, so always return 0
         * (flush complete) and clear the shadow value.
         */
        uint32_t idx = ((uint32_t)addr - 0x1C00) / 4;
        if (idx < 2304) {
            uint32_t shadow = s->regs.regs_3d[idx];
            if (shadow & 0x3) {
                gpu_debug_log("STATUS_RD RB3D_DSTCACHE_CTLSTAT -> 0 "
                              "(was 0x%08x, flush complete)", shadow);
            }
            s->regs.regs_3d[idx] = 0; /* clear flush bits */
        }
        val = 0;
        break;
    }
    case R200_AIC_CTRL:           /* 0x01D0 */
        val = s->regs.aic_ctrl;
        break;
    case R200_AIC_PT_BASE:        /* 0x01D8 */
        val = s->regs.aic_pt_base;
        break;
    case R200_AIC_LO_ADDR:        /* 0x01DC */
        val = s->regs.aic_lo_addr;
        break;
    case R200_AIC_HI_ADDR:        /* 0x01E0 */
        val = s->regs.aic_hi_addr;
        break;

    default:
        /*
         * 3D engine register range (0x1C00-0x3FFF) — return shadow values.
         * This allows the kext to read back any 3D register it has written.
         */
        if (addr >= 0x1C00 && addr < 0x4000) {
            uint32_t idx = ((uint32_t)addr - 0x1C00) / 4;
            if (idx < 2304) {
                val = s->regs.regs_3d[idx];
            }
            break;
        }

        /*
         * PCI config space mirror at 0x0F00-0x0F3F.
         * ATI GPUs mirror their PCI config space into MMIO so the driver
         * can read PCI identity (vendor/device ID) without going through
         * the host bridge. The kext reads DEVICE_ID at 0x0F02 to verify
         * the card before doing anything else.
         */
        if (addr >= R200_PCI_MIRROR_BASE && addr < R200_PCI_MIRROR_END) {
            uint32_t pci_off = addr - R200_PCI_MIRROR_BASE;
            val = pci_default_read_config(&s->pci, pci_off, size);
            break;
        }

        /*
         * VBE DISPI registers (0x500-0x514) - qemu_vga.ndrv compatibility.
         * The NDRV accesses these as 16-bit reads at (reg << 1) + 0x500.
         * This provides the Bochs VBE interface for mode query/set.
         */
        if (addr >= PPC_MAC_GPU_VBE_OFFSET &&
            addr < PPC_MAC_GPU_VBE_END) {
            uint32_t vbe_idx = (addr - PPC_MAC_GPU_VBE_OFFSET) >> 1;
            if (vbe_idx < VBE_DISPI_INDEX_NB) {
                val = s->regs.vbe_regs[vbe_idx];
                static const char *vbe_names[] = {
                    "ID", "XRES", "YRES", "BPP", "ENABLE",
                    "BANK", "VIRT_WIDTH", "VIRT_HEIGHT",
                    "X_OFF", "Y_OFF", "VMEM_64K"
                };
                const char *name = vbe_idx < 11 ? vbe_names[vbe_idx] : "?";
                gpu_debug_log("VBE_RD [%u] %s -> 0x%04x",
                              vbe_idx, name, (uint32_t)val);
            }
            break;
        }

        /*
         * VGA DAC registers (0x400-0x41F) - qemu_vga.ndrv palette access.
         * Port mapping: MMIO offset = port + 0x400 - 0x3C0.
         */
        if (addr >= PPC_MAC_GPU_VGA_OFFSET &&
            addr < PPC_MAC_GPU_VGA_END) {
            uint16_t port = (addr - PPC_MAC_GPU_VGA_OFFSET) + 0x3C0;
            switch (port) {
            case 0x3C0:  /* AR Index/Data */
                val = s->regs.vga_ar_index;
                break;
            case 0x3C7:  /* DAC State */
                val = 3;  /* always in read mode state */
                break;
            case 0x3C8:  /* DAC Write Address */
                val = s->regs.vga_dac_write_index;
                break;
            case 0x3C9: {  /* DAC Data (read) */
                uint8_t idx = s->regs.vga_dac_read_index;
                val = s->regs.vga_dac_palette[idx][s->regs.vga_dac_sub_index];
                s->regs.vga_dac_sub_index++;
                if (s->regs.vga_dac_sub_index >= 3) {
                    s->regs.vga_dac_sub_index = 0;
                    s->regs.vga_dac_read_index++;
                }
                break;
            }
            default:
                val = 0;
                break;
            }
            break;
        }

        /*
         * VBE extended registers (0x600-0x61F) - interrupt control.
         * Accessed as 32-bit at (reg << 2) + 0x600.
         */
        if (addr >= PPC_MAC_GPU_EXT_OFFSET &&
            addr < PPC_MAC_GPU_EXT_END) {
            uint32_t ext_reg = (addr - PPC_MAC_GPU_EXT_OFFSET) >> 2;
            if (ext_reg == 2) {
                /* VBL interrupt status register */
                val = s->regs.gen_int_status & R200_CRTC_VBLANK_INT;
            }
            break;
        }

        /* R300 register catch-all: any register in R300-specific ranges
         * that isn't handled by a specific case above gets returned from
         * shadow storage.  This lets ATIRadeon9700.kext read back any
         * R300 register it has written. */
        if (addr < 0x5000 && is_r300_register((uint32_t)addr)) {
            val = s->r300_shadow[(uint32_t)addr / 4];
            break;
        }

        trace_ppc_mac_gpu_mmio_read_unimpl(size, addr,
                                            ppc_mac_gpu_reg_name(addr));
        /*
         * Phase 3B: Log unhandled register reads — these are the most
         * suspicious candidates for QE capability gates returning wrong
         * values (we return 0 for all of them).
         */
        {
            static int unhandled_read_count = 0;
            unhandled_read_count++;
            if (unhandled_read_count <= 100 ||
                unhandled_read_count % 500 == 0) {
                qemu_log("[QE_GATE_READ] reg=0x%04x value=0x%08x "
                         "context=unhandled_read sequence_id=%d "
                         "note=UNIMPL_%s_returns_zero\n",
                         (uint32_t)addr, 0, unhandled_read_count,
                         ppc_mac_gpu_reg_name(addr));
            }
        }
        val = 0;
        break;
    }

    /*
     * Post-probe MMIO read audit (Phases A-D).
     *
     * Audit window: starts when draw_count >= 20 (after initial probe batch),
     * runs for the next 2000 reads, then emits summary.
     * This captures ALL registers the guest reads during/after probe draws.
     *
     * Two sub-windows:
     *   Window 1: draws 20-100 (early probe / during active probing)
     *   Window 2: draws 100-500 (late probe / post-probe steady state)
     * After window 2, emit final summary with ranked registers.
     */
    {
        int dc = g_post_probe_draw_count;

        /* Activate audit once probe draws have started */
        if (dc >= 20 && !g_post_probe_audit.summary_logged) {
            if (!g_post_probe_audit.audit_active) {
                g_post_probe_audit.audit_active = 1;
                g_post_probe_audit.first_draw_seen = dc;
                g_post_probe_audit.window_start_draw = dc;
            }

            /* Record this read */
            g_post_probe_audit.total_reads++;
            int idx = post_probe_find_or_add_reg((uint32_t)addr);
            if (idx >= 0) {
                g_post_probe_audit.entries[idx].count++;
                g_post_probe_audit.entries[idx].last_value = (uint32_t)val;
                g_post_probe_audit.entries[idx].last_at_draw = dc;
            }

            /*
             * Detailed per-read logging: log first 200 individual reads,
             * then every 500th read, to see the exact sequence.
             */
            int tr = g_post_probe_audit.total_reads;
            if (tr <= 200 || tr % 500 == 0) {
                qemu_log("[POST_PROBE_READ] sequence_id=%d "
                         "reg=0x%04x value=0x%08x "
                         "count=%d context=%s "
                         "delta_from_fence=%d "
                         "note=%s\n",
                         tr,
                         (uint32_t)addr, (uint32_t)val,
                         idx >= 0 ? g_post_probe_audit.entries[idx].count : 0,
                         addr == R200_RBBM_STATUS ? "status_poll" :
                         (addr >= R200_SCRATCH_REG0 &&
                          addr <= R200_SCRATCH_REG5) ? "fence" :
                         addr == R200_CP_STAT ? "cp_status" :
                         (addr >= 0x1C00 && addr < 0x4000) ? "3d_reg" :
                         "other",
                         dc,
                         ppc_mac_gpu_reg_name(addr));
            }

            /* Emit PROBE_WINDOW marker at window boundaries */
            if (!g_post_probe_audit.window_logged && dc >= 100) {
                g_post_probe_audit.window_logged = 1;
                g_post_probe_audit.window_end_draw = dc;
                qemu_log("[PROBE_WINDOW] "
                         "first_probe_draw=%d "
                         "window_begin=%d "
                         "window_end=%d "
                         "total_reads=%d "
                         "unique_regs=%d "
                         "note=early_window_complete\n",
                         g_post_probe_audit.first_draw_seen,
                         g_post_probe_audit.window_start_draw,
                         dc,
                         g_post_probe_audit.total_reads,
                         g_post_probe_audit.num_entries);
            }

            /* Emit final summary after draw 400 or after 5000 reads */
            if ((dc >= 400 || tr >= 5000) && !g_post_probe_audit.summary_logged) {
                g_post_probe_audit.summary_logged = 1;

                /* Sort entries by count (descending) via simple selection */
                for (int i = 0; i < g_post_probe_audit.num_entries - 1; i++) {
                    for (int j = i + 1; j < g_post_probe_audit.num_entries; j++) {
                        if (g_post_probe_audit.entries[j].count >
                            g_post_probe_audit.entries[i].count) {
                            PostProbeReadEntry tmp = g_post_probe_audit.entries[i];
                            g_post_probe_audit.entries[i] =
                                g_post_probe_audit.entries[j];
                            g_post_probe_audit.entries[j] = tmp;
                        }
                    }
                }

                qemu_log("[POST_PROBE_READ_SUMMARY] "
                         "total_reads=%d unique_regs=%d "
                         "draw_range=%d-%d\n",
                         g_post_probe_audit.total_reads,
                         g_post_probe_audit.num_entries,
                         g_post_probe_audit.window_start_draw, dc);

                /* Log top registers by read count */
                for (int i = 0; i < g_post_probe_audit.num_entries && i < 30;
                     i++) {
                    PostProbeReadEntry *e = &g_post_probe_audit.entries[i];
                    const char *ctx =
                        e->reg == R200_RBBM_STATUS ? "STATUS_POLL" :
                        (e->reg >= R200_SCRATCH_REG0 &&
                         e->reg <= R200_SCRATCH_REG5) ? "FENCE" :
                        e->reg == R200_CP_STAT ? "CP_STATUS" :
                        (e->reg >= 0x1C00 && e->reg < 0x4000) ? "3D_REG" :
                        (e->reg == R200_CRTC_VLINE_CRNT_VLINE ||
                         e->reg == R200_CRTC_STATUS ||
                         e->reg == R200_CRTC_CRNT_FRAME) ? "CRTC" :
                        (e->reg >= 0x0F00 && e->reg < 0x0F40) ? "PCI_MIRROR" :
                        "OTHER";

                    bool suspicious = false;
                    const char *suspicion = "";
                    /* Flag registers that return 0 and might be wrong */
                    if (e->last_value == 0 && e->count >= 5) {
                        suspicious = true;
                        suspicion = "static_zero_high_freq";
                    }
                    /* Flag 3D regs read back during probe */
                    if (e->reg >= 0x1C00 && e->reg < 0x4000 &&
                        e->count >= 3) {
                        suspicious = true;
                        suspicion = "3d_reg_readback_during_probe";
                    }

                    qemu_log("[POST_PROBE_REG_RANK] rank=%d reg=0x%04x "
                             "name=%s count=%u "
                             "last_value=0x%08x first_draw=%u last_draw=%u "
                             "context=%s suspicious=%d suspicion=%s\n",
                             i + 1, e->reg,
                             ppc_mac_gpu_reg_name(e->reg),
                             e->count,
                             e->last_value, e->first_at_draw, e->last_at_draw,
                             ctx, suspicious,
                             suspicious ? suspicion : "none");
                }

                /* Identify gate candidates */
                qemu_log("[QE_GATE_CANDIDATE] "
                         "note=see_POST_PROBE_REG_RANK_above "
                         "look_for=static_zero_or_3d_readback_regs\n");
            }
        }
    }

    if (unlikely(trace_event_get_state(TRACE_PPC_MAC_GPU_MMIO_READ))) {
        trace_ppc_mac_gpu_mmio_read(size, addr, ppc_mac_gpu_reg_name(addr), val);
    }
    if (unlikely(s->r3_dump && s->r3 && s->r3->draws && s->r3_trace_lines < 40000)) {
        static uint32_t last_addr, repeats;
        if (addr == last_addr && ++repeats > 3) {
            return val;             /* collapse polling */
        }
        if (addr != last_addr) {
            repeats = 0;
        }
        last_addr = addr;
        s->r3_trace_lines++;
        fprintf(s->r3_dump, "R %04x %08x %s\n", (unsigned)addr, (uint32_t)val,
                ppc_mac_gpu_reg_name(addr));
    }
    return val;
}

/* ========================================================================
 * MMIO write handler
 * ======================================================================== */

/*
 * R200 TCL constant-memory ports.  The index registers take an address in
 * bits 15:0 and a stride in bits 23:16 (octwords for vectors, dwords for
 * scalars); each data write stores one component and auto-increments —
 * four dwords per vector, then the address advances by the stride.
 */
static void ppc_mac_gpu_tcl_port_write(PPCMacGPUState *s, hwaddr addr,
                                       uint32_t val)
{
    switch (addr) {
    case 0x2200:    /* SE_TCL_VECTOR_INDX_REG */
        s->regs.tcl_vec_addr = val & 0x7ff;
        s->regs.tcl_vec_stride = (val >> 16) & 0xff;
        s->regs.tcl_vec_comp = 0;
        break;
    case 0x2204:    /* SE_TCL_VECTOR_DATA_REG */
        s->regs.tcl_vec[s->regs.tcl_vec_addr & 0x7ff][s->regs.tcl_vec_comp] = val;
        if (++s->regs.tcl_vec_comp == 4) {
            uint32_t a = s->regs.tcl_vec_addr & 0x7ff;
            /* POWEREMU_VP_TRACE: the vertex programs the guest uploads.
             * Instructions sit at 0x080-0x0BF and 0x180-0x1BF, four dwords
             * each; everything else here is constants, matrices or lights. */
            s->regs.tcl_vec_comp = 0;
            s->regs.tcl_vec_addr += s->regs.tcl_vec_stride ? s->regs.tcl_vec_stride : 1;
        }
        break;
    case 0x2208:    /* SE_TCL_SCALAR_INDX_REG */
        s->regs.tcl_scalar_addr = val & 0x1ff;
        s->regs.tcl_scalar_stride = (val >> 16) & 0xff;
        break;
    case 0x220C:    /* SE_TCL_SCALAR_DATA_REG */
        if (getenv("POWEREMU_VP_TRACE")) {
            static uint8_t seen_scalar[0x200];
            uint32_t sa = s->regs.tcl_scalar_addr & 0x1ff;
            if (!seen_scalar[sa]) {
                seen_scalar[sa] = 1;
                fprintf(stderr, "ppc-mac-gpu scalar[%03x] %08x\n", sa, val);
            }
        }
        s->regs.tcl_scalar[s->regs.tcl_scalar_addr & 0x1ff] = val;
        s->regs.tcl_scalar_addr += s->regs.tcl_scalar_stride ? s->regs.tcl_scalar_stride : 1;
        break;
    }
}


static void ppc_mac_gpu_mmio_write(void *opaque, hwaddr addr,
                                    uint64_t val, unsigned int size)
{
    PPCMacGPUState *s = opaque;

    if (addr >= PPC_MAC_GPU_HWC_BASE && addr < PPC_MAC_GPU_HWC_END) {
        if (size == 4) {
            ppc_mac_gpu_hwc_write(s, addr - PPC_MAC_GPU_HWC_BASE, val);
        }
        return;
    }
    if (addr >= 0x780 && addr < 0x7C0) {
        seq_log("DMA  wr reg=%03x val=%08x", (unsigned)addr, (uint32_t)val);
    }
    /* No flush here: CPU-side 2D operations check their own VRAM ranges
     * (r200_vram_access), fences complete asynchronously, and the other
     * registers do not touch VRAM. */
    if (unlikely(r200_traffic_on())) {
        if (r200_in_pm4) {
            r200_traffic.type0_regs++;
        } else {
            r200_traffic.mmio_writes++;
        }
        r200_traffic_tick();
    }
    /*
     * Every PM4 type-0 register also lands here -- about a million a second
     * under a game -- so nothing on this path may cost anything while the
     * loggers are off.
     */
    if (unlikely(gpu_debug_enabled())) {
        const char *nm = ppc_mac_gpu_reg_name(addr);

        gpu_debug_log("MMIO_WR size=%u addr=0x%04"PRIx64" val=0x%"PRIx64" (%s)",
                      size, (uint64_t)addr, val, nm);
        trace_ppc_mac_gpu_mmio_write(size, addr, nm, val);
    }

    if (s->r3 && r300_state_owns(addr)) {
        r300_state_write(s->r3, addr, val);
        if (addr == 0x4F58 || addr == 0x4F5C) {
            r300_zpass_write(s, addr, val);
        }
    }
    if (s->r300 && addr >= 0x0B00 && addr < 0x0B80) {
        s->r300_surf[(addr - 0x0B00) / 4] = val;   /* SURFACEn bounds/info */
        if (addr == 0x0B0C) {
            r300_surface_changed(s);
            r300_surfwatch_update(s);
        }
    }
    if (s->r300 && addr == 0x0AB0) {
        s->r300_aic_pt_base = val & ~0xFFFu;    /* PCI GART table base */
        r200_agp_tc_flush();
    }
    if (unlikely(s->r3_dump && s->r3->draws && s->r3_trace_lines < 40000)) {
        const char *nm = r300_state_owns(addr) ? r300_reg_name(addr) : NULL;
        s->r3_trace_lines++;
        fprintf(s->r3_dump, "W %04x %08x %s%s\n", (unsigned)addr, (uint32_t)val,
                nm ? nm : ppc_mac_gpu_reg_name(addr), r200_in_pm4 ? "" : " (mmio)");
    }

    /* Handle indexed register access */
    if (addr == R200_MM_DATA && s->regs.mm_index) {
        ppc_mac_gpu_mmio_write(s, s->regs.mm_index, val, size);
        return;
    }

    switch (addr) {
    case R200_MM_INDEX:
        s->regs.mm_index = val & 0xFFFF;
        break;

    /* BIOS scratch */
    case R200_BIOS_0_SCRATCH ... R200_BIOS_7_SCRATCH:
        s->regs.bios_scratch[(addr - R200_BIOS_0_SCRATCH) / 4] = val;
        break;

    /* Interrupt */
    case R200_GEN_INT_CNTL:
        s->regs.gen_int_cntl = val;
        if (val & R200_CRTC_VBLANK_INT) {
            /* Enable VBlank timer */
            timer_mod(&s->vblank_timer,
                      qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                      NANOSECONDS_PER_SECOND / 60);
        } else {
            timer_del(&s->vblank_timer);
        }
        ppc_mac_gpu_update_irq(s);
        break;
    case R200_GEN_INT_STATUS:
        /* Writing 1 to a bit acknowledges/clears it */
        s->regs.gen_int_status &= ~val;
        ppc_mac_gpu_update_irq(s);
        break;

    /* Config */
    case R200_CONFIG_CNTL:
        s->regs.config_cntl = val;
        break;
    case R200_MC_FB_LOCATION:
        qemu_log("ppc-mac-gpu: MC_FB_LOCATION 0x%08x -> 0x%08x\n",
                 s->regs.mc_fb_location, val);
        s->regs.mc_fb_location = val;
        break;
    case R200_MC_AGP_LOCATION:
        qemu_log("ppc-mac-gpu: MC_AGP_LOCATION 0x%08x -> 0x%08x\n",
                 s->regs.mc_agp_location, val);
        s->regs.mc_agp_location = val;
        break;

    /* Bus */
    case R200_BUS_CNTL:
        s->regs.bus_cntl = val;
        break;
    case R200_HOST_PATH_CNTL:
        s->regs.host_path_cntl = val;
        break;

    /* RBBM soft reset */
    case R200_RBBM_SOFT_RESET:
        /* Acknowledge the reset request, but don't actually reset much */
        trace_ppc_mac_gpu_soft_reset(val);
        break;

    /* Surface */
    case R200_SURFACE_CNTL:
        s->regs.surface_cntl = val;
        break;

    /*
     * R300/RV350 register write stubs — absorb writes from ATIRadeon9700.kext
     * initialization without crashing.  These are stored in the generic
     * shadow array but not functionally used.
     */
    case 0x4018: /* R300_GB_TILE_CONFIG */
    case 0x4024: /* R300_GB_FIFO_SIZE */
    case 0x4008: /* R300_GB_ENABLE */
    case 0x4010: /* R300_GB_MSPOS0 */
    case 0x4014: /* R300_GB_MSPOS1 */
    case 0x401C: /* R300_GB_SELECT */
    case 0x4020: /* R300_GB_AA_CONFIG */
    case 0x2140: /* R300_VAP_CNTL_STATUS */
    case 0x2150: /* R300_VAP_PVS_STATE_FLUSH_REG */
    case 0x170C: /* R300_RBBM_SOFT_RESET */
    case 0x4100: /* R300_TX_ENABLE */
    case 0x4200: /* R300_TX_FILTER0_0 */
    case 0x4204: /* R300_TX_FILTER0_1 */
    case 0x4210: /* R300_TX_FORMAT0_0 */
    case 0x4230: /* R300_TX_SIZE_0 */
    case 0x4250: /* R300_TX_PITCH_0 */
    case 0x4280: /* R300_TX_OFFSET_0 */
    case 0x42C0: /* R300_TX_BORDER_COLOR_0 */
    case 0x46A4: /* R300_US_OUT_FMT */
    case 0x4600: /* R300_US_CONFIG */
    case 0x4604: /* R300_US_PIXSIZE */
    case 0x4608: /* R300_US_CODE_OFFSET */
    case 0x4610: /* R300_US_CODE_ADDR */
    case 0x4614: /* R300_US_CODE_RANGE */
    case 0x4620: /* R300_US_ALU_RGB_ADDR_0 */
    case 0x4624: case 0x4628: case 0x462C:
    case 0x4630: case 0x4634: case 0x4638: case 0x463C:
    case 0x4640: case 0x4644: case 0x4648: case 0x464C:
    case 0x4650: case 0x4654: case 0x4658: case 0x465C:
    case 0x4660: case 0x4664: case 0x4668: case 0x466C:
    case 0x4670: case 0x4674: case 0x4678: case 0x467C:
    case 0x4680: case 0x4684: case 0x4688: case 0x468C:
    case 0x4690: case 0x4694: case 0x4698: case 0x469C:
    case 0x46A0: /* R300 US ALU registers */
    case 0x4F00: /* R300_ZB_CNTL */
    case 0x4F04: /* R300_ZB_ZSTENCILCNTL */
    case 0x4F08: /* R300_ZB_STENCILREFMASK */
    case 0x4F14: /* R300_ZB_DEPTHOFFSET */
    case 0x4F18: /* R300_ZB_DEPTHPITCH */
    case 0x4F1C: /* R300_ZB_DEPTHCLEARVALUE */
    case 0x4F28: /* R300_ZB_ZMASK_OFFSET */
    case 0x4F2C: /* R300_ZB_ZMASK_PITCH */
        /* Silently absorb R300 register writes, store for readback */
        if (addr < 0x5000) {
            s->r300_shadow[(uint32_t)addr / 4] = val;
        }
        break;

    /* CRTC */
    case R200_CRTC_GEN_CNTL:
        s->regs.crtc_gen_cntl = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_EXT_CNTL:
        s->regs.crtc_ext_cntl = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_H_TOTAL_DISP:
        s->regs.crtc_h_total_disp = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_H_SYNC_STRT_WID:
        s->regs.crtc_h_sync_strt_wid = val;
        break;
    case R200_CRTC_V_TOTAL_DISP:
        s->regs.crtc_v_total_disp = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_V_SYNC_STRT_WID:
        s->regs.crtc_v_sync_strt_wid = val;
        break;
    case R200_CRTC_OFFSET:
        if (s->rv250) {
            val &= ~R200_CRTC_OFFSET_BIAS;
        }
        if (val != s->regs.crtc_offset) {
            blit_path_log("CRTC", "OFFSET changed 0x%x -> 0x%x",
                          s->regs.crtc_offset, val);
            r200_rate.flips++;             /* page flip: one frame */
            r200_perf_present();
            /*
             * A flip is the frame boundary the guest itself defines: go
             * get it now instead of waiting for the next poll, so a
             * 60 Hz video is not sampled and presented on a 30 Hz grid.
             * Kicks no closer than ~8 ms apart coalesce anyway (the UI
             * refresh re-reads the current state), so this is a floor on
             * the cost, not a per-flip interrupt storm.
             */
            if (ppc_gpu_flip_kick) {
                int64_t now = g_get_monotonic_time();
                if (now - s->last_flip_kick >= 8000) {
                    s->last_flip_kick = now;
                    dpy_refresh_soon(s->con);
                }
            }
        }
        s->regs.crtc_offset = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_OFFSET_CNTL:
        if (val != s->regs.crtc_offset_cntl) {
            blit_path_log("CRTC", "OFFSET_CNTL changed 0x%x -> 0x%x",
                          s->regs.crtc_offset_cntl, val);
        }
        s->regs.crtc_offset_cntl = val;
        break;
    case R200_CRTC_PITCH:
        if (val != s->regs.crtc_pitch) {
            blit_path_log("CRTC", "PITCH changed 0x%x -> 0x%x",
                          s->regs.crtc_pitch, val);
        }
        s->regs.crtc_pitch = val;
        s->display_invalid = true;
        break;
    case R200_CRTC_GUI_TRIG_VLINE:
        /* Write-only trigger, no state to store */
        break;

    /* DAC */
    case R200_DAC_CNTL:
        s->regs.dac_cntl = val;
        break;
    case R200_DAC_CNTL2:
        s->regs.dac_cntl2 = val;
        break;

    /* Palette */
    case R200_PALETTE_INDEX:
        s->regs.palette_index = val & 0xFF;
        break;
    case R200_PALETTE_DATA:
        if (s->regs.palette_index < 256) {
            s->regs.palette[s->regs.palette_index] = val;
            s->regs.palette_index = (s->regs.palette_index + 1) & 0xFF;
        }
        break;

    /* Cursor */
    case R200_CUR_OFFSET:
        s->regs.cur_offset = val;
        break;
    case R200_CUR_HORZ_VERT_POSN:
        s->regs.cur_horz_vert_posn = val;
        break;
    case R200_CUR_HORZ_VERT_OFF:
        s->regs.cur_horz_vert_off = val;
        break;
    case R200_CUR_CLR0:
        s->regs.cur_clr0 = val;
        break;
    case R200_CUR_CLR1:
        s->regs.cur_clr1 = val;
        break;

    /* Display */
    case R200_DISP_MISC_CNTL:
        s->regs.disp_misc_cntl = val;
        break;
    case R200_DISP_OUTPUT_CNTL:
        s->regs.disp_output_cntl = val;
        break;
    case R200_DISP_MERGE_CNTL:
        s->regs.disp_merge_cntl = val;
        break;
    case R200_DISP_PWR_MAN:
        /* Power management stub - accept and ignore */
        break;
    case R200_FP_GEN_CNTL:
        s->regs.fp_gen_cntl = val;
        break;

    /* AGP */
    case R200_AGP_BASE:
        s->regs.agp_base = val;
        break;
    case R200_AGP_CNTL:
        s->regs.agp_cntl = val;
        break;
    case R200_AGP_COMMAND:
        s->regs.agp_command = val;
        break;

    /* Clock/PLL - accept writes but don't actually change clocks */
    case R200_CLOCK_CNTL_INDEX:
        s->regs.clock_cntl_index = val;
        break;
    case R200_CLOCK_CNTL_DATA:
        s->regs.clock_cntl_data = val;
        break;

    /* 2D engine registers - store state */
    case R200_SRC_PITCH_OFFSET:
        s->regs.src_pitch_offset = val;
        /* Register/unregister tiled surface from SRC_PITCH_OFFSET */
        {
            uint32_t spo_base  = (val & 0x3FFFFF) << 10;
            uint32_t spo_pitch = (val & 0x3FC00000) >> 16;
            bool spo_macro     = (val >> 30) & 1;
            if (spo_macro && spo_base > 0 && spo_pitch > 0) {
                mc_register_tiled_surface(s, spo_base, spo_pitch, "SRC_PO");
            }
        }
        break;
    case R200_DST_PITCH_OFFSET:
        s->regs.dst_pitch_offset = val;
        /* Register/unregister tiled surface from DST_PITCH_OFFSET */
        {
            uint32_t dpo_base  = (val & 0x3FFFFF) << 10;
            uint32_t dpo_pitch = (val & 0x3FC00000) >> 16;
            bool dpo_macro     = (val >> 30) & 1;
            if (dpo_macro && dpo_base > 0 && dpo_pitch > 0) {
                mc_register_tiled_surface(s, dpo_base, dpo_pitch, "DST_PO");
            }
        }
        break;
    case R200_DST_OFFSET:
        s->regs.dst_offset = val;
        break;
    case R200_DST_PITCH:
        s->regs.dst_pitch = val;
        break;
    case R200_DST_WIDTH:
        /*
         * Writing DST_WIDTH starts a blit from the separate SRC_X/SRC_Y/
         * DST_X/DST_Y/DST_HEIGHT registers.  ATIRadeon9700 copies a surface
         * into its backing store in system memory this way
         * (ATIR300Surface::write_2dblit_cmds_for_copy_buffer_using_DMA, the
         * IOAccelerator read lock); without it every QuartzGL window's
         * backing store stayed white, and HIToolbox reopened menus from
         * that blank copy.
         */
        s->regs.dst_width = val;
        s->regs.dst_width_height = ((val & 0x3FFF) << 16) |
                                   (s->regs.dst_height & 0x3FFF);
        if ((val & 0x3FFF) && (s->regs.dst_height & 0x3FFF) &&
            s->regs.dst_pitch > 0) {
            ppc_mac_gpu_2d_blit_sep(s);
        }
        break;
    case R200_DST_HEIGHT:
        s->regs.dst_height = val;
        break;
    case 0x1414:                                    /* SRC_X */
        s->regs.src_y_x = (s->regs.src_y_x & 0xFFFF0000) | (val & 0xFFFF);
        break;
    case 0x1418:                                    /* SRC_Y */
        s->regs.src_y_x = (val << 16) | (s->regs.src_y_x & 0xFFFF);
        break;
    case 0x141C:                                    /* DST_X */
        s->regs.dst_y_x = (s->regs.dst_y_x & 0xFFFF0000) | (val & 0xFFFF);
        break;
    case 0x1420:                                    /* DST_Y */
        s->regs.dst_y_x = (val << 16) | (s->regs.dst_y_x & 0xFFFF);
        break;
    case R200_DST_HEIGHT_WIDTH: {
        /* Alternate blit trigger with h|w format */
        uint32_t h = (val >> 16) & 0x3FFF;
        uint32_t w = val & 0x3FFF;
        s->regs.dst_width_height = (w << 16) | h;
        if (w > 0 && h > 0 && s->regs.dst_pitch > 0) {
            ppc_mac_gpu_2d_blit_sep(s);
        }
        break;
    }
    case R200_DST_WIDTH_HEIGHT: {
        /* Blit trigger with w|h format (0x1598) */
        s->regs.dst_width_height = val;
        uint32_t w2 = (val >> 16) & 0x3FFF;
        uint32_t h2 = val & 0x3FFF;
        if (w2 > 0 && h2 > 0) {
            ppc_mac_gpu_2d_blit(s);
        }
        break;
    }
    case R200_DST_Y_X:
        s->regs.dst_y_x = val;
        break;
    case R200_DST_X_Y: {
        /* DST_X_Y: bits[31:16]=X, bits[15:0]=Y → convert to Y_X */
        uint32_t x = (val >> 16) & 0xFFFF;
        uint32_t y = val & 0xFFFF;
        s->regs.dst_y_x = (y << 16) | x;
        break;
    }
    case R200_SRC_OFFSET:
        s->regs.src_offset = val;
        break;
    case R200_SRC_PITCH:
        s->regs.src_pitch = val;
        break;
    case R200_SRC_Y_X:
        s->regs.src_y_x = val;
        break;
    case R200_SRC_X_Y: {
        /* SRC_X_Y: bits[31:16]=X, bits[15:0]=Y → convert to Y_X */
        uint32_t x = (val >> 16) & 0xFFFF;
        uint32_t y = val & 0xFFFF;
        s->regs.src_y_x = (y << 16) | x;
        break;
    }
    case R200_DP_GUI_MASTER_CNTL:
        s->regs.dp_gui_master_cntl = val;
        break;
    case R200_DP_BRUSH_FRGD_CLR:
        s->regs.dp_brush_frgd_clr = val;
        break;
    case R200_DP_BRUSH_BKGD_CLR:
        s->regs.dp_brush_bkgd_clr = val;
        break;
    case R200_DP_SRC_FRGD_CLR:
        s->regs.dp_src_frgd_clr = val;
        break;
    case R200_DP_SRC_BKGD_CLR:
        s->regs.dp_src_bkgd_clr = val;
        break;
    case R200_DP_WRITE_MSK:
        s->regs.dp_write_msk = val;
        break;
    case R200_DP_CNTL:
        s->regs.dp_cntl = val;
        break;
    case R200_DP_DATATYPE:
        s->regs.dp_datatype = val;
        /* DP_DATATYPE and DP_GUI_MASTER_CNTL share the dst datatype. */
        s->regs.dp_gui_master_cntl = (s->regs.dp_gui_master_cntl & ~0xF00u) |
                                     ((val & 0xF) << 8);
        break;
    case R200_DP_MIX:
        s->regs.dp_mix = val;
        break;
    case R200_SC_TOP_LEFT:
        s->regs.sc_top_left = val;
        break;
    case R200_SC_BOTTOM_RIGHT:
        s->regs.sc_bottom_right = val;
        break;
    case R200_DEFAULT_SC_BOTTOM_RIGHT:
        s->regs.default_sc_bottom_right = val;
        break;
    case R200_DEFAULT_PITCH_OFFSET:
        /*
         * The pitch-offset a 2D operation falls back to when its GMC
         * pitch-offset control bit is clear.  This was once ignored as
         * "not display timing", but it is load-bearing: Apple's driver
         * issues BITBLT_MULTI window saves with GMC_DST_PITCH_OFFSET_CNTL
         * clear and the compositor staging buffer programmed here.
         */
        s->regs.default_pitch_offset = val;
        break;

    /* CP - command processor */
    case R200_CP_RB_BASE:
        s->regs.cp_rb_base = val;
        gpu_debug_log("CP_SETUP RB_BASE=0x%08x", val);
        trace_ppc_mac_gpu_cp_ring_setup(val, s->regs.cp_rb_cntl);
        break;
    case R200_CP_RB_RPTR_ADDR:
        s->regs.cp_rb_rptr_addr = val;
        ppc_mac_gpu_rptr_writeback(s);
        break;
    case R200_CP_RB_CNTL:
        s->regs.cp_rb_cntl = val;
        gpu_debug_log("CP_SETUP RB_CNTL=0x%08x (log2size=%u)", val, val & 0x3f);
        break;
    case R200_CP_RB_RPTR:
        s->regs.cp_rb_rptr = val;
        ppc_mac_gpu_rptr_writeback(s);
        gpu_debug_log("CP_RING RPTR <- %u", val);
        break;
    case R200_CP_RB_RPTR_WR:
        /*
         * How the driver resets the ring: RB_CNTL with RB_RPTR_WR_ENA set,
         * the new read pointer here (0), then WPTR = 0.  Mac OS X's ATI
         * driver does it at start-up and again later (three times in a
         * Quake III timedemo run, the last as the game starts).  Ignoring
         * it left the read pointer where it was, so the WPTR = 0 that
         * followed made the rest of the ring look new: the previous lap,
         * replayed -- IBs whose buffers had since been refilled, run at
         * their old sizes, until a draw's index data parsed as a packet
         * that kicked an IB of garbage, which kicked itself forever.
         * The CP thread may be part-way through a copy of the old ring;
         * bump the epoch so it drops it.
         */
        if (s->regs.cp_rb_cntl & R200_RB_RPTR_WR_ENA) {
            uint32_t rb_bufsz = s->regs.cp_rb_cntl & 0x3F;
            uint32_t mask = rb_bufsz && rb_bufsz <= 25 ?
                            (2U << rb_bufsz) - 1 : UINT32_MAX;
            s->regs.cp_rb_rptr = val & mask;
            cp.epoch++;
            ppc_mac_gpu_rptr_writeback(s);
            gpu_debug_log("CP_RING RPTR_WR <- %u", val);
        }
        break;
    case R200_CP_RB_WPTR:
        /* Normally the lock-free region takes this; MM_DATA lands here. */
        gpu_debug_log("CP_RING WPTR <- %u (prev=%u, delta=%d)",
                      (uint32_t)val, s->regs.cp_rb_wptr,
                      (int)val - (int)s->regs.cp_rb_wptr);
        ppc_mac_gpu_cp_kick(s, val);
        break;
    case R200_CP_ME_CNTL:
        s->regs.cp_me_cntl = val;
        gpu_debug_log("CP_SETUP ME_CNTL=0x%08x (ME_start=%d)", val, !(val & 0x10000000));
        break;
    case R200_CP_IB_BASE:
        s->regs.cp_ib_base = val;
        gpu_debug_log("CP_IB BASE <- 0x%08x", val);
        break;
    case R200_CP_IB_BUFSZ:
        s->regs.cp_ib_bufsz = val;
        gpu_debug_log("CP_IB BUFSZ <- %u dwords (KICK! base=0x%08x)", val, s->regs.cp_ib_base);
        /* Writing BUFSZ triggers IB execution */
        ppc_mac_gpu_execute_ib(s, s->regs.cp_ib_base, val);
        /*
         * NOTE: Do NOT auto-complete scratch registers here.
         *
         * The IB command stream itself contains PM4 type0 packets that
         * write fence values to SCRATCH_REG0-5.  Those writes flow through
         * ppc_mac_gpu_mmio_write() → scratch_writeback() during IB
         * processing, so the correct fence value is already written to
         * the GART/VRAM writeback address by the time the IB finishes.
         *
         * Previously this code unconditionally overwrote all armed scratch
         * registers with 1 after every IB, which corrupted monotonically
         * increasing fence tokens (e.g. guest writes fence=0x15 via IB,
         * then auto-complete overwrites with 1; guest polls for >=0x15,
         * sees 1, thinks fence hasn't completed).
         *
         * Removing auto-complete matches real R200 hardware behavior:
         * the CP only updates scratch registers when it processes an
         * explicit SCRATCH_REG write packet in the command stream.
         */
        break;
    case R200_CP_CSQ_CNTL:
        /* Store value - kext polls until read-back matches */
        s->regs.cp_csq_cntl = val;
        gpu_debug_log("CP_SETUP CSQ_CNTL=0x%08x (mode=%u)", val, (val >> 28) & 0xf);
        break;
    case R200_SCRATCH_UMSK:
        s->regs.scratch_umsk = val;
        break;
    case R200_SCRATCH_ADDR:
        s->regs.scratch_addr = val;
        break;
    case R200_CP_ME_RAM_ADDR:
        s->regs.cp_me_ram_addr = val;
        break;
    case R200_CP_ME_RAM_DATAH:
        /* CP microcode data high - accept and discard */
        break;
    case R200_CP_ME_RAM_DATAL:
        /* CP microcode data low - accept, auto-increment address */
        s->regs.cp_me_ram_addr++;
        break;
    case R200_CP_CSQ_IND_ADDR:
        /* Indirect CSQ buffer address - accept */
        break;
    case R200_CP_CSQ_IND_DATA:
        /* Indirect CSQ buffer data - accept */
        break;

    /* GPIO / I2C / DDC - RV280 uses DVI_DDC for I2C EDID access */
    case R200_GPIO_VGA_DDC ... R200_GPIO_VGA_DDC + 3:
        ppc_mac_gpu_reg_write_offs(&s->regs.gpio_vga_ddc,
                                    addr - R200_GPIO_VGA_DDC, val, size);
        if ((addr <= R200_GPIO_VGA_DDC + 2 &&
             addr + size > R200_GPIO_VGA_DDC + 2) ||
            (addr == R200_GPIO_VGA_DDC &&
             (s->regs.gpio_vga_ddc & 0x30000))) {
            s->regs.gpio_vga_ddc = ppc_mac_gpu_i2c(&s->bbi2c,
                                                     s->regs.gpio_vga_ddc, 0);
        }
        break;
    case R200_GPIO_DVI_DDC ... R200_GPIO_DVI_DDC + 3:
        ppc_mac_gpu_reg_write_offs(&s->regs.gpio_dvi_ddc,
                                    addr - R200_GPIO_DVI_DDC, val, size);
        if ((addr <= R200_GPIO_DVI_DDC + 2 &&
             addr + size > R200_GPIO_DVI_DDC + 2) ||
            (addr == R200_GPIO_DVI_DDC &&
             (s->regs.gpio_dvi_ddc & 0x30000))) {
            s->regs.gpio_dvi_ddc = ppc_mac_gpu_i2c(&s->bbi2c,
                                                     s->regs.gpio_dvi_ddc, 0);
        }
        break;
    case R200_GPIO_MONID ... R200_GPIO_MONID + 3:
        ppc_mac_gpu_reg_write_offs(&s->regs.gpio_monid,
                                    addr - R200_GPIO_MONID, val, size);
        if ((addr <= R200_GPIO_MONID + 2 &&
             addr + size > R200_GPIO_MONID + 2) ||
            (addr == R200_GPIO_MONID &&
             (s->regs.gpio_monid & 0x30000))) {
            s->regs.gpio_monid = ppc_mac_gpu_i2c(&s->bbi2c,
                                                   s->regs.gpio_monid, 0);
        }
        break;

    /* Wait/sync */
    case R200_WAIT_UNTIL:
        /* Writing WAIT_UNTIL on real hardware causes the CP to stall until
         * the specified conditions are met. Since we execute everything
         * synchronously, all conditions are always met instantly.
         * Store 0 to indicate all waits satisfied (not the pending mask). */
        s->regs.wait_until = 0;
        break;
    case R200_ISYNC_CNTL:
        s->regs.isync_cntl = val;
        break;

    /* Scratch registers with writeback */
    case R200_SCRATCH_REG0 ... R200_SCRATCH_REG5: {
        int idx = (addr - R200_SCRATCH_REG0) / 4;

        s->regs.scratch_reg[idx] = val;
        r200_scratch_write(s, idx, val);
        break;
    }

    /* PM4 PIO FIFO data registers */
    case R200_PM4_FIFO_DATA_BASE ... (R200_PM4_FIFO_DATA_END - 1):
        ppc_mac_gpu_pm4_fifo_push(s, val);
        break;

    /* HOST_DATA - for host pixel uploads */
    case R200_HOST_DATA0 ... R200_HOST_DATA_LAST:
        ppc_mac_gpu_host_data_write(s, val);
        break;

    /* GART (AGP Intelligent Controller) registers */
    case R200_AIC_CTRL:           /* 0x01D0 */
        s->regs.aic_ctrl = val;
        gpu_debug_log("GART AIC_CTRL=0x%x (enabled=%d)", val, val & 1);
        break;
    case R200_AIC_PT_BASE:        /* 0x01D8 */
        s->regs.aic_pt_base = val;
        gpu_debug_log("GART PT_BASE=0x%x", val);
        break;
    case R200_AIC_LO_ADDR:        /* 0x01DC */
        s->regs.aic_lo_addr = val;
        gpu_debug_log("GART LO_ADDR=0x%x", val);
        break;
    case R200_AIC_HI_ADDR:        /* 0x01E0 */
        s->regs.aic_hi_addr = val;
        gpu_debug_log("GART HI_ADDR=0x%x", val);
        break;

    /* CRTC2 / misc stubs - accept writes silently */
    case R200_CRTC2_OFFSET_CNTL:  /* 0x0328 */
    case R200_CRTC2_PITCH:        /* 0x023C */
    case R200_FP2_GEN_CNTL:       /* 0x033C */
    /* NOTE: RB3D_DSTCACHE_MODE (0x3258) removed from this group —
     * it's in the 3D register range and must be stored in regs_3d[]
     * so the kext can read back written values. */
    case R200_RBBM_CNTL:          /* 0x1710 */
    case R200_SUBPIC2_CNTL:       /* 0x043C */
    case R200_OV0_SCALE_CNTL:     /* 0x0420 */
    case R200_SUBPIC_CNTL:        /* 0x0540 */
    case R200_CAP0_TRIG_CNTL:     /* 0x0950 */
    case R200_DISP2_MERGE_CNTL:   /* 0x0D68 */
    case R200_VIPH_CONTROL:       /* 0x0C40 */
    case R200_GPIOPAD_MASK:       /* 0x0198 */
    case R200_GPIOPAD_A:          /* 0x019C */
    case R200_DAC_RANGE_CNTL:     /* 0x0484 */
    case R200_GENMO_WT:           /* 0x03C2 */
    case R200_CUR2_OFFSET:        /* 0x0360 */
        break;

    case R200_CRTC_VLINE_CRNT_VLINE:
        /* The line the guest wants to be told about; the line the beam is
         * on is handed back in the upper bits when this is read. */
        s->regs.crtc_vline = val & 0x7FF;
        break;

    /* 3D engine registers (0x1C00-0x3FFF) — store in shadow array
     * for instrumentation and future 3D/Quartz Extreme support.
     * Previously RB3D_CNTL (0x1C3C) and SE_CNTL (0x1C4C) were silently
     * dropped here; now we store them along with all other 3D state. */
    case R200_RB3D_CNTL:          /* 0x1C3C */
    case R200_SE_CNTL:            /* 0x1C4C */
        /* Fall through to 3D register shadow handler below */
        {
            uint32_t idx = (addr - 0x1C00) / 4;
            if (idx < 2304) {
                s->regs.regs_3d[idx] = val;
                trace_ppc_mac_gpu_3d_reg_write(addr, val);
            }
            gpu_debug_log("3D_MMIO 0x%04x <- 0x%08x%s",
                          (uint32_t)addr, val,
                          addr == 0x1C3C ? " (RB3D_CNTL)" :
                          addr == 0x1C4C ? " (SE_CNTL)" : "");
        }
        break;

    default:
        /*
         * 3D engine register range (0x1C00-0x3FFF) — catch-all shadow.
         * Store all writes so reads return stored values, and log for
         * Phase 0 instrumentation to discover what QE actually needs.
         */
        if (addr >= 0x1C00 && addr < 0x4000) {
            uint32_t idx = ((uint32_t)addr - 0x1C00) / 4;
            ppc_mac_gpu_tcl_port_write(s, addr, val);
            if (idx < 2304) {
                s->regs.regs_3d[idx] = val;
                trace_ppc_mac_gpu_3d_reg_write(addr, val);
                /* Log render-target and key 3D registers specifically */
                if (addr == 0x1C40 || addr == 0x1C48 || addr == 0x325C ||
                    addr == 0x1C3C || addr == 0x1C4C || addr == 0x3258 ||
                    addr == 0x20B0 || addr == 0x20B4 || addr == 0x20BC) {
                    const char *name = "?";
                    switch ((uint32_t)addr) {
                    case 0x1C40: name = "RB3D_COLOROFFSET"; break;
                    case 0x1C48: name = "RB3D_COLORPITCH"; break;
                    case 0x325C: name = "RB3D_DSTCACHE_CTLSTAT"; break;
                    case 0x1C3C: name = "RB3D_CNTL"; break;
                    case 0x1C4C: name = "SE_CNTL"; break;
                    case 0x3258: name = "RB3D_DSTCACHE_MODE"; break;
                    case 0x20B0: name = "RE_WIDTH_HEIGHT"; break;
                    case 0x20B4: name = "RE_MISC"; break;
                    case 0x20BC: name = "RE_STIPPLE_ADDR"; break;
                    }
                    gpu_debug_log("3D_MMIO 0x%04x <- 0x%08x (%s)",
                                  (uint32_t)addr, val, name);
                }
            }
            break;
        }

        /*
         * VBE DISPI registers (0x500-0x514) - qemu_vga.ndrv mode switching.
         * The NDRV writes 16-bit values at (reg << 1) + 0x500.
         */
        if (addr >= PPC_MAC_GPU_VBE_OFFSET &&
            addr < PPC_MAC_GPU_VBE_END) {
            uint32_t vbe_idx = (addr - PPC_MAC_GPU_VBE_OFFSET) >> 1;
            if (vbe_idx < VBE_DISPI_INDEX_NB) {
                s->regs.vbe_regs[vbe_idx] = (uint16_t)val;
                {
                    static const char *vbe_names[] = {
                        "ID", "XRES", "YRES", "BPP", "ENABLE",
                        "BANK", "VIRT_WIDTH", "VIRT_HEIGHT",
                        "X_OFF", "Y_OFF", "VMEM_64K"
                    };
                    const char *name = vbe_idx < 11 ? vbe_names[vbe_idx] : "?";
                    gpu_debug_log("VBE_WR [%u] %s <- 0x%04x",
                                  vbe_idx, name, (uint16_t)val);
                }

                /* When ENABLE is written with the enabled flag, apply the
                 * new VBE mode to our CRTC registers so the display update
                 * picks up the correct resolution. */
                /* Y_OFFSET change → update crtc_offset for page flip */
                if (vbe_idx == VBE_DISPI_INDEX_Y_OFFSET) {
                    uint32_t w = s->regs.vbe_regs[VBE_DISPI_INDEX_XRES];
                    uint32_t bpp = s->regs.vbe_regs[VBE_DISPI_INDEX_BPP];
                    if (w == 0) w = 800;
                    if (bpp == 0) bpp = 32;
                    s->regs.crtc_offset = val * w * ((bpp + 7) / 8);
                    s->display_invalid = true;
                    gpu_debug_log("VBE Y_OFFSET=%u -> crtc_offset=0x%x",
                                  val, s->regs.crtc_offset);
                }

                if (vbe_idx == VBE_DISPI_INDEX_ENABLE &&
                    (val & VBE_DISPI_ENABLED)) {
                    uint32_t w = s->regs.vbe_regs[VBE_DISPI_INDEX_XRES];
                    uint32_t h = s->regs.vbe_regs[VBE_DISPI_INDEX_YRES];
                    uint32_t bpp = s->regs.vbe_regs[VBE_DISPI_INDEX_BPP];
                    uint32_t pix_fmt;
                    if (w == 0) w = 800;
                    if (h == 0) h = 600;
                    if (bpp == 0) bpp = 32;

                    switch (bpp) {
                    case 8:  pix_fmt = R200_CRTC_PIX_WIDTH_8BPP; break;
                    case 15: pix_fmt = R200_CRTC_PIX_WIDTH_15BPP; break;
                    case 16: pix_fmt = R200_CRTC_PIX_WIDTH_16BPP; break;
                    case 24: pix_fmt = R200_CRTC_PIX_WIDTH_24BPP; break;
                    default: pix_fmt = R200_CRTC_PIX_WIDTH_32BPP; break;
                    }

                    s->regs.crtc_gen_cntl = R200_CRTC_EN |
                                            R200_CRTC_EXT_DISP_EN | pix_fmt;
                    s->regs.crtc_h_total_disp = ((w / 8 - 1) << 16) |
                                                (w / 8 + 31);
                    s->regs.crtc_v_total_disp = ((h - 1) << 16) | (h + 27);
                    s->regs.crtc_pitch = w / 8;
                    uint32_t y_off = s->regs.vbe_regs[VBE_DISPI_INDEX_Y_OFFSET];
                    s->regs.crtc_offset = y_off * w * ((bpp + 7) / 8);
                    s->display_invalid = true;

                    gpu_debug_log("VBE mode set: %ux%ux%u", w, h, bpp);

                    /* EDID reading is done once the NDRV sets the first mode */
                    if (!s->edid_read_done) {
                        s->edid_read_done = true;
                        gpu_debug_log("EDID read phase complete - switching to ATI register mode");
                    }
                }
            }
            break;
        }

        /*
         * VGA DAC registers (0x400-0x41F) - qemu_vga.ndrv palette writes.
         */
        if (addr >= PPC_MAC_GPU_VGA_OFFSET &&
            addr < PPC_MAC_GPU_VGA_END) {
            uint16_t port = (addr - PPC_MAC_GPU_VGA_OFFSET) + 0x3C0;
            switch (port) {
            case 0x3C0:  /* AR Index/Data */
                if (s->regs.vga_ar_flip_flop == 0) {
                    s->regs.vga_ar_index = val & 0x3F;
                }
                /* Toggle flip-flop for each write */
                s->regs.vga_ar_flip_flop ^= 1;
                break;
            case 0x3C7:  /* DAC Read Address */
                s->regs.vga_dac_read_index = val;
                s->regs.vga_dac_sub_index = 0;
                break;
            case 0x3C8:  /* DAC Write Address */
                s->regs.vga_dac_write_index = val;
                s->regs.vga_dac_sub_index = 0;
                break;
            case 0x3C9: { /* DAC Data (write) */
                uint8_t idx = s->regs.vga_dac_write_index;
                s->regs.vga_dac_palette[idx][s->regs.vga_dac_sub_index] =
                    val & 0x3F;
                s->regs.vga_dac_sub_index++;
                if (s->regs.vga_dac_sub_index >= 3) {
                    s->regs.vga_dac_sub_index = 0;
                    s->regs.vga_dac_write_index++;
                }
                break;
            }
            default:
                break;
            }
            break;
        }

        /*
         * VBE extended registers (0x600-0x61F) - interrupt control.
         */
        if (addr >= PPC_MAC_GPU_EXT_OFFSET &&
            addr < PPC_MAC_GPU_EXT_END) {
            uint32_t ext_reg = (addr - PPC_MAC_GPU_EXT_OFFSET) >> 2;
            if (ext_reg == 2) {
                /* VBL interrupt acknowledge: writing bit 0 clears,
                 * writing bit 1 enables */
                if (val & 1) {
                    s->regs.gen_int_status &= ~R200_CRTC_VBLANK_INT;
                }
            }
            break;
        }

        /* R300 register catch-all: absorb writes to shadow storage.
         * Any R300-specific register the kext writes that isn't handled
         * by a specific case above gets stored in the shadow array. */
        if (addr == 0x15D4) {
            s->r300_src_swap = val & 3;         /* see below */
        }
        if (addr < 0x5000 && is_r300_register((uint32_t)addr)) {
            s->r300_shadow[(uint32_t)addr / 4] = val;
            break;
        }

        /*
         * 0x15D4: endian swap the 2D engine applies to the source of a
         * copy from system memory (write_2dblit_cmds_for_copy_buffer_using_DMA
         * in ATIRadeon9700 passes 2 for Quartz's big-endian ARGB words and
         * 0 for GL textures, which are byte arrays).
         */
        if (addr == 0x15D4) {
            s->r300_src_swap = val & 3;
            break;
        }
        /*
         * 0x15D0: the swap the 2D engine applies when it writes to system
         * memory (the same writer sets it for copies out of VRAM, see
         * ppc_mac_gpu_2d_blit_to_gart).
         */
        if (addr == 0x15D0) {
            g_r300_dst_swap = val & 3;
            break;
        }
        /* Register 0x19E4 - kext writes, accept silently */
        if (addr == 0x19E4) {
            break;
        }

        /* VAP/TCL and RBBM registers below 0x1C00 — store for readback.
         * 0x15C0 = VAP_OUTPUT_VTX_FMT_0, 0x15CC = VAP_VTX_STATE_CNTL,
         * 0x1700 = RBBM_GUICNTL, 0x1704 = SE_TCL_TEX_PROC_CTL_0 */
        if (addr == 0x15C0 || addr == 0x15CC ||
            addr == 0x1700 || addr == 0x1704) {
            break;
        }

        trace_ppc_mac_gpu_mmio_write_unimpl(size, addr,
                                             ppc_mac_gpu_reg_name(addr), val);
        break;
    }
}

/*
 * The guest's view of the registers: the ring caught up first (see "The
 * command processor on its own thread"), except for the few registers
 * that need no ordering against it.  Type-0 ring packets call the inner
 * handlers directly.
 */
static uint64_t ppc_mac_gpu_mmio_read_guest(void *opaque, hwaddr addr,
                                            unsigned int size)
{
    PPCMacGPUState *s = opaque;

    if (addr != R200_CP_RB_RPTR &&
        !(addr >= PPC_MAC_GPU_HWC_BASE && addr < PPC_MAC_GPU_HWC_END)) {
        ppc_mac_gpu_cp_sync(s, addr);
    }
    return ppc_mac_gpu_mmio_read(opaque, addr, size);
}

static void ppc_mac_gpu_mmio_write_guest(void *opaque, hwaddr addr,
                                         uint64_t val, unsigned int size)
{
    PPCMacGPUState *s = opaque;

    if (addr != R200_CP_RB_WPTR &&
        !(addr >= PPC_MAC_GPU_HWC_BASE && addr < PPC_MAC_GPU_HWC_END)) {
        ppc_mac_gpu_cp_sync(s, addr);
    }
    ppc_mac_gpu_mmio_write(opaque, addr, val, size);
}

static const MemoryRegionOps ppc_mac_gpu_mmio_ops = {
    .read = ppc_mac_gpu_mmio_read_guest,
    .write = ppc_mac_gpu_mmio_write_guest,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/* ========================================================================
 * ROM BAR - traced expansion ROM access
 * ======================================================================== */

static uint64_t ppc_mac_gpu_rom_read(void *opaque, hwaddr addr, unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint64_t val = 0;

    if (addr + size <= s->rom_size) {
        memcpy(&val, s->rom_data + addr, size);
    }
    gpu_debug_log("ROM_RD offset=0x%04"HWADDR_PRIx" size=%u val=0x%"PRIx64,
                  addr, size, val);
    return val;
}

static void ppc_mac_gpu_rom_write(void *opaque, hwaddr addr,
                                   uint64_t val, unsigned size)
{
    /* ROM is read-only, ignore writes */
}

static const MemoryRegionOps ppc_mac_gpu_rom_ops = {
    .read = ppc_mac_gpu_rom_read,
    .write = ppc_mac_gpu_rom_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/* ========================================================================
 * Byte-swapped VRAM aperture (upper half of BAR0)
 *
 * On PPC Macs, the ATI kext accesses VRAM through a byte-swapped aperture
 * so the PPC CPU (big-endian) can write pixel data that the GPU (little-endian)
 * reads correctly.  Each 32-bit write is byte-swapped:
 *   CPU writes BE 0x00RRGGBB → memory stores [BB, GG, RR, 00] (LE format)
 *
 * The byte-swapped aperture occupies the upper half of BAR0.
 * The lower half provides direct (non-swapped) access.
 * ======================================================================== */

static uint64_t ppc_mac_gpu_vram_bswap_read(void *opaque, hwaddr addr,
                                              unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint64_t val = 0;

    ppc_mac_gpu_cp_sync(s, 0xFFFFFFFF);     /* a readback: ring done first */

    if (addr + size > s->vram_size) {
        return 0;
    }

    switch (size) {
    case 1:
        /* Byte access: XOR address bits [1:0] for BE byte lane swap */
        val = vram[addr ^ 3];
        break;
    case 2: {
        /* 16-bit: swap within 32-bit word */
        hwaddr swapped = addr ^ 2;
        val = lduw_le_p(vram + swapped);
        break;
    }
    case 4:
        /* 32-bit: byte-swap the whole dword */
        val = bswap32(ldl_le_p(vram + addr));
        break;
    default:
        break;
    }
    return val;
}

static void ppc_mac_gpu_vram_bswap_write(void *opaque, hwaddr addr,
                                           uint64_t val, unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

    if (addr + size > s->vram_size) {
        return;
    }

    switch (size) {
    case 1:
        vram[addr ^ 3] = (uint8_t)val;
        break;
    case 2: {
        hwaddr swapped = addr ^ 2;
        stw_le_p(vram + swapped, (uint16_t)val);
        break;
    }
    case 4:
        stl_le_p(vram + addr, bswap32((uint32_t)val));
        break;
    default:
        break;
    }
    ppc_vram_dirty_e0(s, addr, size);
}

/*
 * R300 aperture 1.  A Radeon's frame-buffer BAR holds two apertures onto
 * the same VRAM, each with its own CPU byte swap: SURFACE_CNTL.NONSURF_APn_SWP,
 * or SURFACEn_INFO.SURF_APn_SWP inside a surface's bounds.  VRAM here holds
 * what the CPU sees through aperture 0 with its 32-bit swap (mode 2), so an
 * aperture 1 access is converted from that view.  Apple's GL driver reads
 * the depth buffer back (glReadPixels, e.g. Chess's mouse picking) through
 * aperture 1 with a surface that sets its swap.
 */
static unsigned r300_ap_swap(PPCMacGPUState *s, uint32_t off, int ap)
{
    for (int n = 0; n < 8; n++) {
        uint32_t info = s->r300_surf[3 + 4 * n];
        uint32_t lo = s->r300_surf[1 + 4 * n], hi = s->r300_surf[2 + 4 * n];
        if ((lo || hi) && off >= lo && off <= hi) {
            return (info >> (20 + 2 * ap)) & 3;
        }
    }
    return (s->regs.surface_cntl >> (20 + 2 * ap)) & 3;
}

static void r300_permute(uint8_t b[4], unsigned mode)
{
    uint8_t t;
    switch (mode & 3) {
    case 1: t = b[0]; b[0] = b[1]; b[1] = t; t = b[2]; b[2] = b[3]; b[3] = t; break;
    case 2: t = b[0]; b[0] = b[3]; b[3] = t; t = b[1]; b[1] = b[2]; b[2] = t; break;
    case 3: t = b[0]; b[0] = b[2]; b[2] = t; t = b[1]; b[1] = b[3]; b[3] = t; break;
    default: break;
    }
}

static void r300_ap1_note(PPCMacGPUState *s, hwaddr off, bool wr, unsigned m)
{
    static int n;
    if (n++ < 8) {
        qemu_log("ppc-mac-gpu r300: aperture 1 %s at 0x%" HWADDR_PRIx " (swap %u)\n",
                 wr ? "write" : "read", off, m);
    }
}

static uint64_t r300_ap1_read(void *opaque, hwaddr addr, unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    hwaddr base = addr & ~3ull;
    uint8_t b[4];

    ppc_mac_gpu_cp_sync(s, 0xFFFFFFFF);     /* a readback: ring done first */
    uint64_t v = 0;

    if (base + 4 > s->vram_size) {
        return 0;
    }
    r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
    unsigned m = r300_ap_swap(s, base, 1);
    memcpy(b, vram + base, 4);
    r300_permute(b, 2);                 /* aperture 0 view -> card's bytes */
    r300_permute(b, m);                 /* -> aperture 1 view */
    for (unsigned i = 0; i < size && (addr & 3) + i < 4; i++) {
        v = (v << 8) | b[(addr & 3) + i];
    }
    r300_ap1_note(s, addr, false, m);
    return v;
}

static void r300_ap1_write(void *opaque, hwaddr addr, uint64_t val,
                           unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    hwaddr base = addr & ~3ull;
    uint8_t b[4];

    if (base + 4 > s->vram_size) {
        return;
    }
    r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
    unsigned m = r300_ap_swap(s, base, 1);
    memcpy(b, vram + base, 4);
    r300_permute(b, 2);
    r300_permute(b, m);
    for (unsigned i = 0; i < size && (addr & 3) + i < 4; i++) {
        b[(addr & 3) + i] = val >> (8 * (size - 1 - i));
    }
    r300_permute(b, m);
    r300_permute(b, 2);
    memcpy(vram + base, b, 4);
    ppc_vram_dirty_e0(s, base, 4);
    r300_ap1_note(s, addr, true, m);
}

/* $R300_SURFWATCH: while SURFACE0 covers a range, trap aperture 0 over it
 * and log what the CPU reads and writes (e.g. glReadPixels readbacks). */
static uint64_t r300_watch_read(void *opaque, hwaddr addr, unsigned size)
{
    PPCMacGPUState *s = opaque;

    ppc_mac_gpu_cp_sync(s, 0xFFFFFFFF);     /* a readback: ring done first */
    uint8_t *p = (uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                 s->r300_watch_base + addr;
    uint64_t v = 0;

    r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
    for (unsigned i = 0; i < size; i++) {
        v = (v << 8) | p[i];
    }
    if (s->r300_watch_logged++ < 64) {
        qemu_log("r300 surfwatch: rd%u 0x%x = 0x%" PRIx64 "\n",
                 size, (unsigned)(s->r300_watch_base + addr), v);
    }
    return v;
}

static void r300_watch_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    PPCMacGPUState *s = opaque;
    uint8_t *p = (uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                 s->r300_watch_base + addr;

    for (unsigned i = 0; i < size; i++) {
        p[i] = val >> (8 * (size - 1 - i));
    }
    if (s->r300_watch_logged++ < 64) {
        qemu_log("r300 surfwatch: wr%u 0x%x = 0x%" PRIx64 "\n", size,
                 (unsigned)(s->r300_watch_base + addr), val);
    }
}

static const MemoryRegionOps r300_watch_ops = {
    .read = r300_watch_read,
    .write = r300_watch_write,
    .endianness = DEVICE_BIG_ENDIAN,
    .valid = { .min_access_size = 1, .max_access_size = 8 },
    .impl = { .min_access_size = 1, .max_access_size = 8 },
};

/*
 * Depth readback with multisampling.  The card keeps a multisampled depth
 * buffer in its micro-tiled, sample-interleaved layout, and Apple's GL
 * driver reads it that way (r300_msaa_offset) after mapping a surface over
 * it with SURFACE0 - glReadPixels of depth, which Chess uses to pick the
 * square under the mouse.  Depth is rendered here one row per sample
 * (R300DrawPacket.aa_samples), so while such a surface is mapped the buffer
 * is rewritten in the card's layout, and converted back (keeping any CPU
 * writes) when it is unmapped.  Depth drawn without multisampling is one
 * linear sample, which every sample of the card's layout gets a copy of.
 */
static void r300_zconv(PPCMacGPUState *s, bool to_card)
{
    uint8_t *vram = (uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                    s->r3_zconv_off;
    uint32_t pitch = s->r3_zconv_pitch, rows = s->r3_zconv_rows;
    uint32_t ns = s->r3_zconv_ns;
    uint64_t lin = (uint64_t)pitch * 4 * rows, card = lin * ns;
    uint8_t *tmp;

    if (!rows || s->r3_zconv_off + card > s->vram_size) {
        return;
    }
    r200_flush_at(s, R200_WHY_TAG(R200_WHY_DISPLAY));
    tmp = g_malloc(card);
    if (s->r3_zconv_rows_ns) {
        for (uint32_t y = 0; y < rows; y++) {
            for (uint32_t k = 0; k < ns; k++) {
                uint8_t *row = (to_card ? vram : tmp) +
                               ((uint64_t)y * ns + k) * pitch * 4;
                for (uint32_t x = 0; x < pitch; x++) {
                    uint8_t *c = (to_card ? tmp : vram) +
                                 r300_msaa_offset(x, y, ns, pitch, 4, k);
                    if (to_card) {
                        memcpy(c, row + x * 4, 4);
                    } else {
                        memcpy(row + x * 4, c, 4);
                    }
                }
            }
        }
        memcpy(vram, tmp, card);
    } else if (to_card) {
        for (uint32_t y = 0; y < rows; y++) {
            for (uint32_t x = 0; x < pitch; x++) {
                for (uint32_t k = 0; k < ns; k++) {
                    memcpy(tmp + r300_msaa_offset(x, y, ns, pitch, 4, k),
                           vram + ((uint64_t)y * pitch + x) * 4, 4);
                }
            }
        }
        memcpy(vram, tmp, card);
    } else {
        memcpy(tmp, vram, card);
        memset(vram + lin, 0, card - lin);
        for (uint32_t y = 0; y < rows; y++) {
            for (uint32_t x = 0; x < pitch; x++) {
                memcpy(vram + ((uint64_t)y * pitch + x) * 4,
                       tmp + r300_msaa_offset(x, y, ns, pitch, 4, 0), 4);
            }
        }
    }
    g_free(tmp);
    ppc_vram_dirty_e0(s, s->r3_zconv_off, card);
    static int logged;
    if (logged++ < 6) {
        qemu_log("ppc-mac-gpu r300: depth at 0x%x (%ux%u, %u samples) %s the "
                 "card's layout for a CPU readback\n", s->r3_zconv_off, pitch,
                 rows, ns, to_card ? "to" : "back from");
    }
}

static void r300_surface_changed(PPCMacGPUState *s)
{
    uint32_t lo = s->r300_surf[1], hi = s->r300_surf[2];
    uint32_t pitch = r300_reg(s->r3, 0x4F24) & 0x3FFC;
    uint32_t zo = r300_reg(s->r3, 0x4F20) & ~0x1Fu;
    uint32_t fb_base = (s->regs.mc_fb_location & 0xFFFF) << 16;

    if (s->r3_zconv) {
        if (lo == s->r3_zconv_off) {
            return;                             /* still mapped */
        }
        r300_zconv(s, false);
        s->r3_zconv = false;
    }
    if (!s->r3 || hi <= lo || zo < fb_base || lo != zo - fb_base ||
        s->r3_zb_offset != zo || (r300_reg(s->r3, 0x4F10) & 0xF) != 2 ||
        !pitch || !s->r3_zb_height) {
        return;
    }
    /* Samples: GB_AA_CONFIG at the depth draws, else what the size of the
     * mapped surface implies. */
    uint32_t ns = s->r3_zb_ns;
    s->r3_zconv_rows_ns = ns >= 2;
    if (ns < 2) {
        ns = ((hi - lo + 1) + pitch * 2 * s->r3_zb_height) /
             (pitch * 4 * s->r3_zb_height);
    }
    if (ns != 2 && ns != 4) {
        return;
    }
    s->r3_zconv_off = lo;
    s->r3_zconv_pitch = pitch;
    s->r3_zconv_ns = ns;
    s->r3_zconv_rows = MIN(s->r3_zb_height, (hi - lo + 1) / (pitch * 4 * ns));
    r300_zconv(s, true);
    s->r3_zconv = true;
}

static void r300_surfwatch_update(PPCMacGPUState *s)
{
    static int enabled = -1;
    uint32_t lo = s->r300_surf[1], hi = s->r300_surf[2];

    if (enabled < 0) {
        enabled = getenv("R300_SURFWATCH") != NULL;
    }
    if (!enabled || !s->r300) {
        return;
    }
    if (s->r300_watch_on) {
        memory_region_del_subregion(&s->vram_bar, &s->r300_watch);
        object_unparent(OBJECT(&s->r300_watch));
        s->r300_watch_on = false;
    }
    if (hi > lo && hi < s->vram_size) {
        memory_region_init_io(&s->r300_watch, OBJECT(s), &r300_watch_ops, s,
                              "ppc-mac-gpu-surfwatch", hi - lo + 1);
        memory_region_add_subregion_overlap(&s->vram_bar, lo, &s->r300_watch, 10);
        s->r300_watch_base = lo;
        s->r300_watch_on = true;
        qemu_log("r300 surfwatch: surface0 0x%x-0x%x info %08x\n", lo, hi,
                 s->r300_surf[3]);
    }
}

static const MemoryRegionOps r300_ap1_ops = {
    .read = r300_ap1_read,
    .write = r300_ap1_write,
    .endianness = DEVICE_BIG_ENDIAN,
    .valid = { .min_access_size = 1, .max_access_size = 4 },
    .impl = { .min_access_size = 1, .max_access_size = 4 },
};

static const MemoryRegionOps ppc_mac_gpu_vram_bswap_ops = {
    .read = ppc_mac_gpu_vram_bswap_read,
    .write = ppc_mac_gpu_vram_bswap_write,
    .endianness = DEVICE_BIG_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/* ========================================================================
 * Device lifecycle
 * ======================================================================== */

static void ppc_mac_gpu_reset(DeviceState *dev)
{
    PPCMacGPUState *s = PPC_MAC_GPU(dev);

    cp.epoch++;                 /* a CP thread mid-ring drops what it has */
    memset(&s->regs, 0, sizeof(s->regs));
    s->hwc_w = s->hwc_h = s->hwc_idx = 0;
    s->hwc_hot_x = s->hwc_hot_y = 0;
    s->regs.regs_3d[R200_3D_IDX(0x3230)] = 0xFFFFFFFFu;  /* DEPTHCLEARVALUE */
    s->hwc_visible = false;
    if (s->con) {
        dpy_mouse_set(s->con, 0, 0, false);
    }

    /* Initialize register defaults.
     * Reserve 4MB at the top of VRAM for CP ring buffer, indirect buffers,
     * and scratch writeback.  The kext places CP structures starting at
     * exactly CONFIG_MEMSIZE, so we must report less than the full
     * allocation to keep those accesses in-bounds.
     */
    uint64_t cp_headroom = 4 * MiB;
    uint64_t reported_memsize = s->vram_size - cp_headroom;
    s->regs.config_memsize = reported_memsize;
    s->regs.mc_fb_location = 0x00000000 | ((reported_memsize - 1) >> 16) << 16;
    s->regs.rbbm_status = 64;  /* FIFO entries free */
    s->regs.gui_stat = 64;
    s->regs.dac_cntl = 0x00000000;
    s->regs.dp_write_msk = 0xFFFFFFFF;
    s->regs.default_sc_bottom_right = 0x1FFF1FFF;
    s->regs.host_path_cntl = (1 << 23);

    /* Default display mode: use QEMU's -g parameter (graphic_width/height/depth)
     * which OpenBIOS reads via fw_cfg to set line-bytes and other OF properties.
     * The VBE defaults must match so that the NDRV re-applies the same mode,
     * keeping CRTC stride in sync with OpenBIOS's line-bytes. */
    {
        uint32_t boot_w = (graphic_width > 0) ? graphic_width : 800;
        uint32_t boot_h = (graphic_height > 0) ? graphic_height : 600;

        s->regs.crtc_gen_cntl = R200_CRTC_EN | R200_CRTC_EXT_DISP_EN |
                                 R200_CRTC_PIX_WIDTH_32BPP;
        s->regs.crtc_h_total_disp = ((boot_w / 8 - 1) << 16) |
                                     (boot_w / 8 + 31);
        s->regs.crtc_v_total_disp = ((boot_h - 1) << 16) | (boot_h + 27);
        s->regs.crtc_pitch = boot_w / 8;
        s->regs.crtc_offset = 0;
    }

    /* AGP status */
    s->regs.agp_status = 0x1F000203;

    /* VBE DISPI defaults - match the -g boot mode for qemu_vga.ndrv.
     * These values are read by QemuVga_Init() at NDRV startup. The NDRV
     * searches for a matching resolution in its mode list and re-applies it.
     * Must match fw_cfg dimensions so OpenBIOS line-bytes == CRTC stride. */
    {
        uint32_t boot_w = (graphic_width > 0) ? graphic_width : 800;
        uint32_t boot_h = (graphic_height > 0) ? graphic_height : 600;
        uint32_t boot_d = (graphic_depth > 0) ? graphic_depth : 32;

        s->regs.vbe_regs[VBE_DISPI_INDEX_ID] = VBE_DISPI_ID5;
        s->regs.vbe_regs[VBE_DISPI_INDEX_XRES] = boot_w;
        s->regs.vbe_regs[VBE_DISPI_INDEX_YRES] = boot_h;
        s->regs.vbe_regs[VBE_DISPI_INDEX_BPP] = boot_d;
        s->regs.vbe_regs[VBE_DISPI_INDEX_ENABLE] =
            VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED;
        s->regs.vbe_regs[VBE_DISPI_INDEX_BANK] = 0;
        s->regs.vbe_regs[VBE_DISPI_INDEX_VIRT_WIDTH] = boot_w;
        s->regs.vbe_regs[VBE_DISPI_INDEX_VIRT_HEIGHT] = boot_h;
        s->regs.vbe_regs[VBE_DISPI_INDEX_X_OFFSET] = 0;
        s->regs.vbe_regs[VBE_DISPI_INDEX_Y_OFFSET] = 0;
        s->regs.vbe_regs[VBE_DISPI_INDEX_VIDEO_MEMORY_64K] =
            (uint16_t)(s->vram_size >> 16);
    }

    s->mode = PPC_MAC_GPU_MODE_EXT;
    s->display_invalid = true;
    s->scanout_dirty_valid = false;   /* next copy is a whole frame */

    /*
     * State kept outside the register file must come back to what realize
     * leaves too, or a guest restart boots against the previous OS's card:
     * the NDRV would read ATI registers where it expects EDID, and the
     * screen would keep that OS's stride override and tiled surfaces.
     */
    s->edid_read_done = false;
    s->disp_stride_override_active = false;
    s->disp_stride_override_value = 0;
    s->host_data_active = false;
    s->pm4_fifo_idx = 0;
    s->pm4_pkt_count = 0;
    s->compositor_valid = false;
    memset(s->tiled_surfaces, 0, sizeof(s->tiled_surfaces));
    s->metal_rt_count = 0;
    s->vb_count = 0;

    timer_del(&s->vblank_timer);
}

/*
 * PPCMacGPUDirtyFn for renderers that keep copies of VRAM: the pages
 * written since the last call, from the VGA dirty log this device keeps
 * on (the CPU through the BARs, and the device's own writes, which it
 * marks with memory_region_set_dirty).
 */
static void ppc_mac_gpu_vram_dirty(void *arg, unsigned long *bitmap,
                                   uint64_t npages)
{
    PPCMacGPUState *s = arg;
    const uint64_t pg = PPC_MAC_GPU_DIRTY_PAGE, chunk = 256 * pg;
    DirtyBitmapSnapshot *snap =
        memory_region_snapshot_and_clear_dirty(&s->vram, 0, s->vram_size,
                                               DIRTY_MEMORY_VGA);

    for (uint64_t off = 0; off < s->vram_size; off += chunk) {
        uint64_t len = MIN(chunk, s->vram_size - off);
        if (!memory_region_snapshot_get_dirty(&s->vram, snap, off, len)) {
            continue;
        }
        for (uint64_t p = off; p < off + len; p += pg) {
            if (p / pg < npages &&
                memory_region_snapshot_get_dirty(&s->vram, snap, p, pg)) {
                set_bit(p / pg, bitmap);
            }
        }
    }
    g_free(snap);
}

static void ppc_mac_gpu_realize(PCIDevice *dev, Error **errp)
{
    PPCMacGPUState *s = PPC_MAC_GPU(dev);
    Object *obj = OBJECT(dev);

    s->regs_size = sizeof(PPCMacGPURegs);   /* saved with the machine */
    s->r300 = object_dynamic_cast(obj, TYPE_ATI_RADEON_9700) != NULL;
    s->rv250 = object_dynamic_cast(obj, TYPE_ATI_RADEON_9000) != NULL;
    s->r300_src_swap = 2;       /* plain copy until the driver says otherwise */
    if (s->r300) {
        const char *dump = getenv("R300_DUMP");

        s->r3 = g_new0(R300State, 1);
        s->r3_dump = dump ? fopen(dump, "w") : NULL;
        if (s->r3_dump) {
            setvbuf(s->r3_dump, NULL, _IOLBF, 0);
        }
    }

    /* Compute VRAM size from MB property */
    s->vram_size = (uint64_t)s->vram_size_mb * MiB;

    if (s->vram_size < 8 * MiB) {
        error_setg(errp, "ppc-mac-gpu: VRAM must be at least 8 MB");
        return;
    }
    if (s->vram_size > 256 * MiB) {
        error_setg(errp, "ppc-mac-gpu: VRAM must be at most 256 MB");
        return;
    }

    /* Create graphics console */
    s->con = graphic_console_init(DEVICE(dev), 0,
                                  &ppc_mac_gpu_gfx_ops, s);

    /*
     * The renderer ("renderer" property).  Unset means Metal on macOS,
     * falling back to software; naming one makes it required.
     */
    const char *rname = s->renderer_name && *s->renderer_name &&
                        strcmp(s->renderer_name, "auto") ? s->renderer_name : NULL;
    bool want_vulkan = rname && !strcmp(rname, "vulkan");
    if (rname && strcmp(rname, "metal") && !want_vulkan) {
        error_setg(errp, "%s: unknown renderer '%s' (metal or vulkan)",
                   object_get_typename(obj), rname);
        return;
    }
#ifndef CONFIG_DARWIN
    if (rname && !strcmp(rname, "metal")) {
        error_setg(errp, "%s: renderer=metal is only available on macOS",
                   object_get_typename(obj));
        return;
    }
#endif
    if (want_vulkan && !s->r300) {
        error_setg(errp, "%s: renderer=vulkan needs the R300 card (ati-radeon-9700)",
                   object_get_typename(obj));
        return;
    }
#ifndef CONFIG_PPC_MAC_GPU_VULKAN
    if (want_vulkan) {
        error_setg(errp, "%s: renderer=vulkan: this QEMU was built without "
                   "Vulkan (install the Vulkan loader/MoltenVK and rebuild)",
                   object_get_typename(obj));
        return;
    }
#endif

    /* BAR0: VRAM aperture (prefetchable)
     *
     * On macOS hosts, allocate VRAM as a Metal shared buffer (zero-copy).
     * This lets both the guest CPU (WindowServer memcpy) and the Metal GPU
     * (QE compositor draws) access the same physical memory — exactly like
     * real VRAM on an R200.  With Vulkan, VRAM is host-visible Vulkan
     * memory the GPU copies to and from.  Falls back to normal QEMU RAM
     * on other hosts.
     */
#ifdef CONFIG_PPC_MAC_GPU_VULKAN
    if (want_vulkan) {
        s->metal_vram_ptr = ppc_mac_gpu_vulkan_alloc_vram(
            s->vram_size, &s->metal_vram_opaque);
        if (!s->metal_vram_ptr) {
            error_setg(errp, "%s: renderer=vulkan unavailable: %s",
                       object_get_typename(obj), ppc_mac_gpu_vulkan_error());
            return;
        }
    }
#endif
#ifdef CONFIG_DARWIN
    if (!want_vulkan) {
        s->metal_vram_ptr = ppc_mac_gpu_metal_alloc_vram(
            s->vram_size, &s->metal_vram_opaque);
        if (!s->metal_vram_ptr && rname) {
            error_setg(errp, "%s: renderer=metal unavailable: it needs an "
                       "Apple GPU (on an Intel Mac, use renderer=vulkan)",
                       object_get_typename(obj));
            return;
        }
    }
#endif
    if (s->metal_vram_ptr) {
        /* Zero-copy path: QEMU uses the MTLBuffer's memory directly */
        memory_region_init_ram_ptr(&s->vram, obj, "ppc-mac-gpu-vram",
                                   s->vram_size, s->metal_vram_ptr);
        /*
         * This memory belongs to Metal, not to QEMU, so it is not part of
         * the machine's memory unless it is named here.  Without this,
         * saving the machine leaves video memory behind and it comes back
         * to a black screen -- everything Mac OS X had drawn lived here.
         */
        vmstate_register_ram(&s->vram, DEVICE(obj));
    } else {
        /* Fallback: normal QEMU-managed RAM */
        memory_region_init_ram(&s->vram, obj, "ppc-mac-gpu-vram",
                               s->vram_size, &error_fatal);
    }
    if (s->r300) {
        /* Two apertures onto VRAM, as on a real Radeon (see r300_ap1_ops). */
        memory_region_init(&s->vram_bar, obj, "ppc-mac-gpu-vram-bar",
                           2 * s->vram_size);
        memory_region_add_subregion(&s->vram_bar, 0, &s->vram);
        memory_region_init_io(&s->vram_ap1, obj, &r300_ap1_ops, s,
                              "ppc-mac-gpu-vram-ap1", s->vram_size);
        memory_region_add_subregion(&s->vram_bar, s->vram_size, &s->vram_ap1);
        pci_register_bar(dev, PPC_MAC_GPU_VRAM_BAR,
                         PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram_bar);
    } else {
        pci_register_bar(dev, PPC_MAC_GPU_VRAM_BAR,
                         PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram);
    }

    /* BAR2: MMIO register space */
    memory_region_init_io(&s->mmio, obj, &ppc_mac_gpu_mmio_ops, s,
                          "ppc-mac-gpu-mmio", PPC_MAC_GPU_MMIO_SIZE);
    pci_register_bar(dev, PPC_MAC_GPU_MMIO_BAR,
                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    ppc_mac_gpu_cp_start(s);

    /* BAR1: I/O space alias to first 256 bytes of MMIO */
    memory_region_init_alias(&s->io, obj, "ppc-mac-gpu-io",
                             &s->mmio, 0, PPC_MAC_GPU_IO_SIZE);
    pci_register_bar(dev, PPC_MAC_GPU_IO_BAR,
                     PCI_BASE_ADDRESS_SPACE_IO, &s->io);

    /* Enable VRAM dirty tracking for display updates */
    memory_region_set_log(&s->vram, true, DIRTY_MEMORY_VGA);

    /* Initialize VBlank timer */
    timer_init_ns(&s->vblank_timer, QEMU_CLOCK_VIRTUAL,
                  ppc_mac_gpu_vblank, s);
    timer_init_ms(&s->frame_kick_timer, QEMU_CLOCK_REALTIME,
                  ppc_mac_gpu_frame_kick_cb, s);

    /*
     * ROM handling:
     * - romfile: Contains the Mac NDRV (Joy! PEF format) for the PCI ROM BAR.
     *   OpenBIOS reads this and sets the "driver,AAPL,MacOS,PowerPC" property.
     * - biosrom: Contains the full ATI BIOS ROM, copied to VRAM offset 0
     *   (where POST normally shadows it) for kext access.
     *
     * If only romfile is provided (no biosrom), it serves both roles.
     */
    {
        const char *bios_path_str = s->biosrom;
        if (bios_path_str && bios_path_str[0]) {
            /* Load full BIOS ROM into VRAM */
            gchar *rom_data = NULL;
            gsize rom_len = 0;
            char *resolved = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_path_str);
            const char *path = resolved ? resolved : bios_path_str;

            if (g_file_get_contents(path, &rom_data, &rom_len, NULL)) {
                uint8_t *vram_ptr = memory_region_get_ram_ptr(&s->vram);
                size_t copy_size = MIN(rom_len, s->vram_size);
                memcpy(vram_ptr, rom_data, copy_size);
                gpu_debug_log("BIOS ROM copied to VRAM: %zu bytes from %s",
                              (size_t)rom_len, path);
                g_free(rom_data);
            } else {
                gpu_debug_log("BIOS ROM load failed: %s", path);
            }
            g_free(resolved);
        } else if (dev->romfile && dev->romfile[0]) {
            /* Fallback: copy romfile to VRAM if no separate biosrom */
            gchar *rom_data = NULL;
            gsize rom_len = 0;
            char *rom_path = qemu_find_file(QEMU_FILE_TYPE_BIOS, dev->romfile);
            const char *path = rom_path ? rom_path : dev->romfile;

            if (g_file_get_contents(path, &rom_data, &rom_len, NULL)) {
                uint8_t *vram_ptr = memory_region_get_ram_ptr(&s->vram);
                size_t copy_size = MIN(rom_len, s->vram_size);
                memcpy(vram_ptr, rom_data, copy_size);
                gpu_debug_log("ROM copied to VRAM: %zu bytes from %s",
                              (size_t)rom_len, path);
                g_free(rom_data);
            } else {
                gpu_debug_log("ROM load failed: %s", path);
            }
            g_free(rom_path);
        }
        /* Leave dev->romfile intact so QEMU's PCI framework creates
         * the standard RAM-backed ROM BAR via pci_add_option_rom(). */
    }

    /* Initialize I2C/DDC for EDID - provides monitor detection to the kext */
    {
        I2CBus *i2cbus = i2c_init_bus(DEVICE(dev), "ppc-mac-gpu.ddc");
        bitbang_i2c_init(&s->bbi2c, i2cbus);
        I2CSlave *i2cddc = I2C_SLAVE(qdev_new(TYPE_I2CDDC));
        i2c_slave_set_address(i2cddc, 0x50);
        qdev_realize_and_unref(DEVICE(i2cddc), BUS(i2cbus), &error_abort);
    }

    /* Build EDID blob for NDRV byte-read compatibility.
     * The QEMU VGA NDRV reads EDID by doing byte reads at MMIO offsets 0x00-0x7F.
     * On a real Bochs VGA, EDID is mapped there. On our ATI device, those offsets
     * are ATI registers. We intercept byte-sized reads to return EDID data.
     *
     * EDID 1.3 - 128 bytes with standard timing bitmap for common modes. */
    {
        uint8_t *e = s->edid_blob;
        int i;
        uint8_t sum;

        memset(e, 0, 128);

        /* Header */
        e[0] = 0x00; e[1] = 0xFF; e[2] = 0xFF; e[3] = 0xFF;
        e[4] = 0xFF; e[5] = 0xFF; e[6] = 0xFF; e[7] = 0x00;

        /* Manufacturer: "QEM" (EISA ID) */
        e[8] = 0x45; e[9] = 0x4D; /* Q=0x11=10001, E=0x05=00101, M=0x0D=01101 → 10001 00101 01101 = 0x454D */

        /* Product code */
        e[10] = 0x00; e[11] = 0x92; /* 0x9200 - Radeon 9200 */

        /* Serial */
        e[12] = 0x01; e[13] = 0x00; e[14] = 0x00; e[15] = 0x00;

        /* Week 1, Year 2006 */
        e[16] = 1; e[17] = 16; /* year = 2006 - 1990 = 16 */

        /* EDID version 1.3 */
        e[18] = 1; e[19] = 3;

        /* Basic display: digital input, 8-bit color, DFP 1.0 */
        e[20] = 0x80; /* digital input */
        e[21] = 33;   /* max H size cm */
        e[22] = 25;   /* max V size cm */
        e[23] = 120;  /* gamma (2.2 = (2.2-1)*100 = 120) */
        e[24] = 0x0A; /* RGB color, preferred timing in DTD1 */

        /* Chromaticity - default sRGB values */
        e[25] = 0xA5; e[26] = 0x59;
        e[27] = 0xA6; e[28] = 0x54;
        e[29] = 0x13; e[30] = 0xAB;
        e[31] = 0x26; e[32] = 0x0F;
        e[33] = 0x50; e[34] = 0x54;

        /* Established timings (bytes 35-37) - the NDRV parses these as a bitmap.
         * Bit mapping (MSB first):
         * Byte 35: 720x400@70, 720x400@88, 640x480@60, 640x480@67,
         *          640x480@72, 640x480@75, 800x600@56, 800x600@60
         * Byte 36: 800x600@72, 800x600@75, 832x624@75, 1024x768i@87,
         *          1024x768@60, 1024x768@70, 1024x768@75, 1280x1024@75
         * Byte 37: 1152x870@75, (rest reserved)
         */
        e[35] = 0x21; /* 640x480@60, 800x600@60 */
        e[36] = 0x09; /* 1024x768@60 + 1280x1024@75 */
        e[37] = 0x80; /* 1152x870@75 */

        /* Standard timing descriptors (bytes 38-53, 8 entries of 2 bytes).
         * Format: byte0 = (hpixels/8)-31, byte1 = aspect_ratio(6:7) | refresh-60(5:0)
         * 0x0101 = unused entry. */
        /* 1600x1200@60: hpix=1600 → (1600/8)-31=169=0xA9, aspect=4:3(0x00), refresh=0 */
        e[38] = 0xA9; e[39] = 0x40;
        /* 1280x960@60: hpix=1280 → (1280/8)-31=129=0x81, aspect=4:3(0x00), refresh=0 */
        e[40] = 0x81; e[41] = 0x40;
        /* 1920x1200@60: hpix=1920 → (1920/8)-31=209=0xD1, aspect=16:10(0x40), refresh=0 */
        /* Actually 16:10 aspect is bits 7:6 = 00 for 16:10 in EDID 1.3 */
        /* In EDID 1.3: 00=16:10, 01=4:3, 10=5:4, 11=16:9 */
        e[42] = 0xD1; e[43] = 0x00; /* 1920x1200@60 */
        /* Rest unused */
        e[44] = 0x01; e[45] = 0x01;
        e[46] = 0x01; e[47] = 0x01;
        e[48] = 0x01; e[49] = 0x01;
        e[50] = 0x01; e[51] = 0x01;
        e[52] = 0x01; e[53] = 0x01;

        /* Detailed Timing Descriptor #1: 1024x768@60Hz (preferred) */
        /* Pixel clock: 65.00 MHz = 6500 in 10kHz units */
        e[54] = 0x64; e[55] = 0x19; /* pixel clock low/high (6500 = 0x1964) */
        e[56] = 0x00; /* H active low 8 bits (1024 & 0xFF = 0x00) */
        e[57] = 0x18; /* H blanking low 8 bits (280 & 0xFF = 0x18) */
        e[58] = 0x41; /* H active high:H blanking high (4:1) → (1024>>8)<<4 | (280>>8) = 0x41 */
        e[59] = 0x00; /* V active low 8 bits (768 & 0xFF = 0x00) */
        e[60] = 0x23; /* V blanking low 8 bits (35 & 0xFF = 0x23) */
        e[61] = 0x30; /* V active high:V blanking high → (768>>8)<<4 | (35>>8) = 0x30 */
        e[62] = 0x18; /* H sync offset low (24) */
        e[63] = 0x88; /* H sync width low (136) */
        e[64] = 0x36; /* V sync offset(3):V sync width(6) */
        e[65] = 0x00; /* HS offset hi:HS width hi:VS offset hi:VS width hi */
        e[66] = 0x30; /* H image size low (304mm, low 8 bits) */
        e[67] = 0xE4; /* V image size low (228mm, low 8 bits) */
        e[68] = 0x10; /* H/V image size upper */
        e[69] = 0x00; /* H border */
        e[70] = 0x00; /* V border */
        e[71] = 0x18; /* flags: non-interlaced, normal, digital separate */

        /* Descriptor #2: Monitor name */
        e[72] = 0x00; e[73] = 0x00; e[74] = 0x00;
        e[75] = 0xFC; /* Monitor name tag */
        e[76] = 0x00;
        /* "QEMU RV280\n   " */
        e[77] = 'Q'; e[78] = 'E'; e[79] = 'M'; e[80] = 'U';
        e[81] = ' '; e[82] = 'R'; e[83] = 'V'; e[84] = '2';
        e[85] = '8'; e[86] = '0'; e[87] = '\n'; e[88] = ' ';
        e[89] = ' ';

        /* Descriptor #3: Monitor range limits */
        e[90] = 0x00; e[91] = 0x00; e[92] = 0x00;
        e[93] = 0xFD; /* Range limits tag */
        e[94] = 0x00;
        e[95] = 50;   /* min V freq */
        e[96] = 75;   /* max V freq */
        e[97] = 30;   /* min H freq (kHz) */
        e[98] = 81;   /* max H freq (kHz) */
        e[99] = 140;  /* max pixel clock / 10 MHz */
        e[100] = 0x00; /* no GTF */
        /* Pad with 0x0A, 0x20 */
        e[101] = 0x0A; e[102] = 0x20; e[103] = 0x20;
        e[104] = 0x20; e[105] = 0x20; e[106] = 0x20;
        e[107] = 0x20;

        /*
         * Descriptor #4: Established Timings III (tag 0xF7).  The QEMU VGA
         * NDRV maps bit (byte k, bit j) to entry 8k+j of its extended mode
         * table: 19 = 1440x900 and 27 = 1680x1050 (16:10).  Entries 34-37
         * are 1856x1392/1792x1344 in the stock driver; the patched driver
         * (ndrv-hwcursor/build.py) turns them into modes at the Mac's panel
         * aspect ratio - 1440x932, 1280x828, 1152x746, 1680x1088 - and
         * 38-39/44-45 (duplicate 1600x1200, 1920x1440) into the area below
         * the notch (1710x1074 points, 1.592) - 1440x904, 1280x804,
         * 1152x724, 1680x1056 - so they are only offered with
         * host-aspect-modes=on.
         */
        e[108] = 0x00; e[109] = 0x00; e[110] = 0x00;
        e[111] = 0xF7;
        e[112] = 0x00;
        e[113] = 0x0A;                     /* revision */
        for (i = 114; i < 126; i++) e[i] = 0x00;
        e[116] = 0x08;                     /* 1440x900 */
        e[117] = 0x08;                     /* 1680x1050 */
        if (s->host_aspect_modes) {
            e[118] = 0xFC;                 /* entries 34-39 */
            e[119] = 0x30;                 /* entries 44-45 */
        }

        /* Extension count */
        e[126] = 0; /* no extensions */

        /* Checksum: sum of all 128 bytes must equal 0 mod 256 */
        sum = 0;
        for (i = 0; i < 127; i++) sum += e[i];
        e[127] = (uint8_t)(256 - sum);

        qemu_log("ppc-mac-gpu: EDID blob initialized (checksum 0x%02x)\n",
                 e[127]);
    }

    /* Initialize renderer — try Metal first for hardware-accelerated QE compositing,
     * fall back to software renderer if Metal is unavailable. */
    {
        uint8_t *vram_ptr = memory_region_get_ram_ptr(&s->vram);
#ifdef CONFIG_PPC_MAC_GPU_VULKAN
        if (want_vulkan) {
            s->renderer = ppc_mac_gpu_renderer_vulkan();
            s->renderer_opaque = s->renderer->init(vram_ptr, s->vram_size);
            if (!s->renderer_opaque) {
                error_setg(errp, "%s: renderer=vulkan unavailable: %s",
                           object_get_typename(obj), ppc_mac_gpu_vulkan_error());
                return;
            }
            qemu_log("ppc-mac-gpu: using Vulkan renderer for 3D\n");
        }
#endif
#ifdef CONFIG_DARWIN
        /* Try Metal renderer first for hardware-accelerated 3D */
        if (!want_vulkan && (s->renderer = ppc_mac_gpu_renderer_metal())) {
            s->renderer_opaque = s->renderer->init(vram_ptr, s->vram_size);
            if (s->renderer_opaque) {
                qemu_log("ppc-mac-gpu: using Metal renderer for 3D\n");
            } else if (rname) {
                error_setg(errp, "%s: renderer=metal: Metal init failed",
                           object_get_typename(obj));
                return;
            } else {
                qemu_log("ppc-mac-gpu: Metal init failed, falling back to SW\n");
                s->renderer = NULL;
            }
        }
        if (!s->renderer)
#endif
        {
            s->renderer = ppc_mac_gpu_renderer_sw();
            s->renderer_opaque = s->renderer->init(vram_ptr, s->vram_size);
            qemu_log("ppc-mac-gpu: using software renderer (3D textured quad)\n");
        }
    }

    if (s->renderer && s->renderer->set_dirty_source) {
        s->renderer->set_dirty_source(s->renderer_opaque, ppc_mac_gpu_vram_dirty, s);
    }

    /*
     * Scanout-path switches (see ppc_mac_gpu_display_update).  The VRAM
     * dirty log belongs to the renderer when one consumes it (Vulkan);
     * that combination keeps copying whole frames, deferring nothing.
     */
    ppc_gpu_dirty_scanout = ppc_gpu_env_on("PPCGPU_DIRTY_SCANOUT");
    ppc_gpu_flip_kick = ppc_gpu_env_on("PPCGPU_FLIP_KICK");
    ppc_gpu_frame_kick = ppc_gpu_env_on("PPCGPU_FRAME_KICK");
    {
        const char *ap = getenv("PPCGPU_ASYNC_PRESENT");
        ppc_gpu_async_present = ap && strcmp(ap, "0") != 0;
    }
    s->scanout_log_owner = !s->renderer || !s->renderer->set_dirty_source;

    /* Set PCI config space fields */
    pci_set_byte(&dev->config[PCI_REVISION_ID],
                 s->r300 ? 0x00 : PPC_MAC_GPU_PCI_REVISION);

    /* Interrupt pin A - required for kext IRQ handler registration */
    dev->config[PCI_INTERRUPT_PIN] = 1;  /* INTA# */

    /* Subsystem IDs - match the device */
    pci_set_word(dev->config + PCI_SUBSYSTEM_VENDOR_ID,
                 PPC_MAC_GPU_PCI_VENDOR_ID);
    pci_set_word(dev->config + PCI_SUBSYSTEM_ID,
                 s->r300 ? PPC_MAC_GPU_R300_DEVICE_ID
                         : PPC_MAC_GPU_PCI_DEVICE_ID);

    /*
     * AGP capability at offset 0x58 (where real RV280 has it).
     * PCI_AGP_SIZEOF = 12 bytes: cap_id, next, version, rfu,
     * status (4 bytes), command (4 bytes).
     *
     * This is critical: Tiger's ATI kext matches on IOAGPDevice,
     * and IOPCIFamily creates IOAGPDevice nubs for devices that
     * have the AGP PCI capability (cap_id = 0x02).
     */
    if (pci_add_capability(dev, PCI_CAP_ID_AGP, 0x58,
                           PCI_AGP_SIZEOF, errp) < 0) {
        return;
    }
    /* AGP version 2.0 */
    pci_set_byte(dev->config + 0x58 + PCI_AGP_VERSION, 0x20);
    /* AGP Status: SBA, 4x/2x/1x capable, 32 RQ depth */
    pci_set_long(dev->config + 0x58 + PCI_AGP_STATUS,
                 PCI_AGP_STATUS_SBA |
                 PCI_AGP_STATUS_RATE4 |
                 PCI_AGP_STATUS_RATE2 |
                 PCI_AGP_STATUS_RATE1 |
                 (0x1F << 24));  /* max requests - 1 = 31 */
    /* AGP Command: enable AGP + SBA + 4x rate.
     * The kext checks AGP status to decide whether to enable 3D/GL.
     * Without AGP enabled, it only provides 2D acceleration. */
    pci_set_long(dev->config + 0x58 + PCI_AGP_COMMAND,
                 PCI_AGP_COMMAND_AGP |   /* AGP enabled */
                 PCI_AGP_COMMAND_SBA |   /* Sideband addressing */
                 PCI_AGP_COMMAND_RATE4 | /* 4x rate */
                 (0x1F << 24));          /* max requests */
    /* AppleMacRiscAGP::setAGPEnable() writes the command register and spins
     * until the enable bit reads back; it must be writable like real HW. */
    pci_set_long(dev->wmask + 0x58 + PCI_AGP_COMMAND, 0xffffffff);

    /* Power Management capability at offset 0x50 (real RV280 layout) */
    if (pci_add_capability(dev, PCI_CAP_ID_PM, 0x50, 8, errp) < 0) {
        return;
    }
    /* PM version 1.1, D0/D3 supported */
    pci_set_word(dev->config + 0x50 + PCI_PM_PMC,
                 (0x01 << 0) |     /* version 1.1 */
                 (1 << 9)   |      /* D1 supported */
                 (1 << 10));       /* D2 supported */

    trace_ppc_mac_gpu_realize(s->vram_size_mb, s->vram_size);
}

/* PCI config space read/write hooks for debugging */
static uint32_t ppc_mac_gpu_config_read(PCIDevice *dev,
                                         uint32_t addr, int len)
{
    uint32_t val = pci_default_read_config(dev, addr, len);
    gpu_debug_log("CFG_RD off=0x%02x len=%d val=0x%x", addr, len, val);
    return val;
}

static void ppc_mac_gpu_config_write(PCIDevice *dev,
                                      uint32_t addr, uint32_t val, int len)
{
    gpu_debug_log("CFG_WR off=0x%02x len=%d val=0x%x", addr, len, val);
    pci_default_write_config(dev, addr, val, len);
}

static void ppc_mac_gpu_exit(PCIDevice *dev)
{
    PPCMacGPUState *s = PPC_MAC_GPU(dev);

    ppc_mac_gpu_cp_stop();
    timer_del(&s->vblank_timer);
    timer_del(&s->frame_kick_timer);
    g_free(s->shadow_buf);
    s->shadow_buf = NULL;
    g_free(s->draw_verts);
    s->draw_verts = NULL;
    s->draw_verts_cap = 0;
    g_free(s->draw_idx);
    s->draw_idx = NULL;
    s->draw_idx_cap = 0;
    graphic_console_close(s->con);
}

/* ========================================================================
 * QOM registration
 * ======================================================================== */

static const Property ppc_mac_gpu_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", PPCMacGPUState, vram_size_mb, 128),
    DEFINE_PROP_BOOL("host-aspect-modes", PPCMacGPUState, host_aspect_modes, false),
    DEFINE_PROP_BOOL("gl-zero-copy", PPCMacGPUState, gl_zero_copy, true),
    DEFINE_PROP_STRING("biosrom", PPCMacGPUState, biosrom),
    DEFINE_PROP_STRING("renderer", PPCMacGPUState, renderer_name),
};

/* Read-only "perf": the running totals behind PowerEmu's overlay. */
/*
 * Turning a trace on while the machine is running.
 *
 * Every trace in this file asks getenv() at the point it prints, which
 * meant choosing them before the machine started and restarting to change
 * one's mind -- painful when the thing being chased takes ten minutes of
 * booting to reach.  Setting this property sets the environment variable,
 * so all of them can be switched from the monitor mid-flight:
 *
 *   qom-set /machine/peripheral/gpu0 trace POWEREMU_TEX_TRACE
 *   qom-set /machine/peripheral/gpu0 trace -POWEREMU_TEX_TRACE
 *
 * Only names beginning POWEREMU_ are accepted: this is a debugging handle,
 * not a way to set the emulator's environment at large.
 */
static void ppc_mac_gpu_set_trace(Object *obj, const char *value, Error **errp)
{
    const char *name = value;
    bool off = false;

    if (name && name[0] == '-') {
        off = true;
        name++;
    }
    if (!name || (!g_str_has_prefix(name, "POWEREMU_") &&
                  !g_str_has_prefix(name, "PPCGPU_"))) {
        error_setg(errp, "trace names must begin with POWEREMU_ or PPCGPU_");
        return;
    }
    if (off) {
        g_unsetenv(name);
    } else {
        g_setenv(name, "1", true);
    }
    /* Logs that remember whether they were switched on have to be told. */
    g_seq_log_enabled = -1;
    fprintf(stderr, "ppc-mac-gpu: %s %s\n", name, off ? "off" : "on");
}

static char *ppc_mac_gpu_get_trace(Object *obj, Error **errp)
{
    /* What is on, of the traces this device knows about. */
    static const char *const names[] = {
        "POWEREMU_STALL_TRACE", "POWEREMU_FENCE_TRACE", "POWEREMU_TEX_TRACE",
        "POWEREMU_VP_TRACE", "POWEREMU_POLL_TRACE",
        /* The direct R200 path logs per draw only under these. */
        "PPCGPU_SEQ_LOG", "PPCGPU_DIAG", "PPCGPU_WINDOWS",
    };
    GString *out = g_string_new(NULL);
    int i;

    for (i = 0; i < ARRAY_SIZE(names); i++) {
        if (g_getenv(names[i])) {
            g_string_append_printf(out, "%s%s", out->len ? " " : "", names[i]);
        }
    }
    if (!out->len) {
        g_string_append(out, "none");
    }
    return g_string_free(out, false);
}

static char *ppc_mac_gpu_get_perf(Object *obj, Error **errp)
{
    PPCMacGPUState *s = PPC_MAC_GPU(obj);
    return g_strdup_printf("frames=%" PRIu64 " draws=%" PRIu64 " tex_vram=%" PRIu64
                           " tex_agp=%" PRIu64 " agp_bytes=%" PRIu64
                           " vram_high=%" PRIu64 " vram_usable=%u",
                           r200_perf.frames, r200_perf.draws, r200_perf.tex_vram,
                           r200_perf.tex_agp, r200_perf.agp_bytes, r200_perf.vram_high,
                           s->regs.config_memsize);
}

/* ========================================================================
 * Saving the card with the machine
 *
 * PowerEmu's sleep writes a whole machine into its disk and starts it
 * again later.  The card took no part in that: the guest came back
 * believing its windows were still in video memory, on a card that had
 * been reset underneath it, and the screen stayed black although Mac OS X
 * was running perfectly well behind it.
 *
 * The register block is saved whole rather than field by field.  It is
 * plain data with several hundred members, and a hand-written list of
 * them would be wrong within a week.  Its size travels with it, so a
 * machine saved by one build refuses to load into a build whose registers
 * have moved, instead of restoring nonsense.  Nothing outside it needs
 * saving: the display mode, the scanout address and the rest are worked
 * out again from the registers on the next screen update, and the command
 * buffers are refilled by the driver.
 * ======================================================================== */

static int ppc_mac_gpu_pre_save(void *opaque)
{
    ppc_mac_gpu_cp_sync(opaque, 0xFFFFFFFF);  /* save with the ring done */
    return 0;
}

static int ppc_mac_gpu_post_load(void *opaque, int version_id)
{
    PPCMacGPUState *s = opaque;

    /* Ask for one full screen update; everything derived follows from it. */
    s->display_invalid = true;
    s->scanout_dirty_valid = false;
    /* Force the mode -- and the surface handed to the window -- to be
     * built again from the restored registers. */
    s->disp.width = 0;
    s->disp.height = 0;
    s->surface_width = 0;
    s->surface_height = 0;
    s->surface_stride = 0;
    ppc_vram_dirty_e0(s, 0, s->vram_size);
    /*
     * The pointer has to be handed to the window again.  Its picture and
     * position are restored above, but they reached the window in the first
     * place as one-off announcements when the guest uploaded them, and a
     * woken machine's guest has no reason to upload anything -- it carries
     * on from the instant it was stopped.  Left undone, the desktop comes
     * back correctly with an invisible pointer.  Deferred to the next
     * screen update rather than done here, because at this moment the
     * window may not yet be listening.
     */
    s->hwc_announce = true;
    if (s->con) {
        dpy_gfx_update_full(s->con);
    }
    return 0;
}

static const VMStateDescription vmstate_ppc_mac_gpu = {
    .name = "ppc-mac-gpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .pre_save = ppc_mac_gpu_pre_save,
    .post_load = ppc_mac_gpu_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(pci, PPCMacGPUState),
        VMSTATE_UINT32_EQUAL(regs_size, PPCMacGPUState,
                             "this machine was saved by a build whose "
                             "graphics registers were laid out differently"),
        VMSTATE_BUFFER_UNSAFE(regs, PPCMacGPUState, 1, sizeof(PPCMacGPURegs)),
        VMSTATE_BOOL(display_invalid, PPCMacGPUState),
        /*
         * What the card has learned rather than been told: Mac OS X's
         * driver sets the scanline length by drawing at it, not by writing
         * CRTC_PITCH, so this is not in the registers.  Left behind, a
         * woken machine reads its own screen with the wrong line length
         * and shows a sheared picture of the desktop.
         */
        VMSTATE_BOOL(disp_stride_override_active, PPCMacGPUState),
        VMSTATE_UINT32(disp_stride_override_value, PPCMacGPUState),
        /*
         * Where Quartz Extreme composites, which the display path follows
         * instead of the framebuffer.  Learned from the blits that go
         * there, so a woken machine would scan out of the wrong place
         * until the next one.
         */
        VMSTATE_UINT32(compositor_base, PPCMacGPUState),
        VMSTATE_UINT32(compositor_pitch, PPCMacGPUState),
        VMSTATE_BOOL(compositor_valid, PPCMacGPUState),
        /*
         * Which stretches of video memory are macro-tiled.  This is
         * gathered from PITCH_OFFSET writes as the guest runs, and a blit
         * into a surface registered before the machine slept would
         * otherwise be written as though it were linear -- the same fault
         * as the window-drag bug, but only after waking.
         */
        VMSTATE_BUFFER_UNSAFE(tiled_surfaces, PPCMacGPUState, 1,
                              sizeof(((PPCMacGPUState *)0)->tiled_surfaces)),
        /*
         * The hardware cursor's picture.  The guest sends it once and only
         * sends it again when it changes, so without this a woken machine
         * can sit with no pointer at all.
         */
        VMSTATE_BUFFER_UNSAFE(hwc_pix, PPCMacGPUState, 1,
                              sizeof(((PPCMacGPUState *)0)->hwc_pix)),
        VMSTATE_UINT32(hwc_w, PPCMacGPUState),
        VMSTATE_UINT32(hwc_h, PPCMacGPUState),
        VMSTATE_UINT32(hwc_idx, PPCMacGPUState),
        VMSTATE_INT32(hwc_x, PPCMacGPUState),
        VMSTATE_INT32(hwc_y, PPCMacGPUState),
        VMSTATE_BOOL(hwc_visible, PPCMacGPUState),
        /* Registers the 9700 kext writes and reads back later. */
        VMSTATE_BUFFER_UNSAFE(r300_shadow, PPCMacGPUState, 1,
                              sizeof(((PPCMacGPUState *)0)->r300_shadow)),
        VMSTATE_END_OF_LIST()
    }
};

static void ppc_mac_gpu_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->vendor_id = PPC_MAC_GPU_PCI_VENDOR_ID;
    k->device_id = PPC_MAC_GPU_PCI_DEVICE_ID;
    k->class_id  = PPC_MAC_GPU_PCI_CLASS;
    k->realize      = ppc_mac_gpu_realize;
    k->exit         = ppc_mac_gpu_exit;
    k->config_read  = ppc_mac_gpu_config_read;
    k->config_write = ppc_mac_gpu_config_write;

    device_class_set_legacy_reset(dc, ppc_mac_gpu_reset);
    dc->vmsd = &vmstate_ppc_mac_gpu;
    device_class_set_props(dc, ppc_mac_gpu_properties);
    dc->hotpluggable = false;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    object_class_property_add_str(klass, "perf", ppc_mac_gpu_get_perf, NULL);
    object_class_property_add_str(klass, "trace", ppc_mac_gpu_get_trace,
                                  ppc_mac_gpu_set_trace);
}

/*
 * ATI Radeon 9700 PRO (R300).  Same register model as the RV280 device; the
 * PCI identity is what makes Tiger's ATIRadeon9700.kext (IOPCIMatch
 * 0x4E441002) and its GL driver take the card.
 */
static void ati_radeon_9700_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);

    k->device_id = PPC_MAC_GPU_R300_DEVICE_ID;
    k->revision = 0x00;
    dc->desc = "ATI Radeon 9700";
}

static const TypeInfo ati_radeon_9700_type_info = {
    .name          = TYPE_ATI_RADEON_9700,
    .parent        = TYPE_PPC_MAC_GPU,
    .class_init    = ati_radeon_9700_class_init,
};

/*
 * ATI Radeon 9000 PRO (RV250) for Mac OS X 10.2 Jaguar: same R200 register
 * model as the RV280 device, with the PCI ID Jaguar's ATIRadeon8500.kext
 * (and its GL driver) match.
 */
static void ati_radeon_9000_class_init(ObjectClass *klass, void *data)
{
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);

    k->device_id = PPC_MAC_GPU_RV250_DEVICE_ID;
    dc->desc = "ATI Radeon 9000";
}

static const TypeInfo ati_radeon_9000_type_info = {
    .name          = TYPE_ATI_RADEON_9000,
    .parent        = TYPE_PPC_MAC_GPU,
    .class_init    = ati_radeon_9000_class_init,
};

static const TypeInfo ppc_mac_gpu_type_info = {
    .name          = TYPE_PPC_MAC_GPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PPCMacGPUState),
    .class_init    = ppc_mac_gpu_class_init,
    .interfaces    = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void ppc_mac_gpu_register_types(void)
{
    type_register_static(&ppc_mac_gpu_type_info);
    type_register_static(&ati_radeon_9700_type_info);
    type_register_static(&ati_radeon_9000_type_info);
}

type_init(ppc_mac_gpu_register_types)
