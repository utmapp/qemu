/*
 * PPC Mac GPU — Surface Ownership & Pipeline Tracker
 *
 * Defines the surface classes, ownership rules, and frame/pass tracking
 * structures for the Tiger QE compositor pipeline.  This is NOT an
 * abstract render-graph engine — it is a minimal, concrete tracking layer
 * that makes the compositor debuggable by answering:
 *
 *   1. What surfaces exist?
 *   2. Who writes them?
 *   3. Who reads them?
 *   4. When are they valid?
 *   5. How do they reach the visible framebuffer?
 *
 * Copyright (c) 2024
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef PPC_MAC_GPU_SURFACE_H
#define PPC_MAC_GPU_SURFACE_H

#include <stdint.h>
#include <stdbool.h>

/* ====================================================================
 * Phase 1 — Surface Classes
 *
 * Every important pixel buffer in the pipeline belongs to exactly one
 * of these classes.  The class determines who may write it, who may
 * read it, and when it is authoritative.
 * ==================================================================== */

typedef enum {
    /*
     * SURF_TEXTURE_TILE
     *   Storage:     VRAM (e.g. 0x3dc000..0x8c0000), 256x256 or smaller tiles
     *   Byte layout: 32bpp BE ARGB, linear (no tiling in practice)
     *   Writer:      Guest CPU via HOST_DATA uploads, or 2D BLT SRC copies
     *   Reader:      3D draw path — sampled as textures by Metal
     *   Authoritative when:  After HOST_DATA or 2D BLT completes
     *   Invalidated when:    Overwritten by a new HOST_DATA/BLT, or
     *                        clobbered by Metal writeback to overlapping range
     *   Note: These are the compositor's input tiles — wallpaper, window
     *         chrome, icons, text glyphs, etc.  Each tile is uploaded once
     *         per content change.  QE reads them via texture sampling.
     */
    SURF_TEXTURE_TILE = 0,

    /*
     * SURF_COMPOSITOR_RT
     *   Storage:     VRAM (e.g. 0x300000) + Metal RT + Shadow RT
     *   Byte layout: VRAM=BE ARGB, Metal=LE BGRA, Shadow=BE ARGB
     *   Writer:      Metal 3D draws (compositing passes)
     *   Reader:      2D BLT engine (PRESENT_BLIT to framebuffer), and
     *                subsequent 3D draws within the same pass (via LoadAction)
     *   Authoritative when:
     *     - Metal RT: during a 3D draw pass (between bind and present)
     *     - Shadow RT: after Metal writeback, IF pitch conflicts exist
     *     - VRAM: after Metal writeback, IF no pitch conflicts
     *   Invalidated when: New compositor frame begins (pitch change at
     *                     same offset signals frame boundary)
     *   Note: 0x300000 is the PRIMARY compositor RT.  The guest uses
     *         multiple pitches (832, 896, 1024) at the same VRAM offset
     *         for different compositor passes.  On real hardware, MC
     *         macro-tiling separates these; in our linear VRAM, they
     *         alias destructively.  Shadow RTs work around this.
     */
    SURF_COMPOSITOR_RT,

    /*
     * SURF_WINDOW_RT
     *   Storage:     VRAM (e.g. 0x54a000, 0x60a000) + Shadow RT
     *   Byte layout: Same as COMPOSITOR_RT
     *   Writer:      Metal 3D draws (per-window compositing)
     *   Reader:      2D BLT engine (PRESENT_BLIT to framebuffer)
     *   Authoritative when:  Same rules as COMPOSITOR_RT
     *   Invalidated when:    Window content changes, or new frame
     *   Note: Each visible window gets its own RT at a separate VRAM
     *         offset.  Finder window = 0x54a000.  Like the main
     *         compositor RT, the guest may use multiple pitches.
     */
    SURF_WINDOW_RT,

    /*
     * SURF_INTERMEDIATE
     *   Storage:     VRAM (e.g. 0x1ee000, 0x462000)
     *   Byte layout: 32bpp BE ARGB, linear
     *   Writer:      Metal 3D draws (intermediate render passes) or
     *                2D BLT copies from other surfaces
     *   Reader:      2D BLT engine (PRESENT_BLIT to framebuffer) or
     *                3D draw as texture source
     *   Authoritative when:  After the pass that produces it completes
     *   Invalidated when:    Used and overwritten by next frame
     *   Note: 0x1ee000 is the early/boot compositor RT (800x600).
     *         0x462000 appears as a non-SRT PRESENT_BLIT source for
     *         the desktop background below the menu bar.
     */
    SURF_INTERMEDIATE,

    /*
     * SURF_VISIBLE_FB
     *   Storage:     VRAM at CRTC offset (0x0)
     *   Byte layout: 32bpp BE xRGB, linear, stride = CRTC_PITCH
     *   Writer:      2D BLT engine (PRESENT_BLITs from compositor/window RTs),
     *                Metal blit_2d (SRT path), fallback 2D pixel copies
     *   Reader:      CRTC scanout → Cocoa display
     *   Authoritative when:  Always (this IS what's displayed)
     *   Invalidated when:    Never (continuously updated by BLTs)
     *   Note: This is the ONLY surface the user actually sees.  Every
     *         other surface exists to eventually produce pixels here.
     *         Multiple writers may touch different regions per frame:
     *         - PRESENT_BLIT from 0x300000 pitch=1024 → wallpaper (full screen)
     *         - PRESENT_BLIT from 0x300000 pitch=896 → Finder window area
     *         - PRESENT_BLIT from 0x462000 → desktop below menu bar
     *         - PRESENT_BLIT from 0x54a000 → Finder window content
     *         Later BLTs overwrite earlier ones at the same pixel positions.
     */
    SURF_VISIBLE_FB,

    /*
     * SURF_2D_ONLY
     *   Storage:     VRAM
     *   Byte layout: 32bpp BE ARGB, may be tiled per MC config
     *   Writer:      2D BLT engine (BITBLT_MULTI, MMIO BLT), HOST_DATA
     *   Reader:      2D BLT engine
     *   Authoritative when:  After BLT/HOST_DATA completes
     *   Invalidated when:    Overwritten by another BLT
     *   Note: Surfaces that never interact with the 3D pipeline.
     *         Cursor bitmaps, scroll-copies, pure 2D widget updates
     *         when QE falls back to 2D for specific operations.
     */
    SURF_2D_ONLY,

    SURF_CLASS_COUNT,
} SurfaceClass;

/* Human-readable names for logging */
static inline const char *surface_class_name(SurfaceClass c)
{
    switch (c) {
    case SURF_TEXTURE_TILE:  return "tex_tile";
    case SURF_COMPOSITOR_RT: return "comp_rt";
    case SURF_WINDOW_RT:     return "win_rt";
    case SURF_INTERMEDIATE:  return "intermediate";
    case SURF_VISIBLE_FB:    return "visible_fb";
    case SURF_2D_ONLY:       return "2d_only";
    default:                 return "unknown";
    }
}

/* ====================================================================
 * Phase 2 — Frame/Pass/RT Tracker
 *
 * Lightweight per-frame tracking of compositor activity.
 * Records events as they happen; dumped at frame end.
 * ==================================================================== */

typedef enum {
    PASS_EVENT_BIND_RT,       /* 3D RT bound for drawing */
    PASS_EVENT_SAMPLE_TEX,    /* Texture sampled from VRAM */
    PASS_EVENT_RENDER,        /* 3D draw executed */
    PASS_EVENT_WRITEBACK,     /* Metal output written back to VRAM/SRT */
    PASS_EVENT_PRESENT,       /* 2D BLT from RT → visible framebuffer */
    PASS_EVENT_FALLBACK_2D,   /* 2D BLT not involving any 3D-produced surface */
    PASS_EVENT_HOST_UPLOAD,   /* HOST_DATA texture upload */
    PASS_EVENT_BLIT_UPLOAD,   /* 2D BLT texture/surface upload */
    PASS_EVENT_DRAG_COPY,     /* synthetic drag body copy in framebuffer */
} PassEventType;

static inline const char *pass_event_name(PassEventType e)
{
    switch (e) {
    case PASS_EVENT_BIND_RT:      return "bind_rt";
    case PASS_EVENT_SAMPLE_TEX:   return "sample_tex";
    case PASS_EVENT_RENDER:       return "render";
    case PASS_EVENT_WRITEBACK:    return "writeback";
    case PASS_EVENT_PRESENT:      return "present";
    case PASS_EVENT_FALLBACK_2D:  return "fallback_2d";
    case PASS_EVENT_HOST_UPLOAD:  return "host_upload";
    case PASS_EVENT_BLIT_UPLOAD:  return "blit_upload";
    case PASS_EVENT_DRAG_COPY:    return "drag_copy";
    default:                      return "?";
    }
}

/* Single event in the frame log */
typedef struct PassEvent {
    PassEventType type;
    uint32_t pass_id;         /* monotonic within frame */
    uint32_t surface_offset;  /* VRAM offset of primary surface */
    uint32_t pitch;           /* pitch in pixels */
    uint32_t src_offset;      /* for PRESENT/SAMPLE: source surface */
    uint32_t dst_offset;      /* for PRESENT: destination (usually 0) */
    uint16_t width, height;   /* region size */
    uint16_t src_x, src_y;    /* source position if applicable */
    uint16_t dst_x, dst_y;    /* destination position (for PRESENT) */
    int16_t  move_x, move_y;  /* drag delta for synthetic moves */
    SurfaceClass surface_class;
    bool     from_srt;        /* true if SRT was authoritative source */
} PassEvent;

typedef enum {
    SURF_OWNER_NONE        = 0,
    SURF_OWNER_QE_PRESENT  = (1u << 0),
    SURF_OWNER_CPU_PRESENT = (1u << 1),
    SURF_OWNER_DRAG_COPY   = (1u << 2),
    SURF_OWNER_HOST_UPLOAD = (1u << 3),
    SURF_OWNER_BLIT_UPLOAD = (1u << 4),
    SURF_OWNER_RENDER      = (1u << 5),
} SurfaceOwnerMask;

typedef struct SurfaceFrameState {
    bool valid;
    uint32_t offset;
    uint32_t pitch;
    SurfaceClass surface_class;
    uint32_t owner_mask;
    uint16_t qe_presents;
    uint16_t cpu_presents;
    uint16_t drag_copies;
    uint16_t host_uploads;
    uint16_t blit_uploads;
    uint16_t renders;
} SurfaceFrameState;

/* Per-frame tracker */
#define FRAME_EVENT_MAX 512
#define FRAME_SURFACE_STATE_MAX 32
typedef struct FrameTracker {
    uint32_t frame_id;        /* monotonic frame counter */
    uint32_t pass_id;         /* monotonic pass counter within frame */
    uint32_t event_count;
    PassEvent events[FRAME_EVENT_MAX];

    /* Surface presence flags — set during frame, read at dump */
    bool has_compositor_rt;   /* any 3D draw to SURF_COMPOSITOR_RT? */
    bool has_window_rt;       /* any 3D draw to SURF_WINDOW_RT? */
    bool has_present;         /* any PRESENT_BLIT to visible FB? */
    bool has_fallback_2d;     /* any fallback 2D write to visible FB? */
    bool has_host_upload;     /* any HOST_DATA upload? */
    bool has_blit_upload;     /* any 2D copy/upload to offscreen surface? */
    bool has_drag_copy;       /* any synthetic body move this frame? */
    bool drag_observed;       /* drag motion detected in this/forced frames */
    bool has_mixed_surface_owner; /* any surface fed by multiple authorities? */

    uint32_t surface_state_count;
    SurfaceFrameState surface_states[FRAME_SURFACE_STATE_MAX];

    /* Dump control */
    bool dumped;              /* true after frame summary logged */
    uint32_t dump_limit;      /* max frames to auto-dump (then on-demand) */
    uint32_t force_dump_remaining; /* targeted dump budget for drag frames */
} FrameTracker;

/* ====================================================================
 * Tracker API
 * ==================================================================== */

static inline void frame_tracker_init(FrameTracker *ft)
{
    memset(ft, 0, sizeof(*ft));
    ft->dump_limit = 5;  /* auto-dump first 5 frames */
}

static inline void frame_tracker_new_frame(FrameTracker *ft)
{
    /* Dump previous frame if it had events */
    ft->frame_id++;
    ft->pass_id = 0;
    ft->event_count = 0;
    ft->has_compositor_rt = false;
    ft->has_window_rt = false;
    ft->has_present = false;
    ft->has_fallback_2d = false;
    ft->has_host_upload = false;
    ft->has_blit_upload = false;
    ft->has_drag_copy = false;
    ft->drag_observed = false;
    ft->has_mixed_surface_owner = false;
    ft->surface_state_count = 0;
    ft->dumped = false;
}

static inline void frame_tracker_request_dump(FrameTracker *ft,
                                              uint32_t frames)
{
    if (frames > ft->force_dump_remaining) {
        ft->force_dump_remaining = frames;
    }
}

static inline void frame_tracker_note_drag(FrameTracker *ft,
                                           uint32_t dump_frames)
{
    ft->drag_observed = true;
    frame_tracker_request_dump(ft, dump_frames);
}

static inline SurfaceFrameState *frame_tracker_surface_state(FrameTracker *ft,
                                                             uint32_t offset,
                                                             uint32_t pitch,
                                                             SurfaceClass sc)
{
    for (uint32_t i = 0; i < ft->surface_state_count; i++) {
        SurfaceFrameState *st = &ft->surface_states[i];
        if (st->valid && st->offset == offset) {
            if (pitch) {
                st->pitch = pitch;
            }
            st->surface_class = sc;
            return st;
        }
    }

    if (ft->surface_state_count >= FRAME_SURFACE_STATE_MAX) {
        return NULL;
    }

    SurfaceFrameState *st = &ft->surface_states[ft->surface_state_count++];
    memset(st, 0, sizeof(*st));
    st->valid = true;
    st->offset = offset;
    st->pitch = pitch;
    st->surface_class = sc;
    return st;
}

static inline bool frame_tracker_surface_has_owner(FrameTracker *ft,
                                                   uint32_t offset,
                                                   uint32_t owner_mask)
{
    for (uint32_t i = 0; i < ft->surface_state_count; i++) {
        SurfaceFrameState *st = &ft->surface_states[i];
        if (st->valid && st->offset == offset &&
            (st->owner_mask & owner_mask) != 0) {
            return true;
        }
    }
    return false;
}

static inline void frame_tracker_mark_surface_owner(FrameTracker *ft,
                                                    uint32_t offset,
                                                    uint32_t pitch,
                                                    SurfaceClass sc,
                                                    PassEventType type,
                                                    uint32_t dst_offset,
                                                    bool from_srt)
{
    SurfaceFrameState *st = frame_tracker_surface_state(ft, offset, pitch, sc);
    if (!st) {
        return;
    }

    switch (type) {
    case PASS_EVENT_PRESENT:
        if (from_srt) {
            st->owner_mask |= SURF_OWNER_QE_PRESENT;
            st->qe_presents++;
        } else if (dst_offset == 0) {
            st->owner_mask |= SURF_OWNER_CPU_PRESENT;
            st->cpu_presents++;
        }
        break;
    case PASS_EVENT_FALLBACK_2D:
        if (dst_offset == 0) {
            st->owner_mask |= SURF_OWNER_CPU_PRESENT;
            st->cpu_presents++;
        }
        break;
    case PASS_EVENT_DRAG_COPY:
        st->owner_mask |= SURF_OWNER_DRAG_COPY;
        st->drag_copies++;
        break;
    case PASS_EVENT_HOST_UPLOAD:
        st->owner_mask |= SURF_OWNER_HOST_UPLOAD;
        st->host_uploads++;
        break;
    case PASS_EVENT_BLIT_UPLOAD:
        st->owner_mask |= SURF_OWNER_BLIT_UPLOAD;
        st->blit_uploads++;
        break;
    case PASS_EVENT_BIND_RT:
    case PASS_EVENT_RENDER:
    case PASS_EVENT_WRITEBACK:
        st->owner_mask |= SURF_OWNER_RENDER;
        st->renders++;
        break;
    default:
        break;
    }

    if ((st->owner_mask & SURF_OWNER_QE_PRESENT) &&
        ((st->owner_mask & SURF_OWNER_CPU_PRESENT) ||
         (st->owner_mask & SURF_OWNER_DRAG_COPY))) {
        ft->has_mixed_surface_owner = true;
    }
}

static inline void frame_tracker_owner_string(uint32_t mask,
                                              char *buf,
                                              size_t buf_size)
{
    bool first = true;
    int n = 0;

    if (!buf_size) {
        return;
    }
    buf[0] = '\0';

    struct {
        uint32_t bit;
        const char *name;
    } entries[] = {
        { SURF_OWNER_QE_PRESENT,  "qe_present"  },
        { SURF_OWNER_CPU_PRESENT, "cpu_present" },
        { SURF_OWNER_DRAG_COPY,   "drag_copy"   },
        { SURF_OWNER_HOST_UPLOAD, "host_upload" },
        { SURF_OWNER_BLIT_UPLOAD, "blit_upload" },
        { SURF_OWNER_RENDER,      "render"      },
    };

    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        if (!(mask & entries[i].bit)) {
            continue;
        }
        n += snprintf(buf + n, buf_size > (size_t)n ? buf_size - (size_t)n : 0,
                      "%s%s", first ? "" : "|", entries[i].name);
        first = false;
    }

    if (first) {
        snprintf(buf, buf_size, "none");
    }
}

/* Classify a VRAM offset into a surface class based on known ranges */
static inline SurfaceClass classify_surface(uint32_t offset, uint32_t pitch,
                                             uint32_t dst_offset)
{
    /* Visible framebuffer is always at offset 0 */
    if (offset == 0 || dst_offset == 0) {
        return SURF_VISIBLE_FB;
    }
    /* Primary compositor RT — multiple pitches at 0x300000 */
    if (offset == 0x300000) {
        return SURF_COMPOSITOR_RT;
    }
    /* Known window RTs */
    if (offset == 0x54a000 || offset == 0x60a000 || offset == 0x992000) {
        return SURF_WINDOW_RT;
    }
    /* Known intermediate surfaces */
    if (offset == 0x1ee000 || offset == 0x462000) {
        return SURF_INTERMEDIATE;
    }
    /* Small offsets near compositor RT are likely sub-surfaces */
    if (offset >= 0x30e000 && offset < 0x370000) {
        return SURF_INTERMEDIATE;
    }
    /* Everything else — classify by context (caller can override) */
    return SURF_2D_ONLY;
}

static inline void frame_tracker_record_detail(FrameTracker *ft,
                                               PassEventType type,
                                               uint32_t surface_offset,
                                               uint32_t pitch,
                                               uint32_t src_offset,
                                               uint32_t dst_offset,
                                               uint16_t w, uint16_t h,
                                               uint16_t src_x, uint16_t src_y,
                                               uint16_t dst_x, uint16_t dst_y,
                                               int16_t move_x, int16_t move_y,
                                               bool from_srt)
{
    if (ft->event_count >= FRAME_EVENT_MAX) return;

    SurfaceClass sc = classify_surface(surface_offset, pitch, dst_offset);

    PassEvent *e = &ft->events[ft->event_count++];
    e->type = type;
    e->pass_id = ft->pass_id;
    e->surface_offset = surface_offset;
    e->pitch = pitch;
    e->src_offset = src_offset;
    e->dst_offset = dst_offset;
    e->width = w;
    e->height = h;
    e->src_x = src_x;
    e->src_y = src_y;
    e->dst_x = dst_x;
    e->dst_y = dst_y;
    e->move_x = move_x;
    e->move_y = move_y;
    e->surface_class = sc;
    e->from_srt = from_srt;

    /* Update presence flags */
    switch (type) {
    case PASS_EVENT_BIND_RT:
    case PASS_EVENT_RENDER:
        if (sc == SURF_COMPOSITOR_RT) ft->has_compositor_rt = true;
        if (sc == SURF_WINDOW_RT)     ft->has_window_rt = true;
        break;
    case PASS_EVENT_PRESENT:
        ft->has_present = true;
        break;
    case PASS_EVENT_FALLBACK_2D:
        ft->has_fallback_2d = true;
        break;
    case PASS_EVENT_HOST_UPLOAD:
        ft->has_host_upload = true;
        break;
    case PASS_EVENT_BLIT_UPLOAD:
        ft->has_blit_upload = true;
        break;
    case PASS_EVENT_DRAG_COPY:
        ft->has_drag_copy = true;
        ft->drag_observed = true;
        break;
    default:
        break;
    }

    frame_tracker_mark_surface_owner(ft, surface_offset, pitch, sc, type,
                                     dst_offset, from_srt);
}

static inline void frame_tracker_record(FrameTracker *ft,
                                        PassEventType type,
                                        uint32_t surface_offset,
                                        uint32_t pitch,
                                        uint32_t src_offset,
                                        uint32_t dst_offset,
                                        uint16_t w, uint16_t h,
                                        uint16_t dst_x, uint16_t dst_y,
                                        bool from_srt)
{
    frame_tracker_record_detail(ft, type,
                                surface_offset, pitch,
                                src_offset, dst_offset,
                                w, h,
                                0, 0,
                                dst_x, dst_y,
                                0, 0,
                                from_srt);
}

/* Increment pass counter (call when RT offset or pitch changes) */
static inline void frame_tracker_next_pass(FrameTracker *ft)
{
    ft->pass_id++;
}

/* ====================================================================
 * Dump frame tracker summary
 * ==================================================================== */
static inline void frame_tracker_dump(FrameTracker *ft)
{
    bool forced = (ft->force_dump_remaining > 0);

    if (ft->dumped || ft->event_count == 0) return;
    if (ft->frame_id > ft->dump_limit && !forced) return;
    ft->dumped = true;
    if (forced) {
        ft->force_dump_remaining--;
    }

    fprintf(stderr, "\n[FRAME_SUMMARY] frame=%u events=%u comp_rt=%d "
            "win_rt=%d present=%d fallback=%d host_upload=%d "
            "blit_upload=%d drag_copy=%d drag=%d mixed_owner=%d "
            "forced=%d remaining=%u\n",
            ft->frame_id, ft->event_count,
            ft->has_compositor_rt, ft->has_window_rt,
            ft->has_present, ft->has_fallback_2d, ft->has_host_upload,
            ft->has_blit_upload, ft->has_drag_copy, ft->drag_observed,
            ft->has_mixed_surface_owner,
            forced, ft->force_dump_remaining);

    uint32_t last_pass = UINT32_MAX;
    for (uint32_t i = 0; i < ft->event_count; i++) {
        PassEvent *e = &ft->events[i];
        if (e->pass_id != last_pass) {
            last_pass = e->pass_id;
            fprintf(stderr, "[FRAME_PASS] frame=%u pass=%u\n",
                    ft->frame_id, e->pass_id);
        }
        fprintf(stderr, "  [FRAME_PASS] frame=%u pass=%u event=%s "
                "surface_id=0x%x pitch=%u class=%s "
                "src_off=0x%x dst_off=0x%x %ux%u "
                "src=(%u,%u) dst=(%u,%u) move=(%d,%d) srt=%d\n",
                ft->frame_id, e->pass_id,
                pass_event_name(e->type),
                e->surface_offset, e->pitch,
                surface_class_name(e->surface_class),
                e->src_offset, e->dst_offset,
                e->width, e->height,
                e->src_x, e->src_y,
                e->dst_x, e->dst_y,
                e->move_x, e->move_y,
                e->from_srt);
    }
    for (uint32_t i = 0; i < ft->surface_state_count; i++) {
        SurfaceFrameState *st = &ft->surface_states[i];
        char owner_buf[128];

        if (!st->valid || st->owner_mask == SURF_OWNER_NONE) {
            continue;
        }

        frame_tracker_owner_string(st->owner_mask, owner_buf,
                                   sizeof(owner_buf));
        fprintf(stderr,
                "[FRAME_SURFACE] frame=%u surface=0x%x pitch=%u class=%s "
                "owners=%s qe=%u cpu=%u drag=%u host=%u blit=%u render=%u\n",
                ft->frame_id, st->offset, st->pitch,
                surface_class_name(st->surface_class),
                owner_buf, st->qe_presents, st->cpu_presents,
                st->drag_copies, st->host_uploads, st->blit_uploads,
                st->renders);
    }
    fprintf(stderr, "[FRAME_END] frame=%u\n\n", ft->frame_id);
}

/* Global tracker pointer — set by Metal backend, accessed by device code */
extern FrameTracker *g_frame_tracker;

/* ========================================================================
 * BLIT region table
 *
 * Records the (off-screen render target) -> (screen) rectangle mappings
 * observed on SRCCOPY BITBLT_MULTI packets, so the 3D renderer can map
 * render-target coordinates onto screen coordinates.
 * ======================================================================== */

#define BLIT_REGION_RT_MAX          16
#define BLIT_REGION_MAX_PER_RT      64

typedef struct BlitRegion {
    uint32_t src_x, src_y;      /* rect origin within the render target */
    uint32_t dst_x, dst_y;      /* rect origin on screen */
    uint32_t width, height;
    uint32_t sequence;          /* for LRU eviction */
} BlitRegion;

typedef struct BlitRegionRT {
    uint32_t rt_offset;         /* VRAM byte offset of the render target */
    uint32_t rt_pitch;          /* render target pitch in bytes */
    int region_count;
    BlitRegion regions[BLIT_REGION_MAX_PER_RT];
} BlitRegionRT;

typedef struct BlitRegionTable {
    int rt_count;
    uint32_t sequence;          /* monotonic counter for LRU */
    BlitRegionRT rts[BLIT_REGION_RT_MAX];
} BlitRegionTable;

#endif /* PPC_MAC_GPU_SURFACE_H */
