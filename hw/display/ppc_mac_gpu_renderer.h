/*
 * PPC Mac GPU - Renderer Interface
 *
 * Clean boundary between guest device emulation and host rendering backend.
 * All guest device code is pure C. Backend implementations may use
 * platform-specific APIs (Metal via Objective-C, etc.).
 *
 * Backends:
 *   - Software (ppc_mac_gpu_renderer_sw.c): always available, pure C
 *   - Metal (ppc_mac_gpu_metal.m): macOS only, the default there
 *   - Vulkan (ppc_mac_gpu_vulkan.c): R300 only; MoltenVK on macOS, the
 *     Vulkan loader elsewhere
 *
 * This interface is designed to compile cleanly on both Intel and Apple
 * Silicon macOS hosts. No platform-specific types leak into this header.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_PPC_MAC_GPU_RENDERER_H
#define HW_DISPLAY_PPC_MAC_GPU_RENDERER_H

#include <stdint.h>
#include <stdbool.h>

struct R300DrawPacket;     /* hw/display/r300/r300_draw.h */

/*
 * Which VRAM pages the CPU side (the guest CPU through the BARs, or the
 * device) has written since the last call: sets bit p of bitmap (npages
 * bits, pages of PPC_MAC_GPU_DIRTY_PAGE bytes) for each, and forgets them.
 */
#define PPC_MAC_GPU_DIRTY_PAGE 4096
typedef void (*PPCMacGPUDirtyFn)(void *arg, unsigned long *bitmap,
                                 uint64_t npages);

/*
 * Blit operation descriptor.
 * Used for 2D engine operations (fill, copy, host data upload).
 * All coordinates are in pixels; offsets are byte offsets into VRAM.
 */
typedef struct PPCMacGPUBlit {
    /* Source */
    uint32_t src_x, src_y;
    uint32_t src_offset;        /* VRAM byte offset */
    uint32_t src_pitch;         /* bytes per scanline */

    /* Destination */
    uint32_t dst_x, dst_y;
    uint32_t dst_offset;        /* VRAM byte offset */
    uint32_t dst_pitch;         /* bytes per scanline */

    /* Dimensions */
    uint32_t width, height;     /* pixels */

    /* Operation */
    uint32_t rop3;              /* raster operation code */
    uint32_t fg_color;          /* foreground color */
    uint32_t bg_color;          /* background color */
    uint32_t bpp;               /* bits per pixel */
} PPCMacGPUBlit;

/*
 * Scanout descriptor.
 * Describes how to read the framebuffer from VRAM and present it.
 */
typedef struct PPCMacGPUScanout {
    uint32_t width;
    uint32_t height;
    uint32_t bpp;
    uint32_t stride;            /* bytes per scanline */
    uint64_t offset;            /* byte offset into VRAM */
    bool big_endian;            /* true if VRAM is in big-endian byte order */
} PPCMacGPUScanout;

/*
 * 3D draw command descriptor.
 * Passed from PM4 Type 3 3D opcode handlers to the renderer backend.
 */
typedef struct PPCMacGPU3DDrawCmd {
    uint32_t opcode;            /* R200_3D_DRAW_VBUF, _IMMD, _INDX */
    uint32_t prim_type;         /* R200_PRIM_TRIANGLES, _TRI_STRIP, etc. */
    uint32_t num_vertices;      /* number of vertices/indices */

    /* For DRAW_IMMD: inline vertex data */
    const uint32_t *vertex_data;  /* pointer to inline dwords (may be NULL) */
    uint32_t vertex_data_dwords;  /* number of dwords */

    /* For DRAW_VBUF/DRAW_INDX: vertex buffer info */
    uint32_t vb_addr;           /* GPU address of vertex data */
    uint32_t vb_stride;         /* bytes per vertex */

    /* For DRAW_INDX: index data */
    const uint16_t *index_data; /* pointer to indices (may be NULL) */
    uint32_t num_indices;
} PPCMacGPU3DDrawCmd;

/*
 * 3D state snapshot — all registers the renderer needs for a draw call.
 * Populated from the regs_3d[] shadow array before calling draw_3d().
 */
typedef struct PPCMacGPU3DState {
    /* Vertex format */
    uint32_t se_vtx_fmt_0;      /* 0x2088 */
    uint32_t se_vtx_fmt_1;      /* 0x208C */
    uint32_t se_vte_cntl;       /* 0x20B0 */

    /* Pixel pipeline */
    uint32_t pp_cntl;           /* 0x1C38 — which tex units enabled */
    uint32_t pp_misc;           /* 0x1C14 */

    /* Texture unit 0 — R200 register bank */
    uint32_t pp_txfilter_0;     /* 0x2C00 */
    uint32_t pp_txformat_0;     /* 0x2C04 */
    uint32_t pp_txformat_x_0;   /* 0x2C08 */
    uint32_t pp_txsize_0;       /* 0x2C0C */
    uint32_t pp_txpitch_0;      /* 0x2C10 */
    uint32_t pp_txoffset_0;     /* 0x2C14 */

    /* Texture unit 0 — R100 register bank (Apple kext uses these) */
    uint32_t r100_pp_txfilter_0;  /* 0x1C54 */
    uint32_t r100_pp_txformat_0;  /* 0x1C58 */
    uint32_t r100_pp_txoffset_0;  /* 0x1C5C */
    uint32_t r100_pp_tex_size_0;  /* 0x1D04 */
    uint32_t r100_pp_tex_pitch_0; /* 0x1D08 */

    /* Texture combiners (unit 0) */
    uint32_t pp_txcblend_0;     /* 0x2F00 */
    uint32_t pp_txcblend2_0;    /* 0x2F04 */
    uint32_t pp_txablend_0;     /* 0x2F08 */
    uint32_t pp_txablend2_0;    /* 0x2F0C */

    /* Render backend */
    uint32_t rb3d_cntl;         /* 0x1C3C */
    uint32_t rb3d_coloroffset;  /* 0x1C40 */
    uint32_t rb3d_colorpitch;   /* 0x1C48 */
    uint32_t rb3d_blendcntl;    /* 0x3220 */
    uint32_t rb3d_ablendcntl;   /* 0x3224 - separate alpha blend */
    uint32_t rb3d_cblendcntl;   /* 0x3228 - color blend override */
    uint32_t rb3d_zstencilcntl; /* 0x1C2C */

    /* Setup engine */
    uint32_t se_cntl;           /* 0x1C4C */
    uint32_t re_cntl;           /* 0x1C50 */

    /* Viewport transform (as float bits) */
    uint32_t se_vport_xscale;   /* 0x1D98 */
    uint32_t se_vport_xoffset;  /* 0x1D9C */
    uint32_t se_vport_yscale;   /* 0x1DA0 */
    uint32_t se_vport_yoffset;  /* 0x1DA4 */
    uint32_t se_vport_zscale;   /* 0x1DA8 */
    uint32_t se_vport_zoffset;  /* 0x1DAC */

    /* 3D scissor */
    uint32_t re_top_left;       /* 0x26C0 */
    uint32_t re_width_height;   /* 0x26C4 */

    /* Screen dimensions (for coordinate mapping) */
    uint32_t screen_width;
    uint32_t screen_height;
} PPCMacGPU3DState;

/*
 * Renderer backend interface.
 *
 * Each backend implements this struct. The guest device code calls these
 * functions without knowing which backend is active.
 *
 * All functions receive an opaque pointer that the backend can use for
 * its internal state. The vram_ptr and vram_size are provided for backends
 * that need direct VRAM access (the software renderer always does).
 */

/*
 * Direct R200 draw — one fully decoded 3D draw, handed to the backend.
 *
 * The device side does everything that needs guest memory or register
 * semantics (vertex fetch through AGP/GART, TCL transform, viewport,
 * primitive assembly into a triangle list, texture-unit register decode);
 * the backend only rasterises into VRAM.  VRAM is the single source of
 * truth: the backend renders straight into it at the guest's offset and
 * pitch, so 2D blits and scanout see the result with no copies.
 *
 * Pixel convention (see qemu-ppc-gpu-endianness-and-tiling-are-correct):
 * 32bpp VRAM holds big-endian ARGB, i.e. bytes A,R,G,B.
 */
#define R200_MAX_TEX     6
#define R200_MAX_STAGES  8

typedef struct R200Vertex {
    float pos[4];                 /* window x,y (pixels), z, clip w */
    float color[4];               /* diffuse RGBA, 0..1 */
    float spec[4];                /* specular RGBA, 0..1 */
    float tex[R200_MAX_TEX][4];   /* s,t,r,q per texture unit */
} R200Vertex;

typedef struct R200TexUnit {
    uint32_t enabled;
    uint32_t offset;              /* byte offset in VRAM */
    uint32_t pitch;               /* bytes per row */
    uint32_t width, height;
    uint32_t format;              /* R200_TXFORMAT_* (bits 4:0) */
    uint32_t alpha_in_map;
    uint32_t filter;              /* raw PP_TXFILTER */
    uint32_t denorm;              /* s,t are in texels, not 0..1 */
    uint32_t swap;                /* TXOFFSET endian swap: 0 none, 1 16-bit,
                                   * 2 32-bit, 3 half-dword */
    const uint8_t *host_data;     /* non-NULL: texels copied out of AGP/system
                                   * memory (same byte layout as VRAM) */
} R200TexUnit;

typedef struct R200DrawPacket {
    /* Render target */
    uint32_t rt_offset;           /* byte offset in VRAM */
    uint32_t rt_pitch;            /* pixels per row */
    uint32_t rt_format;           /* RB3D_CNTL colour format field */
    uint32_t rt_width, rt_height;
    uint32_t scissor[4];          /* x0, y0, x1, y1 (exclusive) */
    uint32_t excl[3][4];          /* exclusive aux scissors: x0, y0, x1, y1 */
    uint32_t num_excl;

    /* Pixel pipeline */
    uint32_t pp_cntl, pp_misc, rb3d_cntl;
    /* Depth / stencil (RB3D_CNTL Z_ENABLE / STENCIL_ENABLE) */
    uint32_t depth_enable, stencil_enable;
    uint32_t zstencil;            /* RB3D_ZSTENCILCNTL */
    uint32_t stencil_refmask;     /* RB3D_STENCILREFMASK */
    uint32_t depth_offset;        /* byte offset in VRAM */
    uint32_t depth_pitch;         /* pixels */
    uint32_t depth_bpp;           /* 2 (16-bit Z) or 4 (24/32-bit Z) */
    uint32_t cblend, ablend, blend_color;
    uint32_t plane_mask;          /* colour write mask, ARGB (all ones = off) */
    uint32_t fog_color;           /* PP_FOG_COLOR (colour + factor source) */
    uint32_t prim_class;          /* 0 triangles, 1 lines, 2 points */
    uint32_t txcblend[R200_MAX_STAGES], txcblend2[R200_MAX_STAGES];
    uint32_t txablend[R200_MAX_STAGES], txablend2[R200_MAX_STAGES];
    uint32_t tfactor[8];
    R200TexUnit tex[R200_MAX_TEX];

    /* Geometry: an indexed triangle list in window coordinates */
    const R200Vertex *verts;
    uint32_t num_verts;
    const uint32_t *indices;
    uint32_t num_indices;
} R200DrawPacket;

typedef struct PPCMacGPURenderer {
    const char *name;           /* e.g. "software", "metal", "vulkan" */

    /*
     * draw_r200/draw_r300 only enqueue the draw and encode it on a
     * backend thread, so they are safe to call without the BQL (the
     * Metal render queue).  The device skips its big-lock dance for
     * such a backend; flush_r200/submit_r200 still wait for everything.
     * Deterministic rejections (bad format, alignment) are still
     * returned synchronously.
     */
    bool draw_queue_async;

    /* r300_us_to_glsl() flags for the GLSL this backend compiles. */
    uint32_t r300_glsl_flags;

    /*
     * Initialize the renderer backend.
     * Called once during device realize.
     * vram_ptr: host pointer to VRAM (from memory_region_get_ram_ptr)
     * vram_size: total VRAM size in bytes
     * Returns opaque backend state, or NULL on failure.
     */
    void *(*init)(uint8_t *vram_ptr, uint64_t vram_size);

    /*
     * Tear down the renderer backend.
     * Called during device exit.
     */
    void (*fini)(void *opaque);

    /*
     * Perform a scanout: read VRAM and produce a DisplaySurface.
     *
     * The backend should create or update a surface from the VRAM contents
     * described by the scanout descriptor. Returns a DisplaySurface that
     * the caller can pass to dpy_gfx_replace_surface().
     *
     * The returned surface is owned by the backend and must remain valid
     * until the next call to scanout() or fini().
     *
     * For software backend: creates a surface pointing directly into VRAM
     * (zero-copy) or a byte-swapped copy if needed.
     *
     * For Metal backend: could render to a Metal texture and return a
     * surface backed by shared memory.
     */
    void *(*scanout)(void *opaque, const PPCMacGPUScanout *desc);

    /*
     * Execute a 2D blit operation.
     * The backend may use hardware acceleration or fall back to software.
     * vram_ptr is provided for software backends that need direct access.
     * Returns 0 on success, -1 if the operation is not supported.
     */
    int (*blit_2d)(void *opaque, uint8_t *vram_ptr,
                   const PPCMacGPUBlit *blit);

    /*
     * Execute a 2D fill operation.
     * Fills a rectangle in VRAM with a solid color.
     * Returns 0 on success, -1 if not supported.
     */
    int (*fill_2d)(void *opaque, uint8_t *vram_ptr,
                   const PPCMacGPUBlit *fill);

    /*
     * Execute a 3D draw command.
     * Translates R200 3D state + draw command into host GPU operations.
     * The backend reads texture data from VRAM at the offsets specified
     * in the 3D state, renders the primitives, and writes the result
     * back to VRAM at rb3d_coloroffset.
     *
     * vram_ptr: host pointer to VRAM
     * vram_size: total VRAM size in bytes
     * state: snapshot of all relevant R200 3D registers
     * cmd: draw command descriptor
     *
     * Returns 0 on success, -1 if not supported (falls back to 2D path).
     */
    int (*draw_3d)(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                   const PPCMacGPU3DState *state,
                   const PPCMacGPU3DDrawCmd *cmd);


    /*
     * Render one decoded R200 draw straight into VRAM (see R200DrawPacket).
     * May be batched; the caller calls flush_r200() before touching VRAM.
     * Returns 0 on success, -1 if the packet uses something the backend
     * cannot render.
     *
     * A queue-async backend that enqueues the draw takes the tex[].host_data
     * buffers over (nulling them in pkt, which the caller may then read
     * only for its scalar fields); the caller frees what is left.
     */
    int (*draw_r200)(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                     const R200DrawPacket *pkt);

    /*
     * Render one assembled R300 draw (see hw/display/r300/r300_draw.h);
     * rt_gpu_addr and tex[].gpu_addr are already VRAM offsets.  Batched
     * with draw_r200 work: flush_r200() finishes both.
     *
     * A queue-async backend that enqueues the draw takes the packet's owned
     * buffers over (glsl, vs_glsl, verts, vs_in, vs_idx, vs_u and
     * tex[].host_data, nulling them in pkt); after the call the packet's
     * scalar fields may still be read, and r300_draw_free() frees what is
     * left.
     */
    int (*draw_r300)(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                     const struct R300DrawPacket *pkt);

    /*
     * The R300 Z-pass (occlusion) counter that draws with
     * uniforms.zpass_count set add to.  Finishes batched work first, then
     * returns the count; with reset, sets it to value (ZB_ZPASS_DATA).
     */
    uint32_t (*zpass_r300)(void *opaque, bool reset, uint32_t value);

    /*
     * Finish all batched draw_r200 work.  The device calls this before any
     * CPU-side access to VRAM that 3D rendering may have written.
     */
    bool (*flush_r200)(void *opaque);   /* true if work was pending */

    /*
     * Commit batched draw_r200 work without waiting.  Returns a nonzero
     * sequence number and later calls done(arg, seq) — from a backend
     * thread — once that work has completed; returns 0 if nothing was
     * pending.  flush_r200() also waits for everything submitted this way.
     */
    uint32_t (*submit_r200)(void *opaque,
                            void (*done)(void *arg, uint32_t seq), void *arg);

    /*
     * A 2D fill wrote VRAM (pixel rect of the surface at offset/pitch, in
     * bytes, bpp bytes per pixel, fill value).  Lets the backend mirror depth
     * buffer clears into its private depth storage.  Called after the fill.
     */
    /*
     * Does batched, unflushed draw_r200 work touch VRAM [lo, hi)?  If
     * write_access, any pending read or write of the range counts; otherwise
     * only pending writes.  The device flushes only when this says so.
     */
    bool (*range_busy_r200)(void *opaque, uint64_t lo, uint64_t hi,
                            bool write_access);

    void (*fill_notify_r200)(void *opaque, uint32_t offset, uint32_t pitch,
                             uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                             uint32_t bpp, uint32_t value);

    /*
     * Backends that keep copies of VRAM (Vulkan: images on the GPU) learn
     * about CPU-side writes here; the device keeps VRAM dirty logging on
     * and marks its own writes dirty.  Optional — may be NULL.
     */
    void (*set_dirty_source)(void *opaque, PPCMacGPUDirtyFn fn, void *arg);

    /*
     * Notify the backend that a display mode change occurred.
     * The backend may need to resize textures or reallocate surfaces.
     */
    void (*mode_change)(void *opaque, const PPCMacGPUScanout *new_mode);

    /*
     * Query backend capabilities.
     * Returns a bitmask of PPC_MAC_GPU_RENDERER_CAP_* flags.
     */
    uint32_t (*get_caps)(void *opaque);

    /*
     * Mirror a 2D engine write into the backend's shadow render target.
     *
     * The 2D engine writes straight into VRAM, but the Metal backend keeps
     * off-screen render targets in its own shadow textures.  Without this
     * write-through, a 2D blit into a region the backend also renders to
     * would be invisible to subsequent 3D draws.
     *
     * Coordinates are in pixels; offsets and pitch are in bytes.
     * Optional — may be NULL.
     */
    void (*srt_write_through)(void *opaque, uint8_t *vram_ptr,
                              uint32_t dst_offset, uint32_t dst_pitch,
                              uint32_t dst_x, uint32_t dst_y,
                              uint32_t width, uint32_t height,
                              uint32_t bpp);

    /*
     * Report the window-drag rectangle the backend has inferred from the
     * compositor's draw stream.
     *
     * Fills in the drag origin, the dragged surface size and a generation
     * counter that increments each time the origin moves.  Returns true if
     * a drag is currently being tracked.  Optional — may be NULL.
     */
    bool (*get_drag_state)(void *opaque,
                           uint32_t *origin_x, uint32_t *origin_y,
                           uint32_t *blit_w, uint32_t *blit_h,
                           uint32_t *frame_gen);

    /*
     * Hand back the pixels the backend saved from the dragged window body,
     * for diagnostics.  Returns NULL when no snapshot is held.
     * Optional — may be NULL.
     */
    const uint32_t *(*get_drag_snap)(void *opaque,
                                     uint32_t *width, uint32_t *height);

    /*
     * Flush any pending drag-body paste into VRAM.
     *
     * Called from the display update path, after all compositor rendering
     * for the frame has been replayed.  Optional — may be NULL.
     */
    void (*flush_drag_paste)(void *opaque, uint8_t *vram);

} PPCMacGPURenderer;

/* Renderer capability flags */
#define PPC_MAC_GPU_RENDERER_CAP_2D_BLIT   (1 << 0)
#define PPC_MAC_GPU_RENDERER_CAP_2D_FILL   (1 << 1)
#define PPC_MAC_GPU_RENDERER_CAP_3D        (1 << 2)
#define PPC_MAC_GPU_RENDERER_CAP_ACCEL     (1 << 3)  /* hardware accelerated */

/*
 * Get the default renderer (software fallback).
 * Always available on all platforms.
 */
PPCMacGPURenderer *ppc_mac_gpu_renderer_sw(void);

/*
 * Get the Metal renderer (macOS only).
 * Returns NULL if Metal is not available.
 * Defined in ppc_mac_gpu_metal.m, compiled only on macOS.
 */
#ifdef CONFIG_DARWIN
PPCMacGPURenderer *ppc_mac_gpu_renderer_metal(void);

/*
 * Allocate VRAM as a shared MTLBuffer so that the guest CPU and the Metal
 * GPU address the same physical pages, the way real VRAM behaves on an R200.
 *
 * Returns the host pointer to hand to memory_region_init_ram_ptr(), and
 * stores an opaque handle in *opaque_out for ppc_mac_gpu_metal_free_vram().
 * Returns NULL if the shared allocation is unavailable; the caller then
 * falls back to ordinary QEMU-managed RAM.
 */
void *ppc_mac_gpu_metal_alloc_vram(uint64_t vram_size, void **opaque_out);
void ppc_mac_gpu_metal_free_vram(void *opaque);
#endif

#ifdef CONFIG_PPC_MAC_GPU_VULKAN
/*
 * The Vulkan renderer (R300 draws only).  VRAM must come from
 * ppc_mac_gpu_vulkan_alloc_vram(): host-visible Vulkan memory the GPU
 * copies to and from its own images (it never renders into it, so a
 * discrete GPU works the same way).  Both return NULL on failure, with
 * the reason in ppc_mac_gpu_vulkan_error().
 */
PPCMacGPURenderer *ppc_mac_gpu_renderer_vulkan(void);
void *ppc_mac_gpu_vulkan_alloc_vram(uint64_t vram_size, void **opaque_out);
void ppc_mac_gpu_vulkan_free_vram(void *opaque);
const char *ppc_mac_gpu_vulkan_error(void);
#endif

#endif /* HW_DISPLAY_PPC_MAC_GPU_RENDERER_H */
