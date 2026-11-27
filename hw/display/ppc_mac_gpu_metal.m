/*
 * PPC Mac GPU - Metal Rendering Backend
 *
 * Translates ATI Radeon 9200 (R200) 3D commands into Apple Metal draw calls.
 * Primary use case: Quartz Extreme window compositing — textured quads with
 * alpha blending. This is NOT a full OpenGL implementation, just enough R200
 * 3D pipeline emulation to satisfy Tiger's QE compositor.
 *
 * Architecture:
 *   1. Guest writes 3D state via PM4 Type 0 packets (stored in regs_3d[])
 *   2. Guest issues draw commands via PM4 Type 3 (DRAW_VBUF/IMMD/INDX)
 *   3. Device code snapshots 3D state into PPCMacGPU3DState
 *   4. This backend translates state → Metal pipeline, reads textures from
 *      VRAM, renders to an offscreen target, copies result back to VRAM
 *      at rb3d_coloroffset for the existing display scanout to pick up
 *
 * Threading: All Metal work is synchronous on the vCPU thread. We call
 * waitUntilCompleted after each command buffer. This is acceptable because
 * QE draws are infrequent (~60fps, 10-30 textured quads per frame).
 *
 * Copyright (c) 2024
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/atomic.h"
#include "qemu/log.h"

#ifdef CONFIG_DARWIN

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <CommonCrypto/CommonDigest.h>

#include "ppc_mac_gpu_renderer.h"
#include "r300/r300_draw.h"
#include "r300/r300_spirv.h"
#include "ppc_mac_gpu_3d_regs.h"
#include "ppc_mac_gpu_surface.h"

/* g_frame_tracker is defined in ppc_mac_gpu.c (built on every host) */

/* ========================================================================
 * Zero-copy unified VRAM — MTLBuffer shared with guest CPU
 *
 * On real R200 hardware, VRAM is a single physical memory that both the
 * CPU and GPU access.  The WindowServer writes window body pixels via
 * CPU memcpy; the QE compositor draws shadows/corners via the 3D engine;
 * both happen to the same physical bytes.
 *
 * We replicate this by allocating VRAM as an MTLBuffer with Shared storage.
 * The buffer's .contents pointer is registered with QEMU as the guest's
 * VRAM BAR via memory_region_init_ram_ptr.  Metal render targets and
 * texture fetches alias the same buffer.  CPU writes are immediately
 * visible to Metal and vice versa — zero copies.
 * ======================================================================== */

/* Opaque handle for Metal VRAM allocation (stored before Metal renderer init) */
typedef struct {
    id<MTLDevice>  device;
    id<MTLBuffer>  vramBuffer;
    uint64_t       size;
} MetalVRAMAlloc;

/* Global handle — set by alloc_vram, consumed by metal_init */
static MetalVRAMAlloc *g_metal_vram_alloc = NULL;

void *ppc_mac_gpu_metal_alloc_vram(uint64_t vram_size, void **opaque_out)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            qemu_log("ppc-mac-gpu-metal: no Metal device for VRAM alloc\n");
            return NULL;
        }
        /*
         * The shaders read the colour and depth they blend into with
         * framebuffer fetch ([[color(n)]] inputs), and render into linear
         * texture views of this shared buffer.  Only Apple GPUs do either;
         * the Intel and AMD GPUs of Intel Macs need the Vulkan backend
         * (MoltenVK), which works without them.
         */
        if (![dev supportsFamily:MTLGPUFamilyApple1]) {
            qemu_log("ppc-mac-gpu-metal: %s is not an Apple GPU; the Metal "
                     "renderer needs one (use renderer=vulkan)\n",
                     [[dev name] UTF8String]);
            return NULL;
        }

        /* Shared storage: CPU and GPU access the same physical memory.
         * This is the key to zero-copy — guest CPU writes are instantly
         * visible to Metal render passes, and vice versa. */
        id<MTLBuffer> buf = [dev newBufferWithLength:vram_size
                                             options:MTLResourceStorageModeShared];
        if (!buf) {
            qemu_log("ppc-mac-gpu-metal: MTLBuffer alloc failed (%llu bytes)\n",
                     (unsigned long long)vram_size);
            return NULL;
        }

        /* Zero-initialize (matches real VRAM power-on state) */
        memset([buf contents], 0, vram_size);

        MetalVRAMAlloc *alloc = g_new0(MetalVRAMAlloc, 1);
        alloc->device = dev;       /* retained by ARC / manually */
        alloc->vramBuffer = buf;
        alloc->size = vram_size;

        /* Store globally so metal_init can find it */
        g_metal_vram_alloc = alloc;

        if (opaque_out) {
            *opaque_out = alloc;
        }

        qemu_log("ppc-mac-gpu-metal: VRAM allocated as MTLBuffer "
                 "(%llu MB, shared storage, zero-copy)\n",
                 (unsigned long long)(vram_size / (1024 * 1024)));

        return [buf contents];  /* host pointer for memory_region_init_ram_ptr */
    }
}

void ppc_mac_gpu_metal_free_vram(void *opaque)
{
    MetalVRAMAlloc *alloc = opaque;
    if (alloc) {
        alloc->vramBuffer = nil;
        alloc->device = nil;
        g_free(alloc);
    }
}

/* ========================================================================
 * Metal shader source (compiled at runtime)
 * ======================================================================== */

static NSString *const kShaderSource = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct VertexIn {\n"
"    float2 position [[attribute(0)]];\n"
"    float2 texcoord [[attribute(1)]];\n"
"    float4 color    [[attribute(2)]];\n"
"};\n"
"\n"
"struct VertexOut {\n"
"    float4 position [[position]];\n"
"    float2 texcoord;\n"
"    float4 color;\n"
"};\n"
"\n"
"struct Uniforms {\n"
"    float2 viewport_size;  // RT width, height in pixels\n"
"    float2 viewport_scale; // from R200 VPORT regs\n"
"    float2 viewport_offset;\n"
"    uint   use_texture;    // 0=solid color, 1=textured\n"
"    uint   tex_format;     // 0=ARGB8888, 1=I8 intensity\n"
"    float2 tex_size;       // texture width, height in pixels\n"
"    uint   swap_rb;        // 1=swap R<->B for BE VRAM output\n"
"    uint   _pad3;\n"
"};\n"
"\n"
"vertex VertexOut vertex_main(\n"
"    VertexIn in [[stage_in]],\n"
"    constant Uniforms &uniforms [[buffer(1)]])\n"
"{\n"
"    VertexOut out;\n"
"    // Convert pixel coordinates to NDC [-1, 1]\n"
"    float2 ndc;\n"
"    ndc.x = (in.position.x / uniforms.viewport_size.x) * 2.0 - 1.0;\n"
"    ndc.y = 1.0 - (in.position.y / uniforms.viewport_size.y) * 2.0;\n"
"    out.position = float4(ndc, 0.0, 1.0);\n"
"    // Normalize texcoords from pixel space to 0-1\n"
"    if (uniforms.tex_size.x > 0.0 && uniforms.tex_size.y > 0.0) {\n"
"        out.texcoord = in.texcoord / uniforms.tex_size;\n"
"    } else {\n"
"        out.texcoord = in.texcoord;\n"
"    }\n"
"    out.color = in.color;\n"
"    return out;\n"
"}\n"
"\n"
"fragment float4 fragment_textured(\n"
"    VertexOut in [[stage_in]],\n"
"    texture2d<float> tex [[texture(0)]],\n"
"    sampler smp [[sampler(0)]],\n"
"    constant Uniforms &uniforms [[buffer(1)]])\n"
"{\n"
"    float4 result;\n"
"    if (uniforms.use_texture != 0) {\n"
"        float4 texel = tex.sample(smp, in.texcoord);\n"
"        if (uniforms.tex_format == 1) {\n"
"            // I8 intensity: R8Unorm gives (R,0,0,1)\n"
"            // R200 GL_INTENSITY: replicate to all channels (I,I,I,I)\n"
"            texel = float4(texel.r, texel.r, texel.r, texel.r);\n"
"        }\n"
"        result = texel * in.color;\n"
"    } else {\n"
"        result = in.color;\n"
"    }\n"
"    // Endianness correction for direct-to-VRAM rendering:\n"
"    // Metal BGRA8Unorm stores bytes B,G,R,A (LE).\n"
"    // Guest expects ARGB in BE: bytes A,R,G,B.\n"
"    // Swapping R<->B and moving A produces the right byte order\n"
"    // so the display update's bswap32 yields correct colors.\n"
"    if (uniforms.swap_rb != 0) {\n"
"        result = float4(result.b, result.g, result.r, result.a);\n"
"    }\n"
"    return result;\n"
"}\n";

/* ========================================================================
 * Internal types
 * ======================================================================== */

/* Vertex layout matching the Metal shader */
typedef struct MetalVertex {
    float position[2];
    float texcoord[2];
    float color[4];
} MetalVertex;

/* Uniforms buffer */
typedef struct MetalUniforms {
    float viewport_size[2];
    float viewport_scale[2];
    float viewport_offset[2];
    uint32_t use_texture;
    uint32_t tex_format;  /* 0=ARGB8888, 1=I8 (intensity replicated to RGBA) */
    float tex_size[2];  /* texture width, height — for normalizing pixel texcoords */
    uint32_t swap_rb;   /* 1=swap R<->B for BE VRAM output (direct-to-VRAM path) */
    uint32_t _pad3;
} MetalUniforms;

/* Blend pipeline cache entry */
#define BLEND_CACHE_SIZE 8
typedef struct BlendCacheEntry {
    uint32_t key;  /* (src_blend << 8) | dst_blend */
    id<MTLRenderPipelineState> pipeline;
} BlendCacheEntry;

/* Backend state */
typedef struct PPCMacGPUMetalState {
    /* Metal objects */
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;

    /* Zero-copy unified VRAM buffer (shared with guest CPU) */
    id<MTLBuffer> vramBuffer;      /* MTLBuffer aliasing guest VRAM, or nil */
    uint32_t rt_vram_offset;       /* VRAM byte offset of current RT alias */
    id<MTLRenderPipelineState> pipelineState;        /* srcAlpha/oneMinusSrcAlpha */
    id<MTLRenderPipelineState> pipelineStateNopBlend; /* src=Zero, dst=One (NOP) */
    id<MTLSamplerState> samplerNearest;
    id<MTLSamplerState> samplerBilinear;
    id<MTLTexture> whiteDummyTexture;    /* 1x1 opaque white for untextured draws */

    /* Blend pipeline cache — keyed by (src_blend, dst_blend) */
    BlendCacheEntry blend_cache[BLEND_CACHE_SIZE];
    int blend_cache_count;
    id<MTLLibrary> shaderLibrary;  /* kept for creating new pipeline states */
    MTLVertexDescriptor *vertexDescriptor; /* kept for creating new pipelines */

    /* Per-(offset, pitch) render target cache.
     * Each unique (color_offset, color_pitch_pixels) gets its own Metal
     * texture and output buffer.  This prevents cross-offset contamination
     * when MTLLoadActionLoad preserves stale content from a different
     * offset's draws. */
#define RT_CACHE_MAX 16
    struct {
        bool     valid;
        bool     needs_clear;    /* true = first draw should use MTLLoadActionClear */
        uint32_t offset;         /* RB3D_COLOROFFSET */
        uint32_t pitch_pixels;   /* color pitch in pixels */
        uint32_t width;
        uint32_t height;
        uint32_t frame_gen;      /* frame generation when last used */
        id<MTLTexture> texture;
        id<MTLBuffer>  outbuf;
    } rt_cache[RT_CACHE_MAX];
    uint32_t rt_frame_gen;       /* incremented on each compositor frame boundary */
    bool qe_compositing_active;  /* set when we first see semi-transparent
                                  * SRT pixels, indicating QE shadow rendering
                                  * is active and opaque pixels are wallpaper */

    /*
     * Drag tracking — per-compositor-surface origin tracker.
     * Detects PRESENT_BLIT origin shifts (window drag) and performs
     * an intra-framebuffer body copy to simulate the IOAccelerator
     * surface-copy that ATIEnableWideBlitSupport would provide.
     */
#define DRAG_TRACKER_MAX 4
#define DRAG_INSET_LEFT    12  /* skip left border/shadow */
#define DRAG_INSET_RIGHT   12  /* skip right border/shadow */
#define DRAG_INSET_TOP     22  /* leave titlebar/frame to QE */
#define DRAG_INSET_BOTTOM  16  /* leave bottom frame/status edge to QE */
    struct {
        bool     active;
        uint32_t src_offset;       /* compositor RT VRAM offset */
        uint32_t prev_origin_x, prev_origin_y;
        uint32_t prev_frame_gen;
        uint32_t cur_origin_x, cur_origin_y;
        uint32_t cur_frame_gen;
        uint32_t blit_w, blit_h;   /* full PRESENT_BLIT dimensions */
        bool     copy_pending;
        int32_t  dx, dy;           /* drag delta */
    } drag_trackers[DRAG_TRACKER_MAX];

    /* Body protection zone — set by drag_body_copy, checked in SRT pixel loop.
     * When active, fully opaque SRT pixels inside this rectangle are SKIPPED,
     * preserving the just-moved body pixels in the framebuffer. */
    bool     drag_protect_active;
    int32_t  drag_protect_x, drag_protect_y;
    int32_t  drag_protect_w, drag_protect_h;

    /* Legacy fields kept for API compat */
    uint32_t *drag_body_buf;
    uint32_t drag_body_buf_size;
    bool     drag_paste_pending;
    int32_t  drag_paste_x, drag_paste_y;
    int32_t  drag_paste_w, drag_paste_h;
    uint32_t drag_paste_pitch;

    /* Active RT pointers — set by rt_cache_bind(), used by draw code */
    id<MTLTexture> renderTarget;
    uint32_t rt_width;
    uint32_t rt_height;
    id<MTLBuffer> outputBuffer;
    uint32_t scanout_width;
    uint32_t scanout_height;

    /*
     * Per-RT pitch tracking to detect pitch changes.
     * On real R200 hardware, MC macro-tiling means different pitches at the
     * same base address access different physical VRAM locations.  In our
     * linear emulation, reusing the same VRAM offset with a different pitch
     * causes cross-contamination.  When we detect a pitch change, we clear
     * the Metal RT instead of loading garbled VRAM data.
     */
#define RT_PITCH_CACHE_SIZE 16
    struct {
        uint32_t offset;
        uint32_t pitch;
        bool valid;
    } rt_pitch_cache[RT_PITCH_CACHE_SIZE];

    /* VRAM access */
    uint8_t *vram_ptr;
    uint64_t vram_size;

    /*
     * Shadow render target cache.
     *
     * On real R200 hardware, MC macro-tiling means different pitches at the
     * same base VRAM offset access physically separate memory banks.  In our
     * emulator (linear VRAM, no MC), this causes cross-contamination.
     *
     * Instead of fighting VRAM layout issues, we keep a *shadow copy* of
     * every 3D-rendered render target in host memory.  The Metal backend
     * writes composited pixels into these shadow buffers (indexed by offset +
     * pitch).  When the 2D BLT engine does a PRESENT_BLIT from a compositing
     * RT, it reads from the shadow buffer — bypassing VRAM entirely and
     * producing correct output regardless of pitch conflicts.
     */
#define SHADOW_RT_MAX 64
    struct {
        bool   valid;
        bool   has_conflict;    /* true if this offset has had pitch conflicts */
        bool   consumed;        /* true after a BLIT read from this SRT */
        bool   origin_valid;    /* true when content_origin_* are set */
        bool   stale;           /* true = not redrawn since last compositor frame */
        uint32_t offset;        /* RB3D_COLOROFFSET */
        uint32_t pitch_pixels;  /* color pitch in pixels */
        uint32_t width;         /* shadow buffer width = pitch_pixels */
        uint32_t height;        /* shadow buffer height */
        uint32_t *pixels;       /* BE-format pixels, stride = width */
        uint32_t alloc_count;   /* allocated pixel count */
        /* Content origin: the PRESENT_BLIT dst_xy when this SRT was last
         * drawn to and consumed.  Used to read from the correct SRT position
         * when the BLIT dst_xy changes (e.g., window drag repositioning). */
        int32_t  content_origin_x;
        int32_t  content_origin_y;
    } shadow_rts[SHADOW_RT_MAX];

    /* Render state */
    bool initialized;

    /* Pipeline tracker */
    FrameTracker frame_tracker;
} PPCMacGPUMetalState;

/* ========================================================================
 * Helper: reinterpret uint32_t as float (used for viewport regs)
 * ======================================================================== */
static inline float u32_to_float(uint32_t v)
{
    union { uint32_t u; float f; } u;
    u.u = v;
    return u.f;
}

/* ========================================================================
 * Helper: Map R200 blend factor to MTLBlendFactor
 * ======================================================================== */
static MTLBlendFactor r200_to_mtl_blend(uint32_t r200_factor)
{
    switch (r200_factor) {
    case R200_BLEND_ZERO:                  return MTLBlendFactorZero;
    case R200_BLEND_ONE:                   return MTLBlendFactorOne;
    case R200_BLEND_SRC_COLOR:             return MTLBlendFactorSourceColor;
    case R200_BLEND_ONE_MINUS_SRC_COLOR:   return MTLBlendFactorOneMinusSourceColor;
    case R200_BLEND_SRC_ALPHA:             return MTLBlendFactorSourceAlpha;
    case R200_BLEND_ONE_MINUS_SRC_ALPHA:   return MTLBlendFactorOneMinusSourceAlpha;
    case R200_BLEND_DST_ALPHA:             return MTLBlendFactorDestinationAlpha;
    case R200_BLEND_ONE_MINUS_DST_ALPHA:   return MTLBlendFactorOneMinusDestinationAlpha;
    case R200_BLEND_DST_COLOR:             return MTLBlendFactorDestinationColor;
    case R200_BLEND_ONE_MINUS_DST_COLOR:   return MTLBlendFactorOneMinusDestinationColor;
    case R200_BLEND_SRC_ALPHA_SAT:         return MTLBlendFactorSourceAlphaSaturated;
    default:                               return MTLBlendFactorOne;
    }
}

/* ========================================================================
 * Get or create a blend pipeline state for the given R200 blend factors.
 * Caches pipeline states to avoid recompilation overhead.
 * ======================================================================== */
static id<MTLRenderPipelineState> metal_get_blend_pipeline(
    PPCMacGPUMetalState *st, uint32_t src_blend, uint32_t dst_blend)
{
    uint32_t key = (src_blend << 8) | dst_blend;

    /* Search cache */
    for (int i = 0; i < st->blend_cache_count; i++) {
        if (st->blend_cache[i].key == key) {
            return st->blend_cache[i].pipeline;
        }
    }

    /* Create new pipeline state with these blend factors */
    if (!st->shaderLibrary || !st->vertexDescriptor) {
        return st->pipelineState; /* fallback */
    }

    @autoreleasepool {
        id<MTLFunction> vertexFunc = [st->shaderLibrary newFunctionWithName:@"vertex_main"];
        id<MTLFunction> fragmentFunc = [st->shaderLibrary newFunctionWithName:@"fragment_textured"];
        if (!vertexFunc || !fragmentFunc) {
            return st->pipelineState;
        }

        MTLRenderPipelineDescriptor *pipeDesc =
            [[MTLRenderPipelineDescriptor alloc] init];
        pipeDesc.vertexFunction = vertexFunc;
        pipeDesc.fragmentFunction = fragmentFunc;
        pipeDesc.vertexDescriptor = st->vertexDescriptor;
        pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        MTLBlendFactor mtl_src = r200_to_mtl_blend(src_blend);
        MTLBlendFactor mtl_dst = r200_to_mtl_blend(dst_blend);

        pipeDesc.colorAttachments[0].blendingEnabled = YES;
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = mtl_src;
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = mtl_dst;
        /* Alpha channel: use ONE / ONE_MINUS_SRC_ALPHA for premultiplied,
         * or same as color for additive */
        if (mtl_src == MTLBlendFactorOne && mtl_dst == MTLBlendFactorOne) {
            /* Additive: alpha adds too */
            pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        } else {
            /* Premultiplied or standard: preserve alpha channel */
            pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }

        NSError *error = nil;
        id<MTLRenderPipelineState> pipeline =
            [st->device newRenderPipelineStateWithDescriptor:pipeDesc error:&error];
        if (!pipeline) {
            qemu_log("METAL: blend pipeline creation failed for src=0x%x dst=0x%x: %s\n",
                     src_blend, dst_blend,
                     error ? [[error localizedDescription] UTF8String] : "unknown");
            return st->pipelineState;
        }

        /* Cache it */
        if (st->blend_cache_count < BLEND_CACHE_SIZE) {
            st->blend_cache[st->blend_cache_count].key = key;
            st->blend_cache[st->blend_cache_count].pipeline = pipeline;
            st->blend_cache_count++;
        } else {
            /* Evict oldest */
            st->blend_cache[0].pipeline = nil;
            memmove(&st->blend_cache[0], &st->blend_cache[1],
                    (BLEND_CACHE_SIZE - 1) * sizeof(BlendCacheEntry));
            st->blend_cache[BLEND_CACHE_SIZE - 1].key = key;
            st->blend_cache[BLEND_CACHE_SIZE - 1].pipeline = pipeline;
        }

        qemu_log("METAL: created blend pipeline src=0x%x(%d) dst=0x%x(%d) "
                 "cache_count=%d\n",
                 src_blend, (int)mtl_src, dst_blend, (int)mtl_dst,
                 st->blend_cache_count);

        return pipeline;
    }
}

/* ========================================================================
 * Create a render target aliased directly into VRAM (zero-copy path)
 *
 * Instead of creating a standalone MTLTexture, this creates a texture
 * VIEW over the shared VRAM MTLBuffer.  GPU draws go directly into the
 * same memory the guest CPU reads/writes.  This is how real VRAM works:
 * CPU memcpy and GPU 3D draws coexist on the same canvas.
 *
 * The texture is created at a specific VRAM byte offset with a specific
 * pitch (bytes per row).  Different compositor layers may use different
 * pitches at the same offset — Metal handles this via separate texture
 * views over the same buffer.
 * ======================================================================== */
static bool metal_ensure_vram_render_target(PPCMacGPUMetalState *st,
                                             uint32_t vram_offset,
                                             uint32_t width, uint32_t height,
                                             uint32_t pitch_pixels)
{
    if (!st->vramBuffer) {
        /* No zero-copy VRAM — fall through to legacy path */
        return false;
    }

    /*
     * MASTER PITCH: always use 4096 bytes per row (1024 pixels).
     *
     * The PRESENT_BLIT reads at the CRTC pitch (1024 pixels = 4096 bpr).
     * All 3D draws at this offset must use the same stride so the BLIT
     * reads correctly.  The guest's color_pitch (832, 896, etc.) is
     * irrelevant — on real R200, MC tiling reconciles different pitch
     * views.  In our linear VRAM, we force a single consistent stride.
     *
     * 4096 is already 256-byte aligned, satisfying Metal's requirements.
     */
    uint32_t master_bpr = 4096;           /* 1024 pixels * 4 bytes */
    uint32_t master_width = 1024;         /* pixels per row */
    uint64_t buffer_offset = (uint64_t)vram_offset;

    /* Validate offset + size fits within VRAM */
    uint64_t needed = buffer_offset + (uint64_t)height * master_bpr;
    if (needed > st->vram_size) {
        return false;
    }

    uint32_t aligned_bpr = master_bpr;
    uint32_t aligned_width = master_width;

    /* Check if we can reuse the current RT (same offset, pitch, size) */
    if (st->renderTarget &&
        st->rt_vram_offset == vram_offset &&
        st->rt_width == aligned_width &&
        st->rt_height >= height) {
        return true;  /* Already bound */
    }

    /* Create texture descriptor */
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:aligned_width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    /* Create texture as a VIEW over the VRAM MTLBuffer.
     * This is the key zero-copy operation: the texture's pixel data
     * IS the VRAM bytes at the specified offset. */
    id<MTLTexture> tex = [st->vramBuffer
        newTextureWithDescriptor:desc
                          offset:buffer_offset
                     bytesPerRow:aligned_bpr];
    if (!tex) {
        static int fail_log = 0;
        if (fail_log < 5) {
            qemu_log("ppc-mac-gpu-metal: VRAM texture alias failed "
                     "off=0x%x %ux%u pitch=%u bpr=%u\n",
                     vram_offset, width, height, pitch_pixels, aligned_bpr);
            fail_log++;
        }
        return false;
    }

    st->renderTarget = tex;
    st->outputBuffer = nil;  /* Not needed — texture IS VRAM */
    st->rt_width = aligned_width;
    st->rt_height = height;
    st->rt_vram_offset = vram_offset;

    static int alias_log = 0;
    if (alias_log < 20) {
        qemu_log("ppc-mac-gpu-metal: [VRAM_RT] alias off=0x%x %ux%u "
                 "pitch=%u bpr=%u\n",
                 vram_offset, aligned_width, height, pitch_pixels, aligned_bpr);
        alias_log++;
    }

    return true;
}

/* ========================================================================
 * Create / resize the offscreen render target (legacy path)
 * ======================================================================== */
static bool metal_ensure_render_target(PPCMacGPUMetalState *st,
                                        uint32_t width, uint32_t height)
{
    if (st->renderTarget && st->rt_width == width && st->rt_height == height) {
        return true;
    }

    /*
     * Save old RT reference for content preservation during resize.
     * When the RT grows (e.g., vertex pre-scan found wider vertices),
     * we must copy the old content to the new RT so that subsequent draws
     * (with MTLLoadActionLoad) blend against the correct background,
     * not a black/zero texture.
     */
    id<MTLTexture> oldRT = st->renderTarget;
    uint32_t old_w = st->rt_width;
    uint32_t old_h = st->rt_height;

    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    /* Use Shared storage so we can upload VRAM content via replaceRegion
     * before rendering (needed for correct alpha blending with existing
     * framebuffer content, especially for tiled surfaces). */
    desc.storageMode = MTLStorageModeShared;

    st->renderTarget = [st->device newTextureWithDescriptor:desc];
    if (!st->renderTarget) {
        qemu_log_mask(LOG_UNIMP, "ppc-mac-gpu-metal: failed to create render target %ux%u\n",
                      width, height);
        return false;
    }

    /* Shared output buffer for CPU readback */
    uint64_t buf_size = (uint64_t)width * height * 4;
    st->outputBuffer = [st->device newBufferWithLength:buf_size
                                    options:MTLResourceStorageModeShared];
    if (!st->outputBuffer) {
        qemu_log_mask(LOG_UNIMP, "ppc-mac-gpu-metal: failed to create output buffer\n");
        return false;
    }

    /*
     * Copy old RT content to new RT if growing (preserve accumulated draws).
     * Uses a Metal blit command to copy the overlapping region.
     */
    if (oldRT && old_w > 0 && old_h > 0) {
        uint32_t copy_w = (old_w < width) ? old_w : width;
        uint32_t copy_h = (old_h < height) ? old_h : height;

        id<MTLCommandBuffer> blitCmd = [st->commandQueue commandBuffer];
        if (blitCmd) {
            id<MTLBlitCommandEncoder> blitEnc = [blitCmd blitCommandEncoder];
            if (blitEnc) {
                [blitEnc copyFromTexture:oldRT
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(copy_w, copy_h, 1)
                               toTexture:st->renderTarget
                        destinationSlice:0
                        destinationLevel:0
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blitEnc endEncoding];
                [blitCmd commit];
                [blitCmd waitUntilCompleted];
            }
        }
    }

    st->rt_width = width;
    st->rt_height = height;
    return true;
}

/* ========================================================================
 * Per-(offset, pitch) render target cache
 *
 * Each unique (color_offset, color_pitch_pixels) gets its own Metal
 * texture and output buffer.  A new frame_gen does NOT inherit stale
 * content — the texture is cleared on creation or frame boundary.
 * Reuse is allowed only when: same offset, same pitch, same frame_gen.
 * ======================================================================== */
static bool rt_cache_bind(PPCMacGPUMetalState *st,
                          uint32_t offset, uint32_t pitch_pixels,
                          uint32_t width, uint32_t height,
                          bool *out_needs_clear)
{
    /* Search for existing entry matching (offset, pitch, frame_gen) */
    int reuse_slot = -1;
    int free_slot = -1;
    int oldest_slot = 0;
    uint32_t oldest_gen = UINT32_MAX;

    for (int i = 0; i < RT_CACHE_MAX; i++) {
        if (!st->rt_cache[i].valid) {
            if (free_slot < 0) free_slot = i;
            continue;
        }
        if (st->rt_cache[i].offset == offset &&
            st->rt_cache[i].pitch_pixels == pitch_pixels &&
            st->rt_cache[i].frame_gen == st->rt_frame_gen) {
            reuse_slot = i;
            break;
        }
        /* Track oldest for eviction */
        if (st->rt_cache[i].frame_gen < oldest_gen) {
            oldest_gen = st->rt_cache[i].frame_gen;
            oldest_slot = i;
        }
    }

    if (reuse_slot >= 0) {
        /* Reuse — but may need to resize if dimensions grew */
        int s = reuse_slot;
        if (st->rt_cache[s].width >= width &&
            st->rt_cache[s].height >= height) {
            /* Exact match or larger — bind directly */
            st->renderTarget = st->rt_cache[s].texture;
            st->outputBuffer = st->rt_cache[s].outbuf;
            st->rt_width = st->rt_cache[s].width;
            st->rt_height = st->rt_cache[s].height;

            if (out_needs_clear) {
                *out_needs_clear = st->rt_cache[s].needs_clear;
            }
            st->rt_cache[s].needs_clear = false;

            static int reuse_log = 0;
            if (reuse_log < 30) {
                fprintf(stderr, "[RT_CACHE_REUSE] slot=%d off=0x%x pitch=%u "
                        "%ux%u gen=%u\n",
                        s, offset, pitch_pixels,
                        st->rt_cache[s].width, st->rt_cache[s].height,
                        st->rt_frame_gen);
                reuse_log++;
            }
            return true;
        }
        /* Need larger — fall through to create, will reuse this slot */
        free_slot = s;
        st->rt_cache[s].texture = nil;
        st->rt_cache[s].outbuf = nil;
        st->rt_cache[s].valid = false;
    }

    /* Allocate new entry */
    int slot = (free_slot >= 0) ? free_slot : oldest_slot;

    /* Evict old entry if reusing occupied slot */
    if (st->rt_cache[slot].valid) {
        static int inval_log = 0;
        if (inval_log < 20) {
            fprintf(stderr, "[RT_CACHE_INVALIDATE] slot=%d old off=0x%x "
                    "pitch=%u gen=%u → evicted for off=0x%x pitch=%u gen=%u\n",
                    slot,
                    st->rt_cache[slot].offset,
                    st->rt_cache[slot].pitch_pixels,
                    st->rt_cache[slot].frame_gen,
                    offset, pitch_pixels, st->rt_frame_gen);
            inval_log++;
        }
        st->rt_cache[slot].texture = nil;
        st->rt_cache[slot].outbuf = nil;
        st->rt_cache[slot].valid = false;
    }

    /* Create new Metal texture */
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    id<MTLTexture> tex = [st->device newTextureWithDescriptor:desc];
    if (!tex) {
        qemu_log_mask(LOG_UNIMP, "ppc-mac-gpu-metal: RT cache create failed %ux%u\n",
                      width, height);
        return false;
    }

    uint64_t buf_size = (uint64_t)width * height * 4;
    id<MTLBuffer> buf = [st->device newBufferWithLength:buf_size
                                    options:MTLResourceStorageModeShared];
    if (!buf) {
        qemu_log_mask(LOG_UNIMP, "ppc-mac-gpu-metal: RT cache outbuf failed\n");
        return false;
    }

    /* Mark for clear on first draw — the draw path will use
     * MTLLoadActionClear instead of MTLLoadActionLoad.  This avoids
     * a synchronous GPU wait here (which killed frame rate). */
    if (out_needs_clear) {
        *out_needs_clear = true;
    }
    st->rt_cache[slot].valid = true;
    st->rt_cache[slot].needs_clear = false;  /* caller handles via out_needs_clear */
    st->rt_cache[slot].offset = offset;
    st->rt_cache[slot].pitch_pixels = pitch_pixels;
    st->rt_cache[slot].width = width;
    st->rt_cache[slot].height = height;
    st->rt_cache[slot].frame_gen = st->rt_frame_gen;
    st->rt_cache[slot].texture = tex;
    st->rt_cache[slot].outbuf = buf;

    /* Bind as active */
    st->renderTarget = tex;
    st->outputBuffer = buf;
    st->rt_width = width;
    st->rt_height = height;

    static int create_log = 0;
    if (create_log < 40) {
        fprintf(stderr, "[RT_CACHE_CREATE] slot=%d off=0x%x pitch=%u "
                "%ux%u gen=%u\n",
                slot, offset, pitch_pixels, width, height,
                st->rt_frame_gen);
        create_log++;
    }
    return true;
}

/* ========================================================================
 * Create a temporary MTLTexture from guest VRAM data (with byte swap)
 * ======================================================================== */
static id<MTLTexture> metal_texture_from_vram(PPCMacGPUMetalState *st,
                                               uint32_t offset,
                                               uint32_t width,
                                               uint32_t height,
                                               uint32_t pitch,
                                               uint32_t format)
{
    if (width == 0 || height == 0 || pitch == 0) {
        return nil;
    }

    /* Validate bounds (use pitch for row stride, covers all bpp formats) */
    uint64_t tex_end = (uint64_t)offset + (uint64_t)(height - 1) * pitch +
                       (uint64_t)pitch;
    if (tex_end > st->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "ppc-mac-gpu-metal: texture exceeds VRAM "
                      "(offset=0x%x end=0x%"PRIx64" vram=0x%"PRIx64")\n",
                      offset, tex_end, st->vram_size);
        return nil;
    }

    /* Determine Metal pixel format */
    MTLPixelFormat mtlFmt = MTLPixelFormatBGRA8Unorm;
    uint32_t bytes_per_pixel = 4;

    /* Map R200 texture format to Metal pixel format */
    switch (format & 0x1F) {
    case R200_TXFORMAT_ARGB8888:
    case R200_TXFORMAT_XRGB8888:
        mtlFmt = MTLPixelFormatBGRA8Unorm;
        bytes_per_pixel = 4;
        break;
    case R200_TXFORMAT_RGB565:
        mtlFmt = MTLPixelFormatB5G6R5Unorm;
        bytes_per_pixel = 2;
        break;
    case R200_TXFORMAT_ARGB1555:
        mtlFmt = MTLPixelFormatA1BGR5Unorm;
        bytes_per_pixel = 2;
        break;
    case R200_TXFORMAT_I8:
    case R200_TXFORMAT_Y8:
        /* 8-bit intensity/luminance — single channel texture.
         * R200 treats I8 as GL_INTENSITY: replicate to all RGBA channels.
         * We load as R8Unorm and expand in the fragment shader. */
        mtlFmt = MTLPixelFormatR8Unorm;
        bytes_per_pixel = 1;
        break;
    case R200_TXFORMAT_AI88:
        /* 16-bit alpha + intensity */
        mtlFmt = MTLPixelFormatRG8Unorm;
        bytes_per_pixel = 2;
        break;
    default:
        /* Fall back to BGRA8 */
        {
            static int fmt_warn = 0;
            if (fmt_warn < 10) {
                qemu_log_mask(LOG_UNIMP,
                    "ppc-mac-gpu-metal: unhandled texture format 0x%x\n",
                    format & 0x1F);
                fmt_warn++;
            }
        }
        break;
    }

    /* Create texture */
    MTLTextureDescriptor *texDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:mtlFmt
        width:width height:height mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeManaged;

    id<MTLTexture> tex = [st->device newTextureWithDescriptor:texDesc];
    if (!tex) {
        return nil;
    }

    /* Copy texture data from VRAM with byte swap (BE -> LE) */
    uint8_t *src = st->vram_ptr + offset;

    if (bytes_per_pixel == 1) {
        /* 8-bit format (I8, Y8): no byte swap needed, copy directly */
        uint32_t row_bytes = width;

        for (uint32_t y = 0; y < height; y++) {
            uint8_t *src_row = src + y * pitch;

            [tex replaceRegion:MTLRegionMake2D(0, y, width, 1)
                   mipmapLevel:0
                     withBytes:src_row
                   bytesPerRow:row_bytes];
        }
    } else if (bytes_per_pixel == 4) {
        /* Byte-swap ARGB (BE) → BGRA (LE) for each pixel */
        uint32_t row_bytes = width * 4;
        uint8_t *tmp = g_malloc(row_bytes);

        for (uint32_t y = 0; y < height; y++) {
            uint32_t *src_row = (uint32_t *)(src + y * pitch);
            uint32_t *dst_row = (uint32_t *)tmp;

            for (uint32_t x = 0; x < width; x++) {
                /* BE VRAM stores XRGB as bytes [X,R,G,B]
                 * which is uint32_t BE = 0xXRGB.
                 * For Metal BGRA8Unorm, we need bytes [B,G,R,A].
                 * bswap32(0xXRGB_be) = 0xBGRX_le which is [B,G,R,X] = BGRA! */
                dst_row[x] = __builtin_bswap32(src_row[x]);
            }

            [tex replaceRegion:MTLRegionMake2D(0, y, width, 1)
                   mipmapLevel:0
                     withBytes:tmp
                   bytesPerRow:row_bytes];
        }
        g_free(tmp);
    } else {
        /* 16-bit formats: byte-swap each u16 */
        uint32_t row_bytes = width * 2;
        uint8_t *tmp = g_malloc(row_bytes);

        for (uint32_t y = 0; y < height; y++) {
            uint16_t *src_row = (uint16_t *)(src + y * pitch);
            uint16_t *dst_row = (uint16_t *)tmp;

            for (uint32_t x = 0; x < width; x++) {
                dst_row[x] = __builtin_bswap16(src_row[x]);
            }

            [tex replaceRegion:MTLRegionMake2D(0, y, width, 1)
                   mipmapLevel:0
                     withBytes:tmp
                   bytesPerRow:row_bytes];
        }
        g_free(tmp);
    }

    return tex;
}

/* ========================================================================
 * Parse R200 vertex format and extract vertices from raw data
 * ======================================================================== */
static uint32_t metal_calc_vertex_stride(const PPCMacGPU3DState *state)
{
    uint32_t stride = 0;
    uint32_t fmt0 = state->se_vtx_fmt_0;
    uint32_t fmt1 = state->se_vtx_fmt_1;

    /* Position: XY always present */
    stride += 8;  /* float2 = 8 bytes */

    /* Z component */
    if (fmt0 & (1 << 1)) stride += 4;
    /* W0 component */
    if (fmt0 & (1 << 2)) stride += 4;

    /* Color 0: check bits [10:8] for component count */
    uint32_t color0_cnt = (fmt0 >> 8) & 0x7;
    if (color0_cnt > 0) {
        if (fmt0 & (1 << 4)) {
            /* RGBA8 packed format */
            stride += 4;
        } else {
            /* Float components */
            stride += color0_cnt * 4;
        }
    }

    /* Color 1 */
    uint32_t color1_cnt = (fmt0 >> 11) & 0x7;
    if (color1_cnt > 0) {
        if (fmt0 & (1 << 5)) {
            stride += 4;
        } else {
            stride += color1_cnt * 4;
        }
    }

    /* Texture coordinates: check each tex unit in fmt1 */
    for (int t = 0; t < 6; t++) {
        uint32_t tc_cnt = (fmt1 >> (t * 3)) & 0x7;
        if (tc_cnt > 0) {
            stride += (tc_cnt + 1) * 4; /* S=4, ST=8, STR=12, STRQ=16 */
        }
    }

    /* Point size */
    if (fmt0 & (1 << 20)) stride += 4;

    return stride;
}

static void metal_extract_vertices(const PPCMacGPU3DState *state,
                                    const uint32_t *raw_data,
                                    uint32_t num_vertices,
                                    uint32_t raw_stride_bytes,
                                    MetalVertex *out_verts,
                                    bool tex0_enabled)
{
    uint32_t fmt0 = state->se_vtx_fmt_0;
    uint32_t fmt1 = state->se_vtx_fmt_1;
    uint32_t stride_dw = raw_stride_bytes / 4;

    /*
     * The Apple QE kext uses AOS (Array of Structures) descriptors to define
     * the vertex layout, which may not match what VTX_FMT_0/1 registers say.
     * Specifically, the kext writes vtx_fmt0=0x1803 (XY+Z, no W0, no color0,
     * color1_cnt=3) but the actual data has [X,Y,Z,W, R,G,B,A, S,T,R,Q]
     * = 12 floats = 48 bytes.
     *
     * Detect this AOS layout mismatch: if the calculated stride from VTX_FMT
     * doesn't match the actual stride, use the stride to infer the format.
     */
    uint32_t calc_stride = metal_calc_vertex_stride(state);

    /*
     * The Apple QE kext uses AOS (Array of Structures) descriptors that
     * define vertex layout independently of VTX_FMT registers.  Detect
     * layout mismatches by comparing actual stride to calculated stride
     * and use the correct fixed layout.
     *
     * Known AOS layouts:
     *   stride=48: XYZW(16) + RGBA_float(16) + STRQ(16) — 12 floats
     *   stride=32: XYZW(16) + packed_RGBA(4) + ST(8) + pad(4) — 8 dwords
     */
    bool use_fixed_layout = false;
    bool use_fixed_layout_32 = false;
    if (raw_stride_bytes == 48 && calc_stride != 48) {
        use_fixed_layout = true;
        static int fix_log = 0;
        if (fix_log < 3) {
            qemu_log("[VTX_FIX] Using fixed XYZW+RGBA+STRQ layout "
                     "(stride=48 calc=%u fmt0=0x%x fmt1=0x%x)\n",
                     calc_stride, fmt0, fmt1);
            fix_log++;
        }
    } else if (raw_stride_bytes == 32 && calc_stride != 32) {
        use_fixed_layout_32 = true;
        static int fix32_log = 0;
        if (fix32_log < 3) {
            qemu_log("[VTX_FIX32] Using fixed XYZW+packed_RGBA+ST layout "
                     "(stride=32 calc=%u fmt0=0x%x fmt1=0x%x)\n",
                     calc_stride, fmt0, fmt1);
            fix32_log++;
        }
    }

    /* Dump raw vertex data when stride doesn't match calculated to debug
     * AOS layout mismatches */
    if (raw_stride_bytes > 0 && raw_stride_bytes != calc_stride &&
        !use_fixed_layout && num_vertices >= 2) {
        static int raw_dump_count = 0;
        if (raw_dump_count < 10) {
            fprintf(stderr, "[VTX_RAW] stride=%u calc=%u fmt0=0x%x fmt1=0x%x "
                    "nverts=%u\n", raw_stride_bytes, calc_stride, fmt0, fmt1);
            for (uint32_t vi = 0; vi < 2; vi++) {
                const float *s = (const float *)((const uint8_t *)raw_data +
                                                  (uint64_t)vi * raw_stride_bytes);
                uint32_t ndw = raw_stride_bytes / 4;
                fprintf(stderr, "  v%u:", vi);
                for (uint32_t d = 0; d < ndw && d < 12; d++) {
                    uint32_t raw = ((const uint32_t *)s)[d];
                    fprintf(stderr, " [%u]=0x%08x(%.2f)", d, raw, s[d]);
                }
                fprintf(stderr, "\n");
            }
            raw_dump_count++;
        }
    }

    for (uint32_t v = 0; v < num_vertices; v++) {
        const float *src;
        if (raw_stride_bytes > 0) {
            src = (const float *)((const uint8_t *)raw_data +
                                  (uint64_t)v * raw_stride_bytes);
        } else {
            src = (const float *)((const uint8_t *)raw_data +
                                  (uint64_t)v * calc_stride);
        }

        MetalVertex *dst = &out_verts[v];

        if (use_fixed_layout) {
            /*
             * Apple kext AOS layout: 12 floats per vertex
             * [0]  X       [4]  R       [8]  S
             * [1]  Y       [5]  G       [9]  T
             * [2]  Z       [6]  B       [10] R(tc)
             * [3]  W       [7]  A       [11] Q
             */
            dst->position[0] = src[0];    /* X */
            dst->position[1] = src[1];    /* Y */
            /* Z=src[2], W=src[3] — not needed for 2D compositing */
            dst->color[0] = src[4];       /* R */
            dst->color[1] = src[5];       /* G */
            dst->color[2] = src[6];       /* B */
            dst->color[3] = src[7];       /* A */
            dst->texcoord[0] = src[8];    /* S */
            dst->texcoord[1] = src[9];    /* T */
            /* R=src[10], Q=src[11] — not needed */
        } else if (use_fixed_layout_32) {
            /*
             * Apple kext AOS layout: 8 dwords per vertex (stride=32)
             * [0]  X       [4]  unused/padding (always 0 for textured draws)
             * [1]  Y       [5]  S (texcoord)
             * [2]  Z       [6]  T (texcoord)
             * [3]  W       [7]  padding / Q
             *
             * VTX_FMT_0=0x1903 claims Color0(1 float) + Color1(3 floats)
             * but the AOS descriptor defines the actual layout.
             *
             * When TEX0 is ENABLED: [4] is padding (always 0), [5-6] are
             * texcoords.  Default vertex color to white so shader produces
             * tex * white = tex colors.  Texture alpha drives visibility.
             *
             * When TEX0 is DISABLED: fragment color = vertex color only.
             * Read [4] as packed RGBA (VTX_FMT_0 Color0 1-component).
             * For compositor "no-op" draws, [4]=0x00000000 → (0,0,0,0)
             * → draw is invisible (correct: preserves destination).
             */
            dst->position[0] = src[0];    /* X */
            dst->position[1] = src[1];    /* Y */
            /* Z=src[2], W=src[3] — not needed for 2D compositing */
            if (tex0_enabled) {
                dst->color[0] = 1.0f;         /* R = white (default) */
                dst->color[1] = 1.0f;         /* G = white (default) */
                dst->color[2] = 1.0f;         /* B = white (default) */
                dst->color[3] = 1.0f;         /* A = opaque (default) */
            } else {
                /* TEX0 disabled: read packed RGBA from [4] */
                uint32_t rgba = src[4];
                dst->color[0] = ((rgba >> 16) & 0xFF) / 255.0f;  /* R */
                dst->color[1] = ((rgba >>  8) & 0xFF) / 255.0f;  /* G */
                dst->color[2] = ((rgba >>  0) & 0xFF) / 255.0f;  /* B */
                dst->color[3] = ((rgba >> 24) & 0xFF) / 255.0f;  /* A */
            }
            dst->texcoord[0] = src[5];    /* S */
            dst->texcoord[1] = src[6];    /* T */
        } else {
            /* Standard VTX_FMT parsing */
            uint32_t offset = 0;

            /* Position XY (always present) */
            dst->position[0] = src[offset++];
            dst->position[1] = src[offset++];

            /* Skip Z */
            if (fmt0 & (1 << 1)) offset++;
            /* Skip W0 */
            if (fmt0 & (1 << 2)) offset++;

            /* Color 0 */
            uint32_t color0_cnt = (fmt0 >> 8) & 0x7;
            if (color0_cnt > 0 && (fmt0 & (1 << 4))) {
                /* RGBA8 packed: decode uint32 → float4 */
                uint32_t rgba = *(const uint32_t *)&src[offset];
                dst->color[0] = ((rgba >> 16) & 0xFF) / 255.0f;  /* R */
                dst->color[1] = ((rgba >> 8) & 0xFF) / 255.0f;   /* G */
                dst->color[2] = (rgba & 0xFF) / 255.0f;          /* B */
                dst->color[3] = ((rgba >> 24) & 0xFF) / 255.0f;  /* A */
                offset++;
            } else if (color0_cnt > 0) {
                dst->color[0] = (color0_cnt > 0) ? src[offset++] : 1.0f;
                dst->color[1] = (color0_cnt > 1) ? src[offset++] : 1.0f;
                dst->color[2] = (color0_cnt > 2) ? src[offset++] : 1.0f;
                dst->color[3] = (color0_cnt > 3) ? src[offset++] : 1.0f;
            } else {
                dst->color[0] = dst->color[1] = dst->color[2] = dst->color[3] = 1.0f;
            }

            /* Skip color 1 */
            uint32_t color1_cnt = (fmt0 >> 11) & 0x7;
            if (color1_cnt > 0) {
                if (fmt0 & (1 << 5)) {
                    offset++;
                } else {
                    offset += color1_cnt;
                }
            }

            /* Texture coordinates 0 */
            uint32_t tc0_cnt = fmt1 & 0x7;
            if (tc0_cnt > 0) {
                dst->texcoord[0] = src[offset++];
                dst->texcoord[1] = (tc0_cnt >= 1) ? src[offset++] : 0.0f;
                /* Skip R, Q if present */
                for (uint32_t skip = 2; skip <= tc0_cnt; skip++) offset++;
            } else {
                dst->texcoord[0] = dst->texcoord[1] = 0.0f;
            }
        }
    }
}

static bool rqueue_start(PPCMacGPUMetalState *st);
static void rqueue_stop(void);

/* ========================================================================
 * Renderer interface implementation
 * ======================================================================== */

/*
 * The render queue (implemented further down, with the R200 batch state
 * it feeds): a lock-free ring of decoded draws, encoded by a thread that
 * holds no QEMU lock.  g_render_lock serialises that thread against the
 * renderer's other entry points, which the device calls under the BQL.
 * Lock order is always BQL -> g_render_lock; the render thread takes
 * only g_render_lock, so no cycle can form.
 */
static pthread_mutex_t g_render_lock = PTHREAD_MUTEX_INITIALIZER;
static bool rqueue_wanted(void);
static bool rqueue_enabled(void);
static void rqueue_drain(void);

/*
 * A persistent Metal binary archive of compiled pipeline states, in
 * ~/Library/Caches: PSO creation on a later run then skips the GPU-specific
 * compile step (the MSL source compile itself still runs once per program).
 * Stale or missing entries are harmless misses.
 */
static id<MTLBinaryArchive> g_bin_archive;
static NSString *g_bin_archive_path;
static bool g_bin_archive_dirty;
/* MTLBinaryArchive is not thread-safe: adds come from whichever thread
 * compiles a pipeline (setup can run on a draw caller while the render
 * thread encodes), and serialization must not overlap them. */
static pthread_mutex_t g_bin_lock = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBinaryArchive> bin_archive(id<MTLDevice> dev)
{
    static int tried;

    if (qatomic_read(&tried)) {
        return g_bin_archive;
    }
    pthread_mutex_lock(&g_bin_lock);
    if (tried) {
        pthread_mutex_unlock(&g_bin_lock);
        return g_bin_archive;
    }
    if (@available(macOS 11.0, *)) {
        @autoreleasepool {
            NSFileManager *fm = [NSFileManager defaultManager];
            NSString *dir = [NSSearchPathForDirectoriesInDomains(
                                NSCachesDirectory, NSUserDomainMask, YES).firstObject
                             stringByAppendingPathComponent:@"ppcosxkvm"];
            NSOperatingSystemVersion osv =
                [[NSProcessInfo processInfo] operatingSystemVersion];
            NSString *name = [NSString stringWithFormat:@"metal-pso-%ld.%ld-%@.bin",
                              (long)osv.majorVersion, (long)osv.minorVersion,
                              [dev.name stringByReplacingOccurrencesOfString:@"/"
                                                                  withString:@"-"]];
            [fm createDirectoryAtPath:dir
          withIntermediateDirectories:YES attributes:nil error:NULL];
            g_bin_archive_path = [[dir stringByAppendingPathComponent:name] copy];
            MTLBinaryArchiveDescriptor *ad = [[MTLBinaryArchiveDescriptor alloc] init];
            NSError *err = nil;
            if ([fm fileExistsAtPath:g_bin_archive_path]) {
                ad.url = [NSURL fileURLWithPath:g_bin_archive_path];
            }
            g_bin_archive = [dev newBinaryArchiveWithDescriptor:ad error:&err];
            if (!g_bin_archive && ad.url) {
                ad.url = nil;       /* a corrupt cache file: start a new one */
                g_bin_archive = [dev newBinaryArchiveWithDescriptor:ad error:&err];
            }
            [ad release];
            if (!g_bin_archive) {
                qemu_log("ppc-mac-gpu-metal: binary archive unavailable: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "?");
            }
        }
    }
    qatomic_set(&tried, 1);     /* publishes g_bin_archive to the fast path */
    pthread_mutex_unlock(&g_bin_lock);
    return g_bin_archive;
}

/* Only at teardown: serialization must not overlap function adds, and
 * those can come from any thread that compiles a pipeline. */
static void bin_archive_serialize(void)
{
    if (@available(macOS 11.0, *)) {
    } else {
        return;
    }
    pthread_mutex_lock(&g_bin_lock);
    if (g_bin_archive && g_bin_archive_dirty && g_bin_archive_path) {
        [g_bin_archive serializeToURL:[NSURL fileURLWithPath:g_bin_archive_path]
                                error:NULL];
        g_bin_archive_dirty = false;
    }
    pthread_mutex_unlock(&g_bin_lock);
}

/* newRenderPipelineStateWithDescriptor through the archive, and into it. */
static id<MTLRenderPipelineState> bin_pso(id<MTLDevice> dev,
                                          MTLRenderPipelineDescriptor *pd,
                                          NSError **err)
{
    id<MTLBinaryArchive> a = bin_archive(dev);
    id<MTLRenderPipelineState> p;

    if (a) {
        @autoreleasepool {
            pd.binaryArchives = @[ a ];
        }
    }
    p = [dev newRenderPipelineStateWithDescriptor:pd error:err];
    if (a && p) {
        pthread_mutex_lock(&g_bin_lock);
        if ([a addRenderPipelineFunctionsWithDescriptor:pd error:NULL]) {
            g_bin_archive_dirty = true;
        }
        pthread_mutex_unlock(&g_bin_lock);
    }
    return p;
}

static void *metal_init(uint8_t *vram_ptr, uint64_t vram_size)
{
    PPCMacGPUMetalState *st = g_new0(PPCMacGPUMetalState, 1);
    st->vram_ptr = vram_ptr;
    st->vram_size = vram_size;

    @autoreleasepool {
        /*
         * If zero-copy VRAM was pre-allocated, reuse its MTLDevice and
         * MTLBuffer.  Otherwise create a new device (legacy path).
         */
        if (g_metal_vram_alloc &&
            [g_metal_vram_alloc->vramBuffer contents] == vram_ptr) {
            /* Zero-copy path: reuse device and buffer from early alloc */
            st->device = g_metal_vram_alloc->device;
            st->vramBuffer = g_metal_vram_alloc->vramBuffer;
            qemu_log("ppc-mac-gpu-metal: using zero-copy unified VRAM "
                     "(%llu MB MTLBuffer)\n",
                     (unsigned long long)(vram_size / (1024*1024)));
        } else {
            /* Legacy path: create new device (VRAM is normal QEMU RAM) */
            st->device = MTLCreateSystemDefaultDevice();
            st->vramBuffer = nil;
        }

        if (!st->device) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: no Metal device available\n");
            g_free(st);
            return NULL;
        }

        /* Create command queue */
        st->commandQueue = [st->device newCommandQueue];
        if (!st->commandQueue) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: failed to create command queue\n");
            g_free(st);
            return NULL;
        }

        /* Compile shaders */
        NSError *error = nil;
        id<MTLLibrary> library = [st->device newLibraryWithSource:kShaderSource
                                                          options:nil
                                                            error:&error];
        if (!library) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: shader compilation failed: %s\n",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
            g_free(st);
            return NULL;
        }

        id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
        id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_textured"];

        if (!vertexFunc || !fragmentFunc) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: shader function lookup failed\n");
            g_free(st);
            return NULL;
        }

        /* Create vertex descriptor */
        MTLVertexDescriptor *vtxDesc = [[MTLVertexDescriptor alloc] init];
        /* position: float2 at offset 0 */
        vtxDesc.attributes[0].format = MTLVertexFormatFloat2;
        vtxDesc.attributes[0].offset = offsetof(MetalVertex, position);
        vtxDesc.attributes[0].bufferIndex = 0;
        /* texcoord: float2 at offset 8 */
        vtxDesc.attributes[1].format = MTLVertexFormatFloat2;
        vtxDesc.attributes[1].offset = offsetof(MetalVertex, texcoord);
        vtxDesc.attributes[1].bufferIndex = 0;
        /* color: float4 at offset 16 */
        vtxDesc.attributes[2].format = MTLVertexFormatFloat4;
        vtxDesc.attributes[2].offset = offsetof(MetalVertex, color);
        vtxDesc.attributes[2].bufferIndex = 0;
        /* Layout */
        vtxDesc.layouts[0].stride = sizeof(MetalVertex);
        vtxDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

        /* Create pipeline state (with alpha blending enabled by default) */
        MTLRenderPipelineDescriptor *pipeDesc =
            [[MTLRenderPipelineDescriptor alloc] init];
        pipeDesc.vertexFunction = vertexFunc;
        pipeDesc.fragmentFunction = fragmentFunc;
        pipeDesc.vertexDescriptor = vtxDesc;
        pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        /* Enable blending — will be overridden per-draw if needed */
        pipeDesc.colorAttachments[0].blendingEnabled = YES;
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        st->pipelineState = [st->device newRenderPipelineStateWithDescriptor:pipeDesc
                                                                       error:&error];
        if (!st->pipelineState) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: pipeline creation failed: %s\n",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
            g_free(st);
            return NULL;
        }

        /* Create NOP blend pipeline state (src=Zero, dst=One → destination passthrough) */
        pipeDesc.colorAttachments[0].blendingEnabled = YES;
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorZero;
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
        pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;

        st->pipelineStateNopBlend = [st->device newRenderPipelineStateWithDescriptor:pipeDesc
                                                                               error:&error];
        if (!st->pipelineStateNopBlend) {
            qemu_log_mask(LOG_UNIMP,
                          "ppc-mac-gpu-metal: NOP blend pipeline creation failed: %s\n",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
            /* Non-fatal: fall back to main pipeline */
        }

        /* Save library + vertex descriptor for creating blend pipelines on demand */
        st->shaderLibrary = library;
        st->vertexDescriptor = vtxDesc;
        st->blend_cache_count = 0;

        /* Create samplers */
        MTLSamplerDescriptor *sampDesc = [[MTLSamplerDescriptor alloc] init];
        sampDesc.minFilter = MTLSamplerMinMagFilterNearest;
        sampDesc.magFilter = MTLSamplerMinMagFilterNearest;
        sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        st->samplerNearest = [st->device newSamplerStateWithDescriptor:sampDesc];

        sampDesc.minFilter = MTLSamplerMinMagFilterLinear;
        sampDesc.magFilter = MTLSamplerMinMagFilterLinear;
        st->samplerBilinear = [st->device newSamplerStateWithDescriptor:sampDesc];

        /* Create 1x1 opaque white dummy texture for non-textured draws.
         * The shader always does tex.sample() * vertex_color, so binding
         * a white texture makes the multiplication an identity when the
         * R200 hardware has TEX_0 disabled (vertex-color-only draws). */
        {
            MTLTextureDescriptor *dtd = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                width:1 height:1 mipmapped:NO];
            dtd.usage = MTLTextureUsageShaderRead;
            st->whiteDummyTexture = [st->device newTextureWithDescriptor:dtd];
            uint32_t white = 0xFFFFFFFF; /* BGRA all 255 */
            [st->whiteDummyTexture replaceRegion:MTLRegionMake2D(0,0,1,1)
                                     mipmapLevel:0
                                       withBytes:&white
                                     bytesPerRow:4];
        }

        st->initialized = true;
        frame_tracker_init(&st->frame_tracker);
        g_frame_tracker = &st->frame_tracker;
        qemu_log("ppc-mac-gpu-metal: Metal backend initialized (%s)\n",
                 [[st->device name] UTF8String]);
    }

    rqueue_start(st);

    return st;
}

static void metal_fini(void *opaque)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st) return;

    rqueue_stop();
    bin_archive_serialize();

    /* Free shadow RT buffers */
    for (int i = 0; i < SHADOW_RT_MAX; i++) {
        if (st->shadow_rts[i].pixels) {
            g_free(st->shadow_rts[i].pixels);
            st->shadow_rts[i].pixels = NULL;
        }
        st->shadow_rts[i].valid = false;
    }

    @autoreleasepool {
        st->renderTarget = nil;
        st->outputBuffer = nil;
        st->pipelineState = nil;
        st->pipelineStateNopBlend = nil;
        st->samplerNearest = nil;
        st->samplerBilinear = nil;
        st->whiteDummyTexture = nil;
        st->commandQueue = nil;
        st->device = nil;
    }

    g_free(st);
}

/* ========================================================================
 * Shadow RT helpers
 *
 * Find or allocate a shadow RT entry for a given (offset, pitch).
 * The shadow RT stores a BE-format pixel buffer at stride = pitch_pixels.
 * ======================================================================== */

static int shadow_rt_find(PPCMacGPUMetalState *st,
                           uint32_t offset, uint32_t pitch_pixels)
{
    for (int i = 0; i < SHADOW_RT_MAX; i++) {
        if (st->shadow_rts[i].valid &&
            st->shadow_rts[i].offset == offset &&
            st->shadow_rts[i].pitch_pixels == pitch_pixels) {
            return i;
        }
    }
    return -1;
}

/*
 * Find by offset only — returns the entry with the largest height
 * (most likely the final composited result).  Used by metal_blit_2d
 * when the BLT source pitch may not exactly match the 3D render pitch.
 */
static int shadow_rt_find_by_offset(PPCMacGPUMetalState *st, uint32_t offset)
{
    int best = -1;
    uint32_t best_h = 0;
    for (int i = 0; i < SHADOW_RT_MAX; i++) {
        if (st->shadow_rts[i].valid &&
            st->shadow_rts[i].offset == offset &&
            st->shadow_rts[i].height > best_h) {
            best = i;
            best_h = st->shadow_rts[i].height;
        }
    }
    return best;
}

/*
 * Ensure a shadow RT entry exists for (offset, pitch), with at least
 * the specified width×height.  Allocates/resizes buffer as needed.
 * Returns slot index, or -1 if cache is full.
 */
static int shadow_rt_ensure(PPCMacGPUMetalState *st,
                              uint32_t offset, uint32_t pitch_pixels,
                              uint32_t width, uint32_t height)
{
    int idx = shadow_rt_find(st, offset, pitch_pixels);
    if (idx >= 0) {
        uint32_t old_w = st->shadow_rts[idx].width;
        uint32_t old_h = st->shadow_rts[idx].height;
        uint32_t new_w = (width > old_w) ? width : old_w;
        uint32_t new_h = (height > old_h) ? height : old_h;

        if (new_w > old_w || new_h > old_h) {
            /* SRT needs to grow */
            uint32_t need = (uint32_t)((uint64_t)new_w * new_h);

            static int srt_grow_log = 0;
            if (srt_grow_log < 60) {
                fprintf(stderr,
                    "[SRT_GROW] "
                    "rt_off=0x%x rt_pitch=%u "
                    "old_size=(%u,%u) new_size=(%u,%u) "
                    "reason=draw_exceeded_bounds "
                    "note=slot=%d\n",
                    offset, pitch_pixels,
                    old_w, old_h, new_w, new_h, idx);
                srt_grow_log++;
            }

            if (new_w > old_w) {
                /* Width changed — must re-layout rows.
                 * Allocate new buffer and copy row by row. */
                uint32_t *new_pixels =
                    (uint32_t *)g_malloc0((size_t)need * 4);
                if (st->shadow_rts[idx].pixels) {
                    uint32_t copy_h = old_h < new_h ? old_h : new_h;
                    uint32_t copy_w = old_w < new_w ? old_w : new_w;
                    for (uint32_t y = 0; y < copy_h; y++) {
                        memcpy(new_pixels + (uint64_t)y * new_w,
                               st->shadow_rts[idx].pixels +
                                   (uint64_t)y * old_w,
                               (size_t)copy_w * 4);
                    }
                    g_free(st->shadow_rts[idx].pixels);
                }
                st->shadow_rts[idx].pixels = new_pixels;
            } else {
                /* Height-only growth — simple realloc + zero new rows */
                if (need > st->shadow_rts[idx].alloc_count) {
                    st->shadow_rts[idx].pixels =
                        (uint32_t *)g_realloc(st->shadow_rts[idx].pixels,
                                              (size_t)need * 4);
                    uint32_t old_count = old_w * old_h;
                    memset(st->shadow_rts[idx].pixels + old_count, 0,
                           (size_t)(need - old_count) * 4);
                }
            }
            st->shadow_rts[idx].width = new_w;
            st->shadow_rts[idx].height = new_h;
            st->shadow_rts[idx].alloc_count = need;
        }
        return idx;
    }

    /* Find a free slot (or evict oldest / LRU — for now just find free) */
    int free_slot = -1;
    for (int i = 0; i < SHADOW_RT_MAX; i++) {
        if (!st->shadow_rts[i].valid) {
            free_slot = i;
            break;
        }
    }
    if (free_slot < 0) {
        /* Cache full — evict slot 0 (simple policy) */
        if (st->shadow_rts[0].pixels) {
            g_free(st->shadow_rts[0].pixels);
        }
        free_slot = 0;
    }

    uint32_t count = (uint32_t)((uint64_t)width * height);
    st->shadow_rts[free_slot].valid = true;
    st->shadow_rts[free_slot].offset = offset;
    st->shadow_rts[free_slot].pitch_pixels = pitch_pixels;

    /* Check if this offset already has a pitch conflict recorded.
     * If so, mark the new entry as conflicted too. */
    st->shadow_rts[free_slot].has_conflict = false;
    st->shadow_rts[free_slot].consumed = false;
    st->shadow_rts[free_slot].stale = false;
    st->shadow_rts[free_slot].origin_valid = false;
    st->shadow_rts[free_slot].content_origin_x = 0;
    st->shadow_rts[free_slot].content_origin_y = 0;
    for (int ci = 0; ci < SHADOW_RT_MAX; ci++) {
        if (ci != free_slot && st->shadow_rts[ci].valid &&
            st->shadow_rts[ci].offset == offset &&
            st->shadow_rts[ci].has_conflict) {
            st->shadow_rts[free_slot].has_conflict = true;
            break;
        }
    }
    st->shadow_rts[free_slot].width = width;
    st->shadow_rts[free_slot].height = height;
    st->shadow_rts[free_slot].pixels =
        (uint32_t *)g_malloc0((size_t)count * 4);
    st->shadow_rts[free_slot].alloc_count = count;

    /*
     * Shadow RT starts as zero (g_malloc0).  We do NOT pre-fill from VRAM
     * because VRAM may contain garbled data from pitch-conflicting writes
     * (multiple pitches writing to the same linear VRAM offset).
     *
     * The first Metal 3D draw to this (offset, pitch) will load the bbox
     * region from VRAM (which is correct at that point — no pitch conflict
     * has occurred yet for that specific pitch view).  Subsequent draws
     * load from the shadow RT, accumulating correctly.
     */
    return free_slot;
}

/*
 * Save a region from the Metal output buffer into the shadow RT.
 * The Metal output is in LE BGRA; we bswap to BE (matching VRAM/guest format).
 */
static void shadow_rt_save_region(PPCMacGPUMetalState *st,
                                    int slot,
                                    const uint32_t *metal_output,
                                    uint32_t rt_width,
                                    uint32_t rx, uint32_t ry,
                                    uint32_t rw, uint32_t rh)
{
    if (slot < 0 || slot >= SHADOW_RT_MAX) return;
    if (!st->shadow_rts[slot].valid || !st->shadow_rts[slot].pixels) return;

    uint32_t sw = st->shadow_rts[slot].width;
    uint32_t sh = st->shadow_rts[slot].height;
    uint32_t *dst = st->shadow_rts[slot].pixels;
    uint32_t nz_count = 0;
    for (uint32_t y = ry; y < ry + rh && y < sh; y++) {
        const uint32_t *src_row = metal_output + (uint64_t)y * rt_width;
        uint32_t *dst_row = dst + (uint64_t)y * sw;
        for (uint32_t x = rx; x < rx + rw && x < sw; x++) {
            uint32_t val = __builtin_bswap32(src_row[x]);
            dst_row[x] = val;
            if (val != 0) nz_count++;
        }
    }
}

static void *metal_scanout_locked(void *opaque, const PPCMacGPUScanout *desc);

static void *metal_scanout(void *opaque, const PPCMacGPUScanout *desc)
{
    void *r;

    pthread_mutex_lock(&g_render_lock);
    r = metal_scanout_locked(opaque, desc);
    pthread_mutex_unlock(&g_render_lock);
    return r;
}

static void *metal_scanout_locked(void *opaque, const PPCMacGPUScanout *desc)
{
    /* Metal backend doesn't handle scanout differently yet.
     * The 2D scanout path in the main device still works for display. */
    (void)opaque;
    (void)desc;
    return NULL;
}

/*
 * Write-through to SRT: when a 2D BLIT writes to VRAM at an offset that
 * has an active SRT, also write the pixels to the SRT.  This ensures
 * the SRT has BOTH 3D-drawn content (wallpaper, shadows, corners) AND
 * 2D-positioned content (window body during drag).
 *
 * Without this, the compositor's 2D BLITs (which reposition window bodies
 * from their backing stores to screen coordinates) are invisible to the
 * PRESENT_BLIT, which reads from the SRT and skips zero pixels.
 *
 * Pixels are in BE format (same as VRAM / guest).
 */
static void metal_srt_write_through_locked(void *opaque, uint8_t *vram_ptr,
                                           uint32_t dst_offset, uint32_t dst_pitch,
                                           uint32_t dst_x, uint32_t dst_y,
                                           uint32_t width, uint32_t height,
                                           uint32_t bpp);

static void metal_srt_write_through(void *opaque, uint8_t *vram_ptr,
                                    uint32_t dst_offset, uint32_t dst_pitch,
                                    uint32_t dst_x, uint32_t dst_y,
                                    uint32_t width, uint32_t height,
                                    uint32_t bpp)
{
    pthread_mutex_lock(&g_render_lock);
    metal_srt_write_through_locked(opaque, vram_ptr, dst_offset, dst_pitch,
                                   dst_x, dst_y, width, height, bpp);
    pthread_mutex_unlock(&g_render_lock);
}

static void metal_srt_write_through_locked(void *opaque, uint8_t *vram_ptr,
                                           uint32_t dst_offset, uint32_t dst_pitch,
                                           uint32_t dst_x, uint32_t dst_y,
                                           uint32_t width, uint32_t height,
                                           uint32_t bpp)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st || bpp != 32) return;

    uint32_t bpp_bytes = bpp / 8;
    uint32_t dst_pitch_pixels = dst_pitch / bpp_bytes;

    /* Find matching SRT by (offset, pitch) */
    int slot = shadow_rt_find(st, dst_offset, dst_pitch_pixels);
    if (slot < 0) return;

    uint32_t sw = st->shadow_rts[slot].width;
    uint32_t sh = st->shadow_rts[slot].height;
    uint32_t *srt_pixels = st->shadow_rts[slot].pixels;
    if (!srt_pixels) return;

    /* Copy pixels from VRAM to SRT at dst coordinates.
     * VRAM and SRT are both in BE format. */
    uint32_t written = 0;
    for (uint32_t row = 0; row < height; row++) {
        uint32_t y = dst_y + row;
        if (y >= sh) break;

        uint32_t *srt_row = srt_pixels + (uint64_t)y * sw;
        uint64_t vram_row_addr = (uint64_t)dst_offset +
                                 (uint64_t)(dst_y + row) * dst_pitch +
                                 (uint64_t)dst_x * bpp_bytes;

        for (uint32_t col = 0; col < width; col++) {
            uint32_t x = dst_x + col;
            if (x >= sw) break;

            uint64_t vram_addr = vram_row_addr + (uint64_t)col * bpp_bytes;
            if (vram_addr + 4 > st->vram_size) continue;

            uint32_t pixel = *(uint32_t *)(vram_ptr + vram_addr);
            srt_row[x] = pixel;
            written++;
        }
    }

    static int wt_log = 0;
    if (wt_log < 60) {
        fprintf(stderr, "[SRT_WRITE_THROUGH] off=0x%x pitch=%u "
                "dst=(%u,%u) %ux%u srt=%ux%u written=%u\n",
                dst_offset, dst_pitch_pixels,
                dst_x, dst_y, width, height, sw, sh, written);
        wt_log++;
    }
}

/*
 * drag_tracker_update — detect PRESENT_BLIT origin shifts (window drag).
 *
 * Called for each large PRESENT_BLIT.  Tracks previous vs current origin
 * for each compositor source offset.  When the origin shifts, sets
 * copy_pending with the delta so the body copy runs before the SRT pixel
 * loop writes shadows/corners.
 *
 * The compositor sends multiple PRESENT_BLITs per frame with varying
 * sizes (shadow strips, corners, body region).  We track by source offset
 * and use the origin (dst_xy - src_xy) which is consistent across all
 * sub-BLITs for the same window.  We accumulate the max blit extent
 * to determine the full window rectangle.
 */
static void drag_tracker_update(PPCMacGPUMetalState *st,
                                uint32_t src_offset,
                                uint32_t origin_x, uint32_t origin_y,
                                uint32_t blit_w, uint32_t blit_h)
{
    int slot = -1;
    int free_slot = -1;
    uint32_t oldest_gen = UINT32_MAX;
    int oldest_slot = -1;

    for (int i = 0; i < DRAG_TRACKER_MAX; i++) {
        if (st->drag_trackers[i].active &&
            st->drag_trackers[i].src_offset == src_offset) {
            slot = i;
            break;
        }
        if (!st->drag_trackers[i].active && free_slot < 0) {
            free_slot = i;
        }
        if (st->drag_trackers[i].cur_frame_gen < oldest_gen) {
            oldest_gen = st->drag_trackers[i].cur_frame_gen;
            oldest_slot = i;
        }
    }

    if (slot < 0) {
        slot = (free_slot >= 0) ? free_slot : oldest_slot;
        st->drag_trackers[slot].active = true;
        st->drag_trackers[slot].src_offset = src_offset;
        st->drag_trackers[slot].prev_frame_gen = 0;
        st->drag_trackers[slot].cur_origin_x = origin_x;
        st->drag_trackers[slot].cur_origin_y = origin_y;
        st->drag_trackers[slot].cur_frame_gen = st->rt_frame_gen;
        st->drag_trackers[slot].blit_w = blit_w;
        st->drag_trackers[slot].blit_h = blit_h;
        st->drag_trackers[slot].copy_pending = false;
        return;
    }

    typeof(st->drag_trackers[0]) *t = &st->drag_trackers[slot];

    /*
     * Same origin as current — just update max dimensions.
     * Multiple BLITs per frame have the same origin but different sizes.
     */
    if (t->cur_origin_x == origin_x && t->cur_origin_y == origin_y) {
        if (blit_w > t->blit_w) t->blit_w = blit_w;
        if (blit_h > t->blit_h) t->blit_h = blit_h;
        return;
    }

    /*
     * Origin changed — this is a drag movement.
     * Rotate: current becomes previous, new origin becomes current.
     */
    t->prev_origin_x = t->cur_origin_x;
    t->prev_origin_y = t->cur_origin_y;
    t->prev_frame_gen = t->cur_frame_gen;

    uint32_t prev_blit_w = t->blit_w;
    uint32_t prev_blit_h = t->blit_h;

    t->cur_origin_x = origin_x;
    t->cur_origin_y = origin_y;
    t->cur_frame_gen = st->rt_frame_gen;
    t->blit_w = blit_w;
    t->blit_h = blit_h;

    /* Compute delta */
    if (t->prev_frame_gen > 0 && !t->copy_pending) {
        int32_t dx = (int32_t)origin_x - (int32_t)t->prev_origin_x;
        int32_t dy = (int32_t)origin_y - (int32_t)t->prev_origin_y;

        if (dx != 0 || dy != 0) {
            /* Guard: skip if delta is too large (window jump, not drag) */
            if ((uint32_t)abs(dx) < prev_blit_w / 2 &&
                (uint32_t)abs(dy) < prev_blit_h / 2) {
                t->copy_pending = true;
                frame_tracker_note_drag(&st->frame_tracker, 4);
                t->dx = dx;
                t->dy = dy;
                /* Use previous frame's dimensions for the body copy
                 * since those define where the body IS right now */
                t->blit_w = prev_blit_w;
                t->blit_h = prev_blit_h;
            }
        }
    }
}

static bool drag_surface_is_tracked(PPCMacGPUMetalState *st, uint32_t src_offset)
{
    if (!st) {
        return false;
    }
    for (int i = 0; i < DRAG_TRACKER_MAX; i++) {
        if (st->drag_trackers[i].active &&
            st->drag_trackers[i].src_offset == src_offset) {
            return true;
        }
    }
    return false;
}

/*
 * drag_body_copy — direct intra-framebuffer copy of window body pixels.
 *
 * Copies body pixels from old position to new position in VRAM using
 * memmove (handles overlapping regions correctly).  Row iteration
 * direction depends on the drag delta to avoid self-clobbering:
 *   dy > 0 (moving down): iterate bottom-to-top
 *   dy <= 0 (moving up):  iterate top-to-bottom
 *
 * Cleanup of the exposed old position is left to later compositor
 * presents; do not repaint it here with a sampled wallpaper color.
 */
static void drag_body_copy(PPCMacGPUMetalState *st, uint8_t *vram,
                           uint32_t fb_pitch,
                           uint32_t prev_x, uint32_t prev_y,
                           uint32_t blit_w, uint32_t blit_h,
                           int32_t dx, int32_t dy,
                           uint32_t screen_w, uint32_t screen_h)
{
    /*
     * Copy only the client/body area. The Finder drag artifacts strongly
     * suggest a symmetric inset is too coarse: it drags pieces of the
     * titlebar/frame/shadow along with the body and then protects too much
     * of the subsequent QE present. Keep the interior move asymmetric so
     * the QE-owned frame stays in the fresh Metal->VRAM pass.
     */
    int32_t body_x = (int32_t)prev_x + DRAG_INSET_LEFT;
    int32_t body_y = (int32_t)prev_y + DRAG_INSET_TOP;
    int32_t body_w = (int32_t)blit_w - DRAG_INSET_LEFT - DRAG_INSET_RIGHT;
    int32_t body_h = (int32_t)blit_h - DRAG_INSET_TOP - DRAG_INSET_BOTTOM;

    if (body_w <= 0 || body_h <= 0) return;

    /* Clip source (old position) to screen */
    if (body_x < 0) { body_w += body_x; body_x = 0; }
    if (body_y < 0) { body_h += body_y; body_y = 0; }
    if (body_x + body_w > (int32_t)screen_w) body_w = screen_w - body_x;
    if (body_y + body_h > (int32_t)screen_h) body_h = screen_h - body_y;
    if (body_w <= 0 || body_h <= 0) return;

    /* Clip dest (new position) to screen */
    int32_t dst_x = body_x + dx;
    int32_t dst_y = body_y + dy;
    if (dst_x < 0) { body_w += dst_x; body_x -= dst_x; dst_x = 0; }
    if (dst_y < 0) { body_h += dst_y; body_y -= dst_y; dst_y = 0; }
    if (dst_x + body_w > (int32_t)screen_w) body_w = screen_w - dst_x;
    if (dst_y + body_h > (int32_t)screen_h) body_h = screen_h - dst_y;
    if (body_w <= 0 || body_h <= 0) return;

    /* VRAM bounds check for both src and dst */
    uint64_t src_max = (uint64_t)(body_y + body_h - 1) * fb_pitch +
                       (uint64_t)(body_x + body_w) * 4;
    uint64_t dst_max = (uint64_t)(dst_y + body_h - 1) * fb_pitch +
                       (uint64_t)(dst_x + body_w) * 4;
    if (src_max > st->vram_size || dst_max > st->vram_size) return;

    /* Copy body with correct row ordering to handle overlap */
    uint32_t row_bytes = (uint32_t)body_w * 4;
    if (dy > 0) {
        /* Moving down: copy bottom-to-top */
        for (int32_t row = body_h - 1; row >= 0; row--) {
            uint8_t *src = vram + (uint64_t)(body_y + row) * fb_pitch +
                           (uint64_t)body_x * 4;
            uint8_t *dst = vram + (uint64_t)(dst_y + row) * fb_pitch +
                           (uint64_t)dst_x * 4;
            memmove(dst, src, row_bytes);
        }
    } else {
        /* Moving up or horizontally: copy top-to-bottom */
        for (int32_t row = 0; row < body_h; row++) {
            uint8_t *src = vram + (uint64_t)(body_y + row) * fb_pitch +
                           (uint64_t)body_x * 4;
            uint8_t *dst = vram + (uint64_t)(dst_y + row) * fb_pitch +
                           (uint64_t)dst_x * 4;
            memmove(dst, src, row_bytes);
        }
    }

    st->drag_protect_active = true;
    st->drag_protect_x = dst_x;
    st->drag_protect_y = dst_y;
    st->drag_protect_w = body_w;
    st->drag_protect_h = body_h;

    static int drag_log = 0;
    if (drag_log < 200) {
        fprintf(stderr, "[DRAG_BODY_COPY] src=(%d,%d) dst=(%d,%d) "
                "delta=(%d,%d) body=%dx%d\n",
                body_x, body_y, dst_x, dst_y, dx, dy,
                body_w, body_h);
        drag_log++;
    }
}

static uint32_t drag_restore_strip(PPCMacGPUMetalState *st, uint8_t *vram,
                                   uint32_t fb_pitch,
                                   int32_t full_x, int32_t full_y,
                                   int32_t local_x0, int32_t local_y0,
                                   int32_t strip_w, int32_t strip_h,
                                   uint32_t screen_w, uint32_t screen_h,
                                   const uint32_t *srt_pixels,
                                   uint32_t srt_w, uint32_t srt_h,
                                   uint32_t srt_stride,
                                   uint32_t src_x, uint32_t src_y)
{
    uint32_t restored = 0;

    if (!st || !vram || !srt_pixels || strip_w <= 0 || strip_h <= 0) {
        return 0;
    }

    for (int32_t row = 0; row < strip_h; row++) {
        int32_t local_y = local_y0 + row;
        int32_t dst_y = full_y + local_y;
        uint32_t sy = src_y + (uint32_t)local_y;
        if (dst_y < 0 || dst_y >= (int32_t)screen_h || sy >= srt_h) {
            continue;
        }

        for (int32_t col = 0; col < strip_w; col++) {
            int32_t local_x = local_x0 + col;
            int32_t dst_x = full_x + local_x;
            uint32_t sx = src_x + (uint32_t)local_x;
            if (dst_x < 0 || dst_x >= (int32_t)screen_w || sx >= srt_w) {
                continue;
            }

            uint32_t pixel_be = srt_pixels[(uint64_t)sy * srt_stride + sx];
            uint32_t alpha = pixel_be & 0xFF;
            if (alpha < 254) {
                continue;
            }

            uint64_t dst_addr = (uint64_t)dst_y * fb_pitch +
                                (uint64_t)dst_x * 4;
            if (dst_addr + 4 > st->vram_size) {
                continue;
            }
            *(uint32_t *)(vram + dst_addr) = pixel_be;
            restored++;
        }
    }

    return restored;
}

/*
 * Restore the old exposed drag strips from opaque compositor pixels in the SRT.
 *
 * This is a narrower cleanup than the old wallpaper-color fill. We only write
 * fully opaque SRT pixels from the *source* region that is being moved, which
 * gives us real compositor-provided background pixels for the exposed strips
 * without inventing colors or touching semi-transparent shadow data.
 */
static void drag_cleanup_exposed(PPCMacGPUMetalState *st, uint8_t *vram,
                                 uint32_t fb_pitch,
                                 uint32_t prev_x, uint32_t prev_y,
                                 uint32_t blit_w, uint32_t blit_h,
                                 int32_t dx, int32_t dy,
                                 uint32_t screen_w, uint32_t screen_h,
                                 const uint32_t *srt_pixels,
                                 uint32_t srt_w, uint32_t srt_h,
                                 uint32_t srt_stride,
                                 uint32_t src_x, uint32_t src_y)
{
    if (!st || !vram || !srt_pixels || blit_w == 0 || blit_h == 0) {
        return;
    }

    uint32_t restored = 0;
    int32_t full_x = (int32_t)prev_x;
    int32_t full_y = (int32_t)prev_y;
    int32_t full_w = (int32_t)blit_w;
    int32_t full_h = (int32_t)blit_h;

    if (dy != 0) {
        int32_t strip_h = abs(dy);
        if (strip_h > full_h) {
            strip_h = full_h;
        }
        if (dy > 0) {
            restored += drag_restore_strip(st, vram, fb_pitch,
                                           full_x, full_y,
                                           0, 0, full_w, strip_h,
                                           screen_w, screen_h,
                                           srt_pixels, srt_w, srt_h,
                                           srt_stride, src_x, src_y);
        } else {
            restored += drag_restore_strip(st, vram, fb_pitch,
                                           full_x, full_y,
                                           0, full_h - strip_h,
                                           full_w, strip_h,
                                           screen_w, screen_h,
                                           srt_pixels, srt_w, srt_h,
                                           srt_stride, src_x, src_y);
        }
    }

    if (dx != 0) {
        int32_t strip_w = abs(dx);
        if (strip_w > full_w) {
            strip_w = full_w;
        }
        if (dx > 0) {
            restored += drag_restore_strip(st, vram, fb_pitch,
                                           full_x, full_y,
                                           0, 0, strip_w, full_h,
                                           screen_w, screen_h,
                                           srt_pixels, srt_w, srt_h,
                                           srt_stride, src_x, src_y);
        } else {
            restored += drag_restore_strip(st, vram, fb_pitch,
                                           full_x, full_y,
                                           full_w - strip_w, 0,
                                           strip_w, full_h,
                                           screen_w, screen_h,
                                           srt_pixels, srt_w, srt_h,
                                           srt_stride, src_x, src_y);
        }
    }

    static int cleanup_log = 0;
    if (cleanup_log < 120) {
        fprintf(stderr,
                "[DRAG_CLEANUP] old=(%d,%d) size=%dx%d delta=(%d,%d) "
                "src=(%u,%u) restored=%u\n",
                full_x, full_y, full_w, full_h, dx, dy,
                src_x, src_y, restored);
        cleanup_log++;
    }
}

/* flush_drag_paste — no-op now, protection handles everything inline */
static void drag_body_paste(PPCMacGPUMetalState *st, uint8_t *vram)
{
    (void)st; (void)vram;
    /* Body protection in the SRT pixel loop prevents overwrites.
     * No deferred paste needed. */
}

static int metal_blit_2d_locked(void *opaque, uint8_t *vram_ptr,
                                const PPCMacGPUBlit *blit);

static int metal_blit_2d(void *opaque, uint8_t *vram_ptr,
                          const PPCMacGPUBlit *blit)
{
    int r;

    /* Shadow RTs and the drag trackers are shared with the render
     * thread's encodes; serialise (the whole function reads and writes
     * them throughout). */
    pthread_mutex_lock(&g_render_lock);
    r = metal_blit_2d_locked(opaque, vram_ptr, blit);
    pthread_mutex_unlock(&g_render_lock);
    return r;
}

static int metal_blit_2d_locked(void *opaque, uint8_t *vram_ptr,
                                const PPCMacGPUBlit *blit)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st || !blit || blit->rop3 != 0xCC) {
        return -1;  /* Only handle SRC copy blits */
    }

    st->drag_protect_active = false;

    /* (test removed) */

    uint32_t bpp_bytes = blit->bpp / 8;
    if (bpp_bytes != 4) {
        return -1;  /* Only 32bpp */
    }

    /*
     * Look up shadow RT by exact (offset, pitch) match only.
     * The pitch must match because different pitches represent different
     * virtual surfaces at the same VRAM offset (separated by MC tiling on
     * real hardware).  A pitch mismatch means this BLT reads from a
     * different surface than what Metal rendered — fall back to VRAM.
     */
    uint32_t src_pitch_pixels = blit->src_pitch / bpp_bytes;
    int slot = shadow_rt_find(st, blit->src_offset, src_pitch_pixels);
    if (slot < 0) {
        static int srt_miss_log = 0;
        if (srt_miss_log < 60) {
            SurfaceClass sc = classify_surface(blit->src_offset,
                                                src_pitch_pixels,
                                                blit->dst_offset);
            fprintf(stderr, "[SRT_SELECT] src=0x%x pitch=%u slot=-1 "
                    "class=%s dst=0x%x decision=VRAM_FALLBACK\n",
                    blit->src_offset, src_pitch_pixels,
                    surface_class_name(sc), blit->dst_offset);
            srt_miss_log++;
        }
        return -1;  /* No matching shadow RT — fall back to VRAM read */
    }

    /*
     * SRT Authority: Always prefer SRT when a matching (offset, pitch) exists.
     * The has_conflict gate was removed — SRT captures the authoritative Metal
     * 3D output regardless of whether pitch conflicts have been observed.
     */
    static int srt_select_log = 0;
    if (srt_select_log < 60) {
        SurfaceClass sc = classify_surface(blit->src_offset,
                                            src_pitch_pixels,
                                            blit->dst_offset);
        fprintf(stderr, "[SRT_SELECT] src=0x%x pitch=%u slot=%d "
                "has_conflict=%d class=%s dst=0x%x decision=USE_SRT\n",
                blit->src_offset, src_pitch_pixels, slot,
                st->shadow_rts[slot].has_conflict,
                surface_class_name(sc), blit->dst_offset);
        srt_select_log++;
    }

    /*
     * For sources being PRESENT_BLIT'd to the framebuffer:
     * The shadow RT only covers the region drawn by 3D draws (positioned
     * at screen coordinates), leaving the rest zero/black. We use
     * alpha-masked blitting: only write pixels with alpha > 0 (actually
     * rendered content), preserving the existing framebuffer content for
     * undrawn areas. This is safe because undrawn SRT pixels are always
     * 0x00000000 (zero-initialized + cleared on pitch conflict), while
     * rendered pixels have alpha > 0.
     */
    bool alpha_mask_blit = (blit->dst_offset == 0);
    bool stale_srt = alpha_mask_blit && st->shadow_rts[slot].stale;

    /* DIAGNOSTIC: Log every PRESENT_BLIT in sequence to understand ordering */
    if (alpha_mask_blit) {
        static uint32_t blit_seq = 0;
        static uint32_t diag_count = 0;
        blit_seq++;
        if (diag_count < 200) {
            uint32_t src_pitch_px = blit->src_pitch / (blit->bpp / 8);
            fprintf(stderr, "[BLIT_SEQ] seq=%u src=0x%x pitch=%u "
                    "sxy=(%u,%u) dxy=(%u,%u) %ux%u slot=%d gen=%u\n",
                    blit_seq, blit->src_offset, src_pitch_px,
                    blit->src_x, blit->src_y,
                    blit->dst_x, blit->dst_y,
                    blit->width, blit->height,
                    slot, st->rt_frame_gen);
            diag_count++;
        }
    }

    /*
     * Fresh SRT-backed presents are already compositor-authoritative window
     * surfaces. A synthetic framebuffer copy on top of those presents creates
     * the duplicate body/frame ownership we see as ghosting during drags.
     *
     * Keep drag tracking for future stale/unrendered moves, but suppress the
     * body-copy fallback whenever the present is coming from a freshly rendered
     * SRT path.
     */
    if (alpha_mask_blit && !stale_srt) {
        for (int i = 0; i < DRAG_TRACKER_MAX; i++) {
            if (st->drag_trackers[i].active &&
                st->drag_trackers[i].src_offset == blit->src_offset &&
                st->drag_trackers[i].copy_pending) {
                static int drag_skip_log = 0;
                if (drag_skip_log < 60) {
                    fprintf(stderr,
                            "[DRAG_COPY_SKIP] src=0x%x reason=srt_authoritative_surface "
                            "frame=%u\n",
                            blit->src_offset,
                            st->frame_tracker.frame_id);
                    drag_skip_log++;
                }
                st->drag_trackers[i].copy_pending = false;
                break;
            }
        }
    }

    if (alpha_mask_blit && !stale_srt &&
        blit->width > 200 && blit->height > 200) {
        uint32_t cur_origin_x = blit->dst_x - blit->src_x;
        uint32_t cur_origin_y = blit->dst_y - blit->src_y;

        drag_tracker_update(st, blit->src_offset,
                           cur_origin_x, cur_origin_y,
                           blit->width, blit->height);
    }

    uint32_t sw = st->shadow_rts[slot].width;
    uint32_t sh = st->shadow_rts[slot].height;
    const uint32_t *src_pixels = st->shadow_rts[slot].pixels;
    if (!src_pixels) {
        return -1;
    }

    /* Sibling layer re-blit: after this PRESENT_BLIT completes, we may
     * need to re-apply a sibling layer's opaque content on top.  This
     * handles the case where per-pitch SRTs expose separate compositor
     * layers, and a background layer (wallpaper) overwrites a foreground
     * layer (window body) due to BLIT ordering.  Tracked here, applied
     * after the main pixel loop. */

    /*
     * SRT row stride is the SRT width (which may have grown beyond
     * pitch_pixels to accommodate screen-coordinate vertex positions).
     */
    uint32_t srt_pitch = sw;  /* use actual SRT width as stride */

    static int blit_log = 0;
    if (blit_log < 60) {
        /* Count nonzero pixels in blit region using BOTH coord systems */
        uint32_t nz_src = 0, nz_dst = 0, total_src = 0, total_dst = 0;
        /* Sample at src coords */
        for (uint32_t ry = 0; ry < blit->height && ry + blit->src_y < sh; ry++) {
            for (uint32_t rx = 0; rx < blit->width && rx + blit->src_x < sw; rx++) {
                uint32_t px = src_pixels[(blit->src_y + ry) * srt_pitch +
                                          blit->src_x + rx];
                if (px != 0) nz_src++;
                total_src++;
            }
        }
        /* Sample at dst coords */
        for (uint32_t ry = 0; ry < blit->height && ry + blit->dst_y < sh; ry++) {
            for (uint32_t rx = 0; rx < blit->width && rx + blit->dst_x < sw; rx++) {
                uint32_t px = src_pixels[(blit->dst_y + ry) * srt_pitch +
                                          blit->dst_x + rx];
                if (px != 0) nz_dst++;
                total_dst++;
            }
        }
        fprintf(stderr, "[SRT_BLIT] src=0x%x pitch=%u srt=%ux%u "
                 "blit=%ux%u sxy=(%u,%u) dxy=(%u,%u) "
                 "nz_src=%u/%u nz_dst=%u/%u\n",
                 blit->src_offset, src_pitch_pixels, sw, sh,
                 blit->width, blit->height,
                 blit->src_x, blit->src_y,
                 blit->dst_x, blit->dst_y,
                 nz_src, total_src, nz_dst, total_dst);
        blit_log++;
    }

    /*
     * Copy from shadow RT to destination VRAM.
     * Shadow RT pixels are in BE format (same as VRAM / guest).
     * Destination is always linear (framebuffer).
     *
     * Coordinate mapping for SRT reads:
     *
     * The R200 compositor renders at screen coordinates. When screen_x < pitch,
     * the SRT position equals the screen position. When screen_x >= pitch,
     * real hardware wraps via the pitch stride, but our Metal renderer doesn't.
     *
     * The BLT src_xy provides the correct VRAM-space coordinates for reading
     * from the surface. For most BLITs, src_xy matches the screen position
     * (when content is within pitch). For some BLITs (especially shadows),
     * src_xy maps to a LOCAL position within the surface.
     *
     * Strategy: use dst_coords by default (works for screen-coordinate
     * content). When src_xy != (0,0) AND src_xy != dst_xy, the BLT may
     * target a sub-region with local coordinates — use src_coords for
     * those blits. This handles shadow strips and window sub-regions.
     */
    /*
     * Coordinate selection for SRT reads:
     *
     * The R200 3D engine renders at SCREEN coordinates (VTX_XY_FMT=1
     * means vertices are already in screen space). Content is stored
     * at screen positions in the SRT.
     *
     * The BLT's src_xy is relative to the surface origin, and the
     * mapping is: framebuffer_pos = surface_origin + src_xy.
     * Since dst_xy = surface_origin + src_xy, the BLT reads from
     * screen position dst_xy — so we always use dst_coords.
     *
     * For non-framebuffer BLITs (surface-to-surface copies), the
     * source isn't the framebuffer, so src_coords are appropriate.
     */
    /*
     * Coordinate selection for SRT reads:
     *
     * FRESH SRTs (just redrawn this frame): use dst_coords.  The 3D
     * draws placed content at screen positions matching the BLIT's
     * dst_xy, so dst_coords correctly index the SRT.
     *
     * STALE SRTs (not redrawn, content from a prior frame): use
     * src_coords.  This matches real R200 VRAM behavior — the 2D BLT
     * reads from fixed buffer-relative addresses via (src_x, src_y),
     * and dst_xy determines where those bytes appear on screen.  When
     * the compositor changes dst_xy during window drag, the same SRT
     * content appears at a different screen position — the window
     * "moves" without being redrawn.
     *
     * Non-framebuffer BLITs: always use src_coords.
     */
    bool use_dst_coords;
    if (!alpha_mask_blit) {
        use_dst_coords = false;
    } else if (stale_srt) {
        use_dst_coords = false;  /* stale: src_coords like real VRAM */
    } else {
        use_dst_coords = true;   /* fresh: dst_coords match draws */
    }

    uint32_t blit_written = 0, blit_skipped = 0, blit_oob = 0;
    for (uint32_t row = 0; row < blit->height; row++) {
        uint32_t dy = blit->dst_y + row;
        uint32_t sy;
        if (use_dst_coords) {
            sy = blit->dst_y + row;
        } else {
            sy = blit->src_y + row;
        }

        if (sy >= sh) { blit_oob += blit->width; continue; }

        const uint32_t *src_row = src_pixels + (uint64_t)sy * srt_pitch;

        for (uint32_t col = 0; col < blit->width; col++) {
            uint32_t dx = blit->dst_x + col;
            uint32_t sx;
            if (use_dst_coords) {
                sx = blit->dst_x + col;
            } else {
                sx = blit->src_x + col;
            }

            if (sx >= sw) { blit_oob++; continue; }

            uint32_t pixel_be = src_row[sx];

            /*
             * Alpha-masked blit: for zero SRT pixels (undrawn areas),
             * fall back to reading from VRAM at the source address.
             *
             * On real R200, the PRESENT_BLIT reads from VRAM which has
             * BOTH compositor-drawn content (shadows/corners) AND
             * CPU-written content (window body from WindowServer memcpy).
             * Our SRT only captures 3D draws — CPU writes go directly
             * to VRAM.  For zero SRT pixels, VRAM may have CPU-written
             * body pixels that we must copy to the framebuffer.
             */
            if (alpha_mask_blit && pixel_be == 0) {
                blit_skipped++;
                continue;
            }

            /*
             * QE layer compositing: skip fully opaque SRT pixels.
             *
             * In the QE compositing pipeline, the SRT contains two
             * types of content:
             *   - Wallpaper tiles (fully opaque, alpha >= 254): these
             *     are the desktop background BEHIND the window, rendered
             *     by the compositor as textured quads.
             *   - Shadows/corners (semi-transparent, alpha < 254):
             *     these are the actual compositor output (drop shadows,
             *     window frame effects).
             *
             * The CPU (WindowServer via Quartz 2D) writes the actual
             * visible content — wallpaper AND window body — directly
             * to the framebuffer.  The compositor should only ADD
             * semi-transparent shadow effects on top.
             *
             * By skipping opaque pixels, we preserve the CPU-written
             * content (wallpaper + body) and only alpha-blend the
             * shadow layers.  This naturally handles:
             *   - Static desktop: body preserved, shadows on top
             *   - Window drag: body stays at CPU-written position
             *   - Window close/minimize: CPU redraws wallpaper
             */
            /*
             * Skip opaque SRT pixels for WINDOW compositor layers only.
             *
             * The QE compositor has two types of PRESENT_BLITs:
             *   1. Wallpaper layer (full-screen, e.g. 1024x768): draws
             *      the desktop background.  These MUST be written so
             *      the wallpaper appears and old window positions are
             *      cleaned up.
             *   2. Window layers (partial screen): draw shadows/corners
             *      around windows.  Opaque pixels in these are wallpaper
             *      BEHIND the window that would overwrite the CPU-drawn
             *      body.  Skip them to preserve the body.
             *
             * Heuristic: if the BLIT covers >90% of the screen area,
             * it's the wallpaper layer — let opaque pixels through.
             */
            if (alpha_mask_blit && !stale_srt &&
                (pixel_be & 0xFF) >= 254 &&
                st->qe_compositing_active) {
                if (st->drag_protect_active &&
                    (int32_t)dx >= st->drag_protect_x &&
                    (int32_t)dx < st->drag_protect_x + st->drag_protect_w &&
                    (int32_t)dy >= st->drag_protect_y &&
                    (int32_t)dy < st->drag_protect_y + st->drag_protect_h) {
                    /* Preserve the copied body, but allow opaque frame
                     * pixels outside that rect to write through. */
                    blit_skipped++;
                    continue;
                }

                if (!st->drag_protect_active) {
                    uint32_t screen_w = st->scanout_width ?
                        st->scanout_width : (blit->dst_pitch / 4);
                    uint32_t screen_h = st->scanout_height ?
                        st->scanout_height : 768;
                    uint32_t blit_area = blit->width * blit->height;
                    uint32_t screen_area = screen_w * screen_h;
                    if (blit_area < screen_area * 9 / 10) {
                        /* Window layer — skip opaque (wallpaper behind window) */
                        blit_skipped++;
                        continue;
                    }
                }
                /* Wallpaper layer — write through to restore background */
            }

            blit_written++;
            /* Write to destination VRAM (linear, no MC tiling for framebuffer) */
            uint64_t dst_addr = (uint64_t)blit->dst_offset +
                                (uint64_t)dy * blit->dst_pitch +
                                (uint64_t)dx * bpp_bytes;
            if (dst_addr + 4 <= st->vram_size) {
                /*
                 * Premultiplied alpha compositing for PRESENT_BLITs:
                 *
                 * On real R200, QE draws shadows directly to the
                 * framebuffer via the 3D engine with alpha blending.
                 * In our implementation, shadows are drawn to a
                 * separate SRT (against black), then BLTed here.
                 * Semi-transparent pixels (shadows) need alpha
                 * blending with existing framebuffer content.
                 *
                 * The SRT stores bswap32(metal_pixel). Metal outputs
                 * LE BGRA8: as uint32 = A<<24|R<<16|G<<8|B.
                 * After bswap32: B<<24|G<<16|R<<8|A.
                 * So on our LE host:
                 *   alpha = bits[7:0], red = bits[15:8],
                 *   green = bits[23:16], blue = bits[31:24]
                 *
                 * Premultiplied alpha: result = src + dst * (1 - src_a)
                 */
                uint32_t src_a = pixel_be & 0xFF;
                if (!alpha_mask_blit || src_a >= 254) {
                    /* Fully opaque or non-alpha blit: direct copy */
                    *(uint32_t *)(vram_ptr + dst_addr) = pixel_be;
                } else {
                    /* Semi-transparent pixel = shadow/corner from QE compositor.
                     * First occurrence activates the opaque-pixel skip. */
                    if (!st->qe_compositing_active) {
                        st->qe_compositing_active = true;
                        fprintf(stderr, "[QE_ACTIVE] First semi-transparent "
                                "pixel detected — enabling opaque skip\n");
                    }
                    /* Semi-transparent: alpha-blend with framebuffer */
                    uint32_t dst_pixel = *(uint32_t *)(vram_ptr + dst_addr);
                    uint32_t inv_a = 255 - src_a;

                    uint32_t sr = (pixel_be >>  8) & 0xFF;
                    uint32_t sg = (pixel_be >> 16) & 0xFF;
                    uint32_t sb = (pixel_be >> 24) & 0xFF;

                    uint32_t dr = (dst_pixel >>  8) & 0xFF;
                    uint32_t dg = (dst_pixel >> 16) & 0xFF;
                    uint32_t db = (dst_pixel >> 24) & 0xFF;
                    uint32_t da = dst_pixel & 0xFF;

                    uint32_t rr = sr + (dr * inv_a + 127) / 255;
                    uint32_t rg = sg + (dg * inv_a + 127) / 255;
                    uint32_t rb = sb + (db * inv_a + 127) / 255;
                    uint32_t ra = src_a + (da * inv_a + 127) / 255;

                    if (rr > 255) rr = 255;
                    if (rg > 255) rg = 255;
                    if (rb > 255) rb = 255;
                    if (ra > 255) ra = 255;

                    *(uint32_t *)(vram_ptr + dst_addr) =
                        (rb << 24) | (rg << 16) | (rr << 8) | ra;
                }
            }
        }
    }

    {
        static int blit_diag = 0;
        if (blit_diag < 80) {
            fprintf(stderr, "[BLIT_DIAG] src=0x%x p=%u alpha=%d "
                    "sxy=(%u,%u) dxy=(%u,%u) %ux%u srt=%ux%u "
                    "use_dst=%d written=%u skipped=%u oob=%u\n",
                    blit->src_offset, src_pitch_pixels, alpha_mask_blit,
                    blit->src_x, blit->src_y,
                    blit->dst_x, blit->dst_y,
                    blit->width, blit->height, sw, sh,
                    use_dst_coords, blit_written, blit_skipped, blit_oob);
            blit_diag++;
        }
        /*
         * Shadow pixel diagnostic: scan entire SRT for semi-transparent
         * pixels (alpha > 0 && alpha < 255) and sample edge pixels.
         */
        static int shadow_px_diag = 0;
        static int shadow_px_total = 0;
        shadow_px_total++;
        /* Reset counter after 200 BLITs to capture steady-state */
        if (shadow_px_total == 200) shadow_px_diag = 0;
        if (alpha_mask_blit && shadow_px_diag < 40 &&
            blit->width > 10 && blit->height > 10) {
            uint32_t srt_pitch = sw;
            /* Scan for semi-transparent pixels in the BLIT region */
            uint32_t trans_count = 0, opaque_count = 0, zero_count = 0;
            uint32_t first_trans_x = 0, first_trans_y = 0;
            uint32_t first_trans_val = 0;
            for (uint32_t sy = blit->dst_y;
                 sy < blit->dst_y + blit->height && sy < sh; sy++) {
                for (uint32_t sx = blit->dst_x;
                     sx < blit->dst_x + blit->width && sx < sw; sx++) {
                    uint32_t px = src_pixels[(uint64_t)sy * srt_pitch + sx];
                    uint32_t a = px & 0xFF;
                    if (px == 0) zero_count++;
                    else if (a < 254) {
                        if (trans_count == 0) {
                            first_trans_x = sx;
                            first_trans_y = sy;
                            first_trans_val = px;
                        }
                        trans_count++;
                    }
                    else opaque_count++;
                }
            }
            /* Sample edge pixels (top-left corner, 3 pixels in) */
            uint32_t edge_px[4] = {0};
            uint32_t edge_x[4], edge_y[4];
            edge_x[0] = blit->dst_x;     edge_y[0] = blit->dst_y;
            edge_x[1] = blit->dst_x + 2; edge_y[1] = blit->dst_y;
            edge_x[2] = blit->dst_x;     edge_y[2] = blit->dst_y + 2;
            edge_x[3] = blit->dst_x + blit->width - 1;
            edge_y[3] = blit->dst_y;
            for (int ei = 0; ei < 4; ei++) {
                if (edge_x[ei] < sw && edge_y[ei] < sh) {
                    edge_px[ei] = src_pixels[
                        (uint64_t)edge_y[ei] * srt_pitch + edge_x[ei]];
                }
            }
            fprintf(stderr, "[SHADOW_SCAN] src=0x%x p=%u dxy=(%u,%u) %ux%u "
                    "zero=%u trans=%u opaque=%u "
                    "first_trans=(%u,%u)=0x%08x(a=%u) "
                    "edge[TL]=0x%08x edge[+2,0]=0x%08x "
                    "edge[0,+2]=0x%08x edge[TR]=0x%08x\n",
                    blit->src_offset, src_pitch_pixels,
                    blit->dst_x, blit->dst_y,
                    blit->width, blit->height,
                    zero_count, trans_count, opaque_count,
                    first_trans_x, first_trans_y,
                    first_trans_val, first_trans_val & 0xFF,
                    edge_px[0], edge_px[1], edge_px[2], edge_px[3]);
            shadow_px_diag++;
        }
        /* Phase A: detect and log OOB reads during PRESENT_BLIT */
        if (blit_oob > 0 || blit_skipped > blit_written) {
            static int oob_log = 0;
            if (oob_log < 60) {
                /* Compute what coordinates were attempted */
                uint32_t read_max_x = use_dst_coords
                    ? blit->dst_x + blit->width
                    : blit->src_x + blit->width;
                uint32_t read_max_y = use_dst_coords
                    ? blit->dst_y + blit->height
                    : blit->src_y + blit->height;
                fprintf(stderr,
                    "[SRT_EXTENT_OOB] "
                    "rt_off=0x%x rt_pitch=%u "
                    "srt_size=(%u,%u) "
                    "read_max=(%u,%u) "
                    "oob_pixels=%u skipped=%u written=%u "
                    "blit_dst=(%u,%u) blit_size=%ux%u "
                    "alpha_mask=%d "
                    "note=reads_exceed_srt_bounds\n",
                    blit->src_offset, src_pitch_pixels,
                    sw, sh,
                    read_max_x, read_max_y,
                    blit_oob, blit_skipped, blit_written,
                    blit->dst_x, blit->dst_y,
                    blit->width, blit->height,
                    alpha_mask_blit);
                oob_log++;
            }
        }
    }

    /* Track: SRT-backed surface move or visible present. */
    frame_tracker_record_detail(&st->frame_tracker,
                                blit->dst_offset == 0
                                    ? PASS_EVENT_PRESENT
                                    : PASS_EVENT_BLIT_UPLOAD,
                                blit->dst_offset == 0
                                    ? blit->src_offset
                                    : blit->dst_offset,
                                blit->dst_pitch / bpp_bytes,
                                blit->src_offset, blit->dst_offset,
                                blit->width, blit->height,
                                blit->src_x, blit->src_y,
                                blit->dst_x, blit->dst_y,
                                0, 0, true);
    /* If this is a full-screen present to FB, dump the frame */
    if (blit->dst_offset == 0 && blit->width >= 800 && blit->height >= 400) {
        frame_tracker_dump(&st->frame_tracker);
        frame_tracker_new_frame(&st->frame_tracker);
    }

    /*
     * If the SRT provided no pixels at all (blit_written == 0), the SRT
     * has no useful content for this blit region.  Two sub-cases:
     *
     * 1. All pixels were zero (blit_skipped > 0): SRT region is in-bounds
     *    but undrawn.  Return 0 to preserve existing framebuffer content
     *    (don't let VRAM fallback overwrite with potentially garbled data).
     *
     * 2. No pixels were even checked (blit_skipped == 0, blit_oob == 0):
     *    The entire read region was beyond SRT bounds (sy >= sh for all
     *    rows).  SRT truly has nothing for this blit.  Return -1 so the
     *    caller falls back to VRAM, which may have correct content (e.g.,
     *    small tile surfaces where each icon is at a different VRAM y
     *    position but the Metal draws all target y=0-38 in the SRT).
     */
    if (blit_written == 0 && blit_skipped == 0) {
        static int srt_empty_log = 0;
        if (srt_empty_log < 30) {
            fprintf(stderr, "[SRT_EMPTY_FALLBACK] src=0x%x pitch=%u "
                    "srt=%ux%u src=(%u,%u) dst=(%u,%u) %ux%u "
                    "oob=%u -> VRAM_FALLBACK\n",
                    blit->src_offset, src_pitch_pixels, sw, sh,
                    blit->src_x, blit->src_y,
                    blit->dst_x, blit->dst_y,
                    blit->width, blit->height, blit_oob);
            srt_empty_log++;
        }
        return -1;  /* SRT had nothing — let VRAM handle it */
    }

    /*
     * Mark the SRT as "consumed" after a successful BLIT to the framebuffer.
     * On real hardware, the 3D engine reuses the same VRAM RT for each
     * compositor pass. Between passes, the 2D engine BLITs the RT content
     * to the framebuffer. The next 3D pass then writes fresh content.
     *
     * We don't clear immediately because the same SRT may be BLITted
     * multiple times (e.g., right strip, bottom strip, then full screen).
     * Instead, the "consumed" flag tells the 3D draw path to clear the
     * SRT before the first draw of the next pass. This ensures each pass
     * starts with a clean slate, preventing additive accumulation.
     */
    if (alpha_mask_blit && blit_written > 0) {
        /* Capture content_origin on the FIRST consume after a redraw.
         * If the SRT was NOT stale (just redrawn this frame), record
         * the current dst_xy as the content origin.  This is where the
         * 3D-drawn content was placed on screen.  On subsequent BLITs
         * with changed dst_xy (drag), we use this origin to read from
         * the correct SRT position. */
        if (!stale_srt) {
            st->shadow_rts[slot].content_origin_x = (int32_t)blit->dst_x;
            st->shadow_rts[slot].content_origin_y = (int32_t)blit->dst_y;
            st->shadow_rts[slot].origin_valid = true;
        }
        st->shadow_rts[slot].consumed = true;
        static int consumed_log = 0;
        if (consumed_log < 200) {
            fprintf(stderr, "[SRT_CONSUMED] slot=%d off=0x%x pitch=%u "
                    "written=%u stale=%d origin=(%d,%d)\n",
                    slot, blit->src_offset,
                    st->shadow_rts[slot].pitch_pixels,
                    blit_written, stale_srt,
                    st->shadow_rts[slot].content_origin_x,
                    st->shadow_rts[slot].content_origin_y);
            consumed_log++;
        }
    }

    /*
     * Framebuffer pixel diagnostic: after writing shadow strips to the FB,
     * sample actual FB pixels at the shadow position and compare with
     * pixels just outside the shadow (pure wallpaper).
     * This tells us what's actually visible on screen.
     */
    static int fb_diag = 0;
    static int fb_diag_total = 0;
    fb_diag_total++;
    if (fb_diag_total == 300) fb_diag = 0; /* reset for steady state */
    if (alpha_mask_blit && fb_diag < 20 &&
        blit->width >= 10 && blit->width <= 30 &&
        blit->height > 100 && blit->dst_offset == 0) {
        /* This looks like a shadow edge strip (narrow, tall) */
        uint32_t fb_pitch = blit->dst_pitch;
        uint32_t bpp = blit->bpp / 8;
        /* Sample FB at: shadow outer edge, shadow middle, shadow inner edge,
         * and 5px outside the shadow (pure wallpaper) */
        uint32_t mid_y = blit->dst_y + blit->height / 2;
        uint32_t positions[][2] = {
            { blit->dst_x > 5 ? blit->dst_x - 5 : 0, mid_y }, /* outside shadow */
            { blit->dst_x, mid_y },                             /* shadow outer */
            { blit->dst_x + blit->width / 2, mid_y },          /* shadow middle */
            { blit->dst_x + blit->width - 1, mid_y },          /* shadow inner */
            { blit->dst_x + blit->width + 5, mid_y },          /* inside body */
        };
        fprintf(stderr, "[FB_SHADOW_DIAG] shadow strip src=0x%x dxy=(%u,%u) %ux%u "
                "mid_y=%u fb_pixels:",
                blit->src_offset, blit->dst_x, blit->dst_y,
                blit->width, blit->height, mid_y);
        for (int pi = 0; pi < 5; pi++) {
            uint32_t px = positions[pi][0];
            uint32_t py = positions[pi][1];
            uint64_t fb_addr = (uint64_t)blit->dst_offset +
                               (uint64_t)py * fb_pitch +
                               (uint64_t)px * bpp;
            uint32_t fb_pixel = 0;
            if (fb_addr + 4 <= st->vram_size) {
                fb_pixel = *(uint32_t *)(vram_ptr + fb_addr);
            }
            uint32_t a = fb_pixel & 0xFF;
            uint32_t r = (fb_pixel >> 8) & 0xFF;
            uint32_t g = (fb_pixel >> 16) & 0xFF;
            uint32_t b = (fb_pixel >> 24) & 0xFF;
            const char *label = (pi == 0) ? "outside" :
                                (pi == 1) ? "outer" :
                                (pi == 2) ? "mid" :
                                (pi == 3) ? "inner" :
                                            "body";
            fprintf(stderr, " %s[%u,%u]=0x%08x(r%u,g%u,b%u,a%u)",
                    label, px, py, fb_pixel, r, g, b, a);
        }
        fprintf(stderr, "\n");
        fb_diag++;
    }

    return 0;  /* Success — handled by shadow RT */
}

static int metal_fill_2d(void *opaque, uint8_t *vram_ptr,
                          const PPCMacGPUBlit *fill)
{
    /* Fall back to software for 2D fills */
    (void)opaque;
    (void)vram_ptr;
    (void)fill;
    return -1;
}

/*
 * R200 tiling helpers for 32bpp surfaces.
 *
 * Micro-tile = 8 pixels wide × 2 rows tall = 64 bytes.
 * Pixels within a micro-tile are stored row-major.
 * Micro-tiles are arranged left-to-right, top-to-bottom.
 */
static inline uint64_t r200_microtile_offset(uint32_t x, uint32_t y,
                                              uint32_t pitch_pixels)
{
    uint32_t tile_x = x / 8;
    uint32_t tile_y = y / 2;
    uint32_t in_tile_x = x % 8;
    uint32_t in_tile_y = y % 2;
    uint32_t tiles_per_row = pitch_pixels / 8;
    uint64_t tile_offset = ((uint64_t)tile_y * tiles_per_row + tile_x) * 64;
    uint64_t pixel_offset = ((uint64_t)in_tile_y * 8 + in_tile_x) * 4;
    return tile_offset + pixel_offset;
}

/*
 * R200 macro-tile offset for 32bpp surfaces (no micro-tiling within).
 *
 * Macro-tile = 64 pixels wide × 8 rows tall = 2048 bytes.
 * Within each macro-tile, pixels are stored linearly (row-major).
 * Macro-tiles are arranged left-to-right, top-to-bottom in the surface.
 *
 * This layout is used when RB3D_COLORPITCH bit 16 is set and bit 15
 * is NOT set (macro-only, no micro-tiling within).
 *
 * Reference: Mesa r200/radeon drivers, Linux DRM radeon driver.
 * R200/RV280 with 1 pipe, group_size=256 bytes.
 */
/*
 * RV280 (Radeon 9200) macro-tile dimensions for 32bpp color buffers.
 * RV280 is a 1-pipe GPU: macro-tile = 256 bytes wide × 16 rows = 4096 bytes.
 * (2-pipe R200 chips use 256 × 8 = 2048 bytes, but RV280 is single-pipe.)
 */
#define R200_MACROTILE_WIDTH_PX  64
#define R200_MACROTILE_HEIGHT_PX 16
#define R200_MACROTILE_WIDTH_BYTES (R200_MACROTILE_WIDTH_PX * 4) /* 256 */
#define R200_MACROTILE_BYTES (R200_MACROTILE_WIDTH_BYTES * R200_MACROTILE_HEIGHT_PX) /* 4096 */

static inline uint64_t r200_macrotile_offset(uint32_t x, uint32_t y,
                                              uint32_t pitch_pixels)
{
    uint32_t macro_x = x / R200_MACROTILE_WIDTH_PX;
    uint32_t macro_y = y / R200_MACROTILE_HEIGHT_PX;
    uint32_t in_macro_x = x % R200_MACROTILE_WIDTH_PX;
    uint32_t in_macro_y = y % R200_MACROTILE_HEIGHT_PX;
    uint32_t macrotiles_per_row = pitch_pixels / R200_MACROTILE_WIDTH_PX;
    if (macrotiles_per_row == 0) macrotiles_per_row = 1;

    uint64_t tile_offset = ((uint64_t)macro_y * macrotiles_per_row + macro_x)
                           * R200_MACROTILE_BYTES;
    uint64_t pixel_offset = (uint64_t)in_macro_y * R200_MACROTILE_WIDTH_BYTES
                           + (uint64_t)in_macro_x * 4;
    return tile_offset + pixel_offset;
}

/*
 * R200 macro+micro tile offset for 32bpp surfaces.
 *
 * When BOTH macro-tile (bit 16) and micro-tile (bit 15) are set,
 * macro-tiles contain micro-tiles instead of linear pixels.
 * Macro-tile = 64 pixels wide × 16 rows = 4096 bytes (RV280 1-pipe).
 * Within: 8 micro-tile columns × 8 micro-tile rows = 64 micro-tiles.
 * Each micro-tile = 8×2 pixels = 64 bytes.
 */
static inline uint64_t r200_macro_micro_tile_offset(uint32_t x, uint32_t y,
                                                     uint32_t pitch_pixels)
{
    uint32_t macro_x = x / R200_MACROTILE_WIDTH_PX;
    uint32_t macro_y = y / R200_MACROTILE_HEIGHT_PX;
    uint32_t in_macro_x = x % R200_MACROTILE_WIDTH_PX;
    uint32_t in_macro_y = y % R200_MACROTILE_HEIGHT_PX;
    uint32_t macrotiles_per_row = pitch_pixels / R200_MACROTILE_WIDTH_PX;
    if (macrotiles_per_row == 0) macrotiles_per_row = 1;

    /* Micro-tile within macro-tile */
    uint32_t utile_col = in_macro_x / 8;  /* 0..7 */
    uint32_t utile_row = in_macro_y / 2;  /* 0..7 (16 rows / 2 per utile) */
    uint32_t in_utile_x = in_macro_x % 8;
    uint32_t in_utile_y = in_macro_y % 2;

    uint64_t macro_offset = ((uint64_t)macro_y * macrotiles_per_row + macro_x)
                            * R200_MACROTILE_BYTES;
    uint64_t utile_index = (uint64_t)utile_row * 8 + utile_col;
    uint64_t utile_offset = utile_index * 64;
    uint64_t pixel_offset = ((uint64_t)in_utile_y * 8 + in_utile_x) * 4;

    return macro_offset + utile_offset + pixel_offset;
}

/*
 * Unified R200 tile offset resolver for 32bpp.
 * Dispatches to the correct helper based on micro/macro flags.
 */
static inline uint64_t r200_tile_offset(uint32_t x, uint32_t y,
                                         uint32_t pitch_pixels,
                                         bool micro_tiled, bool macro_tiled)
{
    if (macro_tiled && micro_tiled) {
        return r200_macro_micro_tile_offset(x, y, pitch_pixels);
    } else if (macro_tiled) {
        return r200_macrotile_offset(x, y, pitch_pixels);
    } else if (micro_tiled) {
        return r200_microtile_offset(x, y, pitch_pixels);
    } else {
        return (uint64_t)y * pitch_pixels * 4 + (uint64_t)x * 4;
    }
}

/*
 * Validation: write a known pattern to a macro-tiled region, read it back,
 * confirm round-trip is correct.
 */
static void r200_macrotile_validate(uint32_t pitch_pixels,
                                     bool micro_tiled, bool macro_tiled)
{
    /* Small test: 128×16 pixels (covers 2×2 macro tiles at 64×8) */
    uint32_t test_w = (pitch_pixels < 128) ? pitch_pixels : 128;
    uint32_t test_h = 16;
    uint64_t buf_size = (uint64_t)pitch_pixels * test_h * 4 * 2; /* generous */
    if (buf_size > 64 * 1024 * 1024) buf_size = 64 * 1024 * 1024;
    uint8_t *buf = (uint8_t *)g_malloc0(buf_size);

    /* Write known pattern */
    uint32_t mismatch_count = 0;
    uint32_t first_mx = 0, first_my = 0;
    uint32_t first_expected = 0, first_actual = 0;

    for (uint32_t y = 0; y < test_h; y++) {
        for (uint32_t x = 0; x < test_w; x++) {
            uint32_t pattern = ((y & 0xFF) << 24) | ((x & 0xFF) << 16) |
                               (((x ^ y) & 0xFF) << 8) | 0x42;
            uint64_t off = r200_tile_offset(x, y, pitch_pixels,
                                             micro_tiled, macro_tiled);
            if (off + 4 <= buf_size) {
                *(uint32_t *)(buf + off) = pattern;
            }
        }
    }

    /* Read back and verify */
    for (uint32_t y = 0; y < test_h; y++) {
        for (uint32_t x = 0; x < test_w; x++) {
            uint32_t expected = ((y & 0xFF) << 24) | ((x & 0xFF) << 16) |
                                (((x ^ y) & 0xFF) << 8) | 0x42;
            uint64_t off = r200_tile_offset(x, y, pitch_pixels,
                                             micro_tiled, macro_tiled);
            uint32_t actual = 0;
            if (off + 4 <= buf_size) {
                actual = *(uint32_t *)(buf + off);
            }
            if (actual != expected) {
                if (mismatch_count == 0) {
                    first_mx = x;
                    first_my = y;
                    first_expected = expected;
                    first_actual = actual;
                }
                mismatch_count++;
            }
        }
    }

    const char *result = (mismatch_count == 0) ? "pass" : "fail";
    qemu_log("[RT_TILE_TEST] target_off=0x0 target_pitch=%u "
             "size=%ux%u pattern=xy_hash "
             "result=%s mismatch_count=%u "
             "first_mismatch=(%u,%u) expected=0x%08x actual=0x%08x "
             "micro=%d macro=%d\n",
             pitch_pixels, test_w, test_h,
             result, mismatch_count,
             first_mx, first_my, first_expected, first_actual,
             micro_tiled, macro_tiled);

    /* Also check for address collisions (two different pixels mapping
     * to the same offset) */
    memset(buf, 0, buf_size);
    uint32_t collision_count = 0;
    for (uint32_t y = 0; y < test_h; y++) {
        for (uint32_t x = 0; x < test_w; x++) {
            uint64_t off = r200_tile_offset(x, y, pitch_pixels,
                                             micro_tiled, macro_tiled);
            if (off + 4 <= buf_size) {
                uint32_t prev = *(uint32_t *)(buf + off);
                if (prev != 0) {
                    collision_count++;
                }
                /* Store 1-based (x,y) encoding so 0 means empty */
                *(uint32_t *)(buf + off) = ((x + 1) << 16) | (y + 1);
            }
        }
    }
    if (collision_count > 0) {
        qemu_log("[RT_TILE_TEST] COLLISION_CHECK: %u collisions detected "
                 "(two pixels mapped to same offset)\n", collision_count);
    } else {
        qemu_log("[RT_TILE_TEST] COLLISION_CHECK: pass (no collisions)\n");
    }

    g_free(buf);
}

/*
 * Compute CRC of a tiled VRAM region (bounding box).
 * Used by Phase 3A probe surface validation to verify identity
 * before/after NOP-blend draws.
 */
/* Bring-up diagnostics: real per-draw work (CRC scans, probes), off by default. */
static bool r200_diag_on(void)
{
    static int on = -1;

    if (on < 0) {
        on = getenv("PPCGPU_DIAG") != NULL;
    }
    return on;
}

static uint32_t compute_vram_region_crc(uint8_t *vram_ptr, uint64_t vram_size,
                                         uint32_t color_offset,
                                         uint32_t pitch_pixels,
                                         bool micro_tiled, bool macro_tiled,
                                         uint32_t rx, uint32_t ry,
                                         uint32_t rw, uint32_t rh)
{
    uint32_t crc = 0xFFFFFFFF;
    for (uint32_t y = ry; y < ry + rh; y++) {
        for (uint32_t x = rx; x < rx + rw; x++) {
            uint64_t offset = color_offset +
                r200_tile_offset(x, y, pitch_pixels, micro_tiled, macro_tiled);
            uint32_t pixel = 0;
            if (offset + 4 <= vram_size) {
                pixel = *(uint32_t *)(vram_ptr + offset);
            }
            /* Fold pixel into CRC */
            for (int b = 0; b < 4; b++) {
                uint8_t byte = (pixel >> (b * 8)) & 0xFF;
                crc ^= byte;
                for (int j = 0; j < 8; j++) {
                    crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                }
            }
        }
    }
    return ~crc;
}

/* Forward declarations for audit functions */
static void tile_formula_parity_test(void);
static void metal_load_vram_region(PPCMacGPUMetalState *st,
                                    uint8_t *vram_ptr, uint64_t vram_size,
                                    uint32_t color_offset,
                                    uint32_t pitch_pixels,
                                    bool micro_tiled, bool macro_tiled,
                                    uint32_t rx, uint32_t ry,
                                    uint32_t rw, uint32_t rh);
static void metal_writeback_vram_region(PPCMacGPUMetalState *st,
                                         uint8_t *vram_ptr, uint64_t vram_size,
                                         uint32_t color_offset,
                                         uint32_t pitch_pixels,
                                         bool micro_tiled, bool macro_tiled,
                                         uint32_t rt_width,
                                         uint32_t rx, uint32_t ry,
                                         uint32_t rw, uint32_t rh);

/* ========================================================================
 * Phase A-E: Metal Data Path / Byte-Order Validation
 * One-shot audit that runs early in metal_draw_3d lifetime.
 * ======================================================================== */

/*
 * Phase A: Metal round-trip test.
 * Write known BE ARGB patterns to VRAM, load into Metal RT, blit to
 * outputBuffer (no rendering), writeback to VRAM, compare byte-by-byte.
 */
static void metal_roundtrip_test(PPCMacGPUMetalState *st,
                                  uint8_t *vram_ptr, uint64_t vram_size)
{
    @autoreleasepool {
        const uint32_t TW = 4, TH = 4, TP = 4; /* test width, height, pitch */
        const uint32_t TOFF = 0x3E0000;         /* safe VRAM offset */
        const int NPIX = TW * TH;               /* 16 pixels */

        if (TOFF + NPIX * 4 > vram_size) {
            qemu_log("[METAL_ROUNDTRIP] SKIP: test region exceeds VRAM "
                     "(off=0x%x need=%u have=%" PRIu64 ")\n",
                     TOFF, NPIX * 4, vram_size);
            return;
        }

        /* Guest ARGB values we want to simulate */
        static const uint32_t test_argb[16] = {
            0xFF804020, 0xFF102030, 0xFF00FF00, 0xFFFF0000,
            0xFF0000FF, 0xFFFFFFFF, 0xFF000000, 0x80402010,
            0xFEDCBA98, 0x12345678, 0xAABBCCDD, 0x00000000,
            0xFF808080, 0xFF010101, 0xFFCAFE42, 0xDEADBEEF
        };

        /* Save original VRAM so we can restore it */
        uint8_t saved_vram[64];
        memcpy(saved_vram, vram_ptr + TOFF, 64);

        /* Write test patterns as BE ARGB to VRAM.
         * PPC guest writes BE: bytes [A,R,G,B].
         * On LE host, *(uint32_t*) stores in LE order, so we bswap32 to
         * get the correct byte layout. */
        uint32_t *vt = (uint32_t *)(vram_ptr + TOFF);
        for (int i = 0; i < NPIX; i++) {
            vt[i] = __builtin_bswap32(test_argb[i]);
        }

        /* Save copy of what we wrote (the "before" state) */
        uint32_t before[16];
        memcpy(before, vram_ptr + TOFF, 64);

        qemu_log("[METAL_ROUNDTRIP] rt_off=0x%x size=(%u,%u) "
                 "pattern=16_distinctive_ARGB_pixels\n", TOFF, TW, TH);
        qemu_log("[METAL_ROUNDTRIP] before_bytes(host_u32):");
        for (int i = 0; i < 8; i++) qemu_log(" %08x", before[i]);
        qemu_log("\n");

        /* Force a temporary RT at test size.
         * The actual draw will trigger a resize back to the correct size. */
        st->renderTarget = nil;
        st->outputBuffer = nil;
        st->rt_width = 0;
        st->rt_height = 0;

        if (!metal_ensure_render_target(st, TW, TH)) {
            qemu_log("[METAL_ROUNDTRIP] FAIL: RT creation failed\n");
            memcpy(vram_ptr + TOFF, saved_vram, 64);
            return;
        }

        /* Load VRAM → Metal RT (does bswap32: BE→LE BGRA) */
        metal_load_vram_region(st, vram_ptr, vram_size,
                                TOFF, TP, false, false,
                                0, 0, TW, TH);

        /* Blit RT → outputBuffer (no rendering, pure copy) */
        id<MTLCommandBuffer> cmdBuf = [st->commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromTexture:st->renderTarget
                  sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(TW, TH, 1)
                     toBuffer:st->outputBuffer
            destinationOffset:0
       destinationBytesPerRow:TW * 4
     destinationBytesPerImage:TW * TH * 4];
        [blit endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Writeback outputBuffer → VRAM (does bswap32: LE→BE ARGB) */
        metal_writeback_vram_region(st, vram_ptr, vram_size,
                                    TOFF, TP, false, false,
                                    TW, 0, 0, TW, TH);

        /* Compare before/after byte-by-byte */
        uint32_t *after = (uint32_t *)(vram_ptr + TOFF);
        bool byte_exact = true;
        int first_mm = -1;
        uint32_t mm_before = 0, mm_after = 0;

        for (int i = 0; i < NPIX; i++) {
            if (after[i] != before[i]) {
                if (byte_exact) {
                    byte_exact = false;
                    first_mm = i;
                    mm_before = before[i];
                    mm_after = after[i];
                }
            }
        }

        qemu_log("[METAL_ROUNDTRIP] after_bytes(host_u32):");
        for (int i = 0; i < 8; i++) qemu_log(" %08x", after[i]);
        qemu_log("\n");

        if (byte_exact) {
            qemu_log("[METAL_ROUNDTRIP] byte_exact=yes "
                     "note=Metal_load_store_preserves_bytes_correctly\n");
        } else {
            qemu_log("[METAL_ROUNDTRIP] byte_exact=no "
                     "first_mismatch=(%u,%u) "
                     "before_pixel=0x%08x after_pixel=0x%08x "
                     "note=Metal_load_store_CORRUPTS_bytes\n",
                     first_mm % TW, first_mm / TW, mm_before, mm_after);
            /* Dump all 16 pixels for forensic analysis */
            for (int i = 0; i < NPIX; i++) {
                qemu_log("[METAL_ROUNDTRIP] pixel[%d,%d]: "
                         "test_argb=0x%08x vram_before=0x%08x "
                         "vram_after=0x%08x %s\n",
                         i % TW, i / TW, test_argb[i],
                         before[i], after[i],
                         before[i] == after[i] ? "OK" : "MISMATCH");
            }
        }

        /* Restore original VRAM */
        memcpy(vram_ptr + TOFF, saved_vram, 64);

        /* Reset RT so the actual draw recreates it at the right size */
        st->renderTarget = nil;
        st->outputBuffer = nil;
        st->rt_width = 0;
        st->rt_height = 0;
    }
}

/*
 * Phase C: Pixel-format / endianness audit table.
 * Logged once to make the byte representation at each pipeline stage explicit.
 */
static void metal_pixel_format_audit(void)
{
    qemu_log("\n========== [PIXEL_FORMAT_AUDIT] Full Pipeline ==========\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=1_vram_storage "
             "format=BE_ARGB8888 byte_order=[A,R,G,B] "
             "conversion=none swap=no "
             "note=PPC_guest_writes_big_endian_XRGB/ARGB\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=2_metal_load_vram_region "
             "format=BGRA8Unorm byte_order=[B,G,R,A]_LE "
             "conversion=bswap32_per_pixel swap=yes "
             "note=*(u32*)(vram)_reads_LE=0xBGRA_then_bswap32=0xARGB_"
             "stored_as_LE_u32_gives_bytes[B,G,R,A]_matches_BGRA8Unorm\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=3_metal_texture_from_vram "
             "format=BGRA8Unorm byte_order=[B,G,R,A]_LE "
             "conversion=bswap32_per_pixel swap=yes "
             "note=same_as_stage2_for_source_textures\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=4_metal_render_target "
             "format=MTLPixelFormatBGRA8Unorm byte_order=[B,G,R,A] "
             "conversion=none swap=no "
             "note=Metal_standard_BGRA_RT\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=5_fragment_shader_output "
             "format=float4(R,G,B,A) byte_order=N/A "
             "conversion=Metal_auto_maps_RGBA_to_BGRA_storage swap=no "
             "note=shader_returns_float4(r,g,b,a)_Metal_stores_as_BGRA\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=6_metal_writeback_vram_region "
             "format=BE_ARGB8888 byte_order=[A,R,G,B] "
             "conversion=bswap32_per_pixel swap=yes "
             "note=reads_LE_BGRA_u32=0xARGB_bswap32=0xBGRA_reversed_"
             "stored_as_LE_gives_bytes[A,R,G,B]=BE_ARGB\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] stage=7_display_scanout "
             "format=PIXMAN_x8r8g8b8 byte_order=[B,G,R,X]_LE "
             "conversion=bswap32_per_pixel swap=yes "
             "note=VRAM_BE[A,R,G,B]_read_as_LE_u32=0xBGRA_"
             "bswap32=0xARGB_stored_as_LE_bytes[B,G,R,X]_"
             "Cocoa_kCGBitmapByteOrder32Little_interprets_correctly\n");
    qemu_log("[PIXEL_FORMAT_AUDIT] summary="
             "load_bswap+writeback_bswap_cancel_for_NOP_draws | "
             "net_path=VRAM_BE_ARGB->bswap->Metal_BGRA->render->"
             "bswap->VRAM_BE_ARGB->bswap->shadow_LE_XRGB->Cocoa | "
             "total_swaps=3(load+writeback+scanout) | "
             "net=1_swap(scanout_only_for_passthrough) | "
             "double_swap_risk=NONE_if_load_and_writeback_symmetric\n");
    qemu_log("==========================================================\n\n");
}

/*
 * Phase B: Compare one draw's output pixel-level.
 * For a textured draw, dump before/after RT pixels + texture source pixels
 * at sample coordinates so we can verify rendering is correct.
 */
static void metal_draw_pixel_audit(PPCMacGPUMetalState *st,
                                    uint8_t *vram_ptr, uint64_t vram_size,
                                    uint32_t color_offset,
                                    uint32_t color_pitch_pixels,
                                    uint32_t tx_offset,
                                    uint32_t tx_pitch,
                                    uint32_t tx_width, uint32_t tx_height,
                                    uint32_t bb_x, uint32_t bb_y,
                                    uint32_t bb_w, uint32_t bb_h,
                                    uint32_t src_blend, uint32_t dst_blend,
                                    int draw_num,
                                    const char *kind)
{
    /* Sample 4 coordinates in the draw bbox */
    struct { uint32_t x, y; } samples[4];
    int nsamp = 0;

    /* Center of bbox */
    samples[nsamp++] = (typeof(samples[0])){bb_x + bb_w / 2, bb_y + bb_h / 2};
    /* Top-left of bbox */
    samples[nsamp++] = (typeof(samples[0])){bb_x, bb_y};
    /* One pixel in from top-left */
    if (bb_w > 1 && bb_h > 1)
        samples[nsamp++] = (typeof(samples[0])){bb_x + 1, bb_y + 1};
    /* Bottom-right */
    if (bb_w > 2 && bb_h > 2)
        samples[nsamp++] = (typeof(samples[0])){bb_x + bb_w - 1, bb_y + bb_h - 1};

    qemu_log("[DRAW_PIXEL_AUDIT] draw_id=%d kind=%s "
             "target_off=0x%x target_pitch=%u "
             "tex_off=0x%x tex_pitch=%u tex_size=%ux%u "
             "bbox=(%u,%u)-(%u,%u) blend_src=0x%x blend_dst=0x%x\n",
             draw_num, kind, color_offset, color_pitch_pixels,
             tx_offset, tx_pitch, tx_width, tx_height,
             bb_x, bb_y, bb_x + bb_w, bb_y + bb_h,
             src_blend, dst_blend);

    for (int s = 0; s < nsamp; s++) {
        uint32_t sx = samples[s].x, sy = samples[s].y;

        /* Read RT pixel (before draw — already in VRAM as BE ARGB) */
        uint64_t rt_off = (uint64_t)color_offset +
            (uint64_t)sy * color_pitch_pixels * 4 + (uint64_t)sx * 4;
        uint32_t rt_raw = 0;
        if (rt_off + 4 <= vram_size) {
            rt_raw = *(uint32_t *)(vram_ptr + rt_off);
        }
        /* Convert from host-LE representation of BE bytes to ARGB value */
        uint32_t rt_argb = __builtin_bswap32(rt_raw);

        /* Read texture pixel at corresponding coordinate.
         * The texture coordinate maps (sx - bb_x) to texture space.
         * For a simple full-quad draw, tex coord ≈ pixel position. */
        uint32_t tx = sx - bb_x, ty = sy - bb_y;
        if (tx >= tx_width) tx = tx_width - 1;
        if (ty >= tx_height) ty = tx_height - 1;
        uint64_t tex_byte_off = (uint64_t)tx_offset +
            (uint64_t)ty * tx_pitch + (uint64_t)tx * 4;
        uint32_t tex_raw = 0;
        if (tex_byte_off + 4 <= vram_size) {
            tex_raw = *(uint32_t *)(vram_ptr + tex_byte_off);
        }
        uint32_t tex_argb = __builtin_bswap32(tex_raw);

        qemu_log("[DRAW_PIXEL_AUDIT] sample[%d] coord=(%u,%u) "
                 "rt_before_argb=0x%08x tex_argb=0x%08x "
                 "rt_raw_host=0x%08x tex_raw_host=0x%08x\n",
                 s, sx, sy, rt_argb, tex_argb, rt_raw, tex_raw);
    }
}

/*
 * Phase B post-draw: read back pixels after Metal rendered and wrote back.
 */
static void metal_draw_pixel_audit_after(uint8_t *vram_ptr, uint64_t vram_size,
                                          uint32_t color_offset,
                                          uint32_t color_pitch_pixels,
                                          uint32_t bb_x, uint32_t bb_y,
                                          uint32_t bb_w, uint32_t bb_h,
                                          int draw_num)
{
    struct { uint32_t x, y; } samples[4];
    int nsamp = 0;
    samples[nsamp++] = (typeof(samples[0])){bb_x + bb_w / 2, bb_y + bb_h / 2};
    samples[nsamp++] = (typeof(samples[0])){bb_x, bb_y};
    if (bb_w > 1 && bb_h > 1)
        samples[nsamp++] = (typeof(samples[0])){bb_x + 1, bb_y + 1};
    if (bb_w > 2 && bb_h > 2)
        samples[nsamp++] = (typeof(samples[0])){bb_x + bb_w - 1, bb_y + bb_h - 1};

    for (int s = 0; s < nsamp; s++) {
        uint32_t sx = samples[s].x, sy = samples[s].y;
        uint64_t rt_off = (uint64_t)color_offset +
            (uint64_t)sy * color_pitch_pixels * 4 + (uint64_t)sx * 4;
        uint32_t rt_raw = 0;
        if (rt_off + 4 <= vram_size) {
            rt_raw = *(uint32_t *)(vram_ptr + rt_off);
        }
        uint32_t rt_argb = __builtin_bswap32(rt_raw);
        qemu_log("[DRAW_PIXEL_AUDIT] draw_id=%d sample[%d] "
                 "rt_after_argb=0x%08x rt_after_raw_host=0x%08x\n",
                 draw_num, s, rt_argb, rt_raw);
    }
}

/*
 * Phase E: Wallpaper texture source verification.
 * Dump raw texture bytes at sample coordinates for the first wallpaper draw.
 */
static void metal_wallpaper_texture_audit(uint8_t *vram_ptr, uint64_t vram_size,
                                           uint32_t tx_offset,
                                           uint32_t tx_pitch,
                                           uint32_t tx_width, uint32_t tx_height,
                                           uint32_t tx_format)
{
    qemu_log("[WALLPAPER_SOURCE] tex_off=0x%x tex_pitch=%u "
             "tex_format=0x%x tex_size=%ux%u\n",
             tx_offset, tx_pitch, tx_format, tx_width, tx_height);

    /* Sample coordinates */
    struct { uint32_t u, v; } coords[] = {
        {0, 0}, {1, 0}, {0, 1}, {8, 8}, {16, 16},
        {tx_width / 2, tx_height / 2}
    };
    int ncoords = sizeof(coords) / sizeof(coords[0]);

    for (int c = 0; c < ncoords; c++) {
        uint32_t u = coords[c].u, v = coords[c].v;
        if (u >= tx_width || v >= tx_height) continue;

        uint64_t byte_off = (uint64_t)tx_offset +
            (uint64_t)v * tx_pitch + (uint64_t)u * 4;
        if (byte_off + 4 > vram_size) {
            qemu_log("[WALLPAPER_SOURCE] sample_coord=(%u,%u) "
                     "SKIP: exceeds VRAM\n", u, v);
            continue;
        }

        uint32_t raw_host = *(uint32_t *)(vram_ptr + byte_off);
        uint32_t decoded_argb = __builtin_bswap32(raw_host);
        uint8_t a = (decoded_argb >> 24) & 0xFF;
        uint8_t r = (decoded_argb >> 16) & 0xFF;
        uint8_t g = (decoded_argb >> 8) & 0xFF;
        uint8_t b = decoded_argb & 0xFF;

        /* Plausibility: a wallpaper pixel should have non-zero alpha
         * and some color variation (not all zeros unless transparent) */
        bool plausible = (a > 0 || (r == 0 && g == 0 && b == 0));

        qemu_log("[WALLPAPER_SOURCE] sample_coord=(%u,%u) "
                 "raw_bytes=%02x,%02x,%02x,%02x "
                 "decoded_argb=0x%08x (A=%u R=%u G=%u B=%u) "
                 "plausible=%s\n",
                 u, v,
                 (uint8_t)(vram_ptr[byte_off]),
                 (uint8_t)(vram_ptr[byte_off + 1]),
                 (uint8_t)(vram_ptr[byte_off + 2]),
                 (uint8_t)(vram_ptr[byte_off + 3]),
                 decoded_argb, a, r, g, b,
                 plausible ? "yes" : "no");
    }

    /* Dump first 16 raw bytes for forensic analysis */
    if (tx_offset + 16 <= vram_size) {
        qemu_log("[WALLPAPER_SOURCE] raw_bytes(first16):");
        for (int i = 0; i < 16; i++) {
            qemu_log(" %02x", vram_ptr[tx_offset + i]);
        }
        qemu_log("\n");
    }
}

/*
 * Upload a rectangular region from VRAM (with R200 tiled layout) into
 * the Metal render target. VRAM stores BE ARGB; Metal uses LE BGRA.
 * bswap32 converts between them.
 *
 * micro_tiled: RB3D_COLORPITCH bit 15 — 8×2 micro-tiles
 * macro_tiled: RB3D_COLORPITCH bit 16 — 64×8 macro-tiles
 */
static void metal_load_vram_region(PPCMacGPUMetalState *st,
                                    uint8_t *vram_ptr, uint64_t vram_size,
                                    uint32_t color_offset,
                                    uint32_t pitch_pixels,
                                    bool micro_tiled, bool macro_tiled,
                                    uint32_t rx, uint32_t ry,
                                    uint32_t rw, uint32_t rh)
{
    if (rw == 0 || rh == 0) return;
    if (!st->renderTarget) return;

    /* Clamp region to render target bounds to avoid AGX OOB assertions */
    if (rx >= st->rt_width || ry >= st->rt_height) return;
    if (rx + rw > st->rt_width) {
        static int load_clamp_log = 0;
        if (load_clamp_log < 5) {
            qemu_log("[RT_CLAMP] load_vram_region: rx=%u rw=%u > rt_w=%u, "
                     "clamping to %u\n", rx, rw, st->rt_width,
                     st->rt_width - rx);
            load_clamp_log++;
        }
        rw = st->rt_width - rx;
    }
    if (ry + rh > st->rt_height) {
        rh = st->rt_height - ry;
    }
    if (rw == 0 || rh == 0) return;

    static bool logged_tile_mode = false;
    if (!logged_tile_mode) {
        qemu_log("[RT_TILE] load_vram_region: micro=%d macro=%d pitch=%u "
                 "off=0x%x region=(%u,%u %ux%u)\n",
                 micro_tiled, macro_tiled, pitch_pixels,
                 color_offset, rx, ry, rw, rh);
        logged_tile_mode = true;
    }

    uint32_t row_bytes = rw * 4;
    uint32_t *tmp = (uint32_t *)g_malloc(row_bytes);

    for (uint32_t y = ry; y < ry + rh; y++) {
        for (uint32_t x = rx; x < rx + rw; x++) {
            uint64_t offset = color_offset +
                r200_tile_offset(x, y, pitch_pixels, micro_tiled, macro_tiled);
            if (offset + 4 <= vram_size) {
                uint32_t pixel_be = *(uint32_t *)(vram_ptr + offset);
                tmp[x - rx] = __builtin_bswap32(pixel_be);
            } else {
                tmp[x - rx] = 0;
            }
        }
        [st->renderTarget replaceRegion:MTLRegionMake2D(rx, y, rw, 1)
                            mipmapLevel:0
                              withBytes:tmp
                            bytesPerRow:row_bytes];
    }
    g_free(tmp);
}

/*
 * Write back a rectangular region from the Metal output buffer to VRAM
 * with R200 tiled layout and byte-swap (LE BGRA → BE ARGB).
 *
 * micro_tiled: RB3D_COLORPITCH bit 15 — 8×2 micro-tiles
 * macro_tiled: RB3D_COLORPITCH bit 16 — 64×8 macro-tiles
 */
static void metal_writeback_vram_region(PPCMacGPUMetalState *st,
                                         uint8_t *vram_ptr, uint64_t vram_size,
                                         uint32_t color_offset,
                                         uint32_t pitch_pixels,
                                         bool micro_tiled, bool macro_tiled,
                                         uint32_t rt_width,
                                         uint32_t rx, uint32_t ry,
                                         uint32_t rw, uint32_t rh)
{
    if (rw == 0 || rh == 0) return;

    const uint32_t *src = (const uint32_t *)[st->outputBuffer contents];
    if (!src) return;

    static bool logged_tile_mode = false;
    if (!logged_tile_mode) {
        qemu_log("[RT_TILE] writeback_vram_region: micro=%d macro=%d pitch=%u "
                 "off=0x%x region=(%u,%u %ux%u)\n",
                 micro_tiled, macro_tiled, pitch_pixels,
                 color_offset, rx, ry, rw, rh);
        logged_tile_mode = true;
    }

    /*
     * Phase A — Tiling Consistency Audit: log sample pixel addresses
     * from Metal writeback to prove write path + tiling formula used.
     * Only log once per RT offset, for 6 representative pixels.
     */
    static uint32_t tile_audit_rt = 0;
    static bool tile_audit_done = false;
    if (!tile_audit_done && macro_tiled && rw >= 128 && rh >= 32) {
        tile_audit_rt = color_offset;
        tile_audit_done = true;
        /* Sample points: (0,0), (63,0), (64,0), (0,15), (0,16), (100,20) */
        static const uint32_t sample_xy[][2] = {
            {0,0}, {63,0}, {64,0}, {0,15}, {0,16}, {100,20}
        };
        for (int si = 0; si < 6; si++) {
            uint32_t sx = sample_xy[si][0], sy = sample_xy[si][1];
            if (sx < rw && sy < rh) {
                uint64_t off = r200_tile_offset(sx, sy, pitch_pixels,
                                                micro_tiled, macro_tiled);
                uint64_t phys = color_offset + off;
                /* Also compute what the MC formula would give */
                uint32_t mc_pitch_bytes = pitch_pixels * 4;
                uint32_t mc_tiles_per_row = mc_pitch_bytes / 256;
                uint32_t mc_tile_x = sx / 64, mc_tile_y = sy / 16;
                uint32_t mc_in_x = sx % 64, mc_in_y = sy % 16;
                uint64_t mc_tile_idx = (uint64_t)mc_tile_y * mc_tiles_per_row
                                       + mc_tile_x;
                uint64_t mc_phys = color_offset
                                   + mc_tile_idx * (256 * 16)
                                   + (uint64_t)mc_in_y * 256
                                   + (uint64_t)mc_in_x * 4;
                fprintf(stderr,
                    "[TILE_PATH] rt_off=0x%x path=metal_writeback "
                    "tiling_applied=%s formula=metal_macro "
                    "x=%u y=%u pitch=%u "
                    "resolved_addr=0x%llx mc_would_be=0x%llx "
                    "match=%s\n",
                    color_offset,
                    macro_tiled ? "yes" : "no",
                    sx, sy, pitch_pixels,
                    (unsigned long long)phys,
                    (unsigned long long)mc_phys,
                    (phys == mc_phys) ? "yes" : "no");
            }
        }
    }

    /*
     * Clamp writeback x-range to pitch_pixels.  When the Metal RT is wider
     * than color_pitch_pixels (because vertex positions exceed the pitch),
     * writing pixels at x >= pitch_pixels would produce wrong VRAM addresses.
     * Those pixels are still captured by the SRT save path.
     */
    uint32_t wb_rx = rx;
    uint32_t wb_rw = rw;
    if (wb_rx >= pitch_pixels) {
        wb_rw = 0;  /* entirely beyond pitch — skip VRAM writeback */
    } else if (wb_rx + wb_rw > pitch_pixels) {
        wb_rw = pitch_pixels - wb_rx;
    }

    for (uint32_t y = ry; y < ry + rh; y++) {
        const uint32_t *src_row = src + (uint64_t)y * rt_width;
        for (uint32_t x = wb_rx; x < wb_rx + wb_rw; x++) {
            uint64_t offset = color_offset +
                r200_tile_offset(x, y, pitch_pixels, micro_tiled, macro_tiled);
            if (offset + 4 <= vram_size) {
                /* Phase A — VRAM write watch: window texture tile range */
                if (offset >= 0x353000 && offset < 0x413000) {
                    static int wb_watch_log = 0;
                    if (wb_watch_log < 50) {
                        uint32_t pixel_le = src_row[x];
                        fprintf(stderr, "[VRAM_WATCH] metal_writeback "
                                "off=0x%06llx val=0x%08x(LE) x=%u y=%u "
                                "rt_off=0x%x\n",
                                (unsigned long long)offset,
                                pixel_le, x, y, color_offset);
                        wb_watch_log++;
                    }
                }
                uint32_t pixel_le = src_row[x];
                *(uint32_t *)(vram_ptr + offset) = __builtin_bswap32(pixel_le);
            }
        }
    }
}

static int metal_draw_3d_locked(void *opaque, uint8_t *vram_ptr,
                                uint64_t vram_size,
                                const PPCMacGPU3DState *state,
                                const PPCMacGPU3DDrawCmd *cmd);

static int metal_draw_3d(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                          const PPCMacGPU3DState *state,
                          const PPCMacGPU3DDrawCmd *cmd)
{
    int r;

    pthread_mutex_lock(&g_render_lock);
    r = metal_draw_3d_locked(opaque, vram_ptr, vram_size, state, cmd);
    pthread_mutex_unlock(&g_render_lock);
    return r;
}

static int metal_draw_3d_locked(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                          const PPCMacGPU3DState *state,
                          const PPCMacGPU3DDrawCmd *cmd)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st || !st->initialized) {
        qemu_log("METAL_3D: not initialized\n");
        return -1;
    }

    static int metal_draw_count = 0;
    int draw_num = metal_draw_count++;
    bool verbose = r200_diag_on() &&
                   ((draw_num < 50) || (draw_num % 100 == 0));

    /* Phase A — entry-point VRAM check for draws 77-78 */
    if (draw_num == 77 || draw_num == 78) {
        uint32_t entry_val = 0;
        if (0x353000 + 4 <= vram_size) {
            entry_val = *(uint32_t *)(vram_ptr + 0x353000);
        }
        fprintf(stderr, "[ENTRY_CHECK] draw=%d ENTRY vram[0x353000]=0x%08x\n",
                draw_num, entry_val);
    }

    /* Phase B — Run formula parity test on first draw */
    tile_formula_parity_test();

    /* ---- Optional one-shot data path audit (disabled for normal runs) ---- */
    {
        static const bool enable_startup_data_audit = false;
        static bool onetime_audit_done = false;
        if (enable_startup_data_audit && !onetime_audit_done) {
            onetime_audit_done = true;
            qemu_log("\n===== METAL DATA PATH AUDIT START =====\n");
            metal_pixel_format_audit();   /* Phase C */
            metal_roundtrip_test(st, vram_ptr, vram_size);  /* Phase A */
            qemu_log("===== METAL DATA PATH AUDIT END =====\n\n");
        }
    }

    /*
     * Decode blend mode using correct R200 bit positions.
     * R200_SRC_BLEND_SHIFT=16, R200_DST_BLEND_SHIFT=24, mask=0x3F
     */
    uint32_t blend_cntl = state->rb3d_blendcntl;
    uint32_t src_blend = (blend_cntl >> R200_SRC_BLEND_SHIFT) & R200_BLEND_FACTOR_MASK;
    uint32_t dst_blend = (blend_cntl >> R200_DST_BLEND_SHIFT) & R200_BLEND_FACTOR_MASK;
    bool nop_blend = (src_blend == R200_BLEND_GL_ZERO && dst_blend == R200_BLEND_GL_ONE);
    bool is_additive = (src_blend == R200_BLEND_GL_ONE && dst_blend == R200_BLEND_GL_ONE);
    /*
     * RB3D_CNTL bits [3:0] = color format, NOT color write enable.
     * The Apple kext always writes color (rb3d_cntl=0x1800 is normal).
     * Do not skip draws based on bit 0.
     */
    bool color_write_en = true;

    if (verbose) {
        const char *blend_name = nop_blend ? "NOP(zero+one)" :
                                 is_additive ? "ADDITIVE(one+one)" :
                                 (src_blend == R200_BLEND_GL_ONE &&
                                  dst_blend == R200_BLEND_GL_ONE_MINUS_SRC_ALPHA) ?
                                 "PREMUL(one+omsa)" : "other";
        qemu_log("METAL_3D[%d]: draw src=0x%x dst=0x%x blend=%s "
                 "color_wr=%d rb3d_cntl=0x%x pitch_raw=0x%x\n",
                 draw_num, src_blend, dst_blend, blend_name,
                 color_write_en, state->rb3d_cntl, state->rb3d_colorpitch);
    }

    @autoreleasepool {
        /* Decode color buffer params */
        uint32_t color_offset = state->rb3d_coloroffset;
        uint32_t color_pitch_pixels = state->rb3d_colorpitch & 0x3FFF;
        /*
         * Honor macro/micro tile flags from RB3D_COLORPITCH.
         *
         * The guest kext sets macro_tile=1 for render targets.  The 2D BLT
         * engine's MC helpers (mc_vram_read32/write32) apply the same
         * macro-tile swizzle when reading from tiled surfaces that are
         * registered via PITCH_OFFSET registers.  If Metal writes linearly
         * but the 2D BLT reads with tiling, every pixel after the first
         * macro-tile boundary lands at the wrong address → corruption.
         *
         * Fix: use the actual tile flags so Metal's load/writeback layout
         * matches the 2D BLT engine's expectations.
         */
        /*
         * Tiling audit conclusion: MC_TILED_SURFACES_ENABLED=0, so
         * the BLT read path is always linear.  Metal must also write
         * linearly to match.  Tiling will be revisited once the core
         * rendering issues are resolved.
         */
        bool micro_tiled = false;
        bool macro_tiled = false;
        bool color_tiled = false;

        /* Run macro-tile validation once on first tiled draw */
        if (color_tiled) {
            static bool validated = false;
            if (!validated) {
                validated = true;
                qemu_log("[RT_TILE] First tiled draw: pitch=%u micro=%d macro=%d "
                         "pitch_raw=0x%x offset=0x%x\n",
                         color_pitch_pixels, micro_tiled, macro_tiled,
                         state->rb3d_colorpitch, color_offset);
                r200_macrotile_validate(color_pitch_pixels,
                                         micro_tiled, macro_tiled);
            }
        }

        /*
         * Determine render target size.
         * Height comes from the viewport scale: rt_h = 2 * |yscale|.
         *
         * Width: the compositor renders at SCREEN coordinates, so vertices
         * can extend far beyond color_pitch_pixels.  E.g., a window SRT
         * with pitch=832 hosts draws at screen x=145..942.  The Metal RT
         * must be wide enough for the full viewport (screen width), or the
         * rasterizer clips the right side.  Use viewport xoffset + |xscale|
         * to compute the right edge of the drawable area.
         */
        uint32_t rt_width = color_pitch_pixels;
        float vp_yscale = u32_to_float(state->se_vport_yscale);
        uint32_t rt_height = (uint32_t)(2.0f * fabsf(vp_yscale));
        if (rt_height < 16) rt_height = 16;
        if (rt_height > 4096) rt_height = 4096;
        /* Ensure at least as wide as pitch */
        if (rt_width < 16) rt_width = color_pitch_pixels > 0 ? color_pitch_pixels : 1024;
        /*
         * Pre-scan vertex positions to find the maximum x and y.
         * The compositor renders at screen coordinates, so vertices can
         * extend beyond the pitch (e.g., dock at x=949 with pitch=896).
         * The Metal RT must be large enough to rasterize all fragments.
         */
        {
            uint32_t num_verts_prescan = cmd->num_vertices;
            if (num_verts_prescan > 0 && num_verts_prescan <= 65536) {
                float max_vx = 0, max_vy = 0;
                if (cmd->opcode == R200_3D_DRAW_IMMD && cmd->vertex_data) {
                    uint32_t stride_dw = cmd->vb_stride > 0
                        ? cmd->vb_stride / 4
                        : metal_calc_vertex_stride(state) / 4;
                    if (stride_dw < 2) stride_dw = 2;
                    for (uint32_t vi = 0; vi < num_verts_prescan; vi++) {
                        uint32_t base = vi * stride_dw;
                        float vx = u32_to_float(cmd->vertex_data[base]);
                        float vy = u32_to_float(cmd->vertex_data[base + 1]);
                        if (vx > max_vx) max_vx = vx;
                        if (vy > max_vy) max_vy = vy;
                    }
                } else if (cmd->opcode == R200_3D_DRAW_VBUF &&
                           cmd->vb_stride >= 8 &&
                           cmd->vb_addr + (uint64_t)num_verts_prescan *
                               cmd->vb_stride <= vram_size) {
                    const uint32_t *vb = (const uint32_t *)(vram_ptr +
                                                             cmd->vb_addr);
                    uint32_t stride_dw = cmd->vb_stride / 4;
                    for (uint32_t vi = 0; vi < num_verts_prescan; vi++) {
                        float vx = u32_to_float(vb[vi * stride_dw]);
                        float vy = u32_to_float(vb[vi * stride_dw + 1]);
                        if (vx > max_vx) max_vx = vx;
                        if (vy > max_vy) max_vy = vy;
                    }
                }
                uint32_t vert_max_x = (uint32_t)(max_vx + 1.5f);
                uint32_t vert_max_y = (uint32_t)(max_vy + 1.5f);
                if (vert_max_x > rt_width && vert_max_x <= 4096) {
                    rt_width = vert_max_x;
                }
                if (vert_max_y > rt_height && vert_max_y <= 4096) {
                    rt_height = vert_max_y;
                }
            }
        }
        if (rt_width > 4096) rt_width = 4096;
        if (rt_height > 4096) rt_height = 4096;

        /* Ensure render target exists (SRT baseline path). */
        bool using_vram_rt = false;
        if (!metal_ensure_render_target(st, rt_width, rt_height)) {
            qemu_log("METAL_3D[%d]: render target failed %ux%u\n",
                     draw_num, rt_width, rt_height);
            return -1;
        }

        /* Track: RT bound */
        frame_tracker_record(&st->frame_tracker, PASS_EVENT_BIND_RT,
                             color_offset, color_pitch_pixels,
                             0, 0, rt_width, rt_height, 0, 0, false);

        /*
         * Decode texture parameters.
         * The Apple kext writes texture state to R100-compatible registers
         * (0x1C54-0x1C5C) rather than the R200 bank (0x2C00-0x2C14).
         * Prefer R100 registers, fall back to R200. Matches SW renderer's
         * decode_tex_params() logic.
         */
        bool textured = false;
        uint32_t tx_offset = 0, tx_width = 1, tx_height = 1, tx_pitch = 32;
        uint32_t tx_format = 0;

        /*
         * PP_CNTL bit 4 = TEX_0_ENABLE.  When cleared, the R200 hardware
         * does NOT sample a texture — fragment color comes entirely from
         * interpolated vertex color.  Register banks (r100_pp_txoffset_0
         * etc.) may still hold STALE values from a previous draw, so we
         * MUST check the enable bit first.
         */
        bool tex0_hw_enabled = (state->pp_cntl >> 4) & 1;

        if (tex0_hw_enabled) {
            if (state->r100_pp_txoffset_0 != 0) {
                /* R100 register bank (preferred — Apple kext uses these) */
                tx_offset = state->r100_pp_txoffset_0;
                tx_format = state->r100_pp_txformat_0 & 0x1F;
                if (state->r100_pp_txformat_0 & (1 << 6)) {
                    /* Non-power-of-2: size from R100_PP_TEX_SIZE_0 */
                    tx_width = (state->r100_pp_tex_size_0 & 0x7FF) + 1;
                    tx_height = ((state->r100_pp_tex_size_0 >> 16) & 0x7FF) + 1;
                } else {
                    /* Power-of-2: size from format register log2 fields */
                    uint32_t wlog2 = (state->r100_pp_txformat_0 >> 8) & 0xF;
                    uint32_t hlog2 = (state->r100_pp_txformat_0 >> 12) & 0xF;
                    tx_width = 1 << wlog2;
                    tx_height = 1 << hlog2;
                }
                tx_pitch = (state->r100_pp_tex_pitch_0 & 0x3FFF) + 32;
                textured = true;
            } else if ((state->pp_txoffset_0 & ~0x1F) != 0) {
                /* R200 register bank (fallback) */
                tx_offset = state->pp_txoffset_0 & ~0x1F;
                tx_format = state->pp_txformat_0 & 0x1F;
                tx_width = (state->pp_txsize_0 & 0x7FF) + 1;
                tx_height = ((state->pp_txsize_0 >> 16) & 0x7FF) + 1;
                tx_pitch = (state->pp_txpitch_0 & 0x3FFF) + 32;
                textured = true;
            }
        } else {
            /* TEX_0 disabled in PP_CNTL — vertex color only */
            static int tex_disabled_log = 0;
            if (tex_disabled_log < 30) {
                fprintf(stderr, "[TEX0_DISABLED] d=%d pp_cntl=0x%x "
                        "r100_off=0x%x r200_off=0x%x — using vertex color only\n",
                        draw_num, state->pp_cntl,
                        state->r100_pp_txoffset_0,
                        state->pp_txoffset_0);
                tex_disabled_log++;
            }
        }

        id<MTLTexture> texture = nil;
        float tex_w_for_uniform = 0.0f, tex_h_for_uniform = 0.0f;

        /* Phase A — check if texture offset is in GART range.
         * AIC_LO=0x3c00000 AIC_HI=0xbbfffff per guest config.
         * If it is, the texture lives in system RAM, not VRAM. */
        if (textured) {
            static int gart_tex_log = 0;
            if (tx_offset >= 0x3c00000 && tx_offset <= 0xbbfffff) {
                if (gart_tex_log < 20) {
                    fprintf(stderr, "[TEX_GART] draw=%d tex_off=0x%x "
                            "IS IN GART/AIC RANGE — sys RAM texture!\n",
                            draw_num, tx_offset);
                    gart_tex_log++;
                }
            } else if (tx_offset >= 0x10000000 && tx_offset <= 0x17ffffff) {
                if (gart_tex_log < 20) {
                    fprintf(stderr, "[TEX_AGP] draw=%d tex_off=0x%x "
                            "IS IN AGP APERTURE — needs AGP translation!\n",
                            draw_num, tx_offset);
                    gart_tex_log++;
                }
            } else if (tx_offset >= 0x3BF0000) {
                if (gart_tex_log < 20) {
                    fprintf(stderr, "[TEX_GART] draw=%d tex_off=0x%x "
                            "ABOVE MC_FB_TOP (0x3BF0000)!\n",
                            draw_num, tx_offset);
                    gart_tex_log++;
                }
            }
        }

        if (textured) {
            tex_w_for_uniform = (float)tx_width;
            tex_h_for_uniform = (float)tx_height;

            if (verbose) {
                uint32_t txfmt_full = state->r100_pp_txoffset_0 ?
                    state->r100_pp_txformat_0 : state->pp_txformat_0;
                bool tex_micro = (txfmt_full >> 16) & 1;
                bool tex_macro = (txfmt_full >> 17) & 1;
                qemu_log("METAL_3D[%d]: tex off=0x%x %ux%u pitch=%u "
                         "fmt=0x%x txfmt_raw=0x%x tex_micro=%d tex_macro=%d\n",
                         draw_num, tx_offset, tx_width, tx_height,
                         tx_pitch, tx_format, txfmt_full, tex_micro, tex_macro);
            }

            /*
             * Texture content probe: for each unique texture offset, sample
             * pixels to see what content is actually at that VRAM location.
             * This tells us whether the texture contains UI data or is blank.
             */
            {
#define TEX_PROBE_MAX 64
                static struct { uint32_t offset; bool logged; } tex_probes[TEX_PROBE_MAX];
                static int tex_probe_count = 0;
                bool already_probed = false;
                for (int tp = 0; tp < tex_probe_count; tp++) {
                    if (tex_probes[tp].offset == tx_offset) {
                        already_probed = true;
                        break;
                    }
                }
                if (!already_probed && tex_probe_count < TEX_PROBE_MAX) {
                    tex_probes[tex_probe_count].offset = tx_offset;
                    tex_probes[tex_probe_count].logged = true;
                    tex_probe_count++;

                    /* Sample 9 pixels from the texture in a 3x3 grid */
                    uint64_t tex_end = (uint64_t)tx_offset +
                        (uint64_t)(tx_height - 1) * tx_pitch +
                        (uint64_t)tx_width * 4;
                    if (tex_end <= st->vram_size && tx_width > 0 && tx_height > 0) {
                        uint32_t sx[3] = { 0, tx_width / 2, tx_width - 1 };
                        uint32_t sy[3] = { 0, tx_height / 2, tx_height - 1 };
                        fprintf(stderr, "[TEX_PROBE] draw=%d off=0x%x %ux%u "
                                "pitch=%u fmt=0x%x blend=src%x_dst%x "
                                "target=0x%x samples:",
                                draw_num, tx_offset, tx_width, tx_height,
                                tx_pitch, tx_format, src_blend, dst_blend,
                                color_offset);
                        for (int gy = 0; gy < 3; gy++) {
                            for (int gx = 0; gx < 3; gx++) {
                                uint64_t addr = (uint64_t)tx_offset +
                                    (uint64_t)sy[gy] * tx_pitch +
                                    (uint64_t)sx[gx] * 4;
                                if (addr + 4 <= st->vram_size) {
                                    uint32_t pix = *(uint32_t *)(
                                        st->vram_ptr + addr);
                                    fprintf(stderr, " (%u,%u)=0x%08x",
                                            sx[gx], sy[gy], pix);
                                }
                            }
                        }
                        /* Count distinct pixel values */
                        int unique = 0;
                        uint32_t seen[16] = {0};
                        int nonzero = 0;
                        for (uint32_t py = 0; py < tx_height; py += tx_height/4 + 1) {
                            for (uint32_t px = 0; px < tx_width; px += tx_width/4 + 1) {
                                uint64_t addr = (uint64_t)tx_offset +
                                    (uint64_t)py * tx_pitch +
                                    (uint64_t)px * 4;
                                if (addr + 4 <= st->vram_size) {
                                    uint32_t v = *(uint32_t *)(
                                        st->vram_ptr + addr);
                                    if (v != 0) nonzero++;
                                    bool found_v = false;
                                    for (int u = 0; u < unique; u++) {
                                        if (seen[u] == v) { found_v = true; break; }
                                    }
                                    if (!found_v && unique < 16) {
                                        seen[unique++] = v;
                                    }
                                }
                            }
                        }
                        fprintf(stderr, " unique=%d nonzero=%d\n", unique, nonzero);
                    }
                }
            }

            texture = metal_texture_from_vram(st, tx_offset,
                                               tx_width, tx_height,
                                               tx_pitch, tx_format);

            /* Track: texture sampled */
            frame_tracker_record(&st->frame_tracker, PASS_EVENT_SAMPLE_TEX,
                                 tx_offset, tx_pitch / 4,
                                 tx_offset, color_offset,
                                 tx_width, tx_height, 0, 0, false);
        }

        /*
         * Phase E — comprehensive draw dump.
         * Log EVERY draw for the first compositor frame to find window content draws.
         * Compact format: draw#, tex_off, tex_size, blend, target, first 4 VRAM pixels.
         */
        {
            static int full_dump_count = 0;
            if (full_dump_count < 600) {
                uint32_t s0 = 0, s1 = 0;
                if (textured && tx_offset + 4 <= st->vram_size) {
                    s0 = *(uint32_t *)(vram_ptr + tx_offset);
                }
                if (textured && tx_offset + tx_pitch * (tx_height / 2) + 4 <= st->vram_size) {
                    s1 = *(uint32_t *)(vram_ptr + tx_offset + tx_pitch * (tx_height / 2));
                }
                fprintf(stderr, "[DRAW_FULL] d=%d tex=0x%x %ux%u p=%u fmt=%x "
                        "blend=%x_%x tgt=0x%x tp=%u s0=0x%08x smid=0x%08x "
                        "textured=%d\n",
                        draw_num, textured ? tx_offset : 0,
                        textured ? tx_width : 0, textured ? tx_height : 0,
                        textured ? tx_pitch : 0, textured ? tx_format : 0,
                        src_blend, dst_blend, color_offset,
                        color_pitch_pixels,
                        s0, s1, textured);
                full_dump_count++;

                /* Log combiner state for I8 texture draws */
                if (textured && tx_format == R200_TXFORMAT_I8) {
                    fprintf(stderr, "  [I8_COMBINER] d=%d pp_cntl=0x%x "
                            "txcblend_0=0x%x txablend_0=0x%x "
                            "txcblend2_0=0x%x txablend2_0=0x%x "
                            "r100_fmt=0x%x vtx_col=(%.3f,%.3f,%.3f,%.3f)\n",
                            draw_num, state->pp_cntl,
                            state->pp_txcblend_0, state->pp_txablend_0,
                            state->pp_txcblend2_0, state->pp_txablend2_0,
                            state->r100_pp_txformat_0,
                            0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }

        /* Extract vertex data */
        uint32_t num_verts = cmd->num_vertices;
        if (num_verts == 0 || num_verts > 65536) return -1;

        MetalVertex *vertices = g_new0(MetalVertex, num_verts);

        if (cmd->opcode == R200_3D_DRAW_IMMD && cmd->vertex_data) {
            uint32_t stride = cmd->vb_stride > 0 ? cmd->vb_stride
                                                  : metal_calc_vertex_stride(state);
            metal_extract_vertices(state, cmd->vertex_data,
                                    num_verts, stride, vertices,
                                    tex0_hw_enabled);
        } else if (cmd->opcode == R200_3D_DRAW_VBUF) {
            if (cmd->vb_addr + (uint64_t)num_verts * cmd->vb_stride <= vram_size) {
                const uint32_t *vb_data = (const uint32_t *)(vram_ptr + cmd->vb_addr);
                metal_extract_vertices(state, vb_data, num_verts,
                                        cmd->vb_stride, vertices,
                                        tex0_hw_enabled);
            } else {
                g_free(vertices);
                return -1;
            }
        } else {
            g_free(vertices);
            return -1;
        }

        /* Vertex/texcoord dump for first 30 textured draws + Finder/dock draws */
        {
            static int vtx_dump_count = 0;
            static int vtx_dump_finder = 0;
            /* Log Finder (0x900000) and dock (0x300000/896) vertex positions */
            if (num_verts <= 8 && vtx_dump_finder < 20 &&
                ((color_offset == 0x900000 && color_pitch_pixels == 832) ||
                 (color_offset == 0x300000 && color_pitch_pixels == 896))) {
                fprintf(stderr, "[VTX_DUMP_WIN] draw=%d nverts=%u tgt=0x%x tp=%u "
                        "v0=(%.1f,%.1f) v1=(%.1f,%.1f) v2=(%.1f,%.1f) v3=(%.1f,%.1f)\n",
                        draw_num, num_verts, color_offset, color_pitch_pixels,
                        vertices[0].position[0], vertices[0].position[1],
                        num_verts > 1 ? vertices[1].position[0] : 0,
                        num_verts > 1 ? vertices[1].position[1] : 0,
                        num_verts > 2 ? vertices[2].position[0] : 0,
                        num_verts > 2 ? vertices[2].position[1] : 0,
                        num_verts > 3 ? vertices[3].position[0] : 0,
                        num_verts > 3 ? vertices[3].position[1] : 0);
                vtx_dump_finder++;
            }
            if (textured && vtx_dump_count < 30 && num_verts <= 8) {
                fprintf(stderr, "[VTX_DUMP] draw=%d nverts=%u prim=%u "
                        "tex=0x%x blend=src%x_dst%x target=0x%x\n",
                        draw_num, num_verts, cmd->prim_type,
                        tx_offset, src_blend, dst_blend, color_offset);
                for (uint32_t vi = 0; vi < num_verts; vi++) {
                    fprintf(stderr, "  v%u: pos=(%.1f,%.1f) "
                            "tc=(%.3f,%.3f) color=(%.2f,%.2f,%.2f,%.2f)\n",
                            vi,
                            vertices[vi].position[0],
                            vertices[vi].position[1],
                            vertices[vi].texcoord[0],
                            vertices[vi].texcoord[1],
                            vertices[vi].color[0],
                            vertices[vi].color[1],
                            vertices[vi].color[2],
                            vertices[vi].color[3]);
                }
                vtx_dump_count++;
            }
        }

        /* Compute bounding box of draw */
        float min_x = vertices[0].position[0], max_x = min_x;
        float min_y = vertices[0].position[1], max_y = min_y;
        for (uint32_t v = 1; v < num_verts; v++) {
            float vx = vertices[v].position[0], vy = vertices[v].position[1];
            if (vx < min_x) min_x = vx;
            if (vx > max_x) max_x = vx;
            if (vy < min_y) min_y = vy;
            if (vy > max_y) max_y = vy;
        }

        /* Phase C — Gap draw logging: draws 42-69 (between wallpaper and
         * window content). Log EVERY draw in this range regardless of
         * existing TEX_PROBE dedup. */
        if (draw_num >= 42 && draw_num <= 69) {
            const char *role = "unknown";
            if (!textured && nop_blend) role = "clear/fill";
            else if (!textured) role = "solid_quad";
            else if (src_blend == R200_BLEND_GL_ONE &&
                     dst_blend == R200_BLEND_GL_ONE_MINUS_SRC_ALPHA)
                role = "premul_tex";
            else if (src_blend == R200_BLEND_GL_ONE &&
                     dst_blend == R200_BLEND_GL_ONE) role = "additive_tex";
            else if (nop_blend && textured) role = "opaque_tex";
            else role = "other_tex";

            /* Sample 4 corners of texture if textured */
            uint32_t corner[4] = {0, 0, 0, 0};
            if (textured && tx_offset + 4 <= st->vram_size) {
                uint64_t addrs[4] = {
                    (uint64_t)tx_offset,
                    (uint64_t)tx_offset + ((uint64_t)(tx_width - 1)) * 4,
                    (uint64_t)tx_offset + ((uint64_t)(tx_height - 1)) * tx_pitch,
                    (uint64_t)tx_offset + ((uint64_t)(tx_height - 1)) * tx_pitch +
                        ((uint64_t)(tx_width - 1)) * 4
                };
                for (int ci = 0; ci < 4; ci++) {
                    if (addrs[ci] + 4 <= st->vram_size)
                        corner[ci] = *(uint32_t *)(st->vram_ptr + addrs[ci]);
                }
            }

            fprintf(stderr,
                "[GAP_DRAW] draw=%d tex_off=0x%x tex=%ux%u "
                "bbox=(%.0f,%.0f)-(%.0f,%.0f) blend=src%x_dst%x "
                "role=%s corners=0x%08x,0x%08x,0x%08x,0x%08x "
                "target=0x%x\n",
                draw_num, textured ? tx_offset : 0,
                textured ? tx_width : 0, textured ? tx_height : 0,
                min_x, min_y, max_x, max_y,
                src_blend, dst_blend, role,
                corner[0], corner[1], corner[2], corner[3],
                color_offset);
        }

        /* Phase D — VRAM tile range re-check at key draw checkpoints.
         * Sample the window texture range 0x353000-0x413000 at multiple
         * points to detect if data appears later in the frame. */
        {
            static const int checkpoints[] = {0, 30, 50, 70, 71, 72, 73,
                                               74, 75, 76, 77, 78, 90, -1};
            for (int ci = 0; checkpoints[ci] >= 0; ci++) {
                if (draw_num == checkpoints[ci]) {
                    int nonzero_count = 0;
                    uint32_t first_nz_off = 0, first_nz_val = 0;
                    for (uint64_t a = 0x353000; a < 0x413000; a += 256) {
                        if (a + 4 <= st->vram_size) {
                            uint32_t v = *(uint32_t *)(st->vram_ptr + a);
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
                        "[TILE_CHECK] draw=%d range=0x353000-0x413000 "
                        "nonzero_samples=%d/%d first_nz=0x%x(0x%08x)\n",
                        draw_num, nonzero_count,
                        (int)((0x413000 - 0x353000) / 256),
                        first_nz_off, first_nz_val);
                    break;
                }
            }
        }

        /*
         * Expand render target if vertices extend beyond viewport-derived size.
         *
         * The viewport scale/offset define a coordinate MAPPING, not the RT
         * dimensions.  The Apple QE kext provides pre-transformed screen
         * coordinates even when VTX_XY_FMT=0, and the RT may be much larger
         * than 2*|yscale|.  If the vertex positions exceed our current RT
         * dimensions, grow the RT to fit them.
         *
         * Width expansion: The QE compositor renders menu bar extras and other
         * small surfaces at screen coordinates (e.g., x=882 for a Spotlight
         * icon) into an RT whose pitch is only 192 pixels.  On real R200 HW,
         * the viewport transform would map these to local tile coordinates.
         * Since we use pre-transformed screen coords, the Metal RT must be
         * wide enough to contain them.  The VRAM writeback clamps x to
         * color_pitch_pixels so it won't corrupt VRAM rows.  The SRT save
         * stores the full rendered region for correct presentation.
         */
        {
            /*
             * Width expansion: Only expand the Metal RT width when vertices
             * are genuinely at screen coordinates that exceed the pitch.
             * This happens for small-pitch surfaces (e.g., pitch=192 for
             * menu bar extras rendered at x=882-1011).
             *
             * For surfaces where pitch >= screen width (e.g., pitch=1024),
             * vertices near x=1024 are just at the edge and should be
             * clamped, not expanded.  Expanding width beyond pitch causes
             * SRT stride mismatch: the SRT grows wider than the pitch,
             * making row reads drift by (width - pitch) per row.
             *
             * Heuristic: only expand width if max_x exceeds pitch by
             * a significant margin (>= 2x pitch), indicating true
             * screen-coordinate rendering into a small tile surface.
             */
            uint32_t need_w = rt_width;
            if (max_x > 0) {
                uint32_t mx = (uint32_t)(max_x + 2);
                if (mx > rt_width && mx > color_pitch_pixels * 2) {
                    /* Vertices far exceed pitch — screen-coordinate rendering
                     * into a small tile surface (e.g., menu bar extras) */
                    need_w = mx;
                }
            }
            uint32_t need_h = (max_y > 0) ? (uint32_t)(max_y + 2) : rt_height;
            if (need_w > 4096) need_w = 4096;
            if (need_h > 4096) need_h = 4096;
            if (need_w > rt_width || need_h > rt_height) {
                uint32_t new_w = (need_w > rt_width) ? need_w : rt_width;
                uint32_t new_h = (need_h > rt_height) ? need_h : rt_height;
                static int rt_expand_log = 0;
                if (rt_expand_log < 40) {
                    qemu_log("METAL_3D[%d]: RT expand %ux%u -> %ux%u "
                             "(vertices max=%.0f,%.0f pitch=%u vp_yscale=%.1f)\n",
                             draw_num, rt_width, rt_height,
                             new_w, new_h, max_x, max_y,
                             color_pitch_pixels, vp_yscale);
                    rt_expand_log++;
                }
                rt_width = new_w;
                rt_height = new_h;
                if (!metal_ensure_render_target(st, rt_width, rt_height)) {
                    qemu_log("METAL_3D[%d]: RT expand failed %ux%u\n",
                             draw_num, rt_width, rt_height);
                    g_free(vertices);
                    return -1;
                }
            }
        }

        /* Clamp to render target bounds */
        uint32_t bb_x = (min_x < 0) ? 0 : (uint32_t)min_x;
        uint32_t bb_y = (min_y < 0) ? 0 : (uint32_t)min_y;
        uint32_t bb_x2 = (max_x >= rt_width) ? rt_width : (uint32_t)(max_x + 1);
        uint32_t bb_y2 = (max_y >= rt_height) ? rt_height : (uint32_t)(max_y + 1);
        uint32_t bb_w = (bb_x2 > bb_x) ? bb_x2 - bb_x : 0;
        uint32_t bb_h = (bb_y2 > bb_y) ? bb_y2 - bb_y : 0;

        /* Log shadow draws (premul alpha) to the final composite SRT
         * to diagnose the dark vertical bar at cols 0-45. */
        if (src_blend == R200_BLEND_GL_ONE &&
            dst_blend == R200_BLEND_GL_ONE_MINUS_SRC_ALPHA &&
            color_pitch_pixels == 1024 && bb_x < 100) {
            static int shadow_bar_log = 0;
            if (shadow_bar_log < 100) {
                fprintf(stderr,
                    "[SHADOW_BAR] d=%d off=0x%x bbox=(%u,%u %ux%u) "
                    "tex=0x%x %ux%u "
                    "v0=(%.1f,%.1f) v1=(%.1f,%.1f) "
                    "v0_col=(%.3f,%.3f,%.3f,%.3f)\n",
                    draw_num, color_offset,
                    bb_x, bb_y, bb_w, bb_h,
                    textured ? tx_offset : 0,
                    textured ? tx_width : 0, textured ? tx_height : 0,
                    vertices[0].position[0], vertices[0].position[1],
                    num_verts > 1 ? vertices[1].position[0] : 0.0f,
                    num_verts > 1 ? vertices[1].position[1] : 0.0f,
                    vertices[0].color[0], vertices[0].color[1],
                    vertices[0].color[2], vertices[0].color[3]);
                shadow_bar_log++;
            }
        }

        /* Phase A: Log draws targeting small-pitch surfaces to understand
         * vertex positions vs RT bounds */
        if (color_pitch_pixels <= 256) {
            static int small_pitch_log = 0;
            if (small_pitch_log < 80) {
                fprintf(stderr,
                    "[SMALL_PITCH_DRAW] d=%d off=0x%x pitch=%u "
                    "verts=(%.1f,%.1f)-(%.1f,%.1f) "
                    "rt=%ux%u bb=(%u,%u %ux%u) "
                    "vp_yscale=%.1f\n",
                    draw_num, color_offset, color_pitch_pixels,
                    min_x, min_y, max_x, max_y,
                    rt_width, rt_height,
                    bb_x, bb_y, bb_w, bb_h,
                    vp_yscale);
                small_pitch_log++;
            }
        }

        /* Menu bar draw logging — trace vertex positions for draws with
         * 48x28 or 256x28 textures (Aqua menu bar gradient tiles + overlay) */
        if (textured && tx_height == 28 && (tx_width == 48 || tx_width == 256)) {
            static int mb_log = 0;
            if (mb_log < 60) {
                fprintf(stderr,
                    "[MENUBAR_DRAW] d=%d tex=0x%x %ux%u p=%u "
                    "verts=(%.1f,%.1f)-(%.1f,%.1f) "
                    "bb=(%u,%u %ux%u) blend=%x_%x "
                    "v0_tc=(%.2f,%.2f) v0_col=(%.3f,%.3f,%.3f,%.3f)\n",
                    draw_num,
                    textured ? tx_offset : 0,
                    textured ? tx_width : 0, textured ? tx_height : 0,
                    textured ? tx_pitch : 0,
                    min_x, min_y, max_x, max_y,
                    bb_x, bb_y, bb_w, bb_h,
                    src_blend, dst_blend,
                    vertices[0].texcoord[0], vertices[0].texcoord[1],
                    vertices[0].color[0], vertices[0].color[1],
                    vertices[0].color[2], vertices[0].color[3]);
                mb_log++;

                /* Dump combiner state + vertices for 48x28 overlay draws */
                if (textured && tx_width == 48) {
                    fprintf(stderr,
                        "  [COMBINER] pp_cntl=0x%x pp_txcblend_0=0x%x "
                        "pp_txablend_0=0x%x pp_txcblend2_0=0x%x "
                        "pp_txablend2_0=0x%x\n"
                        "  fmt0=0x%x fmt1=0x%x stride=%u "
                        "se_vte_cntl=0x%x se_cntl=0x%x re_cntl=0x%x\n",
                        state->pp_cntl, state->pp_txcblend_0,
                        state->pp_txablend_0, state->pp_txcblend2_0,
                        state->pp_txablend2_0,
                        state->se_vtx_fmt_0, state->se_vtx_fmt_1,
                        cmd->vb_stride,
                        state->se_vte_cntl, state->se_cntl, state->re_cntl);
                    for (uint32_t vi = 0; vi < num_verts && vi < 4; vi++) {
                        fprintf(stderr,
                            "  v%u: pos=(%.1f,%.1f) tc=(%.2f,%.2f) "
                            "col=(%.3f,%.3f,%.3f,%.3f)\n",
                            vi,
                            vertices[vi].position[0],
                            vertices[vi].position[1],
                            vertices[vi].texcoord[0],
                            vertices[vi].texcoord[1],
                            vertices[vi].color[0],
                            vertices[vi].color[1],
                            vertices[vi].color[2],
                            vertices[vi].color[3]);
                    }
                    fprintf(stderr, "  tex_size=(%f,%f) viewport_size=(%u,%u)\n",
                            (float)tx_width, (float)tx_height,
                            rt_width, rt_height);
                    /* Dump first row of texture at multiple columns */
                    fprintf(stderr, "[MB_TEX_DUMP] d=%d tex=0x%x %ux%u pitch=%u:\n",
                            draw_num, tx_offset, tx_width, tx_height, tx_pitch);
                    for (uint32_t col = 0; col < tx_width && col < 48; col += 4) {
                        uint64_t addr = (uint64_t)tx_offset + (uint64_t)col * 4;
                        if (addr + 4 <= st->vram_size) {
                            uint32_t raw = *(uint32_t *)(vram_ptr + addr);
                            uint8_t a = raw & 0xFF;
                            uint8_t r = (raw >> 8) & 0xFF;
                            fprintf(stderr, "  row0,col%2u: A=%3u R=%3u  ",
                                    col, a, r);
                        }
                    }
                    fprintf(stderr, "\n");
                    /* Dump column 0 and column 24 at each row */
                    for (uint32_t row = 0; row < tx_height && row < 28; row++) {
                        uint64_t addr0 = (uint64_t)tx_offset +
                                         (uint64_t)row * tx_pitch;
                        uint64_t addr24 = addr0 + 24 * 4;
                        uint32_t raw0 = 0, raw24 = 0;
                        if (addr0 + 4 <= st->vram_size)
                            raw0 = *(uint32_t *)(vram_ptr + addr0);
                        if (addr24 + 4 <= st->vram_size)
                            raw24 = *(uint32_t *)(vram_ptr + addr24);
                        uint8_t a0 = raw0 & 0xFF, r0 = (raw0>>8) & 0xFF;
                        uint8_t a24 = raw24 & 0xFF, r24 = (raw24>>8) & 0xFF;
                        fprintf(stderr, "  row%2u: col0 A=%3u R=%3u | "
                                "col24 A=%3u R=%3u\n",
                                row, a0, r0, a24, r24);
                    }
                }
            }
        }

        if (verbose) {
            qemu_log("METAL_3D[%d]: prim=%u nverts=%u textured=%d "
                     "tiled=%d(micro=%d,macro=%d) "
                     "v0=(%.1f,%.1f) bbox=(%u,%u %ux%u) blend src=0x%x dst=0x%x\n",
                     draw_num, cmd->prim_type, num_verts, textured,
                     color_tiled, micro_tiled, macro_tiled,
                     vertices[0].position[0], vertices[0].position[1],
                     bb_x, bb_y, bb_w, bb_h, src_blend, dst_blend);
        }

        /* ---- Phase B + E: Pixel-level audit for first relevant draws ---- */
        {
            /* Phase E: First additive (wallpaper) textured draw */
            static bool wallpaper_audit_done = false;
            if (!wallpaper_audit_done && textured && is_additive &&
                bb_w > 4 && bb_h > 4) {
                wallpaper_audit_done = true;
                metal_wallpaper_texture_audit(vram_ptr, vram_size,
                    tx_offset, tx_pitch, tx_width, tx_height, tx_format);
                metal_draw_pixel_audit(st, vram_ptr, vram_size,
                    color_offset, color_pitch_pixels,
                    tx_offset, tx_pitch, tx_width, tx_height,
                    bb_x, bb_y, bb_w, bb_h,
                    src_blend, dst_blend, draw_num, "wallpaper");
            }

            /* Phase B: First premul-alpha (window) textured draw */
            static bool window_audit_done = false;
            bool is_premul = (src_blend == R200_BLEND_GL_ONE &&
                              dst_blend == R200_BLEND_GL_ONE_MINUS_SRC_ALPHA);
            if (!window_audit_done && textured && is_premul &&
                bb_w > 4 && bb_h > 4) {
                window_audit_done = true;
                metal_draw_pixel_audit(st, vram_ptr, vram_size,
                    color_offset, color_pitch_pixels,
                    tx_offset, tx_pitch, tx_width, tx_height,
                    bb_x, bb_y, bb_w, bb_h,
                    src_blend, dst_blend, draw_num, "window");
            }
        }

        /*
         * --- Representative Draw Audit (Phase A-B) ---
         * Emit detailed state for first 10 nop=1+color_wr=1 draws,
         * then every 200th, to support forensic analysis.
         */
        {
            static int audit_count = 0;
            bool do_audit = nop_blend && color_write_en &&
                (audit_count < 10 || audit_count % 200 == 0);
            if (do_audit) {
                /* Decode RB3D_BLENDCNTL fully */
                uint32_t bc = state->rb3d_blendcntl;
                uint32_t c_src = bc & 0xF;
                uint32_t c_dst = (bc >> 16) & 0xF;
                bool blend_en = (bc >> 24) & 1;
                bool sep_alpha = (bc >> 26) & 1;
                /* If separate alpha: alpha factors may be at [7:4] and [23:20]
                 * per R200 convention, OR in RB3D_ABLENDCNTL register */
                uint32_t a_src_inline = (bc >> 4) & 0xF;
                uint32_t a_dst_inline = (bc >> 20) & 0xF;
                uint32_t ablend = state->rb3d_ablendcntl;
                uint32_t cblend = state->rb3d_cblendcntl;
                uint32_t a_src_reg = ablend & 0xF;
                uint32_t a_dst_reg = (ablend >> 16) & 0xF;

                /* Check degenerate geometry */
                bool degenerate = (bb_w == 0 || bb_h == 0 ||
                                   (bb_w == 1 && bb_h == 1));

                /* Texture state (use already-decoded values) */
                uint32_t tex_off = tx_offset;
                uint32_t tex_w = tx_width;
                uint32_t tex_h = tx_height;
                uint32_t tex_pitch_a = tx_pitch;
                uint32_t tex_fmt = tx_format;

                /* RB3D_CNTL decode */
                uint32_t color_write_mask = state->rb3d_cntl & 0xF;

                /* Phase A: pick */
                qemu_log("[REP_DRAW_PICK] draw_id=%d signature=nop_cw1_blend%08x "
                         "target_off=0x%x target_pitch=%u prim=%u "
                         "blend=0x%08x color_write=1 textured=%d "
                         "vertex_count=%u bbox=(%u,%u)-(%u,%u) count_seen=%d\n",
                         draw_num, bc, color_offset, color_pitch_pixels,
                         cmd->prim_type, bc, textured, num_verts,
                         bb_x, bb_y, bb_x + bb_w, bb_y + bb_h, audit_count);

                /* Phase B: full state decode */
                qemu_log("[REP_DRAW_STATE] draw_id=%d "
                         "signature=nop_cw1_blend%08x "
                         "target_off=0x%x target_pitch=%u "
                         "macro=%d micro=%d "
                         "prim=%u vertex_count=%u "
                         "bbox=(%u,%u)-(%u,%u) degenerate=%d "
                         "texture_en=%d tex0_off=0x%x tex0_pitch=%u "
                         "tex0_size=%ux%u tex0_format=0x%x "
                         "blend_raw=0x%08x "
                         "blend_en=%d sep_alpha=%d "
                         "c_src=0x%x c_dst=0x%x "
                         "a_src_inline=0x%x a_dst_inline=0x%x "
                         "ablendcntl_reg=0x%08x a_src_reg=0x%x a_dst_reg=0x%x "
                         "cblendcntl_reg=0x%08x "
                         "color_write_mask=0x%x "
                         "rb3d_cntl=0x%x "
                         "note=color_eq_dst_passthrough "
                         "alpha_note=%s\n",
                         draw_num, bc,
                         color_offset, color_pitch_pixels,
                         macro_tiled, micro_tiled,
                         cmd->prim_type, num_verts,
                         bb_x, bb_y, bb_x + bb_w, bb_y + bb_h, degenerate,
                         textured, tex_off, tex_pitch_a,
                         tex_w, tex_h, tex_fmt,
                         bc, blend_en, sep_alpha,
                         c_src, c_dst,
                         a_src_inline, a_dst_inline,
                         ablend, a_src_reg, a_dst_reg,
                         cblend,
                         color_write_mask,
                         state->rb3d_cntl,
                         sep_alpha ? "separate_alpha_factors_may_differ" :
                                     "same_as_color");
            }
            if (nop_blend && color_write_en) audit_count++;
        }

        /*
         * ALL draws (including NOP blends) go through Metal.
         *
         * NOP blend (src=ZERO, dst=ONE) uses pipelineStateNopBlend which
         * preserves destination unchanged — Metal still exercises the full
         * data path (load VRAM → render → writeback).
         *
         * Non-NOP draws use pipelineState with srcAlpha/oneMinusSrcAlpha.
         */
        if (bb_w == 0 || bb_h == 0 || !color_write_en) {
            /* Empty bbox or no color write — nothing to render */
            g_free(vertices);
            if (verbose) {
                qemu_log("[METAL_DRAW] %d: skip (bb=%ux%u color_wr=%d)\n",
                         draw_num, bb_w, bb_h, color_write_en);
            }
        } else {
            /* Draw counter for summary */
            static int metal_exec_count = 0;
            static int metal_nop_count = 0;
            static int metal_real_count = 0;
            static int probe_mismatch_total = 0;
            metal_exec_count++;
            if (nop_blend) metal_nop_count++;
            else metal_real_count++;

            /* Per-offset draw count tracking */
            {
#define OFFSET_TRACK_MAX 32
                static struct { uint32_t offset; int count; } offset_counts[OFFSET_TRACK_MAX];
                static int offset_count_used = 0;
                int found_ot = -1;
                for (int ot = 0; ot < offset_count_used; ot++) {
                    if (offset_counts[ot].offset == color_offset) {
                        found_ot = ot;
                        break;
                    }
                }
                if (found_ot >= 0) {
                    offset_counts[found_ot].count++;
                } else if (offset_count_used < OFFSET_TRACK_MAX) {
                    offset_counts[offset_count_used].offset = color_offset;
                    offset_counts[offset_count_used].count = 1;
                    offset_count_used++;
                }
                /* Log summary at draws 100, 500, 1000 */
                if (draw_num == 100 || draw_num == 500 || draw_num == 1000) {
                    fprintf(stderr, "[DRAW_SUMMARY] at draw %d: %d total "
                            "(%d nop, %d real)\n",
                            draw_num, metal_exec_count,
                            metal_nop_count, metal_real_count);
                    for (int ot = 0; ot < offset_count_used; ot++) {
                        fprintf(stderr, "  off=0x%x: %d draws\n",
                                offset_counts[ot].offset,
                                offset_counts[ot].count);
                    }
                }
            }

            /*
             * Phase 3A: Probe surface validation.
             * For NOP-blend draws (src=ZERO, dst=ONE), the target surface
             * should be IDENTICAL before and after the draw+writeback cycle.
             * Compute CRC before loading into Metal.
             */
            uint32_t before_crc = 0;
            bool do_probe_check = nop_blend && r200_diag_on() &&
                (metal_exec_count <= 50 || metal_exec_count % 200 == 0);
            if (do_probe_check) {
                before_crc = compute_vram_region_crc(
                    vram_ptr, vram_size, color_offset, color_pitch_pixels,
                    micro_tiled, macro_tiled, bb_x, bb_y, bb_w, bb_h);
            }

            /*
             * Pitch-change detection: check if this RT offset was last
             * written at a different pitch.  On real R200, MC tiling makes
             * different pitches access different physical locations.
             * Without tiling, we clear the RT to avoid cross-contamination.
             */
            bool pitch_conflict = false;
            {
                int found = -1;
                int oldest = 0;
                for (int i = 0; i < RT_PITCH_CACHE_SIZE; i++) {
                    if (st->rt_pitch_cache[i].valid &&
                        st->rt_pitch_cache[i].offset == color_offset) {
                        found = i;
                        break;
                    }
                    if (!st->rt_pitch_cache[i].valid) oldest = i;
                }
                if (found >= 0) {
                    if (st->rt_pitch_cache[found].pitch != color_pitch_pixels) {
                        pitch_conflict = true;
                        /* Pitch change at same offset → new compositor pass.
                         * Bump tracker pass counter. */
                        frame_tracker_next_pass(&st->frame_tracker);
                        static int pc_log = 0;
                        if (pc_log < 20) {
                            qemu_log("METAL_3D[%d]: pitch conflict at 0x%x: "
                                     "was %u now %u - clearing RT\n",
                                     draw_num, color_offset,
                                     st->rt_pitch_cache[found].pitch,
                                     color_pitch_pixels);
                            pc_log++;
                        }
                        st->rt_pitch_cache[found].pitch = color_pitch_pixels;

                        /* Mark ALL shadow RTs at this offset as conflicted.
                         * This tells metal_blit_2d to prefer the shadow RT
                         * over VRAM, since VRAM has garbled pitch-mixed data.
                         *
                         * Do NOT clear the SRT content — the compositor does
                         * dirty-rect compositing and content must persist
                         * across pitch changes within a frame. */
                        for (int si = 0; si < SHADOW_RT_MAX; si++) {
                            if (st->shadow_rts[si].valid &&
                                st->shadow_rts[si].offset == color_offset) {
                                st->shadow_rts[si].has_conflict = true;
                            }
                        }
                    }
                } else {
                    /* New entry */
                    st->rt_pitch_cache[oldest].offset = color_offset;
                    st->rt_pitch_cache[oldest].pitch = color_pitch_pixels;
                    st->rt_pitch_cache[oldest].valid = true;
                }
            }

            /*
             * Load affected region into the Metal render target.
             *
             * Priority:
             * 1. Shadow RT (if one exists for this offset+pitch) — correct
             *    accumulated content from previous draws, immune to VRAM
             *    pitch conflicts.
             * 2. VRAM (if no pitch conflict) — first draw to this RT.
             * 3. Clear to zero (pitch conflict, no shadow) — avoid garbled
             *    cross-contamination.
             */
            /* Clamp bbox to actual RT dimensions to prevent AGX OOB */
            if (bb_x >= st->rt_width || bb_y >= st->rt_height) {
                bb_w = 0; bb_h = 0;
            } else {
                if (bb_x + bb_w > st->rt_width) {
                    static int bb_clamp_log = 0;
                    if (bb_clamp_log < 10) {
                        qemu_log("[RT_CLAMP] draw=%d bb_x=%u bb_w=%u > "
                                 "rt_w=%u, clamping\n",
                                 draw_num, bb_x, bb_w, st->rt_width);
                        bb_clamp_log++;
                    }
                    bb_w = st->rt_width - bb_x;
                }
                if (bb_y + bb_h > st->rt_height) {
                    bb_h = st->rt_height - bb_y;
                }
            }
            if (using_vram_rt) {
                /* VRAM RT path: skip all SRT operations.
                 * The texture aliases VRAM directly — no need to load from
                 * SRT before drawing or save to SRT after drawing.
                 * MTLLoadActionLoad naturally preserves existing VRAM content
                 * (including CPU-written window body pixels). */
            } else {
                int srt_slot = shadow_rt_find(st, color_offset,
                                               color_pitch_pixels);
                /*
                 * Frame lifecycle: when a draw targets a consumed SRT, this
                 * signals the start of a new compositor frame for this render
                 * target.  Clear the SRT pixel buffer AND bump rt_frame_gen
                 * to invalidate the Metal RT cache entry.  This forces a new
                 * Metal texture (starting with zeroed pixels) instead of
                 * reusing the old texture that has stale frame content.
                 *
                 * Also clear ALL consumed SRTs at the same VRAM offset,
                 * since a frame boundary affects all pitch variants at that
                 * offset (wallpaper, shadows, window body all get fresh).
                 */
                if (srt_slot >= 0 && st->shadow_rts[srt_slot].consumed &&
                    st->shadow_rts[srt_slot].pixels) {
                    /*
                     * Do not preserve the actively redrawn QE render target
                     * across frames. Window drag proved that Tiger is still
                     * issuing fresh 3D draws for frame/shadow layers on each
                     * step, so preserving these consumed SRTs causes every
                     * new pass to accumulate on top of the old one.
                     *
                     * A true "stale surface" move needs a separate backing
                     * that is not simultaneously serving as the destination of
                     * new 3D draws. This frame-boundary clear path is the
                     * wrong place to preserve it.
                     */
                    bool preserve_drag_srt = false;
                    /*
                     * Frame boundary: clear ALL consumed SRTs at this
                     * offset (original behavior).  The compositor redraws
                     * the full scene each frame; clearing ensures no stale
                     * content persists.
                     */
                    /* Sample BEFORE clear to see what the compositor drew */
                    uint32_t nz_before = 0;
                    uint32_t num_consumed = 0;
                    {
                        uint32_t sw2 = st->shadow_rts[srt_slot].width;
                        uint32_t sh2 = st->shadow_rts[srt_slot].height;
                        const uint32_t *sp = st->shadow_rts[srt_slot].pixels;
                        if (sp && sw2 > 0 && sh2 > 200) {
                            /* Sample row at y=300 (middle of window area) */
                            for (uint32_t x = 0; x < sw2; x++)
                                if (sp[300 * sw2 + x]) nz_before++;
                        }
                        for (int si = 0; si < SHADOW_RT_MAX; si++)
                            if (st->shadow_rts[si].valid &&
                                st->shadow_rts[si].offset == color_offset &&
                                st->shadow_rts[si].consumed)
                                num_consumed++;
                    }
                    for (int si = 0; si < SHADOW_RT_MAX; si++) {
                        if (st->shadow_rts[si].valid &&
                            st->shadow_rts[si].offset == color_offset &&
                            st->shadow_rts[si].consumed &&
                            st->shadow_rts[si].pixels) {
                            if (preserve_drag_srt) {
                                st->shadow_rts[si].stale = true;
                            } else {
                                memset(st->shadow_rts[si].pixels, 0,
                                       (size_t)st->shadow_rts[si].alloc_count * 4);
                                st->shadow_rts[si].stale = false;
                            }
                            st->shadow_rts[si].consumed = false;
                        }
                    }
                    st->rt_frame_gen++;
                    {
                        static int srt_pre_clear_log = 0;
                        if (srt_pre_clear_log < 300) {
                            fprintf(stderr,
                                "[SRT_FRAME_CLEAR] slot=%d off=0x%x "
                                "pitch=%u d=%d gen=%u bb=(%u,%u %ux%u) "
                                "srt=%ux%u nz_before_y300=%u consumed=%u "
                                "preserve=%d\n",
                                srt_slot, color_offset,
                                color_pitch_pixels, draw_num,
                                st->rt_frame_gen,
                                bb_x, bb_y, bb_w, bb_h,
                                st->shadow_rts[srt_slot].width,
                                st->shadow_rts[srt_slot].height,
                                nz_before, num_consumed,
                                preserve_drag_srt);
                            srt_pre_clear_log++;
                        }
                    }

                    /* Snapshot workaround removed — needs real
                     * IOFBBlitSurfaceCopy implementation. */
                }
                if (srt_slot >= 0 && st->shadow_rts[srt_slot].pixels) {
                    /* Load from shadow RT — BE pixels, stride = srt.width */
                    uint32_t sw = st->shadow_rts[srt_slot].width;
                    uint32_t sh = st->shadow_rts[srt_slot].height;
                    const uint32_t *srt_pixels =
                        st->shadow_rts[srt_slot].pixels;

                    /* Diagnostic: log SRT state for draws targeting
                     * 1024-pitch SRTs with bbox overlapping (0,0)-(260,60)
                     * (the cyan square area) */
                    {
                        static int srt_cyan_log = 0;
                        if (srt_cyan_log < 300 &&
                            color_pitch_pixels >= 832 &&
                            bb_x < 260 && bb_y < 60) {
                            uint32_t p = 0;
                            if (bb_y < sh && bb_x < sw)
                                p = srt_pixels[(uint64_t)bb_y * sw + bb_x];
                            /* Also sample pixel at (128,30) if in range */
                            uint32_t p2 = 0;
                            if (30 < sh && 128 < sw &&
                                128 >= bb_x && 128 < bb_x + bb_w &&
                                30 >= bb_y && 30 < bb_y + bb_h)
                                p2 = srt_pixels[(uint64_t)30 * sw + 128];
                            fprintf(stderr,
                                "[SRT_CYAN_CHECK] d=%d off=0x%x pitch=%u "
                                "blend=%u/%u consumed=%d "
                                "bb=(%u,%u %ux%u) "
                                "px[%u,%u]=0x%08x px[128,30]=0x%08x%s\n",
                                draw_num, color_offset, color_pitch_pixels,
                                src_blend, dst_blend,
                                st->shadow_rts[srt_slot].consumed,
                                bb_x, bb_y, bb_w, bb_h,
                                bb_x, bb_y, p, p2,
                                p ? " STALE" : "");
                            srt_cyan_log++;
                        }
                    }

                    /* Convert BE → LE for Metal texture.
                     *
                     * For ADDITIVE draws (ONE+ONE): load ZEROS.  Additive
                     * blending adds src to dst — loading previous frame's
                     * SRT content would cause wallpaper to accumulate across
                     * frames (doubling brightness each frame → cyan washout).
                     *
                     * For non-additive draws (ONE+OMSA, etc.): load SRT
                     * content normally.  These draws blend/replace over
                     * existing content without accumulation.
                     *
                     * The SRT pixel buffer is NOT cleared — it retains
                     * previous frame content for PRESENT_BLIT reads of
                     * un-redrawn layers (e.g., window body during drag).
                     */
                    bool load_zeros = is_additive;
                    uint32_t row_bytes = bb_w * 4;
                    uint32_t *row_buf = (uint32_t *)g_malloc(row_bytes);
                    for (uint32_t y = bb_y; y < bb_y + bb_h; y++) {
                        if (load_zeros || y >= sh) {
                            /* Additive draw or beyond SRT — clean canvas */
                            memset(row_buf, 0, row_bytes);
                        } else {
                            const uint32_t *srt_row = srt_pixels +
                                                      (uint64_t)y * sw;
                            for (uint32_t x = 0; x < bb_w; x++) {
                                uint32_t sx = bb_x + x;
                                if (sx < sw) {
                                    /* Shadow RT is BE, Metal wants LE */
                                    row_buf[x] =
                                        __builtin_bswap32(srt_row[sx]);
                                } else {
                                    row_buf[x] = 0;
                                }
                            }
                        }
                        [st->renderTarget
                            replaceRegion:MTLRegionMake2D(bb_x, y, bb_w, 1)
                              mipmapLevel:0
                                withBytes:row_buf
                              bytesPerRow:row_bytes];
                    }
                    g_free(row_buf);

                    static int srt_load_log = 0;
                    if (srt_load_log < 10) {
                        qemu_log("[SHADOW_RT] load draw=%d "
                                 "off=0x%x pitch=%u "
                                 "bbox=(%u,%u %ux%u) "
                                 "from shadow slot=%d\n",
                                 draw_num, color_offset,
                                 color_pitch_pixels,
                                 bb_x, bb_y, bb_w, bb_h,
                                 srt_slot);
                        srt_load_log++;
                    }
                } else {
                    /*
                     * No shadow RT exists yet — this is the first 3D draw
                     * to this (offset, pitch) combination.
                     *
                     * Decision: clear vs load VRAM based on blend mode and
                     * whether existing SRT writebacks contaminate this VRAM.
                     *
                     * Additive draws (ONE+ONE): ALWAYS clear. VRAM at the
                     * RT offset may contain stale content from previous 3D
                     * writebacks (e.g., a full-screen SRT at 0x300000/1024
                     * wrote wallpaper to VRAM range 0x300000-0x602000,
                     * contaminating offsets like 0x45b000). Loading this
                     * stale data and adding wallpaper on top doubles it —
                     * the root cause of the "cyan square" artifact.
                     *
                     * Non-additive draws (ONE+OMSA, etc.): load VRAM.
                     * These draws blend onto existing content, which may
                     * include legitimate data from earlier compositor passes
                     * (e.g., menu bar gradient+text written by a previous
                     * pass to the same VRAM offset).
                     */
                    bool vram_contaminated = is_additive;
                    if (!vram_contaminated && pitch_conflict) {
                        vram_contaminated = true;
                    }
                    if (vram_contaminated) {
                        uint32_t clear_row_bytes = bb_w * 4;
                        uint32_t *zeros =
                            (uint32_t *)g_malloc0(clear_row_bytes);
                        for (uint32_t y = bb_y; y < bb_y + bb_h; y++) {
                            [st->renderTarget
                                replaceRegion:MTLRegionMake2D(bb_x, y, bb_w, 1)
                                  mipmapLevel:0
                                    withBytes:zeros
                                  bytesPerRow:clear_row_bytes];
                        }
                        g_free(zeros);
                    } else {
                        /* Load from VRAM — may contain valid content */
                        metal_load_vram_region(st, vram_ptr, vram_size,
                                               color_offset, color_pitch_pixels,
                                               micro_tiled, macro_tiled,
                                               bb_x, bb_y, bb_w, bb_h);
                    }
                }
            }

            /* Build index buffer for QUAD_LIST */
            id<MTLBuffer> indexBuffer = nil;
            uint32_t index_count = 0;
            bool use_indexed_draw = false;

            if (cmd->prim_type == 13 /* QUAD_LIST */) {
                uint32_t num_quads = num_verts / 4;
                if (num_quads == 0) { g_free(vertices); return -1; }
                index_count = num_quads * 6;
                uint16_t *indices = g_new(uint16_t, index_count);
                for (uint32_t q = 0; q < num_quads; q++) {
                    uint16_t base = q * 4;
                    indices[q * 6 + 0] = base + 0;
                    indices[q * 6 + 1] = base + 1;
                    indices[q * 6 + 2] = base + 2;
                    indices[q * 6 + 3] = base + 0;
                    indices[q * 6 + 4] = base + 2;
                    indices[q * 6 + 5] = base + 3;
                }
                indexBuffer = [st->device
                    newBufferWithBytes:indices
                               length:index_count * sizeof(uint16_t)
                              options:MTLResourceStorageModeShared];
                g_free(indices);
                use_indexed_draw = true;
            }

            /* Create vertex buffer */
            id<MTLBuffer> vertexBuffer = [st->device
                newBufferWithBytes:vertices
                           length:num_verts * sizeof(MetalVertex)
                          options:MTLResourceStorageModeShared];
            g_free(vertices);
            if (!vertexBuffer) return -1;

            /* Uniforms — pass texture dimensions for texcoord normalization */
            /* Determine if texture is I8/Y8 intensity format */
            uint32_t tex_fmt_uniform = 0;
            if (textured && (tx_format == R200_TXFORMAT_I8 ||
                             tx_format == R200_TXFORMAT_Y8)) {
                tex_fmt_uniform = 1;  /* Signal I8 intensity expansion */
            }

            MetalUniforms uniforms = {
                .viewport_size = { (float)rt_width, (float)rt_height },
                .viewport_scale = { 1.0f, 1.0f },
                .viewport_offset = { 0.0f, 0.0f },
                .use_texture = textured ? 1 : 0,
                .tex_format = tex_fmt_uniform,
                .tex_size = { tex_w_for_uniform, tex_h_for_uniform },
                .swap_rb = using_vram_rt ? 1 : 0,
            };

            /* Map R200 primitive type to Metal */
            MTLPrimitiveType mtlPrim;
            switch (cmd->prim_type) {
            case 4:  mtlPrim = MTLPrimitiveTypeTriangle;      break;
            case 6:  mtlPrim = MTLPrimitiveTypeTriangleStrip;  break;
            case 5:  mtlPrim = MTLPrimitiveTypeTriangle;       break;
            case 8:  mtlPrim = MTLPrimitiveTypeTriangle;       break;
            case 13: mtlPrim = MTLPrimitiveTypeTriangle;       break;
            case 14: mtlPrim = MTLPrimitiveTypeTriangleStrip;  break;
            case 2:  mtlPrim = MTLPrimitiveTypeLine;           break;
            case 3:  mtlPrim = MTLPrimitiveTypeLineStrip;      break;
            case 1:  mtlPrim = MTLPrimitiveTypePoint;          break;
            default: mtlPrim = MTLPrimitiveTypeTriangle;       break;
            }

            /* Select pipeline state based on actual R200 blend mode.
             * Use the blend pipeline cache for dynamic blend factor combos. */
            id<MTLRenderPipelineState> activePipeline;
            if (nop_blend && st->pipelineStateNopBlend) {
                activePipeline = st->pipelineStateNopBlend;
            } else {
                /* Use dynamic blend pipeline matching the actual R200 blend mode */
                activePipeline = metal_get_blend_pipeline(st, src_blend, dst_blend);
            }

            /* Create command buffer and render pass */
            id<MTLCommandBuffer> cmdBuf = [st->commandQueue commandBuffer];
            if (!cmdBuf) return -1;

            MTLRenderPassDescriptor *rpDesc =
                [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.colorAttachments[0].texture = st->renderTarget;
            rpDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
            rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            id<MTLRenderCommandEncoder> encoder =
                [cmdBuf renderCommandEncoderWithDescriptor:rpDesc];
            if (!encoder) return -1;

            [encoder setRenderPipelineState:activePipeline];
            [encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
            [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:1];
            [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:1];

            /*
             * R200 scissor test: RE_TOP_LEFT (0x26C0) and RE_WIDTH_HEIGHT (0x26C4)
             * define a clip rectangle.  The QE compositor uses this to restrict
             * draws to the dirty region during window drag — wallpaper tiles are
             * clipped to only the exposed area, NOT the window body position.
             * Without this, wallpaper fills the SRT at the body position,
             * causing the PRESENT_BLIT to overwrite the window body.
             */
            {
                uint32_t sc_tl = state->re_top_left;
                uint32_t sc_wh = state->re_width_height;
                uint32_t sc_left = sc_tl & 0x3FFF;
                uint32_t sc_top  = (sc_tl >> 16) & 0x3FFF;
                uint32_t sc_right  = sc_wh & 0x3FFF;
                uint32_t sc_bottom = (sc_wh >> 16) & 0x3FFF;

                /* Only apply if scissor defines a valid sub-region */
                if (sc_right > sc_left && sc_bottom > sc_top &&
                    (sc_left > 0 || sc_top > 0 ||
                     sc_right < rt_width || sc_bottom < rt_height)) {
                    uint32_t s_x = sc_left;
                    uint32_t s_y = sc_top;
                    uint32_t s_w = sc_right - sc_left;
                    uint32_t s_h = sc_bottom - sc_top;
                    /* Clamp to render target bounds */
                    if (s_x + s_w > rt_width)  s_w = rt_width - s_x;
                    if (s_y + s_h > rt_height) s_h = rt_height - s_y;
                    MTLScissorRect sr = { s_x, s_y, s_w, s_h };
                    [encoder setScissorRect:sr];
                }
            }

            if (texture) {
                [encoder setFragmentTexture:texture atIndex:0];

                /*
                 * Build per-draw sampler with correct R200 wrapping mode.
                 * PP_TXFILTER_0 bits:
                 *   [2:0]   MAG_FILTER (0=nearest, 1=linear)
                 *   [16:14] CLAMP_S (0=wrap, 2=clamp_last, 4=clamp_border)
                 *   [19:17] CLAMP_T
                 * Use whichever register bank the kext wrote to.
                 */
                uint32_t txfilter = state->r100_pp_txfilter_0;
                if (txfilter == 0)
                    txfilter = state->pp_txfilter_0;
                uint32_t mag_filter = txfilter & 0x7;
                uint32_t clamp_s = (txfilter >> 14) & 0x7;
                uint32_t clamp_t = (txfilter >> 17) & 0x7;

                /* Map R200 clamp mode to Metal address mode */
                MTLSamplerAddressMode s_mode, t_mode;
                switch (clamp_s) {
                case 0: s_mode = MTLSamplerAddressModeRepeat; break;
                case 1: s_mode = MTLSamplerAddressModeMirrorRepeat; break;
                default: s_mode = MTLSamplerAddressModeClampToEdge; break;
                }
                switch (clamp_t) {
                case 0: t_mode = MTLSamplerAddressModeRepeat; break;
                case 1: t_mode = MTLSamplerAddressModeMirrorRepeat; break;
                default: t_mode = MTLSamplerAddressModeClampToEdge; break;
                }

                /* Create sampler with correct filter + wrapping */
                MTLSamplerDescriptor *sd =
                    [[MTLSamplerDescriptor alloc] init];
                sd.minFilter = (mag_filter == 1)
                    ? MTLSamplerMinMagFilterLinear
                    : MTLSamplerMinMagFilterNearest;
                sd.magFilter = sd.minFilter;
                sd.sAddressMode = s_mode;
                sd.tAddressMode = t_mode;
                id<MTLSamplerState> samp =
                    [st->device newSamplerStateWithDescriptor:sd];
                [encoder setFragmentSamplerState:samp atIndex:0];

                /* Log wrapping mode for menu bar draws */
                if (color_pitch_pixels == 1024 && bb_y < 30 &&
                    bb_h > 0 && bb_h <= 30) {
                    static int wrap_log = 0;
                    if (wrap_log < 40) {
                        fprintf(stderr,
                            "[MB_WRAP] d=%d txfilter=0x%08x "
                            "clamp_s=%u clamp_t=%u "
                            "tex=%ux%u bb=(%u,%u %ux%u)\n",
                            draw_num, txfilter,
                            clamp_s, clamp_t,
                            textured ? tx_width : 0,
                            textured ? tx_height : 0,
                            bb_x, bb_y, bb_w, bb_h);
                        wrap_log++;
                    }
                }
            } else {
                /*
                 * No texture — TEX_0 disabled in PP_CNTL.
                 * Bind a 1x1 opaque white dummy texture so the shader's
                 * `tex.sample() * vertex_color` produces just vertex_color.
                 * This is the correct R200 behavior: when texture unit is
                 * disabled, fragment color = interpolated vertex color.
                 */
                [encoder setFragmentTexture:st->whiteDummyTexture atIndex:0];
                [encoder setFragmentSamplerState:st->samplerNearest atIndex:0];
            }

            if (use_indexed_draw && indexBuffer) {
                [encoder drawIndexedPrimitives:mtlPrim
                                    indexCount:index_count
                                     indexType:MTLIndexTypeUInt16
                                   indexBuffer:indexBuffer
                             indexBufferOffset:0];
            } else {
                [encoder drawPrimitives:mtlPrim
                            vertexStart:0
                            vertexCount:num_verts];
            }
            [encoder endEncoding];

            if (!using_vram_rt) {
                /* Legacy path: blit render target to output buffer for CPU readback */
                id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                [blit copyFromTexture:st->renderTarget
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(rt_width, rt_height, 1)
                             toBuffer:st->outputBuffer
                    destinationOffset:0
               destinationBytesPerRow:rt_width * 4
             destinationBytesPerImage:rt_width * rt_height * 4];
                [blit endEncoding];
            }
            /* else: VRAM RT — texture IS VRAM, no readback needed.
             * The draw output is already in the guest's VRAM buffer.
             * Mark the VRAM region dirty so the display update picks it up. */

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                qemu_log("[METAL_DRAW] %d: command buffer error: %s\n",
                         draw_num,
                         cmdBuf.error ?
                         [[cmdBuf.error localizedDescription] UTF8String] :
                         "unknown");
                return -1;
            }

            /* Phase A — check if writeback will overwrite texture range */
            if (draw_num >= 70 && draw_num <= 78 &&
                color_offset == 0x300000) {
                /* Check 0x353000 before writeback */
                uint32_t pre_val = 0;
                if (0x353000 + 4 <= vram_size) {
                    pre_val = *(uint32_t *)(vram_ptr + 0x353000);
                }
                /* Compute which VRAM row 0x353000 maps to at this pitch */
                uint32_t vram_off = 0x353000 - color_offset;
                uint32_t pitch_bytes = color_pitch_pixels * 4;
                uint32_t row_at_pitch = pitch_bytes ? vram_off / pitch_bytes : 0;
                uint32_t col_at_pitch = pitch_bytes ?
                    (vram_off % pitch_bytes) / 4 : 0;
                bool in_bbox = (row_at_pitch >= bb_y &&
                                row_at_pitch < bb_y + bb_h &&
                                col_at_pitch >= bb_x &&
                                col_at_pitch < bb_x + bb_w);
                fprintf(stderr,
                    "[WB_TRACE] draw=%d BEFORE writeback: "
                    "vram[0x353000]=0x%08x pitch=%u row=%u col=%u "
                    "bbox=(%u,%u %ux%u) in_bbox=%d\n",
                    draw_num, pre_val, color_pitch_pixels,
                    row_at_pitch, col_at_pitch,
                    bb_x, bb_y, bb_w, bb_h, in_bbox);
            }

            /*
             * VRAM writeback is SKIPPED.
             *
             * On real R200, the MC tiling engine separates surfaces at the
             * same VRAM offset but different pitches. Our linear VRAM model
             * can't do this — writeback at pitch P overwrites texture data
             * stored at the same offset with pitch Q, corrupting it.
             *
             * The SRT (shadow render target) captures the authoritative
             * Metal output for BLT reads. VRAM writeback was only needed
             * for texture reads from render target output, but those reads
             * at different pitches would get garbled data anyway.
             *
             * Skipping writeback preserves original texture data in VRAM
             * (especially I8 shadow blur textures) that would otherwise be
             * destroyed by 32-bit render target writes.
             */

            /* Log viewport + coordinate info for draws to non-standard offsets
             * (Finder window surfaces, menu extras, etc.) */
            {
                static int vp_draw_log = 0;
                bool is_finder = (color_offset != 0x300000 &&
                                  color_offset != 0x1ee000 &&
                                  color_offset != 0x325000 &&
                                  color_offset != 0x45b000);
                if (vp_draw_log < 200 || is_finder) {
                    float vp_xs = u32_to_float(state->se_vport_xscale);
                    float vp_xo = u32_to_float(state->se_vport_xoffset);
                    float vp_ys = u32_to_float(state->se_vport_yscale);
                    float vp_yo = u32_to_float(state->se_vport_yoffset);
                    uint32_t vte = state->se_vte_cntl;
                    /* For Finder-like surfaces, also dump first 4 vertex positions */
                    if (is_finder && cmd->vertex_data && cmd->num_vertices >= 2) {
                        uint32_t stride_dw = cmd->vb_stride > 0
                            ? cmd->vb_stride / 4
                            : metal_calc_vertex_stride(state) / 4;
                        if (stride_dw < 2) stride_dw = 2;
                        uint32_t nv = cmd->num_vertices < 4 ? cmd->num_vertices : 4;
                        fprintf(stderr, "[VP_DRAW_VTX] d=%d off=0x%x p=%u "
                                "bbox=(%u,%u %ux%u) tex=0x%x/%s "
                                "blend=%u/%u vte=0x%x vtx=[",
                                draw_num, color_offset, color_pitch_pixels,
                                bb_x, bb_y, bb_w, bb_h,
                                tx_offset,
                                textured ? "YES" : "NONE",
                                src_blend, dst_blend, vte);
                        for (uint32_t vi = 0; vi < nv; vi++) {
                            float vx = u32_to_float(cmd->vertex_data[vi * stride_dw]);
                            float vy = u32_to_float(cmd->vertex_data[vi * stride_dw + 1]);
                            fprintf(stderr, "(%.1f,%.1f)%s", vx, vy,
                                    vi < nv-1 ? " " : "");
                        }
                        fprintf(stderr, "]\n");
                    } else {
                        fprintf(stderr, "[VP_DRAW] d=%d off=0x%x p=%u "
                                "bbox=(%u,%u %ux%u) vp=(%.1f+%.1f, %.1f+%.1f) "
                                "vte=0x%x\n",
                                draw_num, color_offset, color_pitch_pixels,
                                bb_x, bb_y, bb_w, bb_h,
                                vp_xo, vp_xs, vp_yo, vp_ys, vte);
                    }
                    vp_draw_log++;
                }
            }

            /* Track: render + writeback */
            frame_tracker_record(&st->frame_tracker, PASS_EVENT_RENDER,
                                 color_offset, color_pitch_pixels,
                                 textured ? tx_offset : 0, color_offset,
                                 bb_w, bb_h, bb_x, bb_y, false);
            frame_tracker_record(&st->frame_tracker, PASS_EVENT_WRITEBACK,
                                 color_offset, color_pitch_pixels,
                                 color_offset, color_offset,
                                 bb_w, bb_h, bb_x, bb_y, false);

            /* Phase A — check if writeback changed texture range */
            if (draw_num >= 70 && draw_num <= 78 &&
                color_offset == 0x300000) {
                uint32_t post_val = 0;
                if (0x353000 + 4 <= vram_size) {
                    post_val = *(uint32_t *)(vram_ptr + 0x353000);
                }
                fprintf(stderr,
                    "[WB_TRACE] draw=%d AFTER writeback: "
                    "vram[0x353000]=0x%08x\n",
                    draw_num, post_val);
            }

            /*
             * Shadow RT: save the rendered region into a host-side buffer
             * keyed by (color_offset, color_pitch_pixels).  This avoids the
             * VRAM pitch-conflict problem: multiple pitches writing to the
             * same VRAM offset produce garbled data in linear mode, but the
             * shadow buffers are kept separate per pitch.  The PRESENT_BLIT
             * (metal_blit_2d) reads from the shadow buffer instead of VRAM.
             *
             * SRT Extent Growth: ensure the SRT is large enough for the
             * full draw bbox, not just the viewport-derived rt_height.
             * Multiple draws to the same (offset, pitch) may target widely
             * separated y positions (e.g., menu bar extras at y=0, 92, 184, 276).
             */
            if (using_vram_rt) {
                /* VRAM RT path: draw output is already in VRAM.
                 * Endianness is handled by the fragment shader (swap_rb=1)
                 * which outputs pixels in the byte order the guest expects.
                 * No CPU-side bswap needed. */
            } else
            {
                uint32_t srt_need_h = (bb_y + bb_h > rt_height)
                    ? (bb_y + bb_h) : rt_height;
                uint32_t srt_need_w = color_pitch_pixels;
                if (bb_x + bb_w > srt_need_w) {
                    srt_need_w = bb_x + bb_w;
                }
                int srt_slot = shadow_rt_ensure(st, color_offset,
                    color_pitch_pixels, srt_need_w, srt_need_h);
                if (srt_slot >= 0) {
                    const uint32_t *metal_out =
                        (const uint32_t *)[st->outputBuffer contents];
                    if (metal_out) {
                        shadow_rt_save_region(st, srt_slot, metal_out,
                            rt_width, bb_x, bb_y, bb_w, bb_h);
                    }
                    /* Diagnostic: track SRT content for drag layers */
                    if (color_pitch_pixels >= 832 &&
                        color_pitch_pixels <= 960 &&
                        color_offset >= 0x900000) {
                        uint32_t sw2 = st->shadow_rts[srt_slot].width;
                        uint32_t sh2 = st->shadow_rts[srt_slot].height;
                        const uint32_t *sp =
                            st->shadow_rts[srt_slot].pixels;
                        uint32_t nz = 0;
                        if (sp) {
                            uint32_t cnt = sw2 * (sh2 < 50 ? sh2 : 50);
                            for (uint32_t i = 0; i < cnt; i++)
                                if (sp[i]) nz++;
                        }
                        static int drag_srt_log = 0;
                        if (drag_srt_log < 100) {
                            fprintf(stderr,
                                "[SRT_SAVE_DIAG] d=%d slot=%d off=0x%x "
                                "p=%u bb=(%u,%u %ux%u) srt=%ux%u "
                                "nz_first50rows=%u consumed=%d stale=%d\n",
                                draw_num, srt_slot, color_offset,
                                color_pitch_pixels,
                                bb_x, bb_y, bb_w, bb_h, sw2, sh2,
                                nz,
                                st->shadow_rts[srt_slot].consumed,
                                st->shadow_rts[srt_slot].stale);
                            drag_srt_log++;
                        }
                    }
                    /* Log SRT state after saves to menu bar surfaces */
                    if (color_pitch_pixels == 1024 && bb_y < 30 &&
                        bb_h > 0 && bb_h <= 30) {
                        static int mb_srt_log = 0;
                        if (mb_srt_log < 40) {
                            uint32_t sw2 = st->shadow_rts[srt_slot].width;
                            uint32_t sh2 = st->shadow_rts[srt_slot].height;
                            const uint32_t *sp =
                                st->shadow_rts[srt_slot].pixels;
                            uint32_t p0 = 0, p256 = 0, p512 = 0, p768 = 0;
                            if (sp && sw2 > 768 && sh2 > 0) {
                                p0 = sp[0];
                                p256 = sp[256];
                                p512 = sp[512];
                                p768 = sp[768];
                            }
                            /* Also sample Metal output at same positions */
                            uint32_t m0 = 0, m256 = 0, m512 = 0;
                            if (metal_out && rt_width > 512) {
                                m0 = metal_out[0];
                                m256 = metal_out[256];
                                m512 = metal_out[512];
                            }
                            fprintf(stderr,
                                "[MB_SRT_STATE] d=%d off=0x%x "
                                "bb=(%u,%u %ux%u) "
                                "srt[0]=0x%08x srt[256]=0x%08x "
                                "srt[512]=0x%08x srt[768]=0x%08x "
                                "metal[0]=0x%08x metal[256]=0x%08x "
                                "metal[512]=0x%08x\n",
                                draw_num, color_offset,
                                bb_x, bb_y, bb_w, bb_h,
                                p0, p256, p512, p768,
                                m0, m256, m512);
                            mb_srt_log++;
                        }
                    }
                    /* Log all saves to 0x54a000 pitch=768 (Finder window) */
                    if (color_offset == 0x54a000 &&
                        color_pitch_pixels == 768) {
                        static int fw_log = 0;
                        if (fw_log < 30) {
                            /* Sample metal output at bbox corners */
                            uint32_t m0 = 0, m1 = 0;
                            if (bb_w > 0 && bb_h > 0) {
                                m0 = metal_out[bb_y * rt_width + bb_x];
                                m1 = metal_out[(bb_y + bb_h/2) * rt_width +
                                               bb_x + bb_w/2];
                            }
                            float vp_xs = u32_to_float(state->se_vport_xscale);
                            float vp_xo = u32_to_float(state->se_vport_xoffset);
                            float vp_ys = u32_to_float(state->se_vport_yscale);
                            float vp_yo = u32_to_float(state->se_vport_yoffset);
                            uint32_t vte = state->se_vte_cntl;
                            fprintf(stderr, "[FW_SAVE] d=%d bbox=(%u,%u %ux%u) "
                                    "rt=%ux%u m0=0x%08x mmid=0x%08x "
                                    "vp=(%.1f,%.1f,%.1f,%.1f) vte=0x%x\n",
                                    draw_num, bb_x, bb_y, bb_w, bb_h,
                                    rt_width, rt_height,
                                    m0, m1,
                                    vp_xs, vp_xo, vp_ys, vp_yo, vte);
                            fw_log++;
                        }
                    }
                    static int srt_save_log = 0;
                    if (srt_save_log < 10) {
                        qemu_log("[SHADOW_RT] save draw=%d slot=%d "
                                 "off=0x%x pitch=%u %ux%u "
                                 "bbox=(%u,%u %ux%u)\n",
                                 draw_num, srt_slot,
                                 color_offset, color_pitch_pixels,
                                 color_pitch_pixels, rt_height,
                                 bb_x, bb_y, bb_w, bb_h);
                        srt_save_log++;
                    }
                    /* Phase A: SRT extent observation logging */
                    {
                        static int srt_ext_log = 0;
                        static int srt_ext_small_log = 0;
                        bool is_small = (color_pitch_pixels <= 256);
                        if (srt_ext_log < 120 || (is_small && srt_ext_small_log < 60)) {
                            uint32_t alloc_w = st->shadow_rts[srt_slot].width;
                            uint32_t alloc_h = st->shadow_rts[srt_slot].height;
                            fprintf(stderr,
                                "[SRT_EXTENT_OBS] "
                                "rt_off=0x%x rt_pitch=%u "
                                "alloc_size=(%u,%u) "
                                "draw_id=%d "
                                "draw_bbox=(%u,%u)-(%u,%u) "
                                "rt_size=(%u,%u) "
                                "note=srt_slot=%d\n",
                                color_offset, color_pitch_pixels,
                                alloc_w, alloc_h,
                                draw_num,
                                bb_x, bb_y, bb_x + bb_w, bb_y + bb_h,
                                rt_width, rt_height,
                                srt_slot);
                            srt_ext_log++;
                            if (is_small) srt_ext_small_log++;
                        }
                    }
                }
            }

            /*
             * Phase A — Double-swizzle summary.
             * Metal writeback writes directly to raw VRAM at tiled addresses.
             * mc_vram_read32 reads using MC tiling formula.
             * If both paths apply tiling independently with same formula,
             * there is NO double-swizzle — single tiling is applied on
             * write (Metal) and matched on read (MC).
             * Emit summary after draw #10.
             */
            {
                static bool dswiz_summary_done = false;
                if (!dswiz_summary_done && draw_num == 10 && macro_tiled) {
                    dswiz_summary_done = true;
                    fprintf(stderr,
                        "[DOUBLE_SWIZZLE_SUMMARY] rt_off=0x%x "
                        "metal_writes_tiled=%s "
                        "metal_write_path=direct_vram_ptr "
                        "mc_write_retiles=n/a(metal_bypasses_mc) "
                        "mc_read_retiles=yes(if_surface_registered) "
                        "double_swizzle_detected=%s "
                        "confidence=%s "
                        "note=Metal_writes_tiled_via_raw_ptr_"
                        "MC_reads_tiled_via_mc_vram_read32_"
                        "both_use_same_formula_"
                        "see_TILE_PATH_and_TILE_PARITY_for_proof\n",
                        color_offset,
                        macro_tiled ? "yes" : "no",
                        "no",
                        "high");
                }
            }

            /* Phase B post-draw: log RT pixels after writeback for
             * the first wallpaper and window draws */
            {
                static bool post_wallpaper_done = false;
                static bool post_window_done = false;
                if (!post_wallpaper_done && textured && is_additive &&
                    bb_w > 4 && bb_h > 4) {
                    post_wallpaper_done = true;
                    qemu_log("[DRAW_PIXEL_AUDIT] === POST-DRAW wallpaper ===\n");
                    metal_draw_pixel_audit_after(vram_ptr, vram_size,
                        color_offset, color_pitch_pixels,
                        bb_x, bb_y, bb_w, bb_h, draw_num);
                }
                if (!post_window_done && textured &&
                    src_blend == R200_BLEND_GL_ONE &&
                    dst_blend == R200_BLEND_GL_ONE_MINUS_SRC_ALPHA &&
                    bb_w > 4 && bb_h > 4) {
                    post_window_done = true;
                    qemu_log("[DRAW_PIXEL_AUDIT] === POST-DRAW window ===\n");
                    metal_draw_pixel_audit_after(vram_ptr, vram_size,
                        color_offset, color_pitch_pixels,
                        bb_x, bb_y, bb_w, bb_h, draw_num);
                }
            }

            /*
             * Phase 3A: Compare VRAM CRC after writeback for NOP-blend draws.
             * If the surface changed, find the first mismatched pixel.
             */
            if (do_probe_check) {
                uint32_t after_crc = compute_vram_region_crc(
                    vram_ptr, vram_size, color_offset, color_pitch_pixels,
                    micro_tiled, macro_tiled, bb_x, bb_y, bb_w, bb_h);

                bool match = (before_crc == after_crc);
                if (!match) {
                    probe_mismatch_total++;
                    /* Find first mismatched pixel */
                    uint32_t fm_x = 0, fm_y = 0;
                    uint32_t fm_before = 0, fm_after = 0;
                    bool found = false;
                    /* Re-scan: we need before data, but it was overwritten.
                     * We can't recover exact before values cheaply, so report
                     * the first non-zero pixel as evidence of the mismatch. */
                    for (uint32_t sy = bb_y; sy < bb_y + bb_h && !found; sy++) {
                        for (uint32_t sx = bb_x; sx < bb_x + bb_w && !found; sx++) {
                            uint64_t off = color_offset +
                                r200_tile_offset(sx, sy, color_pitch_pixels,
                                                  micro_tiled, macro_tiled);
                            if (off + 4 <= vram_size) {
                                uint32_t pix = *(uint32_t *)(vram_ptr + off);
                                if (pix != 0) {
                                    fm_x = sx; fm_y = sy;
                                    fm_after = pix;
                                    found = true;
                                }
                            }
                        }
                    }
                    qemu_log("[PROBE_SURFACE] draw_id=%d target_off=0x%x "
                             "target_pitch=%u macro=%d "
                             "bbox=(%u,%u)-(%u,%u) "
                             "before_crc=0x%08x after_crc=0x%08x "
                             "expected=unchanged result=mismatch "
                             "mismatch_total=%d "
                             "first_nonzero=(%u,%u) after=0x%08x\n",
                             draw_num, color_offset, color_pitch_pixels,
                             macro_tiled,
                             bb_x, bb_y, bb_x + bb_w, bb_y + bb_h,
                             before_crc, after_crc,
                             probe_mismatch_total,
                             fm_x, fm_y, fm_after);
                } else if (metal_exec_count <= 20 ||
                           metal_exec_count % 200 == 0) {
                    qemu_log("[PROBE_SURFACE] draw_id=%d target_off=0x%x "
                             "target_pitch=%u macro=%d "
                             "bbox=(%u,%u)-(%u,%u) "
                             "before_crc=0x%08x after_crc=0x%08x "
                             "expected=unchanged result=match\n",
                             draw_num, color_offset, color_pitch_pixels,
                             macro_tiled,
                             bb_x, bb_y, bb_x + bb_w, bb_y + bb_h,
                             before_crc, after_crc);
                }
            }

            if (verbose) {
                qemu_log("[METAL_DRAW] %d: OK prim=%u nverts=%u nop=%d "
                         "textured=%d bbox=(%u,%u %ux%u) "
                         "off=0x%x pitch=%u micro=%d macro=%d\n",
                         draw_num, cmd->prim_type, num_verts, nop_blend,
                         textured, bb_x, bb_y, bb_w, bb_h,
                         color_offset, color_pitch_pixels,
                         micro_tiled, macro_tiled);
            }

            /* Periodic draw summary */
            if (metal_exec_count == 10 || metal_exec_count == 50 ||
                metal_exec_count == 100 || metal_exec_count == 500 ||
                metal_exec_count % 1000 == 0) {
                qemu_log("[DRAW_SUMMARY] total_metal_exec=%d nop=%d real=%d\n",
                         metal_exec_count, metal_nop_count, metal_real_count);
            }
        } /* end draw execution */
    } /* end @autoreleasepool */

    return 0;
}

static void metal_mode_change(void *opaque, const PPCMacGPUScanout *new_mode)
{
    PPCMacGPUMetalState *st = opaque;

    pthread_mutex_lock(&g_render_lock);
    if (!st || !new_mode) {
        pthread_mutex_unlock(&g_render_lock);
        return;
    }

    st->scanout_width = new_mode->width;
    st->scanout_height = new_mode->height;

    /* Invalidate render target so it gets recreated at new size */
    @autoreleasepool {
        st->renderTarget = nil;
        st->outputBuffer = nil;
        st->rt_width = 0;
        st->rt_height = 0;
    }
    pthread_mutex_unlock(&g_render_lock);
}

static uint32_t metal_get_caps(void *opaque)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st || !st->initialized) return 0;

    return PPC_MAC_GPU_RENDERER_CAP_3D | PPC_MAC_GPU_RENDERER_CAP_ACCEL;
}

/* ========================================================================
 * Phase B — Deterministic macro-tile formula parity test.
 *
 * Compares Metal's r200_macrotile_offset() vs the MC path's formula
 * (r200_macro_tile_addr minus base) for a grid of (x, y, pitch) values.
 * Runs once at first 3D draw. Reports first mismatch or all-match.
 * ======================================================================== */
static void tile_formula_parity_test(void)
{
    static bool done = false;
    if (done) return;
    done = true;

    static const uint32_t test_pitches[] = {64, 128, 192, 832, 896, 1024};
    int n_pitches = sizeof(test_pitches) / sizeof(test_pitches[0]);
    int tested = 0, mismatches = 0;
    uint32_t first_mm_pitch = 0, first_mm_x = 0, first_mm_y = 0;
    uint64_t first_mm_metal = 0, first_mm_mc = 0;

    for (int pi = 0; pi < n_pitches; pi++) {
        uint32_t pitch_px = test_pitches[pi];
        uint32_t pitch_bytes = pitch_px * 4;
        uint32_t max_x = pitch_px < 256 ? pitch_px : 256;
        uint32_t max_y = 64;

        for (uint32_t y = 0; y < max_y; y++) {
            for (uint32_t x = 0; x < max_x; x++) {
                /* Metal formula: r200_macrotile_offset (returns offset from base) */
                uint64_t metal_off = r200_macrotile_offset(x, y, pitch_px);

                /* MC formula: r200_macro_tile_addr returns base + offset.
                 * Inline the MC formula here to avoid calling into ppc_mac_gpu.c */
                uint32_t tile_w_pixels = 64; /* 256/4 for 32bpp */
                uint32_t tile_h = 16;
                uint32_t tiles_per_row = pitch_bytes / 256;
                uint32_t tile_x = x / tile_w_pixels;
                uint32_t tile_y = y / tile_h;
                uint32_t in_x = x % tile_w_pixels;
                uint32_t in_y = y % tile_h;
                uint64_t tile_idx = (uint64_t)tile_y * tiles_per_row + tile_x;
                uint64_t mc_off = tile_idx * (256 * tile_h)
                                  + (uint64_t)in_y * 256
                                  + (uint64_t)in_x * 4;

                tested++;
                if (metal_off != mc_off) {
                    mismatches++;
                    if (mismatches == 1) {
                        first_mm_pitch = pitch_px;
                        first_mm_x = x;
                        first_mm_y = y;
                        first_mm_metal = metal_off;
                        first_mm_mc = mc_off;
                    }
                    if (mismatches <= 5) {
                        fprintf(stderr,
                            "[TILE_PARITY] pitch_px=%u x=%u y=%u "
                            "metal_off=0x%llx mc_off=0x%llx match=no\n",
                            pitch_px, x, y,
                            (unsigned long long)metal_off,
                            (unsigned long long)mc_off);
                    }
                }
            }
        }
    }

    fprintf(stderr,
        "[TILE_PARITY_SUMMARY] tested_cases=%d mismatches=%d "
        "first_mismatch=%s all_match=%s "
        "note=%s\n",
        tested, mismatches,
        mismatches > 0 ? "see_above" : "none",
        mismatches == 0 ? "yes" : "no",
        mismatches == 0
            ? "Metal_and_MC_macrotile_formulas_are_identical"
            : "MISMATCH_FOUND_formulas_disagree");

    if (mismatches > 0) {
        fprintf(stderr,
            "[TILE_PARITY] FIRST_MISMATCH: pitch_px=%u x=%u y=%u "
            "metal_off=0x%llx mc_off=0x%llx\n",
            first_mm_pitch, first_mm_x, first_mm_y,
            (unsigned long long)first_mm_metal,
            (unsigned long long)first_mm_mc);
    }
}

static bool metal_get_drag_state(void *opaque,
                                  uint32_t *origin_x, uint32_t *origin_y,
                                  uint32_t *blit_w, uint32_t *blit_h,
                                  uint32_t *frame_gen)
{
    PPCMacGPUMetalState *st = opaque;
    if (!st) return false;

    pthread_mutex_lock(&g_render_lock);
    /* Find most recently active tracker */
    for (int i = 0; i < DRAG_TRACKER_MAX; i++) {
        if (st->drag_trackers[i].active &&
            st->drag_trackers[i].cur_frame_gen == st->rt_frame_gen) {
            *origin_x = st->drag_trackers[i].cur_origin_x;
            *origin_y = st->drag_trackers[i].cur_origin_y;
            *blit_w = st->drag_trackers[i].blit_w;
            *blit_h = st->drag_trackers[i].blit_h;
            *frame_gen = st->drag_trackers[i].cur_frame_gen;
            pthread_mutex_unlock(&g_render_lock);
            return true;
        }
    }
    pthread_mutex_unlock(&g_render_lock);
    return false;
}

static void metal_flush_drag_paste(void *opaque, uint8_t *vram)
{
    PPCMacGPUMetalState *st = opaque;
    if (st) {
        pthread_mutex_lock(&g_render_lock);
        drag_body_paste(st, vram);
        pthread_mutex_unlock(&g_render_lock);
    }
}

/* ========================================================================
 * Renderer vtable
 * ======================================================================== */

/* ========================================================================
 * Direct R200 renderer (draw_r200)
 *
 * Renders an R200DrawPacket straight into VRAM: the render target and the
 * textures are linear texture views over the VRAM MTLBuffer at the guest's
 * own offsets and pitches, so there is no staging, shadow copy or
 * write-back, and every 2D blit or scanout that follows sees the pixels.
 *
 * Byte order: 32bpp VRAM holds A,R,G,B bytes.  Viewed as BGRA8Unorm those
 * read back as (r,g,b,a) = (G,R,A,B), so true colour is raw.yxwz and the
 * same swizzle converts back.  Because the stored "alpha" channel is really
 * blue, fixed-function blending cannot be used; blending is done in the
 * shader with framebuffer fetch ([[color(0)]], Apple GPUs).
 * ======================================================================== */

typedef struct R200Uniforms {
    float    rt_size[2];
    uint32_t pp_cntl, pp_misc;
    uint32_t rb3d_cntl, cblend, ablend, pad0;
    float    blend_color[4];
    uint32_t stage[R200_MAX_STAGES][4];   /* txcblend, txcblend2, txablend, txablend2 */
    float    tfactor[8][4];
    uint32_t texinfo[R200_MAX_TEX][4];    /* enabled, format, alpha_in_map, denorm */
    float    texsize[R200_MAX_TEX][4];    /* w, h, 1/w, 1/h */
    float    excl[3][4];                  /* exclusive scissors x0,y0,x1,y1 */
    uint32_t num_excl, plane_mask, pad1[2];
    uint32_t texfilt[R200_MAX_TEX][4];    /* x: raw PP_TXFILTER */
    float    fog_color[4];                /* rgb, w: factor source */
    uint32_t zinfo[4];                    /* x flags, y ZSTENCILCNTL, z STENCILREFMASK */
} R200Uniforms;

static NSString *const kR200ShaderSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
/* Specialization: how many combiner stages run and which texture units are
 * bound, per pipeline (the defaults keep the generic pipelines as before). */
"constant uint FC_STAGES [[function_constant(0)]];\n"
"constant uint FC_TEXMASK [[function_constant(1)]];\n"
"struct Vtx { float4 pos; float4 color; float4 spec; float4 tex[6]; };\n"
"struct U {\n"
"    float2 rt_size; uint pp_cntl; uint pp_misc;\n"
"    uint rb3d_cntl; uint cblend; uint ablend; uint pad0;\n"
"    float4 blend_color;\n"
"    uint4 stage[8]; float4 tfactor[8];\n"
"    uint4 texinfo[6]; float4 texsize[6];\n"
"    float4 excl[3]; uint4 nexcl;   /* x: count, y: plane mask */\n""    uint4 texfilt[6];\n"
"    float4 fog_color;\n"
"    uint4 zinfo;   /* x: 1 z-test, 2 stencil, 4 z-write, 8 z16; y zstencil; z refmask */\n"
"};\n"
"struct VOut {\n"
"    float4 position [[position]];\n"
"    float psize [[point_size]];\n"
"    float4 color; float4 spec;\n"
"    float4 t0; float4 t1; float4 t2; float4 t3; float4 t4; float4 t5;\n"
"};\n"
"vertex VOut r200_vs(uint vid [[vertex_id]],\n"
"                    const device Vtx *v [[buffer(0)]],\n"
"                    constant U &u [[buffer(1)]]) {\n"
"    Vtx i = v[vid];\n"
"    float w = i.pos.w != 0.0 ? i.pos.w : 1.0;\n"
"    float2 ndc = float2(i.pos.x / u.rt_size.x * 2.0 - 1.0,\n"
"                        1.0 - i.pos.y / u.rt_size.y * 2.0);\n"
"    VOut o;\n"
"    o.position = float4(ndc * w, clamp(i.pos.z, 0.0, 1.0) * w, w);\n"
"    o.psize = 1.0;\n"
"    o.color = i.color; o.spec = i.spec;\n"
"    o.t0 = i.tex[0]; o.t1 = i.tex[1]; o.t2 = i.tex[2];\n"
"    o.t3 = i.tex[3]; o.t4 = i.tex[4]; o.t5 = i.tex[5];\n"
"    return o;\n"
"}\n"
"/* 16-bit packed texel, stored big-endian (byte 0 = high byte) */\n"
"static float4 unpack16v(uint v, uint f) {\n"
"    if (f == 4u) return float4(float((v >> 11) & 31u) / 31.0, float((v >> 5) & 63u) / 63.0,\n"
"                               float(v & 31u) / 31.0, 1.0);                 /* RGB565 */\n"
"    if (f == 3u) return float4(float((v >> 10) & 31u) / 31.0, float((v >> 5) & 31u) / 31.0,\n"
"                               float(v & 31u) / 31.0, float(v >> 15));      /* ARGB1555 */\n"
"    return float4(float((v >> 8) & 15u) / 15.0, float((v >> 4) & 15u) / 15.0,\n"
"                  float(v & 15u) / 15.0, float((v >> 12) & 15u) / 15.0);    /* ARGB4444 */\n"
"}\n"
"static float4 unpack16(float4 rg, uint f) {\n"
"    return unpack16v((uint(rg.r * 255.0 + 0.5) << 8) | uint(rg.g * 255.0 + 0.5), f);\n"
"}\n"
"static uint2 wrapc(int2 p, uint2 size, uint filt) {\n"
"    int2 sz = int2(size);\n"
"    int2 r;\n"
"    r.x = ((filt >> 23) & 7u) == 0u ? ((p.x % sz.x) + sz.x) % sz.x : clamp(p.x, 0, sz.x - 1);\n"
"    r.y = ((filt >> 27) & 7u) == 0u ? ((p.y % sz.y) + sz.y) % sz.y : clamp(p.y, 0, sz.y - 1);\n"
"    return uint2(r);\n"
"}\n"
"static float4 texel(texture2d<float> t, sampler s, float4 tc, uint4 info, float4 sz, uint filt) {\n"
"    if (info.x == 0) return float4(0.0);\n"
"    float2 st = tc.xy / (tc.w != 0.0 ? tc.w : 1.0);\n"
"    if (info.w != 0) st *= sz.zw;\n"
"    uint f = info.y;\n"
"    if (f >= 3u && f <= 5u) {                      /* packed 16-bit: manual filter */\n"
"        uint2 size = uint2(sz.xy);\n"
"        float2 p = st * sz.xy - 0.5;\n"
"        if ((filt & 1u) == 0u) {\n"
"            return unpack16(t.read(wrapc(int2(floor(p + 0.5)), size, filt)), f);\n"
"        }\n"
"        int2 i0 = int2(floor(p));\n"
"        float2 fr = p - floor(p);\n"
"        float4 a = unpack16(t.read(wrapc(i0, size, filt)), f);\n"
"        float4 b = unpack16(t.read(wrapc(i0 + int2(1, 0), size, filt)), f);\n"
"        float4 c = unpack16(t.read(wrapc(i0 + int2(0, 1), size, filt)), f);\n"
"        float4 d = unpack16(t.read(wrapc(i0 + int2(1, 1), size, filt)), f);\n"
"        return mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);\n"
"    }\n"
"    float4 raw = t.sample(s, st);\n"
"    if (f == 100u) return raw;                      /* decoded (DXT) */\n"
"    if (f == 7u) {                                  /* RGBA8888: BE bytes R,G,B,A */\n"
"        float4 c = raw.zyxw;\n"
"        if (info.z == 0) c.a = 1.0;\n"
"        return c;\n"
"    }\n"
"    if (f == 22u) {                                 /* ABGR8888: BE bytes A,B,G,R */\n"
"        float4 c = raw.wxyz;\n"
"        if (info.z == 0) c.a = 1.0;\n"
"        return c;\n"
"    }\n"
"    if (f == 1u) {                                  /* AI88: BE bytes A, I */\n"
"        float i = raw.g, a = raw.r;\n"
"        return float4(i, i, i, info.z != 0 ? a : 1.0);\n"
"    }\n"
"    if (f == 0u || f == 8u) {                       /* I8 / Y8 */\n"
"        float i = raw.r;\n"
"        return (f == 0u && info.z != 0) ? float4(i) : float4(i, i, i, 1.0);\n"
"    }\n"
"    float4 c = raw.yxwz;                            /* A,R,G,B bytes */\n"
"    if (info.z == 0) c.a = 1.0;\n"
"    return c;\n"
"}\n"
"static float3 cmod(float3 v, uint m) {              /* comp, bias, scale, neg */\n"
"    if (m & 1u) v = 1.0 - v;\n"
"    if (m & 2u) v = v - 0.5;\n"
"    if (m & 4u) v = v * 2.0;\n"
"    if (m & 8u) v = -v;\n"
"    return v;\n"
"}\n"
"static float amod(float v, uint m) {\n"
"    if (m & 1u) v = 1.0 - v;\n"
"    if (m & 2u) v = v - 0.5;\n"
"    if (m & 4u) v = v * 2.0;\n"
"    if (m & 8u) v = -v;\n"
"    return v;\n"
"}\n"
"static float3 carg(uint sel, float4 cur, float4 dif, float4 spc, float4 tf, float4 tf1, thread float4 *R) {\n"
"    switch (sel) {\n"
"    case 2: return cur.rgb;   case 3: return float3(cur.a);\n"
"    case 4: return dif.rgb;   case 5: return float3(dif.a);\n"
"    case 6: return spc.rgb;   case 7: return float3(spc.a);\n"
"    case 8: return tf.rgb;    case 9: return float3(tf.a);\n"
"    case 26: return tf1.rgb;  case 27: return float3(tf1.a);\n"
"    default:\n"
"        if (sel >= 10 && sel <= 21) {\n"
"            float4 r = R[(sel - 10) / 2];\n"
"            return (sel & 1u) ? float3(r.a) : r.rgb;\n"
"        }\n"
"        return float3(0.0);\n"
"    }\n"
"}\n"
"static float aarg(uint sel, float4 cur, float4 dif, float4 spc, float4 tf, float4 tf1, thread float4 *R) {\n"
"    switch (sel) {\n"
"    case 2: return cur.a;   case 3: return cur.b;\n"
"    case 4: return dif.a;   case 5: return dif.b;\n"
"    case 6: return spc.a;   case 7: return spc.b;\n"
"    case 8: return tf.a;    case 9: return tf.b;\n"
"    case 26: return tf1.a;  case 27: return tf1.b;\n"
"    default:\n"
"        if (sel >= 10 && sel <= 21) {\n"
"            float4 r = R[(sel - 10) / 2];\n"
"            return (sel & 1u) ? r.b : r.a;\n"
"        }\n"
"        return 0.0;\n"
"    }\n"
"}\n"
"static float oscale(uint c2) {\n"
"    switch ((c2 >> 8) & 7u) {\n"
"    case 1: return 2.0; case 2: return 4.0; case 3: return 8.0;\n"
"    case 5: return 0.5; case 6: return 0.25; case 7: return 0.125;\n"
"    default: return 1.0;\n"
"    }\n"
"}\n"
"static float4 bfactor(uint f, float4 s, float4 d, float4 k) {\n"
"    switch (f) {\n"
"    case 32: return float4(0.0);          case 33: return float4(1.0);\n"
"    case 34: return s;                    case 35: return 1.0 - s;\n"
"    case 36: return d;                    case 37: return 1.0 - d;\n"
"    case 38: return float4(s.a);          case 39: return float4(1.0 - s.a);\n"
"    case 40: return float4(d.a);          case 41: return float4(1.0 - d.a);\n"
"    case 42: { float m = min(s.a, 1.0 - d.a); return float4(m, m, m, 1.0); }\n"
"    case 43: return k;                    case 44: return 1.0 - k;\n"
"    case 45: return float4(k.a);          case 46: return float4(1.0 - k.a);\n"
"    default: return float4(1.0);\n"
"    }\n"
"}\n"
"static float4 bcombine(uint fn, float4 a, float4 b) {\n"
"    switch (fn) {\n"
"    case 0: return saturate(a + b);  case 1: return a + b;\n"
"    case 2: return saturate(a - b);  case 3: return a - b;\n"
"    case 4: return min(a, b);        case 5: return max(a, b);\n"
"    case 6: return saturate(b - a);  default: return b - a;\n"
"    }\n"
"}\n"
"static float4 r200_shade(VOut in, float4 fb, constant U &u,\n"
"                        texture2d<float> x0, texture2d<float> x1, texture2d<float> x2,\n"
"                        texture2d<float> x3, texture2d<float> x4, texture2d<float> x5,\n"
"                        sampler s0, sampler s1, sampler s2, sampler s3, sampler s4, sampler s5) {\n"
"    for (uint e = 0; e < u.nexcl.x; e++) {\n"
"        float4 r = u.excl[e];\n"
"        if (all(in.position.xy >= r.xy) && all(in.position.xy < r.zw)) discard_fragment();\n"
"    }\n"
"    float4 R[6];\n"
"    R[0] = (FC_TEXMASK & 1u) != 0u ? texel(x0, s0, in.t0, u.texinfo[0], u.texsize[0], u.texfilt[0].x) : float4(0.0);\n"
"    R[1] = (FC_TEXMASK & 2u) != 0u ? texel(x1, s1, in.t1, u.texinfo[1], u.texsize[1], u.texfilt[1].x) : float4(0.0);\n"
"    R[2] = (FC_TEXMASK & 4u) != 0u ? texel(x2, s2, in.t2, u.texinfo[2], u.texsize[2], u.texfilt[2].x) : float4(0.0);\n"
"    R[3] = (FC_TEXMASK & 8u) != 0u ? texel(x3, s3, in.t3, u.texinfo[3], u.texsize[3], u.texfilt[3].x) : float4(0.0);\n"
"    R[4] = (FC_TEXMASK & 16u) != 0u ? texel(x4, s4, in.t4, u.texinfo[4], u.texsize[4], u.texfilt[4].x) : float4(0.0);\n"
"    R[5] = (FC_TEXMASK & 32u) != 0u ? texel(x5, s5, in.t5, u.texinfo[5], u.texsize[5], u.texfilt[5].x) : float4(0.0);\n"
"    float4 dif = in.color, spc = in.spec;\n"
"    float4 cur = dif;\n"
"    bool any = false;\n"
"    for (uint st = 0; st < FC_STAGES; st++) {\n"
"        uint en = st == 7u ? 0x800u : (0x1000u << st);\n"
"        if ((u.pp_cntl & en) == 0u) continue;\n"
"        any = true;\n"
"        uint4 g = u.stage[st];\n"
"        float4 tf = u.tfactor[g.y & 7u], tf1 = u.tfactor[(g.y >> 4) & 7u];\n"
"        float3 A = cmod(carg(g.x & 31u, cur, dif, spc, tf, tf1, R), (g.x >> 16) & 15u);\n"
"        float3 B = cmod(carg((g.x >> 5) & 31u, cur, dif, spc, tf, tf1, R), (g.x >> 20) & 15u);\n"
"        float3 C = cmod(carg((g.x >> 10) & 31u, cur, dif, spc, tf, tf1, R), (g.x >> 24) & 15u);\n"
"        float3 rc;\n"
"        switch ((g.x >> 28) & 7u) {\n"
"        case 2: rc = select(B, A, C > 0.5); break;                 /* CND0 */\n"
"        case 3: rc = A * B + (1.0 - A) * C; break;                  /* LERP */\n"
"        case 4: rc = float3(dot(A, B)); break;                      /* DOT3 */\n"
"        case 5: rc = float3(dot(A, B)); break;                      /* DOT4 (approx) */\n"
"        case 6: rc = select(B, A, C > 0.5); break;                  /* CONDITIONAL */\n"
"        case 7: rc = float3(A.x * B.x + A.y * B.y + C.x); break;    /* DOT2_ADD */\n"
"        default: rc = A * B + C; break;                             /* MADD */\n"
"        }\n"
"        rc *= oscale(g.y);\n"
"        uint cc = (g.y >> 12) & 3u;\n"
"        rc = cc == 1u ? saturate(rc) : cc == 2u ? clamp(rc, -8.0, 8.0) : rc;\n"
"        float Aa = amod(aarg(g.z & 31u, cur, dif, spc, tf, tf1, R), (g.z >> 16) & 15u);\n"
"        float Ba = amod(aarg((g.z >> 5) & 31u, cur, dif, spc, tf, tf1, R), (g.z >> 20) & 15u);\n"
"        float Ca = amod(aarg((g.z >> 10) & 31u, cur, dif, spc, tf, tf1, R), (g.z >> 24) & 15u);\n"
"        float ra;\n"
"        switch ((g.z >> 28) & 7u) {\n"
"        case 2: case 6: ra = Ca > 0.5 ? Aa : Ba; break;\n"
"        case 3: ra = Aa * Ba + (1.0 - Aa) * Ca; break;\n"
"        default: ra = Aa * Ba + Ca; break;\n"
"        }\n"
"        ra *= oscale(g.w);\n"
"        uint ac = (g.w >> 12) & 3u;\n"
"        ra = ac == 1u ? saturate(ra) : ac == 2u ? clamp(ra, -8.0, 8.0) : ra;\n"
"        uint oc = (g.y >> 16) & 7u, oa = (g.w >> 16) & 7u;\n"
"        if (oc >= 1u && oc <= 6u) R[oc - 1u].rgb = rc;\n"
"        if (oa >= 1u && oa <= 6u) R[oa - 1u].a = ra;\n"
"        cur = float4(rc, ra);\n"
"    }\n"
"    float4 col = any ? R[0] : dif;\n"
"    if (u.pp_cntl & 0x200000u) col.rgb += spc.rgb;                /* specular */\n"
"    col = saturate(col);\n"
"    if (u.pp_cntl & 0x400000u) {                                   /* fog */\n"
"        uint src = uint(u.fog_color.w);\n"
"        float f = src == 2u ? dif.a : (src == 0u || src == 1u) ? 1.0 : spc.a;\n"
"        col.rgb = mix(u.fog_color.rgb, col.rgb, saturate(f));\n"
"    }\n"
"    if (u.pp_cntl & 0x800000u) {                                   /* alpha test */\n"
"        float ref = float(u.pp_misc & 0xffu) / 255.0;\n"
"        float a = col.a;\n"
"        bool pass;\n"
"        switch ((u.pp_misc >> 8) & 7u) {\n"
"        case 0: pass = false; break;       case 1: pass = a < ref; break;\n"
"        case 2: pass = a <= ref; break;    case 3: pass = a == ref; break;\n"
"        case 4: pass = a >= ref; break;    case 5: pass = a > ref; break;\n"
"        case 6: pass = a != ref; break;    default: pass = true; break;\n"
"        }\n"
"        if (!pass) discard_fragment();\n"
"    }\n"
"    if (u.rb3d_cntl & 1u) {                                        /* blend */\n"
"        float4 d = fb.yxwz;\n"
"        uint cb = u.cblend;\n"
"        uint ab = (u.rb3d_cntl & (1u << 16)) ? u.ablend : cb;\n"
"        float4 k = u.blend_color;\n"
"        float4 sc = col * bfactor((cb >> 16) & 63u, col, d, k);\n"
"        float4 dc = d * bfactor((cb >> 24) & 63u, col, d, k);\n"
"        float4 sa = col * bfactor((ab >> 16) & 63u, col, d, k);\n"
"        float4 da = d * bfactor((ab >> 24) & 63u, col, d, k);\n"
"        float3 rgb = bcombine((cb >> 12) & 7u, sc, dc).rgb;\n"
"        float alpha = bcombine((ab >> 12) & 7u, sa, da).a;\n"
"        col = saturate(float4(rgb, alpha));\n"
"    }\n"
"    if (u.nexcl.y != 0xffffffffu) {                              /* plane mask */\n"
"        float4 d = fb.yxwz;\n"
"        uint m = u.nexcl.y;\n"
"        col = float4((m & 0x00ff0000u) ? col.r : d.r, (m & 0x0000ff00u) ? col.g : d.g,\n"
"                     (m & 0x000000ffu) ? col.b : d.b, (m & 0xff000000u) ? col.a : d.a);\n"
"    }\n"
"    return col.yxwz;\n"
"}\n"
"#define R200_FS_ARGS constant U &u [[buffer(0)]], \\\n"
"    texture2d<float> x0 [[texture(0)]], texture2d<float> x1 [[texture(1)]], \\\n"
"    texture2d<float> x2 [[texture(2)]], texture2d<float> x3 [[texture(3)]], \\\n"
"    texture2d<float> x4 [[texture(4)]], texture2d<float> x5 [[texture(5)]], \\\n"
"    sampler s0 [[sampler(0)]], sampler s1 [[sampler(1)]], sampler s2 [[sampler(2)]], \\\n"
"    sampler s3 [[sampler(3)]], sampler s4 [[sampler(4)]], sampler s5 [[sampler(5)]]\n"
"fragment float4 r200_fs(VOut in [[stage_in]], float4 fb [[color(0)]], R200_FS_ARGS) {\n"
"    return r200_shade(in, fb, u, x0, x1, x2, x3, x4, x5, s0, s1, s2, s3, s4, s5);\n"
"}\n"
"/* Depth/stencil in VRAM: attachment 1 is the guest depth buffer as uint words\n"
" * in the CPU (big-endian) view — Z24S8 = stencil<<24 | z, or Z16. */\n"
"static uint bs32(uint v) { return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24); }\n"
"static uint bs16(uint v) { return ((v >> 8) & 0xffu) | ((v & 0xffu) << 8); }\n"
"static bool zcmp(uint f, uint a, uint b) {\n"
"    switch (f & 7u) {\n"
"    case 0: return false;  case 1: return a < b;   case 2: return a <= b;\n"
"    case 3: return a == b; case 4: return a >= b;  case 5: return a > b;\n"
"    case 6: return a != b; default: return true;\n"
"    }\n"
"}\n"
"static uint sop(uint op, uint s, uint ref) {\n"
"    switch (op & 7u) {\n"
"    case 1: return 0u;                      case 2: return ref;\n"
"    case 3: return min(s + 1u, 255u);       case 4: return s > 0u ? s - 1u : 0u;\n"
"    case 5: return ~s & 0xffu;              case 6: return (s + 1u) & 0xffu;\n"
"    case 7: return (s - 1u) & 0xffu;        default: return s;\n"
"    }\n"
"}\n"
"struct ZOut { float4 c [[color(0)]]; uint z [[color(1)]]; };\n"
"static ZOut ztest(float4 col, float4 fb, uint zb, float fz, constant U &u) {\n"
"    uint fl = u.zinfo.x, zc = u.zinfo.y, rm = u.zinfo.z;\n"
"    bool z16 = (fl & 8u) != 0u;\n"
"    uint v = z16 ? bs16(zb & 0xffffu) : bs32(zb);\n"
"    uint zold = z16 ? v : (v & 0xffffffu), sold = z16 ? 0u : (v >> 24);\n"
"    /* rint + clamp: 1.0*16777215 + 0.5 rounds to 2^24 in float and wraps to 0 */\n"
"    uint zmax = z16 ? 0xffffu : 0xffffffu;\n"
"    uint znew = min(uint(rint(saturate(fz) * float(zmax))), zmax);\n"
"    uint ref = rm & 0xffu, mask = (rm >> 16) & 0xffu, wmask = (rm >> 24) & 0xffu;\n"
"    ZOut o; o.c = col;\n"
"    uint z = zold, sn = sold;\n"
"    bool spass = true, zpass = true;\n"
"    if ((fl & 2u) != 0u) spass = zcmp(zc >> 12, ref & mask, sold & mask);\n"
"    if (!spass) {\n"
"        sn = sop(zc >> 16, sold, ref); o.c = fb;\n"
"    } else {\n"
"        if ((fl & 1u) != 0u) zpass = zcmp(zc >> 4, znew, zold);\n"
"        if (zpass) {\n"
"            if ((fl & 4u) != 0u) z = znew;\n"
"            if ((fl & 2u) != 0u) sn = sop(zc >> 20, sold, ref);\n"
"        } else {\n"
"            if ((fl & 2u) != 0u) sn = sop(zc >> 24, sold, ref);\n"
"            o.c = fb;\n"
"        }\n"
"    }\n"
"    sn = (sold & ~wmask) | (sn & wmask);\n"
"    o.z = z16 ? bs16(z & 0xffffu) : bs32((sn << 24) | (z & 0xffffffu));\n""    if (u.zinfo.w == 0x5a5au) o.z = bs32(0x55000000u | uint(saturate(fz) * 1000.0));\n"
"    return o;\n"
"}\n"
"fragment ZOut r200_fs_z(VOut in [[stage_in]], float4 fb [[color(0)]], uint zb [[color(1)]],\n"
"                        R200_FS_ARGS) {\n"
"    float4 col = r200_shade(in, fb, u, x0, x1, x2, x3, x4, x5, s0, s1, s2, s3, s4, s5);\n"
"    return ztest(col, fb, zb, in.position.z, u);\n"
"}\n"
"/* 16-bit colour targets (RB3D_CNTL colour format 3 ARGB1555, 4 RGB565,\n"
" * 15 ARGB4444): attachment 0 is an R16Uint view of the big-endian surface.\n"
" * r200_shade works on the raw BGRA8 layout, so convert in and out. */\n"
"static uint rt16fmt(uint rb3d) { uint c = (rb3d >> 10) & 15u; return c == 15u ? 5u : c; }\n"
"static uint pack16(float4 c, uint f) {\n"
"    c = saturate(c);\n"
"    if (f == 4u) return (uint(rint(c.r * 31.0)) << 11) | (uint(rint(c.g * 63.0)) << 5) |\n"
"                        uint(rint(c.b * 31.0));\n"
"    if (f == 3u) return (c.a >= 0.5 ? 0x8000u : 0u) | (uint(rint(c.r * 31.0)) << 10) |\n"
"                        (uint(rint(c.g * 31.0)) << 5) | uint(rint(c.b * 31.0));\n"
"    return (uint(rint(c.a * 15.0)) << 12) | (uint(rint(c.r * 15.0)) << 8) |\n"
"           (uint(rint(c.g * 15.0)) << 4) | uint(rint(c.b * 15.0));\n"
"}\n"
"struct C16Out { uint c [[color(0)]]; };\n"
"struct C16ZOut { uint c [[color(0)]]; uint z [[color(1)]]; };\n"
"fragment C16Out r200_fs16(VOut in [[stage_in]], uint fb [[color(0)]], R200_FS_ARGS) {\n"
"    uint f = rt16fmt(u.rb3d_cntl);\n"
"    float4 fbr = unpack16v(bs16(fb & 0xffffu), f).yxwz;\n"
"    float4 col = r200_shade(in, fbr, u, x0, x1, x2, x3, x4, x5, s0, s1, s2, s3, s4, s5);\n"
"    C16Out o; o.c = bs16(pack16(col.yxwz, f)); return o;\n"
"}\n"
"fragment C16ZOut r200_fs16_z(VOut in [[stage_in]], uint fb [[color(0)]], uint zb [[color(1)]],\n"
"                             R200_FS_ARGS) {\n"
"    uint f = rt16fmt(u.rb3d_cntl);\n"
"    float4 fbr = unpack16v(bs16(fb & 0xffffu), f).yxwz;\n"
"    float4 col = r200_shade(in, fbr, u, x0, x1, x2, x3, x4, x5, s0, s1, s2, s3, s4, s5);\n"
"    ZOut zo = ztest(col, fbr, zb, in.position.z, u);\n"
"    C16ZOut o; o.c = bs16(pack16(zo.c.yxwz, f)); o.z = zo.z; return o;\n"
"}\n"
"struct DOut { float depth [[depth(any)]]; };\n"
"vertex float4 r200_clear_vs(uint vid [[vertex_id]]) {\n"
"    float2 p = float2((vid << 1) & 2, vid & 2);\n"
"    return float4(p * 2.0 - 1.0, 0.0, 1.0);\n"
"}\n"
"fragment DOut r200_clear_fs(constant float &z [[buffer(0)]]) {\n"
"    DOut o; o.depth = z; return o;\n"
"}\n";

static id<MTLRenderPipelineState> g_r200_pipeline;
static id<MTLRenderPipelineState> g_r200_pipeline_ds;   /* unused (old private depth) */
static bool r200_zcheck_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        on = getenv("PPCGPU_ZCHECK") != NULL;
    }
    return on;
}

static id<MTLRenderPipelineState> g_r200_pipeline_z32;  /* + R32Uint depth in VRAM */
static id<MTLRenderPipelineState> g_r200_pipeline_z16;  /* + R16Uint depth in VRAM */
static id<MTLRenderPipelineState> g_r200_pipeline_c16;      /* 16-bit colour target */
static id<MTLRenderPipelineState> g_r200_pipeline_c16_z32;
static id<MTLRenderPipelineState> g_r200_pipeline_c16_z16;
static id<MTLRenderPipelineState> g_r200_clear_pipeline;

/*
 * Function-constant specialization of the uber-shader: a pass's fragment
 * function is recompiled with the number of combiner stages that run and
 * the bound texture units, so a two-stage, one-texture compositor draw no
 * longer pays for six texel() fetches and an eight-iteration stage loop.
 * Variants are indexed as the generic pipelines (below) and cached; the
 * generic pipeline remains the fallback.
 */
static id<MTLLibrary> g_r200_lib;
/* 0 fs, 1 fs_z+R32Uint, 2 fs_z+R16Uint, 3 fs16, 4 fs16_z+R32Uint,
 * 5 fs16_z+R16Uint */
static id<MTLRenderPipelineState> g_r200_pipes_gen[6];

#define R200_SPEC_SLOTS 256
static struct { uint32_t key; id<MTLRenderPipelineState> pso; } g_r200_spec[R200_SPEC_SLOTS];

static id<MTLRenderPipelineState> r200_spec_pso(id<MTLDevice> dev,
                                                uint32_t variant,
                                                uint32_t stages, uint32_t mask)
{
    static const char *const fname[6] = {
        "r200_fs", "r200_fs_z", "r200_fs_z", "r200_fs16", "r200_fs16_z", "r200_fs16_z",
    };
    static const MTLPixelFormat cfmt[6] = {
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm,
        MTLPixelFormatR16Uint, MTLPixelFormatR16Uint, MTLPixelFormatR16Uint,
    };
    static const MTLPixelFormat zfmt[6] = {
        MTLPixelFormatInvalid, MTLPixelFormatR32Uint, MTLPixelFormatR16Uint,
        MTLPixelFormatInvalid, MTLPixelFormatR32Uint, MTLPixelFormatR16Uint,
    };

    if (!g_r200_lib || variant >= 6) {
        return nil;
    }
    if (stages >= 8 && mask == 0x3F) {
        return g_r200_pipes_gen[variant];
    }
    uint32_t key = variant | stages << 3 | mask << 7;
    uint32_t slot = ((key * 2654435761u) >> 20) & (R200_SPEC_SLOTS - 1);
    for (uint32_t probe = 0; probe < 4; probe++) {
        uint32_t i = (slot + probe) & (R200_SPEC_SLOTS - 1);
        if (!g_r200_spec[i].pso) {
            break;
        }
        if (g_r200_spec[i].key == key) {
            return g_r200_spec[i].pso;
        }
    }
    @autoreleasepool {
        NSError *err = nil;
        MTLFunctionConstantValues *fcv = [[MTLFunctionConstantValues alloc] init];
        [fcv setConstantValue:&stages type:MTLDataTypeUInt atIndex:0];
        [fcv setConstantValue:&mask type:MTLDataTypeUInt atIndex:1];
        id<MTLFunction> vs = [g_r200_lib newFunctionWithName:@"r200_vs"];
        id<MTLFunction> fs = [g_r200_lib
            newFunctionWithName:[NSString stringWithUTF8String:fname[variant]]
                 constantValues:fcv error:&err];
        [fcv release];
        if (!vs || !fs) {
            [vs release];
            [fs release];
            return g_r200_pipes_gen[variant];
        }
        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = vs;
        pd.fragmentFunction = fs;
        pd.colorAttachments[0].pixelFormat = cfmt[variant];
        pd.colorAttachments[0].blendingEnabled = NO;
        if (zfmt[variant] != MTLPixelFormatInvalid) {
            pd.colorAttachments[1].pixelFormat = zfmt[variant];
        }
        id<MTLRenderPipelineState> p = bin_pso(dev, pd, &err);
        [pd release];
        [vs release];
        [fs release];
        if (!p) {
            return g_r200_pipes_gen[variant];
        }
        uint32_t store = slot;
        for (uint32_t probe = 0; probe < 4; probe++) {
            uint32_t i = (slot + probe) & (R200_SPEC_SLOTS - 1);
            if (!g_r200_spec[i].pso) {
                store = i;
                break;
            }
        }
        [g_r200_spec[store].pso release];
        g_r200_spec[store].key = key;
        g_r200_spec[store].pso = p;   /* table keeps the reference */
        return p;
    }
}
#define R200_DS_FORMAT MTLPixelFormatDepth32Float_Stencil8
static id<MTLSamplerState> g_r200_samplers[256];
static id<MTLTexture> g_r200_dummy;

static pthread_mutex_t g_r200_setup_lock = PTHREAD_MUTEX_INITIALIZER;

/* The fragment functions are specialised by FC_STAGES/FC_TEXMASK; the generic
 * pipelines get "all stages, all six texture units". */
static id<MTLFunction> r200_generic_fs(id<MTLLibrary> lib, NSString *name)
{
    MTLFunctionConstantValues *fcv = [[MTLFunctionConstantValues alloc] init];
    uint32_t stages = 8, mask = 0x3F;
    NSError *e = nil;

    [fcv setConstantValue:&stages type:MTLDataTypeUInt atIndex:0];
    [fcv setConstantValue:&mask type:MTLDataTypeUInt atIndex:1];
    id<MTLFunction> f = [lib newFunctionWithName:name constantValues:fcv error:&e];
    [fcv release];
    if (!f) {
        qemu_log("ppc-mac-gpu r200: %s: %s\n", [name UTF8String],
                 e ? [[e localizedDescription] UTF8String] : "?");
    }
    return f;
}

static bool r200_metal_setup_locked(id<MTLDevice> dev)
{
    if (g_r200_pipeline) {
        return true;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kR200ShaderSource
                                           options:nil error:&err];
    if (!lib) {
        qemu_log("ppc-mac-gpu r200: shader compile failed: %s\n",
                 err ? [[err localizedDescription] UTF8String] : "?");
        return false;
    }
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    id<MTLFunction> vs = [lib newFunctionWithName:@"r200_vs"];
    id<MTLFunction> fs = r200_generic_fs(lib, @"r200_fs");
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].blendingEnabled = NO;
    g_r200_pipeline = bin_pso(dev, pd, &err);
    {
        id<MTLFunction> zfs = r200_generic_fs(lib, @"r200_fs_z");
        MTLRenderPipelineDescriptor *zd = [pd copy];
        zd.fragmentFunction = zfs;
        zd.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;
        g_r200_pipeline_z32 = bin_pso(dev, zd, &err);
        zd.colorAttachments[1].pixelFormat = MTLPixelFormatR16Uint;
        g_r200_pipeline_z16 = bin_pso(dev, zd, &err);
        [zd release];
        [zfs release];
    }
    {
        id<MTLFunction> f16 = r200_generic_fs(lib, @"r200_fs16");
        id<MTLFunction> f16z = r200_generic_fs(lib, @"r200_fs16_z");
        MTLRenderPipelineDescriptor *cd = [pd copy];
        cd.colorAttachments[0].pixelFormat = MTLPixelFormatR16Uint;
        cd.fragmentFunction = f16;
        g_r200_pipeline_c16 = bin_pso(dev, cd, &err);
        cd.fragmentFunction = f16z;
        cd.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;
        g_r200_pipeline_c16_z32 = bin_pso(dev, cd, &err);
        cd.colorAttachments[1].pixelFormat = MTLPixelFormatR16Uint;
        g_r200_pipeline_c16_z16 = bin_pso(dev, cd, &err);
        if (!g_r200_pipeline_c16 || !g_r200_pipeline_c16_z32 || !g_r200_pipeline_c16_z16) {
            qemu_log("ppc-mac-gpu r200: 16-bit colour pipelines failed: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "?");
        }
        [cd release];
        [f16 release];
        [f16z release];
    }
    pd.depthAttachmentPixelFormat = R200_DS_FORMAT;
    pd.stencilAttachmentPixelFormat = R200_DS_FORMAT;
    g_r200_pipeline_ds = bin_pso(dev, pd, &err);
    [pd release];
    g_r200_pipes_gen[0] = g_r200_pipeline;
    g_r200_pipes_gen[1] = g_r200_pipeline_z32;
    g_r200_pipes_gen[2] = g_r200_pipeline_z16;
    g_r200_pipes_gen[3] = g_r200_pipeline_c16;
    g_r200_pipes_gen[4] = g_r200_pipeline_c16_z32;
    g_r200_pipes_gen[5] = g_r200_pipeline_c16_z16;
    {
        MTLRenderPipelineDescriptor *cd = [[MTLRenderPipelineDescriptor alloc] init];
        id<MTLFunction> cvs = [lib newFunctionWithName:@"r200_clear_vs"];
        id<MTLFunction> cfs = [lib newFunctionWithName:@"r200_clear_fs"];
        cd.vertexFunction = cvs;
        cd.fragmentFunction = cfs;
        cd.depthAttachmentPixelFormat = R200_DS_FORMAT;
        cd.stencilAttachmentPixelFormat = R200_DS_FORMAT;
        g_r200_clear_pipeline = bin_pso(dev, cd, &err);
        [cd release];
        [cvs release];
        [cfs release];
    }
    [vs release];
    [fs release];
    g_r200_lib = lib;           /* kept: r200_spec_pso specializes from it */
    if (!g_r200_pipeline_z16) {
        qemu_log("ppc-mac-gpu r200: 16-bit depth pipeline unavailable: %s\n",
                 err ? [[err localizedDescription] UTF8String] : "?");
    }
    if (!g_r200_pipeline || !g_r200_pipeline_z32) {
        qemu_log("ppc-mac-gpu r200: pipeline failed: %s\n",
                 err ? [[err localizedDescription] UTF8String] : "?");
        return false;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:1 height:1 mipmapped:NO];
    g_r200_dummy = [dev newTextureWithDescriptor:td];
    uint32_t zero = 0;
    [g_r200_dummy replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0
                      withBytes:&zero bytesPerRow:4];
    qemu_log("ppc-mac-gpu r200: direct renderer ready\n");
    return true;
}

/* Lazy pipeline compile, now reachable from two threads (the producer's
 * draw validation and the render thread's encode). */
static bool r200_metal_setup(id<MTLDevice> dev)
{
    bool ok;

    if (g_r200_pipeline) {
        return true;
    }
    pthread_mutex_lock(&g_r200_setup_lock);
    ok = r200_metal_setup_locked(dev);
    pthread_mutex_unlock(&g_r200_setup_lock);
    return ok;
}

static MTLSamplerAddressMode r200_addr_mode(uint32_t clamp)
{
    switch (clamp & 7) {
    case 0: return MTLSamplerAddressModeRepeat;
    case 1: return MTLSamplerAddressModeMirrorRepeat;
    case 3: case 7: return MTLSamplerAddressModeMirrorClampToEdge;
    case 4: case 5: return MTLSamplerAddressModeClampToBorderColor;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

static id<MTLSamplerState> r200_sampler(id<MTLDevice> dev, uint32_t filter)
{
    uint32_t mag = filter & 1, min = ((filter >> 1) & 0xF) ? 1 : 0;
    uint32_t cs = (filter >> 23) & 7, ct = (filter >> 27) & 7;
    uint32_t key = mag | (min << 1) | (cs << 2) | (ct << 5);
    if (!g_r200_samplers[key]) {
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.magFilter = mag ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.minFilter = min ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.sAddressMode = r200_addr_mode(cs);
        sd.tAddressMode = r200_addr_mode(ct);
        sd.borderColor = MTLSamplerBorderColorTransparentBlack;
        g_r200_samplers[key] = [dev newSamplerStateWithDescriptor:sd];
        [sd release];
    }
    return g_r200_samplers[key];
}

static uint32_t r200_metal_warned;

/*
 * Depth/stencil buffers.  Metal cannot alias a depth texture onto the VRAM
 * buffer, so each guest depth buffer (VRAM offset + pitch) gets a private
 * Depth32Float_Stencil8 texture.  Guest writes to depth memory arrive only
 * as 3D draws (rendered here) or 2D fills (mirrored by fill_notify_r200).
 */
#define R200_MAX_DEPTH 4
static struct {
    uint32_t offset, pitch, bpp;
    id<MTLTexture> tex;
    uint64_t used;
    bool fresh;                 /* never rendered: first pass clears it */
} g_r200_depth[R200_MAX_DEPTH];
static uint64_t g_r200_depth_clock;

static id<MTLTexture> r200_depth_tex(id<MTLDevice> dev, uint32_t offset,
                                     uint32_t pitch, uint32_t bpp, bool create)
{
    int lru = 0;
    for (int i = 0; i < R200_MAX_DEPTH; i++) {
        if (g_r200_depth[i].tex && g_r200_depth[i].offset == offset &&
            g_r200_depth[i].pitch == pitch) {
            g_r200_depth[i].used = ++g_r200_depth_clock;
            return g_r200_depth[i].tex;
        }
        if (g_r200_depth[i].used < g_r200_depth[lru].used) {
            lru = i;
        }
    }
    if (!create) {
        return nil;
    }
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:R200_DS_FORMAT width:pitch
                                    height:2048 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    if (!t) {
        return nil;
    }
    [g_r200_depth[lru].tex release];
    g_r200_depth[lru].offset = offset;
    g_r200_depth[lru].pitch = pitch;
    g_r200_depth[lru].bpp = bpp;
    g_r200_depth[lru].tex = t;
    g_r200_depth[lru].used = ++g_r200_depth_clock;
    g_r200_depth[lru].fresh = true;
    qemu_log("ppc-mac-gpu r200: depth buffer at 0x%x, pitch %u, %u-bit\n",
             offset, pitch, bpp * 8);
    return t;
}

static MTLCompareFunction r200_cmp(uint32_t f)
{
    static const MTLCompareFunction map[8] = {
        MTLCompareFunctionNever, MTLCompareFunctionLess,
        MTLCompareFunctionLessEqual, MTLCompareFunctionEqual,
        MTLCompareFunctionGreaterEqual, MTLCompareFunctionGreater,
        MTLCompareFunctionNotEqual, MTLCompareFunctionAlways,
    };
    return map[f & 7];
}

static MTLStencilOperation r200_sop(uint32_t f)
{
    static const MTLStencilOperation map[8] = {
        MTLStencilOperationKeep, MTLStencilOperationZero,
        MTLStencilOperationReplace, MTLStencilOperationIncrementClamp,
        MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
        MTLStencilOperationIncrementWrap, MTLStencilOperationDecrementWrap,
    };
    return map[f & 7];
}

#define R200_DSS_CACHE 64
static struct { uint64_t key; id<MTLDepthStencilState> st; } g_r200_dss[R200_DSS_CACHE];
static int g_r200_ndss;

static id<MTLDepthStencilState> r200_dss(id<MTLDevice> dev, const R200DrawPacket *pkt)
{
    uint32_t z = pkt->zstencil, rm = pkt->stencil_refmask;
    uint64_t key = ((uint64_t)(pkt->depth_enable | pkt->stencil_enable << 1) << 62) |
                   ((uint64_t)(z & 0x47FFF0F0u) << 24) | ((rm >> 16) & 0xFFFF);
    for (int i = 0; i < g_r200_ndss; i++) {
        if (g_r200_dss[i].key == key) {
            return g_r200_dss[i].st;
        }
    }
    MTLDepthStencilDescriptor *d = [[MTLDepthStencilDescriptor alloc] init];
    if (pkt->depth_enable) {
        d.depthCompareFunction = r200_cmp(z >> 4);
        d.depthWriteEnabled = (z >> 30) & 1;
    } else {
        d.depthCompareFunction = MTLCompareFunctionAlways;
        d.depthWriteEnabled = NO;
    }
    if (pkt->stencil_enable) {
        MTLStencilDescriptor *sd = [[MTLStencilDescriptor alloc] init];
        sd.stencilCompareFunction = r200_cmp(z >> 12);
        sd.stencilFailureOperation = r200_sop(z >> 16);
        sd.depthStencilPassOperation = r200_sop(z >> 20);
        sd.depthFailureOperation = r200_sop(z >> 24);
        sd.readMask = (rm >> 16) & 0xFF;
        sd.writeMask = (rm >> 24) & 0xFF;
        d.frontFaceStencil = sd;
        d.backFaceStencil = sd;
        [sd release];
    }
    id<MTLDepthStencilState> st = [dev newDepthStencilStateWithDescriptor:d];
    [d release];
    if (g_r200_ndss == R200_DSS_CACHE) {
        [g_r200_dss[0].st release];
        memmove(g_r200_dss, g_r200_dss + 1, sizeof(g_r200_dss[0]) * (R200_DSS_CACHE - 1));
        g_r200_ndss--;
    }
    g_r200_dss[g_r200_ndss].key = key;
    g_r200_dss[g_r200_ndss++].st = st;
    return st;
}

static void r200_metal_warn(uint32_t bit, const char *msg, uint32_t a, uint32_t b)
{
    if (!(r200_metal_warned & bit)) {
        r200_metal_warned |= bit;
        /* Each of these once, where they can be seen: they say which part
         * of a scene the renderer left out (a texture format it doesn't
         * know yet shows up as a missing surface). */
        fprintf(stderr, "ppc-mac-gpu r200: %s (0x%x, 0x%x)\n", msg, a, b);
    }
}

/*
 * Batching: draws are encoded into one open command buffer, and consecutive
 * draws to the same render target share one render pass.  Nothing waits on
 * the GPU until flush_r200(), which the device calls before anything reads
 * VRAM on the CPU side (2D engine ops, register reads, scanout).
 */
typedef struct R200TexKey {
    uint32_t offset, width, height, pitch, format;
} R200TexKey;

#define R200_VIEW_CACHE 64
static struct {
    R200TexKey key;
    id<MTLTexture> tex;
    uint64_t used;
} g_r200_views[R200_VIEW_CACHE];
static uint64_t g_r200_view_clock;

static id<MTLCommandBuffer> g_r200_cb;
static id<MTLCommandBuffer> g_r200_inflight;   /* newest committed, unwaited */
static id<MTLRenderCommandEncoder> g_r200_enc;
static R200TexKey g_r200_enc_key;
static bool g_r300_enc_mine;        /* open encoder belongs to draw_r300 */
static uint32_t g_r200_enc_variant;                 /* g_r200_pipes_gen index */
static id<MTLRenderPipelineState> g_r200_enc_pipe;  /* last set on the pass */
static uint32_t g_r200_enc_depth_off = ~0u, g_r200_enc_depth_pitch; /* ~0: none */
static bool g_r200_enc_depth_z16;
static uint64_t g_r200_stat_draws, g_r200_stat_passes, g_r200_stat_flushes,
                g_r200_stat_conflicts, g_r200_stat_view_hit, g_r200_stat_view_new;

/* VRAM ranges written by render passes in the open batch.  Views that alias
 * the same memory are distinct Metal objects, so Metal does not order a
 * write through one against a read through another: a draw that would read
 * (or re-target) written memory through a different view flushes first. */
#define R200_MAX_WRITTEN 128
static struct { uint64_t lo, hi; R200TexKey key; uint32_t epoch; } g_r200_written[R200_MAX_WRITTEN];
static int g_r200_nwritten;

/*
 * Instead of stopping the vCPU until the GPU is done, a conflict can end
 * the batch and start a new one that the GPU runs only after the previous
 * one completes (a shared event signalled at the end of each batch and
 * waited for at the start of the next).  The written ranges stay for the
 * CPU-side checks; conflicts only look at the current batch's (epoch).
 * PPCGPU_SPLIT=0 goes back to waiting.
 */
static uint32_t g_r200_epoch;
static id<MTLSharedEvent> g_r200_event;
static uint64_t g_r200_event_val;
static uint64_t g_r200_stat_splits;

static bool r200_split_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PPCGPU_SPLIT");
        on = !(e && e[0] == '0');
    }
    return on;
}

/*
 * --------------------------------------------------------------------
 * Render queue: Metal encoding off the BQL.
 *
 * The CP thread parses the ring under the BQL because registers, GART
 * translation and interrupts are QEMU state.  The Metal half of a draw
 * (view lookup, arena staging, the encoder, the commit) is not: it
 * touches only this file's state and the GPU.  draw_r200/draw_r300
 * therefore copy the decoded packet into a job and push it here, and
 * the render thread encodes jobs in FIFO order with no QEMU lock held,
 * so the vCPU, the CP thread and the GPU overlap completely.
 *
 * The ring itself is lock-free (one position counter per side, cells
 * claimed by sequence number).  The pthread primitives appear only
 * where a thread must sleep: producers when the ring is full (draw
 * order must stay FIFO, so a draw can never be encoded on the
 * producer's thread instead), the render thread when the ring is
 * empty, and drain waiters until every pushed job has been encoded.
 * A producer blocked on a full ring while holding the BQL cannot
 * deadlock: whoever holds g_render_lock holds the BQL too, finishes
 * without needing the queue, and the render thread drains.
 *
 * PPCGPU_RENDER_QUEUE=0 encodes synchronously on the caller's thread,
 * as before (the device then also keeps taking the BQL around draws).
 */

enum { RQ_R200 = 0, RQ_R300 = 1, RQ_COMMIT = 2 };

/* Defined at the bottom of the file; rqueue_start flags it. */
static PPCMacGPURenderer metal_renderer;

static pthread_mutex_t g_rq_mtx = PTHREAD_MUTEX_INITIALIZER;

/*
 * The VRAM ranges a queued job will touch, recorded when it is pushed so
 * that a CPU-side access to unrelated VRAM neither drains the queue nor
 * waits for the GPU (metal_range_busy_r200).  Overestimating is safe (an
 * unnecessary flush); underestimating would race, so anything that does
 * not fit marks the job as touching all of VRAM.
 */
#define RQ_MAX_WR (R300_US_MAX_TARGETS + 2)
#define RQ_MAX_RD R300_NUM_TEX_UNITS
typedef struct RQRanges {
    uint64_t wr[RQ_MAX_WR][2];
    uint64_t rd[RQ_MAX_RD][2];
    unsigned nwr, nrd;
    bool all;
} RQRanges;

typedef struct RQJob {
    int kind;
    void (*done)(void *, uint32_t);     /* RQ_COMMIT: the fence's completion */
    void *done_arg;
    uint32_t seq;
    R200DrawPacket r200;    /* copies of verts/indices; host_data taken over */
    R300DrawPacket r300;    /* the packet's owned buffers, taken over */
    RQRanges ranges;
    struct RQJob *next;     /* the free list */
} RQJob;

/* Jobs are recycled: the packets they embed are kilobytes each, and a
 * busy guest pushes thousands of draws a second. */
static RQJob *g_rq_pool;        /* g_rq_mtx */

static RQJob *rq_job_alloc(void)
{
    RQJob *job;

    pthread_mutex_lock(&g_rq_mtx);
    job = g_rq_pool;
    if (job) {
        g_rq_pool = job->next;
    }
    pthread_mutex_unlock(&g_rq_mtx);
    if (!job) {
        job = g_new(RQJob, 1);
    }
    return job;
}

static void rq_job_release(RQJob *job)
{
    pthread_mutex_lock(&g_rq_mtx);
    job->next = g_rq_pool;
    g_rq_pool = job;
    pthread_mutex_unlock(&g_rq_mtx);
}

static void r200_job_ranges(RQJob *job)
{
    const R200DrawPacket *p = &job->r200;
    RQRanges *r = &job->ranges;
    uint32_t cfmt = (p->rb3d_cntl >> 10) & 0xF;
    uint32_t bpr = p->rt_pitch * (cfmt == 3 || cfmt == 4 || cfmt == 15 ? 2 : 4);

    r->nwr = r->nrd = 0;
    r->all = false;
    if (bpr && p->rt_height) {
        r->wr[r->nwr][0] = p->rt_offset;
        r->wr[r->nwr++][1] = p->rt_offset + (uint64_t)bpr * p->rt_height;
    }
    if (p->depth_enable || p->stencil_enable) {
        if (r->nwr == RQ_MAX_WR) {
            r->all = true;
            return;
        }
        r->wr[r->nwr][0] = p->depth_offset;
        r->wr[r->nwr++][1] = p->depth_offset + (uint64_t)p->depth_pitch *
                             p->depth_bpp * p->rt_height;
    }
    for (int t = 0; t < R200_MAX_TEX; t++) {
        const R200TexUnit *tu = &p->tex[t];
        if (!tu->enabled || tu->host_data) {
            continue;
        }
        if (r->nrd == RQ_MAX_RD) {
            r->all = true;
            return;
        }
        r->rd[r->nrd][0] = tu->offset;
        r->rd[r->nrd++][1] = tu->offset + (uint64_t)tu->pitch * tu->height;
    }
}

static void r300_job_ranges(RQJob *job)
{
    const R300DrawPacket *p = &job->r300;
    RQRanges *r = &job->ranges;
    uint32_t ns = MIN(MAX(p->aa_samples, 1u), 6u);
    uint32_t ncb = MAX(p->num_cb, 1u);

    r->nwr = r->nrd = 0;
    r->all = false;
    if (ncb > R300_US_MAX_TARGETS) {
        r->all = true;
        return;
    }
    for (uint32_t k = 0; k < ncb; k++) {
        uint32_t addr = k ? p->cb[k].gpu_addr : p->rt_gpu_addr;
        uint64_t bpr = (uint64_t)ns * (k ? (uint64_t)p->cb[k].pitch * p->cb[k].bpp
                                         : (uint64_t)p->rt_pitch * p->rt_bpp);
        if (r->nwr == RQ_MAX_WR) {
            r->all = true;
            return;
        }
        r->wr[r->nwr][0] = addr;
        r->wr[r->nwr++][1] = addr + bpr * p->rt_height;
    }
    if (p->depth.attach) {
        if (r->nwr == RQ_MAX_WR) {
            r->all = true;
            return;
        }
        r->wr[r->nwr][0] = p->depth.gpu_addr;
        r->wr[r->nwr++][1] = p->depth.gpu_addr + (uint64_t)ns * p->depth.pitch *
                             p->depth.bpp * p->rt_height;
    }
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &p->tex[t];
        if (!td->bound || td->host_data) {
            continue;
        }
        if (r->nrd == RQ_MAX_RD) {
            r->all = true;
            return;
        }
        r->rd[r->nrd][0] = td->gpu_addr;
        r->rd[r->nrd++][1] = td->gpu_addr + td->size_bytes;
    }
}

static bool rq_ranges_busy(const RQRanges *r, uint64_t lo, uint64_t hi,
                           bool write_access)
{
    if (r->all) {
        return true;
    }
    for (unsigned i = 0; i < r->nwr; i++) {
        if (lo < r->wr[i][1] && r->wr[i][0] < hi) {
            return true;
        }
    }
    if (write_access) {
        for (unsigned i = 0; i < r->nrd; i++) {
            if (lo < r->rd[i][1] && r->rd[i][0] < hi) {
                return true;
            }
        }
    }
    return false;
}

/*
 * The queued job takes the packet's heap buffers over (nulling them in
 * pkt, whose owner then frees only what is left): copying kilobytes of
 * vertices, texels and shader text per draw -- twice, producer and again
 * here -- was pure allocator and memcpy traffic.  The R200 vertices and
 * indices are the device's reusable scratch, so those are still copied.
 */
static RQJob *r200_job_new(R200DrawPacket *pkt)
{
    RQJob *job = rq_job_alloc();

    job->kind = RQ_R200;
    job->r200 = *pkt;
    job->r200.verts = g_memdup2(pkt->verts,
                                (size_t)pkt->num_verts * sizeof(R200Vertex));
    job->r200.indices = g_memdup2(pkt->indices,
                                  (size_t)pkt->num_indices * 4);
    for (int t = 0; t < R200_MAX_TEX; t++) {
        pkt->tex[t].host_data = NULL;
    }
    r200_job_ranges(job);
    return job;
}

static void r200_job_free(RQJob *job)
{
    g_free((void *)job->r200.verts);
    g_free((void *)job->r200.indices);
    for (int t = 0; t < R200_MAX_TEX; t++) {
        g_free((void *)job->r200.tex[t].host_data);
    }
    rq_job_release(job);
}

/* The R300 packet's owned fields (r300_draw_free's list) are taken over;
 * the device's r300_draw_free then frees only what was not queued. */
static RQJob *r300_job_new(R300DrawPacket *pkt)
{
    RQJob *job = rq_job_alloc();
    R300DrawPacket *p = &job->r300;

    job->kind = RQ_R300;
    *p = *pkt;
    pkt->glsl = NULL;
    pkt->vs_glsl = NULL;
    pkt->verts = NULL;
    pkt->vs_in = NULL;
    pkt->vs_idx = NULL;
    pkt->vs_u = NULL;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        pkt->tex[t].host_data = NULL;
    }
    r300_job_ranges(job);
    return job;
}

static void r300_job_free(RQJob *job)
{
    R300DrawPacket *p = &job->r300;

    free(p->glsl);
    free(p->vs_glsl);
    free(p->verts);
    free(p->vs_in);
    free(p->vs_idx);
    free(p->vs_u);
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        free(p->tex[t].host_data);
    }
    rq_job_release(job);
}

#define RQ_BITS 10
#define RQ_SIZE (1u << RQ_BITS)
#define RQ_MASK (RQ_SIZE - 1)
typedef struct { RQJob *job; RQRanges ranges; uintptr_t seq; } RQCell;
static RQCell g_rq_cells[RQ_SIZE];
static uintptr_t g_rq_enq, g_rq_deq;    /* next position to fill / to take */
static uintptr_t g_rq_done;             /* jobs encoded and freed */
/* The job being encoded right now: its ranges move here when the cell is
 * freed, and stay until the encode has recorded them in g_r200_written /
 * g_r200_read (g_rq_mtx guards the handover and the range scan). */
static RQRanges g_rq_enc_ranges;
static bool g_rq_enc_valid;
static pthread_cond_t g_rq_space = PTHREAD_COND_INITIALIZER;   /* ring not full */
static pthread_cond_t g_rq_work = PTHREAD_COND_INITIALIZER;    /* work arrived */
static bool g_rq_sleeping;      /* the render thread waits on g_rq_work */
static pthread_cond_t g_rq_idle = PTHREAD_COND_INITIALIZER;    /* all encoded */
static pthread_t g_rq_thread;
static bool g_rq_thread_started;
static volatile bool g_rq_stop;
static PPCMacGPUMetalState *g_rq_st;
static uint64_t g_rq_stat_jobs, g_rq_stat_full;

static bool rqueue_wanted(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PPCGPU_RENDER_QUEUE");
        on = !(e && e[0] == '0');
    }
    return on;
}

static bool rqueue_enabled(void)
{
    return rqueue_wanted() && g_rq_thread_started;
}

/* Wait until every job pushed so far has been encoded and freed.  Never
 * called while holding g_render_lock (the render thread is what makes
 * progress), so this cannot self-deadlock. */
static void rqueue_drain(void)
{
    pthread_mutex_lock(&g_rq_mtx);
    while (qatomic_read(&g_rq_done) != qatomic_read(&g_rq_enq)) {
        pthread_cond_wait(&g_rq_idle, &g_rq_mtx);
    }
    pthread_mutex_unlock(&g_rq_mtx);
}

static int r200_draw_encode(PPCMacGPUMetalState *st, uint8_t *vram_ptr,
                            uint64_t vram_size, const R200DrawPacket *pkt);
static int r300_draw_encode(PPCMacGPUMetalState *st, uint8_t *vram_ptr,
                            uint64_t vram_size, const R300DrawPacket *pkt);
static id<MTLCommandBuffer> r200_new_cb(PPCMacGPUMetalState *st);
static uint32_t r200_commit_as(void (*done)(void *, uint32_t), void *arg,
                               uint32_t forced_seq);

/* Take the next job, or NULL when the ring is empty.  One consumer;
 * g_rq_mtx held, so a range scan never sees a job in neither the cell
 * nor the encoding slot. */
static RQJob *rq_pop_locked(void)
{
    uintptr_t pos = qatomic_read(&g_rq_deq);
    RQCell *c = &g_rq_cells[pos & RQ_MASK];

    /* Acquire pairs with rq_push's release: once seq says the cell is
     * full, its job pointer and ranges are the ones just published (ARM
     * reorders plain accesses; reading a stale pointer freed it twice). */
    if ((uintptr_t)qatomic_load_acquire(&c->seq) != pos + 1) {
        return NULL;
    }
    RQJob *job = qatomic_read(&c->job);
    g_rq_enc_ranges = c->ranges;
    g_rq_enc_valid = true;
    qatomic_store_release(&c->seq, pos + RQ_SIZE);   /* the cell is free again */
    qatomic_set(&g_rq_deq, pos + 1);
    return job;
}

static RQJob *rq_pop(void)
{
    RQJob *job;

    pthread_mutex_lock(&g_rq_mtx);
    job = rq_pop_locked();
    pthread_mutex_unlock(&g_rq_mtx);
    return job;
}

static void *rqueue_thread(void *unused)
{
    PPCMacGPUMetalState *st = g_rq_st;

    for (;;) {
        RQJob *job = rq_pop();
        if (!job) {
            /*
             * Say we are going to sleep, then look once more: a producer
             * publishes its cell and then checks g_rq_sleeping, each side
             * with a full barrier between its store and its load, so at
             * least one of us sees the other (see rq_push).
             */
            pthread_mutex_lock(&g_rq_mtx);
            for (;;) {
                qatomic_set(&g_rq_sleeping, true);
                smp_mb();
                if (g_rq_stop || (job = rq_pop_locked())) {
                    break;
                }
                pthread_cond_wait(&g_rq_work, &g_rq_mtx);
            }
            qatomic_set(&g_rq_sleeping, false);
            pthread_mutex_unlock(&g_rq_mtx);
            if (!job) {
                break;      /* stopped, and the ring is empty */
            }
        }
        pthread_mutex_lock(&g_render_lock);
        if (job->kind == RQ_COMMIT) {
            /* An empty batch still completes in queue order, which is all
             * the fence needs when nothing was drawn since the last one. */
            if (!g_r200_cb) {
                g_r200_cb = r200_new_cb(st);
            }
            r200_commit_as(job->done, job->done_arg, job->seq);
        } else if (job->kind == RQ_R200) {
            r200_draw_encode(st, st->vram_ptr, st->vram_size, &job->r200);
        } else {
            r300_draw_encode(st, st->vram_ptr, st->vram_size, &job->r300);
        }
        pthread_mutex_unlock(&g_render_lock);
        pthread_mutex_lock(&g_rq_mtx);
        /* The encode recorded the job's writes and reads in the batch
         * trackers, which the range checks see from here on. */
        g_rq_enc_valid = false;
        g_rq_done++;
        pthread_cond_broadcast(&g_rq_idle);
        pthread_cond_broadcast(&g_rq_space);
        pthread_mutex_unlock(&g_rq_mtx);
        if (job->kind == RQ_COMMIT) {
            rq_job_release(job);
        } else if (job->kind == RQ_R200) {
            r200_job_free(job);
        } else {
            r300_job_free(job);
        }
    }
    return NULL;
}

static void rq_push(RQJob *job)
{
    uintptr_t pos = qatomic_fetch_add(&g_rq_enq, (uintptr_t)1);
    RQCell *c = &g_rq_cells[pos & RQ_MASK];

    /* Full: wait for the render thread to finish a job.  FIFO order
     * means this draw cannot be encoded here instead. */
    while ((uintptr_t)qatomic_load_acquire(&c->seq) != pos) {
        pthread_mutex_lock(&g_rq_mtx);
        if ((uintptr_t)qatomic_load_acquire(&c->seq) != pos) {
            g_rq_stat_full++;
            pthread_cond_wait(&g_rq_space, &g_rq_mtx);
        }
        pthread_mutex_unlock(&g_rq_mtx);
    }
    qatomic_set(&c->job, job);
    c->ranges = job->ranges;
    qatomic_store_release(&c->seq, pos + 1);    /* publishes job and ranges */
    /*
     * Wake the render thread if it is (about to be) asleep.  This used to
     * test "the ring was empty" (pos == g_rq_deq) with a plain load after
     * the seq store, which ARM may satisfy first: the producer read a stale
     * g_rq_deq and skipped the wakeup while the render thread missed the
     * new cell and slept, and with the ring no longer empty no later push
     * woke it either.  The queue then never drained (a 2D blit waiting in
     * rqueue_drain hung Doom 3).  The barrier pairs with the one in
     * rqueue_thread; holding g_rq_mtx for the broadcast means it cannot
     * fall between the render thread's last look and its wait.
     */
    smp_mb();
    if (qatomic_read(&g_rq_sleeping)) {
        pthread_mutex_lock(&g_rq_mtx);
        pthread_cond_broadcast(&g_rq_work);
        pthread_mutex_unlock(&g_rq_mtx);
    }
    g_rq_stat_jobs++;
}

static bool rqueue_start(PPCMacGPUMetalState *st)
{
    if (!rqueue_wanted()) {
        return false;
    }
    for (uint32_t i = 0; i < RQ_SIZE; i++) {
        g_rq_cells[i].seq = i;
        g_rq_cells[i].job = NULL;
    }
    g_rq_enq = g_rq_deq = g_rq_done = 0;
    g_rq_enc_valid = false;
    g_rq_st = st;
    g_rq_stop = false;
    if (pthread_create(&g_rq_thread, NULL, rqueue_thread, NULL) != 0) {
        qemu_log("ppc-mac-gpu-metal: render queue thread failed to start; "
                 "encoding on the caller's thread\n");
        return false;
    }
    g_rq_thread_started = true;
    metal_renderer.draw_queue_async = true;
    qemu_log("ppc-mac-gpu-metal: render queue on (%u draws deep, "
             "Metal encode off the BQL)\n", RQ_SIZE);
    return true;
}

static void rqueue_stop(void)
{
    if (!g_rq_thread_started) {
        return;
    }
    pthread_mutex_lock(&g_rq_mtx);
    g_rq_stop = true;
    pthread_cond_broadcast(&g_rq_work);
    pthread_mutex_unlock(&g_rq_mtx);
    /* The thread exits only with the ring empty, so this is a drain. */
    pthread_join(g_rq_thread, NULL);
    g_rq_thread_started = false;
    metal_renderer.draw_queue_async = false;
    qemu_log("ppc-mac-gpu-metal: render queue off: %llu jobs, "
             "%llu full-waits\n",
             (unsigned long long)g_rq_stat_jobs,
             (unsigned long long)g_rq_stat_full);
}

/* A new command buffer, ordered after everything committed before it. */
static id<MTLCommandBuffer> r200_new_cb(PPCMacGPUMetalState *st)
{
    id<MTLCommandBuffer> cb = [[st->commandQueue commandBuffer] retain];
    if (g_r200_event && g_r200_event_val) {
        [cb encodeWaitForEvent:g_r200_event value:g_r200_event_val];
    }
    return cb;
}

/*
 * Vertex staging.
 *
 * Every draw hands Metal the vertices it expanded from the guest's index
 * list.  Anything over a few dozen vertices is too big to pass inline, and
 * a game sends hundreds of draws a frame, so allocating a buffer per draw
 * means millions of allocations a session.  Bump-allocate out of one large
 * buffer instead, and put it back in the pool once the command buffer that
 * referenced it has completed.
 */
#define R200_ARENA_SIZE (8 * 1024 * 1024)

static NSMutableArray *g_r200_arena_free;     /* idle, ready to reuse */
static NSMutableArray *g_r200_arena_used;     /* referenced by the open batch */
static id<MTLBuffer> g_r200_arena;
static size_t g_r200_arena_off;
static pthread_mutex_t g_r200_arena_lock = PTHREAD_MUTEX_INITIALIZER;

/* Space for @len bytes, or NULL when a one-off buffer is needed instead. */
static void *r200_arena_alloc(PPCMacGPUMetalState *st, size_t len,
                              id<MTLBuffer> *buf, size_t *off)
{
    len = (len + 255) & ~(size_t)255;         /* keep offsets aligned */
    if (len > R200_ARENA_SIZE) {
        return NULL;
    }
    if (!g_r200_arena || g_r200_arena_off + len > [g_r200_arena length]) {
        if (g_r200_arena) {
            if (!g_r200_arena_used) {
                g_r200_arena_used = [[NSMutableArray alloc] init];
            }
            [g_r200_arena_used addObject:g_r200_arena];
            [g_r200_arena release];
            g_r200_arena = nil;
        }
        pthread_mutex_lock(&g_r200_arena_lock);
        id<MTLBuffer> reuse = [g_r200_arena_free lastObject];
        if (reuse) {
            g_r200_arena = [reuse retain];
            [g_r200_arena_free removeLastObject];
        }
        pthread_mutex_unlock(&g_r200_arena_lock);
        if (!g_r200_arena) {
            /* newBuffer... already returns a reference we own. */
            g_r200_arena = [st->device newBufferWithLength:R200_ARENA_SIZE
                                options:MTLResourceStorageModeShared];
            if (!g_r200_arena) {
                return NULL;
            }
        }
        g_r200_arena_off = 0;
    }
    *buf = g_r200_arena;
    *off = g_r200_arena_off;
    void *p = (char *)[g_r200_arena contents] + g_r200_arena_off;
    g_r200_arena_off += len;
    return p;
}

/* Hand this batch's staging buffers back once the GPU is done with them. */
static void r200_arena_recycle_on(id<MTLCommandBuffer> cb)
{
    NSMutableArray *mine = g_r200_arena_used;
    g_r200_arena_used = nil;
    if (g_r200_arena) {
        if (!mine) {
            mine = [[NSMutableArray alloc] init];
        }
        [mine addObject:g_r200_arena];
        [g_r200_arena release];
        g_r200_arena = nil;
        g_r200_arena_off = 0;
    }
    if (!mine) {
        return;
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        pthread_mutex_lock(&g_r200_arena_lock);
        if (!g_r200_arena_free) {
            g_r200_arena_free = [[NSMutableArray alloc] init];
        }
        /* Keep a few around; a deep batch's extras can go. */
        for (id<MTLBuffer> b in mine) {
            if ([g_r200_arena_free count] < 8) {
                [g_r200_arena_free addObject:b];
            }
        }
        pthread_mutex_unlock(&g_r200_arena_lock);
        [mine release];
    }];
}

static void r200_note_written(uint64_t lo, uint64_t hi, const R200TexKey *key)
{
    int w = 0;
    while (w < g_r200_nwritten && memcmp(&g_r200_written[w].key, key, sizeof(*key))) {
        w++;
    }
    if (w < g_r200_nwritten) {
        g_r200_written[w].epoch = g_r200_epoch;       /* written again in this batch */
        /* the R300 notes each draw's scissored rows: grow to cover them */
        g_r200_written[w].lo = MIN(g_r200_written[w].lo, lo);
        g_r200_written[w].hi = MAX(g_r200_written[w].hi, hi);
        return;
    }
    if (w == R200_MAX_WRITTEN) {
        return;                /* full: the next draw flushes (see the conflict check) */
    }
    g_r200_written[w].lo = lo;
    g_r200_written[w].hi = hi;
    g_r200_written[w].key = *key;
    g_r200_written[w].epoch = g_r200_epoch;
    g_r200_nwritten++;
}

/* VRAM ranges read (as textures) by draws in the open or in-flight batches. */
#define R200_MAX_READ 64
static struct { uint64_t lo, hi; } g_r200_read[R200_MAX_READ];
static int g_r200_nread;
static bool g_r200_read_overflow;

static void r200_note_read(uint64_t lo, uint64_t hi)
{
    for (int i = 0; i < g_r200_nread; i++) {
        if (g_r200_read[i].lo == lo && g_r200_read[i].hi == hi) {
            return;
        }
    }
    if (g_r200_nread == R200_MAX_READ) {
        g_r200_read_overflow = true;         /* be conservative */
        return;
    }
    g_r200_read[g_r200_nread].lo = lo;
    g_r200_read[g_r200_nread++].hi = hi;
}

static bool metal_range_busy_r200_locked(uint64_t lo, uint64_t hi,
                                         bool write_access);

static bool metal_range_busy_r200(void *opaque, uint64_t lo, uint64_t hi,
                                  bool write_access)
{
    /*
     * Queued and mid-encode draws say which VRAM ranges they touch, so an
     * access to unrelated memory keeps the GPU running instead of draining
     * the queue and waiting for it.  A cell is only read while its seq says
     * "published at this position", checked again afterwards: a producer
     * mid-push or a wraparound lands in one of the checks and is treated
     * conservatively.
     */
    if (g_rq_thread_started) {
        bool busy = false;
        pthread_mutex_lock(&g_rq_mtx);
        if (g_rq_enc_valid && rq_ranges_busy(&g_rq_enc_ranges, lo, hi,
                                             write_access)) {
            busy = true;
        }
        uintptr_t deq = qatomic_read(&g_rq_deq);
        uintptr_t enq = qatomic_read(&g_rq_enq);
        for (uintptr_t pos = deq; pos < enq && !busy; pos++) {
            RQCell *c = &g_rq_cells[pos & RQ_MASK];
            if ((uintptr_t)qatomic_load_acquire(&c->seq) != pos + 1) {
                busy = true;          /* being published right now */
                break;
            }
            if (rq_ranges_busy(&c->ranges, lo, hi, write_access)) {
                busy = true;
            }
            if ((uintptr_t)qatomic_read(&c->seq) != pos + 1) {
                busy = true;          /* recycled while being read */
            }
        }
        pthread_mutex_unlock(&g_rq_mtx);
        if (busy) {
            return true;
        }
    }
    pthread_mutex_lock(&g_render_lock);
    bool busy = metal_range_busy_r200_locked(lo, hi, write_access);
    pthread_mutex_unlock(&g_render_lock);
    return busy;
}

static bool metal_range_busy_r200_locked(uint64_t lo, uint64_t hi,
                                         bool write_access)
{
    if (!g_r200_cb && !g_r200_inflight) {
        return false;
    }
    for (int i = 0; i < g_r200_nwritten; i++) {
        if (lo < g_r200_written[i].hi && g_r200_written[i].lo < hi) {
            return true;
        }
    }
    if (write_access) {
        if (g_r200_read_overflow) {
            return true;
        }
        for (int i = 0; i < g_r200_nread; i++) {
            if (lo < g_r200_read[i].hi && g_r200_read[i].lo < hi) {
                return true;
            }
        }
    }
    return false;
}

static bool r200_batch_conflict(uint64_t lo, uint64_t hi, const R200TexKey *same)
{
    for (int i = 0; i < g_r200_nwritten; i++) {
        if (g_r200_written[i].epoch != g_r200_epoch) {
            continue;          /* an earlier batch: the GPU orders it before this one */
        }
        if (lo < g_r200_written[i].hi && g_r200_written[i].lo < hi &&
            !(same && !memcmp(same, &g_r200_written[i].key, sizeof(*same)))) {
            return true;
        }
    }
    return false;
}
static int64_t g_r200_stat_flush_us;

/* Bytes per pixel of the formats r200_view is used with; 0 = unknown. */
static unsigned r200_pf_bytes(MTLPixelFormat pf)
{
    switch (pf) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Uint:
    case MTLPixelFormatA8Unorm:
        return 1;
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Uint:
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Float: case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm: case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
        return 2;
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatRGBA8Uint: case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Float: case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatBGR10A2Unorm:
        return 4;
    case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Float:
        return 8;
    case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        return 0;
    }
}

/*
 * Linear texture view over VRAM, cached; views stay alive while cached.
 *
 * The geometry comes from guest registers, and Metal aborts the whole
 * process (not just fails the call) on a linear texture whose rows don't
 * fit its bytesPerRow, a misaligned pitch/offset, or one running past the
 * buffer.  So check first.  A sampled texture that doesn't fit is shrunk
 * to what does (sampling past it clamps); a render target isn't, since
 * callers size viewports and scissors from the guest's numbers: it gets
 * no view, and the draw is skipped with a warning.
 */
static id<MTLTexture> r200_view(PPCMacGPUMetalState *st, R200TexKey k,
                                MTLPixelFormat pf, bool render_target)
{
    int lru = 0;
    for (int i = 0; i < R200_VIEW_CACHE; i++) {
        if (g_r200_views[i].tex && !memcmp(&g_r200_views[i].key, &k, sizeof(k))) {
            g_r200_views[i].used = ++g_r200_view_clock;
            g_r200_stat_view_hit++;
            return g_r200_views[i].tex;
        }
        if (g_r200_views[i].used < g_r200_views[lru].used) {
            lru = i;
        }
    }
    uint64_t w = k.width, h = k.height, bpp = r200_pf_bytes(pf);
    uint64_t align = [st->device minimumLinearTextureAlignmentForPixelFormat:pf];
    uint64_t len = st->vramBuffer.length;
    const char *why = NULL;

    if (!w || !h || !k.pitch) {
        why = "empty";
    } else if (align && (k.pitch % align || k.offset % align)) {
        why = "pitch/offset not aligned for Metal";
    } else if (k.offset >= len) {
        why = "offset outside VRAM";
    } else if (bpp) {
        if (w * bpp > k.pitch) {
            w = k.pitch / bpp;             /* rows wider than the pitch */
        }
        if (k.offset + (h - 1) * k.pitch + w * bpp > len) {
            h = (len - k.offset - w * bpp) / k.pitch + 1;   /* past the end */
        }
        if (!w || !h || k.offset + w * bpp > len) {
            why = "does not fit";
        } else if (render_target && (w != k.width || h != k.height)) {
            why = "render target does not fit";
        }
    }
    if (why || w != k.width || h != k.height) {
        static int logged;
        if (logged < 20) {
            logged++;
            qemu_log("ppc-mac-gpu-metal: VRAM view %ux%u pitch %u offset 0x%x "
                     "fmt %lu: %s\n", k.width, k.height, k.pitch, k.offset,
                     (unsigned long)pf,
                     why ? why : "clamped to fit the pitch/VRAM");
        }
        if (why) {
            return nil;
        }
    }
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pf width:w height:h
                                 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead |
              (render_target ? MTLTextureUsageRenderTarget : 0);
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [st->vramBuffer newTextureWithDescriptor:d
                                                          offset:k.offset
                                                     bytesPerRow:k.pitch];
    if (!t) {
        return nil;
    }
    /* An evicted view may still be referenced by the open command buffer;
     * the command buffer retains what it uses, so releasing is safe. */
    g_r200_stat_view_new++;
    [g_r200_views[lru].tex release];
    g_r200_views[lru].key = k;
    g_r200_views[lru].tex = t;
    g_r200_views[lru].used = ++g_r200_view_clock;
    return t;
}

static uint32_t g_r200_seq;

/* Sequence numbers come from the CP thread (async fences) and the render
 * thread (internal commits): allocate them atomically. */
static uint32_t r200_next_seq(void)
{
    uint32_t seq = qatomic_fetch_inc(&g_r200_seq) + 1;

    if (seq == 0) {
        seq = qatomic_fetch_inc(&g_r200_seq) + 1;
    }
    return seq;
}

/* r200_decode_tex_unit()'s AGP copy is pitch*height (DXT: block rows). */
static uint64_t r200_tex_host_bytes(const R200TexUnit *t)
{
    uint64_t len = (uint64_t)t->pitch * t->height;
    if (t->format == 12 || t->format == 14 || t->format == 15) {
        len = (uint64_t)t->pitch * ((t->height + 3) / 4);
    }
    return len;
}

/* POWEREMU_TEX_TRACE, cached: getenv() per draw was a linear environ scan. */
static bool metal_tex_trace(void)
{
    static int on = -1;
    if (on < 0) {
        on = getenv("POWEREMU_TEX_TRACE") != NULL;
    }
    return on;
}

/*
 * Textures that cannot be views over VRAM -- AGP/host copies, converted
 * YUV frames, DXT blocks -- were rebuilt and re-uploaded every draw.
 * Cache them by a hash of their source bytes instead: an unchanged
 * texture (a glyph atlas, a menu, a paused video frame) then costs a
 * hash, not a conversion and an upload.  g_render_lock serialises all
 * users (the encode paths).
 */
#define AUX_TEX_CACHE 32
typedef struct {
    uint64_t hash;
    uint32_t k[6];
    id<MTLTexture> tex;         /* retained */
    uint64_t used;
} AuxTex;
static AuxTex g_aux_tex[AUX_TEX_CACHE];
static uint64_t g_aux_tex_clock;

static id<MTLTexture> aux_tex_find(uint64_t hash, const uint32_t k[6])
{
    for (int i = 0; i < AUX_TEX_CACHE; i++) {
        if (g_aux_tex[i].tex && g_aux_tex[i].hash == hash &&
            !memcmp(g_aux_tex[i].k, k, sizeof(g_aux_tex[i].k))) {
            g_aux_tex[i].used = ++g_aux_tex_clock;
            return g_aux_tex[i].tex;
        }
    }
    return nil;
}

static void aux_tex_store(uint64_t hash, const uint32_t k[6], id<MTLTexture> t)
{
    int lru = 0;

    for (int i = 0; i < AUX_TEX_CACHE; i++) {
        if (g_aux_tex[i].used < g_aux_tex[lru].used) {
            lru = i;
        }
    }
    [g_aux_tex[lru].tex release];
    g_aux_tex[lru].hash = hash;
    memcpy(g_aux_tex[lru].k, k, sizeof(g_aux_tex[lru].k));
    g_aux_tex[lru].tex = [t retain];
    g_aux_tex[lru].used = ++g_aux_tex_clock;
}

/* Finish pending GPU writes to VRAM [lo, hi) before the CPU reads it. */
static bool r200_flush_locked(PPCMacGPUMetalState *st);

static void aux_tex_sync_vram(PPCMacGPUMetalState *st, uint64_t lo, uint64_t hi)
{
    if (metal_range_busy_r200_locked(lo, hi, false)) {
        r200_flush_locked(st);
    }
}

/* Close the open batch and commit it; returns its sequence number or 0. */
static uint32_t r200_commit_as(void (*done)(void *, uint32_t), void *arg,
                               uint32_t forced_seq)
{
    if (!g_r200_cb) {
        return 0;
    }
    if (g_r200_enc) {
        [g_r200_enc endEncoding];
        [g_r200_enc release];
        g_r200_enc = nil;
    }
    g_r200_enc_depth_off = ~0u;
    uint32_t seq = forced_seq ? forced_seq : r200_next_seq();
    if (done) {
        [g_r200_cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            done(arg, seq);
        }];
    }
    if (g_r200_event) {
        [g_r200_cb encodeSignalEvent:g_r200_event value:++g_r200_event_val];
    }
    r200_arena_recycle_on(g_r200_cb);
    [g_r200_cb commit];
    [g_r200_inflight release];
    g_r200_inflight = g_r200_cb;           /* keeps the reference */
    g_r200_cb = nil;
    /* Committed work is ordered on the queue, but memory it writes may
     * still be read by later batches through other views: keep the written
     * ranges until everything in flight is known complete. */
    return seq;
}

static uint32_t r200_commit(void (*done)(void *, uint32_t), void *arg)
{
    return r200_commit_as(done, arg, 0);
}

static uint32_t metal_submit_r200(void *opaque,
                                  void (*done)(void *, uint32_t), void *arg)
{
    uint32_t seq;

    if (rqueue_enabled() &&
        qatomic_read(&g_rq_done) != qatomic_read(&g_rq_enq)) {
        /*
         * Draws are still queued.  Waiting for the render thread to encode
         * them (rqueue_drain) while the CP thread holds the BQL starved the
         * vCPU: in Doom 3 the guest spent a third of its time waiting for
         * the lock.  Queue the commit behind them and hand back its
         * sequence number; the render thread commits in order and the fence
         * fires from the batch's completion.  (Releasing the BQL around the
         * wait instead is unsafe: the vCPU then changes register state
         * under a half-processed packet.)
         */
        RQJob *job = rq_job_alloc();

        memset(&job->ranges, 0, sizeof(job->ranges));   /* touches nothing */
        job->kind = RQ_COMMIT;
        job->done = done;
        job->done_arg = arg;
        job->seq = r200_next_seq();
        seq = job->seq;
        rq_push(job);
        return seq;
    }
    pthread_mutex_lock(&g_render_lock);
    seq = r200_commit(done, arg);
    pthread_mutex_unlock(&g_render_lock);
    return seq;
}

/* Commit and wait: the encode paths call this with g_render_lock held. */
static bool r200_flush_locked(PPCMacGPUMetalState *st)
{
    if (!g_r200_cb && !g_r200_inflight) {
        return false;
    }
    int64_t t0 = g_get_monotonic_time();
    r200_commit(NULL, NULL);
    [g_r200_inflight waitUntilCompleted];
    if (g_r200_inflight.status == MTLCommandBufferStatusError) {
        r200_metal_warn(32, "command buffer failed", 0, 0);
    }
    [g_r200_inflight release];
    g_r200_inflight = nil;
    g_r200_nwritten = 0;
    g_r200_nread = 0;
    g_r200_read_overflow = false;
    g_r200_stat_flushes++;
    g_r200_stat_flush_us += g_get_monotonic_time() - t0;
    return true;
}

static bool metal_flush_r200(void *opaque)
{
    PPCMacGPUMetalState *st = opaque;
    bool did;

    /* The queued draws have to be encoded before anything can wait for
     * them; drain first, then take the lock (never in the other order:
     * the render thread is what drains). */
    if (rqueue_enabled()) {
        rqueue_drain();
    }
    pthread_mutex_lock(&g_render_lock);
    did = r200_flush_locked(st);
    pthread_mutex_unlock(&g_render_lock);
    return did;
}

/* A 2D fill that lands on a guest depth buffer clears our private copy. */
static void metal_fill_notify_r200_locked(void *opaque, uint32_t offset,
                                          uint32_t pitch, uint32_t x, uint32_t y,
                                          uint32_t w, uint32_t h, uint32_t bpp,
                                          uint32_t value);

static void metal_fill_notify_r200(void *opaque, uint32_t offset, uint32_t pitch,
                                   uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                                   uint32_t bpp, uint32_t value)
{
    /* Touches the open batch (g_r200_cb, encoders): serialise against
     * the render thread. */
    pthread_mutex_lock(&g_render_lock);
    metal_fill_notify_r200_locked(opaque, offset, pitch, x, y, w, h, bpp,
                                  value);
    pthread_mutex_unlock(&g_render_lock);
}

static void metal_fill_notify_r200_locked(void *opaque, uint32_t offset,
                                          uint32_t pitch, uint32_t x, uint32_t y,
                                          uint32_t w, uint32_t h, uint32_t bpp,
                                          uint32_t value)
{
    PPCMacGPUMetalState *st = opaque;
    /* Depth lives in VRAM (attachment 1), so 2D fills already cleared it. */
    if (true || !st || !st->vramBuffer || !g_r200_clear_pipeline) {
        return;
    }
    for (int i = 0; i < R200_MAX_DEPTH; i++) {
        if (!g_r200_depth[i].tex || g_r200_depth[i].offset != offset ||
            g_r200_depth[i].pitch * g_r200_depth[i].bpp != pitch) {
            continue;
        }
        id<MTLTexture> dt = g_r200_depth[i].tex;
        bool z16 = g_r200_depth[i].bpp == 2;
        float z = z16 ? (value & 0xFFFF) / 65535.0f
                      : (value & 0xFFFFFF) / 16777215.0f;
        uint32_t sten = z16 ? 0 : value >> 24;
        uint32_t x1 = MIN(x + w, (uint32_t)dt.width), y1 = MIN(y + h, (uint32_t)dt.height);
        if (x >= x1 || y >= y1) {
            return;
        }
        @autoreleasepool {
            if (!g_r200_cb) {
                g_r200_cb = r200_new_cb(st);
            }
            if (g_r200_enc) {
                [g_r200_enc endEncoding];
                [g_r200_enc release];
                g_r200_enc = nil;
            }
            g_r200_enc_depth_off = ~0u;
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            bool fresh = g_r200_depth[i].fresh;
            g_r200_depth[i].fresh = false;
            rp.depthAttachment.texture = dt;
            rp.depthAttachment.loadAction = fresh ? MTLLoadActionClear : MTLLoadActionLoad;
            rp.depthAttachment.clearDepth = 1.0;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            rp.stencilAttachment.texture = dt;
            rp.stencilAttachment.clearStencil = 0;
            rp.stencilAttachment.loadAction = fresh ? MTLLoadActionClear : MTLLoadActionLoad;
            rp.stencilAttachment.storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> e = [g_r200_cb renderCommandEncoderWithDescriptor:rp];
            R200DrawPacket q = { 0 };
            q.depth_enable = 1;
            q.zstencil = (7u << 4) | (1u << 30);               /* ALWAYS, write */
            if (!z16) {
                q.stencil_enable = 1;
                q.zstencil |= (7u << 12) | (2u << 16) | (2u << 20) | (2u << 24);
                q.stencil_refmask = 0xFFFF0000u | sten;         /* replace, all bits */
            }
            [e setRenderPipelineState:g_r200_clear_pipeline];
            [e setDepthStencilState:r200_dss(st->vramBuffer.device, &q)];
            [e setStencilReferenceValue:sten];
            [e setViewport:(MTLViewport){ 0, 0, dt.width, dt.height, 0, 1 }];
            [e setScissorRect:(MTLScissorRect){ x, y, x1 - x, y1 - y }];
            [e setFragmentBytes:&z length:sizeof(z) atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [e endEncoding];
        }
        static int logged;
        if (logged++ < 5) {
            qemu_log("ppc-mac-gpu r200: depth clear at 0x%x (%u,%u %ux%u) "
                     "z=%.3f stencil=%u\n", offset, x, y, w, h, z, sten);
        }
        return;
    }
}

static int r200_draw_encode(PPCMacGPUMetalState *st, uint8_t *vram_ptr,
                            uint64_t vram_size, const R200DrawPacket *pkt)
{
    if (!st || !st->vramBuffer) {
        return -1;
    }
    id<MTLDevice> dev = st->vramBuffer.device;
    if (!r200_metal_setup(dev)) {
        return -1;
    }
    /* 16-bit colour targets (3/4/15) are handled below with an R16Uint
     * view; anything else is skipped there. */

    uint32_t sx0 = pkt->scissor[0], sy0 = pkt->scissor[1];
    uint32_t sx1 = MIN(pkt->scissor[2], pkt->rt_width);
    uint32_t sy1 = MIN(pkt->scissor[3], pkt->rt_height);
    if (sx0 >= sx1 || sy0 >= sy1) {
        return 0;
    }

    uint32_t cfmt = (pkt->rb3d_cntl >> 10) & 0xF;
    bool rt16 = cfmt == 3 || cfmt == 4 || cfmt == 15;
    if (cfmt != 6 && !rt16) {
        r200_metal_warn(0x4000, "colour buffer format not supported", cfmt,
                        pkt->rt_offset);
        return -1;
    }
    if (rt16 && !g_r200_pipeline_c16) {
        return -1;
    }
    MTLPixelFormat rtpf = rt16 ? MTLPixelFormatR16Uint : MTLPixelFormatBGRA8Unorm;
    NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:rtpf];
    uint32_t bpr = pkt->rt_pitch * (rt16 ? 2 : 4);
    if ((pkt->rt_offset % align) || (bpr % align)) {
        r200_metal_warn(2, "render target not aligned for a linear view",
                        pkt->rt_offset, bpr);
        return -1;
    }

    @autoreleasepool {
        R200Uniforms u;
        memset(&u, 0, sizeof(u));
        u.rt_size[0] = pkt->rt_width;
        u.rt_size[1] = pkt->rt_height;
        u.pp_cntl = pkt->pp_cntl;
        u.pp_misc = pkt->pp_misc;
        u.rb3d_cntl = pkt->rb3d_cntl;
        u.cblend = pkt->cblend;
        u.ablend = pkt->ablend;
        u.num_excl = MIN(pkt->num_excl, 3u);
        u.plane_mask = pkt->plane_mask;
        u.fog_color[0] = ((pkt->fog_color >> 16) & 0xFF) / 255.0f;
        u.fog_color[1] = ((pkt->fog_color >> 8) & 0xFF) / 255.0f;
        u.fog_color[2] = (pkt->fog_color & 0xFF) / 255.0f;
        u.fog_color[3] = (float)((pkt->fog_color >> 25) & 7);
        for (uint32_t e = 0; e < u.num_excl; e++) {
            for (int k = 0; k < 4; k++) {
                u.excl[e][k] = pkt->excl[e][k];
            }
        }
        for (int i = 0; i < 4; i++) {
            u.blend_color[i] = ((pkt->blend_color >> (i == 3 ? 24 : 16 - 8 * i)) & 0xFF) / 255.0f;
        }
        for (int i = 0; i < R200_MAX_STAGES; i++) {
            u.stage[i][0] = pkt->txcblend[i];
            u.stage[i][1] = pkt->txcblend2[i];
            u.stage[i][2] = pkt->txablend[i];
            u.stage[i][3] = pkt->txablend2[i];
        }
        for (int i = 0; i < 8; i++) {
            uint32_t v = pkt->tfactor[i];      /* ARGB */
            u.tfactor[i][0] = ((v >> 16) & 0xFF) / 255.0f;
            u.tfactor[i][1] = ((v >> 8) & 0xFF) / 255.0f;
            u.tfactor[i][2] = (v & 0xFF) / 255.0f;
            u.tfactor[i][3] = (v >> 24) / 255.0f;
        }

        id<MTLTexture> tex[R200_MAX_TEX];
        id<MTLSamplerState> smp[R200_MAX_TEX];
        for (int t = 0; t < R200_MAX_TEX; t++) {
            const R200TexUnit *tu = &pkt->tex[t];
            tex[t] = g_r200_dummy;
            smp[t] = r200_sampler(dev, 0);
            if (!tu->enabled) {
                continue;
            }
            if (tu->format == 10 || tu->format == 11) {
                /*
                 * YUV 4:2:2 (QuickTime movie frames: Warcraft III's
                 * cinematics, Tiger's welcome movie, Halo's opening logos).
                 * Undo the TXOFFSET swap, then read the bytes as they lie.
                 *
                 * Which comes first, chroma or luma, is not the format
                 * number alone.  Frames the driver has put in VRAM are
                 * always '2vuy' (Cb Y0 Cr Y1): Tiger's welcome movie
                 * (format 0xca, first texels a8 58 62 59) and Warcraft
                 * III's cinematics both decode correctly that way.  A frame
                 * the program keeps in its own memory and lets the card
                 * read over AGP (client storage, as video players do) is
                 * laid out as the program wrote it, and format 10 there is
                 * 'yuvs' (Y0 Cb Y1 Cr): Halo's second logo reads
                 * e6 7f e6 80, which is near-white as 'yuvs' and the
                 * magenta we used to show as '2vuy'.  Its first logo, in
                 * format 11 over AGP, is '2vuy' and was always right.
                 * BT.601 video range into a BGRA texture.
                 */
                uint32_t w = tu->width, h = tu->height;
                bool luma_first = tu->host_data && tu->format == 10;
                if (!tu->host_data &&
                    (uint64_t)tu->offset + (uint64_t)tu->pitch * h > vram_size) {
                    r200_metal_warn(0x8000, "YUV texture outside VRAM", tu->offset, h);
                    continue;
                }
                const uint8_t *src = tu->host_data ? tu->host_data : vram_ptr + tu->offset;
                uint64_t slen = (uint64_t)tu->pitch * h;
                if (!tu->host_data) {
                    aux_tex_sync_vram(st, tu->offset, tu->offset + slen);
                }
                uint64_t yhash = r300_hash_bytes(src, slen);
                uint32_t akey[6] = { 0, tu->format,
                                     tu->swap | (luma_first ? 8u : 0) |
                                     (tu->host_data ? 16u : 0),
                                     w, h, tu->pitch };
                id<MTLTexture> yt = aux_tex_find(yhash, akey);
                if (!yt) {
                    uint8_t *out = g_malloc((size_t)w * h * 4);
                    for (uint32_t y = 0; y < h; y++) {
                        const uint8_t *row = src + (uint64_t)y * tu->pitch;
                        uint8_t *o = out + (size_t)y * w * 4;
                        for (uint32_t x = 0; x + 1 < w + 1; x += 2) {
                            uint8_t b[4] = { row[x * 2], row[x * 2 + 1],
                                             row[x * 2 + 2], row[x * 2 + 3] };
                            uint8_t g[4];
                            switch (tu->swap) {
                            case 1: g[0] = b[1]; g[1] = b[0]; g[2] = b[3]; g[3] = b[2]; break;
                            case 2: g[0] = b[3]; g[1] = b[2]; g[2] = b[1]; g[3] = b[0]; break;
                            case 3: g[0] = b[2]; g[1] = b[3]; g[2] = b[0]; g[3] = b[1]; break;
                            default: memcpy(g, b, 4); break;
                            }
                            int cb, y0, cr, y1;
                            if (luma_first) {       /* 'yuvs': Y0 Cb Y1 Cr */
                                y0 = g[0]; cb = g[1]; y1 = g[2]; cr = g[3];
                            } else {                /* '2vuy': Cb Y0 Cr Y1 */
                                cb = g[0]; y0 = g[1]; cr = g[2]; y1 = g[3];
                            }
                            for (int k = 0; k < 2 && x + k < w; k++) {
                                float yy = 1.164f * ((k ? y1 : y0) - 16);
                                float r = yy + 1.596f * (cr - 128);
                                float gg = yy - 0.813f * (cr - 128) - 0.391f * (cb - 128);
                                float bb = yy + 2.018f * (cb - 128);
                                uint8_t *p8 = o + (x + k) * 4;
                                p8[0] = (uint8_t)MIN(MAX(bb, 0.0f), 255.0f);
                                p8[1] = (uint8_t)MIN(MAX(gg, 0.0f), 255.0f);
                                p8[2] = (uint8_t)MIN(MAX(r, 0.0f), 255.0f);
                                p8[3] = 255;
                            }
                        }
                    }
                    if (metal_tex_trace()) {
                        static uint64_t last_kind;
                        static int yuv_drawn;
                        uint64_t kind = ((uint64_t)tu->format << 40) | ((uint64_t)tu->swap << 36)
                                      | ((uint64_t)w << 20) | ((uint64_t)h << 4)
                                      | (tu->host_data ? 1 : 0);
                        if (kind != last_kind && yuv_drawn++ < 40) {
                            last_kind = kind;
                            fprintf(stderr, "ppc-mac-gpu yuv draw: fmt %u %ux%u swap %u%s "
                                    "src=%02x %02x %02x %02x -> bgra=%02x %02x %02x %02x\n",
                                    tu->format, w, h, tu->swap,
                                    luma_first ? " (AGP, luma first)" :
                                    tu->host_data ? " (AGP)" : " (VRAM)",
                                    src[0], src[1], src[2], src[3],
                                    out[0], out[1], out[2], out[3]);
                        }
                    }
                    MTLTextureDescriptor *yd = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                        width:w height:h mipmapped:NO];
                    yd.usage = MTLTextureUsageShaderRead;
                    yt = [dev newTextureWithDescriptor:yd];
                    [yt replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                            withBytes:out bytesPerRow:w * 4];
                    g_free(out);
                    if (yt) {
                        aux_tex_store(yhash, akey, yt);
                    }
                    [yt autorelease];
                }
                tex[t] = yt;
                smp[t] = r200_sampler(dev, tu->filter);
                u.texinfo[t][0] = 1;
                u.texinfo[t][1] = 100;      /* already true RGBA */
                u.texinfo[t][2] = 1;
                u.texinfo[t][3] = tu->denorm;
                u.texsize[t][0] = w;
                u.texsize[t][1] = h;
                u.texsize[t][2] = 1.0f / w;
                u.texsize[t][3] = 1.0f / h;
                u.texfilt[t][0] = tu->filter;

                continue;
            }
            if (tu->format == 12 || tu->format == 14 || tu->format == 15) {
                /*
                 * DXT1/3/5.  Compressed textures cannot be linear views of
                 * the VRAM buffer, so copy the blocks into a BC texture.
                 *
                 * The blocks are copied as they lie, wherever they live.
                 * They used to be byte-swapped on the reasoning that VRAM
                 * holds the big-endian CPU's view, and that turned every
                 * surface in Halo into coloured speckle.  Reading one of
                 * its textures straight out of VRAM settles it: decoded
                 * untouched it is a metal hull with panels and a hatch,
                 * while swapped by words, by halfwords, or reversed it is
                 * noise.  The driver hands the card the blocks from the
                 * game's files unchanged, and the card reads bytes.
                 */
                uint32_t bs = tu->format == 12 ? 8 : 16;
                uint32_t bw = (tu->width + 3) / 4, bh = (tu->height + 3) / 4;
                uint32_t row = bw * bs;
                /*
                 * The blocks of one row, end to end.  The pitch register is
                 * not it: for DXT1 the guest leaves the figure it would use
                 * for 16-byte blocks, so it reads twice too large (Halo's
                 * 2048x128 texture says 8192 where the rows are 4096 apart),
                 * and following it skipped every other row of blocks and ran
                 * off the end -- the whole title screen came out speckled.
                 * A padded texture would say exactly what it means, so only
                 * an exact match is taken.
                 */
                if (tu->pitch == row) {
                    row = tu->pitch;
                }
                if (!dev.supportsBCTextureCompression ||
                    (!tu->host_data &&
                     (uint64_t)tu->offset + (uint64_t)row * bh > vram_size)) {
                    r200_metal_warn(64, "DXT texture unusable", tu->format, tu->offset);
                    continue;
                }
                const uint8_t *dsrc = tu->host_data ? tu->host_data
                                                    : vram_ptr + tu->offset;
                if (!tu->host_data) {
                    aux_tex_sync_vram(st, tu->offset,
                                      tu->offset + (uint64_t)row * bh);
                }
                uint64_t dhash = r300_hash_bytes(dsrc, (size_t)row * bh);
                uint32_t dkey[6] = { 1, tu->format, tu->host_data ? 1u : 0u,
                                     bw * 4, bh * 4, row };
                id<MTLTexture> ct = aux_tex_find(dhash, dkey);
                if (!ct) {
                    uint32_t *blk = g_malloc((size_t)bw * bs * bh);
                    for (uint32_t y = 0; y < bh; y++) {
                        memcpy(blk + y * (bw * bs / 4), dsrc + (uint64_t)y * row,
                               bw * bs);
                    }
                    MTLTextureDescriptor *cd = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:
                            tu->format == 12 ? MTLPixelFormatBC1_RGBA :
                            tu->format == 14 ? MTLPixelFormatBC2_RGBA : MTLPixelFormatBC3_RGBA
                        width:bw * 4 height:bh * 4 mipmapped:NO];
                    cd.usage = MTLTextureUsageShaderRead;
                    ct = [dev newTextureWithDescriptor:cd];
                    [ct replaceRegion:MTLRegionMake2D(0, 0, bw * 4, bh * 4) mipmapLevel:0
                            withBytes:blk bytesPerRow:bw * bs];
                    g_free(blk);
                    if (ct) {
                        aux_tex_store(dhash, dkey, ct);
                    }
                    [ct autorelease];      /* the cache or command buffer retains it */
                }
                tex[t] = ct;
                smp[t] = r200_sampler(dev, tu->filter);
                u.texinfo[t][0] = 1;
                u.texinfo[t][1] = 100;      /* already true RGBA */
                u.texinfo[t][2] = 1;
                u.texinfo[t][3] = tu->denorm;
                u.texsize[t][0] = tu->width;
                u.texsize[t][1] = tu->height;
                u.texsize[t][2] = 1.0f / tu->width;
                u.texsize[t][3] = 1.0f / tu->height;
                u.texfilt[t][0] = tu->filter;
                static int dxt_logged;
                if (dxt_logged++ < 3 ||
                    (metal_tex_trace() && dxt_logged < 12)) {
                    fprintf(stderr, "ppc-mac-gpu dxt: DXT%u %ux%u pitch %u at 0x%x%s\n",
                            tu->format == 12 ? 1 : tu->format == 14 ? 3 : 5,
                            tu->width, tu->height, row, tu->offset,
                            tu->host_data ? " (AGP)" : " (VRAM)");
                }
                continue;
            }
            MTLPixelFormat pf;
            uint32_t bpp;
            switch (tu->format) {
            case 0: case 8: pf = MTLPixelFormatR8Unorm;    bpp = 1; break;
            case 1: case 3: case 4: case 5:
                            pf = MTLPixelFormatRG8Unorm;   bpp = 2; break;
            case 7: case 22: pf = MTLPixelFormatBGRA8Unorm; bpp = 4; break;
            case 6:         pf = MTLPixelFormatBGRA8Unorm; bpp = 4; break;
            default:
                r200_metal_warn(8, "texture format not supported yet",
                                tu->format, t);
                continue;
            }
            if (tu->host_data) {
                /* AGP texture: the copied texels, cached by content */
                uint64_t hlen = r200_tex_host_bytes(tu);
                uint64_t hhash = r300_hash_bytes(tu->host_data, hlen);
                uint32_t hkey[6] = { 2, (uint32_t)pf, tu->format,
                                     tu->width, tu->height, tu->pitch };
                id<MTLTexture> ht = aux_tex_find(hhash, hkey);
                if (!ht) {
                    MTLTextureDescriptor *hd = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:pf width:tu->width
                                                    height:tu->height mipmapped:NO];
                    hd.usage = MTLTextureUsageShaderRead;
                    ht = [dev newTextureWithDescriptor:hd];
                    [ht replaceRegion:MTLRegionMake2D(0, 0, tu->width, tu->height)
                          mipmapLevel:0 withBytes:tu->host_data bytesPerRow:tu->pitch];
                    if (ht) {
                        aux_tex_store(hhash, hkey, ht);
                    }
                    [ht autorelease];
                }
                tex[t] = ht;
                smp[t] = r200_sampler(dev, tu->filter);
                u.texinfo[t][0] = 1;
                u.texinfo[t][1] = tu->format;
                u.texinfo[t][2] = tu->alpha_in_map;
                u.texinfo[t][3] = tu->denorm;
                u.texsize[t][0] = tu->width;
                u.texsize[t][1] = tu->height;
                u.texsize[t][2] = 1.0f / tu->width;
                u.texsize[t][3] = 1.0f / tu->height;
                u.texfilt[t][0] = tu->filter;
                continue;
            }
            NSUInteger talign = [dev minimumLinearTextureAlignmentForPixelFormat:pf];
            if ((tu->offset % talign) || (tu->pitch % talign) ||
                tu->pitch < tu->width * bpp ||
                (uint64_t)tu->offset + (uint64_t)tu->pitch * tu->height > vram_size) {
                r200_metal_warn(16, "texture view unusable (offset, pitch)",
                                tu->offset, tu->pitch);
                continue;
            }
            R200TexKey k = { tu->offset, tu->width, tu->height, tu->pitch,
                             (uint32_t)pf };
            id<MTLTexture> x = r200_view(st, k, pf, false);
            if (!x) {
                continue;
            }
            tex[t] = x;
            smp[t] = r200_sampler(dev, tu->filter);
            u.texinfo[t][0] = 1;
            u.texinfo[t][1] = tu->format;
            u.texinfo[t][2] = tu->alpha_in_map;
            u.texinfo[t][3] = tu->denorm;
            u.texfilt[t][0] = tu->filter;
            u.texsize[t][0] = tu->width;
            u.texsize[t][1] = tu->height;
            u.texsize[t][2] = 1.0f / tu->width;
            u.texsize[t][3] = 1.0f / tu->height;
        }

        /* Open (or continue) the batch and the render pass for this target. */
        R200TexKey rk = { pkt->rt_offset, pkt->rt_width, pkt->rt_height, bpr,
                          (uint32_t)rtpf };
        uint64_t rlo = pkt->rt_offset, rhi = rlo + (uint64_t)bpr * pkt->rt_height;
        bool full = g_r200_nwritten == R200_MAX_WRITTEN;
        bool conflict = r200_batch_conflict(rlo, rhi, &rk);
        for (int t = 0; t < R200_MAX_TEX && !conflict; t++) {
            const R200TexUnit *tu = &pkt->tex[t];
            if (u.texinfo[t][0] && !tu->host_data) {
                conflict = r200_batch_conflict(tu->offset, tu->offset +
                               (uint64_t)tu->pitch * tu->height, NULL);
            }
        }
        if (full) {
            r200_flush_locked(st);                 /* forget everything written */
        } else if (conflict) {
            g_r200_stat_conflicts++;
            if (r200_split_enabled()) {
                if (!g_r200_event) {
                    g_r200_event = [st->device newSharedEvent];
                }
                r200_commit(NULL, NULL);           /* the GPU keeps the order */
                g_r200_epoch++;
                g_r200_stat_splits++;
            } else {
                r200_flush_locked(st);
            }
        }
        if (!g_r200_cb) {
            g_r200_cb = r200_new_cb(st);
        }
        bool want_ds = pkt->depth_enable || pkt->stencil_enable;
        bool enc_ds = g_r200_enc_depth_off != ~0u;
        if (g_r200_enc &&
            (g_r300_enc_mine || memcmp(&g_r200_enc_key, &rk, sizeof(rk)) ||
             (want_ds && (!enc_ds || g_r200_enc_depth_off != pkt->depth_offset ||
                          g_r200_enc_depth_pitch != pkt->depth_pitch)))) {
            [g_r200_enc endEncoding];
            [g_r200_enc release];
            g_r200_enc = nil;
        }
        if (!g_r200_enc) {
            id<MTLTexture> rt = r200_view(st, rk, rtpf, true);
            if (!rt) {
                r200_metal_warn(4, "render target view failed", pkt->rt_offset, bpr);
                return -1;
            }
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = rt;
            rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLTexture> dtex = nil;
            MTLPixelFormat zpf = pkt->depth_bpp == 2 ? MTLPixelFormatR16Uint
                                                     : MTLPixelFormatR32Uint;
            if (want_ds) {
                /* guest depth buffer, in place, as a second colour attachment */
                R200TexKey dk = { pkt->depth_offset, pkt->rt_width, pkt->rt_height,
                                  pkt->depth_pitch * pkt->depth_bpp, (uint32_t)zpf };
                NSUInteger za = [dev minimumLinearTextureAlignmentForPixelFormat:zpf];
                if ((pkt->depth_bpp == 2 && !g_r200_pipeline_z16) ||
                    pkt->depth_pitch < pkt->rt_width || (dk.offset % za) ||
                    (dk.pitch % za) ||
                    (uint64_t)dk.offset + (uint64_t)dk.pitch * dk.height > vram_size) {
                    r200_metal_warn(128, "depth buffer not usable in place",
                                    pkt->depth_offset, pkt->depth_pitch);
                } else {
                    dtex = r200_view(st, dk, zpf, true);
                }
                if (dtex) {
                    uint64_t dlo = dk.offset, dhi = dlo + (uint64_t)dk.pitch * dk.height;
                    r200_note_written(dlo, dhi, &dk);
                }
            }
            if (dtex) {
                rp.colorAttachments[1].texture = dtex;
                rp.colorAttachments[1].loadAction = MTLLoadActionLoad;
                rp.colorAttachments[1].storeAction = MTLStoreActionStore;
                g_r200_enc_depth_off = pkt->depth_offset;
                g_r200_enc_depth_pitch = pkt->depth_pitch;
                g_r200_enc_depth_z16 = pkt->depth_bpp == 2;
            } else {
                g_r200_enc_depth_off = ~0u;
            }
            g_r200_enc = [[g_r200_cb renderCommandEncoderWithDescriptor:rp] retain];
            g_r300_enc_mine = false;
            /* The R200 does not clip on z for these draws: a depth clear drawn
             * exactly at z = 1.0 (Chess) would be clipped away by Metal. */
            [g_r200_enc setDepthClipMode:MTLDepthClipModeClamp];
            g_r200_enc_variant = rt16 ? (!dtex ? 3 : pkt->depth_bpp == 2 ? 5 : 4)
                                      : (!dtex ? 0 : pkt->depth_bpp == 2 ? 2 : 1);
            g_r200_enc_pipe = g_r200_pipes_gen[g_r200_enc_variant];
            [g_r200_enc setRenderPipelineState:g_r200_enc_pipe];
            [g_r200_enc setViewport:(MTLViewport){ 0, 0, pkt->rt_width,
                                                  pkt->rt_height, 0, 1 }];
            g_r200_enc_key = rk;
            g_r200_stat_passes++;
            r200_note_written(rlo, rhi, &rk);
        }
        id<MTLRenderCommandEncoder> enc = g_r200_enc;
        [enc setScissorRect:(MTLScissorRect){ sx0, sy0, sx1 - sx0, sy1 - sy0 }];
        if (g_r200_enc_depth_off != ~0u) {
            /* depth-less draws reuse a depth pass with test/write off */
            uint32_t fl = g_r200_enc_depth_z16 ? 8u : 0u;
            if (want_ds) {
                fl |= (pkt->depth_enable ? 1u : 0u) | (pkt->stencil_enable ? 2u : 0u) |
                      (pkt->depth_enable && (pkt->zstencil >> 30) & 1 ? 4u : 0u);
            }
            u.zinfo[0] = fl;
            u.zinfo[1] = pkt->zstencil;
            u.zinfo[2] = pkt->stencil_refmask;
            static int zdebug = -1;
            if (zdebug < 0) {
                zdebug = getenv("PPCGPU_ZDEBUG") != NULL;
            }
            u.zinfo[3] = zdebug ? 0x5a5a : 0;
        }
        {
            static int zlog;
            if (want_ds && zlog++ < 12) {
                qemu_log("r200 zdraw: rt=%x depth=%x/%u bpp=%u enc_depth=%x fl=%x "
                         "z=%08x rm=%08x rb3d=%08x prim=%u nv=%u v0z=%.3f v0w=%.3f\n",
                         pkt->rt_offset, pkt->depth_offset, pkt->depth_pitch,
                         pkt->depth_bpp, g_r200_enc_depth_off, u.zinfo[0],
                         pkt->zstencil, pkt->stencil_refmask, pkt->rb3d_cntl,
                         pkt->prim_class, pkt->num_verts,
                         pkt->verts[pkt->indices[0]].pos[2],
                         pkt->verts[pkt->indices[0]].pos[3]);
            }
        }

        /* Specialize the uber-shader to this draw's actual stage count and
         * bound texture units (the generic pipeline stays the fallback). */
        {
            uint32_t stages = 0, tmask = 0;
            for (int st = 7; st >= 0; st--) {
                if (pkt->pp_cntl & (st == 7 ? 0x800u : 0x1000u << st)) {
                    stages = (uint32_t)st + 1;
                    break;
                }
            }
            for (int t = 0; t < R200_MAX_TEX; t++) {
                tmask |= u.texinfo[t][0] ? 1u << t : 0u;
            }
            id<MTLRenderPipelineState> p =
                r200_spec_pso(dev, g_r200_enc_variant, stages, tmask);
            if (p && p != g_r200_enc_pipe) {
                [enc setRenderPipelineState:p];
                g_r200_enc_pipe = p;
            }
        }

        /*
         * Geometry staging.  When the index list reuses vertices (strips,
         * fans, quads), uploading the vertex list and the indices and
         * drawing indexed moves far fewer bytes than expanding every
         * index into a 144-byte vertex; for a plain list (indices ==
         * vertices in order) the expansion is the same size without the
         * index buffer, so keep it.  The vertex shader addresses v[vid],
         * which an indexed draw feeds from the index buffer.
         */
        size_t vbytes = (size_t)pkt->num_verts * sizeof(R200Vertex);
        size_t ibytes = (size_t)pkt->num_indices * 4;
        size_t fbytes = (size_t)pkt->num_indices * sizeof(R200Vertex);
        id<MTLBuffer> ib = nil;
        size_t ioff = 0;
        bool indexed = false;
        if (pkt->num_verts && vbytes + ibytes < fbytes) {
            id<MTLBuffer> vb = nil;
            size_t voff = 0;
            R200Vertex *vp = r200_arena_alloc(st, vbytes, &vb, &voff);
            uint32_t *ip = vp ? r200_arena_alloc(st, ibytes, &ib, &ioff) : NULL;
            if (ip) {
                memcpy(vp, pkt->verts, vbytes);
                memcpy(ip, pkt->indices, ibytes);
                [enc setVertexBuffer:vb offset:voff atIndex:0];
                [ib retain];      /* released after the draw */
                indexed = true;
            } else {
                /* Larger than the arena: one-off buffers for this draw. */
                id<MTLBuffer> v1 = [dev newBufferWithBytes:pkt->verts length:vbytes
                                                   options:MTLResourceStorageModeShared];
                ib = [dev newBufferWithBytes:pkt->indices length:ibytes
                                     options:MTLResourceStorageModeShared];
                if (v1 && ib) {
                    [enc setVertexBuffer:v1 offset:0 atIndex:0];
                    ioff = 0;
                    indexed = true;
                } else {
                    [ib release];
                    ib = nil;
                }
                [v1 release];
                /* a retained ib is released after the draw */
            }
        }
        if (!indexed) {
            /* Expand the index list straight into the staging buffer. */
            id<MTLBuffer> vb = nil;
            size_t voff = 0;
            R200Vertex *flat = r200_arena_alloc(st, fbytes, &vb, &voff);
            if (flat) {
                for (uint32_t i = 0; i < pkt->num_indices; i++) {
                    flat[i] = pkt->verts[pkt->indices[i]];
                }
                [enc setVertexBuffer:vb offset:voff atIndex:0];
            } else {
                /* Larger than the arena: a one-off buffer for this draw. */
                R200Vertex *tmp = g_malloc(fbytes);
                for (uint32_t i = 0; i < pkt->num_indices; i++) {
                    tmp[i] = pkt->verts[pkt->indices[i]];
                }
                id<MTLBuffer> one = [dev newBufferWithBytes:tmp length:fbytes
                                                    options:MTLResourceStorageModeShared];
                [enc setVertexBuffer:one offset:0 atIndex:0];
                [one release];
                g_free(tmp);
            }
        }
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        for (int t = 0; t < R200_MAX_TEX; t++) {
            [enc setFragmentTexture:tex[t] atIndex:t];
            [enc setFragmentSamplerState:smp[t] atIndex:t];
        }
        if (indexed) {
            [enc drawIndexedPrimitives:pkt->prim_class == 2 ? MTLPrimitiveTypePoint :
                                       pkt->prim_class == 1 ? MTLPrimitiveTypeLine :
                                                              MTLPrimitiveTypeTriangle
                            indexCount:pkt->num_indices
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:ib
                     indexBufferOffset:ioff];
            [ib release];
        } else {
            [enc drawPrimitives:pkt->prim_class == 2 ? MTLPrimitiveTypePoint :
                                pkt->prim_class == 1 ? MTLPrimitiveTypeLine :
                                                       MTLPrimitiveTypeTriangle
                    vertexStart:0 vertexCount:pkt->num_indices];
        }
        g_r200_stat_draws++;
        {   /* debug: read the depth word back after the first depth draws */
            static int zchk;
            if (want_ds && g_r200_enc_depth_off != ~0u && zchk < 12 &&
                r200_zcheck_enabled()) {
                zchk++;
                uint32_t px = (sx0 + sx1) / 2, py = (sy0 + sy1) / 2;
                uint64_t a = (uint64_t)pkt->depth_offset +
                             (uint64_t)py * pkt->depth_pitch * 4 + px * 4;
                const R200Vertex *v0 = &pkt->verts[pkt->indices[0]];
                r200_flush_locked(st);
                qemu_log("r200 zcheck: fl=%x z=%08x at (%u,%u) depthword=%02x%02x%02x%02x "
                         "v0=(%.1f,%.1f,%.3f,%.1f) sc=(%u,%u)-(%u,%u)\n",
                         u.zinfo[0], pkt->zstencil, px, py,
                         vram_ptr[a], vram_ptr[a + 1], vram_ptr[a + 2], vram_ptr[a + 3],
                         v0->pos[0], v0->pos[1], v0->pos[2], v0->pos[3],
                         sx0, sy0, sx1, sy1);
            }
        }
        for (int t = 0; t < R200_MAX_TEX; t++) {
            if (u.texinfo[t][0] && !pkt->tex[t].host_data) {
                r200_note_read(pkt->tex[t].offset, pkt->tex[t].offset +
                               (uint64_t)pkt->tex[t].pitch * pkt->tex[t].height);
            }
        }
    }

    if (g_r200_stat_draws % 2000 == 0) {
        qemu_log("ppc-mac-gpu r200: %llu draws, %llu passes, %llu flushes "
                 "(%llu hazard), avg flush %llu us, views %llu hit/%llu new, "
                 "rq %llu jobs %llu full\n",
                 (unsigned long long)g_r200_stat_draws,
                 (unsigned long long)g_r200_stat_passes,
                 (unsigned long long)g_r200_stat_flushes,
                 (unsigned long long)g_r200_stat_conflicts,
                 (unsigned long long)(g_r200_stat_flushes ?
                     g_r200_stat_flush_us / g_r200_stat_flushes : 0),
                 (unsigned long long)g_r200_stat_view_hit,
                 (unsigned long long)g_r200_stat_view_new,
                 (unsigned long long)g_rq_stat_jobs,
                 (unsigned long long)g_rq_stat_full);
    }
    return 0;
}

/*
 * A draw's deterministic rejections (format, alignment, empty scissor),
 * so the producer still gets -1 synchronously; the encode side re-runs
 * them.  Returns -1 (rejected), 1 (nothing to draw) or 0 (encode it).
 */
static int r200_draw_rejects(PPCMacGPUMetalState *st, const R200DrawPacket *pkt)
{
    if (!st || !st->vramBuffer) {
        return -1;
    }
    id<MTLDevice> dev = st->vramBuffer.device;
    if (!r200_metal_setup(dev)) {
        return -1;
    }
    uint32_t sx0 = pkt->scissor[0], sy0 = pkt->scissor[1];
    uint32_t sx1 = MIN(pkt->scissor[2], pkt->rt_width);
    uint32_t sy1 = MIN(pkt->scissor[3], pkt->rt_height);
    if (sx0 >= sx1 || sy0 >= sy1) {
        return 1;
    }
    uint32_t cfmt = (pkt->rb3d_cntl >> 10) & 0xF;
    bool rt16 = cfmt == 3 || cfmt == 4 || cfmt == 15;
    if (cfmt != 6 && !rt16) {
        r200_metal_warn(0x4000, "colour buffer format not supported", cfmt,
                        pkt->rt_offset);
        return -1;
    }
    if (rt16 && !g_r200_pipeline_c16) {
        return -1;
    }
    MTLPixelFormat rtpf = rt16 ? MTLPixelFormatR16Uint : MTLPixelFormatBGRA8Unorm;
    NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:rtpf];
    uint32_t bpr = pkt->rt_pitch * (rt16 ? 2 : 4);
    if ((pkt->rt_offset % align) || (bpr % align)) {
        r200_metal_warn(2, "render target not aligned for a linear view",
                        pkt->rt_offset, bpr);
        return -1;
    }
    return 0;
}

static int metal_draw_r200(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                           const R200DrawPacket *pkt)
{
    PPCMacGPUMetalState *st = opaque;
    int r;

    static int drawlog = -1;
    if (drawlog < 0) {
        drawlog = getenv("R200_DRAWLOG") != NULL;
    }
    if (drawlog) {
        /* One line per draw: where it goes, how it blends, what it samples. */
        qemu_log("R200DRAW t=%lld rt=%x/%u %ux%u sc=%u,%u-%u,%u rb3d=%x pp=%x "
                 "cb=%x ab=%x mask=%x verts=%u idx=%u tex0=%u:%x/%u %ux%u f=%x "
                 "alphain=%u swap=%u v0=(%.0f,%.0f) c0=(%.3f,%.3f,%.3f,%.3f) v2=(%.0f,%.0f)\n",
                 (long long)(g_get_monotonic_time() / 1000), pkt->rt_offset,
                 pkt->rt_pitch, pkt->rt_width, pkt->rt_height, pkt->scissor[0],
                 pkt->scissor[1], pkt->scissor[2], pkt->scissor[3],
                 pkt->rb3d_cntl, pkt->pp_cntl, pkt->cblend, pkt->ablend,
                 pkt->plane_mask, pkt->num_verts, pkt->num_indices,
                 pkt->tex[0].enabled, pkt->tex[0].offset, pkt->tex[0].pitch,
                 pkt->tex[0].width, pkt->tex[0].height, pkt->tex[0].format,
                 pkt->tex[0].alpha_in_map, pkt->tex[0].swap,
                 pkt->verts ? pkt->verts[0].pos[0] : 0.f,
                 pkt->verts ? pkt->verts[0].pos[1] : 0.f,
                 pkt->verts ? pkt->verts[0].color[0] : 0.f,
                 pkt->verts ? pkt->verts[0].color[1] : 0.f,
                 pkt->verts ? pkt->verts[0].color[2] : 0.f,
                 pkt->verts ? pkt->verts[0].color[3] : 0.f,
                 pkt->num_verts > 2 ? pkt->verts[2].pos[0] : 0.f,
                 pkt->num_verts > 2 ? pkt->verts[2].pos[1] : 0.f);
    }
    if (!rqueue_enabled()) {
        pthread_mutex_lock(&g_render_lock);
        r = r200_draw_encode(st, vram_ptr, vram_size, pkt);
        pthread_mutex_unlock(&g_render_lock);
        return r;
    }
    r = r200_draw_rejects(st, pkt);
    if (r) {
        return r < 0 ? -1 : 0;      /* rejected, or nothing to draw */
    }
    /* The packet the device passes is its own; the job takes its heap
     * buffers over (see r200_job_new). */
    rq_push(r200_job_new((R200DrawPacket *)(uintptr_t)pkt));
    return 0;
}


/* ========================================================================
 * R300 (ati-radeon-9700) draws
 *
 * The device hands over post-transform vertices and the GLSL generated
 * from the fragment program (r300/r300_us.c, compiled on to MSL by
 * r300/r300_spirv.c), which also does
 * alpha test, blending and packing into the colour buffer's byte order.
 * Draws share the R200 path's batch, views and hazard tracking, so the
 * device's flush_r200() covers them too.
 * ======================================================================== */

static NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *g_r300_pipes;

/*
 * Samplers in a direct-mapped table: an NSNumber box and a dictionary
 * hash for each of 16 units each draw added up.  A collision evicts;
 * distinct sampler states in real workloads number in the tens.
 */
#define R300_SMP_SLOTS 512
static struct { uint32_t key; id<MTLSamplerState> smp; } g_r300_smp[R300_SMP_SLOTS];

static void r300_metal_warn(uint32_t bit, const char *msg)
{
    static uint32_t warned;
    if (!(warned & bit)) {
        warned |= bit;
        qemu_log("ppc-mac-gpu r300: %s\n", msg);
    }
}

/*
 * A stage of a r300_us_to_glsl() source as a Metal function: GLSL ->
 * SPIR-V -> MSL (r300_spirv.c), compiled once per source and stage.
 */
static NSMutableDictionary<NSString *, id<MTLFunction>> *g_r300_fns;

/*
 * On-disk cache of the GLSL -> SPIR-V -> MSL step (shaderc and SPIRV-Cross
 * are the slow half of a first-time shader; a game uses hundreds of them and
 * every launch used to redo all of it).  One file per (toolchain, stage,
 * source), named by the SHA-256 of those, holding the MSL and a trailing
 * marker so a file cut short is a miss.  Files are written under a private
 * name and renamed into place, so concurrent launches never see half of one.
 * PPCGPU_MSL_CACHE=0 turns it off.  Stale or damaged entries are harmless:
 * MSL that fails to compile is deleted and regenerated.
 */
#define MSL_CACHE_END "\n// ppcosxkvm-msl-end\n"
#define MSL_CACHE_MAX_FILES 8192

static NSString *g_msl_cache_dir;
static uint64_t g_msl_hits, g_msl_misses, g_msl_load_us, g_msl_conv_us;

static void msl_cache_prune(NSFileManager *fm, NSString *dir)
{
    NSArray<NSString *> *names = [fm contentsOfDirectoryAtPath:dir error:NULL];

    if (names.count <= MSL_CACHE_MAX_FILES) {
        return;
    }
    NSMutableArray<NSArray *> *aged = [NSMutableArray array];
    for (NSString *n in names) {
        NSString *path = [dir stringByAppendingPathComponent:n];
        NSDate *d = [fm attributesOfItemAtPath:path error:NULL].fileModificationDate;
        [aged addObject:@[ path, d ?: [NSDate distantPast] ]];
    }
    [aged sortUsingComparator:^NSComparisonResult(NSArray *a, NSArray *b) {
        return [a[1] compare:b[1]];
    }];
    /* drop the oldest quarter */
    for (NSUInteger i = 0; i < aged.count / 4; i++) {
        [fm removeItemAtPath:aged[i][0] error:NULL];
    }
}

static NSString *msl_cache_dir(void)
{
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        const char *off = getenv("PPCGPU_MSL_CACHE");
        if (off && !strcmp(off, "0")) {
            return;
        }
        @autoreleasepool {
            NSFileManager *fm = [NSFileManager defaultManager];
            NSString *dir = [[NSSearchPathForDirectoriesInDomains(
                                  NSCachesDirectory, NSUserDomainMask, YES).firstObject
                              stringByAppendingPathComponent:@"ppcosxkvm"]
                             stringByAppendingPathComponent:@"msl"];
            if ([fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                               attributes:nil error:NULL]) {
                msl_cache_prune(fm, dir);
                g_msl_cache_dir = [dir copy];
            }
        }
    });
    return g_msl_cache_dir;
}

/* The cache file for this source and stage, or nil when the cache is off. */
static NSString *msl_cache_path(const char *glsl, R300Stage stage)
{
    NSString *dir = msl_cache_dir();
    CC_SHA256_CTX c;
    unsigned char md[CC_SHA256_DIGEST_LENGTH];
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    const char *tc = r300_msl_toolchain_id();
    unsigned char st = (unsigned char)stage;

    if (!dir) {
        return nil;
    }
    CC_SHA256_Init(&c);
    CC_SHA256_Update(&c, tc, (CC_LONG)strlen(tc) + 1);
    CC_SHA256_Update(&c, &st, 1);
    CC_SHA256_Update(&c, glsl, (CC_LONG)strlen(glsl));
    CC_SHA256_Final(md, &c);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        snprintf(hex + 2 * i, 3, "%02x", md[i]);
    }
    return [dir stringByAppendingPathComponent:
            [NSString stringWithFormat:@"%s.%s.metal", hex, r300_stage_entry(stage)]];
}

/* The cached MSL (malloc'd), or NULL. */
static char *msl_cache_load(NSString *path)
{
    static const char end[] = MSL_CACHE_END;
    const size_t en = sizeof(end) - 1;
    NSData *d = [NSData dataWithContentsOfFile:path];
    char *msl;

    if (d.length <= en || memcmp((const char *)d.bytes + d.length - en, end, en)) {
        return NULL;
    }
    msl = malloc(d.length + 1);
    memcpy(msl, d.bytes, d.length);
    msl[d.length] = 0;
    return msl;
}

static void msl_cache_store(NSString *path, const char *msl)
{
    static const char end[] = MSL_CACHE_END;
    NSString *tmp = [path stringByAppendingFormat:@".%d.%x.tmp", getpid(),
                     (unsigned)pthread_mach_thread_np(pthread_self())];
    FILE *f = fopen(tmp.fileSystemRepresentation, "wb");
    bool ok;

    if (!f) {
        return;
    }
    ok = fwrite(msl, 1, strlen(msl), f) == strlen(msl) &&
         fwrite(end, 1, sizeof(end) - 1, f) == sizeof(end) - 1;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.fileSystemRepresentation, path.fileSystemRepresentation)) {
        unlink(tmp.fileSystemRepresentation);
    }
}

static void msl_cache_note(bool hit, int64_t us)
{
    uint64_t n;

    if (hit) {
        g_msl_hits++;
        g_msl_load_us += us;
    } else {
        g_msl_misses++;
        g_msl_conv_us += us;
    }
    n = g_msl_hits + g_msl_misses;
    if (n == 1 || n % 100 == 0) {
        qemu_log("ppc-mac-gpu r300: MSL cache %llu hits (%llu us each) / %llu "
                 "converted (%llu us each)\n",
                 (unsigned long long)g_msl_hits,
                 (unsigned long long)(g_msl_hits ? g_msl_load_us / g_msl_hits : 0),
                 (unsigned long long)g_msl_misses,
                 (unsigned long long)(g_msl_misses ? g_msl_conv_us / g_msl_misses : 0));
    }
}

static id<MTLFunction> r300_function(id<MTLDevice> dev, NSString *glsl,
                                     R300Stage stage)
{
    NSString *key = [glsl stringByAppendingFormat:@"\n// stage %d\n", stage];
    id<MTLFunction> fn;
    NSError *err = nil;
    char *msg = NULL;

    if (!g_r300_fns) {
        g_r300_fns = [[NSMutableDictionary alloc] init];
    }
    fn = g_r300_fns[key];
    if (fn) {
        return fn;
    }
    NSString *cpath = msl_cache_path(glsl.UTF8String, stage);
    char *msl = NULL;
    bool cached = false;
    id<MTLLibrary> lib = nil;

    for (int attempt = 0; attempt < 2 && !lib; attempt++) {
        int64_t t0 = g_get_monotonic_time();

        if (attempt == 0 && cpath) {
            msl = msl_cache_load(cpath);
            cached = msl != NULL;
        }
        if (!msl) {
            msl = r300_glsl_to_msl(glsl.UTF8String, stage, &msg);
            if (!msl) {
                qemu_log("ppc-mac-gpu r300: %s: %s\n%s\n", r300_stage_entry(stage), msg,
                         glsl.UTF8String);
                free(msg);
                return nil;
            }
        }
        msl_cache_note(cached, g_get_monotonic_time() - t0);
        /* Vertex programs keep IEEE infinities and NaNs, as the interpreter
         * does (fast math broke the specials the _DX/_FF opcodes produce). */
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        if (stage == R300_STAGE_VS) {
            opt.mathMode = MTLMathModeSafe;
            /* gl_Position is invariant: a depth pre-pass and the passes
             * drawn over it with an EQUAL depth test (Doom 3's lights)
             * must land on identical depths, which separately optimised
             * shaders otherwise miss here and there (dotted surfaces). */
            opt.preserveInvariance = YES;
        }
        lib = [dev newLibraryWithSource:@(msl) options:opt error:&err];
        [opt release];
        if (lib) {
            if (cpath && !cached) {
                msl_cache_store(cpath, msl);
            }
        } else if (cached) {
            /* a stale or damaged entry: drop it and convert afresh */
            unlink(cpath.fileSystemRepresentation);
            free(msl);
            msl = NULL;
            cached = false;
            continue;
        } else {
            qemu_log("ppc-mac-gpu r300: shader compile failed: %s\n%s\n",
                     err.localizedDescription.UTF8String, msl);
            free(msl);
            return nil;
        }
        free(msl);
        msl = NULL;
    }
    if (!lib) {
        return nil;
    }
    fn = [lib newFunctionWithName:@(r300_stage_entry(stage))];
    [lib release];
    if (fn) {
        g_r300_fns[key] = fn;
        [fn release];
    }
    return fn;
}

/*
 * cfmt[0..ncb-1]: the colour buffers' formats; zfmt MTLPixelFormatInvalid
 * for a colour-only pass (r300_fs), else the format of the depth/stencil
 * buffer bound as colour attachment ncb (r300_fs_z).
 */
static id<MTLRenderPipelineState> r300_pipeline_by_text(id<MTLDevice> dev,
                                                        const char *glsl,
                                                        const char *vs_glsl,
                                                        const MTLPixelFormat *cfmt,
                                                        uint32_t ncb,
                                                        MTLPixelFormat zfmt);

/*
 * Pipelines by (r300_us_glsl_cached id, formats), in front of the lookup
 * by text: building a key from the whole source and hashing it cost more
 * per draw than the draw.  Direct-mapped; a miss falls back to the text.
 */
typedef struct R300PipeSlot {
    uint32_t id, vs_id, ncb;
    MTLPixelFormat cfmt[4], zfmt;
    id<MTLRenderPipelineState> pipe;    /* owned by g_r300_pipes */
} R300PipeSlot;

static R300PipeSlot g_r300_pipe_slots[1024];

static id<MTLRenderPipelineState> r300_pipeline(id<MTLDevice> dev,
                                                const char *glsl, uint32_t prog_id,
                                                const char *vs_glsl, uint32_t vs_id,
                                                const MTLPixelFormat *cfmt,
                                                uint32_t ncb,
                                                MTLPixelFormat zfmt)
{
    if (!prog_id || (vs_glsl && !vs_id) || ncb > 4) {
        return r300_pipeline_by_text(dev, glsl, vs_glsl, cfmt, ncb, zfmt);
    }
    uint32_t h = prog_id * 2654435761u ^ (uint32_t)zfmt * 40503u ^ vs_id * 97u;
    for (uint32_t k = 0; k < ncb; k++) {
        h = h * 31u + (uint32_t)cfmt[k];
    }
    R300PipeSlot *sl = &g_r300_pipe_slots[(h ^ (h >> 16)) & 1023];
    if (sl->pipe && sl->id == prog_id && sl->vs_id == vs_id && sl->ncb == ncb &&
        sl->zfmt == zfmt &&
        memcmp(sl->cfmt, cfmt, ncb * sizeof(*cfmt)) == 0) {
        return sl->pipe;
    }
    id<MTLRenderPipelineState> p = r300_pipeline_by_text(dev, glsl, vs_glsl, cfmt, ncb, zfmt);
    if (p) {
        sl->id = prog_id;
        sl->vs_id = vs_id;
        sl->ncb = ncb;
        sl->zfmt = zfmt;
        memcpy(sl->cfmt, cfmt, ncb * sizeof(*cfmt));
        sl->pipe = p;
    }
    return p;
}

static id<MTLRenderPipelineState> r300_pipeline_by_text(id<MTLDevice> dev,
                                                        const char *glsl,
                                                        const char *vs_glsl,
                                                        const MTLPixelFormat *cfmt,
                                                        uint32_t ncb,
                                                        MTLPixelFormat zfmt)
{
    NSString *src = [NSString stringWithUTF8String:glsl];
    NSString *vsrc = vs_glsl ? [NSString stringWithUTF8String:vs_glsl] : src;
    NSMutableString *key = [NSMutableString stringWithString:src];
    id<MTLRenderPipelineState> p;
    NSError *err = nil;

    for (uint32_t k = 0; k < ncb; k++) {
        [key appendFormat:@"\n// c%u %lu", k, (unsigned long)cfmt[k]];
    }
    [key appendFormat:@"\n// z %lu\n", (unsigned long)zfmt];
    if (vs_glsl) {
        [key appendString:vsrc];
    }
    if (!g_r300_pipes) {
        g_r300_pipes = [[NSMutableDictionary alloc] init];
    }
    p = g_r300_pipes[key];
    if (p) {
        return p;
    }
    id<MTLFunction> vs = r300_function(dev, vsrc, R300_STAGE_VS);
    id<MTLFunction> fs = r300_function(dev, src, zfmt == MTLPixelFormatInvalid ?
                                       R300_STAGE_FS : R300_STAGE_FS_Z);
    if (!vs || !fs) {
        return nil;
    }
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    for (uint32_t k = 0; k < ncb; k++) {
        pd.colorAttachments[k].pixelFormat = cfmt[k];
    }
    pd.colorAttachments[ncb].pixelFormat = zfmt;
    p = bin_pso(dev, pd, &err);
    [pd release];
    if (!p) {
        qemu_log("ppc-mac-gpu r300: pipeline failed: %s\n",
                 err.localizedDescription.UTF8String);
        return nil;
    }
    g_r300_pipes[key] = p;
    [p release];
    if (g_r300_pipes.count % 16 == 1) {
        qemu_log("ppc-mac-gpu r300: %lu fragment pipelines\n",
                 (unsigned long)g_r300_pipes.count);
    }
    return p;
}

/* ZB_ZPASS counter, bound at fragment buffer 1 of every R300 draw. */
static id<MTLBuffer> g_r300_zpass;

static id<MTLBuffer> r300_zpass_buf(id<MTLDevice> dev)
{
    if (!g_r300_zpass) {
        g_r300_zpass = [dev newBufferWithLength:16
                                        options:MTLResourceStorageModeShared];
        memset(g_r300_zpass.contents, 0, 16);
    }
    return g_r300_zpass;
}

static uint32_t metal_zpass_r300(void *opaque, bool reset, uint32_t value)
{
    PPCMacGPUMetalState *st = opaque;
    uint32_t v;

    if (!st || !st->vramBuffer) {
        return 0;
    }
    metal_flush_r200(st);
    uint32_t *c = r300_zpass_buf(st->vramBuffer.device).contents;
    v = c[0];
    if (reset) {
        c[0] = value;
    }
    return v;
}

static MTLSamplerAddressMode r300_wrap(uint32_t w)
{
    switch (w & 7) {
    case 0:  return MTLSamplerAddressModeRepeat;
    case 1:  return MTLSamplerAddressModeMirrorRepeat;
    case 3:
    case 5:
    case 7:  return MTLSamplerAddressModeMirrorClampToEdge;
    case 6:  return MTLSamplerAddressModeClampToBorderColor;  /* r300_tex adds the colour */
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

/*
 * TX_FILTER0: wrap S/T/R, MAG/MIN_FILTER (1 nearest, else linear; 3
 * anisotropic), MIP_FILTER (0 none, 1 nearest, 2 linear), MAX_MIP_LEVEL
 * (the finest level used), MAX_ANISO (1:1 .. 16:1); levels the texture's
 * mip count.  The LOD bias is applied at the sample.
 */
static id<MTLSamplerState> r300_sampler(id<MTLDevice> dev, uint32_t f0,
                                        uint32_t levels)
{
    uint32_t key = (f0 & 0xFFFFFF) | (MIN(levels, 15u) << 24);
    uint32_t slot = (key * 2654435761u) >> 20;      /* spread over the table */
    id<MTLSamplerState> s;

    for (uint32_t probe = 0; probe < 4; probe++) {
        uint32_t i = (slot + probe) & (R300_SMP_SLOTS - 1);
        if (!g_r300_smp[i].smp) {
            break;
        }
        if (g_r300_smp[i].key == key) {
            return g_r300_smp[i].smp;
        }
    }
    uint32_t mag = (f0 >> 9) & 3, min = (f0 >> 11) & 3, mip = (f0 >> 13) & 3;
    MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
    d.sAddressMode = r300_wrap(f0);
    d.tAddressMode = r300_wrap(f0 >> 3);
    d.rAddressMode = r300_wrap(f0 >> 6);
    d.borderColor = MTLSamplerBorderColorTransparentBlack;
    d.magFilter = mag == 1 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    d.minFilter = min == 1 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    d.mipFilter = levels <= 1 || mip == 0 ? MTLSamplerMipFilterNotMipmapped :
                  mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    if ((mag == 3 || min == 3) && ((f0 >> 21) & 7)) {
        d.maxAnisotropy = 1u << MIN((f0 >> 21) & 7, 4u);
    }
    d.lodMinClamp = MIN((f0 >> 17) & 0xF, levels ? levels - 1 : 0);
    d.lodMaxClamp = levels ? levels - 1 : 0;
    s = [dev newSamplerStateWithDescriptor:d];
    [d release];
    /* First free probe slot, else the base slot (evicting its sampler). */
    uint32_t store = slot & (R300_SMP_SLOTS - 1);
    for (uint32_t probe = 0; probe < 4; probe++) {
        uint32_t i = (slot + probe) & (R300_SMP_SLOTS - 1);
        if (!g_r300_smp[i].smp) {
            store = i;
            break;
        }
    }
    [g_r300_smp[store].smp release];
    g_r300_smp[store].key = key;
    g_r300_smp[store].smp = s;      /* table keeps the reference */
    return s;
}

/*
 * Write-after-read: a render target over memory the open batch samples
 * through a different view.  Views aliasing one MTLBuffer are separate
 * Metal resources, so nothing orders the write after those reads.  The
 * compositor hits this while dragging a window: it reuses one scratch
 * buffer at a new pitch every frame, draws into it and samples it back,
 * and without the flush a frame's writes land under the previous frame's
 * reads.  Rendering to the view already bound as this batch's target is
 * ordered by the encoder and needs nothing.
 */
static bool r300_read_conflict(uint64_t lo, uint64_t hi, const R200TexKey *rt)
{
    if (g_r200_enc && !memcmp(&g_r200_enc_key, rt, sizeof(*rt)) &&
        !g_r200_read_overflow) {
        bool any = false;
        for (int i = 0; i < g_r200_nread; i++) {
            if (lo < g_r200_read[i].hi && g_r200_read[i].lo < hi) {
                any = true;
                break;
            }
        }
        if (!any) {
            return false;
        }
    }
    if (g_r200_read_overflow) {
        return g_r200_cb != nil;
    }
    for (int i = 0; i < g_r200_nread; i++) {
        if (lo < g_r200_read[i].hi && g_r200_read[i].lo < hi) {
            return true;
        }
    }
    return false;
}

/* Metal format of a colour-buffer view (R300_RTV_*). */
static MTLPixelFormat r300_rt_pf(uint32_t view)
{
    switch (view) {
    case R300_RTV_R8U:     return MTLPixelFormatR8Uint;
    case R300_RTV_R16U:    return MTLPixelFormatR16Uint;
    case R300_RTV_R32U:    return MTLPixelFormatR32Uint;
    case R300_RTV_RG32U:   return MTLPixelFormatRG32Uint;
    case R300_RTV_RGBA32U: return MTLPixelFormatRGBA32Uint;
    default:               return MTLPixelFormatRGBA8Unorm;
    }
}

static MTLPixelFormat r300_raw_pf(uint32_t view_bpp)
{
    return view_bpp == 16 ? MTLPixelFormatRGBA32Uint :
           view_bpp == 8 ? MTLPixelFormatRG32Uint : MTLPixelFormatR32Uint;
}

/*
 * Textures the device rebuilds as real Metal textures (mip chains, 3D,
 * cube maps: a linear view cannot have levels or faces).  Cached by the
 * unit's layout and a hash of its bytes, so unchanged textures cost a
 * hash, not an upload.
 */
#define R300_TCACHE 48
typedef struct R300TexCacheKey {
    uint32_t addr, format, kind, width, height, depth, dim, levels, pitch;
    uint32_t host;
    uint64_t hash;
} R300TexCacheKey;
static struct {
    R300TexCacheKey key;
    id<MTLTexture> tex;
    uint64_t used;
} g_r300_tcache[R300_TCACHE];
static uint64_t g_r300_tcache_clock;

static id<MTLTexture> r300_texture_full(PPCMacGPUMetalState *st, id<MTLDevice> dev,
                                        uint8_t *vram_ptr, const R300TexDesc *td)
{
    const uint8_t *src = td->host_data ? td->host_data : vram_ptr + td->gpu_addr;
    uint64_t lo = td->gpu_addr, hi = lo + td->size_bytes;
    MTLPixelFormat pf;
    bool dxt = td->kind >= R300_TEXK_DXT1;

    switch (td->kind) {
    case R300_TEXK_R8:    pf = MTLPixelFormatR8Unorm; break;
    case R300_TEXK_RG8:   pf = MTLPixelFormatRG8Unorm; break;
    case R300_TEXK_DXT1:  pf = MTLPixelFormatBC1_RGBA; break;
    case R300_TEXK_DXT3:  pf = MTLPixelFormatBC2_RGBA; break;
    case R300_TEXK_DXT5:  pf = MTLPixelFormatBC3_RGBA; break;
    default:              pf = MTLPixelFormatRGBA8Unorm; break;
    }
    if (td->dim == R300_TEXDIM_CUBE && td->width != td->height) {
        r300_metal_warn(64, "cube map with non-square faces");
        return nil;
    }
    /* The CPU reads the texels: finish pending draws that write them. */
    if (!td->host_data && r200_batch_conflict(lo, hi, NULL)) {
        r200_flush_locked(st);
    }

    R300TexCacheKey key = { td->gpu_addr, td->format, td->kind, td->width, td->height,
                            td->depth, td->dim, td->levels, td->pitch_bytes,
                            td->host_data != NULL ? 1u + (td->endian << 1) : td->endian << 1,
                            r300_tex_identity(td, src) };
    int lru = 0;
    for (int i = 0; i < R300_TCACHE; i++) {
        if (g_r300_tcache[i].tex && !memcmp(&g_r300_tcache[i].key, &key, sizeof(key))) {
            g_r300_tcache[i].used = ++g_r300_tcache_clock;
            return g_r300_tcache[i].tex;
        }
        if (g_r300_tcache[i].used < g_r300_tcache[lru].used) {
            lru = i;
        }
    }

    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = td->dim == R300_TEXDIM_3D ? MTLTextureType3D :
                    td->dim == R300_TEXDIM_CUBE ? MTLTextureTypeCube : MTLTextureType2D;
    d.pixelFormat = pf;
    /* BC textures: the base level a whole number of blocks */
    d.width = dxt ? (td->width + 3) & ~3u : td->width;
    d.height = dxt ? (td->height + 3) & ~3u : td->height;
    d.depth = td->dim == R300_TEXDIM_3D ? td->depth : 1;
    d.mipmapLevelCount = td->levels;
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    [d release];
    if (!t) {
        return nil;
    }
    for (uint32_t l = 0; l < td->levels; l++) {
        uint32_t w, h, n, bpr;
        r300_tex_level_dims(td, l, &w, &h, &n);
        /* Metal's level size (BC: from the rounded-up base) */
        uint32_t mw = MAX((uint32_t)t.width >> l, 1u), mh = MAX((uint32_t)t.height >> l, 1u);
        size_t face = (size_t)td->lvl_pitch[l] * td->lvl_rows[l];
        for (uint32_t f = 0; f < n; f++) {
            const uint8_t *fs = src + td->lvl_off[l] + f * face;
            uint8_t *bytes = r300_tex_level_bytes(td, fs, l, dxt ? mw : w, dxt ? mh : h, &bpr);
            if (td->dim == R300_TEXDIM_3D) {
                [t replaceRegion:MTLRegionMake3D(0, 0, f, dxt ? mw : w, dxt ? mh : h, 1)
                     mipmapLevel:l slice:0 withBytes:bytes bytesPerRow:bpr bytesPerImage:0];
            } else {
                [t replaceRegion:MTLRegionMake2D(0, 0, dxt ? mw : w, dxt ? mh : h)
                     mipmapLevel:l slice:f withBytes:bytes bytesPerRow:bpr bytesPerImage:0];
            }
            free(bytes);
        }
    }
    [g_r300_tcache[lru].tex release];
    g_r300_tcache[lru].key = key;
    g_r300_tcache[lru].tex = t;
    g_r300_tcache[lru].used = ++g_r300_tcache_clock;
    return t;
}

/*
 * Units the shader decodes (R300_TEXK_RAW): the whole chain as one uint
 * view, *rowel elements a row, addressed by byte offset in the shader
 * (r300_lvl, r300_texel).  Zero-copy over VRAM when the level-0 pitch
 * suits a linear view; otherwise (GART texels, odd pitches) a copy with
 * GART dwords byte-reversed to VRAM's side of the aperture (see
 * r300_tex_raw_bpp).
 */
static id<MTLTexture> r300_texture_raw(PPCMacGPUMetalState *st, id<MTLDevice> dev,
                                       uint8_t *vram_ptr, uint64_t vram_size,
                                       const R300TexDesc *td, uint32_t *rowel)
{
    uint32_t eb = td->view_bpp;
    MTLPixelFormat rpf = r300_raw_pf(eb);
    NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:rpf];
    uint32_t p0 = td->pitch_bytes;

    if (!td->host_data && p0 && !(p0 % eb) && !(p0 % align) && !(td->gpu_addr % align)) {
        uint32_t rows = (td->size_bytes + p0 - 1) / p0;
        if (rows <= 16384 && (uint64_t)td->gpu_addr + (uint64_t)rows * p0 <= vram_size) {
            R200TexKey k = { td->gpu_addr, p0 / eb, rows, p0, (uint32_t)rpf };
            *rowel = p0 / eb;
            return r200_view(st, k, rpf, false);
        }
    }
    if (!td->host_data && r200_batch_conflict(td->gpu_addr, (uint64_t)td->gpu_addr +
                                              td->size_bytes, NULL)) {
        r200_flush_locked(st);
    }
    const uint8_t *src = td->host_data ? td->host_data : vram_ptr + td->gpu_addr;
    uint32_t n = (td->size_bytes + eb - 1) / eb;        /* elements */
    uint32_t w = 64;
    while (w < 16384 && (uint64_t)w * w < n) {
        w <<= 1;
    }
    uint32_t h = (n + w - 1) / w;
    if (!h || h > 16384) {
        r300_metal_warn(128, "shader-decoded texture too large");
        return nil;
    }
    size_t bytes = (size_t)w * h * eb;
    /* GART texels: the swapped copy is cached by content, since rebuilding
     * it every draw costs a full copy, bswap and upload. */
    uint64_t rhash = td->host_data ? r300_hash_bytes(td->host_data, td->size_bytes) : 0;
    uint32_t rkey[6] = { 3, eb, w, h, td->gpu_addr, td->size_bytes };
    if (td->host_data) {
        id<MTLTexture> c = aux_tex_find(rhash, rkey);
        if (c) {
            *rowel = w;
            return c;
        }
    }
    uint8_t *buf = g_malloc0(bytes);
    memcpy(buf, src, td->size_bytes);
    if (td->host_data) {
        uint32_t *dw = (uint32_t *)buf;
        for (size_t i = 0; i < bytes / 4; i++) {
            dw[i] = __builtin_bswap32(dw[i]);
        }
    }
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:rpf width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [[dev newTextureWithDescriptor:d] autorelease];
    [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
           withBytes:buf bytesPerRow:(size_t)w * eb];
    g_free(buf);
    if (td->host_data && t) {
        aux_tex_store(rhash, rkey, t);
    }
    *rowel = w;
    return t;
}

/* The Metal texture for one R300 texture unit (see R300_TEXK_*). */
static id<MTLTexture> r300_texture(PPCMacGPUMetalState *st, id<MTLDevice> dev,
                                   uint8_t *vram_ptr, uint64_t vram_size,
                                   const R300TexDesc *td, uint32_t *rowel)
{
    uint64_t lo = td->gpu_addr;
    uint64_t hi = lo + td->size_bytes;
    MTLPixelFormat pf;

    if (!td->host_data && hi > vram_size) {
        r300_metal_warn(2, "texture outside VRAM");
        return nil;
    }
    if (td->kind == R300_TEXK_RAW) {
        return r300_texture_raw(st, dev, vram_ptr, vram_size, td, rowel);
    }
    if (td->levels > 1 || td->dim != R300_TEXDIM_2D) {
        return r300_texture_full(st, dev, vram_ptr, td);
    }
    switch (td->kind) {
    case R300_TEXK_RGBA8: pf = MTLPixelFormatRGBA8Unorm; break;
    case R300_TEXK_R8:    pf = MTLPixelFormatR8Unorm; break;
    case R300_TEXK_RG8:   pf = MTLPixelFormatRG8Unorm; break;
    default:              pf = MTLPixelFormatInvalid; break;
    }
    if (pf != MTLPixelFormatInvalid &&
        (td->host_data || (td->kind == R300_TEXK_R8 && (td->endian & 3) != 2))) {
        /* Copied out of the GART by the device, or 8-bit texels whose
         * words the card reads in another byte order than VRAM holds
         * them (r300_tex_level_bytes reorders them): uploaded through
         * the content-hash cache, so an unchanged texture costs a hash. */
        return r300_texture_full(st, dev, vram_ptr, td);
    }
    if (pf != MTLPixelFormatInvalid) {
        /* Zero-copy view over VRAM. */
        NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:pf];
        if ((td->gpu_addr % align) || (td->pitch_bytes % align)) {
            r300_metal_warn(8, "texture not aligned for a linear view");
            return nil;
        }
        R200TexKey k = { td->gpu_addr, td->width, td->height, td->pitch_bytes,
                         (uint32_t)pf };
        return r200_view(st, k, pf, false);
    }
    /* 16bpp formats and DXT: converted or copied by the CPU (cached). */
    return r300_texture_full(st, dev, vram_ptr, td);
}

/* Render targets B-D of the open R300 pass (target A is g_r200_enc_key). */
static uint32_t g_r300_enc_ncb = 1;
static R200TexKey g_r300_enc_cb[R300_US_MAX_TARGETS];

/*
 * What the open R300 pass already has bound.  A frame is hundreds of draws
 * in one pass that mostly differ in an index list; re-setting the pipeline,
 * sixteen textures and samplers, and a few KB of constants for each of them
 * is encoder work that does nothing.  The cache lives exactly as long as
 * the encoder it describes (r300_draw_encode clears it whenever it opens
 * one, and no other path draws into an R300 pass).  The ids are not
 * retained: the encoder keeps everything it was given alive until its
 * command buffer completes, so a pointer here cannot be recycled meanwhile.
 */
static struct {
    id<MTLRenderPipelineState> pipe;
    int winding, cull;                  /* -1: not set yet */
    bool zpass;                         /* fragment buffer 1 */
    id<MTLTexture> tex[R300_NUM_TEX_UNITS];
    id<MTLSamplerState> smp[R300_NUM_TEX_UNITS];
    bool fsu_valid, ms_valid, vsu_valid, rect_valid;
    R300FSUniforms fsu;
    float ms[4];
    R300VSUniforms *vsu;                /* last bytes bound as vertex buffer 2 */
    MTLScissorRect rect;
} g_r300_ec;
static uint64_t g_r300_ec_set, g_r300_ec_skipped;   /* encoder calls made / saved */

static void r300_ec_reset(void)
{
    R300VSUniforms *keep = g_r300_ec.vsu;

    memset(&g_r300_ec, 0, sizeof(g_r300_ec));
    g_r300_ec.vsu = keep;
    g_r300_ec.winding = g_r300_ec.cull = -1;
}

/* A 1x1 stand-in for an unbound unit, of the type the shader declares. */
static id<MTLTexture> r300_dummy(id<MTLDevice> dev, bool raw, uint32_t dim)
{
    static id<MTLTexture> d[4];
    int i = raw ? 3 : dim;

    if (!d[i]) {
        MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
        td.textureType = i == 1 ? MTLTextureType3D : i == 2 ? MTLTextureTypeCube
                                                              : MTLTextureType2D;
        td.pixelFormat = raw ? MTLPixelFormatR32Uint : MTLPixelFormatRGBA8Unorm;
        td.width = td.height = td.depth = 1;
        td.usage = MTLTextureUsageShaderRead;
        d[i] = [dev newTextureWithDescriptor:td];
        [td release];
    }
    return d[i];
}

/* Copy len bytes into this frame's arena and bind them as vertex buffer
 * index (a one-off buffer when the arena can't take them). */
static bool r300_bind_bytes(PPCMacGPUMetalState *st, id<MTLDevice> dev,
                            id<MTLRenderCommandEncoder> enc, const void *data,
                            size_t len, NSUInteger index)
{
    id<MTLBuffer> b = nil;
    size_t off = 0;
    void *dst = r200_arena_alloc(st, len ? len : 16, &b, &off);

    if (dst) {
        memcpy(dst, data, len);
        [enc setVertexBuffer:b offset:off atIndex:index];
        return true;
    }
    b = [dev newBufferWithBytes:data length:len options:MTLResourceStorageModeShared];
    if (!b) {
        return false;
    }
    [enc setVertexBuffer:b offset:0 atIndex:index];
    [b release];
    return true;
}

static int r300_draw_encode(PPCMacGPUMetalState *st, uint8_t *vram_ptr,
                            uint64_t vram_size, const R300DrawPacket *pkt)
{
    if (!st || !st->vramBuffer) {
        return -1;
    }
    id<MTLDevice> dev = st->vramBuffer.device;
    uint32_t ncb = MAX(pkt->num_cb, 1u);
    MTLPixelFormat cpf[R300_US_MAX_TARGETS];
    R200TexKey ck[R300_US_MAX_TARGETS];
    uint32_t sx0 = pkt->scissor[0], sy0 = pkt->scissor[1];
    uint32_t sx1 = MIN(pkt->scissor[2], pkt->rt_width);
    uint32_t sy1 = MIN(pkt->scissor[3], pkt->rt_height);
    /*
     * Multisampling: sample k of row y is row y * ns + k of every buffer
     * (R300DrawPacket.aa_samples), so one view ns rows' worth of bytes
     * wide holds sample k in columns k * pitch onwards, and each sample
     * is drawn into its columns in turn, with the geometry shifted by the
     * sample's offset from the pixel centre.  The column offset is shared
     * by all attachments, so they need the same pitch in pixels.
     */
    uint32_t ns = MIN(MAX(pkt->aa_samples, 1u), 6u);

    if (sx0 >= sx1 || sy0 >= sy1 || !(pkt->num_verts + pkt->num_line_verts)) {
        return 0;
    }
    if (ns > 1) {
        bool same_pitch = !pkt->depth.attach || pkt->depth.pitch == pkt->rt_pitch;
        for (uint32_t k = 1; k < ncb; k++) {
            same_pitch &= pkt->cb[k].pitch == pkt->rt_pitch;
        }
        if (!same_pitch || (uint64_t)ns * pkt->rt_pitch > 16384) {
            r300_metal_warn(512, "multisampled buffers of different pitches");
            return -1;
        }
    }
    /* Colour buffers: A from the packet's rt_*, B-D from cb[]. */
    for (uint32_t k = 0; k < ncb; k++) {
        uint32_t view = k ? pkt->cb[k].view : pkt->rt_view;
        uint32_t addr = k ? pkt->cb[k].gpu_addr : pkt->rt_gpu_addr;
        uint32_t bpr = ns * (k ? pkt->cb[k].pitch * pkt->cb[k].bpp
                               : pkt->rt_pitch * pkt->rt_bpp);
        cpf[k] = r300_rt_pf(view);
        NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:cpf[k]];
        if ((addr % align) || (bpr % align) || !bpr ||
            (uint64_t)addr + (uint64_t)bpr * pkt->rt_height > vram_size) {
            if (k == 0) {
                r300_metal_warn(1, "colour buffer unusable as a linear view");
                return -1;
            }
            r300_metal_warn(256, "render target B-D unusable as a linear view");
            ncb = k;
            break;
        }
        ck[k] = (R200TexKey){ addr, ns > 1 ? ns * pkt->rt_pitch : pkt->rt_width,
                              pkt->rt_height, bpr, (uint32_t)cpf[k] };
    }

    if ((pkt->cull & (R300_CULL_FRONT | R300_CULL_BACK)) ==
        (R300_CULL_FRONT | R300_CULL_BACK) && !pkt->num_line_verts) {
        return 0;                       /* every polygon culled */
    }

    @autoreleasepool {
        /* Depth/stencil buffer in place, as the colour attachment after
         * the colour buffers. */
        const R300DepthDesc *zd = &pkt->depth;
        MTLPixelFormat zpf = zd->bpp == 2 ? MTLPixelFormatR16Uint
                                          : MTLPixelFormatR32Uint;
        R200TexKey dk = { zd->gpu_addr, ns > 1 ? ns * zd->pitch : pkt->rt_width,
                          pkt->rt_height, ns * zd->pitch * zd->bpp, (uint32_t)zpf };
        bool want_ds = zd->attach;
        if (want_ds) {
            NSUInteger za = [dev minimumLinearTextureAlignmentForPixelFormat:zpf];
            if (zd->pitch < pkt->rt_width || (dk.offset % za) || (dk.pitch % za) ||
                (uint64_t)dk.offset + (uint64_t)dk.pitch * dk.height > vram_size) {
                r300_metal_warn(16, "depth buffer not usable in place");
                want_ds = false;
            }
        }
        /*
         * The views are pkt->rt_height rows (one height per buffer, see
         * r300_draw_build), but a draw only touches the rows above its
         * scissor's bottom: hazards and written ranges use those, or a
         * buffer's view reaching past its real end over textures would
         * flush the batch at almost every draw.
         */
        uint64_t dlo = dk.offset, dhi = dlo + (uint64_t)dk.pitch * sy1;

        id<MTLTexture> tex[R300_NUM_TEX_UNITS];
        id<MTLSamplerState> smp[R300_NUM_TEX_UNITS];
        R300FSUniforms u = pkt->uniforms;
        for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
            const R300TexDesc *td = &pkt->tex[t];
            bool raw = td->kind == R300_TEXK_RAW;
            tex[t] = r300_dummy(dev, raw, raw ? 0 : td->dim);
            smp[t] = r300_sampler(dev, td->filter0, td->levels);
            if (!td->bound) {
                u.tex_info[t][0] = 0;
                continue;
            }
            uint32_t rowel = u.tex_info[t][1];
            id<MTLTexture> x = r300_texture(st, dev, vram_ptr, vram_size, td, &rowel);
            if (!x) {
                u.tex_info[t][0] = 0;
                continue;
            }
            if (raw) {
                u.tex_info[t][1] = rowel;
            }
            tex[t] = x;
        }

        /* Batch hazards, as in the R200 path. */
        bool conflict = g_r200_nwritten >= R200_MAX_WRITTEN - 1 - (int)ncb ||
                        (want_ds && (r200_batch_conflict(dlo, dhi, &dk) ||
                                     r300_read_conflict(dlo, dhi, &dk)));
        for (uint32_t k = 0; k < ncb && !conflict; k++) {
            uint64_t lo = ck[k].offset, hi = lo + (uint64_t)ck[k].pitch * sy1;
            conflict = r200_batch_conflict(lo, hi, &ck[k]) || r300_read_conflict(lo, hi, &ck[k]);
        }
        for (int t = 0; t < R300_NUM_TEX_UNITS && !conflict; t++) {
            if (u.tex_info[t][0] && !pkt->tex[t].host_data) {
                conflict = r200_batch_conflict(pkt->tex[t].gpu_addr,
                    (uint64_t)pkt->tex[t].gpu_addr + pkt->tex[t].size_bytes, NULL);
            }
        }
        if (conflict) {
            g_r200_stat_conflicts++;
            r200_flush_locked(st);
        }
        if (!g_r200_cb) {
            g_r200_cb = r200_new_cb(st);
        }
        /* A pass is keyed by its colour buffers and, when it has one, its
         * depth buffer.  Depth-less draws may run in a pass with depth
         * (test and write off); depth draws need their own buffer bound. */
        bool enc_ds = g_r200_enc_depth_off != ~0u;
        bool same = g_r200_enc && g_r300_enc_mine && g_r300_enc_ncb == ncb &&
                    !memcmp(&g_r200_enc_key, &ck[0], sizeof(ck[0])) &&
                    (ncb == 1 || !memcmp(&g_r300_enc_cb[1], &ck[1], sizeof(ck[0]) * (ncb - 1))) &&
                    !(want_ds && (!enc_ds || g_r200_enc_depth_off != dk.offset ||
                                  g_r200_enc_depth_pitch != zd->pitch ||
                                  g_r200_enc_depth_z16 != (zd->bpp == 2)));
        if (g_r200_enc && !same) {
            [g_r200_enc endEncoding];
            [g_r200_enc release];
            g_r200_enc = nil;
        }
        if (!g_r200_enc) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            for (uint32_t k = 0; k < ncb; k++) {
                id<MTLTexture> rt = r200_view(st, ck[k], cpf[k], true);
                if (!rt) {
                    r300_metal_warn(4, "render target view failed");
                    return -1;
                }
                rp.colorAttachments[k].texture = rt;
                rp.colorAttachments[k].loadAction = MTLLoadActionLoad;
                rp.colorAttachments[k].storeAction = MTLStoreActionStore;
            }
            id<MTLTexture> ztex = want_ds ? r200_view(st, dk, zpf, true) : nil;
            if (want_ds && !ztex) {
                r300_metal_warn(32, "depth buffer view failed");
                want_ds = false;
            }
            if (ztex) {
                rp.colorAttachments[ncb].texture = ztex;
                rp.colorAttachments[ncb].loadAction = MTLLoadActionLoad;
                rp.colorAttachments[ncb].storeAction = MTLStoreActionStore;
                g_r200_enc_depth_off = dk.offset;
                g_r200_enc_depth_pitch = zd->pitch;
                g_r200_enc_depth_z16 = zd->bpp == 2;
                r200_note_written(dlo, dhi, &dk);
            } else {
                g_r200_enc_depth_off = ~0u;
            }
            g_r200_enc = [[g_r200_cb renderCommandEncoderWithDescriptor:rp] retain];
            [g_r200_enc setDepthClipMode:MTLDepthClipModeClamp];
            [g_r200_enc setViewport:(MTLViewport){ 0, 0, pkt->rt_width,
                                                  pkt->rt_height, 0, 1 }];
            g_r200_enc_key = ck[0];
            g_r300_enc_ncb = ncb;
            memcpy(g_r300_enc_cb, ck, sizeof(ck[0]) * ncb);
            g_r300_enc_mine = true;
            r300_ec_reset();
            g_r200_stat_passes++;
        }
        for (uint32_t k = 0; k < ncb; k++) {
            r200_note_written(ck[k].offset, ck[k].offset + (uint64_t)ck[k].pitch * sy1, &ck[k]);
        }
        if (want_ds && g_r200_enc_depth_off == dk.offset) {
            r200_note_written(dlo, dhi, &dk);
        }
        enc_ds = g_r200_enc_depth_off != ~0u;
        MTLPixelFormat pass_zpf = !enc_ds ? MTLPixelFormatInvalid :
                                  g_r200_enc_depth_z16 ? MTLPixelFormatR16Uint
                                                       : MTLPixelFormatR32Uint;
        if (enc_ds && !want_ds) {
            /* keep the pass's buffer as it is */
            u.zinfo[0] = 0;
            u.zinfo[3] = g_r200_enc_depth_z16 ? R300_ZFMT_Z16 : 0;
            u.poly_en = 0;
        }
        if (ncb != MAX(pkt->num_cb, 1u)) {
            return -1;                  /* the MSL declares targets we cannot bind */
        }
        id<MTLRenderPipelineState> pipe = r300_pipeline(dev, pkt->glsl, pkt->glsl_id, pkt->vs_glsl, pkt->vs_id,
                                                      cpf, ncb, pass_zpf);
        if (!pipe) {
            return -1;
        }
        id<MTLRenderCommandEncoder> enc = g_r200_enc;
        if (pipe != g_r300_ec.pipe) {
            [enc setRenderPipelineState:pipe];
            g_r300_ec.pipe = pipe;
            g_r300_ec_set++;
        } else {
            g_r300_ec_skipped++;
        }
        /* SU_CULL_MODE; the winding also decides [[front_facing]] for
         * two-sided stencil. */
        int winding = r300_front_ccw(pkt->cull) ? MTLWindingCounterClockwise
                                                : MTLWindingClockwise;
        if (winding != g_r300_ec.winding) {
            [enc setFrontFacingWinding:(MTLWinding)winding];
            g_r300_ec.winding = winding;
            g_r300_ec_set++;
        } else {
            g_r300_ec_skipped++;
        }
        MTLCullMode cull = (pkt->cull & R300_CULL_FRONT) ? MTLCullModeFront :
                           (pkt->cull & R300_CULL_BACK) ? MTLCullModeBack
                                                        : MTLCullModeNone;

        id<MTLBuffer> ib = nil;
        size_t ioff = 0;
        if (pkt->vs_glsl) {
            /* Vertex program on the GPU: input vertices, their uniforms
             * (R300_BIND_VSU = buffer 2) and the triangle list's indices. */
            if (!r300_bind_bytes(st, dev, enc, pkt->vs_in, pkt->vs_in_vecs * 16u, 0)) {
                return -1;
            }
            /* The 4.6 KB of vertex-program constants rarely change between
             * draws; keep the arena copy that is already bound. */
            if (!g_r300_ec.vsu) {
                g_r300_ec.vsu = g_malloc(sizeof(*g_r300_ec.vsu));
            }
            if (g_r300_ec.vsu_valid &&
                !memcmp(g_r300_ec.vsu, pkt->vs_u, sizeof(*pkt->vs_u))) {
                g_r300_ec_skipped++;
            } else {
                g_r300_ec.vsu_valid = false;
                if (!r300_bind_bytes(st, dev, enc, pkt->vs_u, sizeof(*pkt->vs_u), 2)) {
                    return -1;
                }
                memcpy(g_r300_ec.vsu, pkt->vs_u, sizeof(*pkt->vs_u));
                g_r300_ec.vsu_valid = true;
                g_r300_ec_set++;
            }
            size_t ibytes = (size_t)pkt->num_verts * 4;
            void *idst = r200_arena_alloc(st, ibytes, &ib, &ioff);
            if (idst) {
                memcpy(idst, pkt->vs_idx, ibytes);
                [ib retain];
            } else {
                ib = [dev newBufferWithBytes:pkt->vs_idx length:ibytes
                                     options:MTLResourceStorageModeShared];
                ioff = 0;
            }
        } else {
            uint32_t nv = pkt->num_verts + pkt->num_line_verts;
            if (!r300_bind_bytes(st, dev, enc, pkt->verts, (size_t)nv * sizeof(R300Vertex), 0)) {
                return -1;
            }
        }
        if (!g_r300_ec.zpass) {
            [enc setFragmentBuffer:r300_zpass_buf(dev) offset:0 atIndex:1];
            g_r300_ec.zpass = true;
        }
        for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
            if (tex[t] != g_r300_ec.tex[t]) {
                [enc setFragmentTexture:tex[t] atIndex:t];
                g_r300_ec.tex[t] = tex[t];
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
            if (smp[t] != g_r300_ec.smp[t]) {
                [enc setFragmentSamplerState:smp[t] atIndex:t];
                g_r300_ec.smp[t] = smp[t];
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
        }
        for (uint32_t k = 0; k < ns; k++) {
            /* Sample k: its columns, and the geometry moved so that the
             * pixel centres land where the sample is. */
            uint32_t cx = k * pkt->rt_pitch;
            float ms[4] = { 0, 0, 0, 0 };
            R300FSUniforms uk = u;
            if (ns > 1) {
                ms[0] = -2.0f * pkt->aa_pos[k][0] / pkt->rt_width;
                ms[1] = 2.0f * pkt->aa_pos[k][1] / pkt->rt_height;
                for (int i = 0; i < 4; i++) {
                    uk.cliprect[i][0] += cx;
                    uk.cliprect[i][2] += cx;
                }
                [enc setViewport:(MTLViewport){ cx, 0, pkt->rt_width,
                                                pkt->rt_height, 0, 1 }];
            }
            MTLScissorRect rect = { cx + sx0, sy0, sx1 - sx0, sy1 - sy0 };
            if (!g_r300_ec.rect_valid || memcmp(&rect, &g_r300_ec.rect, sizeof(rect))) {
                [enc setScissorRect:rect];
                g_r300_ec.rect = rect;
                g_r300_ec.rect_valid = true;
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
            if (!g_r300_ec.ms_valid || memcmp(ms, g_r300_ec.ms, sizeof(ms))) {
                [enc setVertexBytes:ms length:sizeof(ms) atIndex:1];
                memcpy(g_r300_ec.ms, ms, sizeof(ms));
                g_r300_ec.ms_valid = true;
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
            if (!g_r300_ec.fsu_valid || memcmp(&uk, &g_r300_ec.fsu, sizeof(uk))) {
                [enc setFragmentBytes:&uk length:sizeof(uk) atIndex:0];
                memcpy(&g_r300_ec.fsu, &uk, sizeof(uk));
                g_r300_ec.fsu_valid = true;
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
            if ((int)cull != g_r300_ec.cull) {
                [enc setCullMode:cull];
                g_r300_ec.cull = (int)cull;
                g_r300_ec_set++;
            } else {
                g_r300_ec_skipped++;
            }
            if (pkt->num_verts && ib) {
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:pkt->num_verts
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:ib
                         indexBufferOffset:ioff];
            } else if (pkt->num_verts) {
                [enc drawPrimitives:pkt->prim_class == 1 ? MTLPrimitiveTypeLine
                                                         : MTLPrimitiveTypeTriangle
                        vertexStart:0 vertexCount:pkt->num_verts];
            }
            if (pkt->num_line_verts) {
                /* polygon-mode edges: lines, never culled */
                [enc setCullMode:MTLCullModeNone];
                g_r300_ec.cull = (int)MTLCullModeNone;
                [enc drawPrimitives:MTLPrimitiveTypeLine vertexStart:pkt->num_verts
                        vertexCount:pkt->num_line_verts];
            }
        }
        [ib release];
        g_r200_stat_draws++;
        if (g_r200_stat_draws % 2000 == 0) {
            qemu_log("ppc-mac-gpu r300: %llu draws, %llu passes, encoder calls "
                     "%llu made / %llu skipped as already bound\n",
                     (unsigned long long)g_r200_stat_draws,
                     (unsigned long long)g_r200_stat_passes,
                     (unsigned long long)g_r300_ec_set,
                     (unsigned long long)g_r300_ec_skipped);
        }
        for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
            if (u.tex_info[t][0] && !pkt->tex[t].host_data) {
                r200_note_read(pkt->tex[t].gpu_addr,
                               (uint64_t)pkt->tex[t].gpu_addr + pkt->tex[t].size_bytes);
            }
        }
    }
    return 0;
}

/* A draw's deterministic rejections, as r200_draw_rejects().  Returns
 * -1 (rejected), 1 (nothing to draw) or 0 (encode it). */
static int r300_draw_rejects(PPCMacGPUMetalState *st, uint64_t vram_size,
                             const R300DrawPacket *pkt)
{
    if (!st || !st->vramBuffer) {
        return -1;
    }
    id<MTLDevice> dev = st->vramBuffer.device;
    uint32_t ncb = MAX(pkt->num_cb, 1u);
    uint32_t sx0 = pkt->scissor[0], sy0 = pkt->scissor[1];
    uint32_t sx1 = MIN(pkt->scissor[2], pkt->rt_width);
    uint32_t sy1 = MIN(pkt->scissor[3], pkt->rt_height);
    uint32_t ns = MIN(MAX(pkt->aa_samples, 1u), 6u);

    if (sx0 >= sx1 || sy0 >= sy1 || !(pkt->num_verts + pkt->num_line_verts)) {
        return 1;
    }
    if (ns > 1) {
        bool same_pitch = !pkt->depth.attach || pkt->depth.pitch == pkt->rt_pitch;
        for (uint32_t k = 1; k < ncb; k++) {
            same_pitch &= pkt->cb[k].pitch == pkt->rt_pitch;
        }
        if (!same_pitch || (uint64_t)ns * pkt->rt_pitch > 16384) {
            r300_metal_warn(512, "multisampled buffers of different pitches");
            return -1;
        }
    }
    for (uint32_t k = 0; k < ncb; k++) {
        uint32_t view = k ? pkt->cb[k].view : pkt->rt_view;
        uint32_t addr = k ? pkt->cb[k].gpu_addr : pkt->rt_gpu_addr;
        uint32_t bpr = ns * (k ? pkt->cb[k].pitch * pkt->cb[k].bpp
                               : pkt->rt_pitch * pkt->rt_bpp);
        MTLPixelFormat pf = r300_rt_pf(view);
        NSUInteger align = [dev minimumLinearTextureAlignmentForPixelFormat:pf];
        if ((addr % align) || (bpr % align) || !bpr ||
            (uint64_t)addr + (uint64_t)bpr * pkt->rt_height > vram_size) {
            if (k == 0) {
                r300_metal_warn(1, "colour buffer unusable as a linear view");
                return -1;
            }
            break;      /* targets B-D are dropped, as the encode side does */
        }
    }
    if ((pkt->cull & (R300_CULL_FRONT | R300_CULL_BACK)) ==
        (R300_CULL_FRONT | R300_CULL_BACK) && !pkt->num_line_verts) {
        return 1;                   /* every polygon culled */
    }
    return 0;
}

static int metal_draw_r300(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                           const R300DrawPacket *pkt)
{
    PPCMacGPUMetalState *st = opaque;
    int r;

    if (!rqueue_enabled()) {
        pthread_mutex_lock(&g_render_lock);
        r = r300_draw_encode(st, vram_ptr, vram_size, pkt);
        pthread_mutex_unlock(&g_render_lock);
        return r;
    }
    r = r300_draw_rejects(st, vram_size, pkt);
    if (r) {
        return r < 0 ? -1 : 0;      /* rejected, or nothing to draw */
    }
    /* The job takes the packet's heap buffers over (see r300_job_new);
     * the device's r300_draw_free frees what is left. */
    rq_push(r300_job_new((R300DrawPacket *)(uintptr_t)pkt));
    return 0;
}

static PPCMacGPURenderer metal_renderer = {
    .name              = "metal",
    .init              = metal_init,
    .fini              = metal_fini,
    .scanout           = metal_scanout,
    .blit_2d           = metal_blit_2d,
    .fill_2d           = metal_fill_2d,
    .draw_3d           = metal_draw_3d,
    .mode_change       = metal_mode_change,
    .srt_write_through = metal_srt_write_through,
    .draw_r200         = metal_draw_r200,
    .draw_r300         = metal_draw_r300,
    .r300_glsl_flags   = R300_GLSL_GPU_VS,
    .zpass_r300        = metal_zpass_r300,
    .flush_r200        = metal_flush_r200,
    .submit_r200       = metal_submit_r200,
    .fill_notify_r200  = metal_fill_notify_r200,
    .range_busy_r200   = metal_range_busy_r200,
    .get_caps          = metal_get_caps,
    .get_drag_state    = metal_get_drag_state,
    .get_drag_snap     = NULL,
    .flush_drag_paste  = metal_flush_drag_paste,
};

PPCMacGPURenderer *ppc_mac_gpu_renderer_metal(void)
{
    return &metal_renderer;
}

#endif /* CONFIG_DARWIN */
