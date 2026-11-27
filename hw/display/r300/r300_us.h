/*
 * R300-family fragment shader unit (US) -> GLSL.
 *
 * Encodings follow AMD's "R3xx 3D Registers" reference (US_* registers).
 * The generated source (Vulkan GLSL 4.50) holds a pass-through vertex
 * shader for post-transform vertices and a fragment shader that runs the
 * R300 program, then does the alpha test, blending and colour-buffer
 * packing in the shader (the guest's big-endian byte order rules out
 * fixed-function blending on the host).  r300_spirv.c compiles it to
 * SPIR-V for Vulkan and on to MSL for Metal.
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_US_H
#define HW_DISPLAY_R300_US_H

#include <stdbool.h>
#include <stdint.h>

#include "r300_state.h"

#define R300_US_NUM_TEMPS       32
#define R300_US_NUM_CONSTS      32
#define R300_US_MAX_ALU         64
#define R300_US_MAX_TEX         32
#define R300_NUM_TEX_UNITS      16
#define R300_NUM_VARYINGS       10
#define R300_US_MAX_TARGETS     4       /* render targets A-D */

/*
 * Post-transform vertex as the device hands it to the renderer: pos is in
 * clip space with y up (Metal's convention; Vulkan flips its viewport), v[k] are the values the rasterizer interpolates into
 * fragment temporaries (see R300FSDesc.route).
 */
typedef struct R300Vertex {
    float pos[4];
    float v[R300_NUM_VARYINGS][4];
    float aux[4];                   /* x: fog value (GB_SELECT.FOG_SELECT, GA_FOG_*) */
    float ucp[8];                   /* user clip plane distances (VAP_CLIP_CNTL) */
} R300Vertex;

/* Must match uniform block R300FSUniforms in the generated GLSL (std140). */
typedef struct R300FSUniforms {
    float consts[R300_US_NUM_CONSTS][4];
    float blend_color[4];
    uint32_t tex_swz[R300_NUM_TEX_UNITS][4];  /* R,G,B,A selects: 0-3 XYZW, 4 zero, 5 one */
    uint32_t tex_info[R300_NUM_TEX_UNITS][4]; /* x: 1 bound, y: raw->XYZW (0 as is, 1 .abgr, 2 .grba),
                                                 z, w: width, height in texels */
    uint32_t cblend, ablend, chanmask, alpha_func;
    uint32_t out_sel[4];                      /* US_OUT_FMT_0 C0..C3 (0 A, 1 R, 2 G, 3 B) */
    uint32_t rt_swap32, clip_rule;            /* stored bytes are C3,C2,C1,C0 */
    uint32_t rt_endian, rop;                  /* RB3D_COLORPITCH0.COLOR_ENDIAN, RB3D_ROPCNTL */
    int32_t cliprect[4][4];                   /* SC_CLIPRECT x0, y0, x1, y1 (inclusive) */
    /* Depth/stencil: ZB_CNTL, ZB_ZSTENCILCNTL, ZB_STENCILREFMASK, and
     * R300_ZFMT_* (layout of the buffer bound after the colour buffers). */
    uint32_t zinfo[4];
    uint32_t zpass_count, poly_en, pad2[2];   /* count Z-pass samples (ZB_ZPASS_*);
                                                 SU_POLY_OFFSET_ENABLE */
    float tex_border[R300_NUM_TEX_UNITS][4];  /* TX_BORDER_COLOR, XYZW before the swizzle */
    float tex_lod[R300_NUM_TEX_UNITS][4];     /* LOD bias, finest level (MAX_MIP_LEVEL),
                                                 coarsest level, mip filter (0 none,
                                                 1 nearest, 2 linear) */
    uint32_t tex_dim[R300_NUM_TEX_UNITS][4];  /* R300_TEXDIM_*, level-0 depth, level-0
                                                 pitch in bytes, R300_TEXF_* */
    float fog_color[4];                       /* FG_FOG_COLOR_*, w: FG_FOG_FACTOR */
    uint32_t fog_blend, depth_src, pad3[2];   /* FG_FOG_BLEND, FG_DEPTH_SRC */
    float poly_offset[4];                     /* front scale, offset, back scale, offset,
                                                 in units of the [0,1] depth range */
    uint32_t tex_addr[R300_NUM_TEX_UNITS][4]; /* x: byte address of a raw unit's texels,
                                                 y: 0 in VRAM, 1 in the aux buffer
                                                 (R300_GLSL_VRAM_SSBO only) */
} R300FSUniforms;

/* tex_dim[k].w */
#define R300_TEXF_SIGNED_X      (1u << 0)     /* .. W at bit 3 (TX_FORMAT1.SIGNED_*) */
#define R300_TEXF_GAMMA         (1u << 4)
#define R300_TEXF_POT_ROWS      (1u << 5)     /* heights round up to a power of two */

#define R300_ZFMT_ENDIAN_MASK   3u            /* ZB_DEPTHPITCH.DEPTHENDIAN */
#define R300_ZFMT_Z16           (1u << 2)     /* 16-bit Z, no stencil */

/* What the translator needs besides the US registers themselves. */
typedef struct R300FSDesc {
    int8_t route[R300_US_NUM_TEMPS];    /* temp <- varying index, or -1 */
} R300FSDesc;

/*
 * Colour buffers a draw binds: 1 + the highest render target (B-D) the
 * program writes whose US_OUT_FMT is not UNUSED, or RB3D_CCTL's
 * NUM_MULTIWRITES (target A replicated), whichever is larger.
 */
uint32_t r300_us_num_targets(const R300State *st);

/* US_OUT_FMT of render target k (target A's for a multiwrite target left
 * UNUSED). */
uint32_t r300_us_out_fmt(const R300State *st, unsigned k);

/*
 * Resource bindings of the generated GLSL (descriptor set 0).  Metal gets
 * them as: uniforms fragment buffer 0, Z-pass counter fragment buffer 1,
 * vertices vertex buffer 0, sample shift vertex buffer 1, unit k texture
 * and sampler k, input attachment k colour attachment k.
 */
enum {
    R300_BIND_UNIFORMS = 0,     /* UBO R300FSUniforms (fragment) */
    R300_BIND_ZPASS = 1,        /* SSBO uint: ZB_ZPASS counter (fragment) */
    R300_BIND_VERTS = 2,        /* SSBO R300Vertex[] (vertex) */
    R300_BIND_MS = 3,           /* UBO vec4: multisample clip-space shift */
    R300_BIND_VRAM = 4,         /* SSBO uint[]: VRAM (R300_GLSL_VRAM_SSBO) */
    R300_BIND_AUX = 5,          /* SSBO uint[]: GART texels (R300_GLSL_VRAM_SSBO) */
    R300_BIND_VSU = 6,          /* UBO R300VSU: GPU vertex shader uniforms */
    R300_BIND_FB0 = 8,          /* input attachments: colour buffers, then Z */
    R300_BIND_TEX0 = 16,        /* combined image samplers, one per unit */
    R300_BIND_COUNT = 32,
};

/* r300_us_to_glsl flags */
#define R300_GLSL_VRAM_SSBO     (1u << 0)   /* raw units read VRAM from a storage
                                               buffer, not a uint texture view */
#define R300_GLSL_GPU_VS        (1u << 1)   /* the renderer runs vertex programs
                                               (r300_draw.h vs_glsl) */

/*
 * Build the GLSL source for the current US program.  It compiles as three
 * shaders by the macro defined: R300_VS (vertex), R300_FS (colour buffers
 * only) or R300_FS_Z (colour buffers plus the depth/stencil buffer as a
 * uint colour attachment after them: R32Uint for Z24S8, R16Uint for Z16).
 * Colour buffer k is output location k and input attachment k
 * (r300_us_num_targets of them); the fragment shaders read the buffers'
 * current contents through the input attachments (framebuffer fetch).
 * Returns a malloc'd string, or NULL with *err set for programs not yet
 * translated.  Every register the source depends on is folded into the
 * text, so the string is its own cache key.
 */
char *r300_us_to_glsl(const R300State *st, const R300FSDesc *desc,
                      uint32_t flags, const char **err);

/*
 * r300_us_to_glsl(), remembered: returns the same text for the same
 * inputs without regenerating it.  *id names the text (equal ids, equal
 * text), so a renderer can cache per id without hashing the string; 0
 * means none (not cacheable).  Not thread-safe.
 */
char *r300_us_glsl_cached(const R300State *st, const R300FSDesc *desc,
                          uint32_t flags, uint32_t *id, const char **err);

/*
 * Texture formats (TX_FORMAT1.TXFORMAT) the shader decodes itself from the
 * texel words, read through a uint view of guest memory: returns the texel
 * size in bytes, or 0 for the formats sampled through a float texture
 * (X8, X16, Y8X8, the 16-bit colour formats, W8Z8Y8X8, DXT) and for ones
 * not implemented.  The uint view's element is 4 bytes for texels of up
 * to 4 bytes, else the texel.
 *
 * The card's view of a texel dword follows the rule the 32-bit formats
 * established: dword = TXO_ENDIAN swap of the big-endian dword in VRAM
 * (VRAM holds the guest CPU's side of its byte-swapping aperture), and
 * the texel at the lowest address is in the low bits.  GART copies are
 * byte-reversed per dword on upload so the same rule holds.
 */
uint32_t r300_tex_raw_bpp(uint32_t txformat);

/* Colour buffer layout, from RB3D_COLORPITCH0.COLORFORMAT and
 * US_OUT_FMT_0.OUT_FMT. */
enum {
    R300_RTV_NONE = 0,          /* unsupported combination */
    R300_RTV_RGBA8,             /* ARGB8888 with C4_8: float RGBA8 view */
    R300_RTV_R8U,               /* 1-byte pixels, uint views ... */
    R300_RTV_R16U,
    R300_RTV_R32U,
    R300_RTV_RG32U,
    R300_RTV_RGBA32U,
};
uint32_t r300_cb_view(uint32_t colorformat, uint32_t outfmt, uint32_t *bpp);

/* R300 US constants are 24-bit floats (1 sign, 7 exponent, 16 mantissa). */
float r300_float24(uint32_t v);

/* Disassemble the active US program, for traces.  Caller frees. */
char *r300_us_disasm(const R300State *st);

#endif
