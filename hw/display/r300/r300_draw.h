/*
 * R300-family draw assembly: turns a 3D_DRAW_* packet plus the register
 * state into a self-contained draw for the host renderer.
 *
 *   vertex fetch (VAP_PROG_STREAM_CNTL, immediate data or 3D_LOAD_VBPNTR
 *   arrays) -> vertex program (PVS, on the CPU) -> viewport transform
 *   (VAP_VTE_CNTL, SE_VPORT_*) -> primitive assembly into a triangle,
 *   line or point list -> rasterizer routing (RS_IP/RS_INST) of vertex
 *   outputs into fragment temporaries.
 *
 * The fragment program, blending and colour-buffer packing are compiled
 * from r300_us_to_glsl(); the renderer only binds and draws.
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_DRAW_H
#define HW_DISPLAY_R300_DRAW_H

#include <stdbool.h>
#include <stdint.h>

#include "r300_state.h"
#include "r300_us.h"

#define R300_MAX_ARRAYS 16

/* 3D_LOAD_VBPNTR state: one entry per vertex array. */
typedef struct R300Arrays {
    unsigned count;
    struct {
        uint32_t addr;          /* GPU address */
        uint32_t size_dw;       /* dwords per element */
        uint32_t stride_dw;     /* dwords between elements */
    } a[R300_MAX_ARRAYS];
} R300Arrays;

/* How the renderer gets a texture unit's texels. */
enum {
    R300_TEXK_RGBA8 = 0,        /* 32bpp view; shader reverses the bytes */
    R300_TEXK_R8,               /* X8 view */
    R300_TEXK_RG8,              /* Y8X8 view; shader swaps the bytes */
    R300_TEXK_CONVERT16,        /* 16bpp packed: decoded to RGBA8 on the CPU */
    R300_TEXK_RAW,              /* uint view of the texel dwords, decoded by
                                   the shader (r300_tex_raw_bpp) */
    R300_TEXK_DXT1,             /* compressed blocks, copied as they lie */
    R300_TEXK_DXT3,
    R300_TEXK_DXT5,
};

/* TX_FORMAT1.TEX_COORD_TYPE */
enum { R300_TEXDIM_2D = 0, R300_TEXDIM_3D = 1, R300_TEXDIM_CUBE = 2 };

#define R300_TEX_MAX_LEVELS     12      /* 2048 .. 1 */

typedef struct R300TexDesc {
    bool bound;
    uint32_t gpu_addr;          /* TX_OFFSET, low bits cleared */
    uint32_t width, height;
    uint32_t endian;            /* TX_OFFSET[1:0]: the card's dword swap */
    uint32_t pitch_bytes;       /* level 0: bytes per row (per row of blocks for DXT) */
    uint32_t format;            /* TX_FORMAT1 & 0x1F */
    uint32_t kind;              /* R300_TEXK_* */
    uint32_t view_bpp;          /* RAW: bytes per uint-view element (4, 8, 16) */
    uint32_t filter0;           /* TX_FILTER0 */
    uint32_t filter1;           /* TX_FILTER1 */
    uint8_t *host_data;         /* texels copied out of the GART (owned), or NULL */

    /*
     * The whole image as the sampler addresses it (r300_tex_layout): mip
     * levels one after another, each holding its cube faces or 3D slices
     * back to back.  Rows are 32-byte aligned below level 0 (and at level
     * 0 without TX_PITCH_EN), and heights round up to a power of two when
     * the texture has mip levels or is 3D or a cube.
     */
    uint32_t dim;               /* R300_TEXDIM_* */
    uint32_t depth;             /* 3D: slices at level 0, else 1 */
    uint32_t levels;            /* TX_FORMAT0.NUM_LEVELS + 1 */
    uint32_t lvl_off[R300_TEX_MAX_LEVELS];      /* bytes from gpu_addr */
    uint32_t lvl_pitch[R300_TEX_MAX_LEVELS];    /* bytes per row (of blocks) */
    uint32_t lvl_rows[R300_TEX_MAX_LEVELS];     /* rows (of blocks) per face/slice */
    uint32_t size_bytes;        /* the whole chain */
} R300TexDesc;

/* Width, height and depth (slices; 6 faces for a cube) of mip level l. */
void r300_tex_level_dims(const R300TexDesc *td, uint32_t l, uint32_t *w,
                         uint32_t *h, uint32_t *d);

/* Fill the layout fields of td from its size, format, dim and levels;
 * bpp is bytes per texel (per 4x4 block for DXT, flagged by dxt). */
void r300_tex_layout(R300TexDesc *td, uint32_t bpp, bool dxt, bool pitch_en);

/* A TX_BORDER_COLOR dword as the unit's XYZW before the swizzle: packed
 * like a texel of the unit's format (8888 for texels wider than 32 bits;
 * DXT as B8G8R8A8). */
void r300_border_color(uint32_t txformat, uint32_t v, float out[4]);

/* Depth/stencil buffer (ZB_*).  It stays in guest memory in the card's
 * layout; the renderer tests and writes it in the fragment shader. */
typedef struct R300DepthDesc {
    bool attach;                /* ZB_CNTL enables the Z or stencil test */
    uint32_t gpu_addr;          /* ZB_DEPTHOFFSET */
    uint32_t pitch;             /* pixels */
    uint32_t bpp;               /* 2 (Z16, Z13E3) or 4 (Z24S8) */
    uint32_t format;            /* ZB_FORMAT.DEPTHFORMAT */
} R300DepthDesc;

/* SU_CULL_MODE */
#define R300_CULL_FRONT         (1u << 0)
#define R300_CULL_BACK          (1u << 1)
#define R300_FACE_CW            (1u << 2)   /* else counter-clockwise front */

/*
 * The card's winding is the one seen on screen with row 0 at the top (as
 * Mesa's r300 maps gallium's front_ccw straight to FRONT_FACE_CCW), which
 * is also how Metal judges winding for the clip-space positions this
 * module produces (their y is up).  So FACE_CCW is MTLWindingCounterClockwise.
 */
static inline bool r300_front_ccw(uint32_t cull)
{
    return !(cull & R300_FACE_CW);
}

/* Colour buffers 1-3 of a multiple-render-target draw (buffer 0 is the
 * packet's rt_* fields). */
typedef struct R300ColorDesc {
    uint32_t gpu_addr;          /* RB3D_COLOROFFSETn */
    uint32_t pitch;             /* pixels */
    uint32_t format;            /* RB3D_COLORPITCHn format field */
    uint32_t bpp;
    uint32_t view;              /* R300_RTV_* */
} R300ColorDesc;

typedef struct R300DrawPacket {
    /* Colour buffer 0 */
    uint32_t rt_gpu_addr;
    uint32_t rt_pitch;          /* pixels */
    uint32_t rt_width, rt_height;
    uint32_t rt_format;         /* RB3D_COLORPITCH0 format field */
    uint32_t rt_bpp;            /* bytes per pixel */
    uint32_t rt_view;           /* R300_RTV_* */
    uint32_t scissor[4];        /* x0, y0, x1, y1 (exclusive) */
    /* Colour buffers 1..num_cb-1 (render targets B-D the program writes,
     * or RB3D_CCTL multiwrites); the depth buffer is bound after them, at
     * colour attachment num_cb. */
    uint32_t num_cb;
    R300ColorDesc cb[R300_US_MAX_TARGETS];
    R300DepthDesc depth;
    uint32_t cull;              /* SU_CULL_MODE (0 for lines and points) */

    /*
     * Multisampling (GB_AA_CONFIG): with aa_samples > 1 every sample is
     * rendered, as its own copy of the image.  Sample k of row y is row
     * y * aa_samples + k of each colour and depth buffer, which fills the
     * aa_samples-times footprint the driver allocates for them.  aa_pos
     * is each sample's offset from the pixel centre in pixels (GB_MSPOS,
     * twelfths of a pixel from the pixel's corner).
     */
    uint32_t aa_samples;
    float aa_pos[6][2];

    /* Fragment stage */
    char *glsl;                 /* r300_us_to_glsl() source; owned by the packet */
    uint32_t glsl_id;           /* r300_us_glsl_cached() id of glsl, 0 if none */

    /*
     * Vertex shading on the host GPU (R300_GLSL_GPU_VS).  When vs_glsl is
     * set, verts is NULL and the renderer draws num_verts indices vs_idx
     * into vs_in (vs_u->info[0] vec4s per vertex) with the vertex shader
     * vs_glsl (id vs_id, as glsl_id) and uniforms vs_u at binding
     * R300_BIND_VSU.  Triangles only (prim_class 0).
     */
    char *vs_glsl;
    uint32_t vs_id;
    float (*vs_in)[4];
    uint32_t vs_in_vecs;
    uint32_t *vs_idx;
    struct R300VSUniforms *vs_u;
    R300FSUniforms uniforms;
    R300TexDesc tex[R300_NUM_TEX_UNITS];

    /* Geometry, expanded (no indices): num_verts vertices of a triangle
     * or line list (prim_class), then num_line_verts more of a line list
     * (polygon-mode edges drawn with filled polygons).  Points and wide
     * lines arrive as triangles. */
    R300Vertex *verts;          /* owned by the packet */
    uint32_t num_verts;
    uint32_t num_line_verts;
    uint32_t prim_class;        /* 0 triangles, 1 lines (of num_verts) */

    uint32_t warn;              /* R300_WARN_* bits for things skipped */
} R300DrawPacket;

#define R300_WARN_PVS           (1u << 0)
#define R300_WARN_DEPTH         (1u << 1)
#define R300_WARN_TEXFMT        (1u << 2)
#define R300_WARN_VTXFMT        (1u << 3)
#define R300_WARN_PRIM          (1u << 4)
#define R300_WARN_RTFMT         (1u << 5)
#define R300_WARN_VARYINGS      (1u << 6)
#define R300_WARN_FLOW          (1u << 7)   /* PVS flow control ran away */

/* Reads guest GPU memory (VRAM or GART) for vertex arrays and indices.
 * Returns false if the range is not mapped. */
typedef bool (*R300ReadFn)(void *opaque, uint32_t gpu_addr, void *dst,
                           uint32_t len);

/* Index buffer from memory (INDX_BUFFER after a DRAW_INDX_2 with no
 * inline indices), already fetched; NULL for other draws. */
typedef struct R300Indices {
    const uint32_t *dw;         /* raw dwords as fetched (VC_SWAP applied here) */
    uint32_t ndw;
} R300Indices;

/* Decode a 3D_LOAD_VBPNTR (type-3 0x2F) body. */
void r300_load_vbpntr(R300Arrays *arr, const uint32_t *d, uint32_t ndw);

/*
 * Build a draw from a 3D_DRAW_VBUF_2 (0x34), 3D_DRAW_IMMD_2 (0x35) or
 * 3D_DRAW_INDX_2 (0x36) body.  Returns false (with *err) when the draw
 * cannot be assembled at all; smaller omissions are flagged in pkt->warn.
 * On success the caller owns pkt->glsl and pkt->verts (r300_draw_free).
 */
bool r300_draw_build(const R300State *st, const R300Arrays *arr,
                     uint32_t opcode, const uint32_t *d, uint32_t ndw,
                     R300ReadFn read, void *opaque,
                     R300DrawPacket *pkt, const char **err);

/*
 * The same for DRAW_INDX_2 whose indices come from memory: vf is the
 * packet's VAP_VF_CNTL and idx the dwords INDX_BUFFER pointed at, read
 * as they lie (VC_SWAP is applied here, as the card's vertex cache does
 * for all fetched data).
 */
bool r300_draw_build_indexed(const R300State *st, const R300Arrays *arr,
                             uint32_t vf, const R300Indices *idx,
                             R300ReadFn read, void *opaque,
                             R300DrawPacket *pkt, const char **err);

/* Index i (0-based) of a DRAW_INDX_2 index stream: 16-bit indices are
 * packed two per dword, the first in the low half. */
uint32_t r300_index_at(const uint32_t *dw, uint32_t ndw, bool i32, uint32_t i);

/* GB_AA_CONFIG's sample count (1 without antialiasing), and GB_MSPOS's
 * sample positions as offsets from the pixel centre, in pixels. */
uint32_t r300_aa_samples(const R300State *st);
void r300_aa_positions(const R300State *st, float pos[6][2]);

/*
 * Byte offset of sample `sample` of pixel (x, y) in a multisampled R300
 * colour/depth buffer (ns = 2 or 4 samples) as Apple's ATIRadeon9700
 * computes it for CPU access (ATIR300Surface get_offset_of_sample_0):
 * 4x4-pixel micro tiles holding all samples, in 8-row bands, pitch_px
 * pixels wide, bpp 2 or 4.  Returns ~0u for other sample counts.
 */
uint32_t r300_msaa_offset(uint32_t x, uint32_t y, uint32_t ns,
                          uint32_t pitch_px, uint32_t bpp, uint32_t sample);

/* Assemble a primitive list (VAP_VF_CNTL.PRIM_TYPE) of n vertices into
 * triangle/line/point list indices; returns the index count (0 for an
 * unsupported type).  list needs room for 3n + 6 entries. */
uint32_t r300_assemble(unsigned prim, uint32_t n, uint32_t *list,
                       uint32_t *cls);

/* Must match uniform block R300VSU in the GPU vertex shader (std140). */
typedef struct R300VSUniforms {
    float c[256][4];            /* PVS constants from CONST_BASE_OFFSET */
    float ucp[6][4];            /* user clip planes */
    float vp0[4];               /* viewport x scale, x offset, y scale, y offset */
    float vp1[4];               /* z scale, z offset, colour buffer W, H */
    float fogp[4];              /* GA_FOG_SCALE, GA_FOG_OFFSET */
    uint32_t info[4];           /* x: vec4s per input vertex */
} R300VSUniforms;

/* Per output primitive: the source primitive's FIRST, SECOND, THIRD and
 * LAST vertex (flat-shading candidates) and its triangle edge flags. */
typedef struct R300Prov {
    uint32_t v[4];
    uint32_t edges;
} R300Prov;

/* r300_assemble, also filling prov (room for 2n + 2 entries) or NULL. */
uint32_t r300_assemble_prov(unsigned prim, uint32_t n, uint32_t *list,
                            uint32_t *cls, R300Prov *prov);

void r300_draw_free(R300DrawPacket *pkt);

/*
 * One face/slice of mip level l of a texture a renderer uploads (kinds
 * other than R300_TEXK_RAW), repacked with tight rows: 16bpp formats
 * decoded to RGBA8, DXT blocks and 8/16/32-bit texels copied as they lie.
 * src is the face's first byte; w, h the level's size (for DXT, rounded
 * up to whole blocks).  Returns malloc'd bytes and their row pitch.
 */
uint8_t *r300_tex_level_bytes(const R300TexDesc *td, const uint8_t *src,
                              uint32_t l, uint32_t w, uint32_t h, uint32_t *bpr);

/* 64-bit content hash, for caches of textures copied out of guest memory. */
uint64_t r300_hash_bytes(const uint8_t *p, size_t n);
uint64_t r300_hash_tex(const uint8_t *p, size_t n);
uint64_t r300_tex_identity(const R300TexDesc *td, const uint8_t *src);

#endif
