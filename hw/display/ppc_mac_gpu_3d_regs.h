/*
 * PPC Mac GPU - R200/RV280 3D Engine Register Definitions
 *
 * Register addresses for the ATI Radeon 9200 (RV280) 3D engine,
 * needed for Quartz Extreme / OpenGL acceleration.
 *
 * References:
 *   - Mesa r200_reg.h
 *   - Linux radeon_reg.h
 *   - ATI R200 programming guide
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_PPC_MAC_GPU_3D_REGS_H
#define HW_DISPLAY_PPC_MAC_GPU_3D_REGS_H

/* ========================================================================
 * 3D Register Shadow Array
 *
 * All 3D registers in range 0x1C00-0x3FFF are stored in:
 *   regs_3d[(addr - 0x1C00) / 4]
 *
 * Helper macros for accessing shadow registers:
 * ======================================================================== */

#define R200_3D_REG_BASE     0x1C00
#define R200_3D_REG_END      0x4000
#define R200_3D_REG_COUNT    ((R200_3D_REG_END - R200_3D_REG_BASE) / 4) /* 2304 */

/* Convert 3D register address to shadow array index */
#define R200_3D_IDX(addr)    (((addr) - R200_3D_REG_BASE) / 4)

/* ========================================================================
 * Pixel Pipeline Control
 * ======================================================================== */

#define R200_PP_MISC                    0x1C14
#define R200_PP_CNTL                    0x1C38
/*   PP_CNTL bits:
 *   [0]     TEX_0_ENABLE
 *   [1]     TEX_1_ENABLE
 *   [2]     TEX_2_ENABLE
 *   [3]     TEX_3_ENABLE (R200 has up to 6, but QE likely uses 1-2)
 *   [4]     TEX_4_ENABLE
 *   [5]     TEX_5_ENABLE
 *   [11]    ANTI_ALIAS_NONE
 *   [13:12] NUM_OUTPUT_PIXELS
 */

/* ========================================================================
 * Render Backend (RB3D)
 * ======================================================================== */

/* Already in main header but repeated here for reference:
 * #define R200_RB3D_CNTL               0x1C3C */
#define R200_RB3D_COLOROFFSET           0x1C40
#define R200_RB3D_COLORPITCH            0x1C48
/*   RB3D_COLORPITCH bits:
 *   [13:0]  COLORPITCH (pitch in pixels / 8, or dwords depending on fmt)
 *   [23:16] COLORTILE (tiling)
 */

#define R200_RB3D_ZSTENCILCNTL          0x1C2C
/*   RB3D_ZSTENCILCNTL bits:
 *   [3:0]   Z_TEST_FUNC
 *   [4]     STENCIL_ENABLE
 *   [5]     Z_ENABLE
 *   [6]     Z_WRITE_ENABLE
 */

#define R200_RB3D_BLENDCNTL             0x3220
/*   RB3D_BLENDCNTL bits:
 *   [3:0]   SRC_BLEND_FACTOR
 *   [7:4]   reserved
 *   [11:8]  reserved
 *   [15:12] reserved
 *   [19:16] DST_BLEND_FACTOR
 *   [23:20] reserved
 *   [24]    ALPHA_BLEND_ENABLE
 *   [25]    reserved
 *   [26]    SEPARATE_ALPHA_ENABLE
 *
 * Blend factors:
 *   0x0 = ZERO, 0x1 = ONE
 *   0x2 = SRC_COLOR, 0x3 = ONE_MINUS_SRC_COLOR
 *   0x4 = SRC_ALPHA, 0x5 = ONE_MINUS_SRC_ALPHA
 *   0x6 = DST_ALPHA, 0x7 = ONE_MINUS_DST_ALPHA
 *   0x8 = DST_COLOR, 0x9 = ONE_MINUS_DST_COLOR
 *   0xA = SRC_ALPHA_SAT
 *   0xC = CONST_COLOR, 0xD = ONE_MINUS_CONST_COLOR
 *   0xE = CONST_ALPHA, 0xF = ONE_MINUS_CONST_ALPHA
 */

#define R200_RB3D_ABLENDCNTL            0x3224  /* Separate alpha blend */
#define R200_RB3D_CBLENDCNTL            0x3228  /* Color blend override */

#define R200_RB3D_DEPTHOFFSET           0x1C24
#define R200_RB3D_DEPTHPITCH            0x1C28
#define R200_RB3D_DEPTHCLEARVALUE       0x3230

#define R200_RB3D_DSTCACHE_CTLSTAT      0x325C
/*   bit 0: DC_FLUSH — flush destination cache
 *   bit 1: DC_FREE  — free dest cache lines */
#define R200_RB3D_ZCACHE_CTLSTAT        0x3254

#define R200_RB3D_ROPCNTL               0x1D98  /* Raster op (unused for QE) */

/* ========================================================================
 * Setup Engine (SE)
 * ======================================================================== */

/* Already in main header:
 * #define R200_SE_CNTL                 0x1C4C */
/*   SE_CNTL bits:
 *   [1:0]   FFACE_CULL_DIR
 *   [3:2]   BFACE_CULL_DIR
 *   [4]     FFACE_SOLID
 *   [5]     BFACE_SOLID
 *   [6]     FLAT_SHADE_VTX_LAST
 *   [7]     DIFFUSE_SHADE_FLAT
 *   [8]     ALPHA_SHADE_FLAT
 *   [9]     SPECULAR_SHADE_FLAT
 *   [10]    FOG_SHADE_FLAT
 *   [11]    ROUND_MODE
 *   [13:12] VTX_PIX_CENTER
 *   [15:14] reserved
 *   [20:16] VPORT_X_SCALE_ENA ... VPORT_Z_OFFSET_ENA
 */

#define R200_SE_CNTL_STATUS             0x2140

/* ========================================================================
 * Vertex Format
 * ======================================================================== */

#define R200_SE_VTX_FMT_0               0x2088
/*   SE_VTX_FMT_0 bits:
 *   [0]     VTX_XY     — position has XY (always set)
 *   [1]     VTX_Z      — position has Z
 *   [2]     VTX_W0     — position has W
 *   [3]     VTX_W0_FMT — W format
 *   [4]     VTX_COLOR_0_FMT — 0=RGBA float, 1=RGBA8
 *   [5]     VTX_COLOR_1_FMT
 *   [6]     VTX_COLOR_2_FMT
 *   [7]     VTX_COLOR_3_FMT
 *   [10:8]  VTX_COLOR_0 (# color components: 0=none, 1=1, 2=2, 3=3, 4=4)
 *   [13:11] VTX_COLOR_1
 *   [16:14] VTX_COLOR_2
 *   [19:17] VTX_COLOR_3
 *   [20]    VTX_PT_SIZE — has point size
 */

#define R200_SE_VTX_FMT_1               0x208C
/*   SE_VTX_FMT_1 bits:
 *   [2:0]   TEX_0_COMP_CNT  (0=S, 1=ST, 2=STR, 3=STRQ)
 *   [5:3]   TEX_1_COMP_CNT
 *   [8:6]   TEX_2_COMP_CNT
 *   [11:9]  TEX_3_COMP_CNT
 *   [14:12] TEX_4_COMP_CNT
 *   [17:15] TEX_5_COMP_CNT
 */

#define R200_SE_VTE_CNTL                0x20B0
/*   SE_VTE_CNTL bits:
 *   [0]     VPORT_X_SCALE_ENA
 *   [1]     VPORT_X_OFFSET_ENA
 *   [2]     VPORT_Y_SCALE_ENA
 *   [3]     VPORT_Y_OFFSET_ENA
 *   [4]     VPORT_Z_SCALE_ENA
 *   [5]     VPORT_Z_OFFSET_ENA
 *   [8]     VTX_XY_FMT (0=float, 1=int)
 *   [9]     VTX_Z_FMT
 *   [10]    VTX_W0_FMT
 */

#define R200_SE_TCL_OUTPUT_VTX_FMT_0    0x2090
#define R200_SE_TCL_OUTPUT_VTX_FMT_1    0x2094

/* ========================================================================
 * Viewport Transform
 * ======================================================================== */

#define R200_SE_VPORT_XSCALE            0x1D98
#define R200_SE_VPORT_XOFFSET           0x1D9C
#define R200_SE_VPORT_YSCALE            0x1DA0
#define R200_SE_VPORT_YOFFSET           0x1DA4
#define R200_SE_VPORT_ZSCALE            0x1DA8
#define R200_SE_VPORT_ZOFFSET           0x1DAC

/* ========================================================================
 * Rasterizer / Scissor (3D path)
 * ======================================================================== */

#define R200_RE_CNTL                    0x1C50
/*   RE_CNTL bits:
 *   [0]     SCISSOR_ENABLE
 *   [1]     ZFUNC_ENABLE (Z testing)
 *   [3:2]   CULL_MODE
 *   [4]     DITHER_ENABLE
 */

#define R200_RE_TOP_LEFT                0x26C0
#define R200_RE_WIDTH_HEIGHT            0x26C4
#define R200_RE_MISC                    0x26C8

/* ========================================================================
 * Texture Unit 0
 * ======================================================================== */

#define R200_PP_TXFILTER_0              0x2C00
/*   PP_TXFILTER bits:
 *   [2:0]   MAG_FILTER (0=nearest, 1=linear, 2=aniso)
 *   [6:4]   MIN_FILTER
 *   [10:8]  MIP_FILTER (0=none, 1=nearest, 2=linear)
 *   [13:11] MAX_ANISO
 */

#define R200_PP_TXFORMAT_0              0x2C04
/*   PP_TXFORMAT bits:
 *   [4:0]   TXFORMAT
 *     0x00 = I8
 *     0x01 = AI88
 *     0x02 = RGB332
 *     0x03 = reserved
 *     0x04 = ARGB1555
 *     0x05 = RGB565
 *     0x06 = ARGB8888
 *     0x07 = RGBA8888
 *     0x08 = reserved
 *     0x09 = Y8
 *     0x0A = AVYU4444
 *     0x0B = VYUY422
 *     0x0C = DXT1
 *     0x0D = DXT23
 *     0x0E = DXT45
 *     0x0F = APAL8888 (palettized)
 *     0x11 = ARGB4444
 *     0x15 = XRGB8888 (no alpha)
 *   [5]     APPLE_YUV (Apple extension)
 *   [7:6]   reserved
 *   [8]     ALPHA_IN_MAP
 *   [9]     NON_POWER2
 *   [11:10] reserved
 *   [15:12] CUBE_MAP_FACE_*
 *   [19:16] TXFORMAT_X (extended format, on some R200 variants)
 *   [20]    ENDIAN_SWAP (byte swap for BE hosts)
 *   [21]    SQ_FORMAT (R200 specific)
 *   [25:22] NUM_LEVELS (mipmap count)
 */

#define R200_PP_TXFORMAT_X_0            0x2C08
#define R200_PP_TXSIZE_0                0x2C0C
/*   PP_TXSIZE bits:
 *   [10:0]  WIDTH  (actual width - 1)
 *   [15:11] reserved
 *   [26:16] HEIGHT (actual height - 1)
 */

#define R200_PP_TXPITCH_0               0x2C10
/*   PP_TXPITCH bits:
 *   [13:0]  TXPITCH (pitch in bytes - 32, must be aligned)
 */

#define R200_PP_TXOFFSET_0              0x2C14
/*   PP_TXOFFSET bits:
 *   [31:5]  TXOFFSET (byte offset in VRAM/GART, 32-byte aligned)
 */

#define R200_PP_CUBIC_OFFSET_T0_0       0x2C18
#define R200_PP_CUBIC_OFFSET_T0_1       0x2C1C
#define R200_PP_CUBIC_OFFSET_T0_2       0x2C20
#define R200_PP_CUBIC_OFFSET_T0_3       0x2C24
#define R200_PP_CUBIC_OFFSET_T0_4       0x2C28

/* Texture border color (per-unit) */
#define R200_PP_BORDER_COLOR_0          0x2C2C

/* ========================================================================
 * Texture Unit 1
 * ======================================================================== */

#define R200_PP_TXFILTER_1              0x2C40
#define R200_PP_TXFORMAT_1              0x2C44
#define R200_PP_TXFORMAT_X_1            0x2C48
#define R200_PP_TXSIZE_1                0x2C4C
#define R200_PP_TXPITCH_1               0x2C50
#define R200_PP_TXOFFSET_1              0x2C54

/* ========================================================================
 * Texture Unit 2
 * ======================================================================== */

#define R200_PP_TXFILTER_2              0x2C80
#define R200_PP_TXFORMAT_2              0x2C84
#define R200_PP_TXFORMAT_X_2            0x2C88
#define R200_PP_TXSIZE_2                0x2C8C
#define R200_PP_TXPITCH_2               0x2C90
#define R200_PP_TXOFFSET_2              0x2C94

/* ========================================================================
 * Pixel Shader / Texture Combiners (R200-specific)
 *
 * R200 uses a fixed-function combiner pipeline, not programmable shaders.
 * Each texture unit has color and alpha blend instructions.
 * ======================================================================== */

/* Texture combiner unit 0 */
#define R200_PP_TXCBLEND_0              0x2F00
#define R200_PP_TXCBLEND2_0             0x2F04
#define R200_PP_TXABLEND_0              0x2F08
#define R200_PP_TXABLEND2_0             0x2F0C

/* Texture combiner unit 1 */
#define R200_PP_TXCBLEND_1              0x2F10
#define R200_PP_TXCBLEND2_1             0x2F14
#define R200_PP_TXABLEND_1              0x2F18
#define R200_PP_TXABLEND2_1             0x2F1C

/* Texture combiner unit 2 */
#define R200_PP_TXCBLEND_2              0x2F20
#define R200_PP_TXCBLEND2_2             0x2F24
#define R200_PP_TXABLEND_2              0x2F28
#define R200_PP_TXABLEND2_2             0x2F2C

/* Output combiner */
#define R200_PP_TF_CNTL_0               0x2D00
#define R200_PP_TF_CNTL_1               0x2D04
#define R200_PP_TF_CNTL_2               0x2D08

/* ========================================================================
 * TCL (Transform, Clip, Lighting) Engine
 *
 * R200 has hardware T&L but QE may bypass it (VTX_XY_FMT=1 in VTE_CNTL
 * means pre-transformed vertices in screen space).
 * ======================================================================== */

#define R200_SE_TCL_MATERIAL_AMBIENT_R  0x2220
#define R200_SE_TCL_MATERIAL_AMBIENT_G  0x2224
#define R200_SE_TCL_MATERIAL_AMBIENT_B  0x2228
#define R200_SE_TCL_MATERIAL_AMBIENT_A  0x222C

#define R200_SE_TCL_LIGHT_MODEL_CTL_0   0x2268
#define R200_SE_TCL_LIGHT_MODEL_CTL_1   0x226C
#define R200_SE_TCL_PER_LIGHT_CTL_0     0x2270
#define R200_SE_TCL_PER_LIGHT_CTL_1     0x2274
#define R200_SE_TCL_PER_LIGHT_CTL_2     0x2278
#define R200_SE_TCL_PER_LIGHT_CTL_3     0x227C

#define R200_SE_TCL_TEX_PROC_CTL_0      0x2284
#define R200_SE_TCL_TEX_PROC_CTL_1      0x2288
#define R200_SE_TCL_TEX_PROC_CTL_2      0x2298

/* TCL input/output route */
#define R200_SE_TCL_INPUT_VTX_VECTOR_ADDR_0  0x2254

/* ========================================================================
 * Fog
 * ======================================================================== */

#define R200_RE_FOG_COEFF               0x1C30
#define R200_RE_FOG_TABLE_BASE          0x1C00

/* ========================================================================
 * Point / Line
 * ======================================================================== */

#define R200_RE_POINTSIZE               0x2648
#define R200_RE_LINE_CNT                0x2234
#define R200_SE_LINE_WIDTH              0x1DB8

/* ========================================================================
 * PM4 Type 3 3D Command Opcodes
 * ======================================================================== */

#define R200_3D_DRAW_VBUF               0x06
#define R200_3D_DRAW_IMMD               0x07
#define R200_3D_DRAW_INDX               0x08
#define R200_3D_LOAD_VBPNTR             0x21
#define R200_3D_DRAW_VBUF_2             0x23  /* R200: draw from VB (variant 2) */
#define R200_3D_DRAW_IMMD_2             0x24  /* R200: draw inline (variant 2) */
#define R200_3D_DRAW_INDX_2             0x25  /* R200: draw indexed (variant 2) */
#define R200_WAIT_FOR_IDLE              0x26
#define R200_3D_CLEAR_CMASK             0x28  /* color mask clear */
#define R200_3D_CLEAR_ZMASK             0x29  /* Z mask clear */
#define R200_3D_DRAW_VBUF_2_ALT        0x34  /* R200: draw from VB (alt variant) */

/* Draw command primitive types (in bits [3:0] of draw body) */
#define R200_PRIM_POINTS                0x0
#define R200_PRIM_LINES                 0x1
#define R200_PRIM_LINE_STRIP            0x2
#define R200_PRIM_TRIANGLES             0x3
#define R200_PRIM_TRI_FAN               0x4
#define R200_PRIM_TRI_STRIP             0x5
#define R200_PRIM_TRI_TYPE_2            0x6
#define R200_PRIM_TRI_TYPE_2_ALT        0x7  /* alternate tri type */
#define R200_PRIM_RECT_LIST             0x8  /* rectangle list (2D accel) */
#define R200_PRIM_3VRT_POINTS           0x9
#define R200_PRIM_3VRT_LINES            0xA
#define R200_PRIM_POINT_SPRITES         0xB
#define R200_PRIM_LINE_LOOP             0xC
#define R200_PRIM_QUAD_LIST             0xD
#define R200_PRIM_QUAD_STRIP            0xE

/* VF_CNTL (Vertex Fetch Control) body dword bits — used in DRAW_VBUF_2 etc. */
#define R200_VF_PRIM_TYPE_MASK          0x0000000F
#define R200_VF_PRIM_WALK_SHIFT         4
#define R200_VF_PRIM_WALK_MASK          0x00000030
#define R200_VF_PRIM_WALK_STATE         (0 << 4)
#define R200_VF_PRIM_WALK_INDEX         (1 << 4)
#define R200_VF_PRIM_WALK_LIST          (2 << 4)
#define R200_VF_PRIM_WALK_RING          (3 << 4)
#define R200_VF_COLOR_ORDER_RGBA        (1 << 6)
#define R200_VF_TCL_OUTPUT_ENA          (1 << 7)
#define R200_VF_PROG_STREAM_ENA         (1 << 8)
#define R200_VF_INDEX_SZ_4              (1 << 11)
#define R200_VF_EN_MAOS                 (1 << 15)
#define R200_VF_NUM_VERTICES_SHIFT      16
#define R200_VF_NUM_VERTICES_MASK       0xFFFF0000

/* ========================================================================
 * Vertex Fetch / AOS (Array of Structures) Registers
 *
 * These are written directly by the kext instead of using LOAD_VBPNTR.
 * The addresses at 0x20C0-0x20D8 store AOS descriptors and base addresses
 * for up to 6 vertex arrays.
 * ======================================================================== */

#define R200_SE_VF_CNTL                 0x2084  /* Vertex fetch control */
#define R200_SE_VF_CNTL_ALT             0x20C0  /* Alt VF_CNTL (used by Apple kext) */

/* AOS register block: descriptor + address pairs
 * Descriptor format: stride0[7:0] | size0[15:8] | stride1[23:16] | size1[31:24]
 * stride/size are in dwords
 */
#define R200_AOS_DESC_0                 0x20C4  /* AOS descriptor for arrays 0,1 */
#define R200_AOS_ADDR_0                 0x20C8  /* AOS base address array 0 */
#define R200_AOS_ADDR_1                 0x20CC  /* AOS base address array 1 */
#define R200_AOS_DESC_1                 0x20D0  /* AOS descriptor for arrays 2,3 */
#define R200_AOS_ADDR_2                 0x20D4  /* AOS base address array 2 */
#define R200_AOS_ADDR_3                 0x20D8  /* AOS base address array 3 */

/* 3D_DRAW_VBUF body format:
 *   dword 0: bits[3:0]=prim_type, bits[15:4]=num_vertices,
 *            bit[14]=PRIM_WALK (0=vertex_data, 1=vertex_list)
 * For DRAW_IMMD:
 *   dword 0: bits[3:0]=prim_type, bits[15:4]=num_vertices
 *   dwords 1..N: inline vertex data
 */

/* 3D_LOAD_VBPNTR body format:
 *   dword 0: number of vertex buffer arrays (1-based count)
 *   For each array:
 *     dword: bits[7:0]=size (bytes per vertex), bits[31:8]=reserved
 *     dword: GPU address of vertex data
 */

/* ========================================================================
 * Texture Format Values (for PP_TXFORMAT bits[4:0])
 * ======================================================================== */

#define R200_TXFORMAT_I8                0x00
#define R200_TXFORMAT_AI88              0x01
#define R200_TXFORMAT_RGB332            0x02
#define R200_TXFORMAT_ARGB1555          0x04
#define R200_TXFORMAT_RGB565            0x05
#define R200_TXFORMAT_ARGB8888          0x06
#define R200_TXFORMAT_RGBA8888          0x07
#define R200_TXFORMAT_Y8                0x09
#define R200_TXFORMAT_VYUY422           0x0B
#define R200_TXFORMAT_DXT1              0x0C
#define R200_TXFORMAT_DXT23             0x0D
#define R200_TXFORMAT_DXT45             0x0E
#define R200_TXFORMAT_ARGB4444          0x11
#define R200_TXFORMAT_XRGB8888          0x15

/* ========================================================================
 * Blend Factor Values (for RB3D_BLENDCNTL at 0x3220)
 *
 * R200 uses 0x20-based blend factor encoding in hardware.
 * SRC blend factor is at bits [21:16], DST blend factor at bits [29:24].
 * COMB_FCN (blend equation) is at bits [14:12].
 *
 * Reference: Mesa r200_reg.h — R200_SRC_BLEND_SHIFT=16, R200_DST_BLEND_SHIFT=24
 * ======================================================================== */

/* Blend factor shift positions in RB3D_BLENDCNTL */
#define R200_SRC_BLEND_SHIFT            16
#define R200_DST_BLEND_SHIFT            24
#define R200_BLEND_FACTOR_MASK          0x3F

/* R200 blend factor values (0x20-based hardware encoding) */
#define R200_BLEND_GL_ZERO                  0x20
#define R200_BLEND_GL_ONE                   0x21
#define R200_BLEND_GL_SRC_COLOR             0x22
#define R200_BLEND_GL_ONE_MINUS_SRC_COLOR   0x23
#define R200_BLEND_GL_DST_COLOR             0x24
#define R200_BLEND_GL_ONE_MINUS_DST_COLOR   0x25
#define R200_BLEND_GL_SRC_ALPHA             0x26
#define R200_BLEND_GL_ONE_MINUS_SRC_ALPHA   0x27
#define R200_BLEND_GL_DST_ALPHA             0x28
#define R200_BLEND_GL_ONE_MINUS_DST_ALPHA   0x29
#define R200_BLEND_GL_SRC_ALPHA_SATURATE    0x2A
#define R200_BLEND_GL_CONST_COLOR           0x2B
#define R200_BLEND_GL_ONE_MINUS_CONST_COLOR 0x2C
#define R200_BLEND_GL_CONST_ALPHA           0x2D
#define R200_BLEND_GL_ONE_MINUS_CONST_ALPHA 0x2E

/* Blend equation (COMB_FCN) at bits [14:12] */
#define R200_COMB_FCN_SHIFT             12
#define R200_COMB_FCN_MASK              (7 << 12)
#define R200_COMB_FCN_ADD_CLAMP         (0 << 12)
#define R200_COMB_FCN_ADD_NOCLAMP       (1 << 12)
#define R200_COMB_FCN_SUB_CLAMP         (2 << 12)
#define R200_COMB_FCN_SUB_NOCLAMP       (3 << 12)

/* Legacy 0-based blend factor aliases (for code compatibility) */
#define R200_BLEND_ZERO                 R200_BLEND_GL_ZERO
#define R200_BLEND_ONE                  R200_BLEND_GL_ONE
#define R200_BLEND_SRC_COLOR            R200_BLEND_GL_SRC_COLOR
#define R200_BLEND_ONE_MINUS_SRC_COLOR  R200_BLEND_GL_ONE_MINUS_SRC_COLOR
#define R200_BLEND_SRC_ALPHA            R200_BLEND_GL_SRC_ALPHA
#define R200_BLEND_ONE_MINUS_SRC_ALPHA  R200_BLEND_GL_ONE_MINUS_SRC_ALPHA
#define R200_BLEND_DST_ALPHA            R200_BLEND_GL_DST_ALPHA
#define R200_BLEND_ONE_MINUS_DST_ALPHA  R200_BLEND_GL_ONE_MINUS_DST_ALPHA
#define R200_BLEND_DST_COLOR            R200_BLEND_GL_DST_COLOR
#define R200_BLEND_ONE_MINUS_DST_COLOR  R200_BLEND_GL_ONE_MINUS_DST_COLOR
#define R200_BLEND_SRC_ALPHA_SAT        R200_BLEND_GL_SRC_ALPHA_SATURATE

/* ========================================================================
 * R100 Texture Registers (Apple kext uses these instead of R200 bank)
 *
 * The ATI kext writes texture state to R100-compatible register addresses
 * (0x1C54, 0x1C58, 0x1C5C) rather than the R200 bank (0x2C00-0x2C14).
 * These are in the 3D shadow range (>= 0x1C00) and stored in regs_3d[].
 * ======================================================================== */

#define R100_PP_TXFILTER_0              0x1C54
#define R100_PP_TXFORMAT_0              0x1C58
/*   R100 PP_TXFORMAT bits:
 *   [4:0]   TXFORMAT (same values as R200: 6=ARGB8888, etc.)
 *   [5]     APPLE_YUV
 *   [6]     NON_POWER2
 *   [7]     ALPHA_IN_MAP
 *   [11:8]  WIDTH_LOG2  (power-of-2 textures only)
 *   [15:12] HEIGHT_LOG2 (power-of-2 textures only)
 */
#define R100_PP_TXOFFSET_0              0x1C5C

/* R100 texture size / pitch (separate registers, used for NON_POWER2) */
#define R100_PP_TEX_SIZE_0              0x1D04
/*   PP_TEX_SIZE bits:
 *   [10:0]  USIZE (width - 1)
 *   [26:16] VSIZE (height - 1)
 */
#define R100_PP_TEX_PITCH_0             0x1D08
/*   PP_TEX_PITCH bits:
 *   [13:0]  TXPITCH (pitch in bytes - 32)
 */

#endif /* HW_DISPLAY_PPC_MAC_GPU_3D_REGS_H */
