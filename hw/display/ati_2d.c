/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI SVGA emulation
 * 2D engine functions
 *
 * Copyright (c) 2019 BALATON Zoltan
 * Copyright (c) 2026 RetroBoot contributors (general ROP3 engine, brushes,
 * colour compare, write mask, mono sources, CP 2D packets)
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "ati_regs.h"
#include "qemu/log.h"
#include "qemu/bswap.h"
#include "ui/pixel_ops.h"
#include "ui/console.h"
#include "ui/rect.h"
#include "trace.h"

/*
 * NOTE:
 * This is 2D _acceleration_ and supposed to be fast. Common operations
 * (copies and solid fills) go to pixman; the general per-pixel ROP3 path
 * handles everything else the guest drivers use.
 *
 * Pixels are processed in VRAM byte order (the CPU's view of the frame
 * buffer); colours from registers are converted to that order once.
 */

#define BRUSH_DATA0      0x1480

enum { SRC_NONE, SRC_MEM_COLOR, SRC_MEM_MONO, SRC_BUF_COLOR, SRC_BUF_MONO };

typedef struct ATI2DCtx {
    int bpp, bypp;
    uint32_t rop3;
    bool left_to_right, top_to_bottom;
    bool need_swap;
    uint8_t *vram;
    const uint8_t *vram_end;
    uint32_t vram_size;
    QemuRect scissor;

    QemuRect dst;
    int dst_stride;
    uint32_t dst_offset;

    /* source */
    int src_kind;
    int sx, sy;
    int src_stride;
    uint32_t src_offset;
    const uint8_t *src_buf;     /* host data block (row-major) */
    bool lsb_first;
    uint32_t sfg, sbg;
    bool src_la;                /* mono: leave background alone */

    /* brush */
    int brush;
    uint32_t bfg, bbg;
    uint8_t bmono[8];
    uint32_t bcolor[64];        /* colour brushes, VRAM byte order */
    uint32_t solid;

    /* write mask and colour compare */
    bool wmask_en;
    uint32_t wmask;
    bool cmp_en;
    int cmp_fcn;
    bool cmp_src;
    uint32_t cmp_clr, cmp_mask;
} ATI2DCtx;

static int ati_bpp_from_datatype(const ATIVGAState *s)
{
    switch (s->regs.dp_datatype & 0xf) {
    case 2:
    case 7:
    case 8:
        return 8;
    case 3:
    case 4:
    case 15:
        return 16;
    case 5:
        return 24;
    case 6:
        return 32;
    default:
        qemu_log_mask(LOG_UNIMP, "Unknown dst datatype %d\n",
                      s->regs.dp_datatype & 0xf);
        return 0;
    }
}

/* register colour (pixel value) -> VRAM byte order word for this depth */
static uint32_t vram_color(const ATI2DCtx *c, uint32_t v)
{
    switch (c->bpp) {
    case 8:
        return v & 0xff;
    case 16:
        v &= 0xffff;
        return c->need_swap ? bswap16(v) : v;
    case 24:
        v &= 0xffffff;
        return v;
    default:
        return c->need_swap ? bswap32(v) : v;
    }
}

static inline uint32_t ld_px(const uint8_t *p, int bypp)
{
    switch (bypp) {
    case 1: return *p;
    case 2: return lduw_he_p(p);
    case 3: return p[0] | (p[1] << 8) | (p[2] << 16);
    default: return ldl_he_p(p);
    }
}

static inline void st_px(uint8_t *p, int bypp, uint32_t v)
{
    switch (bypp) {
    case 1: *p = v; break;
    case 2: stw_he_p(p, v); break;
    case 3: p[0] = v; p[1] = v >> 8; p[2] = v >> 16; break;
    default: stl_he_p(p, v); break;
    }
}

static inline uint32_t rop3(uint8_t rop, uint32_t p, uint32_t s, uint32_t d)
{
    uint32_t r = 0;

    switch (rop) {
    case 0x00: return 0;
    case 0xff: return ~0u;
    case 0xcc: return s;
    case 0xf0: return p;
    case 0xaa: return d;
    case 0x55: return ~d;
    case 0x66: return s ^ d;
    case 0x5a: return p ^ d;
    case 0x88: return s & d;
    case 0xee: return s | d;
    case 0x33: return ~s;
    case 0x22: return ~s & d;
    case 0xbb: return ~s | d;
    case 0x44: return s & ~d;
    case 0x99: return ~(s ^ d);
    case 0x11: return ~(s | d);
    case 0xa0: return p & d;
    case 0xfa: return p | d;
    case 0xb8: return (p & s) | (~s & d);   /* PSDPxax */
    case 0xe2: return (d & ~s) | (p & s);
    default:
        break;
    }
    for (int i = 0; i < 8; i++) {
        if (rop & (1 << i)) {
            r |= ((i & 4) ? p : ~p) & ((i & 2) ? s : ~s) & ((i & 1) ? d : ~d);
        }
    }
    return r;
}

static inline bool rop_uses_src(uint8_t rop)
{
    return ((rop >> 2) ^ rop) & 0x33;
}

static inline bool rop_uses_pat(uint8_t rop)
{
    return ((rop >> 4) ^ rop) & 0x0f;
}

static void ati_set_dirty(VGACommonState *vga, const ATI2DCtx *ctx,
                          const QemuRect *r)
{
    hwaddr start = ctx->dst_offset + r->x * ctx->bypp + r->y * ctx->dst_stride;
    hwaddr end = start + r->width * ctx->bypp +
                 (r->height - 1) * ctx->dst_stride;

    if (r->width && r->height && end <= vga->vram_size) {
        memory_region_set_dirty(&vga->vram, start, end - start);
    }
}

/*
 * Radeon: pitch/offset registers hold card (MC) addresses; the frame buffer
 * sits at MC_FB_LOCATION.  Offsets below the VRAM size are taken as plain
 * VRAM offsets (Rage 128, and drivers that keep MC_FB_LOCATION at 0).
 */
static uint32_t ati_2d_vram_offset(const ATIVGAState *s, uint32_t mc)
{
    uint32_t loc = s->regfile[MC_FB_LOCATION / 4];
    uint32_t start = (loc & 0xffff) << 16, end = (loc & 0xffff0000) | 0xffff;

    if (ati_is_radeon(s) && end > start && mc >= start && mc <= end) {
        return mc - start;
    }
    return mc;
}

static void setup_2d_blt_ctx(const ATIVGAState *s, ATI2DCtx *ctx)
{
    uint32_t gmc = s->regs.dp_gui_master_cntl;
    uint32_t cmp = s->regfile[CLR_CMP_CNTL / 4];

    memset(ctx, 0, sizeof(*ctx));
    ctx->bpp = ati_bpp_from_datatype(s);
    ctx->bypp = ctx->bpp / 8;
    ctx->rop3 = s->regs.dp_mix & GMC_ROP3_MASK;
    ctx->left_to_right = s->regs.dp_cntl & DST_X_LEFT_TO_RIGHT;
    ctx->top_to_bottom = s->regs.dp_cntl & DST_Y_TOP_TO_BOTTOM;
    ctx->need_swap = (HOST_BIG_ENDIAN != s->vga.big_endian_fb);
    ctx->vram = s->vga.vram_ptr;
    ctx->vram_size = s->vga.vram_size;
    ctx->vram_end = s->vga.vram_ptr + s->vga.vram_size;

    if (ati_is_radeon(s)) {
        /* RetroBoot: Radeon scissor right/bottom are exclusive (the Mac OS
         * accelerator clips dword-aligned blits with them; X's radeon
         * driver also adds one to inclusive coordinates) */
        ctx->scissor.width = MAX((int)s->regs.sc_right - (int)s->regs.sc_left, 0);
        ctx->scissor.height = MAX((int)s->regs.sc_bottom - (int)s->regs.sc_top, 0);
    } else {
        ctx->scissor.width = s->regs.sc_right - s->regs.sc_left + 1;
        ctx->scissor.height = s->regs.sc_bottom - s->regs.sc_top + 1;
    }
    ctx->scissor.x = s->regs.sc_left;
    ctx->scissor.y = s->regs.sc_top;

    ctx->dst.width = s->regs.dst_width;
    ctx->dst.height = s->regs.dst_height;
    ctx->dst.x = (ctx->left_to_right ?
                 s->regs.dst_x : s->regs.dst_x + 1 - ctx->dst.width);
    ctx->dst.y = (ctx->top_to_bottom ?
                 s->regs.dst_y : s->regs.dst_y + 1 - ctx->dst.height);
    ctx->dst_stride = s->regs.dst_pitch;
    ctx->dst_offset = ati_2d_vram_offset(s, s->regs.dst_offset);
    if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
        ctx->dst_stride *= ctx->bpp;
    }

    ctx->sx = (ctx->left_to_right ?
               s->regs.src_x : s->regs.src_x + 1 - ctx->dst.width);
    ctx->sy = (ctx->top_to_bottom ?
               s->regs.src_y : s->regs.src_y + 1 - ctx->dst.height);
    ctx->src_stride = s->regs.src_pitch;
    ctx->src_offset = ati_2d_vram_offset(s, s->regs.src_offset);
    if (s->dev_id == PCI_DEVICE_ID_ATI_RAGE128_PF) {
        ctx->src_stride *= ctx->bpp;
    }
    ctx->lsb_first = s->regs.dp_datatype & DP_BYTE_PIX_ORDER;
    ctx->sfg = vram_color(ctx, s->regs.dp_src_frgd_clr);
    ctx->sbg = vram_color(ctx, s->regs.dp_src_bkgd_clr);

    ctx->brush = (s->regs.dp_datatype >> 8) & 0xf;
    ctx->bfg = vram_color(ctx, s->regs.dp_brush_frgd_clr);
    ctx->bbg = vram_color(ctx, s->regs.dp_brush_bkgd_clr);
    for (int i = 0; i < 2; i++) {
        uint32_t w = s->regfile[(BRUSH_DATA0 + 4 * i) / 4];
        for (int b = 0; b < 4; b++) {
            ctx->bmono[i * 4 + b] = w >> (8 * b);
        }
    }
    if (ctx->brush >= 10 && ctx->brush <= 12 && ctx->bypp && ctx->bypp != 3) {
        /* colour brushes: BRUSH_DATA holds the pixels as a little-endian
         * array (8x8, 8x1 or 1x8) */
        unsigned np = ctx->brush == 10 ? 64 : 8;
        for (unsigned j = 0; j < np; j++) {
            uint32_t v = 0;
            for (int b = 0; b < ctx->bypp; b++) {
                unsigned k = j * ctx->bypp + b;
                uint32_t w = s->regfile[(BRUSH_DATA0 + (k & ~3u)) / 4];
                v |= ((w >> (8 * (k & 3))) & 0xff) << (8 * b);
            }
            ctx->bcolor[j] = vram_color(ctx, v);
        }
    }

    ctx->wmask_en = !(gmc & BIT(30)) && s->regs.dp_write_mask != 0xffffffff;
    ctx->wmask = vram_color(ctx, s->regs.dp_write_mask);
    ctx->cmp_en = ati_is_radeon(s) && !(gmc & BIT(28)) && (cmp & 0x707);
    ctx->cmp_src = cmp & BIT(24);
    ctx->cmp_fcn = ctx->cmp_src ? (cmp & 7) : ((cmp >> 8) & 7);
    ctx->cmp_clr = vram_color(ctx, s->regfile[(ctx->cmp_src ? CLR_CMP_CLR_SRC
                                               : CLR_CMP_CLR_DST) / 4]);
    ctx->cmp_mask = vram_color(ctx, s->regfile[CLR_CMP_MASK / 4]);
}

static inline uint32_t brush_px(const ATI2DCtx *c, int x, int y, bool *la)
{
    *la = false;
    switch (c->brush) {
    case 0:     /* 8x8 mono, fg/bg */
    case 1:     /* 8x8 mono, fg/leave alone */
    {
        uint8_t row = c->bmono[y & 7];
        bool on = c->lsb_first ? (row >> (x & 7)) & 1 : (row >> (7 - (x & 7))) & 1;
        if (!on && c->brush == 1) {
            *la = true;
        }
        return on ? c->bfg : c->bbg;
    }
    case 10:    /* 8x8 colour */
        return c->bcolor[(y & 7) * 8 + (x & 7)];
    case 11:    /* 8x1 colour */
        return c->bcolor[x & 7];
    case 12:    /* 1x8 colour */
        return c->bcolor[y & 7];
    case 15:    /* none */
        return 0;
    default:    /* solid colour and anything not decoded */
        return c->bfg;
    }
}

static inline bool mono_bit(const ATI2DCtx *c, const uint8_t *row, int x)
{
    uint8_t b = row[x >> 3];
    return c->lsb_first ? (b >> (x & 7)) & 1 : (b >> (7 - (x & 7))) & 1;
}

/*
 * General per-pixel blit over the (already clipped) rectangle vis_dst,
 * with sx/sy the source position of its top-left pixel.
 */
static void ati_2d_generic(ATI2DCtx *c, const QemuRect *vis, int sx, int sy,
                           const uint8_t *src_rows, int src_row_stride)
{
    uint8_t rop = c->rop3 >> 16;
    bool use_s = rop_uses_src(rop), use_p = rop_uses_pat(rop);
    int bypp = c->bypp;
    uint8_t *tmp = NULL;

    /* overlapping memory-to-memory copies: snapshot the source first */
    if (c->src_kind == SRC_MEM_COLOR && use_s) {
        size_t len = (size_t)vis->width * bypp * vis->height;
        tmp = g_malloc(len);
        for (int y = 0; y < vis->height; y++) {
            memcpy(tmp + (size_t)y * vis->width * bypp,
                   c->vram + c->src_offset + (sy + y) * c->src_stride + sx * bypp,
                   vis->width * bypp);
        }
    }
    for (int y = 0; y < vis->height; y++) {
        uint8_t *d = c->vram + c->dst_offset + (vis->y + y) * c->dst_stride +
                     vis->x * bypp;
        for (int x = 0; x < vis->width; x++, d += bypp) {
            uint32_t S = 0, P = 0, D = ld_px(d, bypp), res;
            bool skip = false;

            if (use_s) {
                switch (c->src_kind) {
                case SRC_MEM_COLOR:
                    S = ld_px(tmp + ((size_t)y * vis->width + x) * bypp, bypp);
                    break;
                case SRC_MEM_MONO:
                {
                    const uint8_t *row = c->vram + c->src_offset +
                                         (sy + y) * c->src_stride;
                    bool on = mono_bit(c, row, sx + x);
                    S = on ? c->sfg : c->sbg;
                    skip = !on && c->src_la;
                    break;
                }
                case SRC_BUF_COLOR:
                    S = ld_px(src_rows + (size_t)y * src_row_stride +
                              (sx + x) * bypp, bypp);
                    break;
                case SRC_BUF_MONO:
                {
                    bool on = mono_bit(c, src_rows + (size_t)y * src_row_stride,
                                       sx + x);
                    S = on ? c->sfg : c->sbg;
                    skip = !on && c->src_la;
                    break;
                }
                default:
                    break;
                }
            }
            if (skip) {
                continue;
            }
            if (use_p) {
                bool la;
                P = brush_px(c, vis->x + x, vis->y + y, &la);
                if (la) {
                    continue;
                }
            }
            if (c->cmp_en) {
                uint32_t v = (c->cmp_src ? S : D) & c->cmp_mask;
                uint32_t ref = c->cmp_clr & c->cmp_mask;
                bool hit;
                switch (c->cmp_fcn) {
                case 0: hit = false; break;
                case 1: hit = true; break;
                case 4: hit = v == ref; break;
                case 5: hit = v != ref; break;
                default: hit = false; break;
                }
                if (hit) {
                    continue;       /* comparison true: pixel not written */
                }
            }
            res = rop3(rop, P, S, D);
            if (c->wmask_en) {
                res = (res & c->wmask) | (D & ~c->wmask);
            }
            st_px(d, bypp, res);
        }
    }
    g_free(tmp);
}

static bool ati_2d_fast(ATI2DCtx *c, const QemuRect *vis, int sx, int sy,
                        uint8_t use_pixman)
{
    uint8_t rop = c->rop3 >> 16;

    if (c->cmp_en || c->wmask_en || c->bpp == 24) {
        return false;
    }
    if (rop == 0xcc && c->src_kind == SRC_MEM_COLOR) {
        uint8_t *src_bits = c->vram + c->src_offset;
        uint8_t *dst_bits = c->vram + c->dst_offset;
        bool same = c->src_offset == c->dst_offset &&
                    c->src_stride == c->dst_stride;
        QemuRect srect = { sx, sy, vis->width, vis->height }, isect;
        bool overlap = same && qemu_rect_intersect(&srect, vis, &isect);
#ifdef CONFIG_PIXMAN
        if ((use_pixman & BIT(1)) && !overlap && !(c->src_stride & 3) &&
            !(c->dst_stride & 3) && !(c->src_offset & 3) &&
            !(c->dst_offset & 3) &&
            pixman_blt((uint32_t *)src_bits, (uint32_t *)dst_bits,
                       c->src_stride / 4, c->dst_stride / 4, c->bpp, c->bpp,
                       sx, sy, vis->x, vis->y, vis->width, vis->height)) {
            return true;
        }
#endif
        /* rows in the order that is safe for overlapping copies */
        for (int i = 0; i < vis->height; i++) {
            int y = (overlap && sy < vis->y) ? vis->height - 1 - i : i;
            memmove(dst_bits + (vis->y + y) * c->dst_stride + vis->x * c->bypp,
                    src_bits + (sy + y) * c->src_stride + sx * c->bypp,
                    vis->width * c->bypp);
        }
        return true;
    }
    if (rop == 0xf0 && (c->brush == 0 || (c->brush >= 10 && c->brush <= 12)) &&
        c->bpp != 24) {
        /* 8x8 pattern (mono fg/bg or colour): build each of the (up to) 8
         * distinct rows once, then copy whole rows */
        uint8_t *dst = c->vram + c->dst_offset;
        size_t rowlen = (size_t)vis->width * c->bypp;
        uint8_t *rows = g_malloc(rowlen * MIN(vis->height, 8));
        for (int y = 0; y < MIN(vis->height, 8); y++) {
            int py = vis->y + y;
            uint8_t *r = rows + y * rowlen;
            for (int k = 0; k < 8 && k < vis->width; k++) {
                bool la;
                st_px(r + k * c->bypp, c->bypp,
                      brush_px(c, (vis->x + k) & 7, py, &la));
            }
            for (size_t n = 8 * c->bypp; n < rowlen; n *= 2) {
                memcpy(r + n, r, MIN(n, rowlen - n));
            }
        }
        for (int y = 0; y < vis->height; y++) {
            memcpy(dst + (vis->y + y) * c->dst_stride + vis->x * c->bypp,
                   rows + (y & 7) * rowlen, rowlen);
        }
        g_free(rows);
        return true;
    }
    if (rop == 0x55) {          /* DSTINVERT */
        uint8_t *dst = c->vram + c->dst_offset;
        for (int y = 0; y < vis->height; y++) {
            uint8_t *d = dst + (vis->y + y) * c->dst_stride + vis->x * c->bypp;
            size_t n = (size_t)vis->width * c->bypp, i = 0;
            for (; i + 8 <= n; i += 8) {
                stq_he_p(d + i, ~ldq_he_p(d + i));
            }
            for (; i < n; i++) {
                d[i] = ~d[i];
            }
        }
        return true;
    }
    if ((rop == 0xf0 && (c->brush == 13 || c->brush == 14)) ||
        rop == 0x00 || rop == 0xff) {
        uint32_t filler = rop == 0x00 ? 0 : rop == 0xff ? ~0u : c->bfg;
        uint8_t *dst = c->vram + c->dst_offset;
        if (c->bpp == 16) {
            filler = (filler & 0xffff) | (filler << 16);
        } else if (c->bpp == 8) {
            filler &= 0xff;
            filler |= filler << 8;
            filler |= filler << 16;
        }
#ifdef CONFIG_PIXMAN
        if ((use_pixman & BIT(0)) && !(c->dst_stride & 3) &&
            !(c->dst_offset & 3) &&
            pixman_fill((uint32_t *)dst, c->dst_stride / 4, c->bpp, vis->x,
                        vis->y, vis->width, vis->height, filler)) {
            return true;
        }
#endif
        for (int y = 0; y < vis->height; y++) {
            uint8_t *d = dst + (vis->y + y) * c->dst_stride + vis->x * c->bypp;
            for (int x = 0; x < vis->width; x++, d += c->bypp) {
                st_px(d, c->bypp, filler);
            }
        }
        return true;
    }
    return false;
}

/* clip, bounds-check and run one operation; returns false if nothing done */
static bool ati_2d_run(ATIVGAState *s, ATI2DCtx *c, const uint8_t *buf,
                       int buf_stride)
{
    QemuRect vis;
    int sx, sy;
    uint64_t dst_end;

    if (!c->bpp) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: 2D blit with invalid bpp\n");
        return false;
    }
    if (!c->dst_stride) {
        qemu_log_mask(LOG_GUEST_ERROR, "ati: 2D blit with zero dest pitch\n");
        return false;
    }
    if (!qemu_rect_intersect(&c->dst, &c->scissor, &vis)) {
        return false;
    }
    dst_end = (uint64_t)c->dst_offset + (uint64_t)(vis.y + vis.height) *
              c->dst_stride;
    if (vis.x + vis.width > c->dst_stride / c->bypp + 1 ||
        dst_end > c->vram_size) {
        qemu_log_mask(LOG_UNIMP, "ati: 2D blit outside VRAM\n");
        return false;
    }
    sx = c->sx + (vis.x - c->dst.x);
    sy = c->sy + (vis.y - c->dst.y);
    trace_ati_2d_op(c->bpp, c->rop3 >> 16, c->src_kind, vis.x, vis.y,
                    vis.width, vis.height, c->dst_offset, c->dst_stride);
    if (c->src_kind == SRC_MEM_COLOR || c->src_kind == SRC_MEM_MONO) {
        uint64_t send = (uint64_t)c->src_offset +
                        (uint64_t)(sy + vis.height) * c->src_stride;
        if (!c->src_stride || sx < 0 || sy < 0 || send > c->vram_size) {
            qemu_log_mask(LOG_UNIMP, "ati: 2D source outside VRAM\n");
            return false;
        }
    }
    if (!ati_2d_fast(c, &vis, sx, sy, s->use_pixman)) {
        ati_2d_generic(c, &vis, sx, sy, buf, buf_stride);
    }
    ati_set_dirty(&s->vga, c, &vis);
    if (ati_is_radeon(s)) {
        /* GL's copy of a 3D target here is stale now */
        ati_3d_vram_written(s, c->dst_offset + vis.x * c->bypp +
                               (uint64_t)vis.y * c->dst_stride,
                            (uint64_t)(vis.height - 1) * c->dst_stride +
                            vis.width * c->bypp);
    }
    s->accel.ops2d++;
    s->accel.pixels2d += (uint64_t)vis.width * vis.height;
    return true;
}

static unsigned host_row_bytes(ATIVGAState *s, const ATI2DCtx *c, bool mono);

/*
 * RetroBoot: a solid fill of an off-screen 3D render target is done by the
 * renderer as a GL clear (ati_3d_fill): Mac OS's ATI OpenGL clears the next
 * back buffer this way every frame.  Same operations and clipping as the
 * fill in ati_2d_fast, on the same 2D state; false = do it in VRAM.
 */
static bool ati_2d_gl_fill(ATIVGAState *s, const ATI2DCtx *c)
{
    uint8_t rop = c->rop3 >> 16;
    QemuRect vis;
    uint32_t filler;

    if (!((rop == 0xf0 && (c->brush == 13 || c->brush == 14)) ||
          rop == 0x00 || rop == 0xff) ||
        c->cmp_en || c->wmask_en || (c->bpp != 16 && c->bpp != 32) ||
        !c->dst_stride || (c->dst_stride & 3) || (c->dst_offset & 3) ||
        !qemu_rect_intersect(&c->dst, &c->scissor, &vis) ||
        vis.x + vis.width > c->dst_stride / c->bypp ||
        (uint64_t)c->dst_offset + (uint64_t)(vis.y + vis.height) *
        c->dst_stride > c->vram_size) {
        return false;
    }
    if (c->src_kind == SRC_MEM_COLOR || c->src_kind == SRC_MEM_MONO) {
        /* unused by these ROPs, but ati_2d_run refuses an operation whose
         * source lies outside VRAM: leave that case to it */
        int sx = c->sx + (vis.x - c->dst.x), sy = c->sy + (vis.y - c->dst.y);
        if (!c->src_stride || sx < 0 || sy < 0 ||
            (uint64_t)c->src_offset + (uint64_t)(sy + vis.height) *
            c->src_stride > c->vram_size) {
            return false;
        }
    }
    filler = rop == 0x00 ? 0 : rop == 0xff ? ~0u : c->bfg;
    if (c->bpp == 16) {
        filler = (filler & 0xffff) | (filler << 16);
    }
    if (!ati_3d_fill(s, c->dst_offset, c->dst_stride, c->bypp,
                     vis.x, vis.y, vis.width, vis.height, filler)) {
        return false;
    }
    trace_ati_2d_op(c->bpp, rop, c->src_kind, vis.x, vis.y, vis.width,
                    vis.height, c->dst_offset, c->dst_stride);
    s->accel.ops2d++;
    s->accel.pixels2d += (uint64_t)vis.width * vis.height;
    return true;
}

void ati_2d_blt(ATIVGAState *s)
{
    ATI2DCtx ctx;
    uint32_t src_source = s->regs.dp_mix & DP_SRC_SOURCE;
    uint32_t src_dt = (s->regs.dp_datatype >> 16) & 3;

    if (ati_is_radeon(s)) {
        if (!s->host_data.active && src_source != DP_SRC_HOST &&
            src_source != DP_SRC_HOST_BYTEALIGN) {
            setup_2d_blt_ctx(s, &ctx);
            if (src_source == DP_SRC_RECT) {
                ctx.src_kind = src_dt == 3 ? SRC_MEM_COLOR : SRC_MEM_MONO;
            }
            if (ati_2d_gl_fill(s, &ctx)) {
                return;
            }
        }
        ati_3d_sync(s, ATI_SYNC_2D);     /* 2D reads/writes see 3D results */
    }
    /* Finish any active HOST_DATA blits before starting a new blit */
    ati_host_data_finish(s);

    if (src_source == DP_SRC_HOST || src_source == DP_SRC_HOST_BYTEALIGN) {
        /* Begin a HOST_DATA blit */
        unsigned rowb;
        setup_2d_blt_ctx(s, &ctx);
        ctx.src_kind = src_dt == 3 ? SRC_BUF_COLOR : SRC_BUF_MONO;
        ctx.src_la = src_dt == 1;
        /* the engine latches the blit parameters when it starts */
        QEMU_BUILD_BUG_ON(sizeof(ATI2DCtx) > sizeof(s->host_data.ctx));
        memcpy(s->host_data.ctx, &ctx, sizeof(ctx));
        rowb = host_row_bytes(s, &ctx, src_dt != 3);
        s->host_data.total_bits = (uint64_t)ctx.dst.width * ctx.dst.height *
                                  (src_dt != 3 ? 1 : ctx.bpp);
        s->host_data.recv_bits = 0;
        s->host_data.active = true;
        if ((s->debug & 2) && src_dt == 3) {
            fprintf(stderr, "ati hostdata: (%d,%d) %dx%d bpp %d dt 0x%x mix 0x%x "
                    "gmc 0x%x sc (%d,%d)-(%d,%d) bits %" PRIu64 "\n",
                    ctx.dst.x, ctx.dst.y, ctx.dst.width, ctx.dst.height, ctx.bpp,
                    s->regs.dp_datatype, s->regs.dp_mix, s->regs.dp_gui_master_cntl,
                    ctx.scissor.x, ctx.scissor.y, ctx.scissor.width,
                    ctx.scissor.height, s->host_data.total_bits);
        }
        s->host_data.next = 0;
        s->host_data.col = 0;
        s->host_data.row = 0;
        s->host_data.nbytes = 0;
        s->host_data.bit0 = 0;
        /* bands of up to 32 rows, at least one row, within the buffer */
        s->host_data.flush_at = MAX(MIN(rowb * 32, (unsigned)sizeof(s->host_data.buf) - 64),
                                    MIN(rowb, (unsigned)sizeof(s->host_data.buf) - 64));
        if (!s->host_data.flush_at) {
            s->host_data.flush_at = 64;
        }
        return;
    }
    setup_2d_blt_ctx(s, &ctx);
    if (src_source == DP_SRC_RECT) {
        ctx.src_kind = src_dt == 3 ? SRC_MEM_COLOR : SRC_MEM_MONO;
        ctx.src_la = src_dt == 1;
    }
    ati_2d_run(s, &ctx, NULL, 0);
}

/*
 * HOST_DATA: the guest streams the source through HOST_DATA0..7 and
 * HOST_DATA_LAST (or a CNTL_HOSTDATA_BLT packet).  The stream is packed:
 * pixel i of the w x h rectangle starts at bit i * bits-per-pixel (1 for
 * monochrome sources), rows are not padded (Mac OS's ATI accelerator sends
 * exactly ceil(w * h * bpp / 32) dwords).  Bytes come in the order the
 * guest stored them; monochrome bits are MSB first unless BYTE_PIX_ORDER.
 * Whole rows are drawn as soon as they are complete.
 */
static unsigned host_row_bytes(ATIVGAState *s, const ATI2DCtx *c, bool mono)
{
    unsigned bits = c->dst.width * (mono ? 1 : c->bpp);

    return DIV_ROUND_UP(bits, 8);
}

bool ati_host_data_flush(ATIVGAState *s)
{
    ATI2DCtx ctx;
    bool mono;
    uint64_t row_bits, avail_bits;
    unsigned rows, rowb;
    uint8_t *band = NULL;

    if (!s->host_data.active) {
        return false;
    }
    s->host_data.next = 0;
    memcpy(&ctx, s->host_data.ctx, sizeof(ctx));
    mono = ctx.src_kind == SRC_BUF_MONO;
    if (!ctx.dst.width || !ctx.dst.height || (!mono && !ctx.bypp)) {
        s->host_data.active = false;
        return false;
    }
    row_bits = (uint64_t)ctx.dst.width * (mono ? 1 : ctx.bpp);
    rowb = host_row_bytes(s, &ctx, mono);
    avail_bits = (uint64_t)s->host_data.nbytes * 8 - s->host_data.bit0;
    rows = MIN(avail_bits / row_bits, ctx.dst.height - s->host_data.row);
    if (rows) {
        ATI2DCtx b = ctx;
        const uint8_t *src = s->host_data.buf;
        uint64_t used_bits = rows * row_bits;
        QemuRect vis;

        if (mono || s->host_data.bit0) {
            /* repack into byte-aligned rows */
            band = g_malloc0((size_t)rows * rowb);
            for (unsigned r = 0; r < rows; r++) {
                for (uint64_t k = 0; k < row_bits; k++) {
                    uint64_t bit = s->host_data.bit0 + r * row_bits + k;
                    uint8_t byte = s->host_data.buf[bit >> 3];
                    bool on = ctx.lsb_first ? (byte >> (bit & 7)) & 1
                                            : (byte >> (7 - (bit & 7))) & 1;
                    if (on) {
                        band[r * rowb + (k >> 3)] |= ctx.lsb_first ?
                            1 << (k & 7) : 0x80 >> (k & 7);
                    }
                }
            }
            src = band;
        }
        if ((s->debug & 2) && !mono) {
            fprintf(stderr, "ati hostdata band row %u rows %u:", s->host_data.row, rows);
            for (unsigned k = 0; k < MIN(rowb, 400u); k += 4) {
                fprintf(stderr, " %08x", ldl_be_p(src + k));
            }
            fprintf(stderr, "\n");
        }
        b.dst.y = ctx.dst.y + s->host_data.row;
        b.dst.height = rows;
        if (qemu_rect_intersect(&b.dst, &b.scissor, &vis)) {
            src += (size_t)(vis.y - b.dst.y) * rowb;
            b.sx = vis.x - b.dst.x;
            b.sy = 0;
            b.dst = vis;
            ati_2d_run(s, &b, src, rowb);
        }
        g_free(band);
        s->host_data.row += rows;
        used_bits += s->host_data.bit0;
        memmove(s->host_data.buf, s->host_data.buf + (used_bits >> 3),
                s->host_data.nbytes - (used_bits >> 3));
        s->host_data.nbytes -= used_bits >> 3;
        s->host_data.bit0 = used_bits & 7;
    }
    if (s->host_data.row >= ctx.dst.height) {
        s->host_data.active = false;
    }
    return s->host_data.active;
}

void ati_host_data_finish(ATIVGAState *s)
{
    if (ati_host_data_flush(s)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "HOST_DATA blit ended before all data was written\n");
    }
    s->host_data.active = false;
}

/* ---------------------------------------------------------------------
 * RetroBoot: 2D packets from the command processor (CNTL_*).  The fields
 * after GUI_CNTL depend on its bits, in this order: source pitch/offset,
 * destination pitch/offset, source clip, destination clip (2), brush
 * colours/pattern, source colours (mono sources); then the operation's own
 * data.  Each field is applied as the corresponding register write.
 */
static unsigned ati_2d_packet_header(ATIVGAState *s, const uint32_t *d,
                                     unsigned n)
{
    uint32_t gmc = d[0];
    unsigned i = 1, brush = (gmc >> 4) & 15;

    ati_reg_write(s, DP_GUI_MASTER_CNTL, gmc);
    if ((gmc & GMC_SRC_PITCH_OFFSET_CNTL) && i < n) {
        ati_reg_write(s, SRC_PITCH_OFFSET, d[i++]);
    }
    if ((gmc & GMC_DST_PITCH_OFFSET_CNTL) && i < n) {
        ati_reg_write(s, DST_PITCH_OFFSET, d[i++]);
    }
    if ((gmc & GMC_SRC_CLIPPING) && i < n) {
        ati_reg_write(s, SRC_SC_BOTTOM_RIGHT, d[i++]);
    }
    if ((gmc & GMC_DST_CLIPPING) && i + 1 < n) {
        ati_reg_write(s, SC_TOP_LEFT, d[i++]);
        ati_reg_write(s, SC_BOTTOM_RIGHT, d[i++]);
    }
    switch (brush) {
    case 0:     /* 8x8 mono fg/bg: bg, fg, 2 dwords of pattern */
        if (i + 3 < n) {
            ati_reg_write(s, DP_BRUSH_BKGD_CLR, d[i++]);
            ati_reg_write(s, DP_BRUSH_FRGD_CLR, d[i++]);
            ati_reg_write(s, BRUSH_DATA0, d[i++]);
            ati_reg_write(s, BRUSH_DATA0 + 4, d[i++]);
        }
        break;
    case 1:     /* 8x8 mono fg/leave alone */
        if (i + 2 < n) {
            ati_reg_write(s, DP_BRUSH_FRGD_CLR, d[i++]);
            ati_reg_write(s, BRUSH_DATA0, d[i++]);
            ati_reg_write(s, BRUSH_DATA0 + 4, d[i++]);
        }
        break;
    case 10:    /* 8x8 colour: the pattern's pixels */
    case 11:    /* 8x1 colour */
    case 12:    /* 1x8 colour */
    {
        static const uint8_t bypp_of[16] = { [2] = 1, [3] = 2, [4] = 2, [6] = 4, [7] = 1 };
        unsigned bypp = bypp_of[(gmc >> 8) & 15];
        unsigned ndw = (brush == 10 ? 64 : 8) * bypp / 4;
        if (!bypp) {
            qemu_log_mask(LOG_UNIMP, "ati: CP 2D colour brush at this depth\n");
            break;
        }
        for (unsigned k = 0; k < ndw && i < n; k++) {
            ati_reg_write(s, BRUSH_DATA0 + 4 * k, d[i++]);
        }
        break;
    }
    case 13:    /* solid colour */
        if (i < n) {
            ati_reg_write(s, DP_BRUSH_FRGD_CLR, d[i++]);
        }
        break;
    case 15:    /* none */
        break;
    default:
        qemu_log_mask(LOG_UNIMP, "ati: CP 2D brush type %u\n", brush);
        break;
    }
    return i;
}

void ati_2d_packet(ATIVGAState *s, uint8_t op, const uint32_t *d, unsigned n)
{
    unsigned i;

    trace_ati_2d_packet(op, n);
    if (!n) {
        return;
    }
    trace_ati_2d_packet_data(op, n, d[0], n > 1 ? d[1] : 0, n > 2 ? d[2] : 0,
                             n > 3 ? d[3] : 0, n > 4 ? d[4] : 0,
                             n > 5 ? d[5] : 0, n > 6 ? d[6] : 0,
                             n > 7 ? d[7] : 0);
    s->accel.packets2d++;
    if (s->debug & 1) {
        static int dumped;
        if (dumped < 12) {
            dumped++;
            fprintf(stderr, "ati 2D packet 0x%02x n=%u:", op, n);
            for (unsigned k = 0; k < n && k < 600; k++) {
                fprintf(stderr, "%s%08x", k % 8 ? " " : "\n  ", d[k]);
            }
            fprintf(stderr, "\n");
        }
    }
    switch (op) {
    case 0x91:  /* CNTL_PAINT: corners (top<<16|left), (bottom<<16|right),
                   right/bottom exclusive (unlike the _MULTI forms) */
        i = ati_2d_packet_header(s, d, n);
        for (; i + 1 < n; i += 2) {
            uint32_t y1 = d[i] >> 16, x1 = d[i] & 0xffff;
            uint32_t y2 = d[i + 1] >> 16, x2 = d[i + 1] & 0xffff;
            if (x2 > x1 && y2 > y1) {
                ati_reg_write(s, DST_X_Y, (x1 << 16) | y1);
                ati_reg_write(s, DST_WIDTH_HEIGHT,
                              ((x2 - x1) << 16) | (y2 - y1));
            }
        }
        break;
    case 0x9a:  /* CNTL_PAINT_MULTI: rects (x<<16|y), (w<<16|h) */
        i = ati_2d_packet_header(s, d, n);
        for (; i + 1 < n; i += 2) {
            ati_reg_write(s, DST_X_Y, d[i]);
            ati_reg_write(s, DST_WIDTH_HEIGHT, d[i + 1]);
        }
        break;
    case 0x92:  /* CNTL_BITBLT */
    case 0x9b:  /* CNTL_BITBLT_MULTI: src (x<<16|y), dst, (w<<16|h) */
    case 0x9c:  /* CNTL_TRANS_BITBLT: + CLR_CMP_CNTL, CLR_CMP_CLR, mask */
        i = ati_2d_packet_header(s, d, n);
        if (op == 0x9c && i + 2 < n) {
            ati_reg_write(s, CLR_CMP_CNTL, d[i++]);
            ati_reg_write(s, CLR_CMP_CLR_SRC, d[i++]);
            ati_reg_write(s, CLR_CMP_MASK, d[i++]);
        }
        for (; i + 2 < n; i += 3) {
            ati_reg_write(s, SRC_X_Y, d[i]);
            ati_reg_write(s, DST_X_Y, d[i + 1]);
            ati_reg_write(s, DST_WIDTH_HEIGHT, d[i + 2]);
        }
        break;
    case 0x94:
        /*
         * CNTL_HOSTDATA_BLT: header, source fg and bg colours (always),
         * DST_Y_X (y<<16|x), DST_HEIGHT_WIDTH (h<<16|w), the number of data
         * dwords, the data.
         */
        i = ati_2d_packet_header(s, d, n);
        if (i + 4 < n) {
            uint32_t ndw;
            ati_reg_write(s, DP_SRC_FRGD_CLR, d[i++]);
            ati_reg_write(s, DP_SRC_BKGD_CLR, d[i++]);
            ati_reg_write(s, DST_Y_X, d[i]);
            ati_reg_write(s, DST_HEIGHT_WIDTH, d[i + 1]);
            ndw = MIN(d[i + 2], n - (i + 3));
            i += 3;
            for (uint32_t k = 0; k < ndw; k++) {
                ati_reg_write(s, k + 1 < ndw ? HOST_DATA0 : HOST_DATA_LAST,
                              d[i + k]);
            }
        }
        break;
    default:
        qemu_log_mask(LOG_UNIMP, "ati: 2D packet 0x%02x (%u dwords) not "
                      "implemented\n", op, n);
        break;
    }
}
