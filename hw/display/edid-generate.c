/*
 * QEMU EDID generator.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */
#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "hw/display/edid.h"

/*
 * The established timing bits (.byte), the additional standard timings 3
 * bits (.xtra3) and the CTA video data block VICs (.dta) each name a fixed
 * refresh rate. Everything else, including .xtra3 entries while a standard
 * timing slot is still free, goes into the standard timings, which carry
 * the host's refresh rate.
 */
static const struct edid_mode {
    uint32_t xres;
    uint32_t yres;
    uint32_t byte;
    uint32_t xtra3;
    uint32_t bit;
    uint32_t dta;
} modes[] = {
    /* dea/dta extension timings (all @ 50 Hz) */
    { .xres = 5120,   .yres = 2160,   .dta = 125 },
    { .xres = 4096,   .yres = 2160,   .dta = 101 },
    { .xres = 3840,   .yres = 2160,   .dta =  96 },
    { .xres = 2560,   .yres = 1080,   .dta =  89 },
    { .xres = 2048,   .yres = 1152 },
    { .xres = 1920,   .yres = 1080,   .dta =  31 },

    /* dea/dta extension timings (all @ 60 Hz) */
    { .xres = 3840,   .yres = 2160,   .dta =  97 },

    /* additional standard timings 3 (all @ 60Hz) */
    { .xres = 1920,   .yres = 1200,   .xtra3 = 10,   .bit = 0 },
    { .xres = 1600,   .yres = 1200,   .xtra3 =  9,   .bit = 2 },
    { .xres = 1680,   .yres = 1050,   .xtra3 =  9,   .bit = 5 },
    { .xres = 1440,   .yres =  900,   .xtra3 =  8,   .bit = 5 },
    { .xres = 1280,   .yres = 1024,   .xtra3 =  7,   .bit = 1 },
    { .xres = 1280,   .yres =  960,   .xtra3 =  7,   .bit = 3 },
    { .xres = 1280,   .yres =  768,   .xtra3 =  7,   .bit = 6 },

    { .xres = 1920,   .yres = 1440,   .xtra3 = 11,   .bit = 5 },
    { .xres = 1856,   .yres = 1392,   .xtra3 = 10,   .bit = 3 },
    { .xres = 1792,   .yres = 1344,   .xtra3 = 10,   .bit = 5 },
    { .xres = 1440,   .yres = 1050,   .xtra3 =  8,   .bit = 1 },
    { .xres = 1360,   .yres =  768,   .xtra3 =  8,   .bit = 7 },

    /* established timings (all @ 60Hz) */
    { .xres = 1024,   .yres =  768,   .byte  = 36,   .bit = 3 },
    { .xres =  800,   .yres =  600,   .byte  = 35,   .bit = 0 },
    { .xres =  640,   .yres =  480,   .byte  = 35,   .bit = 5 },
};

/*
 * The most a display range limits descriptor can express: a byte plus the
 * EDID 1.4 +255 offset for the rates, a byte of 10 MHz for the clock.
 */
#define EDID_RANGE_MAX_RATE     510     /* Hz and kHz */
#define EDID_RANGE_MAX_CLOCK    2550    /* MHz */

typedef struct Timings {
    uint32_t xfront;
    uint32_t xsync;
    uint32_t xblank;

    uint32_t yfront;
    uint32_t ysync;
    uint32_t yblank;

    uint64_t clock;     /* 10 kHz */

    /* as a guest derives them from the fields above */
    uint32_t hfreq;     /* line rate, Hz */
    uint32_t vfreq;     /* refresh rate, mHz */
} Timings;

/* the display range limits, base block units */
typedef struct Ranges {
    uint32_t vmin;      /* Hz */
    uint32_t vmax;
    uint32_t hmin;      /* kHz */
    uint32_t hmax;
} Ranges;

static void generate_timings(Timings *timings, uint32_t refresh_rate,
                             uint32_t xres, uint32_t yres, bool reduced)
{
    uint32_t xtotal, ytotal;

    if (reduced) {
        /*
         * The horizontal blanking of CVT reduced blanking v2, which is what
         * a flat panel drives. It keeps the pixel clock of a fast timing
         * inside the fields that carry it: 655.35 MHz in a detailed timing,
         * 2550 MHz in the range limits.
         */
        timings->xfront = 8;
        timings->xsync  = 32;
        timings->xblank = 80;
    } else {
        /* pull some realistic looking timings out of thin air */
        timings->xfront = xres * 25 / 100;
        timings->xsync  = xres *  3 / 100;
        timings->xblank = xres * 35 / 100;
    }
    xtotal = xres + timings->xblank;

    timings->yfront = yres *  5 / 1000;
    timings->ysync  = yres *  5 / 1000;
    timings->yblank = yres * 35 / 1000;
    ytotal = yres + timings->yblank;

    timings->clock = ((uint64_t)refresh_rate * xtotal * ytotal) / 10000000;

    /*
     * Derive the rates from the published clock rather than from
     * refresh_rate: the clock lost up to 10 kHz to rounding, and the guest
     * only ever sees the clock.
     */
    timings->hfreq = timings->clock * 10000 / xtotal;
    timings->vfreq = timings->clock * 10000 * 1000 /
                     ((uint64_t)xtotal * ytotal);
}

/*
 * The range has to contain every listed timing, the preferred one included
 * (E-EDID 1.4, display range limits descriptor). A guest that finds its
 * preferred timing outside the declared range treats the EDID as invalid
 * and falls back to a default mode list. Widen the range QEMU has always
 * declared to the preferred timing, rounding its minimum down and its
 * maximum up, so the timing stays inside however the guest rounds its own
 * computation of the rate. Zero is reserved.
 */
static void generate_ranges(Ranges *ranges, const Timings *timings)
{
    ranges->vmin = MIN(50, MAX(timings->vfreq / 1000, 1));
    ranges->vmax = MAX(125, DIV_ROUND_UP(timings->vfreq, 1000));
    ranges->hmin = MIN(30, MAX(timings->hfreq / 1000, 1));
    ranges->hmax = MAX(160, DIV_ROUND_UP(timings->hfreq, 1000));
}

static void edid_ext_dta(uint8_t *dta)
{
    dta[0] = 0x02;
    dta[1] = 0x03;
    dta[2] = 0x05;
    dta[3] = 0x00;

    /* video data block */
    dta[4] = 0x40;
}

static void edid_ext_dta_mode(uint8_t *dta, uint8_t nr)
{
    dta[dta[2]] = nr;
    dta[2]++;
    dta[4]++;
}

static int edid_std_mode(uint8_t *mode, uint32_t xres, uint32_t yres,
                         uint32_t hz)
{
    uint32_t aspect;

    if (xres == 0 || yres == 0) {
        mode[0] = 0x01;
        mode[1] = 0x01;
        return 0;

    } else if (xres * 10 == yres * 16) {
        aspect = 0;
    } else if (xres * 3 == yres * 4) {
        aspect = 1;
    } else if (xres * 4 == yres * 5) {
        aspect = 2;
    } else if (xres * 9 == yres * 16) {
        aspect = 3;
    } else {
        return -1;
    }

    if ((xres / 8) - 31 > 255) {
        return -1;
    }

    /*
     * The 6-bit field spans 60..123 Hz. Saturate rather than fall back to
     * 60 Hz, so a guest that builds its mode list from these entries gets
     * the closest rate the field can express to the preferred timing.
     */
    hz = MIN(MAX(hz, 60), 123);

    mode[0] = (xres / 8) - 31;
    mode[1] = ((aspect << 6) | (hz - 60));
    return 0;
}

static void edid_fill_modes(uint8_t *edid, uint8_t *xtra3, uint8_t *dta,
                            uint32_t maxx, uint32_t maxy, uint32_t hz)
{
    const struct edid_mode *mode;
    int std = 38;
    int rc, i;

    for (i = 0; i < ARRAY_SIZE(modes); i++) {
        mode = modes + i;

        if ((maxx && mode->xres > maxx) ||
            (maxy && mode->yres > maxy)) {
            continue;
        }

        if (mode->byte) {
            edid[mode->byte] |= (1 << mode->bit);
        } else if (std < 54) {
            rc = edid_std_mode(edid + std, mode->xres, mode->yres, hz);
            if (rc == 0) {
                std += 2;
            }
        } else if (mode->xtra3 && xtra3) {
            xtra3[mode->xtra3] |= (1 << mode->bit);
        }

        if (dta && mode->dta) {
            edid_ext_dta_mode(dta, mode->dta);
        }
    }

    while (std < 54) {
        edid_std_mode(edid + std, 0, 0, hz);
        std += 2;
    }
}

static void edid_checksum(uint8_t *edid, size_t len)
{
    uint32_t sum = 0;
    int i;

    for (i = 0; i < len; i++) {
        sum += edid[i];
    }
    sum &= 0xff;
    if (sum) {
        edid[len] = 0x100 - sum;
    }
}

static uint8_t *edid_desc_next(uint8_t *edid, uint8_t *dta, uint8_t *desc)
{
    if (desc == NULL) {
        return NULL;
    }
    if (desc + 18 + 18 < edid + 127) {
        return desc + 18;
    }
    if (dta) {
        if (desc < edid + 127) {
            return dta + dta[2];
        }
        if (desc + 18 + 18 < dta + 127) {
            return desc + 18;
        }
    }
    return NULL;
}

static void edid_desc_type(uint8_t *desc, uint8_t type)
{
    desc[0] = 0;
    desc[1] = 0;
    desc[2] = 0;
    desc[3] = type;
    desc[4] = 0;
}

static void edid_desc_text(uint8_t *desc, uint8_t type,
                           const char *text)
{
    size_t len;

    edid_desc_type(desc, type);
    memset(desc + 5, ' ', 13);

    len = strlen(text);
    if (len > 12) {
        len = 12;
    }
    memcpy(desc + 5, text, len);
    desc[5 + len] = '\n';
}

/* the preferred aspect ratio code of the CVT support bytes */
static uint8_t edid_cvt_aspect(uint32_t xres, uint32_t yres)
{
    if (xres * 3 == yres * 4) {
        return 0;
    } else if (xres * 10 == yres * 16) {
        return 2;
    } else if (xres * 4 == yres * 5) {
        return 3;
    } else if (xres * 9 == yres * 15) {
        return 4;
    }
    return 1; /* 16:9 */
}

/*
 * cvt_hz is the refresh rate the standard timings carry when it is not one
 * the VESA DMT modes have, and zero when it is.
 */
static void edid_desc_ranges(uint8_t *desc, const Ranges *ranges,
                             uint32_t cvt_hz, uint32_t xres, uint32_t yres)
{
    uint32_t vmax = MIN(ranges->vmax, EDID_RANGE_MAX_RATE);
    uint32_t hmax = MIN(ranges->hmax, EDID_RANGE_MAX_RATE);

    edid_desc_type(desc, 0xfd);

    /*
     * EDID 1.4 offset flags: a maximum above 255 is stored less 255. The
     * minima only ever shrink from their defaults, so they never need one.
     */
    if (vmax > 255) {
        desc[4] |= 0x02;
        vmax -= 255;
    }
    if (hmax > 255) {
        desc[4] |= 0x08;
        hmax -= 255;
    }

    /* vertical (Hz) */
    desc[5] = ranges->vmin;
    desc[6] = vmax;

    /* horizontal (kHz) */
    desc[7] = ranges->hmin;
    desc[8] = hmax;

    /* max dot clock: 2550 MHz, the most the field can express */
    desc[9] = EDID_RANGE_MAX_CLOCK / 10;

    if (!cvt_hz) {
        /* range limits only */
        desc[10] = 0x01;

        /* padding */
        desc[11] = '\n';
        memset(desc + 12, ' ', 6);
        return;
    }

    /*
     * A standard timing at a rate no DMT mode has asks the guest to derive
     * its raster from a formula, so declare one; GTF is deprecated in
     * EDID 1.4. The continuous frequency bit of the feature byte has to
     * accompany this.
     */
    desc[10] = 0x04;
    desc[11] = 0x11; /* CVT 1.1 */

    /* no dot clock correction, no limit on active pixels per line */
    desc[12] = 0;
    desc[13] = 0;

    /* the aspect ratios a standard timing can express: 4:3 16:9 16:10 5:4 */
    desc[14] = 0xf0;

    /* preferred aspect ratio, reduced blanking */
    desc[15] = (edid_cvt_aspect(xres, yres) << 5) | 0x10;

    /* no scaling */
    desc[16] = 0;

    /* preferred refresh rate (Hz) */
    desc[17] = MIN(cvt_hz, 255);
}

/* additional standard timings 3 */
static void edid_desc_xtra3_std(uint8_t *desc)
{
    edid_desc_type(desc, 0xf7);
    desc[5] = 10;
}

static void edid_desc_dummy(uint8_t *desc)
{
    edid_desc_type(desc, 0x10);
}

static void edid_desc_timing(uint8_t *desc, const Timings *timings,
                             uint32_t xres, uint32_t yres,
                             uint32_t xmm, uint32_t ymm)
{
    stw_le_p(desc, timings->clock);

    desc[2] = xres   & 0xff;
    desc[3] = timings->xblank & 0xff;
    desc[4] = (((xres   & 0xf00) >> 4) |
               ((timings->xblank & 0xf00) >> 8));

    desc[5] = yres   & 0xff;
    desc[6] = timings->yblank & 0xff;
    desc[7] = (((yres   & 0xf00) >> 4) |
               ((timings->yblank & 0xf00) >> 8));

    desc[8] = timings->xfront & 0xff;
    desc[9] = timings->xsync  & 0xff;

    desc[10] = (((timings->yfront & 0x00f) << 4) |
                ((timings->ysync  & 0x00f) << 0));
    desc[11] = (((timings->xfront & 0x300) >> 2) |
                ((timings->xsync  & 0x300) >> 4) |
                ((timings->yfront & 0x030) >> 2) |
                ((timings->ysync  & 0x030) >> 4));

    desc[12] = xmm & 0xff;
    desc[13] = ymm & 0xff;
    desc[14] = (((xmm & 0xf00) >> 4) |
                ((ymm & 0xf00) >> 8));

    desc[17] = 0x18;
}

static uint32_t edid_to_10bit(float value)
{
    return (uint32_t)(value * 1024 + 0.5);
}

static void edid_colorspace(uint8_t *edid,
                            float rx, float ry,
                            float gx, float gy,
                            float bx, float by,
                            float wx, float wy)
{
    uint32_t red_x   = edid_to_10bit(rx);
    uint32_t red_y   = edid_to_10bit(ry);
    uint32_t green_x = edid_to_10bit(gx);
    uint32_t green_y = edid_to_10bit(gy);
    uint32_t blue_x  = edid_to_10bit(bx);
    uint32_t blue_y  = edid_to_10bit(by);
    uint32_t white_x = edid_to_10bit(wx);
    uint32_t white_y = edid_to_10bit(wy);

    edid[25] = (((red_x   & 0x03) << 6) |
                ((red_y   & 0x03) << 4) |
                ((green_x & 0x03) << 2) |
                ((green_y & 0x03) << 0));
    edid[26] = (((blue_x  & 0x03) << 6) |
                ((blue_y  & 0x03) << 4) |
                ((white_x & 0x03) << 2) |
                ((white_y & 0x03) << 0));
    edid[27] = red_x   >> 2;
    edid[28] = red_y   >> 2;
    edid[29] = green_x >> 2;
    edid[30] = green_y >> 2;
    edid[31] = blue_x  >> 2;
    edid[32] = blue_y  >> 2;
    edid[33] = white_x >> 2;
    edid[34] = white_y >> 2;
}

static uint32_t qemu_edid_dpi_from_mm(uint32_t mm, uint32_t res)
{
    return res * 254 / 10 / mm;
}

uint32_t qemu_edid_dpi_to_mm(uint32_t dpi, uint32_t res)
{
    return res * 254 / 10 / dpi;
}

/*
 * The range limits descriptor stops at 510 Hz, 510 kHz per line and
 * 2550 MHz. A timing past any of those needs the room a DisplayID 2.0
 * range block has, and only then is it worth asking a guest to read one:
 * type VII timings have been understood since Linux 5.18, type I since
 * long before.
 */
static bool displayid_needs_2_0(const Timings *timings, const Ranges *ranges)
{
    return ranges->vmax > EDID_RANGE_MAX_RATE ||
           ranges->hmax > EDID_RANGE_MAX_RATE ||
           timings->clock > EDID_RANGE_MAX_CLOCK * 100;
}

/* every DisplayID field is stored one less than it counts */
static void displayid_clock(uint8_t *dst, uint32_t clock)
{
    clock--;
    dst[0] = clock & 0xff;
    dst[1] = (clock & 0xff00) >> 8;
    dst[2] = (clock & 0xff0000) >> 16;
}

/*
 * A type I (DisplayID 1.3) or type VII (DisplayID 2.0) timing; the two are
 * laid out alike and differ only in the unit of the clock.
 */
static void displayid_timing(uint8_t *timing, const Timings *timings,
                             uint32_t xres, uint32_t yres, uint32_t clock)
{
    displayid_clock(timing, clock);

    timing[3] = 0x88; /* preferred timing, aspect ratio undefined */

    stw_le_p(timing + 4,  0xffff & (xres - 1));
    stw_le_p(timing + 6,  0xffff & (timings->xblank - 1));
    stw_le_p(timing + 8,  0xffff & (timings->xfront - 1));
    stw_le_p(timing + 10, 0xffff & (timings->xsync - 1));

    stw_le_p(timing + 12, 0xffff & (yres - 1));
    stw_le_p(timing + 14, 0xffff & (timings->yblank - 1));
    stw_le_p(timing + 16, 0xffff & (timings->yfront - 1));
    stw_le_p(timing + 18, 0xffff & (timings->ysync - 1));
}

static void displayid_generate_1_3(uint8_t *did, const Timings *timings,
                                   uint32_t xres, uint32_t yres)
{
    did[0] = 0x70; /* display id extension */
    did[1] = 0x13; /* version 1.3 */
    did[2] = 23;   /* length */
    did[3] = 0x03; /* product type (0x03 == standalone display device) */

    did[5] = 0x03; /* Type I Detailed Timing Data Block */
    did[6] = 0x00; /* revision */
    did[7] = 0x14; /* block length */

    displayid_timing(did + 8, timings, xres, yres, timings->clock);

    edid_checksum(did + 1, did[2] + 4);
}

static void displayid_generate_2_0(uint8_t *did, const Timings *timings,
                                   const Ranges *ranges,
                                   uint32_t xres, uint32_t yres)
{
    uint32_t clock_khz = timings->clock * 10;
    uint32_t vmax = MIN(ranges->vmax, 1023);
    uint8_t *block;

    did[0] = 0x70; /* display id extension */
    did[1] = 0x20; /* version 2.0 */
    did[2] = 35;   /* length */
    did[3] = 0x04; /* primary use case: desktop productivity display */

    block = did + 5;
    block[0] = 0x22; /* Type VII Timing Data Block */
    block[1] = 0x00; /* revision */
    block[2] = 0x14; /* block length */

    displayid_timing(block + 3, timings, xres, yres, clock_khz);

    block += 3 + block[2];
    block[0] = 0x25; /* Dynamic Video Timing Range Limits Data Block */
    block[1] = 0x01; /* revision 1: the vertical maximum has ten bits */
    block[2] = 0x09; /* block length */

    /* pixel clock (kHz): a scanout the host composites has no lower limit */
    displayid_clock(block + 3, 1);
    displayid_clock(block + 6, clock_khz);

    /* vertical (Hz) */
    block[9]  = ranges->vmin;
    block[10] = vmax & 0xff;
    block[11] = (vmax & 0x300) >> 8;

    edid_checksum(did + 1, did[2] + 4);
}

void qemu_edid_generate(uint8_t *edid, size_t size,
                        qemu_edid_info *info)
{
    Timings timings;
    Ranges ranges;
    uint8_t *desc = edid + 54;
    uint8_t *xtra3 = NULL;
    uint8_t *dta = NULL;
    uint8_t *did = NULL;
    uint32_t width_mm, height_mm;
    /*
     * Without a host rate the detailed timing is synthesised at 75 Hz to
     * give it a plausible pixel clock, while the standard timings keep the
     * 60 Hz VESA DMT modes.
     */
    uint32_t refresh_rate = info->refresh_rate ? info->refresh_rate : 75000;
    uint32_t std_hz = info->refresh_rate ?
        ((uint64_t)info->refresh_rate + 500) / 1000 : 60;
    /*
     * A rate the standard timings end up carrying, once saturated to the
     * 60..123 Hz their field spans, pulls in the formula declaration and
     * the raster that goes with it. Anything else generates the EDID QEMU
     * always has, byte for byte.
     */
    bool host_rate = info->refresh_rate && std_hz > 60;
    uint32_t dpi = 100; /* if no width_mm/height_mm */
    bool large_screen;

    /* =============== set defaults  =============== */

    if (!info->vendor || strlen(info->vendor) != 3) {
        info->vendor = "RHT";
    }
    if (!info->name) {
        info->name = "QEMU Monitor";
    }
    if (!info->prefx) {
        info->prefx = 1280;
    }
    if (!info->prefy) {
        info->prefy = 800;
    }
    if (info->width_mm && info->height_mm) {
        width_mm = info->width_mm;
        height_mm = info->height_mm;
        dpi = qemu_edid_dpi_from_mm(width_mm, info->prefx);
    } else {
        width_mm = qemu_edid_dpi_to_mm(dpi, info->prefx);
        height_mm = qemu_edid_dpi_to_mm(dpi, info->prefy);
    }

    generate_timings(&timings, refresh_rate, info->prefx, info->prefy,
                     host_rate);
    generate_ranges(&ranges, &timings);

    /*
     * The base block describes the preferred timing only while its detailed
     * timing descriptor holds the resolution and the pixel clock, and its
     * range limits descriptor holds the rates. Past that the DisplayID
     * extension carries it.
     */
    large_screen = info->prefx >= 4096 || info->prefy >= 4096 ||
                   timings.clock >= 65536 ||
                   displayid_needs_2_0(&timings, &ranges);

    /* =============== extensions  =============== */

    if (size >= 256) {
        dta = edid + 128;
        edid[126]++;
        edid_ext_dta(dta);
    }

    if (size >= 384 && large_screen) {
        did = edid + 256;
        edid[126]++;
    }

    /* =============== header information =============== */

    /* fixed */
    edid[0] = 0x00;
    edid[1] = 0xff;
    edid[2] = 0xff;
    edid[3] = 0xff;
    edid[4] = 0xff;
    edid[5] = 0xff;
    edid[6] = 0xff;
    edid[7] = 0x00;

    /* manufacturer id, product code, serial number */
    uint16_t vendor_id = ((((info->vendor[0] - '@') & 0x1f) << 10) |
                          (((info->vendor[1] - '@') & 0x1f) <<  5) |
                          (((info->vendor[2] - '@') & 0x1f) <<  0));
    uint16_t model_nr = 0x1234;
    uint32_t serial_nr = info->serial ? atoi(info->serial) : 0;
    stw_be_p(edid +  8, vendor_id);
    stw_le_p(edid + 10, model_nr);
    stl_le_p(edid + 12, serial_nr);

    /* manufacture week and year */
    edid[16] = 42;
    edid[17] = 2014 - 1990;

    /* edid version */
    edid[18] = 1;
    edid[19] = 4;


    /* =============== basic display parameters =============== */

    /* video input: digital, 8bpc, displayport */
    edid[20] = 0xa5;

    /* screen size: undefined */
    edid[21] = width_mm / 10;
    edid[22] = height_mm / 10;

    /* display gamma: 2.2 */
    edid[23] = 220 - 100;

    /*
     * supported features bitmap: preferred timing, std sRGB, and continuous
     * frequency when the range limits declare a formula
     */
    edid[24] = host_rate ? 0x07 : 0x06;


    /* =============== chromaticity coordinates =============== */

    /* standard sRGB colorspace */
    edid_colorspace(edid,
                    0.6400, 0.3300,   /* red   */
                    0.3000, 0.6000,   /* green */
                    0.1500, 0.0600,   /* blue  */
                    0.3127, 0.3290);  /* white point  */

    /* =============== established timing bitmap =============== */
    /* =============== standard timing information =============== */

    /* both filled by edid_fill_modes() */


    /* =============== descriptor blocks =============== */

    if (!large_screen) {
        edid_desc_timing(desc, &timings, info->prefx, info->prefy,
                         width_mm, height_mm);
        desc = edid_desc_next(edid, dta, desc);
    }

    xtra3 = desc;
    edid_desc_xtra3_std(xtra3);
    desc = edid_desc_next(edid, dta, desc);
    edid_fill_modes(edid, xtra3, dta, info->maxx, info->maxy, std_hz);
    /*
     * dta video data block is finished at thus point,
     * so dta descriptor offsets don't move any more.
     */

    edid_desc_ranges(desc, &ranges, host_rate ? std_hz : 0,
                     info->prefx, info->prefy);
    desc = edid_desc_next(edid, dta, desc);

    if (desc && info->name) {
        edid_desc_text(desc, 0xfc, info->name);
        desc = edid_desc_next(edid, dta, desc);
    }

    if (desc && info->serial) {
        edid_desc_text(desc, 0xff, info->serial);
        desc = edid_desc_next(edid, dta, desc);
    }

    while (desc) {
        edid_desc_dummy(desc);
        desc = edid_desc_next(edid, dta, desc);
    }

    /* =============== display id extensions =============== */

    if (did) {
        if (displayid_needs_2_0(&timings, &ranges)) {
            displayid_generate_2_0(did, &timings, &ranges,
                                   info->prefx, info->prefy);
        } else {
            displayid_generate_1_3(did, &timings, info->prefx, info->prefy);
        }
    }

    /* =============== finish up =============== */

    edid_checksum(edid, 127);
    if (dta) {
        edid_checksum(dta, 127);
    }
    if (did) {
        edid_checksum(did, 127);
    }
}

size_t qemu_edid_size(uint8_t *edid)
{
    uint32_t exts;

    if (edid[0] != 0x00 ||
        edid[1] != 0xff) {
        /* doesn't look like a valid edid block */
        return 0;
    }

    exts = edid[126];
    return 128 * (exts + 1);
}
