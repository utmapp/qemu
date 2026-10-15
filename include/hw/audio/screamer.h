/*
 * QEMU PowerMac Awacs Screamer device support
 *
 * Copyright (c) 2016 Mark Cave-Ayland
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef HW_AUDIO_SCREAMER_H
#define HW_AUDIO_SCREAMER_H

#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "hw/ppc/mac_dbdma.h"
#include "audio/audio.h"

#define TYPE_SCREAMER "screamer"
OBJECT_DECLARE_SIMPLE_TYPE(ScreamerState, SCREAMER)

#define SCREAMER_BUFFER_SIZE 0x4000

struct ScreamerState {
    /*< private >*/
    SysBusDevice parent_obj;

    /*< public >*/
    MemoryRegion mem;
    qemu_irq irq;
    void *dbdma;
    qemu_irq dma_tx_irq;
    qemu_irq dma_rx_irq;

    QEMUSoundCard card;
    SWVoiceOut *voice;
    uint8_t *mixbuf;        /* ring of TX frames waiting for the host */
    int samples;            /* ring size in frames */
    int shift;

    uint32_t wpos;          /* ring write index (frames) */
    uint32_t rpos;          /* ring read index (frames) */
    uint32_t fill;          /* frames in the ring */

    QEMUTimer *tx_timer;    /* paces TX DMA at the sample rate */
    int64_t tx_last_ns;
    uint32_t tx_credit;     /* frames TX DMA may consume now */

    uint32_t bpos;
    uint32_t ppos;
    uint32_t rate;
    DBDMA_io io;

    QEMUTimer *rx_timer;    /* completes RX DMA at the sample rate */
    DBDMA_io rx_io;

    uint32_t regs[6];
    uint32_t codec_ctrl_regs[8];
};

void macio_screamer_register_dma(ScreamerState *s, void *dbdma, int txchannel, int rxchannel);

#endif
