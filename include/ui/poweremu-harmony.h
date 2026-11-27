/*
 * Harmony mode: which parts of the guest's screen are its windows.
 *
 * The display bridge needs to know which pixels belong to the guest's
 * windows and which to its desktop, so it can hand PowerEmu a frame whose
 * desktop is transparent and let the windows sit on this Mac's own desktop
 * instead.  Only the GPU model can tell: it watches the copies the guest's
 * compositor makes to the screen.
 *
 * The answer is a grid of tiles over the screen, each holding the kind of
 * the last copy that covered it.  `generation` counts up whenever the grid
 * changes, so a reader can tell when there is nothing new.
 *
 * Copyright (c) 2026 Spartan0285
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef POWEREMU_HARMONY_H
#define POWEREMU_HARMONY_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    PE_AREA_UNKNOWN = 0,        /* nothing has been copied here yet */
    PE_AREA_DESKTOP,            /* the guest's own wallpaper */
    PE_AREA_MENUBAR,            /* the guest's menu bar */
    PE_AREA_WINDOW,             /* a window, or part of one */
} PEHarmonyArea;

#define PE_HARMONY_TILE 8      /* pixels per tile, each way */

/*
 * Turn the grid on or off.  While it is off nothing is tracked and
 * ppc_mac_gpu_harmony_tiles() returns false, so a machine that is not
 * in harmony mode pays nothing for it.
 */
void ppc_mac_gpu_harmony_enable(bool on);

/*
 * The grid as it stands.  Returns false when harmony is off, or when
 * the guest has not drawn anything yet.  The pointer stays valid until
 * the next call; the caller must not free it.
 */
bool ppc_mac_gpu_harmony_tiles(const uint8_t **tiles, int *cols, int *rows,
                                 uint32_t *generation);

#endif /* POWEREMU_HARMONY_H */
