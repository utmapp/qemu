/*
 * R300 shader compilation: the GLSL from r300_us_to_glsl() -> SPIR-V
 * (shaderc) -> MSL (SPIRV-Cross) for the Metal backend.
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_SPIRV_H
#define HW_DISPLAY_R300_SPIRV_H

#include <stddef.h>
#include <stdint.h>

typedef enum R300Stage {
    R300_STAGE_VS,              /* R300_VS: entry point r300_vs */
    R300_STAGE_FS,              /* R300_FS: r300_fs */
    R300_STAGE_FS_Z,            /* R300_FS_Z: r300_fs_z */
} R300Stage;

/* The stage's entry point name in MSL ("r300_vs", "r300_fs", "r300_fs_z"). */
const char *r300_stage_entry(R300Stage stage);

/*
 * Compile one stage of a r300_us_to_glsl() source to SPIR-V.  Returns
 * malloc'd words (*nwords of them), or NULL with a malloc'd *err.
 */
uint32_t *r300_glsl_to_spirv(const char *glsl, R300Stage stage,
                             size_t *nwords, char **err);

/*
 * Cross-compile a stage's SPIR-V to MSL (macOS, framebuffer fetch for the
 * input attachments, R300_BIND_* mapped as r300_us.h describes).  The
 * entry point is renamed r300_stage_entry(stage).  Returns a malloc'd
 * string, or NULL with a malloc'd *err.
 */
char *r300_spirv_to_msl(const uint32_t *spv, size_t nwords, R300Stage stage,
                        char **err);

/* Both steps. */
char *r300_glsl_to_msl(const char *glsl, R300Stage stage, char **err);

/*
 * Names everything that decides what r300_glsl_to_msl() returns for a given
 * source besides the source itself: the shaderc and SPIRV-Cross builds
 * linked in and when this converter was compiled.  An on-disk cache of its
 * output keys on this, so changing any of them starts a fresh cache.
 */
const char *r300_msl_toolchain_id(void);

#endif
