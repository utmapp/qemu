/*
 * R300 shader compilation: GLSL -> SPIR-V (shaderc) -> MSL (SPIRV-Cross).
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_spirv.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shaderc/shaderc.h>
#include <spirv_cross_c.h>

#include "r300_us.h"

const char *r300_stage_entry(R300Stage stage)
{
    static const char *names[] = { "r300_vs", "r300_fs", "r300_fs_z" };
    return names[stage];
}

static char *r300_strdup_err(const char *what, const char *msg)
{
    size_t n = strlen(what) + strlen(msg ? msg : "") + 3;
    char *s = malloc(n);
    snprintf(s, n, "%s: %s", what, msg ? msg : "");
    return s;
}

uint32_t *r300_glsl_to_spirv(const char *glsl, R300Stage stage,
                             size_t *nwords, char **err)
{
    static shaderc_compiler_t compiler;
    static const char *macro[] = { "R300_VS", "R300_FS", "R300_FS_Z" };
    shaderc_compile_options_t opts;
    shaderc_compilation_result_t res;
    uint32_t *words = NULL;

    *err = NULL;
    if (!compiler) {
        compiler = shaderc_compiler_initialize();
        if (!compiler) {
            *err = r300_strdup_err("shaderc", "cannot initialise the compiler");
            return NULL;
        }
    }
    opts = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_1);
    shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_3);
    shaderc_compile_options_add_macro_definition(opts, macro[stage],
                                                 strlen(macro[stage]), "1", 1);
    if (stage == R300_STAGE_FS_Z) {
        shaderc_compile_options_add_macro_definition(opts, "R300_FS", 7, "1", 1);
    }
    res = shaderc_compile_into_spv(compiler, glsl, strlen(glsl),
                                   stage == R300_STAGE_VS ? shaderc_glsl_vertex_shader
                                                          : shaderc_glsl_fragment_shader,
                                   r300_stage_entry(stage), "main", opts);
    if (shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        *err = r300_strdup_err("GLSL", shaderc_result_get_error_message(res));
    } else {
        size_t len = shaderc_result_get_length(res);
        words = malloc(len);
        memcpy(words, shaderc_result_get_bytes(res), len);
        *nwords = len / 4;
    }
    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    return words;
}

/* The pre-"_2" API: SPIRV-Cross in Linux distributions still lacks _2. */
static void msl_bind(spvc_compiler c, SpvExecutionModel model, unsigned binding,
                     unsigned buffer, unsigned texture, unsigned sampler)
{
    spvc_msl_resource_binding b;

    spvc_msl_resource_binding_init(&b);
    b.stage = model;
    b.desc_set = 0;
    b.binding = binding;
    b.msl_buffer = buffer;
    b.msl_texture = texture;
    b.msl_sampler = sampler;
    spvc_compiler_msl_add_resource_binding(c, &b);
}

char *r300_spirv_to_msl(const uint32_t *spv, size_t nwords, R300Stage stage,
                        char **err)
{
    spvc_context ctx = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler c = NULL;
    spvc_compiler_options o = NULL;
    const char *src = NULL;
    char *out = NULL;
    SpvExecutionModel model = stage == R300_STAGE_VS ? SpvExecutionModelVertex
                                                     : SpvExecutionModelFragment;

    *err = NULL;
    if (spvc_context_create(&ctx) != SPVC_SUCCESS) {
        *err = r300_strdup_err("SPIRV-Cross", "cannot create a context");
        return NULL;
    }
    if (spvc_context_parse_spirv(ctx, spv, nwords, &ir) != SPVC_SUCCESS ||
        spvc_context_create_compiler(ctx, SPVC_BACKEND_MSL, ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &c) != SPVC_SUCCESS ||
        spvc_compiler_create_compiler_options(c, &o) != SPVC_SUCCESS) {
        goto fail;
    }
    spvc_compiler_options_set_uint(o, SPVC_COMPILER_OPTION_MSL_PLATFORM,
                                   SPVC_MSL_PLATFORM_MACOS);
    spvc_compiler_options_set_uint(o, SPVC_COMPILER_OPTION_MSL_VERSION,
                                   SPVC_MAKE_MSL_VERSION(2, 4, 0));
    /* input attachment k -> [[color(k)]] */
    spvc_compiler_options_set_bool(o, SPVC_COMPILER_OPTION_MSL_IOS_FRAMEBUFFER_FETCH_SUBPASS,
                                   SPVC_TRUE);
    if (spvc_compiler_install_compiler_options(c, o) != SPVC_SUCCESS) {
        goto fail;
    }
    if (stage == R300_STAGE_VS) {
        msl_bind(c, model, R300_BIND_VERTS, 0, 0, 0);
        msl_bind(c, model, R300_BIND_MS, 1, 0, 0);
        msl_bind(c, model, R300_BIND_VSU, 2, 0, 0);
    } else {
        msl_bind(c, model, R300_BIND_UNIFORMS, 0, 0, 0);
        msl_bind(c, model, R300_BIND_ZPASS, 1, 0, 0);
        msl_bind(c, model, R300_BIND_VRAM, 2, 0, 0);
        msl_bind(c, model, R300_BIND_AUX, 3, 0, 0);
        for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
            msl_bind(c, model, R300_BIND_TEX0 + k, 0, k, k);
        }
    }
    if (spvc_compiler_rename_entry_point(c, "main", r300_stage_entry(stage),
                                         model) != SPVC_SUCCESS ||
        spvc_compiler_compile(c, &src) != SPVC_SUCCESS) {
        goto fail;
    }
    out = strdup(src);
    spvc_context_destroy(ctx);
    return out;

fail:
    *err = r300_strdup_err("SPIRV-Cross", spvc_context_get_last_error_string(ctx));
    spvc_context_destroy(ctx);
    return NULL;
}

char *r300_glsl_to_msl(const char *glsl, R300Stage stage, char **err)
{
    size_t n;
    uint32_t *spv = r300_glsl_to_spirv(glsl, stage, &n, err);
    char *msl;

    if (!spv) {
        return NULL;
    }
    msl = r300_spirv_to_msl(spv, n, stage, err);
    free(spv);
    return msl;
}

const char *r300_msl_toolchain_id(void)
{
    static char id[256];

    if (!id[0]) {
        unsigned sv = 0, sr = 0;

        shaderc_get_spv_version(&sv, &sr);
        snprintf(id, sizeof(id), "r300-msl-1 spvc %s shaderc-spv %x.%u built %s %s",
                 spvc_get_commit_revision_and_timestamp(), sv, sr,
                 __DATE__, __TIME__);
    }
    return id;
}
