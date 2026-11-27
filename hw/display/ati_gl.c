/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: run-time host GL loader.
 *
 * Copyright (c) 2026 RetroBoot contributors
 * Copyright (c) 2026 ClassicMac contributors (macOS CGL back end)
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 *
 * Written without QEMU dependencies so it can be built into a small
 * stand-alone test (tools/gltest in research/gfx; ati/tests in ClassicMac).
 *
 * Linux: EGL surfaceless + OpenGL ES 3.0 (libEGL/libGLESv2, dlopen()ed).
 * macOS: CGL + OpenGL 4.1 Core profile (OpenGL.framework, dlopen()ed).
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ati_gl.h"

#ifdef __APPLE__
/*
 * ClassicMac: Apple's OpenGL.framework through CGL with an OpenGL 4.1 Core
 * profile context and no drawable; the renderer only draws into framebuffer
 * objects.  The framework is loaded when the card is realized, as libEGL is
 * on Linux, so the emulator gains no link-time dependency and machines
 * without the card never load it.  Only CGLTypes.h (enums and opaque types)
 * is included and every entry point is resolved with dlsym(), so none of the
 * framework's deprecated declarations are used.
 */
#define GL_SILENCE_DEPRECATION 1
#include <OpenGL/CGLTypes.h>

#define ATI_CGL_FRAMEWORK "/System/Library/Frameworks/OpenGL.framework/OpenGL"

static struct {
    void *lib;
    CGLContextObj ctx;
    CGLError (*ChoosePixelFormat)(const CGLPixelFormatAttribute *,
                                  CGLPixelFormatObj *, GLint *);
    CGLError (*DestroyPixelFormat)(CGLPixelFormatObj);
    CGLError (*CreateContext)(CGLPixelFormatObj, CGLContextObj,
                              CGLContextObj *);
    CGLError (*DestroyContext)(CGLContextObj);
    CGLError (*SetCurrentContext)(CGLContextObj);
    const char *(*ErrorString)(CGLError);
} cgl;

static const char *ati_cgl_error(CGLError e)
{
    const char *s = cgl.ErrorString ? cgl.ErrorString(e) : NULL;

    return s ? s : "unknown CGL error";
}

bool ati_gl_open(ATIGL *gl, char *info, size_t infolen)
{
    /*
     * A hardware renderer first.  Without one (a Mac without a usable GPU),
     * any OpenGL 4.1 Core renderer, which is Apple's software renderer; the
     * info string then says so, and RAVE/OpenGL still draw correctly.
     */
    static const CGLPixelFormatAttribute accelerated[] = {
        kCGLPFAOpenGLProfile,
        (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFAAccelerated,
        kCGLPFAAllowOfflineRenderers,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)24,
        kCGLPFAAlphaSize, (CGLPixelFormatAttribute)8,
        (CGLPixelFormatAttribute)0
    };
    static const CGLPixelFormatAttribute any[] = {
        kCGLPFAOpenGLProfile,
        (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFAAllowOfflineRenderers,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)24,
        kCGLPFAAlphaSize, (CGLPixelFormatAttribute)8,
        (CGLPixelFormatAttribute)0
    };
    const CGLPixelFormatAttribute *const choices[] = { accelerated, any };
    CGLPixelFormatObj pix = NULL;
    CGLError err = kCGLNoError;
    GLint npix = 0;
    bool software = false;
    const char *version, *renderer;

    memset(gl, 0, sizeof(*gl));
    if (!cgl.lib) {
        cgl.lib = dlopen(ATI_CGL_FRAMEWORK, RTLD_NOW | RTLD_LOCAL);
    }
    if (!cgl.lib) {
        snprintf(info, infolen, "OpenGL.framework not available (%s)",
                 dlerror());
        return false;
    }
#define CGLSYM(n) (*(void **)&cgl.n = dlsym(cgl.lib, "CGL" #n))
    CGLSYM(ChoosePixelFormat);
    CGLSYM(DestroyPixelFormat);
    CGLSYM(CreateContext);
    CGLSYM(DestroyContext);
    CGLSYM(SetCurrentContext);
    CGLSYM(ErrorString);
#undef CGLSYM
    if (!cgl.ChoosePixelFormat || !cgl.DestroyPixelFormat ||
        !cgl.CreateContext || !cgl.DestroyContext || !cgl.SetCurrentContext) {
        snprintf(info, infolen, "OpenGL.framework lacks CGL entry points");
        goto fail;
    }
    for (size_t i = 0; i < sizeof(choices) / sizeof(choices[0]); i++) {
        pix = NULL;
        npix = 0;
        err = cgl.ChoosePixelFormat(choices[i], &pix, &npix);
        if (err == kCGLNoError && pix && npix > 0) {
            software = i > 0;
            break;
        }
        if (pix) {
            cgl.DestroyPixelFormat(pix);
            pix = NULL;
        }
    }
    if (!pix) {
        snprintf(info, infolen, "no OpenGL 4.1 Core pixel format (%s)",
                 ati_cgl_error(err));
        goto fail;
    }
    err = cgl.CreateContext(pix, NULL, &cgl.ctx);
    cgl.DestroyPixelFormat(pix);
    if (err != kCGLNoError || !cgl.ctx) {
        snprintf(info, infolen, "CGLCreateContext failed (%s)",
                 ati_cgl_error(err));
        cgl.ctx = NULL;
        goto fail;
    }
    err = cgl.SetCurrentContext(cgl.ctx);
    if (err != kCGLNoError) {
        snprintf(info, infolen, "CGLSetCurrentContext failed (%s)",
                 ati_cgl_error(err));
        goto fail;
    }
#define ATI_GL_LOAD(ret, name, args) \
    *(void **)&gl->name = dlsym(cgl.lib, "gl" #name); \
    if (!gl->name) { \
        snprintf(info, infolen, "missing gl" #name); \
        goto fail; \
    }
    ATI_GL_FUNCS(ATI_GL_LOAD)
#undef ATI_GL_LOAD
    version = (const char *)gl->GetString(GL_VERSION);
    renderer = (const char *)gl->GetString(GL_RENDERER);
    if (!version || version[0] < '4' || version[0] > '9' || version[1] != '.') {
        snprintf(info, infolen, "not an OpenGL 4 Core context (%s)",
                 version ? version : "no GL_VERSION");
        goto fail;
    }
    if (renderer && strstr(renderer, "Software")) {
        software = true;
    }
    snprintf(info, infolen, "CGL; GL %s; %s%s", version,
             renderer ? renderer : "unknown renderer",
             software ? " (software)" : "");
    return true;

fail:
    ati_gl_close();
    return false;
}

void ati_gl_close(void)
{
    if (cgl.SetCurrentContext) {
        cgl.SetCurrentContext(NULL);
    }
    if (cgl.ctx && cgl.DestroyContext) {
        cgl.DestroyContext(cgl.ctx);
    }
    cgl.ctx = NULL;
    /* the framework stays loaded, as Mesa does on Linux */
}

#else /* !__APPLE__: EGL surfaceless + OpenGL ES 3.0 (RetroBoot) */

typedef void *EGLDisplay;
typedef void *EGLContext;
typedef void *EGLSurface;
typedef void *EGLConfig;
typedef int32_t EGLint;
typedef unsigned int EGLBoolean;
typedef unsigned int EGLenum;
typedef intptr_t EGLAttrib;

#define EGL_NONE                        0x3038
#define EGL_VENDOR                      0x3053
#define EGL_VERSION                     0x3054
#define EGL_EXTENSIONS                  0x3055
#define EGL_OPENGL_ES_API               0x30A0
#define EGL_CONTEXT_MAJOR_VERSION       0x3098
#define EGL_CONTEXT_MINOR_VERSION       0x30FB
#define EGL_PLATFORM_SURFACELESS_MESA   0x31DD

static struct {
    void *lib_egl;
    void *lib_gles;
    EGLDisplay dpy;
    EGLContext ctx;
    void *(*GetProcAddress)(const char *);
    EGLDisplay (*GetPlatformDisplay)(EGLenum, void *, const EGLint *);
    EGLBoolean (*Initialize)(EGLDisplay, EGLint *, EGLint *);
    EGLBoolean (*Terminate)(EGLDisplay);
    EGLBoolean (*BindAPI)(EGLenum);
    EGLContext (*CreateContext)(EGLDisplay, EGLConfig, EGLContext,
                                const EGLint *);
    EGLBoolean (*DestroyContext)(EGLDisplay, EGLContext);
    EGLBoolean (*MakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
    const char *(*QueryString)(EGLDisplay, EGLint);
    EGLint (*GetError)(void);
} egl;

static void *ati_gl_sym(const char *name)
{
    void *p = NULL;

    if (egl.lib_gles) {
        p = dlsym(egl.lib_gles, name);
    }
    if (!p && egl.GetProcAddress) {
        p = egl.GetProcAddress(name);
    }
    return p;
}

bool ati_gl_open(ATIGL *gl, char *info, size_t infolen)
{
    EGLint major, minor;
    static const EGLint ctx_attr[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE
    };
    const char *ext;

    memset(gl, 0, sizeof(*gl));
    egl.lib_egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!egl.lib_egl) {
        snprintf(info, infolen, "libEGL.so.1 not available");
        return false;
    }
    egl.lib_gles = dlopen("libGLESv2.so.2", RTLD_NOW | RTLD_LOCAL);
#define EGLSYM(n) (*(void **)&egl.n = dlsym(egl.lib_egl, "egl" #n))
    EGLSYM(GetProcAddress);
    EGLSYM(Initialize);
    EGLSYM(Terminate);
    EGLSYM(BindAPI);
    EGLSYM(CreateContext);
    EGLSYM(DestroyContext);
    EGLSYM(MakeCurrent);
    EGLSYM(QueryString);
    EGLSYM(GetError);
#undef EGLSYM
    if (!egl.GetProcAddress || !egl.Initialize || !egl.CreateContext ||
        !egl.MakeCurrent || !egl.QueryString) {
        snprintf(info, infolen, "libEGL.so.1 lacks core entry points");
        goto fail;
    }
    ext = egl.QueryString(NULL, EGL_EXTENSIONS);
    if (!ext || !strstr(ext, "EGL_MESA_platform_surfaceless")) {
        snprintf(info, infolen, "EGL has no surfaceless platform");
        goto fail;
    }
    *(void **)&egl.GetPlatformDisplay = egl.GetProcAddress("eglGetPlatformDisplayEXT");
    if (!egl.GetPlatformDisplay) {
        snprintf(info, infolen, "eglGetPlatformDisplayEXT missing");
        goto fail;
    }
    egl.dpy = egl.GetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
    if (!egl.dpy || !egl.Initialize(egl.dpy, &major, &minor)) {
        snprintf(info, infolen, "eglInitialize failed (0x%x)",
                 egl.GetError ? egl.GetError() : 0);
        goto fail;
    }
    ext = egl.QueryString(egl.dpy, EGL_EXTENSIONS);
    if (!ext || !strstr(ext, "EGL_KHR_surfaceless_context") ||
        !strstr(ext, "EGL_KHR_no_config_context")) {
        snprintf(info, infolen, "EGL lacks surfaceless/no-config contexts");
        goto fail;
    }
    if (!egl.BindAPI(EGL_OPENGL_ES_API)) {
        snprintf(info, infolen, "no OpenGL ES API");
        goto fail;
    }
    egl.ctx = egl.CreateContext(egl.dpy, NULL, NULL, ctx_attr);
    if (!egl.ctx) {
        snprintf(info, infolen, "no OpenGL ES 3.0 context (0x%x)",
                 egl.GetError());
        goto fail;
    }
    if (!egl.MakeCurrent(egl.dpy, NULL, NULL, egl.ctx)) {
        snprintf(info, infolen, "eglMakeCurrent failed (0x%x)", egl.GetError());
        goto fail;
    }
#define ATI_GL_LOAD(ret, name, args) \
    *(void **)&gl->name = ati_gl_sym("gl" #name); \
    if (!gl->name) { \
        snprintf(info, infolen, "missing gl" #name); \
        goto fail; \
    }
    ATI_GL_FUNCS(ATI_GL_LOAD)
#undef ATI_GL_LOAD
    snprintf(info, infolen, "EGL %d.%d %s; GL %s; %s", major, minor,
             egl.QueryString(egl.dpy, EGL_VENDOR),
             (const char *)gl->GetString(GL_VERSION),
             (const char *)gl->GetString(GL_RENDERER));
    return true;

fail:
    ati_gl_close();
    return false;
}

void ati_gl_close(void)
{
    if (egl.dpy && egl.MakeCurrent) {
        egl.MakeCurrent(egl.dpy, NULL, NULL, NULL);
    }
    if (egl.ctx && egl.DestroyContext) {
        egl.DestroyContext(egl.dpy, egl.ctx);
    }
    if (egl.dpy && egl.Terminate) {
        egl.Terminate(egl.dpy);
    }
    egl.ctx = NULL;
    egl.dpy = NULL;
    /* the libraries stay loaded: Mesa does not like being unloaded */
}

#endif /* __APPLE__ */
