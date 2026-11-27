/*
 * Emulated ATI Radeon R100 (ati-vga,model=r100): RetroBoot 2.0.1
 * (432a43d4ba8502a7afeaa5dd25ecf959c23aed1c) as imported by amcchord/ClassicMac
 * (bcee0a7d400312d3f37fb82d5abf19452416f3bb, ati/), rebased by hand from
 * QEMU 11.0.2 onto QEMU 10.0 (UTM).  GPL-2.0-or-later.  See
 * ATI-R100-PROVENANCE.md in this directory.
 */
/*
 * QEMU ATI Radeon (R100) emulation: host OpenGL ES 3.0 back end.
 *
 * Copyright (c) 2026 RetroBoot contributors
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 *
 * libEGL/libGLESv2 are loaded at run time (dlopen), so the emulator has no
 * link-time GL dependency and still runs (as a 2D card) where none exists.
 * The context is surfaceless (EGL_MESA_platform_surfaceless): Mesa then uses
 * the first render node with a hardware driver, else llvmpipe.  No window
 * system is involved.
 *
 * ClassicMac (macOS): OpenGL.framework is loaded the same way and a CGL
 * OpenGL 4.1 Core profile context without a drawable is used instead.  Every
 * entry point below is core in both OpenGL ES 3.0 and OpenGL 4.1; the shaders
 * get the matching GLSL version line (ATI_GLSL_VERSION).
 */
#ifndef ATI_GL_H
#define ATI_GL_H

#include <stdbool.h>
#include <stdint.h>

/* minimal GL ES 3.0 declarations (no GL headers needed at build time) */
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef float GLfloat;
typedef char GLchar;
typedef intptr_t GLintptr;
typedef intptr_t GLsizeiptr;
typedef unsigned char GLubyte;
typedef struct __GLsync *GLsync;
typedef uint64_t GLuint64;

#define GL_NO_ERROR                    0
#define GL_FALSE                       0
#define GL_TRUE                        1
#define GL_POINTS                      0x0000
#define GL_LINES                       0x0001
#define GL_LINE_LOOP                   0x0002
#define GL_LINE_STRIP                  0x0003
#define GL_TRIANGLES                   0x0004
#define GL_TRIANGLE_STRIP              0x0005
#define GL_TRIANGLE_FAN                0x0006
#define GL_NEVER                       0x0200
#define GL_LESS                        0x0201
#define GL_EQUAL                       0x0202
#define GL_LEQUAL                      0x0203
#define GL_GREATER                     0x0204
#define GL_NOTEQUAL                    0x0205
#define GL_GEQUAL                      0x0206
#define GL_ALWAYS                      0x0207
#define GL_ZERO                        0
#define GL_ONE                         1
#define GL_SRC_COLOR                   0x0300
#define GL_ONE_MINUS_SRC_COLOR         0x0301
#define GL_SRC_ALPHA                   0x0302
#define GL_ONE_MINUS_SRC_ALPHA         0x0303
#define GL_DST_ALPHA                   0x0304
#define GL_ONE_MINUS_DST_ALPHA         0x0305
#define GL_DST_COLOR                   0x0306
#define GL_ONE_MINUS_DST_COLOR         0x0307
#define GL_SRC_ALPHA_SATURATE          0x0308
#define GL_CONSTANT_COLOR              0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR    0x8002
#define GL_CONSTANT_ALPHA              0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA    0x8004
#define GL_FUNC_ADD                    0x8006
#define GL_MIN                         0x8007
#define GL_MAX                         0x8008
#define GL_FUNC_SUBTRACT               0x800A
#define GL_FUNC_REVERSE_SUBTRACT       0x800B
#define GL_FRONT                       0x0404
#define GL_BACK                        0x0405
#define GL_FRONT_AND_BACK              0x0408
#define GL_CULL_FACE                   0x0B44
#define GL_DEPTH_TEST                  0x0B71
#define GL_STENCIL_TEST                0x0B90
#define GL_BLEND                       0x0BE2
#define GL_SCISSOR_TEST                0x0C11
#define GL_DITHER                      0x0BD0
#define GL_POLYGON_OFFSET_FILL         0x8037
#define GL_CW                          0x0900
#define GL_CCW                         0x0901
#define GL_KEEP                        0x1E00
#define GL_REPLACE                     0x1E01
#define GL_INCR                        0x1E02
#define GL_DECR                        0x1E03
#define GL_INVERT                      0x150A
#define GL_INCR_WRAP                   0x8507
#define GL_DECR_WRAP                   0x8508
#define GL_TEXTURE_2D                  0x0DE1
#define GL_TEXTURE0                    0x84C0
#define GL_TEXTURE_MAG_FILTER          0x2800
#define GL_TEXTURE_MIN_FILTER          0x2801
#define GL_TEXTURE_WRAP_S              0x2802
#define GL_TEXTURE_WRAP_T              0x2803
#define GL_TEXTURE_MAX_LEVEL           0x813D
#define GL_TEXTURE_BASE_LEVEL          0x813C
#define GL_NEAREST                     0x2600
#define GL_LINEAR                      0x2601
#define GL_NEAREST_MIPMAP_NEAREST      0x2700
#define GL_LINEAR_MIPMAP_NEAREST       0x2701
#define GL_NEAREST_MIPMAP_LINEAR       0x2702
#define GL_LINEAR_MIPMAP_LINEAR        0x2703
#define GL_REPEAT                      0x2901
#define GL_CLAMP_TO_EDGE               0x812F
#define GL_MIRRORED_REPEAT             0x8370
#define GL_RGBA                        0x1908
#define GL_RGBA8                       0x8058
#define GL_UNSIGNED_BYTE               0x1401
#define GL_UNSIGNED_SHORT              0x1403
#define GL_UNSIGNED_INT                0x1405
#define GL_FLOAT                       0x1406
#define GL_DEPTH_COMPONENT             0x1902
#define GL_DEPTH24_STENCIL8            0x88F0
#define GL_DEPTH_STENCIL               0x84F9
#define GL_UNSIGNED_INT_24_8           0x84FA
#define GL_FRAMEBUFFER                 0x8D40
#define GL_READ_FRAMEBUFFER            0x8CA8
#define GL_DRAW_FRAMEBUFFER            0x8CA9
#define GL_RENDERBUFFER                0x8D41
#define GL_COLOR_ATTACHMENT0           0x8CE0
#define GL_DEPTH_STENCIL_ATTACHMENT    0x821A
#define GL_FRAMEBUFFER_COMPLETE        0x8CD5
#define GL_COLOR_BUFFER_BIT            0x00004000
#define GL_DEPTH_BUFFER_BIT            0x00000100
#define GL_STENCIL_BUFFER_BIT          0x00000400
#define GL_ARRAY_BUFFER                0x8892
#define GL_ELEMENT_ARRAY_BUFFER        0x8893
#define GL_PIXEL_PACK_BUFFER           0x88EB
#define GL_STREAM_DRAW                 0x88E0
#define GL_STREAM_READ                 0x88E1
#define GL_VERTEX_SHADER               0x8B31
#define GL_FRAGMENT_SHADER             0x8B30
#define GL_COMPILE_STATUS              0x8B81
#define GL_LINK_STATUS                 0x8B82
#define GL_INFO_LOG_LENGTH             0x8B84
#define GL_UNPACK_ALIGNMENT            0x0CF5
#define GL_PACK_ALIGNMENT              0x0D05
#define GL_UNPACK_ROW_LENGTH           0x0CF2
#define GL_PACK_ROW_LENGTH             0x0D02
#define GL_VENDOR                      0x1F00
#define GL_RENDERER                    0x1F01
#define GL_VERSION                     0x1F02
#define GL_MAP_READ_BIT                0x0001
#define GL_SYNC_GPU_COMMANDS_COMPLETE  0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT     0x00000001
#define GL_NEAREST_                    GL_NEAREST

/* the first line of every shader: GLSL ES 3.00, or GLSL 4.10 on macOS */
#ifdef __APPLE__
#define ATI_GLSL_VERSION "#version 410 core\n"
#else
#define ATI_GLSL_VERSION "#version 300 es\n"
#endif

#define ATI_GL_FUNCS(X) \
    X(const GLubyte *, GetString, (GLenum)) \
    X(GLenum, GetError, (void)) \
    X(void, Viewport, (GLint, GLint, GLsizei, GLsizei)) \
    X(void, Scissor, (GLint, GLint, GLsizei, GLsizei)) \
    X(void, Enable, (GLenum)) \
    X(void, Disable, (GLenum)) \
    X(void, ClearColor, (GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, ClearDepthf, (GLfloat)) \
    X(void, ClearStencil, (GLint)) \
    X(void, Clear, (GLbitfield)) \
    X(void, ColorMask, (GLboolean, GLboolean, GLboolean, GLboolean)) \
    X(void, DepthMask, (GLboolean)) \
    X(void, DepthFunc, (GLenum)) \
    X(void, DepthRangef, (GLfloat, GLfloat)) \
    X(void, StencilFunc, (GLenum, GLint, GLuint)) \
    X(void, StencilOp, (GLenum, GLenum, GLenum)) \
    X(void, StencilMask, (GLuint)) \
    X(void, BlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum)) \
    X(void, BlendEquationSeparate, (GLenum, GLenum)) \
    X(void, BlendColor, (GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, CullFace, (GLenum)) \
    X(void, FrontFace, (GLenum)) \
    X(void, PolygonOffset, (GLfloat, GLfloat)) \
    X(void, LineWidth, (GLfloat)) \
    X(void, GenTextures, (GLsizei, GLuint *)) \
    X(void, DeleteTextures, (GLsizei, const GLuint *)) \
    X(void, BindTexture, (GLenum, GLuint)) \
    X(void, TexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, \
                         GLenum, GLenum, const void *)) \
    X(void, TexSubImage2D, (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, \
                            GLenum, GLenum, const void *)) \
    X(void, TexParameteri, (GLenum, GLenum, GLint)) \
    X(void, PixelStorei, (GLenum, GLint)) \
    X(void, ActiveTexture, (GLenum)) \
    X(void, GenFramebuffers, (GLsizei, GLuint *)) \
    X(void, DeleteFramebuffers, (GLsizei, const GLuint *)) \
    X(void, BindFramebuffer, (GLenum, GLuint)) \
    X(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint)) \
    X(void, FramebufferRenderbuffer, (GLenum, GLenum, GLenum, GLuint)) \
    X(GLenum, CheckFramebufferStatus, (GLenum)) \
    X(void, GenRenderbuffers, (GLsizei, GLuint *)) \
    X(void, DeleteRenderbuffers, (GLsizei, const GLuint *)) \
    X(void, BindRenderbuffer, (GLenum, GLuint)) \
    X(void, RenderbufferStorage, (GLenum, GLenum, GLsizei, GLsizei)) \
    X(void, BlitFramebuffer, (GLint, GLint, GLint, GLint, GLint, GLint, \
                              GLint, GLint, GLbitfield, GLenum)) \
    X(void, ReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, \
                         void *)) \
    X(GLuint, CreateShader, (GLenum)) \
    X(void, ShaderSource, (GLuint, GLsizei, const GLchar *const *, \
                           const GLint *)) \
    X(void, CompileShader, (GLuint)) \
    X(void, GetShaderiv, (GLuint, GLenum, GLint *)) \
    X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(GLuint, CreateProgram, (void)) \
    X(void, AttachShader, (GLuint, GLuint)) \
    X(void, LinkProgram, (GLuint)) \
    X(void, GetProgramiv, (GLuint, GLenum, GLint *)) \
    X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(void, UseProgram, (GLuint)) \
    X(GLint, GetUniformLocation, (GLuint, const GLchar *)) \
    X(void, BindAttribLocation, (GLuint, GLuint, const GLchar *)) \
    X(void, Uniform1i, (GLint, GLint)) \
    X(void, Uniform1f, (GLint, GLfloat)) \
    X(void, Uniform2f, (GLint, GLfloat, GLfloat)) \
    X(void, Uniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, Uniform4fv, (GLint, GLsizei, const GLfloat *)) \
    X(void, Uniform1iv, (GLint, GLsizei, const GLint *)) \
    X(void, Uniform4iv, (GLint, GLsizei, const GLint *)) \
    X(void, DeleteShader, (GLuint)) \
    X(void, DeleteProgram, (GLuint)) \
    X(void, GenBuffers, (GLsizei, GLuint *)) \
    X(void, DeleteBuffers, (GLsizei, const GLuint *)) \
    X(void, BindBuffer, (GLenum, GLuint)) \
    X(void, BufferData, (GLenum, GLsizeiptr, const void *, GLenum)) \
    X(void, VertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, \
                                  const void *)) \
    X(void, EnableVertexAttribArray, (GLuint)) \
    X(void, DisableVertexAttribArray, (GLuint)) \
    X(void, VertexAttrib4f, (GLuint, GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, GenVertexArrays, (GLsizei, GLuint *)) \
    X(void, BindVertexArray, (GLuint)) \
    X(void, DrawArrays, (GLenum, GLint, GLsizei)) \
    X(void, DrawElements, (GLenum, GLsizei, GLenum, const void *)) \
    X(void, Finish, (void)) \
    X(void, Flush, (void))

typedef struct ATIGL {
#define ATI_GL_PTR(ret, name, args) ret (*name) args;
    ATI_GL_FUNCS(ATI_GL_PTR)
#undef ATI_GL_PTR
} ATIGL;

/* loads EGL/GLES and makes a context current in the calling thread */
bool ati_gl_open(ATIGL *gl, char *info, size_t infolen);
void ati_gl_close(void);

#endif
