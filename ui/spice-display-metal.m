/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "ui/egl-helpers.h"
#include "ui/spice-display.h"
#include <CoreGraphics/CoreGraphics.h>
#include <IOSurface/IOSurfaceRef.h>
#include <Metal/Metal.h>

@interface SpiceDisplayMetal : NSObject

@property (nonatomic, readonly) id<MTLTexture> renderTarget;
@property (nonatomic, readonly) id<MTLCommandQueue> commandQueue;
@property (nonatomic, nullable, retain) id<MTLTexture> scanoutTexture;
@property (nonatomic, assign) CGRect scanoutRect;
/* draws a scanout whose texel layout differs from the surface's (a 10-bit
 * fullscreen primary into the 8-bit surface): a blit copies raw bits */
@property (nonatomic, nullable, retain) id<MTLRenderPipelineState> convertPipeline;
@property (nonatomic, assign) BOOL convertPipelineFailed;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       surface:(IOSurfaceRef)surface
                         width:(NSInteger)width
                        height:(NSInteger)height;
- (void)scanoutTexture:(id<MTLTexture>)texture rect:(CGRect)rect;
- (void)scanoutDisable;
- (void)drawFrameAtRect:(CGRect)rect completion:(void (^)(void))completion;

@end

@implementation SpiceDisplayMetal

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       surface:(IOSurfaceRef)surface
                         width:(NSInteger)width
                        height:(NSInteger)height
{
    if (self = [super init]) {
        MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
        /* must match the surface, which follows the guest scanout's order */
        textureDescriptor.pixelFormat =
            IOSurfaceGetPixelFormat(surface) == 'RGBA' ? MTLPixelFormatRGBA8Unorm
                                                       : MTLPixelFormatBGRA8Unorm;
        textureDescriptor.width = width;
        textureDescriptor.height = height;
        textureDescriptor.usage = MTLTextureUsageRenderTarget;
        _renderTarget = [device newTextureWithDescriptor:textureDescriptor iosurface:surface plane:0];
        [textureDescriptor release];
        if (!_renderTarget) {
            return nil;
        }
        _commandQueue = [device newCommandQueue];
        if (!_commandQueue) {
            [_renderTarget release];
            return nil;
        }
    }
    return self;
}

- (void)dealloc
{
    [_scanoutTexture release];
    [_convertPipeline release];
    [_commandQueue release];
    [_renderTarget release];
    [super dealloc];
}

- (void)scanoutTexture:(id<MTLTexture>)texture rect:(CGRect)rect
{
    self.scanoutTexture = texture;
    self.scanoutRect = rect;
}

- (void)scanoutDisable
{
    self.scanoutTexture = nil;
    self.scanoutRect = CGRectMake(0, 0, 0, 0);
}

/* A full-screen triangle whose fragments read the source texel at the same
 * position the blit would copy from (destination pixel + scanout origin).
 * Scanout alpha carries nothing (DXGI ignores it), so the output is opaque. */
static NSString *const convert_shader_source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct V { float4 pos [[position]]; };\n"
     "vertex V convert_vs(uint vid [[vertex_id]]) {\n"
     "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
     "    return { float4(p * 2.0 - 1.0, 0.0, 1.0) };\n"
     "}\n"
     "fragment float4 convert_fs(V in [[stage_in]],\n"
     "                           texture2d<float> src [[texture(0)]],\n"
     "                           constant uint2 &origin [[buffer(0)]]) {\n"
     "    return float4(src.read(uint2(in.pos.xy) + origin).rgb, 1.0);\n"
     "}\n";

/* Formats the conversion may copy unencoded into the 8-bit surface: the
 * 10-bit UNORM scanouts, which carry the same (sRGB-encoded) values as an
 * 8-bit one.  A 16-bit-float scanout is normally scRGB (linear, can
 * exceed 1.0) and would need clamping and sRGB encoding, so it is not
 * converted here. */
static bool convertible(MTLPixelFormat f)
{
    switch (f) {
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatBGR10A2Unorm:
        return true;
    default:
        return false;
    }
}

- (id<MTLRenderPipelineState>)pipeline
{
    if (!self.convertPipeline && !self.convertPipelineFailed) {
        id<MTLDevice> device = self.commandQueue.device;
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:convert_shader_source
                                                  options:nil
                                                    error:&err];
        if (!lib) {
            error_report("spice-display-metal: conversion shader: %s",
                         err.localizedDescription.UTF8String);
            self.convertPipelineFailed = YES;
            return nil;
        }
        MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = [[lib newFunctionWithName:@"convert_vs"] autorelease];
        desc.fragmentFunction = [[lib newFunctionWithName:@"convert_fs"] autorelease];
        desc.colorAttachments[0].pixelFormat = self.renderTarget.pixelFormat;
        self.convertPipeline = [[device newRenderPipelineStateWithDescriptor:desc
                                                                       error:&err] autorelease];
        [desc release];
        [lib release];
        if (!self.convertPipeline) {
            error_report("spice-display-metal: conversion pipeline: %s",
                         err.localizedDescription.UTF8String);
            self.convertPipelineFailed = YES;
        }
    }
    return self.convertPipeline;
}

- (void)drawFrameAtRect:(CGRect)rect completion:(void (^)(void))completion
{
    @autoreleasepool {
        if (!self.scanoutTexture) {
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
        MTLPixelFormat srcFormat = self.scanoutTexture.pixelFormat;
        /* the rect comes from the guest's flush: keep the scissor inside
         * the attachment, as Metal requires, and the reads inside the
         * scanout texture past its origin */
        NSUInteger ox = (NSUInteger)self.scanoutRect.origin.x;
        NSUInteger oy = (NSUInteger)self.scanoutRect.origin.y;
        NSUInteger srcw = self.scanoutTexture.width > ox ? self.scanoutTexture.width - ox : 0;
        NSUInteger srch = self.scanoutTexture.height > oy ? self.scanoutTexture.height - oy : 0;
        NSUInteger tw = MIN(self.renderTarget.width, srcw);
        NSUInteger th = MIN(self.renderTarget.height, srch);
        NSUInteger sx = MIN((NSUInteger)MAX(rect.origin.x, 0), tw);
        NSUInteger sy = MIN((NSUInteger)MAX(rect.origin.y, 0), th);
        NSUInteger sw = MIN((NSUInteger)MAX(rect.size.width, 0), tw - sx);
        NSUInteger sh = MIN((NSUInteger)MAX(rect.size.height, 0), th - sy);
        if (convertible(srcFormat)) {
            /* never blit a 10-bit scanout raw: with nothing to draw (an
             * empty rect, or no pipeline) the frame is skipped */
            if (sw && sh && [self pipeline]) {
                MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = self.renderTarget;
                pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLRenderCommandEncoder> enc =
                    [commandBuffer renderCommandEncoderWithDescriptor:pass];
                uint32_t origin[2] = { (uint32_t)self.scanoutRect.origin.x,
                                       (uint32_t)self.scanoutRect.origin.y };
                [enc setRenderPipelineState:self.convertPipeline];
                [enc setScissorRect:(MTLScissorRect){ sx, sy, sw, sh }];
                [enc setFragmentTexture:self.scanoutTexture atIndex:0];
                [enc setFragmentBytes:origin length:sizeof(origin) atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [enc endEncoding];
            }
        } else {
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            MTLOrigin origin = MTLOriginMake(self.scanoutRect.origin.x + rect.origin.x,
                                             self.scanoutRect.origin.y + rect.origin.y,
                                             0);
            MTLSize size = MTLSizeMake(rect.size.width, rect.size.height, 1);

            [blit copyFromTexture:self.scanoutTexture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:origin
                       sourceSize:size
                        toTexture:self.renderTarget
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:(MTLOrigin){rect.origin.x,rect.origin.y,0}];

            [blit endEncoding];
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            completion();
        }];

        [commandBuffer commit];
    }
}

@end

uint32_t qemu_spice_display_metal_texture_fourcc(void *texture)
{
    id<MTLTexture> tex = (id<MTLTexture>)texture;

    if (!tex) {
        return 0;
    }
    switch (tex.pixelFormat) {
    case MTLPixelFormatBGRA8Unorm:
        return 'BGRA';
    case MTLPixelFormatRGBA8Unorm:
        return 'RGBA';
    default:
        /* no IOSurface equivalent wired up; caller keeps its default */
        return 0;
    }
}

SpiceDisplayMetalContext qemu_spice_display_metal_create_context(IOSurfaceRef surface,
                                                                 uint32_t width,
                                                                 uint32_t height)
{
    id<MTLDevice> device = (id<MTLDevice>)qemu_egl_angle_native_device;

    if (!device) {
        return NULL;
    }

    return [[SpiceDisplayMetal alloc] initWithDevice:device
                                             surface:surface
                                               width:width
                                              height:height];
}

void qemu_spice_display_metal_destroy_context(SpiceDisplayMetalContext ctx)
{
    [(SpiceDisplayMetal *)ctx release];
}

void qemu_spice_display_metal_scanout_texture(SpiceDisplayMetalContext ctx,
                                              MTLTexture_id tex, uint32_t x, uint32_t y,
                                              uint32_t w, uint32_t h)
{
    CGRect rect = CGRectMake(x, y, w, h);

    [(SpiceDisplayMetal *)ctx scanoutTexture:(id<MTLTexture>)tex
                                        rect:rect];
}

void qemu_spice_display_metal_scanout_disable(SpiceDisplayMetalContext ctx)
{
    [(SpiceDisplayMetal *)ctx scanoutDisable];
}

bool qemu_spice_display_metal_has_scanout(SpiceDisplayMetalContext ctx)
{
    return [(SpiceDisplayMetal *)ctx scanoutTexture] != nil;
}

void qemu_spice_display_metal_draw_frame(SpiceDisplayMetalContext ctx,
                                         uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                                         SpiceDisplayMetalCompletion completion,
                                         void *data)
{
    CGRect rect = CGRectMake(x, y, w, h);

    [(SpiceDisplayMetal *)ctx drawFrameAtRect:rect completion:^{
        completion(data);
    }];
}
