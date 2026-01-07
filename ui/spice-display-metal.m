/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
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
        textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
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

- (void)drawFrameAtRect:(CGRect)rect completion:(void (^)(void))completion
{
    @autoreleasepool {
        if (!self.scanoutTexture) {
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
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

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            completion();
        }];

        [commandBuffer commit];
    }
}

@end

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
