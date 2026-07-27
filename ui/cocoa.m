/*
 * QEMU Cocoa CG display driver
 *
 * Copyright (c) 2008 Mike Kronenberg
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

#define GL_SILENCE_DEPRECATION

#include "qemu/osdep.h"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#include <crt_externs.h>

#include "qemu/help-texts.h"
#include "qemu-main.h"
#include "ui/clipboard.h"
#include "ui/console.h"
#include "ui/input.h"
#include "ui/kbd-state.h"
#include "system/system.h"
#include "system/runstate.h"
#include "system/runstate-action.h"
#include "system/cpu-throttle.h"
#include "qapi/error.h"
#include "qapi/qapi-commands-block.h"
#include "qapi/qapi-commands-machine.h"
#include "qapi/qapi-commands-misc.h"
#include "system/blockdev.h"
#include "qemu-version.h"
#include "qemu/cutils.h"
#include "qemu/main-loop.h"
#include "qemu/module.h"
#include "qemu/error-report.h"
#include <Carbon/Carbon.h>
#include "hw/core/cpu.h"

#ifdef CONFIG_EGL
#include "ui/egl-context.h"
#endif

#if defined(CONFIG_METAL) && defined(CONFIG_OPENGL) && defined(CONFIG_EGL)
#define USE_METAL
#endif

#ifdef USE_METAL
#include <Metal/Metal.h>
#endif

#ifndef MAC_OS_VERSION_14_0
#define MAC_OS_VERSION_14_0 140000
#endif

//#define DEBUG

#ifdef DEBUG
#define COCOA_DEBUG(...)  { (void) fprintf (stdout, __VA_ARGS__); }
#else
#define COCOA_DEBUG(...)  ((void) 0)
#endif

#define cgrect(nsrect) (*(CGRect *)&(nsrect))

#define UC_CTRL_KEY "\xe2\x8c\x83"
#define UC_ALT_KEY "\xe2\x8c\xa5"

typedef struct {
    int width;
    int height;
} QEMUScreen;

@class QemuCocoaPasteboardTypeOwner;

static DisplayChangeListener dcl;
static DisplaySurface *surface;
static QKbdState *kbd;
static int cursor_hide = 1;
static int left_command_key_enabled = 1;
static bool swap_opt_cmd;

static bool zoom_interpolation;
static NSTextField *pauseLabel;

static bool allow_events;

static NSInteger cbchangecount = -1;
static QemuClipboardInfo *cbinfo;
static QemuEvent cbevent;
static QemuCocoaPasteboardTypeOwner *cbowner;

#ifdef CONFIG_OPENGL

@interface QemuCGLLayer : CAOpenGLLayer
@end

static bool gl_dirty;
static uint32_t gl_scanout_id;
static bool gl_scanout_y0_top;
static QEMUGLContext gl_view_ctx;

#ifdef CONFIG_EGL
static EGLSurface egl_surface;
#endif

static void cocoa_gl_switch(DisplayChangeListener *dcl,
                            DisplaySurface *new_surface);

static void cocoa_gl_render(void);

static bool cocoa_gl_is_compatible_dcl(DisplayGLCtx *dgc,
                                       DisplayChangeListener *dcl);

static QEMUGLContext cocoa_gl_create_context(DisplayGLCtx *dgc,
                                             QEMUGLParams *params);

static void cocoa_gl_destroy_context(DisplayGLCtx *dgc, QEMUGLContext ctx);

static int cocoa_gl_make_context_current(DisplayGLCtx *dgc, QEMUGLContext ctx);

static const DisplayGLCtxOps dgc_ops = {
    .dpy_gl_ctx_is_compatible_dcl = cocoa_gl_is_compatible_dcl,
    .dpy_gl_ctx_create            = cocoa_gl_create_context,
    .dpy_gl_ctx_destroy           = cocoa_gl_destroy_context,
    .dpy_gl_ctx_make_current      = cocoa_gl_make_context_current,
};

static DisplayGLCtx dgc = {
    .ops = &dgc_ops,
};

@implementation QemuCGLLayer
- (id)init
{
    self = [super init];
    if (self) {
        [self setAsynchronous:NO];
    }
    return self;
}

- (CGLContextObj)copyCGLContextForPixelFormat:(CGLPixelFormatObj)pf
{
    CGLContextObj ctx;
    CGLCreateContext(pf, gl_view_ctx, &ctx);
    return ctx;
}

- (CGLPixelFormatObj)copyCGLPixelFormatForDisplayMask:(uint32_t)mask
{
    CGLPixelFormatObj pix;
    GLint npix;
    CGLPixelFormatAttribute attribs[] = {
        kCGLPFADisplayMask,
        mask,
        kCGLPFAOpenGLProfile,
        (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        0
    };

    CGLChoosePixelFormat(attribs, &pix, &npix);

    return pix;
}

- (void)drawInCGLContext:(CGLContextObj)ctx
             pixelFormat:(CGLPixelFormatObj)pf
            forLayerTime:(CFTimeInterval)t
             displayTime:(const CVTimeStamp *)ts
{
    BQL_LOCK_GUARD();
    cocoa_gl_render();

    [super drawInCGLContext:ctx
                pixelFormat:pf
               forLayerTime:t
                displayTime:ts];
}
@end

#endif

// Utility functions to run specified code block with the BQL held
typedef void (^CodeBlock)(void);
typedef bool (^BoolCodeBlock)(void);

static void with_bql(CodeBlock block)
{
    bool locked = bql_locked();
    if (!locked) {
        bql_lock();
    }
    block();
    if (!locked) {
        bql_unlock();
    }
}

static bool bool_with_bql(BoolCodeBlock block)
{
    bool locked = bql_locked();
    bool val;

    if (!locked) {
        bql_lock();
    }
    val = block();
    if (!locked) {
        bql_unlock();
    }
    return val;
}

static int cocoa_keycode_to_qemu(int keycode)
{
    if (qemu_input_map_osx_to_qcode_len <= keycode) {
        error_report("(cocoa) warning unknown keycode 0x%x", keycode);
        return 0;
    }
    return qemu_input_map_osx_to_qcode[keycode];
}

/* Displays an alert dialog box with the specified message */
static void QEMU_Alert(NSString *message)
{
    NSAlert *alert;
    alert = [NSAlert new];
    [alert setMessageText: message];
    [alert runModal];
}

/* Handles any errors that happen with a device transaction */
static void handleAnyDeviceErrors(Error * err)
{
    if (err) {
        QEMU_Alert([NSString stringWithCString: error_get_pretty(err)
                                      encoding: NSASCIIStringEncoding]);
        error_free(err);
    }
}

typedef NS_ENUM(NSInteger, QemuCocoaViewScanout) {
    QemuCocoaViewScanoutNone,
    QemuCocoaViewScanoutGL,
#ifdef USE_METAL
    QemuCocoaViewScanoutMetal,
#endif
};

/*
 ------------------------------------------------------
    QemuCocoaView
 ------------------------------------------------------
*/
@interface QemuCocoaView : NSView
{
    QEMUScreen screen;
    /* The state surrounding mouse grabbing is potentially confusing.
     * isAbsoluteEnabled tracks qemu_input_is_absolute() [ie "is the emulated
     *   pointing device an absolute-position one?"], but is only updated on
     *   next refresh.
     * isMouseGrabbed tracks whether GUI events are directed to the guest;
     *   it controls whether special keys like Cmd get sent to the guest,
     *   and whether we capture the mouse when in non-absolute mode.
     */
    BOOL isMouseGrabbed;
    BOOL isAbsoluteEnabled;
    /* isSelfResizing is set while resizeWindow changes the window frame
     *   itself so that windowDidResize does not report the resize back to
     *   the guest as a ui_info resolution request.
     */
    BOOL isSelfResizing;
    CFMachPortRef eventsTap;
    CGColorSpaceRef colorspace;
    QEMUCursor *cursor;
    int mouseX;
    int mouseY;
    bool mouseOn;
}

#ifdef CONFIG_OPENGL
@property (nonatomic,readonly) CALayer* glLayer;
#endif
@property (nonatomic,readonly) CALayer* cursorLayer;
@property (nonatomic,assign) QemuCocoaViewScanout scanout;

- (void) grabMouse;
- (void) ungrabMouse;
- (void) setFullGrab:(id)sender;
- (void) handleMonitorInput:(NSEvent *)event;
- (bool) handleEvent:(NSEvent *)event;
- (bool) handleEventLocked:(NSEvent *)event;
- (void) notifyMouseModeChange;
- (BOOL) isMouseGrabbed;
- (BOOL) isSelfResizing;
- (void) raiseAllKeys;
- (void) setScanout:(QemuCocoaViewScanout)scanout;

#ifdef USE_METAL
@property (nonatomic,readonly) CAMetalLayer* metalLayer;
@property (nonatomic,readonly) id<MTLCommandQueue> commandQueue;
@property (nonatomic,retain) id<MTLTexture> fbTexture;
@property (nonatomic,assign) MTLOrigin fbOrigin;
@property (nonatomic,assign) MTLSize fbSize;

- (void)scanoutMetalTexture:(id<MTLTexture>)metalTexture origin:(MTLOrigin)origin size:(MTLSize)size;
- (void)drawFrame;
#endif
@end

QemuCocoaView *cocoaView;

static CGEventRef handleTapEvent(CGEventTapProxy proxy, CGEventType type, CGEventRef cgEvent, void *userInfo)
{
    QemuCocoaView *view = userInfo;
    NSEvent *event = [NSEvent eventWithCGEvent:cgEvent];
    if ([view isMouseGrabbed] && [view handleEvent:event]) {
        COCOA_DEBUG("Global events tap: qemu handled the event, capturing!\n");
        return NULL;
    }
    COCOA_DEBUG("Global events tap: qemu did not handle the event, letting it through...\n");

    return cgEvent;
}

@implementation QemuCocoaView
- (id)initWithFrame:(NSRect)frameRect
#ifdef CONFIG_OPENGL
                cgl:(BOOL)cgl
#ifdef USE_METAL
          mtlDevice:(id<MTLDevice>)mtlDevice
#endif
#endif
{
    COCOA_DEBUG("QemuCocoaView: initWithFrame\n");

    self = [super initWithFrame:frameRect];
    if (self) {

        NSTrackingAreaOptions options = NSTrackingActiveInKeyWindow |
                                        NSTrackingMouseEnteredAndExited |
                                        NSTrackingMouseMoved |
                                        NSTrackingInVisibleRect;

        NSTrackingArea *trackingArea =
            [[NSTrackingArea alloc] initWithRect:CGRectZero
                                         options:options
                                           owner:self
                                        userInfo:nil];

        [self addTrackingArea:trackingArea];
        [trackingArea release];
        screen.width = frameRect.size.width;
        screen.height = frameRect.size.height;
        colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_14_0
        [self setClipsToBounds:YES];
#endif
        [self setWantsLayer:YES];
#ifdef USE_METAL
        if (mtlDevice) {
            _metalLayer = [[CAMetalLayer alloc] init];
            _metalLayer.device = mtlDevice;
            _metalLayer.framebufferOnly = NO;
            _metalLayer.autoresizingMask = kCALayerNotSizable;
            _commandQueue = [mtlDevice newCommandQueue];
        }
#endif
#ifdef CONFIG_OPENGL
        if (cgl) {
            _glLayer = [[QemuCGLLayer alloc] init];
        } else {
            _glLayer = [[CALayer alloc] init];
        }
        /*
         * Unlike a backing layer, a sublayer does not track the view's
         * geometry. The frame must be non-zero before the EGL window
         * surface is created on this layer, and is kept in sync with the
         * view in updateScale until a GL scanout takes ownership of it
         * (cocoa_gl_scanout_texture).
         */
        _glLayer.frame = self.bounds;
#endif
        _cursorLayer = [[CALayer alloc] init];
        [_cursorLayer setAnchorPoint:CGPointMake(0, 1)];
        [_cursorLayer setZPosition:1];
#ifdef CONFIG_OPENGL
        [[self layer] addSublayer:_glLayer];
#endif
        [[self layer] addSublayer:_cursorLayer];
    }
    return self;
}

- (void) dealloc
{
    COCOA_DEBUG("QemuCocoaView: dealloc\n");

    if (eventsTap) {
        CFRelease(eventsTap);
    }

    CGColorSpaceRelease(colorspace);
    [_cursorLayer release];
    cursor_unref(cursor);
#ifdef CONFIG_OPENGL
    [_glLayer release];
#endif
#ifdef USE_METAL
    [_fbTexture release];
    [_commandQueue release];
    [_metalLayer release];
#endif
    [super dealloc];
}

- (BOOL) isOpaque
{
    return YES;
}

#ifdef CONFIG_OPENGL
- (BOOL)wantsUpdateLayer
{
    return display_opengl;
}
#endif

- (void) viewDidMoveToWindow
{
    [self resizeWindow];
}

- (void) selectConsoleLocked:(unsigned int)index
{
    QemuConsole *con = qemu_console_lookup_by_index(index);
    if (!con) {
        return;
    }

    unregister_displaychangelistener(&dcl);
    qkbd_state_switch_console(kbd, con);
    dcl.con = con;
    register_displaychangelistener(&dcl);
    [self notifyMouseModeChange];
    [self updateUIInfo];
}

- (void) hideCursor
{
    if (!cursor_hide) {
        return;
    }
    [NSCursor hide];
}

- (void) unhideCursor
{
    if (!cursor_hide) {
        return;
    }
    [NSCursor unhide];
}

- (void)updateCursorLayout
{
    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    if (cursor) {
        CGFloat scale = [self bounds].size.width / screen.width;
        CGPoint position;
        CGRect bounds = CGRectZero;

        position.x = mouseX * scale;
        position.y = (screen.height - mouseY) * scale;

        bounds.size.width = cursor->width * scale;
        bounds.size.height = cursor->height * scale;

        [self.cursorLayer setBounds:bounds];
        [self.cursorLayer setContentsScale:scale];
        [self.cursorLayer setPosition:position];
    }

    [self.cursorLayer setHidden:!mouseOn];
    [CATransaction commit];
}

- (void)setMouseX:(int)x y:(int)y on:(bool)on
{
    mouseX = x;
    mouseY = y;
    mouseOn = on;
    [self updateCursorLayout];
}

- (void)setCursor:(QEMUCursor *)given_cursor
{
    CGDataProviderRef provider;
    CGImageRef image;

    cursor_unref(cursor);
    cursor = given_cursor;

    if (!cursor) {
        return;
    }

    cursor_ref(cursor);

    provider = CGDataProviderCreateWithData(
        NULL,
        cursor->data,
        cursor->width * cursor->height * 4,
        NULL
    );

    image = CGImageCreate(
        cursor->width, //width
        cursor->height, //height
        8, //bitsPerComponent
        32, //bitsPerPixel
        cursor->width * 4, //bytesPerRow
        colorspace, //colorspace
        kCGBitmapByteOrder32Little | kCGImageAlphaFirst, //bitmapInfo
        provider, //provider
        NULL, //decode
        0, //interpolate
        kCGRenderingIntentDefault //intent
    );

    CGDataProviderRelease(provider);
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [self.cursorLayer setContents:(id)image];
    [self updateCursorLayout];
    [CATransaction commit];
    CGImageRelease(image);
}

- (void) drawRect:(NSRect) rect
{
    COCOA_DEBUG("QemuCocoaView: drawRect\n");

    // get CoreGraphic context
    CGContextRef viewContextRef = [[NSGraphicsContext currentContext] CGContext];
    BQL_LOCK_GUARD();

    CGContextSetInterpolationQuality(viewContextRef,
                                     zoom_interpolation ? kCGInterpolationLow :
                                                          kCGInterpolationNone);
    CGContextSetShouldAntialias (viewContextRef, NO);

    // draw screen bitmap directly to Core Graphics context
    int w = surface_width(surface);
    int h = surface_height(surface);
    int bitsPerPixel = PIXMAN_FORMAT_BPP(surface_format(surface));
    int stride = surface_stride(surface);

    CGDataProviderRef dataProviderRef = CGDataProviderCreateWithData(
        NULL,
        surface_data(surface),
        stride * h,
        NULL
    );
    CGImageRef imageRef = CGImageCreate(
        w, //width
        h, //height
        DIV_ROUND_UP(bitsPerPixel, 8) * 2, //bitsPerComponent
        bitsPerPixel, //bitsPerPixel
        stride, //bytesPerRow
        colorspace, //colorspace
        kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst, //bitmapInfo
        dataProviderRef, //provider
        NULL, //decode
        0, //interpolate
        kCGRenderingIntentDefault //intent
    );
    // selective drawing code (draws only dirty rectangles) (OS X >= 10.4)
    const NSRect *rectList;
    NSInteger rectCount;
    int i;
    CGImageRef clipImageRef;
    CGRect clipRect;

    [self getRectsBeingDrawn:&rectList count:&rectCount];
    for (i = 0; i < rectCount; i++) {
        clipRect = rectList[i];
        clipRect.origin.y = (float)h - (clipRect.origin.y + clipRect.size.height);
        clipImageRef = CGImageCreateWithImageInRect(
                                                    imageRef,
                                                    clipRect
                                                    );
        CGContextDrawImage (viewContextRef, cgrect(rectList[i]), clipImageRef);
        CGImageRelease (clipImageRef);
    }
    CGImageRelease (imageRef);
    CGDataProviderRelease(dataProviderRef);
}

- (NSSize)fixAspectRatio:(NSSize)max
{
    NSSize scaled;
    NSSize fixed;

    scaled.width = screen.width * max.height;
    scaled.height = screen.height * max.width;

    /*
     * Here screen is our guest's output size, and max is the size of the
     * largest possible area of the screen we can display on.
     * We want to scale up (screen.width x screen.height) by either:
     *   1) max.height / screen.height
     *   2) max.width / screen.width
     * With the first scale factor the scale will result in an output height of
     * max.height (i.e. we will fill the whole height of the available screen
     * space and have black bars left and right) and with the second scale
     * factor the scaling will result in an output width of max.width (i.e. we
     * fill the whole width of the available screen space and have black bars
     * top and bottom). We need to pick whichever keeps the whole of the guest
     * output on the screen, which is to say the smaller of the two scale
     * factors.
     * To avoid doing more division than strictly necessary, instead of directly
     * comparing scale factors 1 and 2 we instead calculate and compare those
     * two scale factors multiplied by (screen.height * screen.width).
     */
    if (scaled.width < scaled.height) {
        fixed.width = scaled.width / screen.height;
        fixed.height = max.height;
    } else {
        fixed.width = max.width;
        fixed.height = scaled.height / screen.width;
    }

    return fixed;
}

- (NSSize) screenSafeAreaSize
{
    NSSize size = [[[self window] screen] frame].size;
    NSEdgeInsets insets = [[[self window] screen] safeAreaInsets];
    size.width -= insets.left + insets.right;
    size.height -= insets.top + insets.bottom;
    return size;
}

- (void) resizeWindow
{
    /* setContentSize: delivers windowDidResize synchronously */
    isSelfResizing = YES;

    [[self window] setContentAspectRatio:NSMakeSize(screen.width, screen.height)];

    if (!([[self window] styleMask] & NSWindowStyleMaskResizable)) {
        CGFloat width = screen.width / [[self window] backingScaleFactor];
        CGFloat height = screen.height / [[self window] backingScaleFactor];

        [[self window] setContentSize:NSMakeSize(width, height)];
        [[self window] center];
    } else if ([[self window] styleMask] & NSWindowStyleMaskFullScreen) {
        [[self window] setContentSize:[self fixAspectRatio:[self screenSafeAreaSize]]];
        [[self window] center];
    } else {
        [[self window] setContentSize:[self fixAspectRatio:[self frame].size]];
    }

    isSelfResizing = NO;
}

- (void) updateScale
{
    if (display_opengl) {
#ifdef CONFIG_OPENGL
        if (self.scanout != QemuCocoaViewScanoutGL) {
            [self.glLayer setFrame:[self.layer bounds]];
        }
        [self.glLayer setContentsScale:[[self window] backingScaleFactor]];
#ifdef USE_METAL
        [self.metalLayer setContentsScale:[[self window] backingScaleFactor]];
#endif
#endif
    } else {
        [self setBoundsSize:NSMakeSize(screen.width, screen.height)];
    }

    [self updateCursorLayout];
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

- (void) updateUIInfoLocked
{
    /* Must be called with the BQL, i.e. via updateUIInfo */
    NSSize frameSize;
    QemuUIInfo info;

    if (!qemu_console_is_graphic(dcl.con)) {
        return;
    }

    if ([self window]) {
        NSDictionary *description = [[[self window] screen] deviceDescription];
        CGDirectDisplayID display = [[description objectForKey:@"NSScreenNumber"] unsignedIntValue];
        NSSize screenSize = [[[self window] screen] frame].size;
        CGSize screenPhysicalSize = CGDisplayScreenSize(display);
        bool isFullscreen = ([[self window] styleMask] & NSWindowStyleMaskFullScreen) != 0;
        CVDisplayLinkRef displayLink;

        frameSize = isFullscreen ? [self screenSafeAreaSize] : [self frame].size;

        if (!CVDisplayLinkCreateWithCGDisplay(display, &displayLink)) {
            CVTime period = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(displayLink);
            CVDisplayLinkRelease(displayLink);
            if (!(period.flags & kCVTimeIsIndefinite)) {
                update_displaychangelistener(&dcl,
                                             1000 * period.timeValue / period.timeScale);
                info.refresh_rate = (int64_t)1000 * period.timeScale / period.timeValue;
            }
        }

        info.width_mm = frameSize.width / screenSize.width * screenPhysicalSize.width;
        info.height_mm = frameSize.height / screenSize.height * screenPhysicalSize.height;
    } else {
        frameSize = [self frame].size;
        info.width_mm = 0;
        info.height_mm = 0;
    }

    info.xoff = 0;
    info.yoff = 0;
    info.width = frameSize.width * [[self window] backingScaleFactor];
    info.height = frameSize.height * [[self window] backingScaleFactor];

    dpy_set_ui_info(dcl.con, &info, TRUE);
}

#pragma clang diagnostic pop

- (void) updateUIInfo
{
    if (!allow_events) {
        /*
         * Don't try to tell QEMU about UI information in the application
         * startup phase -- we haven't yet registered dcl with the QEMU UI
         * layer.
         * When cocoa_display_init() does register the dcl, the UI layer
         * will call cocoa_switch(), which will call updateUIInfo, so
         * we don't lose any information here.
         */
        return;
    }

    with_bql(^{
        [self updateUIInfoLocked];
    });
}

- (void) updateScreenWidth:(int)w height:(int)h
{
    COCOA_DEBUG("QemuCocoaView: updateScreenWidth:height:\n");

    if (w != screen.width || h != screen.height) {
        COCOA_DEBUG("updateScreenWidth:height: new size %d x %d\n", w, h);
        screen.width = w;
        screen.height = h;
        [self resizeWindow];
        [self updateScale];
    }
}

- (void) setFullGrab:(id)sender
{
    COCOA_DEBUG("QemuCocoaView: setFullGrab\n");

    CGEventMask mask = CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp) | CGEventMaskBit(kCGEventFlagsChanged);
    eventsTap = CGEventTapCreate(kCGHIDEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault,
                                 mask, handleTapEvent, self);
    if (!eventsTap) {
        warn_report("Could not create event tap, system key combos will not be captured.\n");
        return;
    } else {
        COCOA_DEBUG("Global events tap created! Will capture system key combos.\n");
    }

    CFRunLoopRef runLoop = CFRunLoopGetCurrent();
    if (!runLoop) {
        warn_report("Could not obtain current CF RunLoop, system key combos will not be captured.\n");
        return;
    }

    CFRunLoopSourceRef tapEventsSrc = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventsTap, 0);
    if (!tapEventsSrc ) {
        warn_report("Could not obtain current CF RunLoop, system key combos will not be captured.\n");
        return;
    }

    CFRunLoopAddSource(runLoop, tapEventsSrc, kCFRunLoopDefaultMode);
    CFRelease(tapEventsSrc);
}

- (void) toggleKey: (int)keycode {
    qkbd_state_key_event(kbd, keycode, !qkbd_state_key_get(kbd, keycode));
}

// Does the work of sending input to the monitor
- (void) handleMonitorInput:(NSEvent *)event
{
    int keysym = 0;
    int control_key = 0;

    // if the control key is down
    if ([event modifierFlags] & NSEventModifierFlagControl) {
        control_key = 1;
    }

    /* translates Macintosh keycodes to QEMU's keysym */

    static const int without_control_translation[] = {
        [0 ... 0xff] = 0,   // invalid key

        [kVK_UpArrow]       = QEMU_KEY_UP,
        [kVK_DownArrow]     = QEMU_KEY_DOWN,
        [kVK_RightArrow]    = QEMU_KEY_RIGHT,
        [kVK_LeftArrow]     = QEMU_KEY_LEFT,
        [kVK_Home]          = QEMU_KEY_HOME,
        [kVK_End]           = QEMU_KEY_END,
        [kVK_PageUp]        = QEMU_KEY_PAGEUP,
        [kVK_PageDown]      = QEMU_KEY_PAGEDOWN,
        [kVK_ForwardDelete] = QEMU_KEY_DELETE,
        [kVK_Delete]        = QEMU_KEY_BACKSPACE,
    };

    static const int with_control_translation[] = {
        [0 ... 0xff] = 0,   // invalid key

        [kVK_UpArrow]       = QEMU_KEY_CTRL_UP,
        [kVK_DownArrow]     = QEMU_KEY_CTRL_DOWN,
        [kVK_RightArrow]    = QEMU_KEY_CTRL_RIGHT,
        [kVK_LeftArrow]     = QEMU_KEY_CTRL_LEFT,
        [kVK_Home]          = QEMU_KEY_CTRL_HOME,
        [kVK_End]           = QEMU_KEY_CTRL_END,
        [kVK_PageUp]        = QEMU_KEY_CTRL_PAGEUP,
        [kVK_PageDown]      = QEMU_KEY_CTRL_PAGEDOWN,
    };

    if (control_key != 0) { /* If the control key is being used */
        if ([event keyCode] < ARRAY_SIZE(with_control_translation)) {
            keysym = with_control_translation[[event keyCode]];
        }
    } else {
        if ([event keyCode] < ARRAY_SIZE(without_control_translation)) {
            keysym = without_control_translation[[event keyCode]];
        }
    }

    // if not a key that needs translating
    if (keysym == 0) {
        NSString *ks = [event characters];
        if ([ks length] > 0) {
            keysym = [ks characterAtIndex:0];
        }
    }

    if (keysym) {
        QemuTextConsole *con = QEMU_TEXT_CONSOLE(dcl.con);
        qemu_text_console_put_keysym(con, keysym);
    }
}

- (bool) handleEvent:(NSEvent *)event
{
    return bool_with_bql(^{
        return [self handleEventLocked:event];
    });
}

- (bool) handleEventLocked:(NSEvent *)event
{
    /* Return true if we handled the event, false if it should be given to OSX */
    COCOA_DEBUG("QemuCocoaView: handleEvent\n");
    InputButton button;
    int keycode = 0;
    NSUInteger modifiers = [event modifierFlags];

    /*
     * Check -[NSEvent modifierFlags] here.
     *
     * There is a NSEventType for an event notifying the change of
     * -[NSEvent modifierFlags], NSEventTypeFlagsChanged but these operations
     * are performed for any events because a modifier state may change while
     * the application is inactive (i.e. no events fire) and we don't want to
     * wait for another modifier state change to detect such a change.
     *
     * NSEventModifierFlagCapsLock requires a special treatment. The other flags
     * are handled in similar manners.
     *
     * NSEventModifierFlagCapsLock
     * ---------------------------
     *
     * If CapsLock state is changed, "up" and "down" events will be fired in
     * sequence, effectively updates CapsLock state on the guest.
     *
     * The other flags
     * ---------------
     *
     * If a flag is not set, fire "up" events for all keys which correspond to
     * the flag. Note that "down" events are not fired here because the flags
     * checked here do not tell what exact keys are down.
     *
     * If one of the keys corresponding to a flag is down, we rely on
     * -[NSEvent keyCode] of an event whose -[NSEvent type] is
     * NSEventTypeFlagsChanged to know the exact key which is down, which has
     * the following two downsides:
     * - It does not work when the application is inactive as described above.
     * - It malfactions *after* the modifier state is changed while the
     *   application is inactive. It is because -[NSEvent keyCode] does not tell
     *   if the key is up or down, and requires to infer the current state from
     *   the previous state. It is still possible to fix such a malfanction by
     *   completely leaving your hands from the keyboard, which hopefully makes
     *   this implementation usable enough.
     */
    if (!!(modifiers & NSEventModifierFlagCapsLock) !=
        qkbd_state_modifier_get(kbd, QKBD_MOD_CAPSLOCK)) {
        qkbd_state_key_event(kbd, Q_KEY_CODE_CAPS_LOCK, true);
        qkbd_state_key_event(kbd, Q_KEY_CODE_CAPS_LOCK, false);
    }

    if (!(modifiers & NSEventModifierFlagShift)) {
        qkbd_state_key_event(kbd, Q_KEY_CODE_SHIFT, false);
        qkbd_state_key_event(kbd, Q_KEY_CODE_SHIFT_R, false);
    }
    if (!(modifiers & NSEventModifierFlagControl)) {
        qkbd_state_key_event(kbd, Q_KEY_CODE_CTRL, false);
        qkbd_state_key_event(kbd, Q_KEY_CODE_CTRL_R, false);
    }
    if (!(modifiers & NSEventModifierFlagOption)) {
        if (swap_opt_cmd) {
            qkbd_state_key_event(kbd, Q_KEY_CODE_META_L, false);
            qkbd_state_key_event(kbd, Q_KEY_CODE_META_R, false);
        } else {
            qkbd_state_key_event(kbd, Q_KEY_CODE_ALT, false);
            qkbd_state_key_event(kbd, Q_KEY_CODE_ALT_R, false);
        }
    }
    if (!(modifiers & NSEventModifierFlagCommand)) {
        if (swap_opt_cmd) {
            qkbd_state_key_event(kbd, Q_KEY_CODE_ALT, false);
            qkbd_state_key_event(kbd, Q_KEY_CODE_ALT_R, false);
        } else {
            qkbd_state_key_event(kbd, Q_KEY_CODE_META_L, false);
            qkbd_state_key_event(kbd, Q_KEY_CODE_META_R, false);
        }
    }

    switch ([event type]) {
        case NSEventTypeFlagsChanged:
            switch ([event keyCode]) {
                case kVK_Shift:
                    if (!!(modifiers & NSEventModifierFlagShift)) {
                        [self toggleKey:Q_KEY_CODE_SHIFT];
                    }
                    break;

                case kVK_RightShift:
                    if (!!(modifiers & NSEventModifierFlagShift)) {
                        [self toggleKey:Q_KEY_CODE_SHIFT_R];
                    }
                    break;

                case kVK_Control:
                    if (!!(modifiers & NSEventModifierFlagControl)) {
                        [self toggleKey:Q_KEY_CODE_CTRL];
                    }
                    break;

                case kVK_RightControl:
                    if (!!(modifiers & NSEventModifierFlagControl)) {
                        [self toggleKey:Q_KEY_CODE_CTRL_R];
                    }
                    break;

                case kVK_Option:
                    if (!!(modifiers & NSEventModifierFlagOption)) {
                        if (swap_opt_cmd) {
                            [self toggleKey:Q_KEY_CODE_META_L];
                        } else {
                            [self toggleKey:Q_KEY_CODE_ALT];
                        }
                    }
                    break;

                case kVK_RightOption:
                    if (!!(modifiers & NSEventModifierFlagOption)) {
                        if (swap_opt_cmd) {
                            [self toggleKey:Q_KEY_CODE_META_R];
                        } else {
                            [self toggleKey:Q_KEY_CODE_ALT_R];
                        }
                    }
                    break;

                /* Don't pass command key changes to guest unless mouse is grabbed */
                case kVK_Command:
                    if (isMouseGrabbed &&
                        !!(modifiers & NSEventModifierFlagCommand) &&
                        left_command_key_enabled) {
                        if (swap_opt_cmd) {
                            [self toggleKey:Q_KEY_CODE_ALT];
                        } else {
                            [self toggleKey:Q_KEY_CODE_META_L];
                        }
                    }
                    break;

                case kVK_RightCommand:
                    if (isMouseGrabbed &&
                        !!(modifiers & NSEventModifierFlagCommand)) {
                        if (swap_opt_cmd) {
                            [self toggleKey:Q_KEY_CODE_ALT_R];
                        } else {
                            [self toggleKey:Q_KEY_CODE_META_R];
                        }
                    }
                    break;
            }
            return true;
        case NSEventTypeKeyDown:
            keycode = cocoa_keycode_to_qemu([event keyCode]);

            // forward command key combos to the host UI unless the mouse is grabbed
            if (!isMouseGrabbed && ([event modifierFlags] & NSEventModifierFlagCommand)) {
                return false;
            }

            // default

            // handle control + alt Key Combos (ctrl+alt+[1..9,g] is reserved for QEMU)
            if (([event modifierFlags] & NSEventModifierFlagControl) && ([event modifierFlags] & NSEventModifierFlagOption)) {
                NSString *keychar = [event charactersIgnoringModifiers];
                if ([keychar length] == 1) {
                    char key = [keychar characterAtIndex:0];
                    switch (key) {

                        // enable graphic console
                        case '1' ... '9':
                            [self selectConsoleLocked:key - '0' - 1]; /* ascii math */
                            return true;

                        // release the mouse grab
                        case 'g':
                            [self ungrabMouse];
                            return true;
                    }
                }
            }

            if (qemu_console_is_graphic(dcl.con)) {
                qkbd_state_key_event(kbd, keycode, true);
            } else {
                [self handleMonitorInput: event];
            }
            return true;
        case NSEventTypeKeyUp:
            keycode = cocoa_keycode_to_qemu([event keyCode]);

            // don't pass the guest a spurious key-up if we treated this
            // command-key combo as a host UI action
            if (!isMouseGrabbed && ([event modifierFlags] & NSEventModifierFlagCommand)) {
                return true;
            }

            if (qemu_console_is_graphic(dcl.con)) {
                qkbd_state_key_event(kbd, keycode, false);
            }
            return true;
        case NSEventTypeScrollWheel:
            /*
             * Send wheel events to the guest regardless of window focus.
             * This is in-line with standard Mac OS X UI behaviour.
             */

            /* Determine if this is a scroll up or scroll down event */
            if ([event deltaY] != 0) {
                button = ([event deltaY] > 0) ?
                    INPUT_BUTTON_WHEEL_UP : INPUT_BUTTON_WHEEL_DOWN;
            } else if ([event deltaX] != 0) {
                button = ([event deltaX] > 0) ?
                    INPUT_BUTTON_WHEEL_LEFT : INPUT_BUTTON_WHEEL_RIGHT;
            } else {
                /*
                 * We shouldn't have got a scroll event when deltaY and delta Y
                 * are zero, hence no harm in dropping the event
                 */
                return true;
            }

            qemu_input_queue_btn(dcl.con, button, true);
            qemu_input_event_sync();
            qemu_input_queue_btn(dcl.con, button, false);
            qemu_input_event_sync();

            return true;
        default:
            return false;
    }
}

- (void) handleMouseEvent:(NSEvent *)event button:(InputButton)button down:(bool)down
{
    if (!isMouseGrabbed) {
        return;
    }

    with_bql(^{
        qemu_input_queue_btn(dcl.con, button, down);
    });

    [self handleMouseEvent:event];
}

- (void) handleMouseEvent:(NSEvent *)event
{
    if (!isMouseGrabbed) {
        return;
    }

    with_bql(^{
        if (isAbsoluteEnabled) {
            CGFloat d = (CGFloat)screen.height / [self frame].size.height;
            NSPoint p = [event locationInWindow];

            /* Note that the origin for Cocoa mouse coords is bottom left, not top left. */
            qemu_input_queue_abs(dcl.con, INPUT_AXIS_X, p.x * d, 0, screen.width);
            qemu_input_queue_abs(dcl.con, INPUT_AXIS_Y, screen.height - p.y * d, 0, screen.height);
        } else {
            qemu_input_queue_rel(dcl.con, INPUT_AXIS_X, [event deltaX]);
            qemu_input_queue_rel(dcl.con, INPUT_AXIS_Y, [event deltaY]);
        }

        qemu_input_event_sync();
    });
}

- (void) mouseExited:(NSEvent *)event
{
    if (isAbsoluteEnabled && isMouseGrabbed) {
        [self ungrabMouse];
    }
}

- (void) mouseEntered:(NSEvent *)event
{
    if (isAbsoluteEnabled && !isMouseGrabbed) {
        [self grabMouse];
    }
}

- (void) mouseMoved:(NSEvent *)event
{
    [self handleMouseEvent:event];
}

- (void) mouseDown:(NSEvent *)event
{
    [self handleMouseEvent:event button:INPUT_BUTTON_LEFT down:true];
}

- (void) rightMouseDown:(NSEvent *)event
{
    [self handleMouseEvent:event button:INPUT_BUTTON_RIGHT down:true];
}

- (void) otherMouseDown:(NSEvent *)event
{
    [self handleMouseEvent:event button:INPUT_BUTTON_MIDDLE down:true];
}

- (void) mouseDragged:(NSEvent *)event
{
    [self handleMouseEvent:event];
}

- (void) rightMouseDragged:(NSEvent *)event
{
    [self handleMouseEvent:event];
}

- (void) otherMouseDragged:(NSEvent *)event
{
    [self handleMouseEvent:event];
}

- (void) mouseUp:(NSEvent *)event
{
    if (!isMouseGrabbed) {
        [self grabMouse];
    }

    [self handleMouseEvent:event button:INPUT_BUTTON_LEFT down:false];
}

- (void) rightMouseUp:(NSEvent *)event
{
    [self handleMouseEvent:event button:INPUT_BUTTON_RIGHT down:false];
}

- (void) otherMouseUp:(NSEvent *)event
{
    [self handleMouseEvent:event button:INPUT_BUTTON_MIDDLE down:false];
}

- (void) grabMouse
{
    COCOA_DEBUG("QemuCocoaView: grabMouse\n");

    if (qemu_name)
        [[self window] setTitle:[NSString stringWithFormat:@"QEMU %s - (Press  " UC_CTRL_KEY " " UC_ALT_KEY " G  to release Mouse)", qemu_name]];
    else
        [[self window] setTitle:@"QEMU - (Press  " UC_CTRL_KEY " " UC_ALT_KEY " G  to release Mouse)"];
    [self hideCursor];
    CGAssociateMouseAndMouseCursorPosition(isAbsoluteEnabled);
    isMouseGrabbed = TRUE; // while isMouseGrabbed = TRUE, QemuCocoaApp sends all events to [cocoaView handleEvent:]
}

- (void) ungrabMouse
{
    COCOA_DEBUG("QemuCocoaView: ungrabMouse\n");

    if (qemu_name)
        [[self window] setTitle:[NSString stringWithFormat:@"QEMU %s", qemu_name]];
    else
        [[self window] setTitle:@"QEMU"];
    [self unhideCursor];
    CGAssociateMouseAndMouseCursorPosition(TRUE);
    isMouseGrabbed = FALSE;
    [self raiseAllButtons];
}

- (void) notifyMouseModeChange {
    bool tIsAbsoluteEnabled = bool_with_bql(^{
        return qemu_input_is_absolute(dcl.con);
    });

    if (tIsAbsoluteEnabled == isAbsoluteEnabled) {
        return;
    }

    isAbsoluteEnabled = tIsAbsoluteEnabled;

    if (isMouseGrabbed) {
        if (isAbsoluteEnabled) {
            [self ungrabMouse];
        } else {
            CGAssociateMouseAndMouseCursorPosition(isAbsoluteEnabled);
        }
    }
}
- (BOOL) isMouseGrabbed {return isMouseGrabbed;}
- (BOOL) isSelfResizing {return isSelfResizing;}

/*
 * Makes the target think all down keys are being released.
 * This prevents a stuck key problem, since we will not see
 * key up events for those keys after we have lost focus.
 */
- (void) raiseAllKeys
{
    with_bql(^{
        qkbd_state_lift_all_keys(kbd);
    });
}

- (void) raiseAllButtons
{
    with_bql(^{
        qemu_input_queue_btn(dcl.con, INPUT_BUTTON_LEFT, false);
        qemu_input_queue_btn(dcl.con, INPUT_BUTTON_RIGHT, false);
        qemu_input_queue_btn(dcl.con, INPUT_BUTTON_MIDDLE, false);
    });
}

- (void)setScanout:(QemuCocoaViewScanout)scanout
{
    COCOA_DEBUG("QemuCocoaView: setScanout:%d\n", scanout);
    if (_scanout != scanout) {
        _scanout = scanout;
        [self.cursorLayer removeFromSuperlayer];
#ifdef CONFIG_OPENGL
        [self.glLayer removeFromSuperlayer];
#endif
#ifdef USE_METAL
        [self.metalLayer removeFromSuperlayer];
        if (scanout != QemuCocoaViewScanoutMetal) {
            self.fbTexture = nil;
        }
#endif
        switch (scanout) {
#ifdef CONFIG_OPENGL
            case QemuCocoaViewScanoutNone:
                /* in GL mode, 2D surface content is presented via glLayer */
                self.glLayer.frame = self.layer.bounds;
                [self.layer addSublayer:self.glLayer];
                break;
            case QemuCocoaViewScanoutGL: [self.layer addSublayer:self.glLayer]; break;
#endif
#ifdef USE_METAL
            case QemuCocoaViewScanoutMetal: [self.layer addSublayer:self.metalLayer]; break;
#endif
            default: break;
        }
        [self.layer addSublayer:self.cursorLayer];
    }
}

#ifdef USE_METAL
- (void)scanoutMetalTexture:(id<MTLTexture>)metalTexture origin:(MTLOrigin)origin size:(MTLSize)size
{
    COCOA_DEBUG("QemuCocoaView: scanoutMetalTexture(%p) origin:(%d, %d) size:(%d, %d)\n",
                metalTexture, origin.x, origin.y, size.width, size.height);
    self.fbTexture = metalTexture;
    self.fbOrigin = origin;
    self.fbSize = size;

    if (metalTexture) {
        CGFloat scale = self.window.backingScaleFactor;
        CGRect frame = CGRectMake(0, 0, size.width / scale, size.height / scale);
        CGSize drawableSize = CGSizeMake(size.width, size.height);
        /*
         * A guest may re-emit SET_SCANOUT_BLOB for every scanned-out frame, so
         * this method can run at frame rate.  Assigning drawableSize or
         * pixelFormat discards the layer's drawable pool, which starves
         * nextDrawable in drawFrame, so only touch the layer when a parameter
         * actually changed.
         */
        if (self.metalLayer.contentsScale != scale ||
            self.metalLayer.pixelFormat != metalTexture.pixelFormat ||
            !CGRectEqualToRect(self.metalLayer.frame, frame) ||
            !CGSizeEqualToSize(self.metalLayer.drawableSize, drawableSize)) {
            self.metalLayer.contentsScale = scale;
            self.metalLayer.pixelFormat = metalTexture.pixelFormat;
            self.metalLayer.frame = frame;
            self.metalLayer.drawableSize = drawableSize;
        }
        self.scanout = QemuCocoaViewScanoutMetal;
    }
}

- (void)drawFrame
{
    @autoreleasepool {
        if (!self.fbTexture) {
            return;
        }
        id<CAMetalDrawable> drawable = [self.metalLayer nextDrawable];
        if (!drawable) {
            /*
             * The refresh that dispatched us already consumed gl_dirty, so
             * re-arm it to retry on the next tick.  Otherwise the frame is
             * dropped and the display keeps the stale image until the guest
             * flushes again.
             */
            gl_dirty = true;
            return;
        }

        id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];

        [blit copyFromTexture:self.fbTexture
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:self.fbOrigin
                   sourceSize:self.fbSize
                    toTexture:drawable.texture
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:(MTLOrigin){0,0,0}];

        [blit endEncoding];

        [commandBuffer presentDrawable:drawable];
        [commandBuffer commit];
    }
}
#endif

@end



/*
 ------------------------------------------------------
    QemuCocoaAppController
 ------------------------------------------------------
*/
@interface QemuCocoaAppController : NSObject
                                       <NSWindowDelegate, NSApplicationDelegate>
{
}
- (void)doToggleFullScreen:(id)sender;
- (void)showQEMUDoc:(id)sender;
- (void)zoomToFit:(id) sender;
- (void)displayConsole:(id)sender;
- (void)pauseQEMU:(id)sender;
- (void)resumeQEMU:(id)sender;
- (void)displayPause;
- (void)removePause;
- (void)restartQEMU:(id)sender;
- (void)powerDownQEMU:(id)sender;
- (void)ejectDeviceMedia:(id)sender;
- (void)changeDeviceMedia:(id)sender;
- (BOOL)verifyQuit;
- (void)openDocumentation:(NSString *)filename;
- (IBAction) do_about_menu_item: (id) sender;
- (void)adjustSpeed:(id)sender;
@end

@implementation QemuCocoaAppController
#ifdef CONFIG_OPENGL
- (id) initWithCGL:(BOOL)cgl
#ifdef USE_METAL
         mtlDevice:(id<MTLDevice>)mtlDevice
#endif
#else
- (id) init
#endif
{
    NSWindow *window;

    COCOA_DEBUG("QemuCocoaAppController: init\n");

    self = [super init];
    if (self) {
        NSRect frame = NSMakeRect(0.0, 0.0, 640.0, 480.0);

        // create a view and add it to the window
#ifdef CONFIG_OPENGL
#ifdef USE_METAL
        cocoaView = [[QemuCocoaView alloc] initWithFrame:frame cgl:cgl mtlDevice:mtlDevice];
#else
        cocoaView = [[QemuCocoaView alloc] initWithFrame:frame cgl:cgl];
#endif
#else
        cocoaView = [[QemuCocoaView alloc] initWithFrame:frame];
#endif
        if(!cocoaView) {
            error_report("(cocoa) can't create a view");
            exit(1);
        }

        // create a window
        window = [[NSWindow alloc] initWithContentRect:frame
            styleMask:NSWindowStyleMaskTitled|NSWindowStyleMaskMiniaturizable|NSWindowStyleMaskClosable
            backing:NSBackingStoreBuffered defer:NO];
        if(!window) {
            error_report("(cocoa) can't create window");
            exit(1);
        }
        [window setAcceptsMouseMovedEvents:YES];
        [window setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
        [window setTitle:qemu_name ? [NSString stringWithFormat:@"QEMU %s", qemu_name] : @"QEMU"];
        [window setContentView:cocoaView];
        [window makeKeyAndOrderFront:self];
        [window center];
        [window setDelegate: self];

        /* Used for displaying pause on the screen */
        pauseLabel = [NSTextField new];
        [pauseLabel setBezeled:YES];
        [pauseLabel setDrawsBackground:YES];
        [pauseLabel setBackgroundColor: [NSColor whiteColor]];
        [pauseLabel setEditable:NO];
        [pauseLabel setSelectable:NO];
        [pauseLabel setStringValue: @"Paused"];
        [pauseLabel setFont: [NSFont fontWithName: @"Helvetica" size: 90]];
        [pauseLabel setTextColor: [NSColor blackColor]];
        [pauseLabel sizeToFit];
    }
    return self;
}

- (void) dealloc
{
    COCOA_DEBUG("QemuCocoaAppController: dealloc\n");

    [cocoaView release];
    [cbowner release];
    cbowner = nil;

    [super dealloc];
}

- (void)applicationDidFinishLaunching: (NSNotification *) note
{
    COCOA_DEBUG("QemuCocoaAppController: applicationDidFinishLaunching\n");
    allow_events = true;
}

- (void)applicationWillTerminate:(NSNotification *)aNotification
{
    COCOA_DEBUG("QemuCocoaAppController: applicationWillTerminate\n");

    with_bql(^{
        shutdown_action = SHUTDOWN_ACTION_POWEROFF;
        qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);
    });

    /*
     * Sleep here, because returning will cause OSX to kill us
     * immediately; the QEMU main loop will handle the shutdown
     * request and terminate the process.
     */
    [NSThread sleepForTimeInterval:INFINITY];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication
{
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:
                                                         (NSApplication *)sender
{
    COCOA_DEBUG("QemuCocoaAppController: applicationShouldTerminate\n");
    return [self verifyQuit];
}

- (void)windowDidChangeScreen:(NSNotification *)notification
{
    [cocoaView updateUIInfo];
}

- (void)windowDidEnterFullScreen:(NSNotification *)notification
{
    [cocoaView grabMouse];
}

- (void)windowDidExitFullScreen:(NSNotification *)notification
{
    [cocoaView resizeWindow];
    [cocoaView ungrabMouse];
}

- (void)windowDidResize:(NSNotification *)notification
{
    [cocoaView updateScale];
    if (![cocoaView isSelfResizing]) {
        [cocoaView updateUIInfo];
    }
}

/* Called when the user clicks on a window's close button */
- (BOOL)windowShouldClose:(id)sender
{
    COCOA_DEBUG("QemuCocoaAppController: windowShouldClose\n");
    [NSApp terminate: sender];
    /* If the user allows the application to quit then the call to
     * NSApp terminate will never return. If we get here then the user
     * cancelled the quit, so we should return NO to not permit the
     * closing of this window.
     */
    return NO;
}

- (NSApplicationPresentationOptions) window:(NSWindow *)window
                                     willUseFullScreenPresentationOptions:(NSApplicationPresentationOptions)proposedOptions;

{
    return (proposedOptions & ~(NSApplicationPresentationAutoHideDock | NSApplicationPresentationAutoHideMenuBar)) |
           NSApplicationPresentationHideDock | NSApplicationPresentationHideMenuBar;
}

/*
 * Called when QEMU goes into the background. Note that
 * [-NSWindowDelegate windowDidResignKey:] is used here instead of
 * [-NSApplicationDelegate applicationWillResignActive:] because it cannot
 * detect that the window loses focus when the deck is clicked on macOS 13.2.1.
 */
- (void) windowDidResignKey: (NSNotification *)aNotification
{
    COCOA_DEBUG("%s\n", __func__);
    [cocoaView ungrabMouse];
    [cocoaView raiseAllKeys];
}

/* We abstract the method called by the Enter Fullscreen menu item
 * because Mac OS 10.7 and higher disables it. This is because of the
 * menu item's old selector's name toggleFullScreen:
 */
- (void) doToggleFullScreen:(id)sender
{
    [[cocoaView window] toggleFullScreen:sender];
}

- (void) setFullGrab:(id)sender
{
    COCOA_DEBUG("QemuCocoaAppController: setFullGrab\n");

    [cocoaView setFullGrab:sender];
}

/* Tries to find then open the specified filename */
- (void) openDocumentation: (NSString *) filename
{
    /* Where to look for local files */
    NSString *path_array[] = {@"../share/doc/qemu/", @"../doc/qemu/", @"docs/"};
    NSString *full_file_path;
    NSURL *full_file_url;

    /* iterate thru the possible paths until the file is found */
    int index;
    for (index = 0; index < ARRAY_SIZE(path_array); index++) {
        full_file_path = [[NSBundle mainBundle] executablePath];
        full_file_path = [full_file_path stringByDeletingLastPathComponent];
        full_file_path = [NSString stringWithFormat: @"%@/%@%@", full_file_path,
                          path_array[index], filename];
        full_file_url = [NSURL fileURLWithPath: full_file_path
                                   isDirectory: false];
        if ([[NSWorkspace sharedWorkspace] openURL: full_file_url] == YES) {
            return;
        }
    }

    /* If none of the paths opened a file */
    NSBeep();
    QEMU_Alert(@"Failed to open file");
}

- (void)showQEMUDoc:(id)sender
{
    COCOA_DEBUG("QemuCocoaAppController: showQEMUDoc\n");

    [self openDocumentation: @"index.html"];
}

/* Stretches video to fit host monitor size */
- (void)zoomToFit:(id) sender
{
    NSWindowStyleMask styleMask = [[cocoaView window] styleMask] ^ NSWindowStyleMaskResizable;

    [[cocoaView window] setStyleMask:styleMask];
    [sender setState:styleMask & NSWindowStyleMaskResizable ? NSControlStateValueOn : NSControlStateValueOff];
    [cocoaView resizeWindow];
}

- (void)toggleZoomInterpolation:(id) sender
{
    qatomic_set(&zoom_interpolation, !zoom_interpolation);
    [sender setState:zoom_interpolation ? NSControlStateValueOn :
                                          NSControlStateValueOff];
}

/* Displays the console on the screen */
- (void)displayConsole:(id)sender
{
    with_bql(^{
        [cocoaView selectConsoleLocked:[sender tag]];
    });
}

/* Pause the guest */
- (void)pauseQEMU:(id)sender
{
    with_bql(^{
        qmp_stop(NULL);
    });
    [sender setEnabled: NO];
    [[[sender menu] itemWithTitle: @"Resume"] setEnabled: YES];
    [self displayPause];
}

/* Resume running the guest operating system */
- (void)resumeQEMU:(id) sender
{
    with_bql(^{
        qmp_cont(NULL);
    });
    [sender setEnabled: NO];
    [[[sender menu] itemWithTitle: @"Pause"] setEnabled: YES];
    [self removePause];
}

/* Displays the word pause on the screen */
- (void)displayPause
{
    /* Coordinates have to be calculated each time because the window can change its size */
    int xCoord, yCoord, width, height;
    xCoord = ([cocoaView frame].size.width - [pauseLabel frame].size.width)/2;
    yCoord = [cocoaView frame].size.height - [pauseLabel frame].size.height - ([pauseLabel frame].size.height * .5);
    width = [pauseLabel frame].size.width;
    height = [pauseLabel frame].size.height;
    [pauseLabel setFrame: NSMakeRect(xCoord, yCoord, width, height)];
    [cocoaView addSubview: pauseLabel];
}

/* Removes the word pause from the screen */
- (void)removePause
{
    [pauseLabel removeFromSuperview];
}

/* Restarts QEMU */
- (void)restartQEMU:(id)sender
{
    with_bql(^{
        qmp_system_reset(NULL);
    });
}

/* Powers down QEMU */
- (void)powerDownQEMU:(id)sender
{
    with_bql(^{
        qmp_system_powerdown(NULL);
    });
}

/* Ejects the media.
 * Uses sender's tag to figure out the device to eject.
 */
- (void)ejectDeviceMedia:(id)sender
{
    NSString * drive;
    drive = [sender representedObject];
    if(drive == nil) {
        NSBeep();
        QEMU_Alert(@"Failed to find drive to eject!");
        return;
    }

    __block Error *err = NULL;
    with_bql(^{
        qmp_eject([drive cStringUsingEncoding: NSASCIIStringEncoding],
                  NULL, false, false, &err);
    });
    handleAnyDeviceErrors(err);
}

/* Displays a dialog box asking the user to select an image file to load.
 * Uses sender's represented object value to figure out which drive to use.
 */
- (void)changeDeviceMedia:(id)sender
{
    /* Find the drive name */
    NSString * drive;
    drive = [sender representedObject];
    if(drive == nil) {
        NSBeep();
        QEMU_Alert(@"Could not find drive!");
        return;
    }

    /* Display the file open dialog */
    NSOpenPanel * openPanel;
    openPanel = [NSOpenPanel openPanel];
    [openPanel setCanChooseFiles: YES];
    [openPanel setAllowsMultipleSelection: NO];
    if([openPanel runModal] == NSModalResponseOK) {
        NSString * file = [[[openPanel URLs] objectAtIndex: 0] path];
        if(file == nil) {
            NSBeep();
            QEMU_Alert(@"Failed to convert URL to file path!");
            return;
        }

        __block Error *err = NULL;
        with_bql(^{
            qmp_blockdev_change_medium([drive cStringUsingEncoding:
                                                  NSASCIIStringEncoding],
                                       NULL,
                                       [file cStringUsingEncoding:
                                                 NSASCIIStringEncoding],
                                       "raw",
                                       true, false,
                                       false, 0,
                                       false, 0,
                                       &err);
        });
        handleAnyDeviceErrors(err);
    }
}

/* Verifies if the user really wants to quit */
- (BOOL)verifyQuit
{
    NSAlert *alert = [NSAlert new];
    [alert autorelease];
    [alert setMessageText: @"Are you sure you want to quit QEMU?"];
    [alert addButtonWithTitle: @"Cancel"];
    [alert addButtonWithTitle: @"Quit"];
    if([alert runModal] == NSAlertSecondButtonReturn) {
        return YES;
    } else {
        return NO;
    }
}

/* The action method for the About menu item */
- (IBAction) do_about_menu_item: (id) sender
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    char *icon_path_c = get_relocated_path(CONFIG_QEMU_ICONDIR "/hicolor/512x512/apps/qemu.png");
    NSString *icon_path = [NSString stringWithUTF8String:icon_path_c];
    g_free(icon_path_c);
    NSImage *icon = [[NSImage alloc] initWithContentsOfFile:icon_path];
    NSString *version = @"QEMU emulator version " QEMU_FULL_VERSION;
    NSString *copyright = @QEMU_COPYRIGHT;
    NSDictionary *options;
    if (icon) {
        options = @{
            NSAboutPanelOptionApplicationIcon : icon,
            NSAboutPanelOptionApplicationVersion : version,
            @"Copyright" : copyright,
        };
        [icon release];
    } else {
        options = @{
            NSAboutPanelOptionApplicationVersion : version,
            @"Copyright" : copyright,
        };
    }
    [NSApp orderFrontStandardAboutPanelWithOptions:options];
    [pool release];
}

/* Used by the Speed menu items */
- (void)adjustSpeed:(id)sender
{
    int throttle_pct; /* throttle percentage */
    NSMenu *menu;

    menu = [sender menu];
    if (menu != nil)
    {
        /* Unselect the currently selected item */
        for (NSMenuItem *item in [menu itemArray]) {
            if (item.state == NSControlStateValueOn) {
                [item setState: NSControlStateValueOff];
                break;
            }
        }
    }

    // check the menu item
    [sender setState: NSControlStateValueOn];

    // get the throttle percentage
    throttle_pct = [sender tag];

    with_bql(^{
        cpu_throttle_set(throttle_pct);
    });
    COCOA_DEBUG("cpu throttling at %d%c\n", cpu_throttle_get_percentage(), '%');
}

@end

@interface QemuApplication : NSApplication
@end

@implementation QemuApplication
- (void)sendEvent:(NSEvent *)event
{
    COCOA_DEBUG("QemuApplication: sendEvent\n");
    if (![cocoaView handleEvent:event]) {
        [super sendEvent: event];
    }
}
@end

static void create_initial_menus(void)
{
    // Add menus
    NSMenu      *menu;
    NSMenuItem  *menuItem;

    [NSApp setMainMenu:[[NSMenu alloc] init]];
    [NSApp setServicesMenu:[[NSMenu alloc] initWithTitle:@"Services"]];

    // Application menu
    menu = [[NSMenu alloc] initWithTitle:@""];
    [menu addItemWithTitle:@"About QEMU" action:@selector(do_about_menu_item:) keyEquivalent:@""]; // About QEMU
    [menu addItem:[NSMenuItem separatorItem]]; //Separator
    menuItem = [menu addItemWithTitle:@"Services" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:[NSApp servicesMenu]];
    [menu addItem:[NSMenuItem separatorItem]];
    [menu addItemWithTitle:@"Hide QEMU" action:@selector(hide:) keyEquivalent:@"h"]; //Hide QEMU
    menuItem = (NSMenuItem *)[menu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"]; // Hide Others
    [menuItem setKeyEquivalentModifierMask:(NSEventModifierFlagOption|NSEventModifierFlagCommand)];
    [menu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""]; // Show All
    [menu addItem:[NSMenuItem separatorItem]]; //Separator
    [menu addItemWithTitle:@"Quit QEMU" action:@selector(terminate:) keyEquivalent:@"q"];
    menuItem = [[NSMenuItem alloc] initWithTitle:@"Apple" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];
    [NSApp performSelector:@selector(setAppleMenu:) withObject:menu]; // Workaround (this method is private since 10.4+)

    // Machine menu
    menu = [[NSMenu alloc] initWithTitle: @"Machine"];
    [menu setAutoenablesItems: NO];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle: @"Pause" action: @selector(pauseQEMU:) keyEquivalent: @""] autorelease]];
    menuItem = [[[NSMenuItem alloc] initWithTitle: @"Resume" action: @selector(resumeQEMU:) keyEquivalent: @""] autorelease];
    [menu addItem: menuItem];
    [menuItem setEnabled: NO];
    [menu addItem: [NSMenuItem separatorItem]];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle: @"Reset" action: @selector(restartQEMU:) keyEquivalent: @""] autorelease]];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle: @"Power Down" action: @selector(powerDownQEMU:) keyEquivalent: @""] autorelease]];
    menuItem = [[[NSMenuItem alloc] initWithTitle: @"Machine" action:nil keyEquivalent:@""] autorelease];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];

    // View menu
    menu = [[NSMenu alloc] initWithTitle:@"View"];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle:@"Enter Fullscreen" action:@selector(doToggleFullScreen:) keyEquivalent:@"f"] autorelease]]; // Fullscreen
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"Zoom To Fit" action:@selector(zoomToFit:) keyEquivalent:@""] autorelease];
    [menuItem setState: [[cocoaView window] styleMask] & NSWindowStyleMaskResizable ? NSControlStateValueOn : NSControlStateValueOff];
    [menu addItem: menuItem];
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"Zoom Interpolation" action:@selector(toggleZoomInterpolation:) keyEquivalent:@""] autorelease];
    [menuItem setState: zoom_interpolation ? NSControlStateValueOn : NSControlStateValueOff];
    [menu addItem: menuItem];
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"View" action:nil keyEquivalent:@""] autorelease];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];

    // Speed menu
    menu = [[NSMenu alloc] initWithTitle:@"Speed"];

    // Add the rest of the Speed menu items
    int p, percentage, throttle_pct;
    for (p = 10; p >= 0; p--)
    {
        percentage = p * 10 > 1 ? p * 10 : 1; // prevent a 0% menu item

        menuItem = [[[NSMenuItem alloc]
                   initWithTitle: [NSString stringWithFormat: @"%d%%", percentage] action:@selector(adjustSpeed:) keyEquivalent:@""] autorelease];

        if (percentage == 100) {
            [menuItem setState: NSControlStateValueOn];
        }

        /* Calculate the throttle percentage */
        throttle_pct = -1 * percentage + 100;

        [menuItem setTag: throttle_pct];
        [menu addItem: menuItem];
    }
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"Speed" action:nil keyEquivalent:@""] autorelease];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];

    // Window menu
    menu = [[NSMenu alloc] initWithTitle:@"Window"];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"] autorelease]]; // Miniaturize
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""] autorelease];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];
    [NSApp setWindowsMenu:menu];

    // Help menu
    menu = [[NSMenu alloc] initWithTitle:@"Help"];
    [menu addItem: [[[NSMenuItem alloc] initWithTitle:@"QEMU Documentation" action:@selector(showQEMUDoc:) keyEquivalent:@"?"] autorelease]]; // QEMU Help
    menuItem = [[[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""] autorelease];
    [menuItem setSubmenu:menu];
    [[NSApp mainMenu] addItem:menuItem];
}

/* Returns a name for a given console */
static NSString * getConsoleName(QemuConsole * console)
{
    g_autofree char *label = qemu_console_get_label(console);

    return [NSString stringWithUTF8String:label];
}

/* Add an entry to the View menu for each console */
static void add_console_menu_entries(void)
{
    NSMenu *menu;
    NSMenuItem *menuItem;
    int index = 0;

    menu = [[[NSApp mainMenu] itemWithTitle:@"View"] submenu];

    [menu addItem:[NSMenuItem separatorItem]];

    while (qemu_console_lookup_by_index(index) != NULL) {
        menuItem = [[[NSMenuItem alloc] initWithTitle: getConsoleName(qemu_console_lookup_by_index(index))
                                               action: @selector(displayConsole:) keyEquivalent: @""] autorelease];
        [menuItem setTag: index];
        [menu addItem: menuItem];
        index++;
    }
}

/* Make menu items for all removable devices.
 * Each device is given an 'Eject' and 'Change' menu item.
 */
static void addRemovableDevicesMenuItems(void)
{
    NSMenu *menu;
    NSMenuItem *menuItem;
    BlockInfoList *currentDevice, *pointerToFree;
    NSString *deviceName;

    currentDevice = qmp_query_block(NULL);
    pointerToFree = currentDevice;

    menu = [[[NSApp mainMenu] itemWithTitle:@"Machine"] submenu];

    // Add a separator between related groups of menu items
    [menu addItem:[NSMenuItem separatorItem]];

    // Set the attributes to the "Removable Media" menu item
    NSString *titleString = @"Removable Media";
    NSMutableAttributedString *attString=[[NSMutableAttributedString alloc] initWithString:titleString];
    NSColor *newColor = [NSColor blackColor];
    NSFontManager *fontManager = [NSFontManager sharedFontManager];
    NSFont *font = [fontManager fontWithFamily:@"Helvetica"
                                          traits:NSBoldFontMask|NSItalicFontMask
                                          weight:0
                                            size:14];
    [attString addAttribute:NSFontAttributeName value:font range:NSMakeRange(0, [titleString length])];
    [attString addAttribute:NSForegroundColorAttributeName value:newColor range:NSMakeRange(0, [titleString length])];
    [attString addAttribute:NSUnderlineStyleAttributeName value:[NSNumber numberWithInt: 1] range:NSMakeRange(0, [titleString length])];

    // Add the "Removable Media" menu item
    menuItem = [NSMenuItem new];
    [menuItem setAttributedTitle: attString];
    [menuItem setEnabled: NO];
    [menu addItem: menuItem];

    /* Loop through all the block devices in the emulator */
    while (currentDevice) {
        deviceName = [[NSString stringWithFormat: @"%s", currentDevice->value->device] retain];

        if(currentDevice->value->removable) {
            menuItem = [[NSMenuItem alloc] initWithTitle: [NSString stringWithFormat: @"Change %s...", currentDevice->value->device]
                                                  action: @selector(changeDeviceMedia:)
                                           keyEquivalent: @""];
            [menu addItem: menuItem];
            [menuItem setRepresentedObject: deviceName];
            [menuItem autorelease];

            menuItem = [[NSMenuItem alloc] initWithTitle: [NSString stringWithFormat: @"Eject %s", currentDevice->value->device]
                                                  action: @selector(ejectDeviceMedia:)
                                           keyEquivalent: @""];
            [menu addItem: menuItem];
            [menuItem setRepresentedObject: deviceName];
            [menuItem autorelease];
        }
        currentDevice = currentDevice->next;
    }
    qapi_free_BlockInfoList(pointerToFree);
}

static void cocoa_mouse_mode_change_notify(Notifier *notifier, void *data)
{
    dispatch_async(dispatch_get_main_queue(), ^{
        [cocoaView notifyMouseModeChange];
    });
}

static Notifier mouse_mode_change_notifier = {
    .notify = cocoa_mouse_mode_change_notify
};

@interface QemuCocoaPasteboardTypeOwner : NSObject<NSPasteboardTypeOwner>
@end

@implementation QemuCocoaPasteboardTypeOwner

- (void)pasteboard:(NSPasteboard *)sender provideDataForType:(NSPasteboardType)type
{
    if (type != NSPasteboardTypeString) {
        return;
    }

    with_bql(^{
        QemuClipboardInfo *info = qemu_clipboard_info_ref(cbinfo);
        qemu_event_reset(&cbevent);
        qemu_clipboard_request(info, QEMU_CLIPBOARD_TYPE_TEXT);

        while (info == cbinfo &&
               info->types[QEMU_CLIPBOARD_TYPE_TEXT].available &&
               info->types[QEMU_CLIPBOARD_TYPE_TEXT].data == NULL) {
            bql_unlock();
            qemu_event_wait(&cbevent);
            bql_lock();
        }

        if (info == cbinfo) {
            NSData *data = [[NSData alloc] initWithBytes:info->types[QEMU_CLIPBOARD_TYPE_TEXT].data
                                           length:info->types[QEMU_CLIPBOARD_TYPE_TEXT].size];
            [sender setData:data forType:NSPasteboardTypeString];
            [data release];
        }

        qemu_clipboard_info_unref(info);
    });
}

@end

static void cocoa_clipboard_notify(Notifier *notifier, void *data);
static void cocoa_clipboard_request(QemuClipboardInfo *info,
                                    QemuClipboardType type);

static QemuClipboardPeer cbpeer = {
    .name = "cocoa",
    .notifier = { .notify = cocoa_clipboard_notify },
    .request = cocoa_clipboard_request
};

static void cocoa_clipboard_update_info(QemuClipboardInfo *info)
{
    if (info->owner == &cbpeer || info->selection != QEMU_CLIPBOARD_SELECTION_CLIPBOARD) {
        return;
    }

    if (info != cbinfo) {
        NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
        qemu_clipboard_info_unref(cbinfo);
        cbinfo = qemu_clipboard_info_ref(info);
        cbchangecount = [[NSPasteboard generalPasteboard] declareTypes:@[NSPasteboardTypeString] owner:cbowner];
        [pool release];
    }

    qemu_event_set(&cbevent);
}

static void cocoa_clipboard_notify(Notifier *notifier, void *data)
{
    QemuClipboardNotify *notify = data;

    switch (notify->type) {
    case QEMU_CLIPBOARD_UPDATE_INFO:
        cocoa_clipboard_update_info(notify->info);
        return;
    case QEMU_CLIPBOARD_RESET_SERIAL:
        /* ignore */
        return;
    }
}

static void cocoa_clipboard_request(QemuClipboardInfo *info,
                                    QemuClipboardType type)
{
    NSAutoreleasePool *pool;
    NSData *text;

    switch (type) {
    case QEMU_CLIPBOARD_TYPE_TEXT:
        pool = [[NSAutoreleasePool alloc] init];
        text = [[NSPasteboard generalPasteboard] dataForType:NSPasteboardTypeString];
        if (text) {
            qemu_clipboard_set_data(&cbpeer, info, type,
                                    [text length], [text bytes], true);
        }
        [pool release];
        break;
    default:
        break;
    }
}

static int cocoa_main(void)
{
    COCOA_DEBUG("Main thread: entering OSX run loop\n");
    [NSApp run];
    COCOA_DEBUG("Main thread: left OSX run loop, which should never happen\n");

    abort();
}



#pragma mark qemu
static void cocoa_update(DisplayChangeListener *dcl,
                         int x, int y, int w, int h)
{
    NSRect rect = NSMakeRect(x, surface_height(surface) - y - h, w, h);

    COCOA_DEBUG("qemu_cocoa: cocoa_update\n");

    dispatch_async(dispatch_get_main_queue(), ^{
        [cocoaView setNeedsDisplayInRect:rect];
    });
}

static void cocoa_switch(DisplayChangeListener *dcl,
                         DisplaySurface *new_surface)
{
    COCOA_DEBUG("qemu_cocoa: cocoa_switch\n");

    surface = new_surface;

    dispatch_async(dispatch_get_main_queue(), ^{
        BQL_LOCK_GUARD();
        int w = surface_width(surface);
        int h = surface_height(surface);

        [cocoaView updateScreenWidth:w height:h];
    });
}

static void cocoa_refresh(DisplayChangeListener *dcl)
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    COCOA_DEBUG("qemu_cocoa: cocoa_refresh\n");
    graphic_hw_update(dcl->con);

    if (cbchangecount != [[NSPasteboard generalPasteboard] changeCount]) {
        qemu_clipboard_info_unref(cbinfo);
        cbinfo = qemu_clipboard_info_new(&cbpeer, QEMU_CLIPBOARD_SELECTION_CLIPBOARD);
        if ([[NSPasteboard generalPasteboard] availableTypeFromArray:@[NSPasteboardTypeString]]) {
            cbinfo->types[QEMU_CLIPBOARD_TYPE_TEXT].available = true;
        }
        qemu_clipboard_update(cbinfo);
        cbchangecount = [[NSPasteboard generalPasteboard] changeCount];
        qemu_event_set(&cbevent);
    }

    [pool release];
}

static void cocoa_mouse_set(DisplayChangeListener *dcl, int x, int y, bool on)
{
    dispatch_async(dispatch_get_main_queue(), ^{
        [cocoaView setMouseX:x y:y on:on];
    });
}

static void cocoa_cursor_define(DisplayChangeListener *dcl, QEMUCursor *cursor)
{
    dispatch_async(dispatch_get_main_queue(), ^{
        BQL_LOCK_GUARD();
        [cocoaView setCursor:qemu_console_get_cursor(dcl->con)];
    });
}

static const DisplayChangeListenerOps dcl_ops = {
    .dpy_name          = "cocoa",
    .dpy_gfx_update = cocoa_update,
    .dpy_gfx_switch = cocoa_switch,
    .dpy_refresh = cocoa_refresh,
    .dpy_mouse_set = cocoa_mouse_set,
    .dpy_cursor_define = cocoa_cursor_define,
};

#ifdef CONFIG_OPENGL

static void with_gl_view_ctx(CodeBlock block)
{
#ifdef CONFIG_EGL
    if (egl_surface) {
        eglMakeCurrent(qemu_egl_display, egl_surface,
                       egl_surface, gl_view_ctx);
        block();
        eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE,
                       EGL_NO_SURFACE, EGL_NO_CONTEXT);
        return;
    }
#endif

    CGLSetCurrentContext((CGLContextObj)gl_view_ctx);
    block();
    CGLSetCurrentContext(NULL);
}

static CGLPixelFormatObj cocoa_gl_create_cgl_pixel_format(int bpp)
{
    CGLPixelFormatObj pix;
    GLint npix;
    CGLPixelFormatAttribute attribs[] = {
        kCGLPFAOpenGLProfile,
        (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFAColorSize,
        bpp,
        kCGLPFADoubleBuffer,
        0,
    };

    CGLChoosePixelFormat(attribs, &pix, &npix);

    return pix;
}

static int cocoa_gl_make_context_current(DisplayGLCtx *dgc, QEMUGLContext ctx)
{
#ifdef CONFIG_EGL
    if (egl_surface) {
        EGLSurface current_surface = ctx == EGL_NO_CONTEXT ? EGL_NO_SURFACE : egl_surface;
        return !eglMakeCurrent(qemu_egl_display, current_surface, current_surface, ctx);
    }
#endif

    return CGLSetCurrentContext((CGLContextObj)ctx);
}

static QEMUGLContext cocoa_gl_create_context(DisplayGLCtx *dgc,
                                             QEMUGLParams *params)
{
    CGLPixelFormatObj format;
    CGLContextObj ctx;
    int bpp;

#ifdef CONFIG_EGL
    if (egl_surface) {
        eglMakeCurrent(qemu_egl_display, egl_surface,
                       egl_surface, gl_view_ctx);
        return qemu_egl_create_context(dgc, params);
    }
#endif

    bpp = PIXMAN_FORMAT_BPP(surface_format(surface));
    format = cocoa_gl_create_cgl_pixel_format(bpp);
    CGLCreateContext(format, gl_view_ctx, &ctx);
    CGLDestroyPixelFormat(format);

    return (QEMUGLContext)ctx;
}

static void cocoa_gl_destroy_context(DisplayGLCtx *dgc, QEMUGLContext ctx)
{
#ifdef CONFIG_EGL
    if (egl_surface) {
        eglDestroyContext(qemu_egl_display, ctx);
        return;
    }
#endif

    CGLDestroyContext(ctx);
}

static void cocoa_gl_update(DisplayChangeListener *dcl,
                            int x, int y, int w, int h)
{
    with_gl_view_ctx(^{
        surface_gl_update_texture(dgc.gls, surface, x, y, w, h);
        gl_dirty = true;
    });
}

static void cocoa_gl_switch(DisplayChangeListener *dcl,
                            DisplaySurface *new_surface)
{
    with_gl_view_ctx(^{
        surface_gl_destroy_texture(dgc.gls, surface);
        surface_gl_create_texture(dgc.gls, new_surface);
    });

    cocoa_switch(dcl, new_surface);
    gl_dirty = true;
}

static void cocoa_gl_render(void)
{
    NSSize size = [cocoaView convertSizeToBacking:[cocoaView frame].size];
    GLint filter = qatomic_read(&zoom_interpolation) ? GL_LINEAR : GL_NEAREST;

    glViewport(0, 0, size.width, size.height);

    if (gl_scanout_id) {
        glBindFramebuffer(GL_FRAMEBUFFER_EXT, 0);
        glBindTexture(GL_TEXTURE_2D, gl_scanout_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        qemu_gl_run_texture_blit(dgc.gls, gl_scanout_y0_top);
    } else {
        glBindTexture(GL_TEXTURE_2D, surface->texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        surface_gl_render_texture(dgc.gls, surface);
    }
}

static void cocoa_gl_refresh(DisplayChangeListener *dcl)
{
    cocoa_refresh(dcl);

    if (gl_dirty) {
        gl_dirty = false;

#ifdef CONFIG_EGL
#ifdef USE_METAL
        if (cocoaView.scanout == QemuCocoaViewScanoutMetal) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [cocoaView drawFrame];
            });
        } else
#endif
        if (egl_surface) {
            with_gl_view_ctx(^{
                cocoa_gl_render();
                eglSwapBuffers(qemu_egl_display, egl_surface);
            });

            return;
        }
#endif

        dispatch_async(dispatch_get_main_queue(), ^{
            /* the CGL drawable is glLayer, not the view's backing layer */
            [cocoaView.glLayer setNeedsDisplay];
        });
    }
}

static void cocoa_gl_scanout_disable(DisplayChangeListener *dcl)
{
    gl_scanout_id = 0;
    gl_dirty = true;
    dispatch_async(dispatch_get_main_queue(), ^{
        cocoaView.scanout = QemuCocoaViewScanoutNone;
    });
}

static void cocoa_gl_scanout_texture(DisplayChangeListener *dcl,
                                     uint32_t backing_id,
                                     bool backing_y_0_top,
                                     uint32_t backing_width,
                                     uint32_t backing_height,
                                     uint32_t x, uint32_t y,
                                     uint32_t w, uint32_t h,
                                     ScanoutTextureNative native)
{
    gl_scanout_id = backing_id;
    gl_scanout_y0_top = backing_y_0_top;
    gl_dirty = true;
#ifdef USE_METAL
    if (native.type == SCANOUT_TEXTURE_NATIVE_TYPE_METAL) {
        id<MTLTexture> mtlTexture = [(id<MTLTexture>)native.handle retain];
        MTLOrigin mtlOrigin = MTLOriginMake(x, y, 0);
        MTLSize mtlSize = MTLSizeMake(w, h, 1);
        dispatch_async(dispatch_get_main_queue(), ^{
            [cocoaView scanoutMetalTexture:mtlTexture origin:mtlOrigin size:mtlSize];
            [mtlTexture release];
        });
    } else
#endif
    {
        dispatch_async(dispatch_get_main_queue(), ^{
            CGFloat scale = cocoaView.window.backingScaleFactor;
            cocoaView.glLayer.contentsScale = scale;
            cocoaView.glLayer.frame = CGRectMake(0, 0, w / scale, h / scale);
            cocoaView.scanout = QemuCocoaViewScanoutGL;
        });
    }
}

static void cocoa_gl_scanout_flush(DisplayChangeListener *dcl,
                                   uint32_t x, uint32_t y,
                                   uint32_t w, uint32_t h)
{
    gl_dirty = true;
}

static const DisplayChangeListenerOps dcl_gl_ops = {
    .dpy_name               = "cocoa-gl",
    .dpy_gfx_update         = cocoa_gl_update,
    .dpy_gfx_switch         = cocoa_gl_switch,
    .dpy_gfx_check_format   = console_gl_check_format,
    .dpy_refresh            = cocoa_gl_refresh,
    .dpy_mouse_set          = cocoa_mouse_set,
    .dpy_cursor_define      = cocoa_cursor_define,

    .dpy_gl_scanout_disable = cocoa_gl_scanout_disable,
    .dpy_gl_scanout_texture = cocoa_gl_scanout_texture,
    .dpy_gl_update          = cocoa_gl_scanout_flush,
};

static bool cocoa_gl_is_compatible_dcl(DisplayGLCtx *dgc,
                                       DisplayChangeListener *dcl)
{
    return dcl->ops == &dcl_gl_ops;
}

#endif

static void cocoa_display_early_init(DisplayOptions *o)
{
    assert(o->type == DISPLAY_TYPE_COCOA);
    if (o->has_gl && o->gl) {
        display_opengl = 1;
    }
}

static void cocoa_display_init(DisplayState *ds, DisplayOptions *opts)
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    QemuCocoaAppController *controller;

    COCOA_DEBUG("qemu_cocoa: cocoa_display_init\n");

    // Pull this console process up to being a fully-fledged graphical
    // app with a menubar and Dock icon
    ProcessSerialNumber psn = { 0, kCurrentProcess };
    TransformProcessType(&psn, kProcessTransformToForegroundApplication);

    [QemuApplication sharedApplication];

    dcl.con = qemu_console_lookup_default();
    kbd = qkbd_state_init(dcl.con);
    surface = qemu_console_surface(dcl.con);

#if defined(CONFIG_OPENGL) && defined(CONFIG_EGL)
    if (display_opengl && opts->gl == DISPLAY_GL_MODE_ES) {
        if (qemu_egl_init_dpy_cocoa(DISPLAY_GL_MODE_ES)) {
            exit(1);
        }
    }
#endif

    // Create an Application controller
#ifdef CONFIG_OPENGL
    controller = [[QemuCocoaAppController alloc] initWithCGL:display_opengl &&
                                                             opts->gl != DISPLAY_GL_MODE_ES
#ifdef USE_METAL
                                                   mtlDevice:(id<MTLDevice>)qemu_egl_angle_native_device
#endif
                 ];
#else
    controller = [[QemuCocoaAppController alloc] init];
#endif
    [NSApp setDelegate:controller];

    if (display_opengl) {
#ifdef CONFIG_OPENGL
        if (opts->gl == DISPLAY_GL_MODE_ES) {
#ifdef CONFIG_EGL
            gl_view_ctx = qemu_egl_init_ctx();
            if (!gl_view_ctx) {
                exit(1);
            }
            egl_surface = qemu_egl_init_surface(gl_view_ctx, cocoaView.glLayer);
            if (!egl_surface) {
                exit(1);
            }
#else
            error_report("OpenGLES without EGL is not supported - exiting");
            exit(1);
#endif
        } else {
            CGLPixelFormatObj format = cocoa_gl_create_cgl_pixel_format(32);
            CGLContextObj ctx;
            CGLCreateContext(format, NULL, &ctx);
            CGLDestroyPixelFormat(format);
            gl_view_ctx = (QEMUGLContext)ctx;
#ifdef CONFIG_EGL
            egl_surface = EGL_NO_SURFACE;
#endif
            cocoa_gl_make_context_current(&dgc, gl_view_ctx);
        }

        dgc.gls = qemu_gl_init_shader();
        dcl.ops = &dcl_gl_ops;

        for (unsigned int index = 0; ; index++) {
            QemuConsole *con = qemu_console_lookup_by_index(index);
            if (!con) {
                break;
            }

            qemu_console_set_display_gl_ctx(con, &dgc);
        }
#else
        error_report("OpenGL is not enabled - exiting");
        exit(1);
#endif
    } else {
        dcl.ops = &dcl_ops;
    }

    /* if fullscreen mode is to be used */
    if (opts->has_full_screen && opts->full_screen) {
        [[cocoaView window] toggleFullScreen: nil];
    }
    if (opts->u.cocoa.has_full_grab && opts->u.cocoa.full_grab) {
        [controller setFullGrab: nil];
    }

    if (opts->has_show_cursor && opts->show_cursor) {
        cursor_hide = 0;
    }
    if (opts->u.cocoa.has_swap_opt_cmd) {
        swap_opt_cmd = opts->u.cocoa.swap_opt_cmd;
    }

    if (opts->u.cocoa.has_left_command_key && !opts->u.cocoa.left_command_key) {
        left_command_key_enabled = 0;
    }

    if (opts->u.cocoa.has_zoom_to_fit && opts->u.cocoa.zoom_to_fit) {
        [cocoaView window].styleMask |= NSWindowStyleMaskResizable;
    }

    zoom_interpolation = opts->u.cocoa.has_zoom_interpolation &&
                         opts->u.cocoa.zoom_interpolation;

    create_initial_menus();
    /*
     * Create the menu entries which depend on QEMU state (for consoles
     * and removable devices). These make calls back into QEMU functions,
     * which is OK because at this point we know that the second thread
     * holds the BQL and is synchronously waiting for us to
     * finish.
     */
    add_console_menu_entries();
    addRemovableDevicesMenuItems();

    // register vga output callbacks
    register_displaychangelistener(&dcl);
    qemu_add_mouse_mode_change_notifier(&mouse_mode_change_notifier);
    [cocoaView notifyMouseModeChange];
    [cocoaView updateUIInfo];

    qemu_event_init(&cbevent, false);
    cbowner = [[QemuCocoaPasteboardTypeOwner alloc] init];
    qemu_clipboard_peer_register(&cbpeer);

    [pool release];

    /*
     * The Cocoa UI will run the NSApplication runloop on the main thread
     * rather than the default Core Foundation one.
     */
    qemu_main = cocoa_main;
}

static QemuDisplay qemu_display_cocoa = {
    .type       = DISPLAY_TYPE_COCOA,
    .early_init = cocoa_display_early_init,
    .init       = cocoa_display_init,
};

static void register_cocoa(void)
{
    qemu_display_register(&qemu_display_cocoa);
}

type_init(register_cocoa);

#ifdef CONFIG_OPENGL
module_dep("ui-opengl");
#endif
