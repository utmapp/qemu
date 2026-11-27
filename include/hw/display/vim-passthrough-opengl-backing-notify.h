/* Local/private research extension, NOT part of the stable stream ABI or an
 * upstream contribution. No process pointer ever crosses the guest wire.
 *
 * An accepted subscription retains the exact namespaced GPU ticket through
 * ONE callback, independent of producer/stream destruction. subscribe returns
 * without waiting for GPU completion. A callback may precede its return; the
 * caller must publish/retain its opaque owner before subscribing. status==0
 * means verified native completion; any other status grants NO CPU ownership.
 * Returns an errno value; a rejected subscription never invokes its callback.
 * There is at most one accepted subscription per ticket, no cancellation ACK
 * and no implicit ticket release. Reset owners must suppress stale publication
 * but remain alive until callback. The loaded library must outlive callbacks.
 */
#ifndef VIM_PASSTHROUGH_OPENGL_BACKING_NOTIFY_H
#define VIM_PASSTHROUGH_OPENGL_BACKING_NOTIFY_H
#include <stdint.h>
#define VIM_GL_BACKING_NOTIFY_ABI 2u
typedef void (*VIMGLBackingNotifyCallback)(void *opaque, int status);
struct VIMGLBackingNotifyAPI {
    uint32_t version, bytes;
    int (*subscribe)(const uint64_t namespace_id[2], uint64_t ticket,
                     VIMGLBackingNotifyCallback callback, void *opaque);
    /* Reset cleanup ONLY after real completion. Returns EBUSY if pending;
     * failure never licenses unpinning or retirement. This removes the
     * registry ownership, not a callback's independently retained reference.
     * A normal live guest releases through its existing service protocol. */
    int (*release_completed)(const uint64_t namespace_id[2], uint64_t ticket);
};
typedef const struct VIMGLBackingNotifyAPI *(*VIMGLGetBackingNotifyAPI)(uint32_t, uint32_t);
#endif
