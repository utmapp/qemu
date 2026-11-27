/* Local/private research host-executor ABI, not an upstream contribution or
 * a guest pointer/wire interface.
 * One opaque endpoint owns one serialized native interpreter; execution can
 * use its worker or the already-unlocked caller. Caller serializes
 * close after its I/O completes; shutdown is the concurrent cancellation API.
 * Current executors return from close only after accepted native GPU uses
 * have drained and the interpreter retired. Shutdown alone makes no such
 * guarantee. This stronger close behavior must not be inferred for an old
 * ABI-1 library. ABI 2 makes this guarantee mandatory for BOTH implementations.
 * The loaded library must outlive all its endpoints.
 */
#ifndef VIM_PASSTHROUGH_OPENGL_INPROCESS_H
#define VIM_PASSTHROUGH_OPENGL_INPROCESS_H
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <errno.h>
#include <limits.h>
#define VIM_GL_INPROCESS_ABI 2u
struct VIMGLInprocessAPI {
    uint32_t version,bytes;
    void *(*open)(void);
    ssize_t (*read)(void *,void *,size_t);
    ssize_t (*write)(void *,const void *,size_t);
    ssize_t (*readv)(void *,const struct iovec *,int);
    ssize_t (*writev)(void *,const struct iovec *,int);
    int (*shutdown)(void *);
    int (*close)(void *);
    unsigned (*live_count)(void); /* Diagnostic only. */
};
typedef const struct VIMGLInprocessAPI *(*VIMGLGetInprocessAPI)(uint32_t,uint32_t);
/* For a protocol consumer that already knows the exact reply span. Keep
 * partial ring progress inside the host, rather than paying another guest
 * pin/map/device round trip for every 64-KiB ring fragment. This is NOT the
 * general read callback: unknown-length callers keep its partial semantics.
 * EOF/error/cancellation may return a short span; never replay a command. */
static inline ssize_t VIMGLInprocessReadExpected(const struct VIMGLInprocessAPI *api,
    void *stream,void *buffer,size_t bytes)
{
    if(bytes>(size_t)SSIZE_MAX){errno=EINVAL;return -1;}
    size_t done=0;
    while(done<bytes){
        ssize_t n=api->read(stream,(char *)buffer+done,bytes-done);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)return done?(ssize_t)done:n;
        done+=(size_t)n;
    }
    return (ssize_t)done;
}
#endif
