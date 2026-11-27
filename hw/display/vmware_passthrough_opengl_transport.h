/* VirtualIntelMac PassthroughOpenGL transport (nfzerox, MIT): loads the host
 * GL executor in-process and drives it with the same partial/vector I/O
 * contract as its socket transport. Explicit memory-mode failure NEVER falls
 * back to a second renderer. Each stream is retained across unlocked waits;
 * shutdown wakes it, and the existing token/epoch owner later closes it.
 */
#ifndef VIM_VMWARE_PASSTHROUGH_OPENGL_TRANSPORT_H
#define VIM_VMWARE_PASSTHROUGH_OPENGL_TRANSPORT_H
#include <sys/uio.h>
#include "hw/display/vim-passthrough-opengl-backing-notify.h"
#ifdef __APPLE__
#include <dlfcn.h>
#include "hw/display/vim-passthrough-opengl-inprocess.h"
static const struct VIMGLInprocessAPI *vim_passthrough_opengl_inprocess;
static const struct VIMGLBackingNotifyAPI *vim_passthrough_opengl_backing_notify;
static gsize vim_passthrough_opengl_inprocess_initialized;
static void vmsvga_gl_load_inprocess(const char *path)
{
    if (g_once_init_enter(&vim_passthrough_opengl_inprocess_initialized)) {
        void *library = g_path_is_absolute(path) ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : NULL;
        VIMGLGetInprocessAPI get = library ? (VIMGLGetInprocessAPI)dlsym(library, "VIMPassthroughOpenGLInprocessGetAPI") : NULL;
        const struct VIMGLInprocessAPI *api = get ? get(VIM_GL_INPROCESS_ABI, sizeof(*api)) : NULL;
        if (api && api->version == VIM_GL_INPROCESS_ABI && api->bytes == sizeof(*api) &&
            api->open && api->read && api->write && api->readv && api->writev && api->shutdown && api->close) {
            vim_passthrough_opengl_inprocess = api;
            VIMGLGetBackingNotifyAPI get_notify = (VIMGLGetBackingNotifyAPI)
                dlsym(library, "VIMPassthroughOpenGLBackingNotifyGetAPI");
            const struct VIMGLBackingNotifyAPI *notify = get_notify ?
                get_notify(VIM_GL_BACKING_NOTIFY_ABI, sizeof(*notify)) : NULL;
            if (notify && notify->version == VIM_GL_BACKING_NOTIFY_ABI &&
                notify->bytes == sizeof(*notify) && notify->subscribe && notify->release_completed) {
                vim_passthrough_opengl_backing_notify = notify;
            }
        } else {
            error_report("PassthroughOpenGL in-process executor initialization failed; refusing socket fallback");
        }
        /* Never unload: CGL owning workers can retire asynchronously. */
        g_once_init_leave(&vim_passthrough_opengl_inprocess_initialized, 1);
    }
}
#endif
static bool vmsvga_gl_backing_notify_available(void)
{
#ifdef __APPLE__
    const char *path = getenv("VIM_PASSTHROUGH_OPENGL_INPROCESS_LIBRARY");
    if (path) {
        vmsvga_gl_load_inprocess(path);
        return vim_passthrough_opengl_backing_notify != NULL;
    }
#endif
    return false;
}
static int vmsvga_gl_backing_subscribe(const uint64_t namespace_id[2], uint64_t ticket,
                                     VIMGLBackingNotifyCallback callback, void *opaque)
{
#ifdef __APPLE__
    if (vim_passthrough_opengl_backing_notify) {
        return vim_passthrough_opengl_backing_notify->subscribe(namespace_id, ticket, callback, opaque);
    }
#endif
    return ENOTSUP;
}
static int vmsvga_gl_backing_release_completed(const uint64_t namespace_id[2], uint64_t ticket)
{
#ifdef __APPLE__
    if (vim_passthrough_opengl_backing_notify) {
        return vim_passthrough_opengl_backing_notify->release_completed(namespace_id, ticket);
    }
#endif
    return ENOTSUP;
}
struct VIMPassthroughOpenGLStream {
    int fd;
    void *memory;
};
static bool vmsvga_gl_stream_valid(struct VIMPassthroughOpenGLStream stream)
{
    return stream.fd >= 0 || stream.memory;
}
static bool vmsvga_gl_stream_drained_close(struct VIMPassthroughOpenGLStream stream)
{
#ifdef __APPLE__
    return stream.memory && vim_passthrough_opengl_inprocess &&
           vim_passthrough_opengl_inprocess->version == 2;
#else
    return false;
#endif
}
static struct VIMPassthroughOpenGLStream vmsvga_gl_stream_open(uint16_t port)
{
    struct VIMPassthroughOpenGLStream stream = { .fd = -1 };
#ifdef __APPLE__
    const char *path = getenv("VIM_PASSTHROUGH_OPENGL_INPROCESS_LIBRARY");
    if (path) {
        vmsvga_gl_load_inprocess(path);
        if (vim_passthrough_opengl_inprocess) {
            stream.memory = vim_passthrough_opengl_inprocess->open();
            if (!stream.memory) {
                error_report("PassthroughOpenGL in-process connection could not be opened");
            }
        }
        return stream;
    }
#endif
    int fd = qemu_socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = { .sin_family = AF_INET,
        .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
    if (fd < 0) {
        return stream;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr))) {
        close(fd);
        return stream;
    }
    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    stream.fd = fd;
    return stream;
}
static void vmsvga_gl_stream_shutdown(struct VIMPassthroughOpenGLStream stream)
{
#ifdef __APPLE__
    if (stream.memory) {
        vim_passthrough_opengl_inprocess->shutdown(stream.memory);
        return;
    }
#endif
    shutdown(stream.fd, SHUT_RDWR);
}
static void vmsvga_gl_stream_close(struct VIMPassthroughOpenGLStream stream)
{
#ifdef __APPLE__
    if (stream.memory) {
        vim_passthrough_opengl_inprocess->close(stream.memory);
        return;
    }
#endif
    close(stream.fd);
}
static ssize_t vmsvga_gl_stream_write(struct VIMPassthroughOpenGLStream stream, const void *p, size_t n)
{
#ifdef __APPLE__
    if (stream.memory) {
        return vim_passthrough_opengl_inprocess->write(stream.memory, p, n);
    }
#endif
    return send(stream.fd, p, n, 0);
}
static ssize_t vmsvga_gl_stream_read(struct VIMPassthroughOpenGLStream stream, void *p, size_t n)
{
#ifdef __APPLE__
    if (stream.memory) {
        return vim_passthrough_opengl_inprocess->read(stream.memory, p, n);
    }
#endif
    return recv(stream.fd, p, n, 0);
}
static ssize_t vmsvga_gl_stream_read_expected(struct VIMPassthroughOpenGLStream stream, void *p, size_t n)
{
#ifdef __APPLE__
    if (stream.memory) {
        return VIMGLInprocessReadExpected(vim_passthrough_opengl_inprocess, stream.memory, p, n);
    }
#endif
    /* Preserve the external comparison path, including its partial reads. */
    return vmsvga_gl_stream_read(stream, p, n);
}
static ssize_t vmsvga_gl_stream_writev(struct VIMPassthroughOpenGLStream stream, const struct iovec *v, int n)
{
#ifdef __APPLE__
    if (stream.memory) {
        return vim_passthrough_opengl_inprocess->writev(stream.memory, v, n);
    }
#endif
    return writev(stream.fd, v, n);
}
static ssize_t vmsvga_gl_stream_readv(struct VIMPassthroughOpenGLStream stream, const struct iovec *v, int n)
{
#ifdef __APPLE__
    if (stream.memory) {
        return vim_passthrough_opengl_inprocess->readv(stream.memory, v, n);
    }
#endif
    return readv(stream.fd, v, n);
}
#endif
