/*
 * Virtio GPU Device
 *
 * Copyright Red Hat, Inc. 2013-2014
 *
 * Authors:
 *     Dave Airlie <airlied@redhat.com>
 *     Gerd Hoffmann <kraxel@redhat.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "qemu/units.h"
#include "qemu/iov.h"
#include "ui/console.h"
#include "hw/virtio/virtio-gpu.h"
#include "hw/virtio/virtio-gpu-pixman.h"
#include "trace.h"
#include "exec/ramblock.h"
#include "system/hostmem.h"
#include <sys/ioctl.h>
#include <linux/memfd.h>
#include "qemu/memfd.h"
#include "standard-headers/linux/udmabuf.h"

static void virtio_gpu_create_udmabuf(struct virtio_gpu_simple_resource *res)
{
    struct udmabuf_create_list *list;
    RAMBlock *rb;
    ram_addr_t offset;
    int udmabuf, i;

    udmabuf = udmabuf_fd();
    if (udmabuf < 0) {
        return;
    }

    list = g_malloc0(sizeof(struct udmabuf_create_list) +
                     sizeof(struct udmabuf_create_item) * res->iov_cnt);

    for (i = 0; i < res->iov_cnt; i++) {
        rcu_read_lock();
        rb = qemu_ram_block_from_host(res->iov[i].iov_base, false, &offset);
        rcu_read_unlock();

        if (!rb || rb->fd < 0) {
            g_free(list);
            return;
        }

        list->list[i].memfd  = rb->fd;
        list->list[i].offset = offset;
        list->list[i].size   = res->iov[i].iov_len;
    }

    list->count = res->iov_cnt;
    list->flags = UDMABUF_FLAGS_CLOEXEC;

    res->dmabuf_fd = ioctl(udmabuf, UDMABUF_CREATE_LIST, list);
    if (res->dmabuf_fd < 0) {
        warn_report("%s: UDMABUF_CREATE_LIST: %s", __func__,
                    strerror(errno));
    }
    g_free(list);
}

static void virtio_gpu_remap_udmabuf(struct virtio_gpu_simple_resource *res)
{
    res->remapped = mmap(NULL, res->blob_size, PROT_READ,
                         MAP_SHARED, res->dmabuf_fd, 0);
    if (res->remapped == MAP_FAILED) {
        warn_report("%s: dmabuf mmap failed: %s", __func__,
                    strerror(errno));
        res->remapped = NULL;
    }
}

static void virtio_gpu_destroy_udmabuf(struct virtio_gpu_simple_resource *res)
{
    if (res->remapped) {
        munmap(res->remapped, res->blob_size);
        res->remapped = NULL;
    }
    if (res->dmabuf_fd >= 0) {
        close(res->dmabuf_fd);
        res->dmabuf_fd = -1;
    }
}

static int find_memory_backend_type(Object *obj, void *opaque)
{
    bool *memfd_backend = opaque;
    int ret;

    if (object_dynamic_cast(obj, TYPE_MEMORY_BACKEND)) {
        HostMemoryBackend *backend = MEMORY_BACKEND(obj);
        RAMBlock *rb = backend->mr.ram_block;

        if (rb && rb->fd > 0) {
            ret = fcntl(rb->fd, F_GET_SEALS);
            if (ret > 0) {
                *memfd_backend = true;
            }
        }
    }

    return 0;
}

bool virtio_gpu_have_udmabuf(void)
{
    Object *memdev_root;
    int udmabuf;
    bool memfd_backend = false;

    udmabuf = udmabuf_fd();
    if (udmabuf < 0) {
        return false;
    }

    memdev_root = object_resolve_path("/objects", NULL);
    object_child_foreach(memdev_root, find_memory_backend_type, &memfd_backend);

    return memfd_backend;
}

void virtio_gpu_init_udmabuf(struct virtio_gpu_simple_resource *res)
{
    void *pdata = NULL;

    res->dmabuf_fd = -1;
    if (res->iov_cnt == 1 &&
        res->iov[0].iov_len < 4096) {
        pdata = res->iov[0].iov_base;
    } else {
        virtio_gpu_create_udmabuf(res);
        if (res->dmabuf_fd < 0) {
            return;
        }
        virtio_gpu_remap_udmabuf(res);
        if (!res->remapped) {
            return;
        }
        pdata = res->remapped;
    }

    res->blob = pdata;
}

void virtio_gpu_fini_udmabuf(struct virtio_gpu_simple_resource *res)
{
    if (res->remapped) {
        virtio_gpu_destroy_udmabuf(res);
    }
}

static void virtio_gpu_free_dmabuf(VirtIOGPU *g, VGPUDMABuf *dmabuf)
{
    struct virtio_gpu_scanout *scanout;

    scanout = &g->parent_obj.scanout[dmabuf->scanout_id];
    dpy_gl_release_dmabuf(scanout->con, dmabuf->buf);
    g_clear_pointer(&dmabuf->buf, qemu_dmabuf_free);
    QTAILQ_REMOVE(&g->dmabuf.bufs, dmabuf, next);
    g_free(dmabuf);
}

static VGPUDMABuf
*virtio_gpu_create_dmabuf(VirtIOGPU *g,
                          uint32_t scanout_id,
                          struct virtio_gpu_simple_resource *res,
                          struct virtio_gpu_framebuffer *fb,
                          struct virtio_gpu_rect *r)
{
    VGPUDMABuf *dmabuf;

    if (res->dmabuf_fd < 0) {
        return NULL;
    }

    dmabuf = g_new0(VGPUDMABuf, 1);
    dmabuf->buf = qemu_dmabuf_new(r->width, r->height, fb->stride,
                                  r->x, r->y, fb->width, fb->height,
                                  qemu_pixman_to_drm_format(fb->format),
                                  0, res->dmabuf_fd, true, false);
    dmabuf->scanout_id = scanout_id;
    /* Remember the request identity so a later identical present can reuse
     * this dmabuf + its EGL import instead of re-creating it. */
    dmabuf->src_fd     = res->dmabuf_fd;
    dmabuf->src_res_id = res->resource_id;
    dmabuf->src_w      = r->width;
    dmabuf->src_h      = r->height;
    dmabuf->src_x      = r->x;
    dmabuf->src_y      = r->y;
    dmabuf->fb_w       = fb->width;
    dmabuf->fb_h       = fb->height;
    dmabuf->fb_stride  = fb->stride;
    dmabuf->fb_format  = fb->format;
    QTAILQ_INSERT_HEAD(&g->dmabuf.bufs, dmabuf, next);

    return dmabuf;
}

static bool virtio_gpu_dmabuf_matches(const VGPUDMABuf *d,
                                      uint32_t scanout_id,
                                      const struct virtio_gpu_simple_resource *res,
                                      const struct virtio_gpu_framebuffer *fb,
                                      const struct virtio_gpu_rect *r)
{
    return d->scanout_id == scanout_id &&
           d->src_fd == res->dmabuf_fd &&
           d->src_res_id == res->resource_id &&
           d->src_w == r->width && d->src_h == r->height &&
           d->src_x == r->x && d->src_y == r->y &&
           d->fb_w == fb->width && d->fb_h == fb->height &&
           d->fb_stride == fb->stride && d->fb_format == fb->format;
}

int virtio_gpu_update_dmabuf(VirtIOGPU *g,
                             uint32_t scanout_id,
                             struct virtio_gpu_simple_resource *res,
                             struct virtio_gpu_framebuffer *fb,
                             struct virtio_gpu_rect *r)
{
    struct virtio_gpu_scanout *scanout = &g->parent_obj.scanout[scanout_id];
    VGPUDMABuf *primary = g->dmabuf.primary[scanout_id];
    VGPUDMABuf *new_primary;
    uint32_t width, height;

    /* Identical re-present of the displayed buffer: only refresh content.
     * The flip-model guest re-presents at ~60 Hz; re-importing every time
     * churns GL state on the display thread and hangs the UI. */
    if (primary && virtio_gpu_dmabuf_matches(primary, scanout_id, res, fb, r)) {
        return 0;
    }

    /* A different buffer (flip chain) or new geometry: import fresh.  The
     * import must be fresh per switch -- a cached EGL import sampled while
     * the guest GPU re-renders that buffer draws torn/stale content under
     * present storms (cursor movement).  The historically expensive part
     * of this path was the unconditional console resize, not the import;
     * only resize when the dimensions actually change. */
    new_primary = virtio_gpu_create_dmabuf(g, scanout_id, res, fb, r);
    if (!new_primary) {
        return -EINVAL;
    }

    width = qemu_dmabuf_get_width(new_primary->buf);
    height = qemu_dmabuf_get_height(new_primary->buf);

    g->dmabuf.primary[scanout_id] = new_primary;
    if (!primary ||
        qemu_dmabuf_get_width(primary->buf) != width ||
        qemu_dmabuf_get_height(primary->buf) != height) {
        qemu_console_resize(scanout->con, width, height);
    }
    dpy_gl_scanout_dmabuf(scanout->con, new_primary->buf);

    if (primary) {
        virtio_gpu_free_dmabuf(g, primary);
    }

    return 0;
}

