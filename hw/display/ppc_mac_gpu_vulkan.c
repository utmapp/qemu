/*
 * PPC Mac GPU - Vulkan backend (R300 draws)
 *
 * The same job as metal_draw_r300 in ppc_mac_gpu_metal.m: bind and draw
 * the packets r300/r300_draw.c assembles, with the GLSL r300/r300_us.c
 * generates (compiled to SPIR-V by r300/r300_spirv.c).  Developed on
 * MoltenVK; written for any Vulkan 1.1 GPU, discrete ones included.
 *
 * Memory model.  Metal renders straight into VRAM through linear texture
 * views of one shared buffer.  Vulkan cannot: an image cannot alias a
 * buffer at the guest's pitch, and on a discrete GPU the guest's VRAM is
 * not the GPU's memory anyway.  So:
 *
 *  - VRAM is host-visible Vulkan memory (ppc_mac_gpu_vulkan_alloc_vram),
 *    the CPU's copy.  The GPU reads it only to copy from it, and for
 *    shader-decoded textures (a storage buffer, R300_GLSL_VRAM_SSBO).
 *  - Colour and depth buffers and textures are device-local images,
 *    cached by their VRAM range and layout.
 *  - Every VRAM page has a write generation.  CPU writes (the device's
 *    dirty log, set_dirty_source) and GPU writes (a draw into an image)
 *    stamp the pages they touch; an image whose range holds a newer
 *    stamp than its own is stale and copied from VRAM before use, after
 *    any image holding newer contents of that range is written back.
 *  - What a batch renders is written back to VRAM at the end of the
 *    batch, so completed work is in VRAM as the device expects (fences,
 *    scanout, CPU reads after flush_r200).  Only the rectangles draws
 *    touched (their scissors) go back: the CPU may write other parts of
 *    the same buffer meanwhile (2D blits while a window is dragged, text
 *    into a surface), and a whole-image write-back would undo that.
 *
 * The fragment shader reads the buffers it blends into as input
 * attachments that are also its colour attachments (a feedback loop, all
 * images in VK_IMAGE_LAYOUT_GENERAL).  Draws are ordered by a by-region
 * self-dependency barrier between them (MoltenVK included).  Without
 * VK_EXT_rasterization_order_attachment_access, overlapping primitives
 * of one draw are not ordered on other GPUs (not handled yet).
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/thread.h"
#include "qemu/atomic.h"
#include "qemu/bitmap.h"
#include "qemu/bswap.h"

#include <vulkan/vulkan.h>

#include "ppc_mac_gpu_renderer.h"
#include "r300/r300_draw.h"
#include "r300/r300_spirv.h"

#define VK_PAGE             PPC_MAC_GPU_DIRTY_PAGE
#define VK_MAX_IMG          128
#define VK_MAX_TEXFULL      48
#define VK_MAX_FB           64
#define VK_MAX_RP           32
#define VK_NBATCH           6
#define VK_ARENA_CHUNK      (8u << 20)
#define VK_SETS_PER_POOL    256
#define VK_MAX_ATT          (R300_US_MAX_TARGETS + 1)
#define VK_MAX_DR           8       /* dirty rectangles per image */

static char vk_err[256];

static void vk_fail(const char *fmt, ...) G_GNUC_PRINTF(1, 2);
static void vk_fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(vk_err, sizeof(vk_err), fmt, ap);
    va_end(ap);
    qemu_log("ppc-mac-gpu vulkan: %s\n", vk_err);
}

const char *ppc_mac_gpu_vulkan_error(void)
{
    return vk_err[0] ? vk_err : "unknown Vulkan error";
}

static void vk_warn(uint32_t bit, const char *msg)
{
    static uint32_t warned;
    if (!(warned & bit)) {
        warned |= bit;
        qemu_log("ppc-mac-gpu vulkan: %s\n", msg);
    }
}

/* ---- objects ---------------------------------------------------------- */

typedef struct VkChunk {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    VkDeviceSize size, off;
} VkChunk;

enum { TRASH_IMAGE, TRASH_VIEW, TRASH_MEM, TRASH_FB, TRASH_BUF };
typedef struct VkTrash {
    int type;
    uint64_t h;
} VkTrash;

typedef struct VkBatch {
    int state;                  /* 0 free, 1 open, 2 submitted */
    uint32_t seq;
    VkCommandBuffer cb;
    VkFence fence;
    GArray *pools;              /* VkDescriptorPool */
    unsigned pool_cur;
    GPtrArray *chunks;          /* VkChunk * used by this batch */
    GArray *trash;              /* VkTrash, destroyed once complete */
} VkBatch;

enum { IMG_RT, IMG_TEX };
typedef struct VkImg {
    bool live;
    int cls;
    uint32_t addr, width, height, pitch;    /* the key, with fmt and cls */
    VkFormat fmt;
    uint32_t bpp;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    bool valid;                 /* holds the contents of generation gen */
    uint64_t gen;
    bool dirty;                 /* rendered into since its last write-back */
    uint32_t ndr;               /* ... in these rectangles (x0, y0, x1, y1) */
    uint32_t dr[VK_MAX_DR][4];
    uint64_t used;
    uint64_t hash;              /* PPCGPU_VK_CHECK: VRAM bytes when loaded, or 0 */
} VkImg;

/* Textures rebuilt on the CPU (mips, 3D, cube, 16bpp, DXT, GART copies). */
typedef struct VkTexKey {
    uint32_t addr, format, kind, width, height, depth, dim, levels, pitch;
    uint32_t host;
    uint64_t hash;
} VkTexKey;
typedef struct VkTexFull {
    bool live;
    VkTexKey key;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    uint64_t used;
} VkTexFull;

typedef struct VkRP {
    uint32_t n;                 /* attachments: colour buffers, then Z */
    VkFormat fmt[VK_MAX_ATT];
    VkRenderPass rp;
} VkRP;

typedef struct VkFB {
    VkRenderPass rp;
    uint32_t n, w, h;
    VkImageView view[VK_MAX_ATT];
    VkFramebuffer fb;
} VkFB;

/* Pipelines of one GLSL source. */
typedef struct VkPipeKey {
    VkRenderPass rp;
    uint32_t ncb;
    bool z;
    VkPrimitiveTopology topo;
    VkCullModeFlags cull;
    VkFrontFace front;
} VkPipeKey;
typedef struct VkProg {
    VkShaderModule mod[3];      /* R300Stage */
    bool failed[3];
    GArray *pipes;              /* VkPipeVar */
} VkProg;
typedef struct VkPipeVar {
    VkPipeKey key;
    VkPipeline pipe;
} VkPipeVar;

static struct {
    bool ready;
    VkInstance inst;
    VkPhysicalDevice pdev;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties memprops;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    bool moltenvk;
    bool depth_clamp, aniso, bc, mirror_clamp;
    VkDeviceSize align;         /* arena allocation alignment */

    VkCommandPool cmdpool;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;

    /* VRAM */
    uint8_t *vram;
    uint64_t vram_size;
    VkDeviceMemory vram_mem;
    VkBuffer vram_buf;
    uint64_t npages;
    uint64_t *page_gen;         /* write generation of each page */
    uint32_t *page_wseq;        /* batch that last wrote each page (GPU) */
    uint32_t *page_rseq;        /* batch that last read each page (GPU) */
    unsigned long *dirty_bm;
    uint64_t gen;
    PPCMacGPUDirtyFn dirty_fn;
    void *dirty_arg;

    VkBuffer zpass_buf;
    VkDeviceMemory zpass_mem;
    uint32_t *zpass;

    /* batches */
    VkBatch batch[VK_NBATCH];
    VkBatch *cur;               /* open batch, or NULL */
    uint32_t seq;               /* last sequence number handed out */
    uint32_t done_seq;          /* every batch up to here completed (atomic) */
    uint32_t worker_seq;        /* ... and the waiter is finished with it */
    GPtrArray *free_chunks;
    VkChunk *chunk;             /* current arena chunk (in cur->chunks) */

    /* the open render pass */
    bool in_pass;
    VkRenderPass pass_rp;
    VkFramebuffer pass_fb;
    uint32_t pass_n;
    VkImg *pass_img[VK_MAX_ATT];
    bool pass_z;                /* last attachment is the depth buffer */
    bool pass_z16;

    VkImg img[VK_MAX_IMG];
    VkTexFull texfull[VK_MAX_TEXFULL];
    uint64_t clock;
    VkRP rps[VK_MAX_RP];
    unsigned nrp;
    VkFB fbs[VK_MAX_FB];
    unsigned fb_next;
    GHashTable *progs;          /* GLSL -> VkProg */
    GHashTable *samplers;       /* key -> VkSampler */
    VkImage dummy_img[3];
    VkDeviceMemory dummy_mem[3];
    VkImageView dummy_view[3];
    bool dummy_ready;

    /* completion waiter */
    QemuThread thread;
    QemuMutex lock;
    QemuCond wake, idle;
    GQueue *waitq;              /* VkWaitItem */
    bool thread_started;

    uint64_t stat_draws, stat_passes, stat_uploads, stat_writebacks, stat_flushes;
} V;

typedef struct VkWaitItem {
    VkFence fence;
    uint32_t seq;
    void (*done)(void *, uint32_t);
    void *arg;
} VkWaitItem;

#define VKCHECK(expr, what) do {                                        \
        VkResult r_ = (expr);                                           \
        if (r_ != VK_SUCCESS) {                                         \
            vk_fail("%s failed (VkResult %d)", what, (int)r_);          \
            goto fail;                                                  \
        }                                                               \
    } while (0)

/* ---- context ---------------------------------------------------------- */

static bool vk_has_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
    for (uint32_t i = 0; i < n; i++) {
        if (!strcmp(e[i].extensionName, name)) {
            return true;
        }
    }
    return false;
}

static int vk_memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < V.memprops.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (V.memprops.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return -1;
}

static void *vk_waiter(void *opaque);

static bool vk_ctx_init(void)
{
    VkExtensionProperties *ie = NULL, *de = NULL;
    VkPhysicalDevice *pd = NULL;
    uint32_t n = 0;
    const char *iext[4], *dext[4];
    uint32_t niext = 0, ndext = 0;
    bool portability = false;

    if (V.ready) {
        return true;
    }
    if (vkEnumerateInstanceExtensionProperties(NULL, &n, NULL) != VK_SUCCESS) {
        vk_fail("no Vulkan loader or driver (vkEnumerateInstanceExtensionProperties failed)");
        return false;
    }
    ie = g_new0(VkExtensionProperties, n);
    vkEnumerateInstanceExtensionProperties(NULL, &n, ie);
    if (vk_has_ext(ie, n, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        iext[niext++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        portability = true;
    }
    g_free(ie);
    ie = NULL;

    /*
     * Ask for 1.3: MoltenVK reports a device only as new as the version
     * asked for, and 1.2 is needed to see the driver ID and 1.2 features.
     * A 1.0 loader only accepts 1.0; devices are still required to be 1.1.
     */
    uint32_t iver = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&iver) != VK_SUCCESS) {
        iver = VK_API_VERSION_1_0;
    }
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "qemu ati-radeon-9700",
        .apiVersion = iver >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_0,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0,
        .pApplicationInfo = &app,
        .enabledExtensionCount = niext,
        .ppEnabledExtensionNames = iext,
    };
    VKCHECK(vkCreateInstance(&ici, NULL, &V.inst), "vkCreateInstance");

    n = 0;
    vkEnumeratePhysicalDevices(V.inst, &n, NULL);
    if (!n) {
        vk_fail("no Vulkan device");
        goto fail;
    }
    pd = g_new0(VkPhysicalDevice, n);
    vkEnumeratePhysicalDevices(V.inst, &n, pd);
    {
        /* PPCGPU_VK_DEVICE=index picks one; else discrete, then integrated. */
        const char *e = getenv("PPCGPU_VK_DEVICE");
        int best = -1, score = -1;
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(pd[i], &p);
            int sc = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 :
                     p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
            if (p.apiVersion < VK_API_VERSION_1_1) {
                continue;
            }
            if (e && (uint32_t)atoi(e) == i) {
                sc = 100;
            }
            if (sc > score) {
                score = sc;
                best = i;
            }
        }
        if (best < 0) {
            vk_fail("no Vulkan 1.1 device");
            goto fail;
        }
        V.pdev = pd[best];
    }
    g_free(pd);
    pd = NULL;
    vkGetPhysicalDeviceProperties(V.pdev, &V.props);
    vkGetPhysicalDeviceMemoryProperties(V.pdev, &V.memprops);

    n = 0;
    vkEnumerateDeviceExtensionProperties(V.pdev, NULL, &n, NULL);
    de = g_new0(VkExtensionProperties, n);
    vkEnumerateDeviceExtensionProperties(V.pdev, NULL, &n, de);
    if (vk_has_ext(de, n, "VK_KHR_portability_subset")) {
        dext[ndext++] = "VK_KHR_portability_subset";
    }
    bool mirror_ext = V.props.apiVersion < VK_API_VERSION_1_2 &&
        vk_has_ext(de, n, VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);
    if (mirror_ext) {
        dext[ndext++] = VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME;
    }
    g_free(de);
    de = NULL;

    {
        VkPhysicalDeviceDriverProperties drv = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
        };
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = V.props.apiVersion >= VK_API_VERSION_1_2 ? &drv : NULL,
        };
        vkGetPhysicalDeviceProperties2(V.pdev, &p2);
        V.moltenvk = V.props.apiVersion >= VK_API_VERSION_1_2 &&
                     drv.driverID == VK_DRIVER_ID_MOLTENVK;
    }

    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = V.props.apiVersion >= VK_API_VERSION_1_2 ? &f12 : NULL,
    };
    vkGetPhysicalDeviceFeatures2(V.pdev, &f2);
    if (!f2.features.fragmentStoresAndAtomics) {
        vk_fail("%s lacks fragmentStoresAndAtomics", V.props.deviceName);
        goto fail;
    }
    VkPhysicalDeviceFeatures want = {
        .fragmentStoresAndAtomics = VK_TRUE,
        .depthClamp = f2.features.depthClamp,
        .shaderClipDistance = f2.features.shaderClipDistance,
        .samplerAnisotropy = f2.features.samplerAnisotropy,
        .textureCompressionBC = f2.features.textureCompressionBC,
    };
    V.depth_clamp = want.depthClamp;
    V.aniso = want.samplerAnisotropy;
    V.bc = want.textureCompressionBC;
    VkPhysicalDeviceVulkan12Features w12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .samplerMirrorClampToEdge = f12.samplerMirrorClampToEdge,
    };
    V.mirror_clamp = mirror_ext || f12.samplerMirrorClampToEdge;

    n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(V.pdev, &n, NULL);
    VkQueueFamilyProperties *qf = g_new0(VkQueueFamilyProperties, n);
    vkGetPhysicalDeviceQueueFamilyProperties(V.pdev, &n, qf);
    V.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            V.qfam = i;
            break;
        }
    }
    g_free(qf);
    if (V.qfam == UINT32_MAX) {
        vk_fail("no graphics queue");
        goto fail;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = V.qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = V.props.apiVersion >= VK_API_VERSION_1_2 ? &w12 : NULL,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = ndext,
        .ppEnabledExtensionNames = dext,
        .pEnabledFeatures = &want,
    };
    VKCHECK(vkCreateDevice(V.pdev, &dci, NULL, &V.dev), "vkCreateDevice");
    vkGetDeviceQueue(V.dev, V.qfam, 0, &V.queue);

    V.align = MAX(MAX(V.props.limits.minUniformBufferOffsetAlignment,
                      V.props.limits.minStorageBufferOffsetAlignment), 256);

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = V.qfam,
    };
    VKCHECK(vkCreateCommandPool(V.dev, &cpi, NULL, &V.cmdpool), "vkCreateCommandPool");

    /* One descriptor set layout for every R300 program (R300_BIND_*). */
    VkDescriptorSetLayoutBinding b[6 + VK_MAX_ATT + R300_NUM_TEX_UNITS];
    uint32_t nb = 0;
    const VkShaderStageFlags all = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_UNIFORMS,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, all, NULL };
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_ZPASS,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, NULL };
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_VERTS,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, NULL };
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_MS,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, all, NULL };
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_VRAM,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, NULL };
    b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_AUX,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, NULL };
    for (uint32_t k = 0; k < VK_MAX_ATT; k++) {
        b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_FB0 + k,
            VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    }
    for (uint32_t k = 0; k < R300_NUM_TEX_UNITS; k++) {
        b[nb++] = (VkDescriptorSetLayoutBinding){ R300_BIND_TEX0 + k,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    }
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nb,
        .pBindings = b,
    };
    VKCHECK(vkCreateDescriptorSetLayout(V.dev, &dli, NULL, &V.dsl), "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &V.dsl,
    };
    VKCHECK(vkCreatePipelineLayout(V.dev, &pli, NULL, &V.layout), "vkCreatePipelineLayout");

    for (int i = 0; i < VK_NBATCH; i++) {
        VkBatch *bt = &V.batch[i];
        VkCommandBufferAllocateInfo cai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = V.cmdpool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VKCHECK(vkAllocateCommandBuffers(V.dev, &cai, &bt->cb), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VKCHECK(vkCreateFence(V.dev, &fci, NULL, &bt->fence), "vkCreateFence");
        bt->pools = g_array_new(FALSE, FALSE, sizeof(VkDescriptorPool));
        bt->chunks = g_ptr_array_new();
        bt->trash = g_array_new(FALSE, FALSE, sizeof(VkTrash));
    }
    V.free_chunks = g_ptr_array_new();
    V.progs = g_hash_table_new(g_str_hash, g_str_equal);
    V.samplers = g_hash_table_new(g_direct_hash, g_direct_equal);

    qemu_mutex_init(&V.lock);
    qemu_cond_init(&V.wake);
    qemu_cond_init(&V.idle);
    V.waitq = g_queue_new();
    qemu_thread_create(&V.thread, "vk-wait", vk_waiter, NULL, QEMU_THREAD_DETACHED);
    V.thread_started = true;

    qemu_log("ppc-mac-gpu vulkan: %s (%s, Vulkan %u.%u)%s\n", V.props.deviceName,
             V.moltenvk ? "MoltenVK" : "native",
             VK_API_VERSION_MAJOR(V.props.apiVersion),
             VK_API_VERSION_MINOR(V.props.apiVersion),
             V.bc ? "" : ", no BC (DXT) textures");
    V.ready = true;
    return true;

fail:
    g_free(ie);
    g_free(de);
    g_free(pd);
    return false;
}

/* ---- buffers and images ----------------------------------------------- */

static bool vk_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags want, VkMemoryPropertyFlags prefer,
                      VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
    };
    VkMemoryRequirements req;
    int mt;

    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    VKCHECK(vkCreateBuffer(V.dev, &bci, NULL, buf), "vkCreateBuffer");
    vkGetBufferMemoryRequirements(V.dev, *buf, &req);
    mt = vk_memtype(req.memoryTypeBits, want | prefer);
    if (mt < 0) {
        mt = vk_memtype(req.memoryTypeBits, want);
    }
    if (mt < 0) {
        vk_fail("no memory type for a %llu-byte buffer", (unsigned long long)size);
        goto fail;
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    VKCHECK(vkAllocateMemory(V.dev, &mai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(V.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (map) {
        VKCHECK(vkMapMemory(V.dev, *mem, 0, VK_WHOLE_SIZE, 0, map), "vkMapMemory");
    }
    return true;
fail:
    if (*buf) {
        vkDestroyBuffer(V.dev, *buf, NULL);
    }
    if (*mem) {
        vkFreeMemory(V.dev, *mem, NULL);
    }
    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    return false;
}

static bool vk_image(VkImageType type, VkImageViewType vtype, VkFormat fmt,
                     uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
                     uint32_t layers, VkImageUsageFlags usage,
                     VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = vtype == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
        .imageType = type,
        .format = fmt,
        .extent = { w, h, d },
        .mipLevels = levels,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkMemoryRequirements req;
    int mt;

    *img = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    *view = VK_NULL_HANDLE;
    VKCHECK(vkCreateImage(V.dev, &ici, NULL, img), "vkCreateImage");
    vkGetImageMemoryRequirements(V.dev, *img, &req);
    mt = vk_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt < 0) {
        mt = vk_memtype(req.memoryTypeBits, 0);
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    VKCHECK(vkAllocateMemory(V.dev, &mai, NULL, mem), "vkAllocateMemory (image)");
    VKCHECK(vkBindImageMemory(V.dev, *img, *mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *img,
        .viewType = vtype,
        .format = fmt,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
    };
    VKCHECK(vkCreateImageView(V.dev, &vci, NULL, view), "vkCreateImageView");
    return true;
fail:
    if (*img) {
        vkDestroyImage(V.dev, *img, NULL);
    }
    if (*mem) {
        vkFreeMemory(V.dev, *mem, NULL);
    }
    *img = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    return false;
}

/* ---- VRAM ------------------------------------------------------------- */

void *ppc_mac_gpu_vulkan_alloc_vram(uint64_t vram_size, void **opaque_out)
{
    void *map = NULL;

    if (!vk_ctx_init()) {
        return NULL;
    }
    if (V.vram) {
        vk_fail("VRAM already allocated (one Vulkan-rendered card only)");
        return NULL;
    }
    /*
     * Host-visible and, where there is a choice, CPU-cached: the guest
     * CPU reads VRAM a lot, and write-combined memory reads slowly.  The
     * GPU only copies from and to it (and reads shader-decoded textures),
     * so it does not need to be device-local.
     */
    if (!vk_buffer(vram_size,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                   &V.vram_buf, &V.vram_mem, &map)) {
        return NULL;
    }
    if ((uintptr_t)map % qemu_real_host_page_size()) {
        vk_fail("mapped VRAM is not page aligned");
        vkDestroyBuffer(V.dev, V.vram_buf, NULL);
        vkFreeMemory(V.dev, V.vram_mem, NULL);
        V.vram_buf = VK_NULL_HANDLE;
        V.vram_mem = VK_NULL_HANDLE;
        return NULL;
    }
    memset(map, 0, vram_size);
    V.vram = map;
    V.vram_size = vram_size;
    V.npages = DIV_ROUND_UP(vram_size, VK_PAGE);
    V.page_gen = g_new0(uint64_t, V.npages);
    V.page_wseq = g_new0(uint32_t, V.npages);
    V.page_rseq = g_new0(uint32_t, V.npages);
    V.dirty_bm = bitmap_new(V.npages);
    *opaque_out = &V;
    return map;
}

void ppc_mac_gpu_vulkan_free_vram(void *opaque)
{
    /* Lives as long as the process: images and batches refer to it. */
}

/* ---- batches ---------------------------------------------------------- */

static bool vk_seq_done(uint32_t seq)
{
    return (int32_t)(qatomic_read(&V.done_seq) - seq) >= 0;
}

static void vk_set_done(uint32_t seq)
{
    uint32_t old = qatomic_read(&V.done_seq);
    while ((int32_t)(seq - old) > 0) {
        uint32_t prev = qatomic_cmpxchg(&V.done_seq, old, seq);
        if (prev == old) {
            break;
        }
        old = prev;
    }
}

static void *vk_waiter(void *opaque)
{
    for (;;) {
        qemu_mutex_lock(&V.lock);
        while (g_queue_is_empty(V.waitq)) {
            qemu_cond_wait(&V.wake, &V.lock);
        }
        VkWaitItem *it = g_queue_peek_head(V.waitq);
        qemu_mutex_unlock(&V.lock);

        vkWaitForFences(V.dev, 1, &it->fence, VK_TRUE, UINT64_MAX);
        vk_set_done(it->seq);
        if (it->done) {
            it->done(it->arg, it->seq);
        }
        qemu_mutex_lock(&V.lock);
        g_queue_pop_head(V.waitq);
        qatomic_set(&V.worker_seq, it->seq);
        qemu_cond_broadcast(&V.idle);
        qemu_mutex_unlock(&V.lock);
        g_free(it);
    }
    return NULL;
}

static void vk_trash(VkBatch *b, int type, uint64_t h)
{
    VkTrash t = { type, h };
    if (!h) {
        return;
    }
    if (!b) {
        b = V.cur;
    }
    if (!b) {
        /* nothing open: destroy once everything submitted is done */
        for (int i = 0; i < VK_NBATCH; i++) {
            if (V.batch[i].state == 2 && (!b || (int32_t)(V.batch[i].seq - b->seq) > 0)) {
                b = &V.batch[i];
            }
        }
    }
    if (!b) {
        switch (type) {
        case TRASH_IMAGE: vkDestroyImage(V.dev, (VkImage)(uintptr_t)h, NULL); break;
        case TRASH_VIEW:  vkDestroyImageView(V.dev, (VkImageView)(uintptr_t)h, NULL); break;
        case TRASH_MEM:   vkFreeMemory(V.dev, (VkDeviceMemory)(uintptr_t)h, NULL); break;
        case TRASH_FB:    vkDestroyFramebuffer(V.dev, (VkFramebuffer)(uintptr_t)h, NULL); break;
        case TRASH_BUF:   vkDestroyBuffer(V.dev, (VkBuffer)(uintptr_t)h, NULL); break;
        }
        return;
    }
    g_array_append_val(b->trash, t);
}

/* Return a completed batch's resources. */
static void vk_recycle(VkBatch *b)
{
    vkResetFences(V.dev, 1, &b->fence);
    vkResetCommandBuffer(b->cb, 0);
    for (guint i = 0; i < b->pools->len; i++) {
        vkResetDescriptorPool(V.dev, g_array_index(b->pools, VkDescriptorPool, i), 0);
    }
    b->pool_cur = 0;
    for (guint i = 0; i < b->chunks->len; i++) {
        VkChunk *c = g_ptr_array_index(b->chunks, i);
        if (c->size == VK_ARENA_CHUNK && V.free_chunks->len < 8) {
            c->off = 0;
            g_ptr_array_add(V.free_chunks, c);
        } else {
            vkDestroyBuffer(V.dev, c->buf, NULL);
            vkFreeMemory(V.dev, c->mem, NULL);
            g_free(c);
        }
    }
    g_ptr_array_set_size(b->chunks, 0);
    for (guint i = 0; i < b->trash->len; i++) {
        VkTrash *t = &g_array_index(b->trash, VkTrash, i);
        switch (t->type) {
        case TRASH_IMAGE: vkDestroyImage(V.dev, (VkImage)(uintptr_t)t->h, NULL); break;
        case TRASH_VIEW:  vkDestroyImageView(V.dev, (VkImageView)(uintptr_t)t->h, NULL); break;
        case TRASH_MEM:   vkFreeMemory(V.dev, (VkDeviceMemory)(uintptr_t)t->h, NULL); break;
        case TRASH_FB:    vkDestroyFramebuffer(V.dev, (VkFramebuffer)(uintptr_t)t->h, NULL); break;
        case TRASH_BUF:   vkDestroyBuffer(V.dev, (VkBuffer)(uintptr_t)t->h, NULL); break;
        }
    }
    g_array_set_size(b->trash, 0);
    b->state = 0;
}

static void vk_reap(void)
{
    uint32_t ws = qatomic_read(&V.worker_seq);
    for (int i = 0; i < VK_NBATCH; i++) {
        VkBatch *b = &V.batch[i];
        if (b->state == 2 && (int32_t)(ws - b->seq) >= 0) {
            vk_recycle(b);
        }
    }
}

static void vk_full_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* The open batch, starting one if needed. */
static VkBatch *vk_batch(void)
{
    VkBatch *b = NULL;

    if (V.cur) {
        return V.cur;
    }
    vk_reap();
    for (;;) {
        for (int i = 0; i < VK_NBATCH && !b; i++) {
            if (V.batch[i].state == 0) {
                b = &V.batch[i];
            }
        }
        if (b) {
            break;
        }
        /* all in flight: wait for the waiter to finish the oldest */
        qemu_mutex_lock(&V.lock);
        uint32_t ws = qatomic_read(&V.worker_seq);
        while (qatomic_read(&V.worker_seq) == ws && !g_queue_is_empty(V.waitq)) {
            qemu_cond_wait(&V.idle, &V.lock);
        }
        qemu_mutex_unlock(&V.lock);
        vk_reap();
    }
    b->state = 1;
    b->seq = ++V.seq;
    if (!b->seq) {
        b->seq = ++V.seq;
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(b->cb, &bi);
    /* ordered after everything submitted before (and host writes) */
    vk_full_barrier(b->cb);
    V.cur = b;
    V.chunk = NULL;
    return b;
}

/* Bump allocation from the batch's host-visible arena. */
static void *vk_arena(VkDeviceSize len, VkBuffer *buf, VkDeviceSize *off)
{
    VkBatch *b = vk_batch();
    VkChunk *c = V.chunk;

    len = ROUND_UP(len, V.align);
    if (!c || c->off + len > c->size) {
        if (len <= VK_ARENA_CHUNK && V.free_chunks->len) {
            c = g_ptr_array_steal_index_fast(V.free_chunks, V.free_chunks->len - 1);
        } else {
            void *map;
            c = g_new0(VkChunk, 1);
            c->size = MAX(len, (VkDeviceSize)VK_ARENA_CHUNK);
            if (!vk_buffer(c->size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                           &c->buf, &c->mem, &map)) {
                g_free(c);
                return NULL;
            }
            c->map = map;
        }
        c->off = 0;
        g_ptr_array_add(b->chunks, c);
        if (len <= VK_ARENA_CHUNK) {
            V.chunk = c;
        }
    }
    *buf = c->buf;
    *off = c->off;
    c->off += len;
    return c->map + *off;
}

static void vk_end_pass(void)
{
    if (V.in_pass) {
        vkCmdEndRenderPass(V.cur->cb);
        V.in_pass = false;
    }
}

/* ---- page generations ------------------------------------------------- */

static bool vk_check(void);

static inline uint64_t vk_pg(uint64_t addr)
{
    return addr / VK_PAGE;
}

/* Is a GPU write of the page still pending (open or unfinished batch)? */
static inline bool vk_page_pending_write(uint64_t p)
{
    uint32_t s = V.page_wseq[p];
    return s && (V.cur && s == V.cur->seq ? true : !vk_seq_done(s));
}

/*
 * Take the CPU's writes from the device's dirty log (which, with this
 * backend, does not include what draws render).  An image with rendering
 * not yet in VRAM over a page the CPU wrote gets that rendering written
 * back before it is reloaded (vk_img_get), so both survive.
 */
static void vk_harvest(void)
{
    if (!V.dirty_fn) {
        return;
    }
    bitmap_zero(V.dirty_bm, V.npages);
    V.dirty_fn(V.dirty_arg, V.dirty_bm, V.npages);
    uint64_t p = find_first_bit(V.dirty_bm, V.npages);
    if (p >= V.npages) {
        return;
    }
    V.gen++;
    for (; p < V.npages; p = find_next_bit(V.dirty_bm, V.npages, p + 1)) {
        V.page_gen[p] = V.gen;
        if (vk_check() && vk_page_pending_write(p)) {
            static int n;
            if (n++ < 300) {
                qemu_log("ppc-mac-gpu vulkan CHECK: CPU wrote VRAM page %06llx while "
                         "rendering to it is pending (no flush first)\n",
                         (unsigned long long)p * VK_PAGE);
            }
        }
    }
}

static uint64_t vk_range_gen(uint64_t lo, uint64_t hi)
{
    uint64_t g = 0;
    for (uint64_t p = vk_pg(lo); p <= vk_pg(hi - 1) && p < V.npages; p++) {
        g = MAX(g, V.page_gen[p]);
    }
    return g;
}

static void vk_note_read(uint64_t lo, uint64_t hi)
{
    uint32_t s = vk_batch()->seq;
    for (uint64_t p = vk_pg(lo); p <= vk_pg(hi - 1) && p < V.npages; p++) {
        V.page_rseq[p] = s;
    }
}

/* PPCGPU_VK_CHECK=1: verify that textures the tracking calls current
 * still match VRAM, and log (then fix) any that don't. */
static bool vk_check(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PPCGPU_VK_CHECK");
        on = e && e[0] == '1';
    }
    return on;
}

static bool vk_range_pending(uint64_t lo, uint64_t hi);

static uint64_t vk_img_hash(uint32_t addr, uint32_t w, uint32_t h, uint32_t pitch,
                            uint32_t bpp)
{
    uint64_t hv = 0;
    for (uint32_t y = 0; y < h; y++) {
        hv = hv * 0x100000001B3ull ^
             r300_hash_bytes(V.vram + addr + (uint64_t)y * pitch, (size_t)w * bpp);
    }
    return hv | 1;
}

static inline uint64_t vk_img_hi(const VkImg *im)
{
    return (uint64_t)im->addr + (uint64_t)im->pitch * (im->height - 1) +
           (uint64_t)im->width * im->bpp;
}

/* ---- image cache ------------------------------------------------------ */

static void vk_fb_forget(VkImageView v)
{
    for (int i = 0; i < VK_MAX_FB; i++) {
        VkFB *f = &V.fbs[i];
        if (!f->fb) {
            continue;
        }
        for (uint32_t k = 0; k < f->n; k++) {
            if (f->view[k] == v) {
                vk_trash(NULL, TRASH_FB, (uint64_t)(uintptr_t)f->fb);
                f->fb = VK_NULL_HANDLE;
                break;
            }
        }
    }
}

/*
 * The copy of image rows/columns [x0, x1) x [y0, y1) to or from VRAM.
 * Buffer offsets must be 4-byte aligned: x0 is rounded down to suit, and
 * a pitch that is not a multiple of 4 falls back to the image's first row.
 */
static VkBufferImageCopy vk_region(const VkImg *im, uint32_t x0, uint32_t y0,
                                   uint32_t x1, uint32_t y1)
{
    if (im->pitch % 4) {
        y1 = y1 > y0 ? y1 : im->height;
        y0 = 0;
    }
    if (im->bpp < 4) {
        x0 &= ~(4 / im->bpp - 1);
    }
    return (VkBufferImageCopy){
        .bufferOffset = im->addr + (uint64_t)y0 * im->pitch + (uint64_t)x0 * im->bpp,
        .bufferRowLength = im->pitch / im->bpp,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageOffset = { (int32_t)x0, (int32_t)y0, 0 },
        .imageExtent = { x1 - x0, y1 - y0, 1 },
    };
}

/* Write the rectangles rendered since the last write-back to VRAM. */
static void vk_copy_to_vram(VkImg *im)
{
    VkCommandBuffer cb = vk_batch()->cb;
    VkBufferImageCopy r[VK_MAX_DR];

    for (uint32_t i = 0; i < im->ndr; i++) {
        r[i] = vk_region(im, im->dr[i][0], im->dr[i][1], im->dr[i][2], im->dr[i][3]);
    }
    if (im->ndr) {
        vk_end_pass();
        vk_full_barrier(cb);
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, V.vram_buf,
                               im->ndr, r);
        vk_full_barrier(cb);
        V.stat_writebacks++;
    }
    im->dirty = false;
    im->ndr = 0;
}

/* Write back every image with rendering newer than VRAM in [lo, hi). */
static void vk_writeback_range(uint64_t lo, uint64_t hi)
{
    for (int i = 0; i < VK_MAX_IMG; i++) {
        VkImg *im = &V.img[i];
        if (im->live && im->dirty && lo < vk_img_hi(im) && im->addr < hi) {
            vk_copy_to_vram(im);
        }
    }
}

static void vk_img_free(VkImg *im)
{
    if (im->dirty) {
        vk_copy_to_vram(im);
    }
    for (uint32_t k = 0; k < V.pass_n; k++) {
        if (V.in_pass && V.pass_img[k] == im) {
            vk_end_pass();
        }
    }
    vk_fb_forget(im->view);
    vk_trash(NULL, TRASH_VIEW, (uint64_t)(uintptr_t)im->view);
    vk_trash(NULL, TRASH_IMAGE, (uint64_t)(uintptr_t)im->img);
    vk_trash(NULL, TRASH_MEM, (uint64_t)(uintptr_t)im->mem);
    memset(im, 0, sizeof(*im));
}

static void vk_layout_general(VkCommandBuffer cb, VkImage img, uint32_t levels,
                              uint32_t layers)
{
    VkImageMemoryBarrier ib = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
}

/*
 * The image for a VRAM range: w x h texels of bpp bytes at addr, rows
 * pitch bytes apart.  Its contents are current: copied from VRAM when a
 * CPU write or another image's rendering made them stale (after writing
 * such rendering back).
 */
static VkImg *vk_img_get(int cls, uint32_t addr, uint32_t w, uint32_t h,
                         uint32_t pitch, VkFormat fmt, uint32_t bpp)
{
    VkImg *im = NULL;
    int lru = -1;

    if (!w || !h || pitch < w * bpp || addr % 4 || addr % bpp || pitch % bpp ||
        (uint64_t)addr + (uint64_t)pitch * (h - 1) + (uint64_t)w * bpp > V.vram_size ||
        w > V.props.limits.maxImageDimension2D || h > V.props.limits.maxImageDimension2D) {
        return NULL;
    }
    for (int i = 0; i < VK_MAX_IMG; i++) {
        VkImg *c = &V.img[i];
        if (c->live && c->cls == cls && c->addr == addr && c->width == w &&
            c->height == h && c->pitch == pitch && c->fmt == fmt) {
            im = c;
            break;
        }
    }
    if (!im) {
        /* a free slot, else the least recently used not in the open pass */
        for (int i = 0; i < VK_MAX_IMG; i++) {
            VkImg *c = &V.img[i];
            bool busy = false;
            if (!c->live) {
                lru = i;
                break;
            }
            for (uint32_t k = 0; V.in_pass && k < V.pass_n; k++) {
                busy |= V.pass_img[k] == c;
            }
            if (!busy && (lru < 0 || c->used < V.img[lru].used)) {
                lru = i;
            }
        }
    }
    if (!im) {
        if (lru < 0) {
            return NULL;
        }
        im = &V.img[lru];
        if (im->live) {
            vk_img_free(im);
        }
        VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  (cls == IMG_RT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                   VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
                                                 : VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!vk_image(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, fmt, w, h, 1, 1, 1, usage,
                      &im->img, &im->mem, &im->view)) {
            return NULL;
        }
        im->live = true;
        im->cls = cls;
        im->addr = addr;
        im->width = w;
        im->height = h;
        im->pitch = pitch;
        im->fmt = fmt;
        im->bpp = bpp;
        vk_end_pass();
        vk_layout_general(vk_batch()->cb, im->img, 1, 1);
    }
    im->used = ++V.clock;

    uint64_t hi = vk_img_hi(im);
    uint64_t gen = vk_range_gen(addr, hi);
    if (vk_check() && cls == IMG_TEX && im->valid && gen <= im->gen && im->hash &&
        !vk_range_pending(addr, hi) &&
        vk_img_hash(addr, w, h, pitch, bpp) != im->hash) {
        static int n;
        if (n++ < 200) {
            qemu_log("ppc-mac-gpu vulkan CHECK: texture %06x %ux%u pitch %u changed "
                     "in VRAM without a dirty mark (gen %llu, page gens %llu)\n",
                     addr, w, h, pitch, (unsigned long long)im->gen,
                     (unsigned long long)gen);
        }
        im->valid = false;
    }
    if (!im->valid || gen > im->gen) {
        /* The rows holding pages newer than the image (all, if new). */
        uint32_t y0 = 0, y1 = h;
        if (im->valid) {
            uint64_t lo_p = UINT64_MAX, hi_p = 0;
            for (uint64_t p = vk_pg(addr); p <= vk_pg(hi - 1) && p < V.npages; p++) {
                if (V.page_gen[p] > im->gen) {
                    lo_p = MIN(lo_p, p);
                    hi_p = p;
                }
            }
            uint64_t blo = MAX(lo_p * VK_PAGE, (uint64_t)addr);
            uint64_t bhi = MIN((hi_p + 1) * VK_PAGE, hi);
            y0 = (blo - addr) / pitch;
            y1 = MIN((uint32_t)((bhi - addr + pitch - 1) / pitch), h);
        }
        /* Rendering over those rows that VRAM lacks goes there first. */
        vk_writeback_range(addr + (uint64_t)y0 * pitch, addr + (uint64_t)y1 * pitch);
        VkCommandBuffer cb = vk_batch()->cb;
        VkBufferImageCopy r = vk_region(im, 0, y0, w, y1);
        vk_end_pass();
        vk_full_barrier(cb);
        vkCmdCopyBufferToImage(cb, V.vram_buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        vk_full_barrier(cb);
        vk_note_read(addr + (uint64_t)y0 * pitch, addr + (uint64_t)y1 * pitch);
        im->valid = true;
        im->gen = gen;
        im->hash = vk_check() && cls == IMG_TEX && !vk_range_pending(addr, hi) ?
                   vk_img_hash(addr, w, h, pitch, bpp) : 0;
        V.stat_uploads++;
    }
    return im;
}

/*
 * A draw renders into rectangle [x0, x1) x [y0, y1) of im: the pages of
 * those rows get a new generation that im holds, and the rectangle is
 * written back with the batch.
 */
static void vk_img_written(VkImg *im, uint32_t x0, uint32_t y0,
                           uint32_t x1, uint32_t y1)
{
    uint32_t s = vk_batch()->seq;
    bool fresh = im->gen == V.gen;     /* nothing else happened since */

    x1 = MIN(x1, im->width);
    y1 = MIN(y1, im->height);
    if (x0 >= x1 || y0 >= y1) {
        return;
    }
    /* Keep the rectangle: inside one we have, merged into the one it
     * grows least, or new. */
    int best = -1;
    uint64_t grow = UINT64_MAX;
    for (uint32_t i = 0; i < im->ndr; i++) {
        uint32_t *d = im->dr[i];
        uint64_t a0 = (uint64_t)(d[2] - d[0]) * (d[3] - d[1]);
        uint64_t a1 = (uint64_t)(MAX(d[2], x1) - MIN(d[0], x0)) *
                      (MAX(d[3], y1) - MIN(d[1], y0));
        if (a1 - a0 < grow) {
            grow = a1 - a0;
            best = i;
        }
    }
    if (best >= 0 && (grow == 0 || im->ndr == VK_MAX_DR)) {
        uint32_t *d = im->dr[best];
        d[0] = MIN(d[0], x0);
        d[1] = MIN(d[1], y0);
        d[2] = MAX(d[2], x1);
        d[3] = MAX(d[3], y1);
    } else {
        uint32_t *d = im->dr[im->ndr++];
        d[0] = x0;
        d[1] = y0;
        d[2] = x1;
        d[3] = y1;
    }
    im->dirty = true;
    im->valid = true;
    if (grow == 0 && fresh) {
        return;                         /* those pages are already stamped */
    }
    uint64_t lo = im->addr + (uint64_t)y0 * im->pitch + (uint64_t)x0 * im->bpp;
    uint64_t hi = im->addr + (uint64_t)(y1 - 1) * im->pitch + (uint64_t)x1 * im->bpp;
    V.gen++;
    for (uint64_t p = vk_pg(lo); p <= vk_pg(hi - 1) && p < V.npages; p++) {
        V.page_gen[p] = V.gen;
        V.page_wseq[p] = s;
    }
    im->gen = V.gen;
}

/* ---- commit / flush --------------------------------------------------- */

static uint32_t vk_commit(void (*done)(void *, uint32_t), void *arg)
{
    VkBatch *b = V.cur;

    if (!b) {
        return 0;
    }
    vk_end_pass();
    /* What the device marked dirty for this batch's draws is ours. */
    vk_harvest();
    for (int i = 0; i < VK_MAX_IMG; i++) {
        if (V.img[i].live && V.img[i].dirty) {
            vk_copy_to_vram(&V.img[i]);
        }
    }
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(b->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(b->cb);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &b->cb,
    };
    VkResult r = vkQueueSubmit(V.queue, 1, &si, b->fence);
    if (r != VK_SUCCESS) {
        vk_fail("vkQueueSubmit failed (VkResult %d)", (int)r);
    }
    b->state = 2;
    V.cur = NULL;
    V.chunk = NULL;

    VkWaitItem *it = g_new0(VkWaitItem, 1);
    it->fence = b->fence;
    it->seq = b->seq;
    it->done = done;
    it->arg = arg;
    qemu_mutex_lock(&V.lock);
    g_queue_push_tail(V.waitq, it);
    qemu_cond_signal(&V.wake);
    qemu_mutex_unlock(&V.lock);
    return b->seq;
}

static uint32_t vk_submit_r200(void *opaque, void (*done)(void *, uint32_t), void *arg)
{
    return vk_commit(done, arg);
}

static bool vk_flush_r200(void *opaque)
{
    VkBatch *newest = NULL;

    vk_commit(NULL, NULL);
    for (int i = 0; i < VK_NBATCH; i++) {
        VkBatch *b = &V.batch[i];
        if (b->state == 2 && !vk_seq_done(b->seq) &&
            (!newest || (int32_t)(b->seq - newest->seq) > 0)) {
            newest = b;
        }
    }
    if (!newest) {
        return false;
    }
    /* A barrier opens every batch, so the newest finishing means all did. */
    vkWaitForFences(V.dev, 1, &newest->fence, VK_TRUE, UINT64_MAX);
    vk_set_done(newest->seq);
    V.stat_flushes++;
    return true;
}

static bool vk_range_busy_r200(void *opaque, uint64_t lo, uint64_t hi, bool write_access)
{
    if (!V.vram || hi <= lo) {
        return false;
    }
    hi = MIN(hi, V.vram_size);
    for (uint64_t p = vk_pg(lo); p <= vk_pg(hi - 1) && p < V.npages; p++) {
        if (vk_page_pending_write(p)) {
            return true;
        }
        if (write_access) {
            uint32_t s = V.page_rseq[p];
            if (s && ((V.cur && s == V.cur->seq) || !vk_seq_done(s))) {
                return true;
            }
        }
    }
    return false;
}

static uint32_t vk_zpass_r300(void *opaque, bool reset, uint32_t value)
{
    uint32_t v;

    if (!V.zpass) {
        return 0;
    }
    vk_flush_r200(opaque);
    v = V.zpass[0];
    if (reset) {
        V.zpass[0] = value;
    }
    return v;
}

static void vk_set_dirty_source(void *opaque, PPCMacGPUDirtyFn fn, void *arg)
{
    V.dirty_fn = fn;
    V.dirty_arg = arg;
}

/* ---- render passes, framebuffers, pipelines --------------------------- */

static VkRenderPass vk_render_pass(uint32_t n, const VkFormat *fmt)
{
    VkAttachmentDescription ad[VK_MAX_ATT];
    VkAttachmentReference ref[VK_MAX_ATT];

    for (unsigned i = 0; i < V.nrp; i++) {
        if (V.rps[i].n == n && !memcmp(V.rps[i].fmt, fmt, n * sizeof(*fmt))) {
            return V.rps[i].rp;
        }
    }
    for (uint32_t k = 0; k < n; k++) {
        ad[k] = (VkAttachmentDescription){
            .format = fmt[k],
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        ref[k] = (VkAttachmentReference){ k, VK_IMAGE_LAYOUT_GENERAL };
    }
    VkSubpassDescription sd = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = n,
        .pInputAttachments = ref,
        .colorAttachmentCount = n,
        .pColorAttachments = ref,
    };
    VkSubpassDependency dep[3] = {
        {   /* between draws: writes before the next draw's input reads */
            .srcSubpass = 0, .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        },
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        },
        {
            .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        },
    };
    VkRenderPassCreateInfo rci = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = n,
        .pAttachments = ad,
        .subpassCount = 1,
        .pSubpasses = &sd,
        .dependencyCount = 3,
        .pDependencies = dep,
    };
    VkRenderPass rp;
    if (vkCreateRenderPass(V.dev, &rci, NULL, &rp) != VK_SUCCESS || V.nrp == VK_MAX_RP) {
        vk_warn(1u << 0, "render pass creation failed");
        return VK_NULL_HANDLE;
    }
    V.rps[V.nrp].n = n;
    memcpy(V.rps[V.nrp].fmt, fmt, n * sizeof(*fmt));
    V.rps[V.nrp++].rp = rp;
    return rp;
}

static VkFramebuffer vk_framebuffer(VkRenderPass rp, uint32_t n, VkImg **att,
                                    uint32_t w, uint32_t h)
{
    VkImageView views[VK_MAX_ATT];

    for (uint32_t k = 0; k < n; k++) {
        views[k] = att[k]->view;
    }
    for (int i = 0; i < VK_MAX_FB; i++) {
        VkFB *f = &V.fbs[i];
        if (f->fb && f->rp == rp && f->n == n && f->w == w && f->h == h &&
            !memcmp(f->view, views, n * sizeof(views[0]))) {
            return f->fb;
        }
    }
    VkFB *f = &V.fbs[V.fb_next];
    V.fb_next = (V.fb_next + 1) % VK_MAX_FB;
    if (f->fb) {
        vk_trash(NULL, TRASH_FB, (uint64_t)(uintptr_t)f->fb);
        f->fb = VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = rp,
        .attachmentCount = n,
        .pAttachments = views,
        .width = w,
        .height = h,
        .layers = 1,
    };
    if (vkCreateFramebuffer(V.dev, &fci, NULL, &f->fb) != VK_SUCCESS) {
        f->fb = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    f->rp = rp;
    f->n = n;
    f->w = w;
    f->h = h;
    memcpy(f->view, views, n * sizeof(views[0]));
    return f->fb;
}

static VkShaderModule vk_module(VkProg *pg, const char *glsl, R300Stage stage)
{
    if (pg->mod[stage] || pg->failed[stage]) {
        return pg->mod[stage];
    }
    size_t nw;
    char *err = NULL;
    uint32_t *spv = r300_glsl_to_spirv(glsl, stage, &nw, &err);
    if (!spv) {
        qemu_log("ppc-mac-gpu vulkan: %s: %s\n%s\n", r300_stage_entry(stage), err, glsl);
        free(err);
        pg->failed[stage] = true;
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo mci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = nw * 4,
        .pCode = spv,
    };
    if (vkCreateShaderModule(V.dev, &mci, NULL, &pg->mod[stage]) != VK_SUCCESS) {
        pg->mod[stage] = VK_NULL_HANDLE;
        pg->failed[stage] = true;
    }
    free(spv);
    return pg->mod[stage];
}

static VkPipeline vk_pipeline(const char *glsl, const VkPipeKey *key)
{
    VkProg *pg = g_hash_table_lookup(V.progs, glsl);

    if (!pg) {
        pg = g_new0(VkProg, 1);
        pg->pipes = g_array_new(FALSE, FALSE, sizeof(VkPipeVar));
        g_hash_table_insert(V.progs, g_strdup(glsl), pg);
        if (g_hash_table_size(V.progs) % 16 == 1) {
            qemu_log("ppc-mac-gpu vulkan: %u fragment programs\n",
                     g_hash_table_size(V.progs));
        }
    }
    for (guint i = 0; i < pg->pipes->len; i++) {
        VkPipeVar *pv = &g_array_index(pg->pipes, VkPipeVar, i);
        if (!memcmp(&pv->key, key, sizeof(*key))) {
            return pv->pipe;
        }
    }
    VkShaderModule vs = vk_module(pg, glsl, R300_STAGE_VS);
    VkShaderModule fs = vk_module(pg, glsl, key->z ? R300_STAGE_FS_Z : R300_STAGE_FS);
    if (!vs || !fs) {
        return VK_NULL_HANDLE;
    }
    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = key->topo,
    };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = V.depth_clamp,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = key->cull,
        .frontFace = key->front,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineColorBlendAttachmentState cba[VK_MAX_ATT];
    uint32_t natt = key->ncb + (key->z ? 1 : 0);
    for (uint32_t k = 0; k < natt; k++) {
        cba[k] = (VkPipelineColorBlendAttachmentState){
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
    }
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = natt,
        .pAttachments = cba,
    };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn,
    };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = st,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &cb,
        .pDynamicState = &ds,
        .layout = V.layout,
        .renderPass = key->rp,
        .subpass = 0,
    };
    VkPipeVar pv = { *key, VK_NULL_HANDLE };
    if (vkCreateGraphicsPipelines(V.dev, VK_NULL_HANDLE, 1, &gci, NULL, &pv.pipe) != VK_SUCCESS) {
        vk_warn(1u << 1, "pipeline creation failed");
        return VK_NULL_HANDLE;
    }
    g_array_append_val(pg->pipes, pv);
    return pv.pipe;
}

/* ---- textures --------------------------------------------------------- */

static VkSamplerAddressMode vk_wrap(uint32_t w)
{
    switch (w & 7) {
    case 0:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 1:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 3:
    case 5:
    case 7:  return V.mirror_clamp ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
                                   : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 6:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;  /* r300_tex adds the colour */
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

/* As r300_sampler in the Metal backend. */
static VkSampler vk_sampler(uint32_t f0, uint32_t levels)
{
    uint32_t key = (f0 & 0xFFFFFF) | (MIN(levels, 15u) << 24);
    VkSampler s = g_hash_table_lookup(V.samplers, GUINT_TO_POINTER(key + 1));

    if (s) {
        return s;
    }
    uint32_t mag = (f0 >> 9) & 3, min = (f0 >> 11) & 3, mip = (f0 >> 13) & 3;
    bool mipped = levels > 1 && mip != 0;
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = mag == 1 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
        .minFilter = min == 1 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
        .mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = vk_wrap(f0),
        .addressModeV = vk_wrap(f0 >> 3),
        .addressModeW = vk_wrap(f0 >> 6),
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .minLod = mipped ? MIN((f0 >> 17) & 0xF, levels - 1) : 0,
        .maxLod = mipped ? levels - 1 : 0,
    };
    if (V.aniso && (mag == 3 || min == 3) && ((f0 >> 21) & 7)) {
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy = MIN((float)(1u << MIN((f0 >> 21) & 7, 4u)),
                                V.props.limits.maxSamplerAnisotropy);
    }
    if (vkCreateSampler(V.dev, &sci, NULL, &s) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    g_hash_table_insert(V.samplers, GUINT_TO_POINTER(key + 1), s);
    return s;
}

/* 1x1 stand-ins for unbound units: 2D, 3D, cube. */
static bool vk_dummies(void)
{
    static const VkImageType it[3] = { VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_2D };
    static const VkImageViewType vt[3] = { VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_3D,
                                           VK_IMAGE_VIEW_TYPE_CUBE };
    VkClearColorValue zero = { { 0, 0, 0, 0 } };

    if (V.dummy_ready) {
        return true;
    }
    VkCommandBuffer cb = vk_batch()->cb;
    for (int i = 0; i < 3; i++) {
        uint32_t layers = i == 2 ? 6 : 1;
        if (!vk_image(it[i], vt[i], VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, layers,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      &V.dummy_img[i], &V.dummy_mem[i], &V.dummy_view[i])) {
            return false;
        }
        vk_layout_general(cb, V.dummy_img[i], 1, layers);
        VkImageSubresourceRange sr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
        vkCmdClearColorImage(cb, V.dummy_img[i], VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &sr);
    }
    vk_full_barrier(cb);
    V.dummy_ready = true;
    return true;
}

static VkFormat vk_tex_format(uint32_t kind)
{
    switch (kind) {
    case R300_TEXK_R8:   return VK_FORMAT_R8_UNORM;
    case R300_TEXK_RG8:  return VK_FORMAT_R8G8_UNORM;
    case R300_TEXK_DXT1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case R300_TEXK_DXT3: return VK_FORMAT_BC2_UNORM_BLOCK;
    case R300_TEXK_DXT5: return VK_FORMAT_BC3_UNORM_BLOCK;
    default:             return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

/*
 * Textures rebuilt from their bytes on the CPU and cached by a hash of
 * them, as r300_texture_full does for Metal: mip chains, 3D and cube
 * maps, 16bpp formats, DXT, and anything copied out of the GART.  The
 * caller has made sure no GPU work still writes the bytes.
 */
static VkImageView vk_texture_full(const R300TexDesc *td)
{
    const uint8_t *src = td->host_data ? td->host_data : V.vram + td->gpu_addr;
    bool dxt = td->kind >= R300_TEXK_DXT1;
    VkFormat fmt = vk_tex_format(td->kind);
    VkTexKey key = { td->gpu_addr, td->format, td->kind, td->width, td->height,
                     td->depth, td->dim, td->levels, td->pitch_bytes,
                     td->host_data != NULL ? 1u + (td->endian << 1) : td->endian << 1, r300_tex_identity(td, src) };
    int lru = 0;

    if (dxt && !V.bc) {
        vk_warn(1u << 2, "DXT texture on a GPU without BC formats");
        return VK_NULL_HANDLE;
    }
    if (td->dim == R300_TEXDIM_CUBE && td->width != td->height) {
        vk_warn(1u << 3, "cube map with non-square faces");
        return VK_NULL_HANDLE;
    }
    for (int i = 0; i < VK_MAX_TEXFULL; i++) {
        VkTexFull *t = &V.texfull[i];
        if (t->live && !memcmp(&t->key, &key, sizeof(key))) {
            t->used = ++V.clock;
            return t->view;
        }
        if (!t->live || (V.texfull[lru].live && t->used < V.texfull[lru].used)) {
            lru = i;
        }
    }
    VkTexFull *t = &V.texfull[lru];
    if (t->live) {
        vk_trash(NULL, TRASH_VIEW, (uint64_t)(uintptr_t)t->view);
        vk_trash(NULL, TRASH_IMAGE, (uint64_t)(uintptr_t)t->img);
        vk_trash(NULL, TRASH_MEM, (uint64_t)(uintptr_t)t->mem);
        t->live = false;
    }
    uint32_t w = dxt ? (td->width + 3) & ~3u : td->width;
    uint32_t h = dxt ? (td->height + 3) & ~3u : td->height;
    bool cube = td->dim == R300_TEXDIM_CUBE, vol = td->dim == R300_TEXDIM_3D;
    if (!vk_image(vol ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                  vol ? VK_IMAGE_VIEW_TYPE_3D : cube ? VK_IMAGE_VIEW_TYPE_CUBE
                                                     : VK_IMAGE_VIEW_TYPE_2D,
                  fmt, w, h, vol ? td->depth : 1, td->levels, cube ? 6 : 1,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  &t->img, &t->mem, &t->view)) {
        return VK_NULL_HANDLE;
    }
    VkCommandBuffer cb = vk_batch()->cb;
    vk_end_pass();
    vk_layout_general(cb, t->img, td->levels, cube ? 6 : 1);
    for (uint32_t l = 0; l < td->levels; l++) {
        uint32_t lw, lh, n, bpr;
        r300_tex_level_dims(td, l, &lw, &lh, &n);
        uint32_t mw = MAX(w >> l, 1u), mh = MAX(h >> l, 1u);
        uint32_t cw = dxt ? mw : lw, ch = dxt ? mh : lh;
        size_t face = (size_t)td->lvl_pitch[l] * td->lvl_rows[l];
        for (uint32_t f = 0; f < n; f++) {
            uint8_t *bytes = r300_tex_level_bytes(td, src + td->lvl_off[l] + f * face,
                                                  l, cw, ch, &bpr);
            size_t len = dxt ? (size_t)bpr * ((ch + 3) / 4) : (size_t)bpr * ch;
            VkBuffer sb;
            VkDeviceSize so;
            void *dst = vk_arena(len, &sb, &so);
            if (!dst) {
                free(bytes);
                continue;
            }
            memcpy(dst, bytes, len);
            free(bytes);
            VkBufferImageCopy r = {
                .bufferOffset = so,
                .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, cube ? f : 0, 1 },
                .imageOffset = { 0, 0, vol ? (int32_t)f : 0 },
                .imageExtent = { cw, ch, 1 },
            };
            cb = vk_batch()->cb;
            vkCmdCopyBufferToImage(cb, sb, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        }
    }
    vk_full_barrier(cb);
    t->live = true;
    t->key = key;
    t->used = ++V.clock;
    return t->view;
}

/* ---- draws ------------------------------------------------------------ */

static VkFormat vk_rt_format(uint32_t view)
{
    switch (view) {
    case R300_RTV_R8U:     return VK_FORMAT_R8_UINT;
    case R300_RTV_R16U:    return VK_FORMAT_R16_UINT;
    case R300_RTV_R32U:    return VK_FORMAT_R32_UINT;
    case R300_RTV_RG32U:   return VK_FORMAT_R32G32_UINT;
    case R300_RTV_RGBA32U: return VK_FORMAT_R32G32B32A32_UINT;
    default:               return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

static bool vk_range_pending(uint64_t lo, uint64_t hi)
{
    for (uint64_t p = vk_pg(lo); p <= vk_pg(hi - 1) && p < V.npages; p++) {
        if (vk_page_pending_write(p)) {
            return true;
        }
    }
    return false;
}

static int vk_draw_r300(void *opaque, uint8_t *vram_ptr, uint64_t vram_size,
                        const R300DrawPacket *pkt)
{
    uint32_t ncb = MAX(pkt->num_cb, 1u);
    uint32_t sx0 = pkt->scissor[0], sy0 = pkt->scissor[1];
    uint32_t sx1 = MIN(pkt->scissor[2], pkt->rt_width);
    uint32_t sy1 = MIN(pkt->scissor[3], pkt->rt_height);
    uint32_t ns = MIN(MAX(pkt->aa_samples, 1u), 6u);
    VkImg *att[VK_MAX_ATT];
    VkFormat fmt[VK_MAX_ATT];
    uint32_t addr[VK_MAX_ATT], bpr[VK_MAX_ATT], bpp[VK_MAX_ATT];
    uint32_t fbw;

    if (!V.vram || vram_ptr != V.vram) {
        return -1;
    }
    if (sx0 >= sx1 || sy0 >= sy1 || !(pkt->num_verts + pkt->num_line_verts)) {
        return 0;
    }
    if ((pkt->cull & (R300_CULL_FRONT | R300_CULL_BACK)) ==
        (R300_CULL_FRONT | R300_CULL_BACK) && !pkt->num_line_verts) {
        return 0;                       /* every polygon culled */
    }
    if (ns > 1) {
        bool same_pitch = !pkt->depth.attach || pkt->depth.pitch == pkt->rt_pitch;
        for (uint32_t k = 1; k < ncb; k++) {
            same_pitch &= pkt->cb[k].pitch == pkt->rt_pitch;
        }
        if (!same_pitch || (uint64_t)ns * pkt->rt_pitch > V.props.limits.maxImageDimension2D) {
            vk_warn(1u << 4, "multisampled buffers of different pitches");
            return -1;
        }
    }
    /* The framebuffer: sample k of row y is row y * ns + k (see Metal). */
    fbw = ns > 1 ? ns * pkt->rt_pitch : pkt->rt_width;
    for (uint32_t k = 0; k < ncb; k++) {
        uint32_t view = k ? pkt->cb[k].view : pkt->rt_view;
        addr[k] = k ? pkt->cb[k].gpu_addr : pkt->rt_gpu_addr;
        bpp[k] = k ? pkt->cb[k].bpp : pkt->rt_bpp;
        bpr[k] = ns * (k ? pkt->cb[k].pitch : pkt->rt_pitch) * bpp[k];
        fmt[k] = vk_rt_format(view);
    }
    const R300DepthDesc *zd = &pkt->depth;
    bool want_z = zd->attach;
    if (want_z && (zd->pitch < pkt->rt_width || zd->gpu_addr % 4)) {
        vk_warn(1u << 5, "depth buffer not usable");
        want_z = false;
    }

    vk_harvest();

    /*
     * Textures whose bytes the CPU reads (rebuilt ones, and anything in
     * VRAM a batch still renders into) first: that may need a flush,
     * which must not come in the middle of recording this draw.
     */
    R300FSUniforms u = pkt->uniforms;
    bool need_flush = false;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &pkt->tex[t];
        bool direct = td->levels <= 1 && td->dim == R300_TEXDIM_2D &&
                      (td->kind == R300_TEXK_RGBA8 || td->kind == R300_TEXK_R8 ||
                       td->kind == R300_TEXK_RG8);
        if (td->bound && !td->host_data && td->kind != R300_TEXK_RAW && !direct &&
            (uint64_t)td->gpu_addr + td->size_bytes <= V.vram_size &&
            vk_range_pending(td->gpu_addr, (uint64_t)td->gpu_addr + td->size_bytes)) {
            need_flush = true;
        }
    }
    if (need_flush) {
        vk_flush_r200(opaque);
    }
    if (!vk_dummies()) {
        return -1;
    }

    VkDescriptorImageInfo tii[R300_NUM_TEX_UNITS];
    uint32_t aux_bytes = 0;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &pkt->tex[t];
        bool raw = td->kind == R300_TEXK_RAW;
        VkImageView view = VK_NULL_HANDLE;

        tii[t] = (VkDescriptorImageInfo){
            vk_sampler(td->filter0, td->levels),
            V.dummy_view[raw ? 0 : MIN(td->dim, 2u)], VK_IMAGE_LAYOUT_GENERAL };
        u.tex_addr[t][0] = u.tex_addr[t][1] = 0;
        if (!td->bound) {
            u.tex_info[t][0] = 0;
            continue;
        }
        if (!td->host_data && (uint64_t)td->gpu_addr + td->size_bytes > V.vram_size) {
            vk_warn(1u << 6, "texture outside VRAM");
            u.tex_info[t][0] = 0;
            continue;
        }
        if (raw) {
            if (td->host_data) {
                u.tex_addr[t][0] = aux_bytes;
                u.tex_addr[t][1] = 1;
                aux_bytes += ROUND_UP(td->size_bytes, 16);
            } else {
                /* The shader reads VRAM: rendering to it must be there. */
                u.tex_addr[t][0] = td->gpu_addr;
                vk_writeback_range(td->gpu_addr, (uint64_t)td->gpu_addr + td->size_bytes);
                vk_note_read(td->gpu_addr, (uint64_t)td->gpu_addr + td->size_bytes);
            }
            continue;
        }
        bool direct = !td->host_data && td->levels <= 1 && td->dim == R300_TEXDIM_2D &&
                      (td->kind == R300_TEXK_RGBA8 || td->kind == R300_TEXK_R8 ||
                       td->kind == R300_TEXK_RG8);
        if (direct) {
            uint32_t b = td->kind == R300_TEXK_RGBA8 ? 4 : td->kind == R300_TEXK_RG8 ? 2 : 1;
            uint32_t w = td->width, h = td->height;
            /* like r200_view: shrink what does not fit the pitch or VRAM */
            if ((uint64_t)w * b > td->pitch_bytes) {
                w = td->pitch_bytes / b;
            }
            if (w && (uint64_t)td->gpu_addr + (uint64_t)(h - 1) * td->pitch_bytes +
                     (uint64_t)w * b > V.vram_size) {
                h = (V.vram_size - td->gpu_addr - (uint64_t)w * b) / td->pitch_bytes + 1;
            }
            VkImg *im = vk_img_get(IMG_TEX, td->gpu_addr, w, h, td->pitch_bytes,
                                   vk_tex_format(td->kind), b);
            if (im) {
                view = im->view;
            } else if (!vk_range_pending(td->gpu_addr, (uint64_t)td->gpu_addr + td->size_bytes)) {
                view = vk_texture_full(td);     /* odd pitch: repacked on the CPU */
            }
        } else {
            view = vk_texture_full(td);
        }
        if (!view) {
            u.tex_info[t][0] = 0;
            continue;
        }
        tii[t].imageView = view;
    }

    /* Shader-decoded texels copied out of the GART. */
    VkBuffer auxb = VK_NULL_HANDLE;
    VkDeviceSize auxo = 0;
    uint8_t *auxp = vk_arena(MAX(aux_bytes, 16u), &auxb, &auxo);
    if (!auxp) {
        return -1;
    }
    for (int t = 0; t < R300_NUM_TEX_UNITS && aux_bytes; t++) {
        const R300TexDesc *td = &pkt->tex[t];
        if (td->bound && td->kind == R300_TEXK_RAW && td->host_data) {
            uint32_t *dw = (uint32_t *)(auxp + u.tex_addr[t][0]);
            memcpy(dw, td->host_data, td->size_bytes);
            /* GART dwords reversed to VRAM's side (r300_tex_raw_bpp) */
            for (uint32_t i = 0; i < ROUND_UP(td->size_bytes, 4) / 4; i++) {
                dw[i] = bswap32(dw[i]);
            }
        }
    }

    /* Colour buffers and the depth buffer. */
    for (uint32_t k = 0; k < ncb; k++) {
        att[k] = vk_img_get(IMG_RT, addr[k], fbw, pkt->rt_height, bpr[k], fmt[k], bpp[k]);
        if (!att[k]) {
            if (k == 0) {
                vk_warn(1u << 7, "colour buffer unusable");
                return -1;
            }
            vk_warn(1u << 8, "render target B-D unusable");
            return -1;
        }
    }
    uint32_t natt = ncb;
    bool z16 = zd->bpp == 2;
    if (want_z) {
        VkFormat zf = z16 ? VK_FORMAT_R16_UINT : VK_FORMAT_R32_UINT;
        uint32_t zw = ns > 1 ? ns * zd->pitch : pkt->rt_width;
        att[ncb] = vk_img_get(IMG_RT, zd->gpu_addr, zw, pkt->rt_height,
                              ns * zd->pitch * zd->bpp, zf, zd->bpp);
        if (att[ncb] && zw == fbw) {
            fmt[ncb] = zf;
            natt++;
        } else {
            vk_warn(1u << 9, "depth buffer not usable in place");
            want_z = false;
        }
    }
    /* Keep the open pass's depth buffer bound for depth-less draws. */
    if (!want_z && V.in_pass && V.pass_z && V.pass_n == ncb + 1) {
        bool same = true;
        for (uint32_t k = 0; k < ncb; k++) {
            same &= V.pass_img[k] == att[k];
        }
        if (same) {
            att[ncb] = V.pass_img[ncb];
            fmt[ncb] = att[ncb]->fmt;
            natt++;
            z16 = V.pass_z16;
            u.zinfo[0] = 0;
            u.zinfo[3] = z16 ? R300_ZFMT_Z16 : 0;
            u.poly_en = 0;
        }
    }
    bool pass_z = natt > ncb;

    VkRenderPass rp = vk_render_pass(natt, fmt);
    if (!rp) {
        return -1;
    }
    VkPipeKey pk;
    memset(&pk, 0, sizeof(pk));         /* compared with memcmp: no stray padding */
    pk.rp = rp;
    pk.ncb = ncb;
    pk.z = pass_z;
    pk.topo = pkt->prim_class == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                   : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pk.cull = (pkt->cull & R300_CULL_FRONT) ? VK_CULL_MODE_FRONT_BIT :
              (pkt->cull & R300_CULL_BACK) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    pk.front = r300_front_ccw(pkt->cull) ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                         : VK_FRONT_FACE_CLOCKWISE;
    VkPipeline pipe = VK_NULL_HANDLE, lpipe = VK_NULL_HANDLE;
    if (pkt->num_verts) {
        pipe = vk_pipeline(pkt->glsl, &pk);
        if (!pipe) {
            return -1;
        }
    }
    if (pkt->num_line_verts) {
        VkPipeKey lk = pk;
        lk.topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        lk.cull = VK_CULL_MODE_NONE;    /* polygon-mode edges: never culled */
        lpipe = vk_pipeline(pkt->glsl, &lk);
        if (!lpipe) {
            return -1;
        }
    }

    /* Per-draw data: vertices, and each sample's uniforms and shift. */
    uint32_t nv = pkt->num_verts + pkt->num_line_verts;
    VkBuffer vb, ub;
    VkDeviceSize vo, uo;
    void *vp = vk_arena((VkDeviceSize)nv * sizeof(R300Vertex), &vb, &vo);
    VkDeviceSize ustride = ROUND_UP(sizeof(R300FSUniforms), V.align);
    VkDeviceSize mstride = V.align;
    uint8_t *up = vk_arena((ustride + mstride) * ns, &ub, &uo);
    if (!vp || !up) {
        return -1;
    }
    memcpy(vp, pkt->verts, (size_t)nv * sizeof(R300Vertex));
    for (uint32_t k = 0; k < ns; k++) {
        R300FSUniforms *uk = (R300FSUniforms *)(up + k * ustride);
        float *ms = (float *)(up + ns * ustride + k * mstride);
        *uk = u;
        memset(ms, 0, 16);
        if (ns > 1) {
            uint32_t cx = k * pkt->rt_pitch;
            ms[0] = -2.0f * pkt->aa_pos[k][0] / pkt->rt_width;
            ms[1] = 2.0f * pkt->aa_pos[k][1] / pkt->rt_height;
            for (int i = 0; i < 4; i++) {
                uk->cliprect[i][0] += cx;
                uk->cliprect[i][2] += cx;
            }
        }
    }

    /* The descriptor set. */
    VkBatch *b = vk_batch();
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (;;) {
        if (b->pool_cur == b->pools->len) {
            VkDescriptorPoolSize ps[4] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 * VK_SETS_PER_POOL },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * VK_SETS_PER_POOL },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_MAX_ATT * VK_SETS_PER_POOL },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  R300_NUM_TEX_UNITS * VK_SETS_PER_POOL },
            };
            VkDescriptorPoolCreateInfo pci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = VK_SETS_PER_POOL,
                .poolSizeCount = 4,
                .pPoolSizes = ps,
            };
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(V.dev, &pci, NULL, &pool) != VK_SUCCESS) {
                return -1;
            }
            g_array_append_val(b->pools, pool);
        }
        VkDescriptorSetAllocateInfo dai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g_array_index(b->pools, VkDescriptorPool, b->pool_cur),
            .descriptorSetCount = 1,
            .pSetLayouts = &V.dsl,
        };
        if (vkAllocateDescriptorSets(V.dev, &dai, &set) == VK_SUCCESS) {
            break;
        }
        b->pool_cur++;
    }
    VkDescriptorBufferInfo bi[6] = {
        { ub, uo, sizeof(R300FSUniforms) },
        { V.zpass_buf, 0, 16 },
        { vb, vo, (VkDeviceSize)nv * sizeof(R300Vertex) },
        { ub, uo + ns * ustride, 16 },
        { V.vram_buf, 0, VK_WHOLE_SIZE },
        { auxb, auxo, MAX(aux_bytes, 16u) },
    };
    VkDescriptorImageInfo ai[VK_MAX_ATT];
    VkWriteDescriptorSet wr[6 + 2];
    uint32_t nwr = 0;
    static const VkDescriptorType bt[6] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    for (uint32_t i = 0; i < 6; i++) {
        wr[nwr++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = bt[i], .pBufferInfo = &bi[i] };
    }
    for (uint32_t k = 0; k < natt; k++) {
        ai[k] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, att[k]->view, VK_IMAGE_LAYOUT_GENERAL };
    }
    wr[nwr++] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = R300_BIND_FB0, .descriptorCount = natt,
        .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, .pImageInfo = ai };
    wr[nwr++] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = R300_BIND_TEX0, .descriptorCount = R300_NUM_TEX_UNITS,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = tii };
    vkUpdateDescriptorSets(V.dev, nwr, wr, 0, NULL);

    /* The render pass: reuse the open one if it has these attachments. */
    VkCommandBuffer cb = b->cb;
    bool same = V.in_pass && V.pass_rp == rp && V.pass_n == natt;
    for (uint32_t k = 0; same && k < natt; k++) {
        same = V.pass_img[k] == att[k];
    }
    if (!same) {
        vk_end_pass();
        VkFramebuffer fb = vk_framebuffer(rp, natt, att, fbw, pkt->rt_height);
        if (!fb) {
            return -1;
        }
        VkRenderPassBeginInfo rbi = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = rp,
            .framebuffer = fb,
            .renderArea = { { 0, 0 }, { fbw, pkt->rt_height } },
        };
        vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        V.in_pass = true;
        V.pass_rp = rp;
        V.pass_fb = fb;
        V.pass_n = natt;
        memcpy(V.pass_img, att, natt * sizeof(att[0]));
        V.pass_z = pass_z;
        V.pass_z16 = z16;
        V.stat_passes++;
    } else {
        /* This draw reads what the previous one wrote.  MoltenVK needs it
         * too: without it blending read stale destination pixels (shadows,
         * translucent menus, text left garbage behind). */
        VkMemoryBarrier mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
        };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &mb, 0, NULL, 0, NULL);
    }

    for (uint32_t k = 0; k < ns; k++) {
        uint32_t cx = k * pkt->rt_pitch * (ns > 1);
        uint32_t dyn[2] = { k * ustride, k * mstride };
        /* Clip space has y up (as Metal's): flip the viewport. */
        VkViewport vpt = { cx, pkt->rt_height, pkt->rt_width, -(float)pkt->rt_height, 0, 1 };
        VkRect2D sc = { { cx + sx0, sy0 }, { sx1 - sx0, sy1 - sy0 } };
        vkCmdSetViewport(cb, 0, 1, &vpt);
        vkCmdSetScissor(cb, 0, 1, &sc);
        if (pkt->num_verts) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.layout, 0, 1,
                                    &set, 2, dyn);
            vkCmdDraw(cb, pkt->num_verts, 1, 0, 0);
        }
        if (pkt->num_line_verts) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, lpipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.layout, 0, 1,
                                    &set, 2, dyn);
            vkCmdDraw(cb, pkt->num_line_verts, 1, pkt->num_verts, 0);
        }
    }
    /* What the scissor let through, over every sample's columns. */
    for (uint32_t k = 0; k < natt; k++) {
        vk_img_written(att[k], sx0, sy0, (ns - 1) * pkt->rt_pitch * (ns > 1) + sx1, sy1);
    }
    V.stat_draws++;
    if (V.stat_draws % 20000 == 0) {
        qemu_log("ppc-mac-gpu vulkan: %llu draws, %llu passes, %llu uploads, "
                 "%llu write-backs, %llu flushes\n",
                 (unsigned long long)V.stat_draws, (unsigned long long)V.stat_passes,
                 (unsigned long long)V.stat_uploads, (unsigned long long)V.stat_writebacks,
                 (unsigned long long)V.stat_flushes);
    }
    return 0;
}

/* ---- the renderer ----------------------------------------------------- */

static void *vk_init(uint8_t *vram_ptr, uint64_t vram_size)
{
    void *map;

    if (!V.ready || !V.vram || vram_ptr != V.vram) {
        vk_fail("VRAM was not allocated by the Vulkan backend");
        return NULL;
    }
    if (!V.zpass_buf) {
        if (!vk_buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                       &V.zpass_buf, &V.zpass_mem, &map)) {
            return NULL;
        }
        V.zpass = map;
        memset(map, 0, 16);
    }
    return &V;
}

static void vk_fini(void *opaque)
{
    vk_flush_r200(opaque);
}

static uint32_t vk_get_caps(void *opaque)
{
    return PPC_MAC_GPU_RENDERER_CAP_3D | PPC_MAC_GPU_RENDERER_CAP_ACCEL;
}

static PPCMacGPURenderer vulkan_renderer = {
    .name             = "vulkan",
    .r300_glsl_flags  = R300_GLSL_VRAM_SSBO,
    .init             = vk_init,
    .fini             = vk_fini,
    .draw_r300        = vk_draw_r300,
    .zpass_r300       = vk_zpass_r300,
    .flush_r200       = vk_flush_r200,
    .submit_r200      = vk_submit_r200,
    .range_busy_r200  = vk_range_busy_r200,
    .set_dirty_source = vk_set_dirty_source,
    .get_caps         = vk_get_caps,
};

PPCMacGPURenderer *ppc_mac_gpu_renderer_vulkan(void)
{
    return &vulkan_renderer;
}
