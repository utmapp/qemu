/*
 * PowerEmu gamepad: a USB game controller for the guest, driven by a real
 * controller on the host.
 *
 *   -device poweremu-gamepad,path=/tmp/x.sock
 *   -device poweremu-gamepad,chardev=pad0   (UTM: a spiceport chardev)
 *
 * The host's controller -- an Xbox or PlayStation pad over Bluetooth, say
 * -- speaks a language Mac OS X 10.4 was never taught.  macOS understands
 * all of them, so PowerEmu reads the controller there and sends its
 * positions here, and this device shows the guest an ordinary USB gamepad
 * of the kind any Mac of that age recognises without a driver.
 *
 * PowerEmu listens on the Unix socket at `path`; this device connects, and
 * reads fixed 9-byte readings:
 *
 *   0  left stick X       128 centred
 *   1  left stick Y
 *   2  right stick X
 *   3  right stick Y
 *   4  left trigger       0 released
 *   5  right trigger
 *   6  direction pad      0 up, then clockwise in eighths, 15 nothing
 *   7  buttons 1-8        one bit each
 *   8  buttons 9-16       13-16 are the direction pad again
 *
 * which is also what the guest reads, so nothing has to be rearranged in
 * between.
 *
 * Copyright (c) 2026 Spartan0285
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"
#include "hw/usb.h"
#include "hw/usb/desc.h"
#include "hw/usb/hid.h"
#include "hw/qdev-properties.h"
#include "hw/qdev-properties-system.h"
#include "io/channel-socket.h"
#include "chardev/char-fe.h"
#include "qom/object.h"

#define TYPE_POWEREMU_GAMEPAD "poweremu-gamepad"
OBJECT_DECLARE_SIMPLE_TYPE(PowerEmuPad, POWEREMU_GAMEPAD)

#define PAD_REPORT_LEN 9

struct PowerEmuPad {
    USBDevice dev;
    char *path;
    CharBackend chr;            /* instead of path: readings from a chardev */

    QIOChannelSocket *sioc;
    guint watch;
    bool connected;
    int64_t last_try_ms;

    uint8_t usage;              /* 4 Joystick (default), 5 Game Pad */
    uint8_t descriptor[128];    /* the report descriptor, with `usage` in it */
    uint8_t report[PAD_REPORT_LEN];
    uint8_t sent[PAD_REPORT_LEN];
    bool changed;
    uint8_t in[PAD_REPORT_LEN];
    size_t in_len;
};

enum {
    STR_MANUFACTURER = 1,
    STR_PRODUCT,
    STR_SERIALNUMBER,
    STR_CONFIG,
};

static const USBDescStrings desc_strings = {
    [STR_MANUFACTURER] = "PowerEmu",
    [STR_PRODUCT]      = "PowerEmu Gamepad",
    [STR_SERIALNUMBER] = "1",
    [STR_CONFIG]       = "Gamepad",
};

/*
 * Two sticks, two triggers, a direction pad and twelve buttons, in the
 * plainest form the HID specification offers: no report id, one byte per
 * axis, and the buttons last.  Mac OS X's own HID support picks this up
 * as a game controller with nothing installed.
 *
 * It calls itself a Joystick rather than a Game Pad by default: programs
 * of the age accept either, but some only ever looked for a joystick
 * (SDL 1.2 did until 2003), and nothing looks only for a game pad.  The
 * `usage` property changes it.
 *
 * "Logical Maximum 255" must be written as three bytes (0x26 0xff 0x00):
 * in the one-byte form Mac OS X 10.4 reads the value as -1 and every axis
 * comes out backwards.
 */
static const uint8_t pad_report_descriptor[] = {
    0x05, 0x01,        /* Usage Page (Generic Desktop)     */
    0x09, 0x04,        /* Usage (Joystick); see `usage`    */
    0xa1, 0x01,        /* Collection (Application)         */
    0xa1, 0x00,        /*   Collection (Physical)          */
    0x05, 0x01,        /*     Usage Page (Generic Desktop) */
    /*
     * The order the Xbox controller driver of this era published, which
     * is what games of the time were written against: the left stick on
     * X and Y, the right stick on Rx and Ry, and the triggers on Z and
     * Rz.  The triggers have to be the two that rest at one end rather
     * than in the middle, because a game reading them as steering axes
     * would otherwise see a control held hard over at all times.
     */
    0x09, 0x30,        /*     Usage (X)   left stick       */
    0x09, 0x31,        /*     Usage (Y)                    */
    0x09, 0x33,        /*     Usage (Rx)  right stick      */
    0x09, 0x34,        /*     Usage (Ry)                   */
    0x09, 0x32,        /*     Usage (Z)   left trigger     */
    0x09, 0x35,        /*     Usage (Rz)  right trigger    */
    0x15, 0x00,        /*     Logical Minimum (0)          */
    0x26, 0xff, 0x00,  /*     Logical Maximum (255)        */
    0x75, 0x08,        /*     Report Size (8)              */
    0x95, 0x06,        /*     Report Count (6)             */
    0x81, 0x02,        /*     Input (Data, Variable, Abs)  */

    0x05, 0x01,        /*     Usage Page (Generic Desktop) */
    0x09, 0x39,        /*     Usage (Hat switch)           */
    0x15, 0x00,        /*     Logical Minimum (0)          */
    0x25, 0x07,        /*     Logical Maximum (7)          */
    0x35, 0x00,        /*     Physical Minimum (0)         */
    0x46, 0x3b, 0x01,  /*     Physical Maximum (315)       */
    0x65, 0x14,        /*     Unit (Degrees)               */
    0x75, 0x04,        /*     Report Size (4)              */
    0x95, 0x01,        /*     Report Count (1)             */
    0x81, 0x42,        /*     Input (Data, Var, Abs, Null) */
    0x65, 0x00,        /*     Unit (None)                  */
    0x75, 0x04,        /*     Report Size (4)              */
    0x95, 0x01,        /*     Report Count (1)             */
    0x81, 0x03,        /*     Input (Constant): padding    */

    /*
     * The direction pad is sent twice over: as the hat above, and as
     * buttons 13-16.  Both drivers that gave Mac OS X a controller in
     * this period settled on buttons -- the Xbox 360 one changed to them
     * on purpose, "for better support in games" -- while a hat is what
     * the pad really is and what Halo reads.  Sending both costs the
     * four bits that were padding, and suits either kind of program.
     */
    0x05, 0x09,        /*     Usage Page (Button)          */
    0x19, 0x01,        /*     Usage Minimum (Button 1)     */
    0x29, 0x10,        /*     Usage Maximum (Button 16)    */
    0x15, 0x00,        /*     Logical Minimum (0)          */
    0x25, 0x01,        /*     Logical Maximum (1)          */
    0x75, 0x01,        /*     Report Size (1)              */
    0x95, 0x10,        /*     Report Count (16)            */
    0x81, 0x02,        /*     Input (Data, Variable, Abs)  */
    0xc0,              /*   End Collection                 */
    0xc0,              /* End Collection                   */
};

static const USBDescIface desc_iface_pad = {
    .bInterfaceNumber              = 0,
    .bNumEndpoints                 = 1,
    .bInterfaceClass               = USB_CLASS_HID,
    .bInterfaceSubClass            = 0x00,      /* no boot protocol */
    .bInterfaceProtocol            = 0x00,
    .ndesc                         = 1,
    .descs = (USBDescOther[]) {
        {
            .data = (uint8_t[]) {
                0x09,               /*  u8  bLength */
                USB_DT_HID,         /*  u8  bDescriptorType */
                0x10, 0x01,         /*  u16 HID_class (1.10) */
                0x00,               /*  u8  country_code */
                0x01,               /*  u8  num_descriptors */
                USB_DT_REPORT,      /*  u8  type: Report */
                sizeof(pad_report_descriptor), 0,   /* u16 len */
            },
        },
    },
    .eps = (USBDescEndpoint[]) {
        {
            .bEndpointAddress      = USB_DIR_IN | 0x01,
            .bmAttributes          = USB_ENDPOINT_XFER_INT,
            .wMaxPacketSize        = PAD_REPORT_LEN,
            .bInterval             = 0x08,      /* 8 ms, as a pad of the era */
        },
    },
};

static const USBDescDevice desc_device_pad = {
    .bcdUSB                        = 0x0110,
    .bMaxPacketSize0               = 8,
    .bNumConfigurations            = 1,
    .confs = (USBDescConfig[]) {
        {
            .bNumInterfaces        = 1,
            .bConfigurationValue   = 1,
            .iConfiguration        = STR_CONFIG,
            .bmAttributes          = USB_CFG_ATT_ONE | USB_CFG_ATT_WAKEUP,
            .bMaxPower             = 50,
            .nif = 1,
            .ifs = &desc_iface_pad,
        },
    },
};

static const USBDesc desc_pad = {
    .id = {
        /*
         * pid.codes hands out identifiers for work like this, so nothing
         * here pretends to be a real product: a guest that recognises a
         * controller by name will not mistake this for one.
         */
        .idVendor          = 0x1209,
        .idProduct         = 0x7050,
        .bcdDevice         = 0x0100,
        .iManufacturer     = STR_MANUFACTURER,
        .iProduct          = STR_PRODUCT,
        .iSerialNumber     = STR_SERIALNUMBER,
    },
    .full = &desc_device_pad,
    .str  = desc_strings,
};

/* ---- the controller on the host ---- */

static gboolean pad_readable(QIOChannel *ioc, GIOCondition cond, gpointer opaque)
{
    PowerEmuPad *p = opaque;
    ssize_t n = qio_channel_read(ioc, (char *)p->in + p->in_len,
                                 sizeof(p->in) - p->in_len, NULL);

    if (n == QIO_CHANNEL_ERR_BLOCK) {
        return G_SOURCE_CONTINUE;
    }
    if (n <= 0) {                       /* PowerEmu went away */
        p->connected = false;
        p->watch = 0;
        return G_SOURCE_REMOVE;
    }
    p->in_len += n;
    if (p->in_len == PAD_REPORT_LEN) {
        memcpy(p->report, p->in, PAD_REPORT_LEN);
        p->in_len = 0;
        p->changed = true;
    }
    return G_SOURCE_CONTINUE;
}

static void pad_try_connect(PowerEmuPad *p)
{
    SocketAddress addr = { .type = SOCKET_ADDRESS_TYPE_UNIX };
    int64_t now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    QIOChannelSocket *sioc;

    if (!p->path || now - p->last_try_ms < 2000) {
        return;
    }
    p->last_try_ms = now;
    addr.u.q_unix.path = p->path;
    sioc = qio_channel_socket_new();
    qio_channel_set_name(QIO_CHANNEL(sioc), "poweremu-gamepad");
    if (qio_channel_socket_connect_sync(sioc, &addr, NULL) < 0) {
        object_unref(OBJECT(sioc));
        return;
    }
    if (p->sioc) {
        object_unref(OBJECT(p->sioc));
    }
    p->sioc = sioc;
    p->in_len = 0;
    p->connected = true;
    p->watch = qio_channel_add_watch(QIO_CHANNEL(p->sioc), G_IO_IN,
                                     pad_readable, p, NULL);
}

/*
 * chardev= (UTM sends the same 9-byte readings over a SPICE port).  The
 * reading in progress is dropped when the other end comes or goes, and the
 * controls are let go when it goes, so nothing stays held.
 */
static void pad_release(PowerEmuPad *p)
{
    memset(p->report, 0, sizeof(p->report));
    p->report[0] = p->report[1] = p->report[2] = p->report[3] = 128;
    p->report[6] = 15;
    p->changed = true;
}

static int pad_chr_can_read(void *opaque)
{
    PowerEmuPad *p = opaque;

    return PAD_REPORT_LEN - p->in_len;
}

static void pad_chr_read(void *opaque, const uint8_t *buf, int size)
{
    PowerEmuPad *p = opaque;

    while (size > 0) {
        size_t n = MIN((size_t)size, PAD_REPORT_LEN - p->in_len);

        memcpy(p->in + p->in_len, buf, n);
        p->in_len += n;
        buf += n;
        size -= n;
        if (p->in_len == PAD_REPORT_LEN) {
            memcpy(p->report, p->in, PAD_REPORT_LEN);
            p->in_len = 0;
            p->changed = true;
        }
    }
}

static void pad_chr_event(void *opaque, QEMUChrEvent event)
{
    PowerEmuPad *p = opaque;

    switch (event) {
    case CHR_EVENT_OPENED:
        p->in_len = 0;
        break;
    case CHR_EVENT_CLOSED:
        p->in_len = 0;
        pad_release(p);
        break;
    default:
        break;
    }
}

/* ---- the gamepad the guest sees ---- */

static void pad_handle_reset(USBDevice *dev)
{
    PowerEmuPad *p = POWEREMU_GAMEPAD(dev);

    /* Sticks centred, nothing held. */
    memset(p->report, 0, sizeof(p->report));
    p->report[0] = p->report[1] = p->report[2] = p->report[3] = 128;
    p->report[6] = 15;                  /* the direction pad at rest */
    memcpy(p->sent, p->report, sizeof(p->sent));
    p->changed = false;
}

static void pad_handle_control(USBDevice *dev, USBPacket *pkt, int request,
                               int value, int index, int length, uint8_t *data)
{
    PowerEmuPad *p = POWEREMU_GAMEPAD(dev);
    int ret = usb_desc_handle_control(dev, pkt, request, value, index, length, data);

    if (ret >= 0) {
        return;
    }
    switch (request) {
    case InterfaceRequest | USB_REQ_GET_DESCRIPTOR:
        if ((value >> 8) == 0x22) {     /* the report descriptor */
            int len = MIN(length, (int)sizeof(pad_report_descriptor));
            memcpy(data, p->descriptor, len);
            pkt->actual_length = len;
        } else {
            goto fail;
        }
        break;
    case HID_GET_REPORT:
        memcpy(data, p->report, MIN(length, PAD_REPORT_LEN));
        pkt->actual_length = MIN(length, PAD_REPORT_LEN);
        memcpy(p->sent, p->report, sizeof(p->sent));
        p->changed = false;
        break;
    case HID_SET_IDLE:
    case HID_SET_PROTOCOL:
    case HID_GET_PROTOCOL:
        break;
    default:
    fail:
        pkt->status = USB_RET_STALL;
        break;
    }
}

static void pad_handle_data(USBDevice *dev, USBPacket *pkt)
{
    PowerEmuPad *p = POWEREMU_GAMEPAD(dev);

    if (pkt->pid != USB_TOKEN_IN || pkt->ep->nr != 1) {
        pkt->status = USB_RET_STALL;
        return;
    }
    if (!p->connected && p->path) {
        pad_try_connect(p);
    }
    /* Nothing has moved: say so, rather than repeat the last reading. */
    if (!p->changed && memcmp(p->sent, p->report, sizeof(p->sent)) == 0) {
        pkt->status = USB_RET_NAK;
        return;
    }
    usb_packet_copy(pkt, p->report, MIN(pkt->iov.size, PAD_REPORT_LEN));
    memcpy(p->sent, p->report, sizeof(p->sent));
    p->changed = false;
}

static void pad_realize(USBDevice *dev, Error **errp)
{
    PowerEmuPad *p = POWEREMU_GAMEPAD(dev);

    if (!p->path && !qemu_chr_fe_backend_connected(&p->chr)) {
        error_setg(errp, "poweremu-gamepad needs path= or chardev=");
        return;
    }
    /*
     * No generated serial: games remember which controls a reader bound
     * against the device's identity, and a serial that changed with the
     * port would quietly lose those bindings between runs.
     */
    usb_desc_init(dev);
    memcpy(p->descriptor, pad_report_descriptor, sizeof(pad_report_descriptor));
    p->descriptor[3] = p->usage;        /* Joystick or Game Pad */
    pad_handle_reset(dev);
    if (qemu_chr_fe_backend_connected(&p->chr)) {
        qemu_chr_fe_set_handlers(&p->chr, pad_chr_can_read, pad_chr_read,
                                 pad_chr_event, NULL, p, NULL, true);
    } else {
        pad_try_connect(p);             /* and again later, if PowerEmu is slow */
    }
}

static void pad_unrealize(USBDevice *dev)
{
    PowerEmuPad *p = POWEREMU_GAMEPAD(dev);

    if (p->watch) {
        g_source_remove(p->watch);
        p->watch = 0;
    }
    if (p->sioc) {
        object_unref(OBJECT(p->sioc));
        p->sioc = NULL;
    }
    qemu_chr_fe_deinit(&p->chr, false);
}

static const Property pad_properties[] = {
    DEFINE_PROP_STRING("path", PowerEmuPad, path),
    DEFINE_PROP_CHR("chardev", PowerEmuPad, chr),
    DEFINE_PROP_UINT8("usage", PowerEmuPad, usage, 0x04),
};

static void pad_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);

    uc->product_desc   = "PowerEmu Gamepad";
    uc->usb_desc       = &desc_pad;
    uc->realize        = pad_realize;
    uc->unrealize      = pad_unrealize;
    uc->handle_reset   = pad_handle_reset;
    uc->handle_control = pad_handle_control;
    uc->handle_data    = pad_handle_data;
    uc->handle_attach  = usb_desc_attach;
    device_class_set_props(dc, pad_properties);
    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);
}

static const TypeInfo pad_info = {
    .name          = TYPE_POWEREMU_GAMEPAD,
    .parent        = TYPE_USB_DEVICE,
    .instance_size = sizeof(PowerEmuPad),
    .class_init    = pad_class_init,
};

static void pad_register_types(void)
{
    type_register_static(&pad_info);
}

type_init(pad_register_types)
