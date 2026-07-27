/*
 * QTest testcase for USB xHCI controller
 *
 * Copyright (c) 2014 HUAWEI TECHNOLOGIES CO., LTD.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "libqtest-single.h"
#include "libqos/usb.h"
#include "qobject/qdict.h"

static void wait_device_deleted_event(QTestState *qtest, const char *id)
{
    QDict *resp, *data;
    const char *device;

    /*
     * Other devices might get removed along with the removed device. Skip
     * these. The device of interest will be the last one.
     */
    for (;;) {
        resp = qtest_qmp_eventwait_ref(qtest, "DEVICE_DELETED");
        data = qdict_get_qdict(resp, "data");
        device = data ? qdict_get_try_str(data, "device") : NULL;
        if (device && !strcmp(device, id)) {
            qobject_unref(resp);
            break;
        }
        qobject_unref(resp);
    }
}

/*
 * Regression test for the xHCI-PCI "host" strong-link reference cycle.
 *
 * The xHCI PCI wrapper embeds an xhci-core child whose strong "host" link
 * points back at the PCI device, forming a refcount cycle. If
 * usb_xhci_pci_exit() does not break that cycle, the device's refcount never
 * reaches 0 on unplug, device_finalize() never runs, and therefore the
 * DEVICE_DELETED event (emitted from device_finalize()) is never sent.
 *
 * This test hot-plugs an xHCI controller into an ACPI-hotpluggable bus,
 * requests its removal and waits for DEVICE_DELETED. Without the fix the event
 * is never delivered (device_finalize() is blocked), so the test would
 * hang/time out.
 */
static void test_xhci_unplug_finalize(void)
{
    QTestState *qtest;
    const char *arch = qtest_get_arch();

    if (strcmp(arch, "i386") != 0 && strcmp(arch, "x86_64") != 0) {
        g_test_skip("Test only runs on x86 (ACPI PCI hotplug)");
        return;
    }
    if (!qtest_has_device("nec-usb-xhci")) {
        g_test_skip("Device nec-usb-xhci not available");
        return;
    }

    qtest = qtest_initf("-machine pc");

    qtest_qmp_device_add(qtest, "nec-usb-xhci", "xhci-finalize", "{}");

    /*
     * Request device removal. As the guest is not running, the unplug request
     * won't be processed until the next system reset, which performs the
     * removal and triggers device_finalize() (and thus DEVICE_DELETED).
     */
    qtest_qmp_device_del_send(qtest, "xhci-finalize");
    qtest_system_reset_nowait(qtest);
    wait_device_deleted_event(qtest, "xhci-finalize");

    qtest_quit(qtest);
}

static void test_xhci_hotplug(void)
{
    usb_test_hotplug(global_qtest, "xhci", "1", NULL);
}

static void test_usb_uas_hotplug(void)
{
    QTestState *qts = global_qtest;

    qtest_qmp_device_add(qts, "usb-uas", "uas", "{}");
    qtest_qmp_device_add(qts, "scsi-hd", "scsihd", "{'drive': 'drive0'}");

    /* TODO:
        UAS HBA driver in libqos, to check that
        added disk is visible after BUS rescan
    */

    qtest_qmp_device_del(qts, "scsihd");
    qtest_qmp_device_del(qts, "uas");
}

static void test_usb_ccid_hotplug(void)
{
    QTestState *qts = global_qtest;

    qtest_qmp_device_add(qts, "usb-ccid", "ccid", "{}");
    qtest_qmp_device_del(qts, "ccid");
    /* check the device can be added again */
    qtest_qmp_device_add(qts, "usb-ccid", "ccid", "{}");
    qtest_qmp_device_del(qts, "ccid");
}

int main(int argc, char **argv)
{
    int ret;

    g_test_init(&argc, &argv, NULL);

    qtest_add_func("/xhci/pci/hotplug", test_xhci_hotplug);
    qtest_add_func("/xhci/pci/unplug/finalize", test_xhci_unplug_finalize);
    if (qtest_has_device("usb-uas")) {
        qtest_add_func("/xhci/pci/hotplug/usb-uas", test_usb_uas_hotplug);
    }
    if (qtest_has_device("usb-ccid")) {
        qtest_add_func("/xhci/pci/hotplug/usb-ccid", test_usb_ccid_hotplug);
    }

    qtest_start("-device nec-usb-xhci,id=xhci"
                " -drive id=drive0,if=none,file=null-co://,"
                "file.read-zeroes=on,format=raw");
    ret = g_test_run();
    qtest_end();

    return ret;
}
