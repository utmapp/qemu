/*
 * HVF stubs for builds without the Hypervisor.framework accelerator.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "system/hvf.h"

bool hvf_kernel_irqchip;
