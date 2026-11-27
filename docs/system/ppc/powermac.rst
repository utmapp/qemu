PowerMac family boards (``g3beige``, ``mac99``)
==================================================================

Use the executable ``qemu-system-ppc`` to simulate a complete PowerMac
PowerPC system.

- ``g3beige``              Heathrow based PowerMac
- ``mac99``                Mac99 based PowerMac

Supported devices
-----------------

QEMU emulates the following PowerMac peripherals:

 *  UniNorth or Grackle PCI Bridge
 *  PCI VGA compatible card with VESA Bochs Extensions
 *  2 PMAC IDE interfaces with hard disk and CD-ROM support
 *  NE2000 PCI adapters
 *  Non Volatile RAM
 *  VIA-CUDA with ADB keyboard and mouse.
 *  ATI Radeon 9000/9700 (R200/R300) accelerated graphics for Mac OS X
   (``mac99`` only, see below).


Missing devices
---------------

 * To be identified

Firmware
--------

Since version 0.9.1, QEMU uses OpenBIOS https://www.openbios.org/ for
the g3beige and mac99 PowerMac and the 40p machines. OpenBIOS is a free
(GPL v2) portable firmware implementation. The goal is to implement a
100% IEEE 1275-1994 (referred to as Open Firmware) compliant firmware.

ATI Radeon 9000/9700 graphics (Mac OS X Tiger and Leopard)
----------------------------------------------------------

The ``ati-radeon-9700`` and ``ati-radeon-9000`` PCI devices present as an
ATI Radeon 9700 PRO and Radeon 9000 PRO, so that Apple's own ATI drivers
in Mac OS X 10.4 and 10.5 (10.2 for the 9000) drive them, with Quartz
Extreme, Core Image and OpenGL.  The device decodes the card's command
processor stream, 2D engine and R300 3D packets, translates the vertex
and fragment programs to GLSL and executes them on the host GPU with
Metal (the default on macOS hosts) or Vulkan (optional, selected with
``renderer=vulkan``), falling back to a software renderer when neither is
available.  It is built into ``qemu-system-ppc`` when ``mac99`` is, and
needs shaderc and SPIRV-Cross at build time; Vulkan needs the Vulkan
headers and loader (MoltenVK on macOS).

The mac99 PCI host must present an AGP bridge and a PCI memory window
large enough for the card's BARs, and the firmware must name the card
so the kext matches it::

    qemu-system-ppc -M mac99,ndrv=qemu_vga_hwc.ndrv -bios openbios-ppc-r300 \
        -global uni-north-pci.agp-capable=on \
        -device ati-radeon-9700,vgamem_mb=128,renderer=metal

Use ``-bios openbios-ppc-rv250`` with ``ati-radeon-9000``.  ``renderer=``
accepts ``auto``, ``metal`` and ``vulkan``.  ``qemu_vga_hwc.ndrv`` drives the
device's hardware cursor.  The model is derived from linuxkid473/ppc-osx-vgpu
(GPL-2.0-or-later).
