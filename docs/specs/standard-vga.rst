
QEMU Standard VGA
=================

Exists in two variants, for isa and pci.

command line switches:

``-vga std``
   picks isa for -M isapc, otherwise pci
``-device VGA``
   pci variant
``-device isa-vga``
   isa variant
``-device secondary-vga``
   legacy-free pci variant


PCI spec
--------

Applies to the pci variant only for obvious reasons.

PCI ID
   ``1234:1111``

PCI Region 0
   Framebuffer memory, 16 MB in size (by default).
   Size is tunable via vga_mem_mb property.

PCI Region 1
   Reserved (so we have the option to make the framebuffer bar 64bit).

PCI Region 2
   MMIO bar, 4096 bytes in size (QEMU 1.3+)

PCI ROM Region
   Holds the vgabios (QEMU 0.14+).


The legacy-free variant has no ROM and has ``PCI_CLASS_DISPLAY_OTHER``
instead of ``PCI_CLASS_DISPLAY_VGA``.


IO ports used
-------------

Doesn't apply to the legacy-free pci variant, use the MMIO bar instead.

``03c0 - 03df``
   standard vga ports
``01ce``
   bochs vbe interface index port
``01cf``
   bochs vbe interface data port (x86 only)
``01d0``
   bochs vbe interface data port


Memory regions used
-------------------

``0xe0000000``
  Framebuffer memory, isa variant only.

The pci variant used to mirror the framebuffer bar here, QEMU 0.14+
stops doing that (except when in ``-M pc-$old`` compat mode).


MMIO area spec
--------------

Likewise applies to the pci variant only for obvious reasons.

``0000 - 03ff``
  edid data blob.
``0400 - 041f``
  vga ioports (``0x3c0`` to ``0x3df``), remapped 1:1. Word access
  is supported, bytes are written in little endian order (aka index
  port first),  so indexed registers can be updated with a single
  mmio write (and thus only one vmexit).
``0500 - 0515``
  bochs dispi interface registers, mapped flat without index/data ports.
  Use ``(index << 1)`` as offset for (16bit) register access.
``0600 - 0607``
  QEMU extended registers.  QEMU 2.2+ only.
  The pci revision is 2 (or greater) when these registers are present.
  The registers are 32bit.
``0600``
  QEMU extended register region size, in bytes.
``0604``
  framebuffer endianness register.
  - ``0xbebebebe`` indicates big endian.
  - ``0x1e1e1e1e`` indicates little endian.
``0800 - 0fff``
  Mac OS paravirtual extension ("PVMAC"), ``VGA`` device with
  ``mac-accel=on`` only (off by default). Used by the Mac OS 9 driver
  ``qemu_vga.ndrv``; a driver must check the ID register and treat the
  region as absent when it does not read ``0x50564d43``. The registers
  are 32bit, little endian.
``0800``
  ID, reads ``0x50564d43`` (``'PVMC'``).
``0804``
  version, reads 2 (1: cursor only; 2 adds the GL stream registers below).
``0808``
  capabilities. Bit 0: hardware cursor. Bit 1: GL stream available (the
  PassthroughOpenGL in-process executor library was loaded).
``080c``
  status. Bit 1 is set when the last cursor definition was rejected;
  bits 8-15 hold the reason (1: size, 2: hotspot).
``0810``
  cursor control. Bit 0: cursor visible.
``0814``
  cursor position: x in bits 0-15, y in bits 16-31 (signed, pixels,
  hotspot position).
``0818``
  cursor hotspot: x in bits 0-15, y in bits 16-31.
``081c``
  cursor size: 32 or 64 (square).
``0820``
  cursor define doorbell. Writing any value makes the host read the
  cursor image (size x size pixels, 32bit ARGB stored in A, R, G, B
  byte order) from the cursor slot in VRAM.
``0824``
  cursor slot offset in VRAM (read only). The slot is the last 16 KiB of
  VRAM; the guest must not use it for the framebuffer.
``0828`` - ``084c``
  GL stream (version 2, capabilities bit 1; all read 0 otherwise): ring
  offset, ring size, ring head (rw), ring tail, reply offset, reply size,
  reply length, doorbell (wo), GL status (write 1 to clear; bits: 0 bad
  head, 1 bad reply expect, 2 write failed, 3 short reply, 4 executor open
  failed, 5 no GL, 6 bad reply destination), reply expect (rw, exact reply bytes to read). The
  1 MiB command ring and 4 MiB reply area sit in VRAM directly below the
  cursor slot and must not be used for the framebuffer. A doorbell
  forwards ring bytes tail..head to the executor, then reads reply expect
  bytes into the reply area. A doorbell with reply expect 0 only queues
  the ring bytes for a host worker thread and returns; one with reply
  expect > 0 first waits for the queue to drain. The GL stream needs at
  least 16 MiB of VRAM.
``0850`` - ``085c``
  Present destination (version 2). Reply destination offset (VRAM byte
  offset; 0 is valid), destination pitch (bytes per row), source row
  bytes (width * 4), flags (bit 0: byte-swap every 32bit pixel, BGRA8 to
  big-endian xRGB; bit 1: source rows are bottom-up; bit 2: convert BGRA8
  to big-endian RGB555, 0RRRRRGGGGGBBBBB, bit 0 then ignored, destination
  rows are source row bytes / 2 and need only 2-byte alignment, and the
  bounds checks use that size; bit 3: present, i.e. store the reply at the
  destination instead of the reply area). With bit 3 set and reply expect > 0 the doorbell stores the reply row by row at
  destination + row * pitch and marks it dirty; reply expect must be a
  multiple of the row bytes, the rows 4-byte aligned and the rectangle
  must end below the ring offset, else GL status bit 6 is set and the reply
  is dropped. The 16 byte reply header read must use destination 0.
``0860``
  Stream reset (version 2, write only). Any value stops the host worker,
  closes the executor stream (the host frees that client's contexts),
  resets the ring head and tail, the reply registers and GL status. The
  stream reopens on the next doorbell. A guest GL process writes it before
  its first DG_HELLO, because the host rejects a second HELLO on one
  stream; only one GL application can run at a time.
