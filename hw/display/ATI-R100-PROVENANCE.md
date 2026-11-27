# Emulated ATI Radeon R100 (`ati-vga,model=r100`): provenance

The R100 model (Radeon 7000, PCI 1002:5144) in `ati.c`, `ati_2d.c`,
`ati_3d.[ch]`, `ati_cp.c`, `ati_gl.[ch]`, `ati_glr.c`, `ati_int.h`,
`ati_regs.h` and `ati_dbg.c` is taken from:

- RetroBoot 2.0.1, `emulator/patches/retroboot-ati-r100.patch` at RetroBoot
  commit `432a43d4ba8502a7afeaa5dd25ecf959c23aed1c` (SHA-256
  `6276b79e4d9e4ec771c1767b53272096272abc782a94a7abaf94cf8b39b3e393`), as
  imported by amcchord/ClassicMac (`ati/`, commit
  `bcee0a7d400312d3f37fb82d5abf19452416f3bb`).
- ClassicMac's `ati/classicmac-ati-macos.patch` (CGL / OpenGL 4.1 Core host
  back end loaded with `dlopen()`), `ati/classicmac-radeon-idle.patch`,
  `ati/classicmac-texture-swap.patch` and `ati/retroboot-radeon-cursor.patch`
  (RetroBoot `88d074a55b00af7507e9412734ee83fd988876b3`, cursor hunks only).
- Not ported: `ati/classicmac-host-resize.patch` and `host_display.[ch]`
  (they need ClassicMac's own display-control mailbox and guest driver); the
  WGL/Windows back end is not part of these patches.

License: GPL-2.0-or-later, as QEMU (`ati*.c` by BALATON Zoltan) and the
upstream patches.  No ATI or Apple binary is included.

## Additions in this tree

- Hardware TCL position transform (`ati_3d.c`, `ati_glr.c`): written from the
  public Mesa r100 and Linux radeon sources, see `ATI-R100-TCL.md`.

## Rebase notes (QEMU 11.0.2 -> 10.0 UTM)

The patches apply to QEMU 11.0.2's `hw/display/ati*`, which carry upstream
changes this tree did not have yet (2D engine rewrite, HOST_DATA, linear
aperture size, scissor registers, I2CDDC embedded in the device). Those
upstream versions of `ati.c`, `ati_2d.c`, `ati_int.h`, `ati_regs.h` and
`ati_dbg.c` were taken whole as the base, then the patches were applied, then:

- `hw/core/qdev-properties.h` -> `hw/qdev-properties.h`;
  `class_init(..., const void *)` -> `void *`; `(const InterfaceInfo[])` ->
  `(InterfaceInfo[])`.
- `ati_3d.c`: `system/physmem.h` does not exist here (and `exec/ram_addr.h`
  is target-specific); the texture dirty test uses
  `memory_region_snapshot_and_clear_dirty()` + `memory_region_snapshot_get_dirty()`
  on the VRAM region instead of `physical_memory_test_and_clear_dirty()`.
- `meson.build` and `trace-events` hunks applied by hand.

Run-time switches: `CLASSICMAC_ATI_3D=off` (or `RETROBOOT_ATI_3D=off`) turns the
3D engine off; QOM `x-accel-stats` and `x-gl-renderer` report status.

## Firmware

`pc-bios/openbios-ppc-r100`: OpenBIOS `e1e703ac0ad615ab0d7edea54db600aaa629bf07`
+ `roms/openbios-ati-r100/openbios-ati-radeon.diff` (ClassicMac
`ati/openbios-ati-radeon.patch`, RetroBoot `emulator/openbios/retroboot-ati-radeon.patch`,
SHA-256 `c6282bb965fdb742d9a5c25bfdacd2aa7c3a1b460971b99b5d64866996bd0a88`),
built as described in `roms/openbios-ati-r100/README.md`. The retail variant
`pc-bios/openbios-ppc-r100-retail` adds `openbios-ati-radeon-retail.diff` (node
named `ATY,RADEONp`). ClassicMac's own Mac OS 8/9
profile patch is kept next to it (`openbios-classic-macos-8.diff`) for
evaluation; it is not built into `pc-bios`.

Command line (mac99, Mac OS 9):

    qemu-system-ppc -M mac99,via=cuda -cpu 7400 -m 512 -bios openbios-ppc-r100 \
      -vga none -device ati-vga,id=ati0,model=r100,vgamem_mb=32,romfile=,guest_hwcursor=on,xres=1024,yres=768,refresh_rate=60000
