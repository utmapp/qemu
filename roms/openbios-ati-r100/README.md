# OpenBIOS for `ati-vga,model=r100`

The emulated Radeon 7000 has no FCode ROM, so the stock OpenBIOS cannot set
a boot mode on it or describe it to Mac OS.  These patches add that.  They
are applied on top of OpenBIOS commit
`e1e703ac0ad615ab0d7edea54db600aaa629bf07` ("macio: add awacs/screamer
device", the tree `pc-bios/openbios-ppc` is built from).

| File | Purpose |
|---|---|
| `openbios-ati-radeon.diff` | Programs a boot mode through the ATI CRTC registers and publishes the Radeon properties (`ATY,Rage6p` node).  Built as `pc-bios/openbios-ppc-r100`.  From ClassicMac `ati/openbios-ati-radeon.patch` / RetroBoot `emulator/openbios/retroboot-ati-radeon.patch` (SHA-256 `c6282bb965fdb742d9a5c25bfdacd2aa7c3a1b460971b99b5d64866996bd0a88`). |
| `openbios-ati-radeon-retail.diff` | On top of the above: names the node `ATY,RADEONp` and adds the retail Radeon Mac Edition properties.  Built as `pc-bios/openbios-ppc-r100-retail`. |
| `openbios-classic-macos-8.diff` | ClassicMac's Mac OS 8/9 profile patch.  Evaluation only; not built into `pc-bios`. |

## Building

The images in `pc-bios/` were built with the OpenBIOS builder container
(`ghcr.io/openbios/openbios-builder`, digest
`sha256:bf806644ccff091323dd56a853866ba09301b1a7fcb91e865fdb289499ef8b38`),
as ClassicMac's `scripts/build-openbios.sh` does:

    git clone https://github.com/openbios/openbios.git
    cd openbios
    git checkout e1e703ac0ad615ab0d7edea54db600aaa629bf07
    git apply /path/to/qemu/roms/openbios-ati-r100/openbios-ati-radeon.diff
    # for the retail variant also:
    #   git apply .../openbios-ati-radeon-retail.diff
    docker run --rm --platform linux/amd64 \
      -e "EXTRACFLAGS=-Wno-array-bounds -Wno-address-of-packed-member" \
      -v "$PWD":/src -w /src \
      ghcr.io/openbios/openbios-builder@sha256:bf806644ccff091323dd56a853866ba09301b1a7fcb91e865fdb289499ef8b38 \
      sh -c 'config/scripts/switch-arch ppc >/dev/null && make -j2'
    cp obj-ppc/openbios-qemu.elf pc-bios/openbios-ppc-r100

The banner date was pinned with a `date` shim so the build is reproducible.
Sanity checks used: `strings openbios-ppc-r100 | grep ATY,Rage6p` and
`grep screamer` match; the retail image contains `ATY,RADEONp` and no
`ATY,Rage6p`.

## Use

    qemu-system-ppc -M mac99,via=cuda -cpu 7400 -m 512 \
      -bios openbios-ppc-r100 -vga none \
      -device ati-vga,id=ati0,model=r100,vgamem_mb=32,romfile=,guest_hwcursor=on

(`*.patch` is ignored by the repository's `.gitignore`, hence `.diff`.)
