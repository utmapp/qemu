/*
 * Raw mac99 firmware image: loaded at 0xfff00000, entered at +0x100.
 * macio-nvram covers 0xfff04000-0xfff07fff, so the code goes above it.
 */
ENTRY(_start)
SECTIONS
{
    . = 0xfff00000;
    .head : { . += 0x100; KEEP(*(.text.start)) }
    . = 0xfff08000;
    .text : { *(.text .text.*) }
    .rodata : { *(.rodata .rodata.* .sdata2 .sdata2.*) }
    /DISCARD/ : { *(.comment) *(.eh_frame) }
}
