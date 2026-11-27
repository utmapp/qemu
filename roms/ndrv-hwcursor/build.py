#!/usr/bin/env python3
"""
Patch QEMU's qemu_vga.ndrv (QemuMacDrivers PEF) to drive a hardware cursor
through ppc-mac-gpu registers at MMIO BAR + 0xFF00.

  0xFF00 SIZE    w | h << 16   (starts an image upload)
  0xFF04 DATA    one ARGB pixel per write, row-major
  0xFF08 COMMIT  define the cursor from the uploaded pixels
  0xFF0C X, 0xFF10 Y           top-left of the image, signed
  0xFF14 SHOW    bit 0 = visible (applies X/Y)
  0xFF18 ID      reads 'HWC1' when the device supports this
  0xFF1C HOT     hotspot x | y << 16 inside the image, written before COMMIT
                 (IOHardwareCursorInfo.cursorHotSpotX/Y, else 4,4; added for UTM)

All registers are little-endian (stwbrx/lwbrx), like the rest of the BAR.
"""
import struct, sys

GLOBAL = 0x1444          # gDriverGlobal, relative to the TOC (data section)
G_ISOPEN = 0x4e
G_REGS = 0x1a            # boardRegAddress (mac68k-aligned struct)
HWC_MAGIC = 0x48574331   # 'HWC1'

# Original stubs (code-section offsets) and the dispatch that reaches them.
STUB_SUPPORTS = 0x1010   # status 22
STUB_SET = 0x1070        # control 22
STUB_DRAW = 0x10a0       # control 23
STUB_STATE = 0x10d0      # status 23

CUR_MAX = 64

# ---------------------------------------------------------------- assembler
def D(op, rt, ra, d): return (op << 26) | (rt << 21) | (ra << 16) | (d & 0xffff)
def X(rt, ra, rb, xo, rc=0): return (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc
def addi(rt, ra, v): return D(14, rt, ra, v)
def li(rt, v): return addi(rt, 0, v)
def addis(rt, ra, v): return D(15, rt, ra, v)
def ori(ra, rs, v): return D(24, rs, ra, v)
def lwz(rt, d, ra): return D(32, rt, ra, d)
def lwzu(rt, d, ra): return D(33, rt, ra, d)
def lbz(rt, d, ra): return D(34, rt, ra, d)
def lhz(rt, d, ra): return D(40, rt, ra, d)
def stw(rs, d, ra): return D(36, rs, ra, d)
def stwu(rs, d, ra): return D(37, rs, ra, d)
def cmplwi(ra, v): return D(10, 0, ra, v)
def cmpw(ra, rb): return X(0, ra, rb, 0)
def stwbrx(rs, ra, rb): return X(rs, ra, rb, 662)
def lwbrx(rt, ra, rb): return X(rt, ra, rb, 534)
def mr(ra, rs): return X(rs, ra, rs, 444)
def or_(ra, rs, rb): return X(rs, ra, rb, 444)
def mullw(rt, ra, rb): return X(rt, ra, rb, 235)
def slwi(ra, rs, n): return (21 << 26) | (rs << 21) | (ra << 16) | (n << 11) | (0 << 6) | ((31 - n) << 1)
def andi_(ra, rs, v): return D(28, rs, ra, v)
MFLR0 = 0x7c0802a6
MTLR0 = 0x7c0803a6
MTCTR0 = 0x7c0903a6
BCTRL = 0x4e800421
BLR = 0x4e800020
SYNC = 0x7c0004ac
EIEIO = 0x7c0006ac

class Asm:
    def __init__(self, base):
        self.base, self.w, self.labels, self.fix = base, [], {}, []
    def pc(self): return self.base + 4 * len(self.w)
    def __call__(self, *ins): self.w.extend(ins)
    def label(self, n): self.labels[n] = self.pc()
    def br(self, kind, n):            # kind: 'b', 'beq', 'bne', 'bdnz'
        self.fix.append((len(self.w), kind, n)); self.w.append(0)
    def link(self):
        for i, kind, n in self.fix:
            off = self.labels[n] - (self.base + 4 * i)
            if kind == 'b':
                self.w[i] = (18 << 26) | (off & 0x3fffffc)
            else:
                bo, bi = {'beq': (12, 2), 'bne': (4, 2), 'bdnz': (16, 0)}[kind]
                assert -0x8000 <= off < 0x8000
                self.w[i] = (16 << 26) | (bo << 21) | (bi << 16) | (off & 0xfffc)
        return b''.join(struct.pack('>I', x) for x in self.w)

# ---------------------------------------------------------------- PEF I/O
def be32(b, o): return struct.unpack_from('>I', b, o)[0]

def build(src, dst):
    d = bytearray(open(src, 'rb').read())
    nsec = struct.unpack_from('>H', d, 32)[0]
    secs = []
    for i in range(nsec):
        o = 40 + 28 * i
        name, addr, total, unpacked, packed, cont, kind, share, align, res = \
            struct.unpack_from('>iIIIIIBBBB', d, o)
        secs.append(dict(name=name, addr=addr, total=total, unpacked=unpacked,
                         packed=packed, off=cont, kind=kind, share=share,
                         align=align, res=res,
                         body=bytearray(d[cont:cont + packed])))
    code = [s for s in secs if s['kind'] == 0][0]
    data = [s for s in secs if s['kind'] == 1][0]
    ldr = [s for s in secs if s['kind'] == 4][0]
    assert code['total'] == code['packed'] and data['total'] == data['packed']

    # ---- loader section: add import VideoServicesLib:VSLPrepareCursorForHardwareCursor
    L = ldr['body']
    hdr = list(struct.unpack_from('>iIiIiIIIIIIIII', L, 0))
    (mainSec, mainOff, initSec, initOff, termSec, termOff, nLib, nSym, nRelSec,
     relOff, strOff, hashOff, hashPow, nExp) = hdr
    libs = [list(struct.unpack_from('>IIIIIBBH', L, 56 + 24 * i)) for i in range(nLib)]
    symoff = 56 + 24 * nLib
    syms = [be32(L, symoff + 4 * i) for i in range(nSym)]
    reloff = symoff + 4 * nSym
    relhdrs = [list(struct.unpack_from('>HHII', L, reloff + 12 * i)) for i in range(nRelSec)]
    strtab = bytes(L[strOff:hashOff])
    hashtab = bytes(L[hashOff:])
    relins = bytes(L[relOff:strOff])      # includes any padding; trimmed below

    def cstr(o): return strtab[o:strtab.index(b'\0', o)].decode()
    vsl = [l for l in libs if cstr(l[0]) == 'VideoServicesLib'][0]
    assert vsl[4] + vsl[3] == nSym, 'VideoServicesLib must own the last imports'
    newidx = nSym
    vsl[3] += 1
    strtab = strtab.rstrip(b'\0') + b'\0'
    nameoff = len(strtab)
    strtab += b'VSLPrepareCursorForHardwareCursor\0'
    while len(strtab) % 4:
        strtab += b'\0'
    syms.append((0x82 << 24) | nameoff)   # weak TVector import, like the others
    nSym += 1

    # ---- widen the mode table: EDID extended-standard entries 34-37
    # (1856x1392 / 1792x1344 CRT modes) become modes at the aspect ratio of
    # a 2880x1864 Retina panel (1440x932 points, 1.545), offered by
    # ppc-mac-gpu's EDID when host-aspect-modes=on.
    EXT_TABLE = 0xd24                   # edidExtStdVModes: {UInt32 w, h}
    # Entries 38-39 (duplicate 1600x1200) and 44-45 (1920x1440) become modes
    # for the area below the notch (1710x1074 points, 1.592).
    MODES = {34: (1440, 932), 35: (1280, 828), 36: (1152, 746), 37: (1680, 1088),
             38: (1440, 904), 39: (1280, 804), 44: (1152, 724), 45: (1680, 1056)}
    for idx, (w, h) in MODES.items():
        o = EXT_TABLE + 8 * idx
        old = struct.unpack_from('>II', data['body'], o)
        assert old in ((1856, 1392), (1792, 1344), (1600, 1200), (1920, 1440)), (idx, old)
        struct.pack_into('>II', data['body'], o, w, h)

    # ---- data section additions
    D0 = len(data['body'])
    assert D0 % 4 == 0
    SLOT = D0                 # import TVector pointer (relocated)
    DESC = SLOT + 4           # IOHardwareCursorDescriptor (100 bytes)
    INFO = DESC + 100         # IOHardwareCursorInfo (44 bytes)
    STATE = INFO + 44         # x, y, visible, set
    BUF = (STATE + 16 + 15) & ~15
    END = BUF + CUR_MAX * CUR_MAX * 4
    ext = bytearray(END - D0)
    struct.pack_into('>HHIIIIII', ext, DESC - D0, 1, 0, CUR_MAX, CUR_MAX, 32, 0, 0, 0)
    struct.pack_into('>HH', ext, INFO - D0, 1, 0)
    data['body'] += ext
    for k in ('total', 'unpacked', 'packed'):
        data[k] = len(data['body'])
    assert END < 0x8000

    # relocation: SetPosition(SLOT), SmByImport(newidx)
    dsec = secs.index(data)
    rh = [h for h in relhdrs if h[0] == dsec][0]
    assert len(relhdrs) == 1, 'only one relocated section expected'
    used = rh[2] * 2
    relins = relins[:used] + struct.pack('>HHH', 0xA000 | (SLOT >> 16), SLOT & 0xffff,
                                         0x6000 | newidx)
    rh[2] += 3
    while len(relins) % 4:
        relins += b'\0\0'

    # ---- new code
    C0 = len(code['body'])
    a = Asm(C0)

    # SupportsHardwareCursor(VDSupportsHardwareCursorRec *r3)
    a.label('supports')
    a(addi(4, 2, GLOBAL), lbz(0, G_ISOPEN, 4), cmplwi(0, 0))
    a.br('bne', 'sup_open')
    a(li(3, -18), BLR)
    a.label('sup_open')
    a(li(0, 0), stw(0, 4, 3), stw(0, 8, 3))
    a(lwz(5, G_REGS, 4), addi(5, 5, 0x7F80), addi(5, 5, 0x7F80))
    a(li(6, 0x18), lwbrx(7, 5, 6))
    a(addis(8, 0, HWC_MAGIC >> 16), ori(8, 8, HWC_MAGIC & 0xffff))
    a(li(0, 0), cmpw(7, 8))
    a.br('bne', 'sup_no')
    a(li(0, 1))
    a.label('sup_no')
    a(stw(0, 0, 3), li(3, 0), BLR)

    # DrawHardwareCursor(VDDrawHardwareCursorRec *r3): x, y, visible
    a.label('draw')
    a(addi(4, 2, GLOBAL), lwz(5, G_REGS, 4), addi(5, 5, 0x7F80), addi(5, 5, 0x7F80))
    a(lwz(6, 0, 3), lwz(7, 4, 3), lwz(8, 8, 3))
    a(addi(9, 2, STATE), stw(6, 0, 9), stw(7, 4, 9), stw(8, 8, 9))
    a(li(10, 0x0C), stwbrx(6, 5, 10))
    a(li(10, 0x10), stwbrx(7, 5, 10))
    a(li(10, 0x14), stwbrx(8, 5, 10))
    a(SYNC, li(3, 0), BLR)

    # GetHardwareCursorDrawState(VDHardwareCursorDrawStateRec *r3)
    a.label('state')
    a(addi(9, 2, STATE))
    for o in (0, 4, 8, 12):
        a(lwz(0, o, 9), stw(0, o, 3))
    a(li(0, 0), stw(0, 16, 3), stw(0, 20, 3), li(3, 0), BLR)

    # SetHardwareCursor(VDSetHardwareCursorRec *r3)
    a.label('set')
    a(MFLR0, stw(0, 8, 1), stwu(1, -64, 1))
    a(lwz(3, 0, 3))                               # csCursorRef
    a(addi(4, 2, DESC), addi(5, 2, INFO))
    a(addi(0, 2, BUF), stw(0, 16, 5))             # hardwareCursorData
    a(li(0, 0), stw(0, 12, 5), stw(0, 4, 5), stw(0, 8, 5), stw(0, 20, 5))  # .. hotspot
    a(lwz(12, SLOT, 2), cmplwi(12, 0))
    a.br('beq', 'set_fail')
    a(stw(2, 20, 1), lwz(0, 0, 12), lwz(2, 4, 12), MTCTR0, BCTRL, lwz(2, 20, 1))
    a(andi_(0, 3, 0xff))
    a.br('beq', 'set_fail')
    a(addi(5, 2, INFO), lwz(6, 4, 5), lwz(7, 8, 5))   # height, width
    a(addi(4, 2, GLOBAL), lwz(8, G_REGS, 4), addi(8, 8, 0x7F80), addi(8, 8, 0x7F80))
    a(slwi(10, 6, 16), or_(10, 10, 7), li(11, 0), stwbrx(10, 8, 11))
    a(mullw(10, 6, 7), cmplwi(10, 0))
    a.br('beq', 'set_commit')
    a(mr(0, 10), MTCTR0, addi(9, 2, BUF - 4), li(11, 4))
    a.label('set_loop')
    a(lwzu(0, 4, 9), stwbrx(0, 8, 11))
    a.br('bdnz', 'set_loop')
    a.label('set_commit')
    # Hotspot: cursorHotSpotX/Y if VSL filled them in. Tiger's IOFramebuffer
    # (IOGraphics-179 convertCursorImage) leaves them 0 and instead shifts the
    # image so the hotspot sits at (4,4) whenever it fits, so 0,0 means (4,4).
    a(addi(5, 2, INFO), lhz(10, 20, 5), lhz(12, 22, 5))
    a(slwi(12, 12, 16), or_(10, 10, 12), cmplwi(10, 0))
    a.br('bne', 'set_hot')
    a(addis(10, 0, 4), ori(10, 10, 4))
    a.label('set_hot')
    a(li(11, 0x1C), stwbrx(10, 8, 11))
    a(li(11, 8), stwbrx(11, 8, 11), SYNC)
    a(addi(9, 2, STATE), li(0, 1), stw(0, 12, 9), li(3, 0))
    a.br('b', 'set_out')
    a.label('set_fail')
    a(addi(9, 2, STATE), li(0, 0), stw(0, 12, 9), li(3, -17))
    a.label('set_out')
    a(lwz(0, 72, 1), addi(1, 1, 64), MTLR0, BLR)

    newcode = a.link()
    code['body'] += newcode
    while len(code['body']) % 16:
        code['body'] += b'\0'
    for k in ('total', 'unpacked', 'packed'):
        code[k] = len(code['body'])

    def patch_branch(at, label):
        off = a.labels[label] - at
        struct.pack_into('>I', code['body'], at, (18 << 26) | (off & 0x3fffffc))
    # sanity: the stubs are where we think (mflr r0 prologues)
    for at in (STUB_SUPPORTS, STUB_SET, STUB_DRAW, STUB_STATE):
        assert be32(code['body'], at) == MFLR0, hex(at)
    patch_branch(STUB_SUPPORTS, 'supports')
    patch_branch(STUB_SET, 'set')
    patch_branch(STUB_DRAW, 'draw')
    patch_branch(STUB_STATE, 'state')

    # ---- reassemble loader
    NL = bytearray(56)
    body = b''.join(struct.pack('>IIIIIBBH', *l) for l in libs)
    body += b''.join(struct.pack('>I', s) for s in syms)
    body += b''.join(struct.pack('>HHII', *h) for h in relhdrs)
    relOff = 56 + len(body)
    body += relins
    strOff = 56 + len(body)
    body += strtab
    hashOff = 56 + len(body)
    assert hashOff % 4 == 0
    body += hashtab
    struct.pack_into('>iIiIiIIIIIIIII', NL, 0, mainSec, mainOff, initSec, initOff,
                     termSec, termOff, nLib, nSym, nRelSec, relOff, strOff, hashOff,
                     hashPow, nExp)
    NL += body
    ldr['body'] = NL
    ldr['packed'] = len(NL)         # loader total/unpacked stay 0

    # ---- reassemble container (keep file order of sections)
    out = bytearray(d[:40 + 28 * nsec])
    pos = len(out)
    for s in sorted(secs, key=lambda s: s['off']):
        pos = (pos + 15) & ~15
        s['off'] = pos
        pos += len(s['body'])
    out += bytes(pos - len(out))
    for i, s in enumerate(secs):
        struct.pack_into('>iIIIIIBBBB', out, 40 + 28 * i, s['name'], s['addr'],
                         s['total'], s['unpacked'], s['packed'], s['off'], s['kind'],
                         s['share'], s['align'], s['res'])
        out[s['off']:s['off'] + len(s['body'])] = s['body']
    open(dst, 'wb').write(out)
    print('code +%d bytes at 0x%x, data 0x%x..0x%x (slot 0x%x desc 0x%x info 0x%x '
          'state 0x%x buf 0x%x), import #%d' % (len(newcode), C0, D0, END, SLOT, DESC,
                                              INFO, STATE, BUF, newidx))
    for n in ('supports', 'set', 'draw', 'state'):
        print('  %s at 0x%x' % (n, a.labels[n]))

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
