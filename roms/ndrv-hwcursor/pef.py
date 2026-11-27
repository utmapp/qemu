import struct, sys
def be32(b,o): return struct.unpack_from('>I',b,o)[0]
class PEF:
    def __init__(s, data):
        s.d=data
        (tag1,tag2,arch,ver,ts,ov,oi,cv,s.nsec,s.ninst,_)=struct.unpack_from('>4s4s4sIIIIIHHI',data,0)
        assert tag1==b'Joy!' and tag2==b'peff'
        s.secs=[]
        strtab_off = 40+28*s.nsec
        for i in range(s.nsec):
            o=40+28*i
            name,addr,total,unpacked,packed,cont,kind,share,align,res=struct.unpack_from('>iIIIIIBBBB',data,o)
            s.secs.append(dict(name=name,addr=addr,total=total,unpacked=unpacked,packed=packed,off=cont,kind=kind,share=share,align=align))
    def loader(s):
        L=[x for x in s.secs if x['kind']==4][0]
        b=s.d[L['off']:L['off']+L['packed']]
        h=struct.unpack_from('>iIiIIIIIIIIIII',b,0)
        keys=['mainSec','mainOff','initSec','initOff','termSec','termOff','nImpLib','nImpSym','nRelocSec','relocInstrOff','strOff','hashOff','hashPow','nExp']
        H=dict(zip(keys,h))
        return b,H
if __name__=='__main__':
    p=PEF(open(sys.argv[1],'rb').read())
    for i,x in enumerate(p.secs): print(i,x)
    b,H=p.loader(); print(H)
    def cstr(o):
        e=b.index(b'\0',H['strOff']+o); return b[H['strOff']+o:e].decode()
    o=56
    libs=[]
    for i in range(H['nImpLib']):
        nameoff,oldimp,curver,nsym,first,opt,flags,res=struct.unpack_from('>IIIIIBBH',b,o); o+=24
        libs.append((cstr(nameoff),nsym,first)); print('lib',cstr(nameoff),nsym,first,hex(opt))
    syms=[]
    for i in range(H['nImpSym']):
        w=be32(b,o); o+=4; cls=w>>24; syms.append(cstr(w&0xffffff)); print(' imp',i,cls,cstr(w&0xffffff))
    for i in range(H['nRelocSec']):
        sec,res,cnt,first=struct.unpack_from('>HHII',b,o); o+=12; print('reloc sec',sec,'count',cnt,'first',first)
    print('exports',H['nExp'])
