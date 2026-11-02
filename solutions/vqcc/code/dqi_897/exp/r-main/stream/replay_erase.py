"""Erase the first nonlinear replay step through its still-live operands.

This changes neither homogeneous forward values nor any completion/failure set.
The scalar replay's second step adds a product to a known constant, so its last
nonlinear reverse operation can use HMR plus a bilinear phase (0 T).
"""
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'playbook/kit/decoder'),str(ROOT/'playbook/kit')]
from final_decoder import decode as ref
from exp.decoder.common import *
from exp.arithmetic.division import phase_product24
from final_decoder.conversion import clear_flag


def scalar_erase_second24(b,head,N,D,q,ct,q0):
    """Reverse second scalar step; its pre-step N is known q0, D arbitrary."""
    scalepair(b,N,inv(q));b.swap(N,D)
    for j in ar.bits(q0):b.x(N[j])
    m=b.alloc_bits(24);b.hmr(N,m)
    a,h=head;cx(b,h,a,ct);phase_product24(b,combine(head),D,m,K);cx(b,h,a,ct)
    b.free(m)
    for j in ar.bits(q0):b.x(N[j])


def external_node_chain(b,words,gi,reverse=False,token=None,normalizer=None):
    if not reverse:return ref.external_node_chain(b,words,gi,False,None,normalizer)
    g=ref.groups[gi];N=combine(words[g['word']:g['word']+2]);L=combine(words[g['word']+2:g['word']+4])
    heads=[words[2*i:2*i+2]for i in range(2*gi)]
    params=[];c=0
    for i in range(2*gi):
        q=ref.ks[i]^g['kap'];assert q
        params.append((q,c));c^=inv(q)
    assert c==0
    if ref.POLE_FREE:
        # Pole-free reverse (see final_decoder/decode.py DQI_POLE_FREE notes) --
        # BUILT AND REFUTED 2026-10-31 (SCORECARD "Pole-free 896"): the (H)/(I)
        # redesign below is semantics-exact (pf1 full-ladder and this chunked
        # version fail blocksim IDENTICALLY: syn_restore_bad 118/512, phase_match
        # 443/512), but the (A) cswap control F=[N=0] over-fires the pole on
        # [N=0 & p=0] shots (E1: +5/36/161/466 per 2048 at gi=13..16), corrupting
        # exactly the over-firing nodes' word registers.  The pole is live data
        # at (A) with no local substitute, so it must survive the middle and the
        # forward peak 48+9 pins Q at 897.  Kept behind the env gate.
        D=b.alloc_qubits(24);b.x(D[0])
        F=ref.zero_flag(b,N);b.cswap(F,N,D)
        m=b.alloc_bits(1);b.hmr(F,m[0]);b.free(F)
        b.x(D[0]);b.cz(D[0],m[0]);b.x(D[0]);b.free(m)
    else:
        pole=token['pole']
        if ref.ZTC:
            pole=ref.zero_flag(b,N);b.cz(pole,token['pole']);b.free(token['pole'])
        D=b.alloc_qubits(24);b.x(D[0]);b.cswap(pole,N,D)
    for head,(q,c)in reversed(list(zip(heads,params))[1:]):ref.external_chain_step(b,head,N,D,L,q,g['c'],c,True)
    q0=params[0][0];scalepair(b,N,inv(q0));b.swap(N,D);scalepair(b,L,inv(q0))
    norm=ref.external_normalize_paired(b,N,D,L,True,token['norm'])
    a,h=heads[0];cx(b,h,a,g['c']);b.cx(combine(heads[0]),N);cx(b,h,a,g['c']);b.cx(h,L[:12])
    invmask=b.alloc_bits(24);b.hmr(D,invmask);b.free(D)
    D=b.alloc_qubits(24);b.x(D[0])
    for i,(head,(q,c))in enumerate(zip(heads,params)):ref.external_scalar_step(b,head,N,D,q,g['c'],first=i==0)
    if ref.POLE_FREE:
        zs=[ref.zero_flag(b,D[lo:lo+6])for lo in(0,6,12,18)]
        w12=b.alloc_qubits(1)[0];b.init_and(zs[0],zs[1],w12)
        w34=b.alloc_qubits(1)[0];b.init_and(zs[2],zs[3],w34)
        F3=b.alloc_qubits(1)[0];b.init_and(w12,w34,F3)
        b.cz(F3,token['pole'])
        b.cswap(F3,N,D)
    else:
        b.cswap(pole,N,D)
    b.cz(D,token['gauge']);b.free(token['gauge'])
    reciprocal,direct=norm
    ref.external_inverse_phase_norm_pair(b,D,invmask,reciprocal,direct);b.free(invmask)
    if ref.POLE_FREE:
        b.cswap(F3,N,D)
        b.del_and(w12,w34,F3);b.free(F3)
        b.del_and(zs[0],zs[1],w12);b.free(w12)
        b.del_and(zs[2],zs[3],w34);b.free(w34)
        for lo,z in zip((0,6,12,18),zs):clear_flag(b,D[lo:lo+6],z)
    else:
        b.cswap(pole,N,D)
        clear_flag(b,D,pole)
    for i in reversed(range(len(heads))):
        if i==1:scalar_erase_second24(b,heads[i],N,D,params[i][0],g['c'],q0)
        else:ref.external_scalar_step(b,heads[i],N,D,params[i][0],g['c'],True,first=i==0)
    b.x(D[0]);b.free(D)

external_extraction=FunctionType(ref.external_extraction.__code__,dict(ref.external_extraction.__globals__,external_node_chain=external_node_chain),'external_extraction',ref.external_extraction.__defaults__,ref.external_extraction.__closure__)
extraction=FunctionType(ref.extraction.__code__,dict(ref.extraction.__globals__,external_extraction=external_extraction),'extraction',ref.extraction.__defaults__,ref.extraction.__closure__)
decode=FunctionType(ref.decode.__code__,dict(ref.decode.__globals__,extraction=extraction),'decode',ref.decode.__defaults__,ref.decode.__closure__)


def scalar_erase_second12(b,head,N,D,q,q0):
    """Subfield version for the trailing zero-node scalar replay."""
    scale(b,N,inv(q));b.swap(N,D)
    for j in ar.bits(q0):b.x(N[j])
    ar.erase_product(b,head[0],D,N)
    for j in ar.bits(q0):b.x(N[j])


def extraction(b,words,reverse=False,token=None):
    if not reverse:return ref.extraction(b,words)
    heads=[words[2*i:2*i+2]for i in range(34)];N,L=words[-2:]
    pole=token['pole'];ref.controlled_inverse(b,L,pole,True,token['chart_inverse'])
    D=b.alloc_qubits(12);b.x(D[0]);b.cswap(pole,N,D)
    for head,kap,c in reversed(list(zip(heads,ref.ks,ref.corrections))[1:]):ref.step(b,head,N,D,L,kap,c,True)
    scale(b,N,inv(ref.ks[0]));b.swap(N,D);scale(b,L,inv(ref.ks[0]))
    extra_z_mask=ref.normal_form(b,N,D,L)
    b.cx(heads[0][0],N);b.cx(heads[0][1],L)
    b.cz(D,token['normalization']);b.free(token['normalization'])
    inverse_mask=b.alloc_bits(12);b.hmr(D,inverse_mask);b.free(D)
    z_mask=token['gauge_mask'];b.cx(extra_z_mask,z_mask);b.free(extra_z_mask)
    D=b.alloc_qubits(12);b.x(D[0]);ref.scalar_step(b,heads[0],N,D,ref.ks[0],first=True)
    for head,kap in zip(heads[1:],ref.ks[1:]):ref.scalar_step(b,head,N,D,kap)
    b.cswap(pole,N,D);b.cz(D,z_mask);b.free(z_mask)
    ref.phase_inverse34(b,D,inverse_mask);b.free(inverse_mask);b.cswap(pole,N,D)
    ref.guarded.zero_toggle(b,D,pole)
    for i in reversed(range(1,len(heads))):
        if i==1:scalar_erase_second12(b,heads[i],N,D,ref.ks[i],ref.ks[0])
        else:ref.scalar_step(b,heads[i],N,D,ref.ks[i],True)
    ref.scalar_step(b,heads[0],N,D,ref.ks[0],True,first=True)
    b.x(D[0]);b.free(D);b.free(pole);external_extraction(b,words,True,token['external'])


decode=FunctionType(ref.decode.__code__,dict(ref.decode.__globals__,extraction=extraction),'decode',ref.decode.__defaults__,ref.decode.__closure__)
