"""GF16 thickened cubic place: truncated lengths2/3 at99/153 Toffolis.

Input banks may be shorter than k, treating missing high coefficients as zero.
Everything is accumulated into a separate k-word target; no extra qubits.
"""
from functools import lru_cache
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'playbook/kit'),str(ROOT/'playbook/kit/decoder'),
             str(ROOT/'playbook/kit/decoder/exp/arithmetic')]
import field as ar
from exp.mul99.kernels import tensor4


@lru_cache(None)
def construction(k):
    assert k in (2,3)
    f,sub=ar.field(12),ar.field(4)
    mul=[[sub.mul(a,b) for b in range(16)] for a in range(16)]
    def prod(a,b):
        out=[0]*(len(a)+len(b)-1)
        for i,u in enumerate(a):
            for j,v in enumerate(b):out[i+j]^=mul[u][v]
        return out
    def mod(a,p):
        out=list(a)
        for i in range(len(out)-len(p),-1,-1):
            c=out[i+len(p)-1]
            if c:
                for j,v in enumerate(p):out[i+j]^=mul[c][v]
        return (out+[0]*(len(p)-1))[:len(p)-1]
    def ev(a,x):
        out=0
        for c in reversed(a):out=mul[out][x]^c
        return out
    def pack(a):return sum(c<<(4*i) for i,c in enumerate(a))
    def unpack(v,n):return [(v>>(4*i))&15 for i in range(n)]
    generator=f.pow(f.primitive_element,4095//15)
    def binary_eval(p,x):
        out=0
        for i in reversed(range(5)):out=f.mul(out,x)^((p>>i)&1)
        return out
    root=next(f.pow(generator,i) for i in range(1,16)
              if binary_eval(sub.modulus,f.pow(generator,i))==0)
    embed=[f.pow(root,i) for i in range(4)]
    alpha=f.primitive_element
    basis=[f.mul(v,f.pow(alpha,j)) for j in range(3) for v in embed]
    to=ar.inverse_columns(basis)
    p=unpack(ar.apply(to,f.pow(alpha,3)),3)+[1]
    modulus=[1]
    for _ in range(k):modulus=prod(modulus,p)
    # The 4096-th power is the unique coefficient-field lift of t.
    theta=[0,1]
    for _ in range(12):theta=mod(prod(theta,theta),modulus)
    tpowers=[[1],theta,mod(prod(theta,theta),modulus)]
    ppowers=[[1]]
    for _ in range(k-1):ppowers.append(prod(ppowers[-1],p))
    cols=[]
    for degree in range(k):
        for bit in range(12):
            co=unpack(to[bit],3)
            v=[0]*(3*k)
            for j,c in enumerate(co):
                for i,x in enumerate(tpowers[j]):v[i]^=mul[c][x]
            cols.append(pack(mod(prod(v,ppowers[degree]),modulus)))
    inv=ar.inverse_columns(cols)
    count=6*k-1
    points=list(range(min(count,16)))
    if count==17:points.append(None)
    terms=[]
    for point in points:
        lag=[1]
        denom=1
        for other in points:
            if other is not None and other!=point:
                lag=prod(lag,[other,1])
                if point is not None:denom=mul[denom][point^other]
        lag=mod([mul[v][sub.invert(denom)] for v in lag],modulus)
        vals=[(unpack(c,3*k)[-1] if point is None else ev(unpack(c,3*k),point))
              for c in cols]
        for smask,svec in tensor4():
            mask=sum(((v&smask).bit_count()&1)<<i for i,v in enumerate(vals))
            vector=ar.apply(inv,pack([mul[v][svec] for v in lag]))
            if mask and vector:terms.append((mask,vector))
    return tuple(terms)


@lru_cache(None)
def tensor(k,nx=None,ny=None):
    nx=k if nx is None else nx
    ny=k if ny is None else ny
    assert 1<=nx<=k and 1<=ny<=k
    xm,ym=(1<<(12*nx))-1,(1<<(12*ny))-1
    merged={}
    for mask,v in construction(k):
        a,c=mask&xm,mask&ym
        if a and c:merged[a,c]=merged.get((a,c),0)^v
    return tuple((a,c,v) for (a,c),v in merged.items() if v)


def accumulate(b,x,y,z,k):
    assert len(x)%12==len(y)%12==0 and len(z)==12*k
    assert len(set(map(str,list(x)+list(y)+list(z))))==len(x)+len(y)+len(z)
    for am,bm,v in tensor(k,len(x)//12,len(y)//12):
        ix,iy,iz=map(lambda q:list(ar.bits(q)),(am,bm,v))
        p,q,t=ix[0],iy[0],iz[0]
        for i in ix[1:]:b.cx(x[i],x[p])
        for j in iy[1:]:b.cx(y[j],y[q])
        for j in iz[1:]:b.cx(z[t],z[j])
        b.ccx(x[p],y[q],z[t])
        for j in reversed(iz[1:]):b.cx(z[t],z[j])
        for j in reversed(iy[1:]):b.cx(y[j],y[q])
        for i in reversed(ix[1:]):b.cx(x[i],x[p])
