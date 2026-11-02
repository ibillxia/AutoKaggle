"""Monic-root full-product head restoration; separate from frozen candidate."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import direct_converter as direct
import truncated_converter as trunc
from converter import clone


def root_restore(b,f,g,ai,F):
 m=len(g);q0,q1=f[-2:]
 pivot(b,g,F);high=b.alloc_qubits(12)
 # G_original*(q or x*q) has degree m and known monic coefficient1.
 # Interpolation at m finite points aliases x^m by a known constant only.
 with dk.shifted_pair(b,[q0,q1],F) as q:
  dk.conv(b,g,q,f[:-2]+[high],m)
 V,W=dk.k.vandermonde(m);field=ar.field(12)
 for j,word in enumerate(f[:-2]+[high]):
  correction=0
  for t in range(m):correction^=field.mul(W[j][t],field.pow(t,m))
  for i in ar.bits(correction):b.x(word[i])
 pivot(b,g,F)
 mask=b.alloc_bits(12);b.hmr(q0,mask);b.swap(q0,high);b.free(high)
 # q0_old=lo*ai+c*ai^2, since the restored high coefficient is1.
 ar.phase_product(b,q0,ai,mask)
 ar.linear_inplace(b,ai,ar.square_columns(12))
 with dk.selected(b,g[-2],g[-3],F) as c:ar.phase_product(b,c,ai,mask)
 ar.linear_inplace(b,ai,[field.pow(1<<i,2048)for i in range(12)])
 b.free(mask)


def root_action(b,f,g,payloads,ai,Z,F,post,undo):
 restore=post!=undo
 if not restore:return ordinary_root_action(b,f,g,payloads,ai,Z,F,post,undo)
 a=g[-1];b.cx(Z,a[0])
 if post:
  for v in payloads:old.scale(b,v[-1],a,ai,False)
  for v in payloads:dk.selected_dot(b,g[:-1],v[:-1],v[-1],F)
  for v in payloads:old.scale(b,v[-1],a,ai,True)
  root_restore(b,f,g,ai,F)
 else:
  root_restore(b,f,g,ai,F)
  for v in payloads:dk.shifted_product(b,g[:-1],v[-1],v[:-1],F)
  for v in payloads:old.scale(b,v[-1],a,ai,False)
 b.cx(Z,a[0])


def make_converter():
 original_factory=direct.mh.make_converter
 def factory(return_scope=False):
  cv,scope,short=original_factory(True);oldconv=scope['convolution_xor']
  def convolution(b,x,y,z,reverse=False):
   k=len(z)
   if k in (2,3)and len(x)==k and len(y)==2:
    trunc.small.accumulate(b,[q for w in x for q in w],[q for w in y for q in w],[q for w in z for q in w],k)
   else:oldconv(b,x,y,z,reverse)
  scope['convolution_xor']=convolution;short['old']=SimpleNamespace(**scope)
  short['root_restore']=clone(root_restore,short)
  short['ordinary_root_action']=clone(direct.root_action,short)
  return (cv,scope,short)if return_scope else cv
 env=dict(vars(direct),mh=SimpleNamespace(make_converter=factory),root_action=root_action)
 return clone(direct.make_converter,env)()
convert_safe=make_converter()
