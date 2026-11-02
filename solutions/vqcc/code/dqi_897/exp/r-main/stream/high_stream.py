"""Separate experiment: full-head product in original divisor coordinates.

Frozen decode_truncated sources are not changed. Nonroot direct levels only.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import direct_converter as direct
import truncated_converter as trunc
import root_fusion as rf
from converter import clone


def high_full_restore(b,f,g,F):
 m=len(g);lo=b.alloc_qubits(12)
 with dk.shifted_pair(b,f[-2:],F) as q:
  dk.conv(b,g,q,f[:-2]+[lo],m)
 masks=[b.alloc_bits(12),b.alloc_bits(12)]
 b.hmr(f[-2],masks[0]);b.swap(f[-2],lo);b.free(lo)
 hi=b.alloc_qubits(12)
 with dk.selected(b,g[-1],g[-2],F) as a:ar.mul_xor0(b,a,f[-1],hi)
 b.hmr(f[-1],masks[1]);b.swap(f[-1],hi);b.free(hi)
 V,W=dk.k.vandermonde(m);field=ar.field(12)
 for j,word in enumerate(f[:-1]):
  correction=0
  for t in range(m):correction^=field.mul(W[j][t],field.pow(t,m))
  dk.k.cx(b,f[-1],word,correction)
 return masks


def high_close_heads(b,f,g,ai,F,masks):
 lo,hi=f[-2:]
 ar.phase_product(b,hi,ai,masks[1]);ar.phase_product(b,lo,ai,masks[0])
 b.cswap(F,g[-2],g[-3]);c=g[-2]
 if True:
  product=b.alloc_qubits(12);ar.mul_xor0(b,hi,c,product)
  ar.linear_inplace(b,ai,ar.square_columns(12))
  ar.phase_product(b,product,ai,masks[0])
  ar.linear_inplace(b,ai,[ar.field(12).pow(1<<i,2048)for i in range(12)])
  ar.erase_product(b,hi,c,product);b.free(product)
 b.cswap(F,g[-2],g[-3])
 for mask in masks:b.free(mask)


def direct_pre(b,f,g,payloads,F,undo=False,incoming=None,defer=False):
 if not undo:return ordinary_pre(b,f,g,payloads,F,False,incoming,defer)
 Z0=fresh_zero_flag(b,list(g[-2])+list(g[-1]));b.cx(Z0,g[-2][0])
 masks=high_full_restore(b,f,g,F)
 # Z0=>F: if F=0 the original top was nonzero. Thus the clipped selected
 # pivot is at the second-top slot whenever the clipping bit is actually1.
 b.cx(Z0,g[-2][0]);pivot(b,g,F);a=g[-1]
 ai,Z=inverse_zero(b,a);b.cx(Z,Z0);b.free(Z0);consume(b,ai,Z,incoming)
 b.cx(Z,a[0]);b.cx(Z,ai[0]);high_close_heads(b,f,g,ai,F,masks)
 for v in payloads:dk.shifted_product(b,g[:-1],v[-1],v[:-1],F)
 for v in payloads:scale(b,v[-1],a,ai,False)
 b.cx(Z,ai[0]);b.cx(Z,a[0]);out=erase(b,a,ai,Z,defer)
 pivot(b,g,F);return out


def direct_post(b,f,g,payloads,F,H,parent,reverse=False,incoming=None,defer=False,phase=None):
 m=len(f)-1;q0,q1=f[-2:]
 if not reverse:
  q_post(b,q1,payloads,H,m,parent,phase=phase)
  pivot(b,g,F);a=g[-1]
  for v in payloads:dk.selected_dot(b,g[:-1],v[:-1],v[-1],F)
  ai,Z=inverse_zero(b,a);consume(b,ai,Z,incoming)
  b.cx(Z,a[0]);b.cx(Z,ai[0])
  for v in payloads:scale(b,v[-1],a,ai,True)
  b.cx(Z,ai[0]);mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai)
  pivot(b,g,F);phase['heads'][m]=high_full_restore(b,f,g,F)
  b.cx(Z,g[-2][0]);zm=b.alloc_bits(1);b.hmr(Z,zm[0]);b.free(Z)
  assert defer
  return mask,zm
 pivot(b,g,F);a=g[-1]
 def action(ai):
  scale(b,q1,a,ai,True);cross(b,g,q1,q0,F);scale(b,q0,a,ai,True)
  for word,mask in zip([q0,q1],phase['heads'].pop(m)):
   b.cz(word,mask);b.free(mask)
  dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F,True)
  for v in payloads:scale(b,v[-1],a,ai,False)
 carry=inverse_apply(b,a,action,incoming,defer)
 for v in payloads:dk.selected_dot(b,g[:-1],v[:-1],v[-1],F)
 pivot(b,g,F);q_post(b,q1,payloads,H,m,parent,True,phase)
 return carry


def make_converter():
 original_factory=direct.mh.make_converter
 def factory(return_scope=False):
  cv,scope,short=original_factory(True);oldconv=scope['convolution_xor']
  def convolution(b,x,y,z,reverse=False):
   k=len(z)
   if k in (2,3)and len(x)==k and len(y)==2:
    trunc.small.accumulate(b,[q for w in x for q in w],[q for w in y for q in w],[q for w in z for q in w],k)
   else:oldconv(b,x,y,z,reverse)
  scope['convolution_xor']=convolution
  for name in ['high_full_restore','high_close_heads']:
   scope[name]=clone(globals()[name],scope)
  scope['ordinary_pre']=clone(direct.direct_pre,scope)
  # These custom functions intentionally replace only high direct levels;
  # ordinary measured-head low pre/post retain their own high_full_restore globals.
  short['old']=SimpleNamespace(**scope)
  short['root_restore']=clone(rf.root_restore,short)
  short['ordinary_root_action']=clone(direct.root_action,short)
  return (cv,scope,short)if return_scope else cv
 env=dict(vars(direct),mh=SimpleNamespace(make_converter=factory),
  direct_pre=direct_pre,direct_post=direct_post,root_action=rf.root_action)
 return clone(direct.make_converter,env)()
convert_safe=make_converter()
