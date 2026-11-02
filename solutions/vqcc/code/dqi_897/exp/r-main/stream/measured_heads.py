"""NOTE1 full-product quotient-head restoration with exact clipped completion.

Nonroot levels only. Forward a inverse is retired before the24 product
outputs while its zero flag stays live. Reverse restoration gets an11T
zero flag, absorbed into the next inverse's correlated flag. This retiming
pays11T/level to remain within the routed workspace.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter import clone
from direct_kernels import conv
from final_decoder import conversion as stable
from types import SimpleNamespace


def full_restore(b,f,g):
 high=[b.alloc_qubits(12),b.alloc_qubits(12)]
 with b.mark('full_head_product'):conv(b,g,f[-2:],f[:-2]+high,len(g)+1)
 masks=[b.alloc_bits(12),b.alloc_bits(12)]
 for old,new,mask in zip(f[-2:],high,masks):
  b.hmr(old,mask);b.swap(old,new);b.free(new)
 return masks


def close_heads(b,f,g,ai,masks):
 lo,hi=f[-2:];c=g[-2]
 with b.mark('head_phase'):
  ar.phase_product(b,hi,ai,masks[1]);ar.phase_product(b,lo,ai,masks[0])
  product=b.alloc_qubits(12);ar.mul_xor0(b,hi,c,product)
  ar.linear_inplace(b,ai,ar.square_columns(12))
  ar.phase_product(b,product,ai,masks[0])
  ar.linear_inplace(b,ai,[ar.field(12).pow(1<<i,2048) for i in range(12)])
  ar.erase_product(b,hi,c,product);b.free(product)
 for mask in masks:b.free(mask)


def pre(b,f,g,payloads,undo=False,incoming=None,defer=False):
 m=len(f)-1;n=m-1;a=g[-1];q0=f[-2];q1=f[-1]
 if not undo:
  def action(ai):
   ops=[('s',v[-1],True)for v in payloads]
   ops +=[('m',g[i],v[-1],v[i])for v in payloads for i in range(n)]
   ops +=[('s',q1,True),('m',q1,g[-2],q0),('s',q0,True),('c',g[:-1],[q0,q1],f[:-2])]
   runops(b,ops,a,ai)
  return inverse_apply(b,a,action,incoming,defer)
 # A one-bit predicate fits beside24 product targets; an inverse does not.
 F=fresh_zero_flag(b,a);b.cx(F,a[0])
 masks=full_restore(b,f,g);b.cx(F,a[0])
 ai,Z=inverse_zero(b,a);b.cx(Z,F);b.free(F);consume(b,ai,Z,incoming)
 b.cx(Z,a[0]);b.cx(Z,ai[0])
 close_heads(b,f,g,ai,masks)
 for v in payloads:
  for i in range(n):ar.mul_xor0(b,g[i],v[-1],v[i])
 for v in payloads:scale(b,v[-1],a,ai,False)
 b.cx(Z,ai[0]);b.cx(Z,a[0])
 return erase(b,a,ai,Z,defer)


def post_look(b,f,g,payloads,H,parent,reverse=False,incoming=None,defer=False,phase=None):
 m=len(f)-1;n=m-1;a=g[-1];q0=f[-2];q1=f[-1]
 def dots():
  for v in payloads:
   for i in range(n):ar.mul_xor0(b,g[i],v[i],v[-1])
 if not reverse:
  q_post(b,q1,payloads,H,m,parent,phase=phase);dots()
  ai,Z=inverse_zero(b,a);consume(b,ai,Z,incoming)
  b.cx(Z,a[0]);b.cx(Z,ai[0])
  for v in payloads:scale(b,v[-1],a,ai,True)
  b.cx(Z,ai[0]);mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai)
  phase['heads'][m]=full_restore(b,f,g)
  b.cx(Z,a[0]);zm=b.alloc_bits(1);b.hmr(Z,zm[0]);b.free(Z)
  assert defer
  return mask,zm
 def action(ai):
  runops(b,[('s',q1,True),('m',q1,g[-2],q0),('s',q0,True)],a,ai)
  for word,mask in zip([q0,q1],phase['heads'].pop(m)):
   b.cz(word,mask);b.free(mask)
  convolution_xor(b,g[:-1],[q0,q1],f[:-2],True)
  for v in payloads:scale(b,v[-1],a,ai,False)
 carry=inverse_apply(b,a,action,incoming,defer);dots()
 q_post(b,q1,payloads,H,m,parent,True,phase)
 return carry


def make_converter(return_scope=False):
 scope=dict(stable.scope)
 for name in ['scale','runops','zero_toggle','inverse_apply','canonical_permute',
              'masked_product','fresh_zero_flag','base','bez_lookahead']:
  scope[name]=clone(scope[name],scope)
 # consume is exported through inverse helper globals, not converter globals.
 scope['consume']=scope['inverse_apply_safe'].__globals__['consume']
 for name,fn in [('full_restore',full_restore),('close_heads',close_heads),('pre',pre),('post_look',post_look)]:
  scope[name]=clone(fn,dict(scope,conv=conv)) if name=='full_restore' else clone(fn,scope)
 scope['base_case']=scope['base']
 from contextlib import contextmanager
 scope['q_inverse']=contextmanager(clone(stable.scope['q_inverse'].__wrapped__,scope))
 scope['q_post']=clone(stable.q_post,scope)
 scope['look_solve']=clone(stable.look_solve,scope)
 short=dict(stable.short,old=SimpleNamespace(**scope))
 for name,fn in [('cached_root_action',stable.cached_root_action),('look_root_cached',stable.look_root_cached),('_bez_monic',stable._bez_monic)]:
  short[name]=clone(fn,short)
 core=clone(stable.base._short_scope['convert_safe'],short)
 def convert_safe(b,A,B,reverse=False,history_scratch12=None,phase_token=None,*,monic_root=True):
  if phase_token is None:
   phase_token={'a':{},'q':{},'heads':{},'_short_monic':bool(monic_root)}
   for key in ['gap_flags','given_gap','incoming_gap','return_mask','returned_gap','held_gap']:phase_token[key]={}
  out=core(b,A,B,reverse,history_scratch12,phase_token,monic_root=monic_root)
  if reverse:assert not phase_token['heads']
  return out
 return (convert_safe,scope,short) if return_scope else convert_safe
convert_safe=make_converter()
