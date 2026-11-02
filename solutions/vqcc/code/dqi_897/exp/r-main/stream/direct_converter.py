"""Aligned gap route, with full-product heads on the small remaining levels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import direct_kernels as dk
import measured_heads as mh
from converter import clone
from types import SimpleNamespace
from final_decoder import conversion as stable
THRESHOLD=7


def pivot(b,g,F):
 with b.mark('head_pivot'):b.cswap(F,g[-1],g[-2])

def aligned(b,f,g,F):
 with b.mark('aligned_route'):
  for x,y in zip(f[:-2],g[:-1]):b.cswap(F,x,y)

def cross(b,g,q1,q0,F):
 with dk.selected(b,g[-2],g[-3],F) as c:ar.mul_xor0(b,q1,c,q0)


def direct_pre(b,f,g,payloads,F,undo=False,incoming=None,defer=False):
 pivot(b,g,F);a=g[-1];q0,q1=f[-2:]
 def action(ai):
  if not undo:
   for v in payloads:scale(b,v[-1],a,ai,True)
   for v in payloads:dk.shifted_product(b,g[:-1],v[-1],v[:-1],F)
   scale(b,q1,a,ai,True);cross(b,g,q1,q0,F);scale(b,q0,a,ai,True)
   dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F)
  else:
   dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F,True)
   scale(b,q0,a,ai,False);cross(b,g,q1,q0,F);scale(b,q1,a,ai,False)
   for v in payloads:dk.shifted_product(b,g[:-1],v[-1],v[:-1],F)
   for v in payloads:scale(b,v[-1],a,ai,False)
 out=inverse_apply(b,a,action,incoming,defer);pivot(b,g,F);return out


def direct_post(b,f,g,payloads,F,H,parent,reverse=False,incoming=None,defer=False,phase=None):
 m=len(f)-1;q0,q1=f[-2:]
 if not reverse:q_post(b,q1,payloads,H,m,parent,phase=phase)
 pivot(b,g,F);a=g[-1]
 def dots():
  for v in payloads:dk.selected_dot(b,g[:-1],v[:-1],v[-1],F)
 def action(ai):
  if not reverse:
   for v in payloads:scale(b,v[-1],a,ai,True)
   dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F)
   scale(b,q0,a,ai,False);cross(b,g,q1,q0,F);scale(b,q1,a,ai,False)
  else:
   scale(b,q1,a,ai,True);cross(b,g,q1,q0,F);scale(b,q0,a,ai,True)
   dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F,True)
   for v in payloads:scale(b,v[-1],a,ai,False)
 if not reverse:dots()
 out=inverse_apply(b,a,action,incoming,defer)
 if reverse:dots()
 pivot(b,g,F)
 if reverse:q_post(b,q1,payloads,H,m,parent,True,phase)
 return out


def look_solve(b,f,g,payloads,H=None,parent=None,reverse=False,phase=None):
 m=len(f)-1
 if m<THRESHOLD:return old_look_solve(b,f,g,payloads,H,parent,reverse,phase)
 keep=m in KEEP_CHILD;full=m in KEEP_FULL
 F=phase['held_gap'].pop(m) if reverse and full else fresh_zero_flag(b,g[-1])
 if reverse and not full:consume_flag(b,F,phase['gap_flags'].pop(m))
 if not reverse:carry=direct_pre(b,f,g,payloads,F,defer=True)
 else:carry=direct_post(b,f,g,payloads,F,H,parent,True,phase['a'].pop(m),True,phase)
 aligned(b,f,g,F)
 if reverse or keep:phase['given_gap'][m]=F
 else:phase['incoming_gap'][m]=hold_flag(b,F)
 childparent=([v[-1]for v in payloads],f[-1],f[-2])
 look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,childparent,reverse,phase)
 if reverse and not keep:
  F=fresh_zero_flag(b,g[-1]);consume_flag(b,F,phase['return_mask'].pop(m))
 else:F=phase['returned_gap'].pop(m)
 aligned(b,f,g,F)
 if not reverse:phase['a'][m]=direct_post(b,f,g,payloads,F,H,parent,incoming=carry,defer=True,phase=phase)
 else:direct_pre(b,f,g,payloads,F,True,incoming=carry)
 if reverse:clear_flag(b,g[-1],F)
 elif full:phase['held_gap'][m]=F
 else:phase['gap_flags'][m]=hold_flag(b,F)


def root_action(b,f,g,payloads,ai,Z,F,post,undo):
 a=g[-1];q0,q1=f[-2:]
 if post:
  ops=[('s',v[-1],False) for v in payloads]+[('d',v) for v in payloads]
  ops +=[('s',v[-1],True) for v in payloads]+[('c',),('s',q0,False),('x',)]
 else:
  ops=[('s',v[-1],True) for v in payloads]+[('u',v) for v in payloads]
  ops +=[('x',),('s',q0,True),('c',)]
 b.cx(Z,a[0])
 for op in (reversed(ops) if undo else ops):
  if op[0]=='s':old.scale(b,op[1],a,ai,op[2]^undo)
  elif op[0]=='u':dk.shifted_product(b,g[:-1],op[1][-1],op[1][:-1],F)
  elif op[0]=='d':dk.selected_dot(b,g[:-1],op[1][:-1],op[1][-1],F)
  elif op[0]=='c':dk.shifted_conv(b,g[:-1],[q0,q1],f[:-2],F,undo)
  elif op[0]=='x':
   with dk.selected(b,g[-2],g[-3],F) as c:ar.mul_xor0(b,q1,c,q0)
 b.cx(Z,a[0])


def look_root_cached(b,O,g,payloads,reverse,phase):
 m=len(O);keep=m in KEEP_CHILD;full=m in KEEP_FULL
 F=phase['held_gap'].pop(m) if reverse and full else old.fresh_zero_flag(b,g[-1])
 if reverse and not full:consume_flag(b,F,phase['gap_flags'].pop(m))
 pivot(b,g,F);a=g[-1];ai,Z=old.inverse_zero(b,a);b.cx(Z,ai[0])
 if reverse:
  mask=phase.pop('_root_inverse_mask');b.cz(ai,mask);b.free(mask)
 f=list(O)+[ai]
 root_action(b,f,g,payloads,ai,Z,F,post=reverse,undo=reverse)
 erase_product_zero(b,a,ai,Z);pivot(b,g,F);aligned(b,f,g,F)
 if reverse or keep:phase['given_gap'][m]=F
 else:phase['incoming_gap'][m]=hold_flag(b,F)
 parent=([v[-1]for v in payloads],f[-1],f[-2])
 old.look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,parent,reverse,phase)
 if reverse and not keep:
  F=old.fresh_zero_flag(b,g[-1]);consume_flag(b,F,phase['return_mask'].pop(m))
 else:F=phase['returned_gap'].pop(m)
 aligned(b,f,g,F);pivot(b,g,F);Z=old.fresh_zero_flag(b,a)
 root_action(b,f,g,payloads,ai,Z,F,post=not reverse,undo=reverse)
 erase_product_zero(b,a,ai,Z)
 mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai)
 if reverse:phase_inverse34(b,a,mask,zero_mask=mask[0]);b.free(mask)
 else:phase['_root_inverse_mask']=mask
 pivot(b,g,F)
 if reverse:clear_flag(b,g[-1],F)
 elif full:phase['held_gap'][m]=F
 else:phase['gap_flags'][m]=hold_flag(b,F)


def make_converter():
 _,scope,short=mh.make_converter(True)
 scope.update(dk=dk,THRESHOLD=THRESHOLD,old_look_solve=scope['look_solve'])
 for name in ['pivot','aligned','cross','direct_pre','direct_post','look_solve']:
  scope[name]=clone(globals()[name],scope)
 short.update(old=SimpleNamespace(**scope),dk=dk,pivot=scope['pivot'],aligned=scope['aligned'])
 for name in ['root_action','look_root_cached']:
  short[name]=clone(globals()[name],short)
 short['_bez_monic']=clone(stable._bez_monic,short)
 core=clone(stable.base._short_scope['convert_safe'],short)
 def convert_safe(b,A,B,reverse=False,history_scratch12=None,phase_token=None,*,monic_root=True):
  if phase_token is None:
   phase_token={'a':{},'q':{},'heads':{},'_short_monic':bool(monic_root)}
   for key in ['gap_flags','given_gap','incoming_gap','return_mask','returned_gap','held_gap']:phase_token[key]={}
  out=core(b,A,B,reverse,history_scratch12,phase_token,monic_root=monic_root)
  if reverse:assert not phase_token['heads']
  return out
 return convert_safe
convert_safe=make_converter()
