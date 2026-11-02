"""Private converter graph, with auditable cost categories.

`route=True` is the baseline's complete gap1/2 converter. `route=False`
removes all gap movement and parent correction for a generic-chart control;
it is NOT a candidate. No stable module globals are mutated.
"""
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'playbook/kit/decoder'))
from final_decoder import conversion as stable


def clone(fn,scope):
 out=FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)
 out.__kwdefaults__=fn.__kwdefaults__;return out

def marked(fn,category):
 def call(b,*args,**kw):
  with b.mark(category):return fn(b,*args,**kw)
 return call

def plain_q_post(b,q,payloads,H,m,parent,reverse=False,phase=None):
 with q_inverse(b,q,m,reverse,phase) as qi:
  for v in payloads:scale(b,v[-1],q,qi,not reverse)

def routed_q_post(b,q,payloads,H,m,parent,reverse=False,phase=None):
 if parent is None:
  with q_inverse(b,q,m,reverse,phase) as qi:
   for v in payloads:scale(b,v[-1],q,qi,not reverse)
  return
 tops,p,c=parent
 if reverse or m+1 in KEEP_CHILD:F=phase['given_gap'].pop(m+1)
 else:
  F=fresh_zero_flag(b,q);consume_flag(b,F,phase['incoming_gap'].pop(m+1))
 with b.mark('gap_routing'):b.cswap(F,q,p)
 if not reverse:
  with b.mark('gap_routing'):
   for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
 with q_inverse(b,q,m,reverse,phase) as qi:
  if not reverse:
   for v in payloads:scale(b,v[-1],q,qi,True)
   for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
  else:
   for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
   for v in payloads:scale(b,v[-1],q,qi,False)
 if reverse:
  with b.mark('gap_routing'):
   for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
 with b.mark('gap_routing'):b.cswap(F,q,p)
 if reverse and m+1 not in KEEP_CHILD:phase['return_mask'][m+1]=hold_flag(b,F)
 else:phase['returned_gap'][m+1]=F

def plain_look_solve(b,f,g,payloads,H=None,parent=None,reverse=False,phase=None):
 m=len(f)-1
 if m==1:return base_case(b,f,g,payloads,H,parent,reverse,phase)
 if not reverse:carry=pre(b,f,g,payloads,defer=True)
 else:carry=post_look(b,f,g,payloads,H,parent,True,incoming=phase['a'].pop(m),defer=True,phase=phase)
 childparent=([v[-1]for v in payloads],f[-1],f[-2])
 look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,childparent,reverse,phase)
 if not reverse:phase['a'][m]=post_look(b,f,g,payloads,H,parent,incoming=carry,defer=True,phase=phase)
 else:pre(b,f,g,payloads,True,incoming=carry)

def plain_root(b,O,g,payloads,reverse,phase):
 a=g[-1];ai,Z=old.inverse_zero(b,a);b.cx(Z,ai[0])
 if reverse:
  mask=phase.pop('_root_inverse_mask');b.cz(ai,mask);b.free(mask)
 f=list(O)+[ai]
 cached_root_action(b,f,g,payloads,ai,Z,post=reverse,undo=reverse)
 erase_product_zero(b,a,ai,Z)
 parent=([v[-1]for v in payloads],f[-1],f[-2])
 old.look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,parent,reverse,phase)
 Z=old.fresh_zero_flag(b,a)
 cached_root_action(b,f,g,payloads,ai,Z,post=not reverse,undo=reverse)
 erase_product_zero(b,a,ai,Z)
 mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai)
 if reverse:phase_inverse34(b,a,mask,zero_mask=mask[0]);b.free(mask)
 else:phase['_root_inverse_mask']=mask


def make_converter(route=True):
 scope=dict(stable.scope)
 targets=[]
 def head_wrapper(fn,post=False):
  def call(b,f,g,payloads,*args,**kw):
   if post:rev=kw.get('reverse',args[2] if len(args)>2 else False)
   else:rev=kw.get('undo',args[0] if args else False)
   targets.append((set(str(q) for w in f[-2:] for q in w),
                   'head_restore' if post!=bool(rev) else 'head_prepare'))
   try:return fn(b,f,g,payloads,*args,**kw)
   finally:targets.pop()
  return call
 def categorized_ops(b,ops,a,ai,undo=False):
  for op in (reversed(ops) if undo else ops):
   dest=op[1] if op[0]=='s' else op[3]
   if op[0]=='c':dest=[q for w in dest for q in w]
   head=bool(targets and any(str(q) in targets[-1][0] for q in dest))
   category=targets[-1][1] if head else 'raw_arithmetic'
   if head and op[0]=='s':category='head_prepare' if op[2]^undo else 'head_restore'
   with b.mark(category):
    if op[0]=='s':scope['scale'](b,op[1],a,ai,op[2]^undo)
    elif op[0]=='m':scope['ar'].mul_xor0(b,op[1],op[2],op[3])
    elif op[0]=='c':scope['convolution_xor'](b,op[1],op[2],op[3],undo)
    else:raise ValueError(op[0])
 invscope=dict(scope['inverse_apply_safe'].__globals__)
 erasescope=dict(invscope['erase'].__globals__)
 erasescope['phase_inverse36_zero']=marked(stable.phase_inverse34,'inverse_phase')
 invscope['erase']=clone(invscope['erase'],erasescope)
 invscope['inverse_zero']=marked(invscope['inverse_zero'],'inverse_compute')
 scope['inverse_apply_safe']=clone(scope['inverse_apply_safe'],invscope)
 scope['inverse_zero']=marked(scope['inverse_zero'],'inverse_compute')
 scope['erase']=invscope['erase']
 # Rebuild call graph so categories are assigned at the innermost operation.
 for name in ['scale','runops','zero_toggle','inverse_apply','pre','canonical_permute',
              'masked_product','fresh_zero_flag','post_look','base','bez_lookahead']:
  scope[name]=clone(scope[name],scope)
 scope['runops']=categorized_ops
 scope['pre']=head_wrapper(scope['pre'])
 scope['post_look']=head_wrapper(scope['post_look'],True)
 scope['base']=head_wrapper(scope['base'],True)
 scope['base_case']=scope['base']
 for name,category in [('scale','raw_scale'),('canonical_permute','gap_routing'),
                       ('rotate','gap_routing'),('masked_product','parent_correction'),
                       ('zero_toggle','flags'),('clear_flag','flags'),
                       ('hold_flag','flag_measure'),('consume_flag','flag_phase')]:
  fn=scope[name]
  if hasattr(fn,'__code__') and fn.__module__==stable.__name__:fn=clone(fn,scope)
  scope[name]=marked(fn,category)
 scope['q_inverse']=contextmanager(clone(stable.scope['q_inverse'].__wrapped__,scope))
 scope['q_post']=clone(routed_q_post if route else plain_q_post,scope)
 scope['look_solve']=clone(stable.look_solve if route else plain_look_solve,scope)
 short=dict(stable.short,old=SimpleNamespace(**scope),
            clear_flag=scope['clear_flag'],hold_flag=scope['hold_flag'],
            consume_flag=scope['consume_flag'],
            phase_inverse34=marked(stable.phase_inverse34,'inverse_phase'))
 for name,fn in [('cached_root_action',stable.cached_root_action),
                 ('look_root_cached',stable.look_root_cached if route else plain_root),
                 ('_bez_monic',stable._bez_monic)]:
  short[name]=clone(fn,short)
 rootaction=short['cached_root_action']
 def root_heads(b,f,g,payloads,ai,Z,post,undo):
  targets.append((set(str(q) for w in f[-2:] for q in w),
                  'head_restore' if post!=bool(undo) else 'head_prepare'))
  try:return rootaction(b,f,g,payloads,ai,Z,post,undo)
  finally:targets.pop()
 short['cached_root_action']=root_heads
 core=clone(stable.base._short_scope['convert_safe'],short)
 def convert_safe(b,A,B,reverse=False,history_scratch12=None,phase_token=None,*,monic_root=True):
  if phase_token is None:
   phase_token={'a':{},'q':{},'_short_monic':bool(monic_root)}
   for key in ['gap_flags','given_gap','incoming_gap','return_mask','returned_gap','held_gap']:phase_token[key]={}
  result=core(b,A,B,reverse,history_scratch12,phase_token,monic_root=monic_root)
  return result
 return convert_safe

convert_safe=make_converter(True)
