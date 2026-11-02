"""Direct gap-chart arithmetic kernels; no stable module mutations.

After the pivot head swap, F*g[-2]=0. Therefore the shifted low-g times
shifted coefficient products have one identically zero highest coefficient.
This algebraic constraint, not an approximation, permits one fewer evaluation.
"""
from pathlib import Path
import sys
from contextlib import contextmanager
from types import FunctionType,SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'playbook/kit/decoder'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'playbook/kit/decoder/exp/convert'))
from exp.convert import kernels as k
from exp.decoder.common import ar

_scope=dict(vars(k),ar=SimpleNamespace(**dict(vars(ar),mul_xor=ar.mul_xor0)))
rank_update=FunctionType(k.rank_update.__code__,_scope,'rank_update',k.rank_update.__defaults__,k.rank_update.__closure__)

def conv(b,g,q,out,N,reverse=False):
 """Rank-N product with known degree<N; target truncation permitted."""
 V,W=k.vandermonde(N)
 for t in (reversed(range(N)) if reverse else range(N)):
  tc=[W[j][t] for j in range(len(out))]
  if any(tc):rank_update(b,g,V[t][:len(g)],q,V[t][:len(q)],out,tc)

@contextmanager
def shifted_pair(b,q,F):
 """Embed(q0,q1) as q or x*q. Restore heads with one measured AND copy."""
 z=b.alloc_qubits(12)
 b.cswap(F,q[1],z);b.cswap(F,q[0],q[1])
 yield [q[0],q[1],z]
 b.cswap(F,q[0],q[1]);b.cx(z,q[1])
 mask=b.alloc_bits(12);b.hmr(z,mask);b.free(z)
 for j in range(12):b.ccz(F,q[1][j],mask[j])
 b.free(mask)

def shifted_conv(b,g,q,out,F,reverse=False):
 with shifted_pair(b,q,F) as coeff:conv(b,g,coeff,out,len(g)+1,reverse)


def shifted_product(b,g,x,out,F):
 z=b.alloc_qubits(12);b.cswap(F,x,z)
 conv(b,g,[x,z],out,len(g))
 b.cx(z,x)
 mask=b.alloc_bits(12);b.hmr(z,mask);b.free(z)
 for j in range(12):b.ccz(F,x[j],mask[j])
 b.free(mask)

@contextmanager
def selected(b,x,y,F):
 """Select x if F0, y if F1, with12T and exact zero-T phase closure."""
 z=b.alloc_qubits(12);b.cx(y,x)
 for j in range(12):b.init_and(F,x[j],z[j])
 b.cx(y,x);b.cx(x,z)
 yield z
 mask=b.alloc_bits(12);b.hmr(z,mask);b.free(z)
 b.cz(x,mask);b.cx(y,x)
 for j in range(12):b.ccz(F,x[j],mask[j])
 b.cx(y,x);b.free(mask)


def selected_dot(b,g,v,target,F):
 """Transposed shifted scalar product;24 fresh wires,12 extraT.

Compute both transpose outputs, select into target, then erase the unselected
bilinear outputs in the unshifted frame. Both HMR repairs are free CZ/CCZ
with a classical control; no deferred cubic phase debt is introduced.
"""
 d=[b.alloc_qubits(12),b.alloc_qubits(12)]
 N=len(g);V,W=k.vandermonde(N)
 for t in range(N):
  rank_update(b,g,V[t],v,[W[i][t] for i in range(N)],d,[1,t])
 b.cx(d[0],target);b.cx(d[1],d[0])
 for j in range(12):b.ccx(F,d[0][j],target[j])
 b.cx(d[1],d[0])
 masks=[b.alloc_bits(12),b.alloc_bits(12)]
 for x,m in zip(d,masks):b.hmr(x,m);b.free(x)
 # Transpose of output map(1,t) into a classical product mask.
 for t in range(N):
  mask=b.alloc_bits(12);b.bit_store(mask,False);b.cx(masks[0],mask)
  # mask dot(t*product) = transpose(multiply t)*mask.
  f=ar.field(12)
  for j in range(12):
   value=f.mul(t,1<<j)
   for i in ar.bits(value):b.cx(masks[1][i],mask[j])
  x,p=k.expose(b,g,V[t]);y,r=k.expose(b,v,[W[i][t] for i in range(N)])
  ar.phase_product(b,x,y,mask)
  k.unexpose(b,v,[W[i][t] for i in range(N)],r);k.unexpose(b,g,V[t],p)
  b.free(mask)
 for m in masks:b.free(m)
