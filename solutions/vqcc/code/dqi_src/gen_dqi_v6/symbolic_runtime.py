"""Exact bounded symbolic partial evaluation for reversible kickmix emission.

Every form is an XOR of independent formal atoms, including constant atom 0.
An AND of forms is interned as a fresh atom; no independence assumption is
used to infer equalities. Oversized forms are forgotten into unique atoms,
which loses optimization opportunities but cannot create false equalities.
HMR returns a wire to zero and permits immediate physical reuse. Declared
input registers remain pinned. No input examples enter this compiler.
SPDX-License-Identifier: Apache-2.0
"""
import kickmix as km
import port_runtime as R

ZERO=frozenset()
ONE=frozenset((0,))
Base=R.Builder

class SymbolicBuilder(Base):
    def __init__(self):
        super().__init__()
        self.forms=[];self.pinned=set();self.atom_id=0;self.products={}
        self.elided={};self.folded={};self.max_native=0
        self.limit=48

    def atom(self):
        self.atom_id+=1
        return frozenset((self.atom_id,))

    def xor(self,a,c):
        v=a^c
        return v if len(v)<=self.limit else self.atom()

    def product(self,a,c):
        if not a or not c:return ZERO
        if a==ONE:return c
        if c==ONE or a==c:return a
        if a^c==ONE:return ZERO
        # Canonical complemented inputs share the same nonlinear atom.
        ac=0 in a;cc=0 in c
        aa=a-ONE;bb=c-ONE
        key=(aa,bb) if tuple(sorted(aa))<=tuple(sorted(bb)) else (bb,aa)
        if key not in self.products:self.products[key]=self.atom()
        v=self.products[key]
        if ac:v=self.xor(v,bb)
        if cc:v=self.xor(v,aa)
        if ac and cc:v=self.xor(v,ONE)
        return v

    def wire(self,q):
        if self.physical[q] is None:
            self.physical[q]=self.k.alloc_qubits(1)[0]
            if self.forms[q]==ONE:self.k.x(self.physical[q])
        return self.physical[q]

    def qubit(self):
        if self.free_qubits:
            q=self.free_qubits.pop();self.free_set.remove(q)
            assert self.forms[q]==ZERO
        else:
            q=len(self.physical);self.physical.append(None);self.forms.append(ZERO)
        assert q not in self.active
        self.active.add(q)
        return q

    def _constant(self,q):
        if q in self.pinned or self.forms[q] not in (ZERO,ONE):return
        if self.physical[q] is not None:
            if self.forms[q]==ONE:self.k.x(self.physical[q])
            self.k.free(self.physical[q]);self.physical[q]=None

    def recycle(self,q):
        assert q in self.active
        if self.forms[q]:
            # Retain actual reset semantics on every approximate path.
            self.k.reset(self.wire(q))
        if self.physical[q] is not None:self.k.free(self.physical[q])
        self.physical[q]=None;self.forms[q]=ZERO
        self.active.remove(q);self.free_qubits.append(q);self.free_set.add(q)
        self.owners.pop(q,None)

    def _native(self,name,qs,cond):
        if cond is not None:self.k.push_condition(cond)
        self.gate_counts[name]=self.gate_counts.get(name,0)+1
        getattr(self.k,name)(*[self.wire(q) for q in qs])
        if cond is not None:self.k.pop_condition()

    def _gate(self,name,*qs,cond=None):
        if getattr(self,'suspended',False):
            self._native(name,qs,cond)
            if name not in ('z','cz','ccz'):
                changed=qs[-2:] if name in ('swap','cswap') else qs[-1:]
                self.invalidate(changed)
            return
        fs=[self.forms[q] for q in qs]
        def skip():self.elided[name]=self.elided.get(name,0)+1
        def fold(n,qq):
            self.folded[name+'->'+n]=self.folded.get(name+'->'+n,0)+1
            self._gate(n,*qq,cond=cond)
        if name in ('z','cz','ccz'):
            if any(f==ZERO for f in fs):skip();return
            keep=[q for q,f in zip(qs,fs) if f!=ONE]
            if len(keep)<len(qs):
                if not keep:
                    if cond is None:
                        # Global half-turn retained explicitly using one bit.
                        q=self.k.alloc_qubits(1)[0];self.k.x(q);self.k.z(q);self.k.x(q);self.k.free(q)
                    else:self.k.neg_if(cond)
                else:fold(('z','cz','ccz')[len(keep)-1],keep)
                return
            self._native(name,qs,cond);return
        if name in ('cx','ccx','cswap'):
            n=1 if name in ('cx','cswap') else 2
            if any(f==ZERO for f in fs[:n]):skip();return
            if name=='cx' and fs[0]==ONE:fold('x',qs[1:]);return
            if name=='ccx':
                if fs[0]==ONE:fold('cx',qs[1:]);return
                if fs[1]==ONE or fs[0]==fs[1]:fold('cx',(qs[0],qs[2]));return
                if fs[0]^fs[1]==ONE:skip();return
            if name=='cswap':
                if fs[1]==fs[2]:skip();return
                if fs[0]==ONE:fold('swap',qs[1:]);return
        if name=='swap' and fs[0]==fs[1]:skip();return
        if name=='swap' and cond is None and not any(q in self.pinned for q in qs):
            a,c=qs;self.physical[a],self.physical[c]=self.physical[c],self.physical[a]
            self.forms[a],self.forms[c]=fs[1],fs[0]
            skip();return
        self._native(name,qs,cond)
        if name=='x':updates={qs[0]:self.xor(fs[0],ONE)}
        elif name=='cx':updates={qs[1]:self.xor(fs[1],fs[0])}
        elif name=='ccx':updates={qs[2]:self.xor(fs[2],self.product(fs[0],fs[1]))}
        elif name=='swap':updates={qs[0]:fs[1],qs[1]:fs[0]}
        elif name=='cswap':
            delta=self.product(fs[0],self.xor(fs[1],fs[2]))
            updates={qs[1]:self.xor(fs[1],delta),qs[2]:self.xor(fs[2],delta)}
        else:raise ValueError(name)
        for q,f in updates.items():
            self.forms[q]=f if cond is None else self.atom()
            self._constant(q)

    def hmr(self,q):
        self.k.hmr(self.wire(q),self.measurement)
        self.forms[q]=ZERO
        self._constant(q)
        return self.measurement

    def invalidate(self,qs):
        # Direct native field operations bypass this adapter. Conservatively
        # forget their outputs; no value-based optimization may cross them.
        for q in qs:self.forms[q]=self.atom()

def declare_registers(b,decls):
    regs=R.declare_registers(b,decls)
    assert not b.forms
    b.forms=[b.atom() for _ in b.physical]
    b.pinned.update(q for v in regs.values() for q in v)
    return regs
