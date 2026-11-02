# SPDX-License-Identifier: Apache-2.0
"""Small explicit adapter for Gu et al.'s published source to official kickmix.

Integer wire handles and LIFO ownership match the published builder. Physical
kickmix qubits stay allocated while this adapter accounts for their lifetimes;
every recycle emits the same reset as the authors' Builder. Named replay holds
are checked, so the journal cannot silently alias live scratch.
"""
from contextlib import contextmanager
import json, os, pathlib, time
import kickmix as km
from ref.gu_classical import GF2n, f2_matinv, f2_transpose, parity
from ref.gu_linear import linear_in_place


def load_contract(_):
    contract=json.loads((pathlib.Path(__file__).parent/'ref/gu_contract.json').read_text())
    # The paper uses 0x10eb. The competition uses 0x1053. All arithmetic
    # formulas are regenerated from this field; no field labels are conflated.
    contract['gf_poly']=km.GF2Field(12).modulus
    return contract


class Journal:
    """Gu et al., uncompute.bennett.Journal; pure conjugate choreography."""
    def __init__(self): self._steps=[];self.recording=True
    def mark(self):return len(self._steps)
    def step(self, fwd, rev=None):
        fwd()
        if self.recording:self._steps.append((fwd,rev if rev is not None else fwd))
    def defer(self, fwd, rev=None):self._steps.append((fwd,rev if rev is not None else fwd))
    def forward(self,since=0,until=None):
        for fwd,_ in self._steps[since:until]:fwd()
    def reverse(self,since=0,until=None):
        for _,rev in reversed(self._steps[since:until]):rev()


class Builder:
    def __init__(self):
        self.k=km.CircuitBuilder()
        self.physical=[]
        self.free_qubits=[];self.free_set=set();self.active=set()
        self.errors=[self.k.create_classical_register(12,name=f'E{j}') for j in range(4095)]
        # Published emitter uses each HMR outcome only for the immediately
        # following phase repayments, before the next HMR. Reusing one live
        # classical bit preserves that ordering and avoids dead classical ids.
        self.measurement=self.k.alloc_bits(1)[0]
        self.phase_stats=[];self.gate_counts={}
        self.phase_name='init';self.started=time.monotonic()
        self.owners={};self.live_profile={};self.current_phase=None
        self.progress=pathlib.Path(os.environ['BRIDGE_OUT'])/'build_progress.json' if 'BRIDGE_OUT' in os.environ else None

    def qubit(self):
        if self.free_qubits:
            q=self.free_qubits.pop();self.free_set.remove(q)
        else:
            q=len(self.physical);self.physical.append(self.k.alloc_qubits(1)[0])
        assert q not in self.active
        self.active.add(q)
        self._profile()
        return q
    def qubits(self,n):return [self.qubit() for _ in range(n)]
    def reclaim(self,qs):
        qs=list(qs);want=set(qs)
        assert len(want)==len(qs) and want<=self.free_set, ('reclaim live ids',want-self.free_set)
        self.free_qubits[:]=[q for q in self.free_qubits if q not in want]
        self.free_set-=want;self.active.update(want)
        self._profile()
    def recycle(self,q):
        assert q in self.active,('recycle not active',q)
        self.k.reset(self.physical[q])
        self.active.remove(q);self.free_qubits.append(q);self.free_set.add(q)
        self.owners.pop(q,None)
    def recycle_all(self,qs):
        for q in qs:self.recycle(q)
    def _profile(self,force=False):
        if self.current_phase is None:return
        p=self.live_profile.setdefault(self.current_phase,{'peak_active':0,'peak_physical':0})
        p['peak_physical']=max(p['peak_physical'],len(self.physical))
        if len(self.active)>p['peak_active'] or (force and len(self.active)==p['peak_active']):
            p['peak_active']=len(self.active)
            counts={}
            for q in self.active:
                name=self.owners.get(q,'unclaimed')
                counts[name]=counts.get(name,0)+1
            p['owners_at_peak']=counts
    def claim(self,name,qs):
        for q in qs:self.owners[q]=name
        self._profile(force=True)
    def release(self,name,qs):
        for q in qs:self.owners.pop(q,None)
    def release_recycle(self,name,qs):self.recycle_all(reversed(qs))
    def _gate(self,name,*qs,cond=None):
        if cond is not None:self.k.push_condition(cond)
        self.gate_counts[name]=self.gate_counts.get(name,0)+1
        getattr(self.k,name)(*[self.physical[q] for q in qs])
        if cond is not None:self.k.pop_condition()
    def x(self,q,cond=None):self._gate('x',q,cond=cond)
    def z(self,q,cond=None):self._gate('z',q,cond=cond)
    def cx(self,c,t,cond=None):self._gate('cx',c,t,cond=cond)
    def cz(self,a,b,cond=None):self._gate('cz',a,b,cond=cond)
    def ccx(self,a,b,t,cond=None):self._gate('ccx',a,b,t,cond=cond)
    def ccz(self,a,b,c,cond=None):self._gate('ccz',a,b,c,cond=cond)
    def swap(self,a,b,cond=None):self._gate('swap',a,b,cond=cond)
    def cswap(self,c,a,b,cond=None):self._gate('cswap',c,a,b,cond=cond)
    def neg(self,cond=None):
        assert cond is not None
        self.k.neg_if(cond)
    def hmr(self,q):
        self.k.hmr(self.physical[q],self.measurement)
        return self.measurement
    @contextmanager
    def phase(self,name):
        self.k.info_tag('@enter:'+name)
        self.phase_name=name
        prev=self.current_phase;self.current_phase=name
        self._profile()
        before=dict(self.gate_counts)
        t=time.monotonic()
        p={'phase':name,'elapsed':t-self.started,'physical_qubits':len(self.physical),'active_qubits':len(self.active)}
        print(json.dumps(p),flush=True)
        if self.progress:self.progress.write_text(json.dumps(p,indent=2))
        yield
        self.k.info_tag('@exit:'+name)
        self.phase_stats.append({'phase':name,'seconds':time.monotonic()-t,
                                 'gates':{k:v-before.get(k,0) for k,v in self.gate_counts.items()}})
        self.live_profile[name]['exit_active']=len(self.active)
        self.current_phase=prev


def declare_registers(b,decls):
    regs={}
    for index,(_,width) in enumerate(decls):
        assert not b.free_qubits
        start=len(b.physical)
        reg=b.k.create_quantum_register(width,name=f'S{index}')
        b.physical.extend(reg)
        qs=list(range(start,start+width));b.active.update(qs)
        regs[f'S{index}']=qs
    return regs


def assert_metrics(b,_):
    return b.gate_counts.get('ccx',0)+b.gate_counts.get('cswap',0)+b.gate_counts.get('ccz',0)


def emit_and_report(b):
    raise RuntimeError('Use run_gu_port.py; no implicit publication or emission destination.')
