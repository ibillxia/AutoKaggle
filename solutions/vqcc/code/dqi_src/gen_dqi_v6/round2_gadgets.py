# SPDX-License-Identifier: Apache-2.0
"""Round-two reversible scheduling changes; original algebra: Gu et al.

The quotient swap schedule, early Euclid quotient parking, and early linear
root erasure were developed in this workspace (OpenAI Codex, 2026-10-23).
"""
from contextlib import contextmanager


def install(G,top_first=False,pebble_last=False,norm_pebble=False,compact_from=70):
    @contextmanager
    def quotient(self, ctl, bb_l):
        E=self.E
        if self.A.round>=compact_from:
            with compact_quotient(G,self,ctl,bb_l) as mq:yield mq
            return
        if pebble_last and self.A.round==68:
            with pebble_quotient(G,self,ctl,bb_l,full=norm_pebble) as mq:yield mq
            return
        with G.opened(E,'pq',G.B) as pv:
            E.mac(G.prod_rt(pv,self.Dw,self.SIG[0]))
            with G.opened(E,'dinv',G.B) as dv, G.chain_live(E,pv,dv):
                # The quotient occupies p's register; swap back before erasing
                # the inversion scaffold, whose source is the original p.
                pairs=G.word_pairs([pv],[dv])
                E.swaps(ctl,pairs)
                yield pv
                E.swaps(ctl,pairs)
            rt=[]
            G.bil_terms(rt,G.KT,G.spec_plain(self.Dw),
                [frozenset((bb_l[0][j],self.SIG[0][j])) for j in range(G.B)],pv)
            rt.append(('L',(ctl,),0,(pv[0],)))
            E.free(rt)
    G.BM.quotient=quotient
    if pebble_last or compact_from<70:
        original_run=G.BM.run
        def run(self):
            original_run(self)
            # An erased scratch window may be reused by a bank later in the
            # SAME round. Replay already owns that bank and restores it to
            # zero before the earlier scratch operations execute. Hold only
            # the other ids; free the bank once at its normal reverse point.
            for rr,regs in self.A.bank_log.items():
                for reg in regs:self.pools[rr].difference_update(reg.wires)
        G.BM.run=run

    # Preserve every journal mark's meaning: q_hi is parked in g17 before
    # q_lo's temporary window opens, and the recorded cofactor operand follows
    # its new location. No polynomial operation is dropped.
    import inspect
    if norm_pebble:
        G.inv_word=lambda E,src:compact_inverse(G,E,src)
    source=inspect.getsource(G.psi_forward)
    source=source.replace('''            with opened(E, "qh", B) as qh, opened(E, "ql", B) as ql:
                E.mac(prod_rt(qh, g[17], fi))''','''            with opened(E, "qh", B) as qh:
                E.mac(prod_rt(qh, g[17], fi))
                E.free(prod_rt(g[17], qh, f[17]))
                E.swaps(None, word_pairs([qh], [g[17]]))
            qh = g[17]
            with opened(E, "ql", B) as ql:''')
    source=source.replace('''                E.free(prod_rt(g[17], qh, f[17]))  # claim g17 == qh f17
''','')
    source=source.replace('''                E.swaps(None, word_pairs([qh, ql], [g[17], g[16]]))''',
                          '''                E.swaps(None, word_pairs([ql], [g[16]]))''')
    if top_first:
        G.top_poly_rt=lambda qh,ql,fw,tgt,high:top_poly_rt(G,qh,ql,fw,tgt,high)
        source=source.replace('''            qh = g[17]
            with opened(E, "ql", B) as ql:
                if j <= 16:
                    with opened(E, "t", B) as t:
                        trt = E.mac(prod_rt(t, qh, f[16]))
                        rtl = []
                        bil_terms(
                            rtl,
                            KT,
                            [frozenset((g[16][i], t[i])) for i in range(B)],
                            spec_plain(fi),
                            ql,
                        )
                        E.mac(rtl)
                        E.free(trt)
                else:
                    E.mac(prod_rt(ql, g[16], fi))''','''            qh = g[17]
            if j <= 16:
                targets = [[g[j - 1 + n]] for n in range(18 - j)]
                fw = [f[k] for k in range(j,17)]
                E.xor(top_poly_rt(qh,None,fw,targets,True))
            with opened(E, "ql", B) as ql:
                E.mac(prod_rt(ql, g[16], fi))''')
        source=source.replace('''                        polyprod_rt(
                            [ql, qh],
                            [f[k] for k in range(j, 17)],
                            [[g[j - 1 + n]] for n in range(18 - j)],
                        ),''','''                        top_poly_rt(qh,ql,fw,targets,False),''')
        assert 'with opened(E, "t", B)' not in source
    ns=dict(vars(G));exec(source,ns)
    G.psi_forward=ns['psi_forward']


@contextmanager
def compact_quotient(G,bm,ctl,bb_l):
    E=bm.E
    with G.opened(E,'pq',12) as pv:
        E.mac(G.prod_rt(pv,bm.Dw,bm.SIG[0]))
        with G.inv_word(E,pv) as dv:
            pairs=G.word_pairs([pv],[dv]);E.swaps(ctl,pairs)
            yield pv
            E.swaps(ctl,pairs)
        rt=[]
        G.bil_terms(rt,G.KT,G.spec_plain(bm.Dw),
            [frozenset((bb_l[0][j],bm.SIG[0][j])) for j in range(12)],pv)
        rt.append(('L',(ctl,),0,(pv[0],)));E.free(rt)


@contextmanager
def pebble_quotient(G,bm,ctl,bb_l,full=False):
    E=bm.E;A=bm.A
    with G.opened(E,'pq',12) as pv:
        E.mac(G.prod_rt(pv,bm.Dw,bm.SIG[0]))
        dv=A.open('dinv',12)
        wins,rts=G.chain_fwd(E,pv,dv)
        nq,m3,s3,yd=wins
        E.free(rts[2]);E.free(rts[1])
        A.close('s3d',s3);A.close('m3d',m3)
        if full:
            E.free(rts[0]);A.close('nqd',nq)
        pairs=G.word_pairs([pv],[dv]);E.swaps(ctl,pairs)
        yield pv
        E.swaps(ctl,pairs)
        # The inverse output needs only p and yd for MBU. Release its twelve
        # wires BEFORE recomputing the two erased three-bit norm windows.
        E.free(rts[4]);A.close('dinv',dv)
        if full:nq=A.open('nqd_re',6)
        nm=A.open('m3d_re',3);ns=A.open('s3d_re',3)
        nr=G.chain_rts(pv,[nq,nm,ns,yd],dv)
        if full:E.mac(nr[0])
        E.mac(nr[1]);E.mac(nr[2]);E.free(nr[3])
        E.free(nr[2]);E.free(nr[1]);E.free(nr[0] if full else rts[0])
        A.close('s3d_re',ns);A.close('m3d_re',nm)
        A.close('yd',yd);A.close('nqd',nq)
        rt=[]
        G.bil_terms(rt,G.KT,G.spec_plain(bm.Dw),
            [frozenset((bb_l[0][j],bm.SIG[0][j])) for j in range(12)],pv)
        rt.append(('L',(ctl,),0,(pv[0],)));E.free(rt)


def norm_rt(G,src,out):
    rt=[]
    G.bil_terms(rt,G.KT6,G.spec_rows(G.A0ROWS,src),G.spec_rows(G.A1ROWS,src),out)
    G.lin_rows_terms(rt,G.LNROWS,src,out)
    return rt


@contextmanager
def compact_inverse(G,E,src):
    """18-wire inverse schedule using Norm(x^-1) = Norm(x)^-1.

    Forward inverse is still 66 Toffolis. Erasing its output costs 39 rather
    than36, but no twelve-bit output overlaps an eighteen-bit scaffold.
    """
    A=E.A
    wins=[A.open(name,n) for name,n in G.CHAIN_WINS]
    nq,m3,s3,yd=wins
    # The dummy output is only used to construct term lists, never emitted.
    rts=G.chain_rts(src,wins,src)
    E.macs(rts[:4])
    E.free(rts[2]);E.free(rts[1]);E.free(rts[0])
    A.close('s3d',s3);A.close('m3d',m3);A.close('nqd',nq)
    out=A.open('inv',12)
    inv=G.chain_rts(src,wins,out)[4];E.mac(inv)
    E.free(norm_rt(G,out,yd));A.close('yd',yd)
    yield out
    y=A.open('yd_un',6);E.mac(norm_rt(G,out,y))
    dummy=[out[:6],out[:3],out[:3],y]
    E.free(G.chain_rts(src,dummy,out)[4]);A.close('inv',out)
    n=A.open('nqd_un',6);m=A.open('m3d_un',3);s=A.open('s3d_un',3)
    rt=G.chain_rts(src,[n,m,s,y],src)
    E.macs(rt[:3]);E.free(rt[3]);E.frees(rt[:3])
    A.close('s3d_un',s);A.close('m3d_un',m);A.close('nqd_un',n);A.close('yd_un',y)


_TOP_V={}

def certify_norm_identity(G):
    """Exhaust all4096 field elements over the actual emitted term lists."""
    src=list(range(12));wins=[list(range(12,18)),list(range(18,21)),list(range(21,24)),list(range(24,30))]
    out=list(range(30,42));normout=list(range(42,48))
    rts=G.chain_rts(src,wins,out);nr=norm_rt(G,out,normout)
    def emit(memory,rt):
        for term in rt:
            if term[0]=='L':v=term[2]^sum(memory[q] for q in term[1])%2
            else:v=(term[2]^sum(memory[q] for q in term[1])%2)&(term[4]^sum(memory[q] for q in term[3])%2)
            for q in term[-1]:memory[q]^=v
    for v in range(4096):
        memory=[(v>>q)&1 for q in range(12)]+[0]*36
        for rt in rts:emit(memory,rt)
        got=sum(memory[q]<<i for i,q in enumerate(out))
        assert got==G.ginv(v),(v,got,G.ginv(v))
        emit(memory,nr)
        assert [memory[q] for q in wins[-1]]==[memory[q] for q in normout],v
    return {'field_elements':4096,'inverse_correct':True,'norm_inverse_identity':True,
            'method':'evaluate exact Gu term lists and norm_rt on every GF4096 element, including zero'}

def top_poly_rt(G,qh,ql,fw,tgt,high):
    """Evaluate a linear-times-polynomial product at finite points + infinity.

    Emit infinity first so its high coefficient changes g16 before q_lo is
    divided from g16. This removes the redundant twelve-bit top-product copy.
    The remaining interpolation columns never write the high coefficient.
    """
    n=len(fw)+1
    if n not in _TOP_V:
        pts=[G.EXP[t] for t in range(n-1)]
        matrix=[[G.gpow(p,k) for k in range(n)] for p in pts]+[[0]*(n-1)+[1]]
        _TOP_V[n]=(pts,G.gf_matinv(matrix))
    pts,iv=_TOP_V[n]
    rt=[]
    for t in ([n-1] if high else range(n-1)):
        if high:
            sa=G.spec_plain(qh);sb=G.spec_plain(fw[-1])
        else:
            sa=[G.xfold([(ql,1<<b),(qh,G.cmrows(pts[t])[b])]) for b in range(G.B)]
            cf=[G.cmrows(G.gpow(pts[t],k)) for k in range(len(fw))]
            sb=[G.xfold([(fw[k],cf[k][b]) for k in range(len(fw))]) for b in range(G.B)]
        def targets(tb,t=t):
            acc={};u=G.tmask(tb)
            for k in range(n):
                v=G.gmul(iv[k][t],u)
                for word in tgt[k]:
                    for b in G.bitlist(v,G.B):acc[word[b]]=acc.get(word[b],0)^1
            return sorted(q for q,odd in acc.items() if odd)
        G.bil_terms(rt,G.KT,sa,sb,tws_of=targets)
    return [term for term in rt if term[-1]]


def lean_sweep(b,E,A,omw,ebw,abw,stream=False):
    import gu_port as G
    import kickmix as km
    from port_runtime import linear_in_place
    raw=b.k;gf=km.GF2Field(12)
    def arr(qs):return km.array([b.physical[q] for q in qs])
    def sum_into(words,target):
        raw.cx(arr([q for word in words for q in word]),arr([q for _ in words for q in target]))
    words=omw+ebw+abw
    logs=list(range(35))+G.LOG_EB+G.LOG_AB
    beta=gf.pow(gf.primitive_element,4094)
    steps=[gf.pow(beta,e) for e in logs]
    arrays=[arr(reg) for reg in words]
    for j in range(4095):
        val=A.open('scan_root',12);sum_into(ebw,val)
        tr=A.open('tree',11);rts=E.ladder(G.eq_rungs(val,0),tr);root=tr[-1]
        linear=[('L',tuple(w[k] for w in ebw),0,(val[k],)) for k in range(12)]
        E.free(linear);A.close('scan_root',val)
        om=A.open('scan_omega',1 if stream else 12);ps=A.open('scan_psi',12);loc=A.open('scan_located',1 if stream else 12)
        sum_into(abw,ps)
        linear_in_place(b,ps,G.SQUARE_ROWS)
        if stream:
            # Bilinearity permits one evaluator bit at a time. The AND
            # ancilla is repaid before reuse: still twelve Toffolis/point.
            for k in range(12):
                raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)))
                raw.init_and(b.physical[root],arr(om),arr(loc))
                for h in range(12):
                    coeff=gf.mul(1<<k,1<<h)
                    for c in range(12):
                        if (coeff>>c)&1:
                            raw.push_condition(b.errors[j][c])
                            raw.cz(b.physical[loc[0]],b.physical[ps[h]])
                            raw.pop_condition()
                raw.del_and(b.physical[root],arr(om),arr(loc))
                raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)))
        else:
            sum_into(omw,om)
            raw.init_and(b.physical[root],arr(om),arr(loc))
            raw.gf2_phase_by_product(arr(loc),arr(ps),mask=b.errors[j],field=gf)
            raw.del_and(b.physical[root],arr(om),arr(loc))
            sum_into(omw,om)
        linear_in_place(b,ps,G.SQRT_ROWS)
        sum_into(abw,ps)
        A.close('scan_located',loc);A.close('scan_psi',ps);A.close('scan_omega',om)
        A.reopen('scan_root',val);sum_into(ebw,val)
        E.frees(rts);A.close('tree',tr)
        sum_into(ebw,val);A.close('scan_root',val)
        if j+1<4095:
            for reg,factor in zip(arrays,steps):
                if factor!=1:raw.gf2_imul(factor,target=reg,field=gf)
        if j%512==0:
            import json,time
            p={'phase':'lean_sweep','position':j,'elapsed':time.monotonic()-b.started}
            print(json.dumps(p),flush=True)
            if b.progress:b.progress.write_text(json.dumps(p))
    for reg,factor in zip(arrays,steps):
        if factor!=1:raw.gf2_idiv(gf.pow(factor,4094),target=reg,field=gf)

def stream_sweep(b,E,A,omw,ebw,abw):
    return lean_sweep(b,E,A,omw,ebw,abw,stream=True)
