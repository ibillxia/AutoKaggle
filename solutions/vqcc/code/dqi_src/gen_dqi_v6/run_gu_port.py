# SPDX-License-Identifier: Apache-2.0
"""Build the paper-source port with arbitrary classical phase masks.

The published source's fixed mask and loop emitter are replaced by an
in-place Chien scan. The decoder and all constants are generated from
source for the competition field, never translated from binary circuits.
"""
import argparse, hashlib, json, os, pathlib, time, traceback
os.environ['OMP_NUM_THREADS']='1'
import kickmix as km


def borrow_template(b,E,A,bm,dW):
    """Route the final template's two high zero words into a scratch loan.

    The common completed recurrence leaves words d and d+1, modulo 37,
    zero in the cyclic template bank. Routing these into fixed words 35
    and 36 permits a 24-qubit loan. This is an approximate support claim,
    tested by check_borrow.py and the official verifier, not an assertion
    for all inputs. The inverse route restores the remaining template.
    """
    import gu_port as G
    bb=bm.final_bb_l
    def swaps(v):
        m=(v+2)%37
        first=(m+35)%37;second=(m+36)%37
        pairs=[]
        if first!=35:pairs.append((35,first))
        if second==35:second=first
        if second!=36:pairs.append((36,second))
        return pairs
    def route(reverse=False):
        with b.phase('borrow/unroute' if reverse else 'borrow/route'):
            if os.environ.get('DQI_BORROW')=='affine':
                # Two 8-bit values form an affine line: choose a differing
                # pivot, XOR it into the other differing coordinates and
                # test the remaining seven coordinates. Thus each modulo-37
                # selector needs one six-AND ladder, no stored control bank.
                for m in range(37):
                    pairs=swaps(m-2)
                    if not pairs:continue
                    values=[v&255 for v in range(-1,69) if (v+2)%37==m]
                    assert len(values) in (1,2)
                    if len(values)==2:
                        u,v=values;diff=u^v;pivot=(diff&-diff).bit_length()-1
                        changed=[i for i in range(8) if i!=pivot and (diff>>i)&1]
                        for i in changed:E.cx(dW[pivot],dW[i])
                        bits=[dW[i] for i in range(8) if i!=pivot]
                        target=sum((((u>>i)&1)^(((u>>pivot)&1) if i in changed else 0))<<k
                                   for k,i in enumerate(i for i in range(8) if i!=pivot))
                    else:
                        changed=[];bits=dW;target=values[0]
                    with G.eq_test(E,bits,target) as ctl:
                        for x,y in (reversed(pairs) if reverse else pairs):
                            E.swaps(ctl,G.word_pairs([bb[x]],[bb[y]]))
                    for i in reversed(changed):E.cx(dW[pivot],dW[i])
            elif os.environ.get('DQI_BORROW')=='trie':
                mods=[m for m in range(37) if swaps(m-2)]
                ctlwords=A.open('loan_controls',len(mods))
                controls=dict(zip(mods,ctlwords))
                entries={v&255:(v+2)%37 for v in range(-1,69) if (v+2)%37 in controls}
                def visit(values,bit,parent=None):
                    if bit<0:
                        assert len(values)==1
                        E.cx(parent,controls[entries[values[0]]])
                        return
                    one=[v for v in values if (v>>bit)&1]
                    zero=[v for v in values if not (v>>bit)&1]
                    node=A.open('loan_trie',1)[0]
                    parity=0 if one else 1
                    rt=G.norm_term(() if parent is None else (parent,),1 if parent is None else 0,
                                   (dW[bit],),parity,(node,))
                    E.mac([rt])
                    if one:visit(one,bit-1,node)
                    if zero:
                        if one:
                            if parent is None:E.x(node)
                            else:E.cx(parent,node)
                            rt=G.norm_term(() if parent is None else (parent,),1 if parent is None else 0,
                                           (dW[bit],),1,(node,))
                        visit(zero,bit-1,node)
                    E.free([rt]);A.close('loan_trie',[node])
                visit(list(entries),7)
                for m,ctl in controls.items():
                    pairs=swaps(m-2)
                    for x,y in (reversed(pairs) if reverse else pairs):
                        E.swaps(ctl,G.word_pairs([bb[x]],[bb[y]]))
                visit(list(entries),7)
                A.close('loan_controls',ctlwords)
            else:
                for v in range(-1,69):
                    pairs=swaps(v)
                    if not pairs:continue
                    with G.eq_test(E,dW,v&255) as ctl:
                        for x,y in (reversed(pairs) if reverse else pairs):
                            E.swaps(ctl,G.word_pairs([bb[x]],[bb[y]]))
    route()
    loan=list(bb[35])+list(bb[36])
    A.close('borrowed_template',loan)
    def restore():
        A.reopen('borrowed_template',loan)
        route(reverse=True)
    return restore


def dynamic_sweep(b,E,A,omw,ebw,abw):
    import gu_port as G
    from port_runtime import linear_in_place
    raw=b.k;gf=km.GF2Field(12)
    def arr(qs):return km.array([b.physical[q] for q in qs])
    def sum_into(words,target):
        raw.cx(arr([q for word in words for q in word]),
               arr([q for _ in words for q in target]))
    names=[('scan_root',12),('scan_omega',12),('scan_psi',12),('scan_located',12)]
    val,om,ps,loc=[A.open(name,n) for name,n in names]
    all_words=omw+ebw+abw
    logs=list(range(35))+G.LOG_EB+G.LOG_AB
    beta=gf.pow(gf.primitive_element,4094)
    steps=[gf.pow(beta,e) for e in logs]
    arrays=[arr(reg) for reg in all_words]
    for j in range(4095):
        sum_into(omw,om);sum_into(ebw,val);sum_into(abw,ps)
        linear_in_place(b,ps,G.SQUARE_ROWS)
        with G.eq_test(E,val,0) as root:
            raw.init_and(b.physical[root],arr(om),arr(loc))
            raw.gf2_phase_by_product(arr(loc),arr(ps),mask=b.errors[j],field=gf)
            raw.del_and(b.physical[root],arr(om),arr(loc))
        linear_in_place(b,ps,G.SQRT_ROWS)
        sum_into(abw,ps);sum_into(ebw,val);sum_into(omw,om)
        if j+1<4095:
            for reg,factor in zip(arrays,steps):
                if factor!=1:raw.gf2_imul(factor,target=reg,field=gf)
        if j%256==0:
            p={'phase':'sweep','position':j,'positions':4095,'elapsed':time.monotonic()-b.started,
               'physical_qubits':len(b.physical)}
            print(json.dumps(p),flush=True)
            if b.progress:b.progress.write_text(json.dumps(p,indent=2))
    for reg,factor in zip(arrays,steps):
        if factor!=1:raw.gf2_idiv(gf.pow(factor,4094),target=reg,field=gf)
    for (name,_),qs in reversed(list(zip(names,[val,om,ps,loc]))):A.close(name,qs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--shots',type=int,default=2048)
    ap.add_argument('--salts',nargs='+',default=['dqi-gu-source-smoke-20261023'])
    ap.add_argument('--load',action='store_true')
    ap.add_argument('--derive-only',action='store_true')
    ap.add_argument('--borrow',action='store_true')
    ap.add_argument('--loan-routing',choices=['direct','trie','affine'],default='direct')
    ap.add_argument('--build-only',action='store_true')
    ap.add_argument('--expected-sha256')
    ap.add_argument('--lean',action='store_true')
    ap.add_argument('--stack',type=int,default=1)
    ap.add_argument('--wide',action='store_true')
    ap.add_argument('--stream',action='store_true')
    ap.add_argument('--top-first',action='store_true')
    ap.add_argument('--pebble-last',action='store_true')
    ap.add_argument('--norm-pebble',action='store_true')
    ap.add_argument('--wide-late',action='store_true')
    ap.add_argument('--compact-bm-from',type=int,default=70)
    a=ap.parse_args()
    out=pathlib.Path(os.environ['BRIDGE_OUT'])
    host=pathlib.Path('/home/admin_pc4/quantum/runs/ws_dqi')/a.name
    host.mkdir(parents=True,exist_ok=True)
    def report(obj,name='summary.json'):
        (out/name).write_text(json.dumps(obj,indent=2))
        print(json.dumps(obj),flush=True)
    from dqi import verify_submission_dqi as V
    t=time.monotonic()
    if a.load:
        circuit=km.Circuit.from_file(str(host/'dqi.kmb'),format='kmb')
    else:
        import gu_formula_derivation as F
        tables,timing=F.run_b12()
        assert F.certify_kt(tables['KT12'],F.FIELD)
        constants=json.loads((pathlib.Path(__file__).parent/'ref/gu_constants.json').read_text())
        constants.update(tables)
        if a.wide or a.wide_late:
            win=constants['bm_windows']
            for r in range(70):
                if a.wide_late and not a.wide and r<64:continue
                win['IMAXV'][r]=min(r,35,(r+1)//2+1)
                win['KMSV'][r]=min(r+1,35,r//2+2)
        constpath=out/'constants_competition.json'
        constpath.write_text(json.dumps(constants,indent=2))
        report({'field_modulus':F.CONTRACT['gf_poly'],'formulas':{k:len(v) for k,v in tables.items()},
                'derivation_seconds':timing},'formula_summary.json')
        if a.derive_only:return
        os.environ['DQI_CONSTANTS']=str(constpath)
        import gu_port as G
        G.emit_sweep=dynamic_sweep
        G.borrow_template=borrow_template
        if a.stack!=1:
            G.STK_CAP=a.stack
            G.CAPS=[min(c,a.stack) for c in G.CAPS_FULL]
        if a.lean:
            import round2_gadgets as R2
            if a.norm_pebble:report(R2.certify_norm_identity(G),'norm_certificate.json')
            R2.install(G,top_first=a.top_first,pebble_last=a.pebble_last,norm_pebble=a.norm_pebble,
                       compact_from=a.compact_bm_from)
            G.emit_sweep=R2.lean_sweep
            if a.stream:G.emit_sweep=R2.stream_sweep
        if a.borrow:os.environ['DQI_BORROW']=a.loan_routing
        b,tof=G.build()
        (out/'phase_stats.json').write_text(json.dumps(b.phase_stats,indent=2))
        (out/'liveness.json').write_text(json.dumps(b.live_profile,indent=2))
        b.k.free(b.measurement)
        circuit=b.k.finish_circuit()
        circuit.to_file(str(host/'dqi.kmb'),format='kmb')
    summary={'name':a.name,'args':vars(a),'path':str(host/'dqi.kmb'),
             'build_seconds':time.monotonic()-t,'qubits':circuit.num_qubits,
             'static_toffoli':circuit.operation_statistics().num_toffoli_like,
             'num_ops':len(circuit),'bytes':(host/'dqi.kmb').stat().st_size,'verification':{}}
    if a.expected_sha256:
        digest=hashlib.sha256()
        with (host/'dqi.kmb').open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):digest.update(chunk)
        summary['rebuilt_sha256']=digest.hexdigest()
        summary['expected_sha256']=a.expected_sha256
        summary['reproducible']=digest.hexdigest()==a.expected_sha256
        report(summary)
        assert summary['reproducible'], 'Rebuilt circuit differs from verified artifact'
    report(summary)
    if a.build_only:return
    spec=V.ProblemSpecDQI.benchmark_instance(num_shots=a.shots)
    for salt in a.salts:
        report({'stage':'verify','salt':salt,'shots':a.shots},'progress.json')
        t=time.monotonic()
        try:r=V.verify(circuit,spec,salt=salt,max_workers=int(os.environ.get('BRIDGE_JOBS','1')))
        except Exception as e:
            traceback.print_exc();r={'status':'FAIL','error':str(e)}
        r.update(salt=salt,seconds=time.monotonic()-t)
        summary['verification'][salt]=r
        report(summary)
    report({'stage':'complete'},'progress.json')


if __name__=='__main__':main()
