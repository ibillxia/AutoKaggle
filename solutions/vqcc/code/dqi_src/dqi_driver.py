"""Full source decoder build with exact symbolic lowering (Apache-2.0)."""
import os,sys,pathlib,json,time,gc

def build(destination,work):
    src=pathlib.Path(__file__).resolve().parent
    work=pathlib.Path(work);work.mkdir(parents=True,exist_ok=True)
    os.environ['DQI_BORROW']='affine'
    if str(src/'gen_dqi_v6') not in sys.path:sys.path.insert(0,str(src/'gen_dqi_v6'))
    import gu_formula_derivation as F
    t,_=F.run_b12();assert F.certify_kt(t['KT12'],F.FIELD)
    c=json.loads((src/'gen_dqi_v6/ref/gu_constants.json').read_text());c.update(t)
    path=work/'dqi_constants.json';path.write_text(json.dumps(c));os.environ['DQI_CONSTANTS']=str(path)
    import gu_port as G,run_gu_port as RG,round2_gadgets as R2
    import importlib
    G=importlib.reload(G);R2=importlib.reload(R2)
    import symbolic_runtime as S
    G.Builder=S.SymbolicBuilder;G.declare_registers=S.declare_registers
    G.borrow_template=RG.borrow_template
    R2.install(G,top_first=True,pebble_last=True,norm_pebble=True,compact_from=70)
    source=(src/'gen_dqi_v6/round2_gadgets.py').read_text()
    # Sweep emits native array operations. Mark every write conservatively.
    source=source[source.index('def lean_sweep('):]
    source=source.replace('b.physical[q]','b.wire(q)').replace('b.physical[root]','b.wire(root)').replace('b.physical[loc[0]]','b.wire(loc[0])').replace('b.physical[ps[h]]','b.wire(ps[h])')
    source=source.replace("raw.cx(arr([q for word in words for q in word]),arr([q for _ in words for q in target]))", "raw.cx(arr([q for word in words for q in word]),arr([q for _ in words for q in target]));b.invalidate(target)")
    source=source.replace("tr=A.open('tree',11);", "b.invalidate(val);tr=A.open('tree',11);")
    source=source.replace("linear=[('L',tuple(w[k] for w in ebw),0,(val[k],)) for k in range(12)]", "linear=[('L',tuple(w[k] for w in ebw),0,(val[k],)) for k in range(12)]")
    source=source.replace("raw.del_and(b.wire(root),arr(om),arr(loc))", "raw.del_and(b.wire(root),arr(om),arr(loc));b.forms[loc[0]]=S.ZERO")
    source=source.replace("raw.init_and(b.wire(root),arr(om),arr(loc))", "raw.init_and(b.wire(root),arr(om),arr(loc));b.invalidate(loc)")
    source=source.replace("raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)))", "raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)));b.invalidate(om)")
    source=source.replace("if factor!=1:raw.gf2_imul(factor,target=reg,field=gf)", "if factor!=1:raw.gf2_imul(factor,target=reg,field=gf)")
    ns={'S':S};exec(source,ns)
    def native_sweep(b,E,A,omw,ebw,abw):
        qs=[q for r in omw+ebw+abw for q in r]
        b.suspended=True
        ns['stream_sweep'](b,E,A,omw,ebw,abw)
        b.invalidate(qs)
        b.suspended=False
    G.emit_sweep=native_sweep
    start=time.monotonic();b,_=G.build();b.k.free(b.measurement)
    cir=b.k.finish_circuit();cir.to_file(str(destination),format='kmb')
    report={'num_qubits':cir.num_qubits,'toffoli_count':int(cir.operation_statistics().num_toffoli_like),'ops':len(cir),'build_seconds':time.monotonic()-start,'elided':b.elided,'folded':b.folded,'symbolic_products':len(b.products)}
    return report
