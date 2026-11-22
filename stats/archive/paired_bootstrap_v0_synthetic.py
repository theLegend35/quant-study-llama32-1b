"""Paired bootstrap over questions for Phase 2 (run once eval_samples/*.json.gz exist).
usage: python3 paired_bootstrap.py <eval_samples_dir> [out.json]
Macro average = mean of the 6 per-task accuracies (as in the paper). Each resample draws questions
with replacement independently within each task (stratified), the same indices for both models.
LoRA cells use per-question correctness averaged over seeds 0-2. Two-sided p = 2*min(P(d*<=0),P(d*>=0))
(percentile bootstrap, +1 smoothing). Holm over the 7 main comparisons. Also seed-level diffs and
per-task exact McNemar (no-LoRA vs no-LoRA, or seed-0 cells)."""
import sys, gzip, json, math, numpy as np
D=sys.argv[1]; OUT=sys.argv[2] if len(sys.argv)>2 else "paired_bootstrap.json"
T=["arc_challenge","arc_easy","hellaswag","winogrande","boolq","piqa"]; B=10000; rng=np.random.default_rng(20261119)
def load(n):
    d=json.load(gzip.open(f"{D}/{n}.json.gz","rt"))["per_task"]
    return {t:np.array(d[t]["correct"],float) for t in T}, {t:d[t]["doc_id"] for t in T}
cell,ids={},None
for b in ["fp16","int8-rtn","int4-rtn"]:
    cell[b],i0=load(b+"-nolora"); ids=ids or i0
    seeds=[load(f"{b}-seed{s}") for s in range(3)]
    for s,(c,i) in enumerate(seeds): assert all(i[t]==ids[t] for t in T); cell[f"{b}+s{s}"]=c
    cell[b+"+L"]={t:np.mean([c[t] for c,_ in seeds],0) for t in T}
idx={t:rng.integers(0,len(ids[t]),(B,len(ids[t]))) for t in T}
def boot(a,b):
    d={t:cell[a][t]-cell[b][t] for t in T}
    obs=np.mean([d[t].mean() for t in T])*100
    bs=np.mean([d[t][idx[t]].mean(1) for t in T],0)*100
    p=min(1,2*min(((bs<=0).sum()+1)/(B+1),((bs>=0).sum()+1)/(B+1)))
    return obs,*np.percentile(bs,[2.5,97.5]),p
def boot_task(a,b,t):
    d=cell[a][t]-cell[b][t]; bs=d[idx[t]].mean(1)*100
    return d.mean()*100,*np.percentile(bs,[2.5,97.5]),min(1,2*min(((bs<=0).sum()+1)/(B+1),((bs>=0).sum()+1)/(B+1)))
def mcnemar(a,b,t):
    x,y=cell[a][t],cell[b][t]; n01=int(((x==1)&(y==0)).sum()); n10=int(((x==0)&(y==1)).sum()); n=n01+n10
    from math import comb
    k=min(n01,n10); p=min(1,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0
    return n01,n10,p
fam=[("RTN INT8 vs FP16 (no LoRA)","int8-rtn","fp16"),("RTN INT4 vs FP16 (no LoRA)","int4-rtn","fp16"),
     ("LoRA effect, FP16","fp16+L","fp16"),("LoRA effect, RTN INT8","int8-rtn+L","int8-rtn"),
     ("LoRA effect, RTN INT4","int4-rtn+L","int4-rtn"),("INT8+LoRA vs FP16+LoRA","int8-rtn+L","fp16+L"),
     ("INT4+LoRA vs FP16+LoRA","int4-rtn+L","fp16+L"),("INT4+LoRA vs plain FP16 (secondary)","int4-rtn+L","fp16")]
res=[boot(a,b) for _,a,b in fam]; main=list(range(7))
order=sorted(main,key=lambda i:res[i][3]); holm={}; run=0
for k,i in enumerate(order): run=max(run,min(1,(7-k)*res[i][3])); holm[i]=run
out={"main":[],"per_task":{},"mcnemar":{}}
for i,(n,a,b) in enumerate(fam):
    o,lo,hi,p=res[i]
    seed=[np.mean([(cell[a.replace("+L",f"+s{s}")][t]-cell[b.replace("+L",f"+s{s}")][t]).mean() for t in T])*100 for s in range(3)] if "+L" in a else None
    out["main"].append(dict(name=n,diff=o,lo=lo,hi=hi,p=p,p_holm=holm.get(i),seed_diffs=seed))
    print(f"{n:38s} {o:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.4f} holm={holm.get(i)} seeds={seed}")
for n,a,b in fam:
    out["per_task"][n]={t:dict(zip(["diff","lo","hi","p"],boot_task(a,b,t))) for t in T}
for n,a,b in [("INT8 vs FP16","int8-rtn","fp16"),("INT4 vs FP16","int4-rtn","fp16"),("INT4 LoRA s0 vs INT4","int4-rtn+s0","int4-rtn"),("INT4+LoRA s0 vs FP16+LoRA s0","int4-rtn+s0","fp16+s0")]:
    out["mcnemar"][n]={t:dict(zip(["a_only","b_only","p"],mcnemar(a,b,t))) for t in T}
json.dump(out,open(OUT,"w"),indent=1,default=float); print("wrote",OUT)
