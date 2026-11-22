"""Paired bootstrap + exact McNemar over questions for Phase 2.
usage: python3 paired_bootstrap.py <eval_samples_dir> [out.json] [saved_eval_dir]

Input: <dir>/<name>.json.gz from the Colab re-score cell, format
  {name, base, adapter, lm_eval_version, per_task: {task: {metric, doc_id[], correct[]}}}
for name in {fp16,int8-rtn,int4-rtn}-{nolora,seed0,seed1,seed2}.

Primary test (macro average = mean of the 6 per-task accuracies, as in the paper):
  stratified paired bootstrap, B=10,000. Each resample draws questions with replacement independently
  within each task; the SAME indices are used for both models. LoRA cells use per-question correctness
  averaged over seeds 0-2 (values 0, 1/3, 2/3, 1). Two-sided p = 2*min(P(d*<=0), P(d*>=0)) with +1
  smoothing, so the smallest reportable p is 2/(B+1) ~= 0.0002. 95% CI = percentile interval.
  Holm over the 7 main comparisons.
Secondary test: exact two-sided McNemar (binomial on discordant pairs), per task and pooled over all
  6 tasks (pooled = question-weighted, so HellaSwag dominates). McNemar needs 0/1 outcomes, so for LoRA
  cells it is run once per seed (seed k vs seed k, or seed k vs the no-LoRA base); the reported p for a
  comparison is the LARGEST of the 3 seed p-values (conservative), then Holm over the 7.
Also: seed-level macro diffs, and a check of each model's accuracy against the saved eval JSONs.
Changes vs the synthetic-tested v0: doc_id alignment is asserted for every model (v0 skipped the
  int8/int4 no-LoRA files), McNemar is computed for every comparison and seed (v0: 4 pairs, seed 0 only),
  pooled McNemar + Holm added, p-value floor recorded, saved-accuracy check added. Bootstrap unchanged."""
import sys, gzip, json, os, numpy as np
from scipy.stats import binomtest
D=sys.argv[1]; OUT=sys.argv[2] if len(sys.argv)>2 else "paired_bootstrap.json"
SAVED=sys.argv[3] if len(sys.argv)>3 else None
T=["arc_challenge","arc_easy","hellaswag","winogrande","boolq","piqa"]; B=10000; rng=np.random.default_rng(20261119)
EXPECT_METRIC={"arc_challenge":"acc_norm","arc_easy":"acc_norm","hellaswag":"acc_norm","piqa":"acc_norm","winogrande":"acc","boolq":"acc"}
NAMES=[f"{b}-{k}" for b in ["fp16","int8-rtn","int4-rtn"] for k in ["nolora","seed0","seed1","seed2"]]
raw,ids={},None
for n in NAMES:
    d=json.load(gzip.open(f"{D}/{n}.json.gz","rt"))
    assert d["name"]==n, (d["name"],n)
    pt=d["per_task"]; assert set(pt)==set(T), set(pt)
    for t in T:
        assert pt[t]["metric"]==EXPECT_METRIC[t], (n,t,pt[t]["metric"])
        assert len(pt[t]["doc_id"])==len(pt[t]["correct"])==len(set(pt[t]["doc_id"]))
        assert set(pt[t]["correct"])<= {0,1}
    if ids is None: ids={t:pt[t]["doc_id"] for t in T}
    assert all(pt[t]["doc_id"]==ids[t] for t in T), f"doc_id order differs in {n}"
    raw[n]={t:np.array(pt[t]["correct"],float) for t in T}
NQ={t:len(ids[t]) for t in T}
cell={}
for b in ["fp16","int8-rtn","int4-rtn"]:
    cell[b]=raw[f"{b}-nolora"]
    for s in range(3): cell[f"{b}+s{s}"]=raw[f"{b}-seed{s}"]
    cell[b+"+L"]={t:np.mean([raw[f"{b}-seed{s}"][t] for s in range(3)],0) for t in T}
acc={n:{t:float(raw[n][t].mean()) for t in T} for n in NAMES}
saved_check={}
if SAVED:
    for n in NAMES:
        s=json.load(open(os.path.join(SAVED,f"{n}.json")))
        saved_check[n]={t:dict(rescore=acc[n][t],saved=s["metrics"][t]["value"],
                               net_questions=int(round((acc[n][t]-s["metrics"][t]["value"])*NQ[t]))) for t in T}
idx={t:rng.integers(0,NQ[t],(B,NQ[t])) for t in T}
PMIN=2/(B+1)
def pval(bs): return min(1,2*min(((bs<=0).sum()+1)/(B+1),((bs>=0).sum()+1)/(B+1)))
def boot(a,b):
    d={t:cell[a][t]-cell[b][t] for t in T}
    obs=np.mean([d[t].mean() for t in T])*100
    bs=np.mean([d[t][idx[t]].mean(1) for t in T],0)*100
    return obs,*np.percentile(bs,[2.5,97.5]),pval(bs),float(bs.std(ddof=1))
def boot_task(a,b,t):
    d=cell[a][t]-cell[b][t]; bs=d[idx[t]].mean(1)*100
    return d.mean()*100,*np.percentile(bs,[2.5,97.5]),pval(bs)
def mcn(x,y):
    n01=int(((x==1)&(y==0)).sum()); n10=int(((x==0)&(y==1)).sum()); n=n01+n10
    p=binomtest(min(n01,n10),n,0.5).pvalue if n else 1.0
    return dict(a_only=n01,b_only=n10,p=float(p))
def seed_pairs(a,b):
    """Binary (seed-level) versions of a comparison: one pair per seed for LoRA cells."""
    if "+L" not in a and "+L" not in b: return [(a,b)]
    return [(a.replace("+L",f"+s{s}"),b.replace("+L",f"+s{s}")) for s in range(3)]
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); out=[None]*len(ps); run=0
    for k,i in enumerate(order): run=max(run,min(1,(len(ps)-k)*ps[i])); out[i]=run
    return out
fam=[("RTN INT8 vs FP16 (no LoRA)","int8-rtn","fp16"),("RTN INT4 vs FP16 (no LoRA)","int4-rtn","fp16"),
     ("LoRA effect, FP16","fp16+L","fp16"),("LoRA effect, RTN INT8","int8-rtn+L","int8-rtn"),
     ("LoRA effect, RTN INT4","int4-rtn+L","int4-rtn"),("INT8+LoRA vs FP16+LoRA","int8-rtn+L","fp16+L"),
     ("INT4+LoRA vs FP16+LoRA","int4-rtn+L","fp16+L"),("INT4+LoRA vs plain FP16 (secondary)","int4-rtn+L","fp16")]
res=[boot(a,b) for _,a,b in fam]
mc_pooled=[]; mc_task={}
for n,a,b in fam:
    per_seed=[]
    for x,y in seed_pairs(a,b):
        X=np.concatenate([cell[x][t] for t in T]); Y=np.concatenate([cell[y][t] for t in T])
        per_seed.append(dict(pair=f"{x} vs {y}",**mcn(X,Y)))
    mc_pooled.append(per_seed)
    mc_task[n]={t:[dict(pair=f"{x} vs {y}",**mcn(cell[x][t],cell[y][t])) for x,y in seed_pairs(a,b)] for t in T}
p_boot=[r[3] for r in res[:7]]; p_mc=[max(d["p"] for d in mc_pooled[i]) for i in range(7)]
h_boot=holm(p_boot); h_mc=holm(p_mc)
out={"settings":dict(B=B,rng_seed=20261119,tasks=T,n_questions=NQ,lora_cell="per-question mean over seeds 0-2",
                     p_floor=PMIN,ci="percentile 95%",mcnemar="exact binomial, two-sided; LoRA: max p over 3 seeds"),
     "accuracy":acc,"saved_check":saved_check,"main":[],"per_task":{},"mcnemar_per_task":mc_task}
for i,(n,a,b) in enumerate(fam):
    o,lo,hi,p,se=res[i]
    seed=[float(np.mean([(cell[x][t]-cell[y][t]).mean() for t in T])*100) for x,y in seed_pairs(a,b)] if "+L" in a else None
    rec=dict(name=n,a=a,b=b,diff=o,lo=lo,hi=hi,boot_se=se,p=p,p_at_floor=bool(p<=PMIN+1e-12),
             p_holm=h_boot[i] if i<7 else None,mcnemar_pooled=mc_pooled[i],
             mcnemar_p=p_mc[i] if i<7 else max(d["p"] for d in mc_pooled[i]),mcnemar_p_holm=h_mc[i] if i<7 else None,seed_diffs=seed)
    out["main"].append(rec)
    print(f"{n:38s} {o:+.2f} [{lo:+.2f},{hi:+.2f}] p_boot={p:.4f} holm={rec['p_holm']} McN={rec['mcnemar_p']:.2e} holmMcN={rec['mcnemar_p_holm']} seeds={None if seed is None else [round(s,2) for s in seed]}")
for n,a,b in fam:
    out["per_task"][n]={t:dict(zip(["diff","lo","hi","p"],boot_task(a,b,t))) for t in T}
json.dump(out,open(OUT,"w"),indent=1,default=float); print("wrote",OUT)
