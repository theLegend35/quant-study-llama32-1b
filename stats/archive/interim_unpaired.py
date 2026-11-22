"""Interim significance analysis for Phase 2 from saved lm-eval aggregates (value + stderr per task).
NO per-example outputs were saved (log_samples=False), so a paired bootstrap is impossible here.
Unpaired normal-approx test: Var(diff) = sum over tasks of (SE_a^2 + SE_b^2)/T^2  (tasks independent).
Ignoring the positive within-question correlation makes this CONSERVATIVE (CIs too wide).
LoRA cells = mean over 3 seeds; eval SE of the seed-mean bounded above by the mean of the 3 SEs
(perfect correlation), plus seed variance s^2/3. Holm over the main family of 7 comparisons."""
import json, math, statistics as st
from statistics import NormalDist
class norm:
    @staticmethod
    def sf(z): return 1-NormalDist().cdf(z)
E="/workspace/lora/results/eval/"
T=["arc_challenge","arc_easy","hellaswag","winogrande","boolq","piqa"]
def load(n): return json.load(open(E+n+".json"))["metrics"]
cells={}
for b in ["fp16","int8-rtn","int4-rtn"]:
    m=load(b+"-nolora"); cells[b]={t:(m[t]["value"]*100,m[t]["stderr"]*100,0.0) for t in T}
    ss=[load(f"{b}-seed{s}") for s in range(3)]
    cells[b+"+L"]={t:(st.mean(x[t]["value"]*100 for x in ss),
                     st.mean(x[t]["stderr"]*100 for x in ss),
                     st.stdev([x[t]["value"]*100 for x in ss])) for t in T}
    cells[b+"+L_seeds"]=[{t:x[t]["value"]*100 for t in T} for x in ss]
def var(c,t): v,se,sd=cells[c][t]; return se**2+(sd**2/3)
def comp(a,b,tasks=T):
    d=st.mean(cells[a][t][0]-cells[b][t][0] for t in tasks)
    se=math.sqrt(sum(var(a,t)+var(b,t) for t in tasks))/len(tasks)
    p=2*norm.sf(abs(d)/se); return d,d-1.96*se,d+1.96*se,p
fam=[("RTN INT8 vs FP16 (no LoRA)","int8-rtn","fp16"),
     ("RTN INT4 vs FP16 (no LoRA)","int4-rtn","fp16"),
     ("LoRA effect, FP16","fp16+L","fp16"),
     ("LoRA effect, RTN INT8","int8-rtn+L","int8-rtn"),
     ("LoRA effect, RTN INT4","int4-rtn+L","int4-rtn"),
     ("INT8+LoRA vs FP16+LoRA","int8-rtn+L","fp16+L"),
     ("INT4+LoRA vs FP16+LoRA","int4-rtn+L","fp16+L"),
     ("INT4+LoRA vs plain FP16","int4-rtn+L","fp16")]
main=[0,1,2,3,4,5,6]   # Holm family; the last row is secondary (descriptive)
res=[comp(a,b) for _,a,b in fam]
order=sorted(main,key=lambda i:res[i][3]); holm={}; run=0
for k,i in enumerate(order):
    run=max(run,min(1,(len(main)-k)*res[i][3])); holm[i]=run
def seedview(a,b):
    if a.endswith("+L"):
        sa=cells[a+"_seeds"]
        sb=cells[b+"_seeds"] if b.endswith("+L") else [{t:cells[b][t][0] for t in T}]*3
        return [st.mean(sa[i][t]-sb[i][t] for t in T) for i in range(3)]
    return None
out=[]
for i,(name,a,b) in enumerate(fam):
    d,lo,hi,p=res[i]; sv=seedview(a,b)
    out.append(dict(name=name,diff=d,lo=lo,hi=hi,p=p,p_holm=holm.get(i),seed_diffs=sv))
    print(f"{name:28s} {d:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2g} holm={holm.get(i) if i in holm else '-'} seeds={['%+.2f'%x for x in sv] if sv else '-'}")
# per-task, LoRA effect on INT4 and INT4+L vs FP16+L
pt={}
for name,a,b in [("LoRA effect, RTN INT4","int4-rtn+L","int4-rtn"),("INT4+LoRA vs FP16+LoRA","int4-rtn+L","fp16+L"),("RTN INT4 vs FP16 (no LoRA)","int4-rtn","fp16"),("RTN INT8 vs FP16 (no LoRA)","int8-rtn","fp16")]:
    pt[name]={}
    for t in T:
        d,lo,hi,p=comp(a,b,[t]); pt[name][t]=dict(diff=d,lo=lo,hi=hi,p=p)
        print(f"  {name:28s} {t:13s} {d:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2g}")
json.dump(dict(main=out,per_task=pt,method=__doc__),open("/workspace/lora/stats/interim_unpaired.json","w"),indent=1)
