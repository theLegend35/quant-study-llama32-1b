"""Collect evidence for the Phase 2 re-score mismatch: batch sizes from the saved Colab notebooks (Drive),
env versions from the saved eval JSONs, and net per-task changes (re-score minus saved)."""
import json, re, gzip, os
NB={"orig_2026-11-12":"/workspace/lora/drive_nb/nb_1112_0909.ipynb","rescore_2026-11-20/21":"/workspace/lora/drive_nb/rescore_1121.ipynb"}
bs={}
for k,f in NB.items():
    nb=json.load(open(f)); cur=None; d={}
    for c in nb["cells"]:
        for o in c.get("outputs",[]):
            for l in re.split(r"[\r\n]","".join(o.get("text",""))):
                m=re.search(r"\[(eval|rescore)\] (\S+) \.\.\.",l)
                if m: cur=m.group(2)
                m=re.search(r"Determined largest batch size: (\d+)",l)
                if m and cur: d[cur]=int(m.group(1))
    bs[k]=d
T=["arc_challenge","arc_easy","hellaswag","winogrande","boolq","piqa"]
rows={}
for n in sorted(os.listdir("/workspace/lora/results/eval")):
    n=n[:-5]; s=json.load(open(f"/workspace/lora/results/eval/{n}.json"))
    r=json.load(gzip.open(f"/workspace/lora/eval_samples/eval_samples/{n}.json.gz","rt"))["per_task"]
    net={t:int(round((sum(r[t]["correct"])/len(r[t]["correct"])-s["metrics"][t]["value"])*len(r[t]["correct"]))) for t in T}
    avg=sum(sum(r[t]["correct"])/len(r[t]["correct"])-s["metrics"][t]["value"] for t in T)/6*100
    rows[n]=dict(net_questions=net,avg_change_points=avg,env=s["env"],lm_eval=s["lm_eval_version"],
                 bs_orig=bs["orig_2026-11-12"].get(n),bs_rescore=bs["rescore_2026-11-20/21"].get(n),exact=all(v==0 for v in net.values()))
drive_times={"eval/*.json (original)":{"fp16-nolora":"2026-11-12T08:25Z","int8-rtn-nolora":"2026-11-12T08:41Z","int4-rtn-nolora":"2026-11-12T08:57Z",
  "fp16-seed0":"2026-11-12T09:14Z","fp16-seed1":"2026-11-12T09:32Z","fp16-seed2":"2026-11-19T01:30Z","int8-rtn-seed0":"2026-11-19T01:47Z",
  "int8-rtn-seed1":"2026-11-19T02:05Z","int8-rtn-seed2":"2026-11-19T02:23Z","int4-rtn-seed0":"2026-11-19T02:40Z","int4-rtn-seed1":"2026-11-19T02:57Z","int4-rtn-seed2":"2026-11-19T03:14Z"},
  "training run folders":"2026-11-12T06:00Z..07:58Z (same session as the first 5 evals)"}
out=dict(batch_sizes=bs,models=rows,drive_modified_times_utc=drive_times)
json.dump(out,open("mismatch_evidence.json","w"),indent=1)
for n,v in rows.items(): print(f"{n:16s} bs {v['bs_orig']}->{v['bs_rescore']} exact={v['exact']} net={v['net_questions']} avgΔ={v['avg_change_points']:+.3f}")
print("envs identical:",len({json.dumps(v['env'],sort_keys=True)+v['lm_eval'] for v in rows.values()})==1)
