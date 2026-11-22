import openvino as ov, numpy as np, torch, re, glob, json
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
torch.set_grad_enabled(False)
tok=AutoTokenizer.from_pretrained("models/fp16-src")
ids=tok("\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"]),add_special_tokens=False).input_ids
model=AutoModelForCausalLM.from_pretrained("models/fp16-src",dtype=torch.float32).eval()
S={}; cur={}
def hook(name):
    def f(m,i,o):
        x=i[0][0].float(); c=cur["c"]; st=1  # exclude pos0 (BOS or first text token)
        d=S.setdefault((c,name),dict(abs=0,sq=0,n=0,p0=0,mx=0))
        d["abs"]=d["abs"]+x[st:].abs().sum(0); d["sq"]=d["sq"]+(x[st:]**2).sum(0); d["n"]+=x.shape[0]-st
        d["p0"]=d["p0"]+x[0].abs(); d["mx"]=torch.maximum(torch.as_tensor(d["mx"]),x[st:].abs().max(0).values)
    return f
for i,l in enumerate(model.model.layers):
    for g,mod in [("self_attn",l.self_attn),("mlp",l.mlp)]:
        for p in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
            if hasattr(mod,p): getattr(mod,p).register_forward_hook(hook(f"layers.{i}.{g}.{p}"))
for c in ["bos","none"]:
    cur["c"]=c
    for s in [0,50000,120000,200000]:
        pre=[tok.bos_token_id] if c=="bos" else []
        model(input_ids=torch.tensor([pre+ids[s:s+2048-len(pre)]]))
    print("done",c,flush=True)
A={k:{kk:(v/d["n"] if kk in("abs","sq") else v/4 if kk=="p0" else v) for kk,v in d.items() if kk!="n"} for k,d in S.items()}
ref={}
for f in glob.glob("models/fp16-src/*.safetensors"): ref.update(load_file(f))
core=ov.Core(); PAT=re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)")
def colerr(md):
    out={}
    for op in core.read_model(md+"/openvino_model.xml").get_ordered_ops():
        if op.get_type_name()!="MatMul": continue
        mm=PAT.search(op.get_friendly_name())
        if not mm: continue
        k=f"layers.{mm.group(1)}.{mm.group(2)}.{mm.group(3)}"
        if k in out: continue
        r=ref["model."+k+".weight"].float().numpy()
        w=np.asarray(list(core.compile_model(ov.Model([op.input_value(1)],[]),"CPU").create_infer_request().infer({}).values())[0],np.float32).reshape(r.shape)
        e=w-r; out[k]=(np.linalg.norm(e,axis=0),np.abs(e).max(0))
    return out
CE={m:colerr("models/"+m) for m in ["ov-gptq-int4","ov-gptq-int4-se","ov-rtn-int4"]}
def rank(v,j): return int((v>v[j]).sum())  # 0 = largest
def desc(k,j):
    b=A[("bos",k)]; n=A[("none",k)]
    return dict(col=j, bos_meanabs=round(float(b["abs"][j]),4), none_meanabs=round(float(n["abs"][j]),4), ratio_none_bos=round(float(n["abs"][j]/b["abs"][j]),2),
      bos_rms=round(float(b["sq"][j]**.5),4), none_rms=round(float(n["sq"][j]**.5),4), bos_rank_meanabs=rank(b["abs"].numpy(),j), none_rank=rank(n["abs"].numpy(),j), nch=len(b["abs"]),
      pos0_abs_bos=round(float(b["p0"][j]),3), pos0_abs_none=round(float(n["p0"][j]),3), bos_max=round(float(b["mx"][j]),3), none_max=round(float(n["mx"][j]),3),
      colerr={m:round(float(CE[m][k][0][j]),3) for m in CE})
out={}
win=json.load(open("diag/se_winner.json")) or {"keys":[]}
targets=[("ov-gptq-int4","layers.1.mlp.down_proj")]+[("ov-gptq-int4-se",k) for k in win["keys"]]+[("ov-gptq-int4-se",k) for k in ["layers.15.mlp.up_proj","layers.15.self_attn.o_proj","layers.15.mlp.gate_proj"]]
for m,k in dict.fromkeys(targets):
    ce,me=CE[m][k]; rc=CE["ov-rtn-int4"][k][0]; top=np.argsort(ce/np.maximum(rc,1e-6))[::-1][:5]
    out[f"{m}|{k}"]=[desc(k,int(j)) for j in dict.fromkeys([int(np.argmax(me))]+[int(t) for t in top])]
    print(m,k,json.dumps(out[f"{m}|{k}"]),flush=True)
# global test: across all matrices, is excess column error (gptq/rtn) correlated with low bos activation or with none/bos ratio?
from scipy.stats import spearmanr
glob_={}
for m in ["ov-gptq-int4","ov-gptq-int4-se"]:
    rb=[];rr=[]; topstats=[]
    for k in CE[m]:
        ex=CE[m][k][0]/np.maximum(CE["ov-rtn-int4"][k][0],1e-6); b=A[("bos",k)]["sq"].numpy(); n=A[("none",k)]["sq"].numpy()
        rb.append(spearmanr(ex,b).statistic); rr.append(spearmanr(ex,n/np.maximum(b,1e-12)).statistic)
        j=int(np.argmax(ex)); topstats.append((float(ex[j]),k,j,float(n[j]/b[j]),rank(b,j)/len(b)))
    topstats.sort(reverse=True)
    glob_[m]=dict(median_spearman_excess_vs_bosact=float(np.median(rb)),median_spearman_excess_vs_ratio=float(np.median(rr)),
       top10_excess_cols=[dict(k=k,col=j,excess=round(e,1),none_over_bos_ms=round(r,2),bos_rank_frac=round(q,3)) for e,k,j,r,q in topstats[:10]])
    print(m,json.dumps(glob_[m]),flush=True)
json.dump(dict(targets=out,globl=glob_),open("diag/act_check.json","w"),indent=1)
