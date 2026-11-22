import openvino as ov, numpy as np, torch, re, shutil, os, json, gc
from openvino import opset13 as ops
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from datasets import load_dataset
core=ov.Core(); PAT=re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)")
def mats(m):
    d={}
    for op in m.get_ordered_ops():
        if op.get_type_name()=="MatMul":
            mm=PAT.search(op.get_friendly_name())
            if mm: d.setdefault(f"layers.{mm.group(1)}.{mm.group(2)}.{mm.group(3)}",op)
    return d
rtn=mats(core.read_model("models/ov-rtn-int4/openvino_model.xml")); RW={}
def rw(k):
    if k not in RW: RW[k]=np.asarray(list(core.compile_model(ov.Model([rtn[k].input_value(1)],[]),"CPU").create_infer_request().infer({}).values())[0],np.float16)
    return RW[k]
SRC="models/ov-gptq-int4-se"; ALL=sorted(mats(core.read_model(SRC+"/openvino_model.xml")), key=lambda k:(int(k.split(".")[1]),k))
def patch(keys,dst):
    g=core.read_model(SRC+"/openvino_model.xml"); gm=mats(g)
    for k in keys:
        go=gm[k]; c=ops.convert(ops.constant(rw(k)), go.input_value(1).get_element_type()); go.input(1).replace_source_output(c.output(0))
    assert not os.path.exists(dst)
    shutil.copytree(SRC,dst,ignore=shutil.ignore_patterns("openvino_model.xml","openvino_model.bin"))
    ov.save_model(g,dst+"/openvino_model.xml",compress_to_fp16=False)
tok=AutoTokenizer.from_pretrained("models/fp16-src")
ids=tok("\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"]),add_special_tokens=False).input_ids
L=2048; starts=[0,50000,120000,200000]; bins=[0,32,128,512,2047]
def ev(path,conds=("none",)):
    mdl=OVModelForCausalLM.from_pretrained(path,device="GPU"); out={}
    for c in conds:
        N=[]
        for s in starts:
            pre=[tok.bos_token_id] if c=="bos" else []
            x=np.array(pre+ids[s:s+L-len(pre)])
            lg=mdl(input_ids=torch.tensor(x[None]),attention_mask=torch.ones(1,len(x),dtype=torch.long)).logits[0].float()
            N.append((-torch.log_softmax(lg,-1)[:-1].gather(1,torch.tensor(x[1:])[:,None])[:,0]).numpy()[len(pre):])
        N=np.stack(N); out[c]=dict(ppl=float(np.exp(N.mean())),bins=[round(float(N[:,lo:hi].mean()),2) for lo,hi in zip(bins[:-1],bins[1:])])
    del mdl; gc.collect(); return out
R={}; J="diag/se_bisect.json"
def log(name,r): R[name]=r; print(name,json.dumps(r),flush=True); json.dump(R,open(J,"w"),indent=1)
R.update(json.load(open(J)))
for m in ["ov-gptq-int4","ov-gptq-int4-l1down-rtn","ov-rtn-int4","ov-fp16","ov-gptq-int4-se"]:
    if m not in R: log(m,ev("models/"+m,("bos","none")))
thr=1.5*R["ov-gptq-int4-se"]["bos"]["ppl"]; CNT=[0]
def test(name,keys):
    CNT[0]+=1; TMP=f"models/tmp-se-patch{CNT[0]}"; patch(keys,TMP); r=ev(TMP); shutil.rmtree(TMP,ignore_errors=True); r["n"]=len(keys); log("se+RTN["+name+"]",r); return r["none"]["ppl"]<thr
L15=[k for k in ALL if k.startswith("layers.15.")]
cands=[("L15all",L15),("L15mlp",[k for k in L15 if ".mlp." in k]),("L15attn",[k for k in L15 if "self_attn" in k])]
for a,b in [(0,3),(4,7),(8,11),(12,14)]: cands.append((f"L{a}-{b}",[k for k in ALL if a<=int(k.split(".")[1])<=b]))
fix=[(n,k) for n,k in cands if test(n,k)]
if not fix: fix=[("ALL",ALL)] if test("ALL",ALL) else []
cur=min(fix,key=lambda t:len(t[1])) if fix else None
while cur and len(cur[1])>1:
    k=cur[1]; h=len(k)//2; A,B=k[:h],k[h:]
    nxt=[(n,s) for n,s in [(f"{cur[0]}/a",A),(f"{cur[0]}/b",B)] if test(n+":"+",".join(s) if len(s)<=4 else n,s)]
    if not nxt: print("no half fixes alone; culprit set =",k,flush=True); break
    cur=nxt[0]

print("WINNER",cur,flush=True); json.dump({"name":cur[0],"keys":cur[1]} if cur else None,open("diag/se_winner.json","w"))
if cur: patch(cur[1],"models/ov-gptq-int4-se-patched"); print("saved models/ov-gptq-int4-se-patched",flush=True)

