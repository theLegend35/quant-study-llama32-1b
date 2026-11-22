import torch, numpy as np, json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("models/fp16-src")
text = "\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"])
ids = tok(text, add_special_tokens=False).input_ids
L = int(sys.argv[1]) if len(sys.argv)>1 else 1024
x_none = torch.tensor([ids[50000:50000+L]]); x_bos = torch.tensor([[tok.bos_token_id]+ids[50000:50000+L-1]])
model = AutoModelForCausalLM.from_pretrained("models/fp16-src", torch_dtype=torch.float32, attn_implementation="eager")
orig = {k:v.clone() for k,v in model.state_dict().items() if "_proj.weight" in k}
deq = {n: torch.load(f"diag/deq_{n}.pt") for n in ["ov-rtn-int4","ov-awq-int4","ov-gptq-int4"]}
def setw(d):
    sd = model.state_dict()
    for k in orig: sd[k].copy_(d.get(k, orig[k]).reshape(orig[k].shape) if k in d else orig[k])
def run(x):
    out = model(input_ids=x, output_hidden_states=True, output_attentions=True)
    lp = torch.log_softmax(out.logits[0].float(),-1)
    nll = -lp[:-1].gather(1,x[0,1:,None])[:,0]
    hs = [h[0] for h in out.hidden_states]
    sink = [a[0,:,:,0].mean().item() for a in out.attentions]  # mean attention mass on position 0
    return nll, hs, sink
res = {}
setw({}); ref = {}
for tag,x in [("none",x_none),("bos",x_bos)]:
    ref[tag] = run(x)
    print("fp16", tag, "ppl", float(torch.exp(ref[tag][0].mean())), flush=True)
for n,d in deq.items():
    setw(d)
    for tag,x in [("none",x_none),("bos",x_bos)]:
        nll, hs, sink = run(x); rn, rhs, rsink = ref[tag]
        rel = [float((h-r).norm()/r.norm()) for h,r in zip(hs,rhs)]
        rel_p0 = [float((h[0]-r[0]).norm()/r[0].norm()) for h,r in zip(hs,rhs)]
        rel_rest = [float((h[1:]-r[1:]).norm()/r[1:].norm()) for h,r in zip(hs,rhs)]
        n0 = [float(h[0].norm()) for h in hs]; rn0=[float(r[0].norm()) for r in rhs]
        res[f"{n}|{tag}"] = dict(ppl=float(torch.exp(nll.mean())), rel=rel, rel_p0=rel_p0, rel_rest=rel_rest, norm_p0=n0, ref_norm_p0=rn0, sink=sink, ref_sink=rsink)
        print(n, tag, "ppl", round(res[f"{n}|{tag}"]["ppl"],2), flush=True)
        print("  layer: rel_all / rel_pos0 / rel_rest / |h0| (ref) / sinkattn (ref)")
        for i in range(len(rel)):
            print(f"  L{i:2d} {rel[i]:.3f} {rel_p0[i]:.3f} {rel_rest[i]:.3f}  {n0[i]:8.1f} ({rn0[i]:8.1f})  {sink[i-1] if i>0 else float('nan'):.3f} ({rsink[i-1] if i>0 else float('nan'):.3f})", flush=True)
json.dump(res, open("diag/hidden_err.json","w"))
# single-layer swap: GPTQ weights in one layer only, none-prefix
g = deq["ov-gptq-int4"]
print("SWAP single layer GPTQ (rest fp16), none-prefix ppl:")
for i in range(16):
    setw({k:v for k,v in g.items() if f"layers.{i}." in k}); print(f"  layer {i}: {float(torch.exp(run(x_none)[0].mean())):.2f}", flush=True)
