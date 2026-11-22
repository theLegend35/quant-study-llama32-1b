import torch, numpy as np, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("models/fp16-src")
text = "\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"])
ids = tok(text, add_special_tokens=False).input_ids
L=512
X = {"none": torch.tensor([ids[50000:50000+L]]), "bos": torch.tensor([[tok.bos_token_id]+ids[50000:50000+L-1]])}
model = AutoModelForCausalLM.from_pretrained("models/fp16-src", dtype=torch.float32)
orig = {k:v.clone() for k,v in model.state_dict().items() if "_proj.weight" in k}
act = {}
model.model.layers[1].mlp.down_proj.register_forward_hook(lambda m,i,o: act.__setitem__("in", i[0][0].abs()))
def setw(d):
    sd = model.state_dict()
    for k in orig: sd[k].copy_(d[k].reshape(orig[k].shape) if k in d else orig[k])
def run(x):
    out = model(input_ids=x, output_hidden_states=True)
    lp = torch.log_softmax(out.logits[0],-1)
    nll = -lp[:-1].gather(1,x[0,1:,None])[:,0]
    return float(torch.exp(nll.mean())), [h[0].clone() for h in out.hidden_states]
setw({}); ref={}
for t,x in X.items():
    ref[t]=run(x); a=act["in"]
    print(f"fp16 {t} ppl {ref[t][0]:.2f}; layer1 down_proj input |ch2558| pos0..3 {a[:4,2558].tolist()} median-over-pos {a[4:,2558].median():.3f}; argmax channel at pos0 {int(a[0].argmax())} ({a[0].max():.1f})", flush=True)
G = torch.load("diag/deq_ov-gptq-int4.pt", mmap=True); R = torch.load("diag/deq_ov-rtn-int4.pt", mmap=True)
def report(name, d):
    setw(d)
    for t,x in X.items():
        p,hs = run(x); rhs=ref[t][1]
        rel=[float((h-r).norm()/r.norm()) for h,r in zip(hs,rhs)]
        rp0=[float((h[0]-r[0]).norm()/r[0].norm()) for h,r in zip(hs,rhs)]
        rr=[float((h[1:]-r[1:]).norm()/r[1:].norm()) for h,r in zip(hs,rhs)]
        print(f"{name} {t} ppl {p:.2f}", flush=True)
        print("   hidden rel err per layer output (all / pos0 / pos1+): " + " ".join(f"L{i-1}:{rel[i]:.2f}/{rp0[i]:.2f}/{rr[i]:.2f}" for i in range(1,len(rel))), flush=True)
report("RTN", {k:R[k] for k in orig})
report("GPTQ", {k:G[k] for k in orig})
k1="model.layers.1.mlp.down_proj.weight"
report("GPTQ+L1down=RTN", {k:(R[k] if k==k1 else G[k]) for k in orig})
report("GPTQ+L1down=FP16", {k:G[k] for k in orig if k!=k1})
print("single-layer swaps (GPTQ weights in layer i only), none-prefix ppl:")
for i in range(16):
    setw({k:G[k] for k in orig if f"layers.{i}." in k}); print(f"  layer {i}: {run(X['none'])[0]:.2f}", flush=True)
