import torch, glob
from safetensors.torch import load_file
ref={}
for f in glob.glob("models/fp16-src/*.safetensors"): ref.update(load_file(f))
k="model.layers.1.mlp.down_proj.weight"; r=ref[k].float()
D={n:torch.load(f"diag/deq_{n}.pt",mmap=True)[k].reshape(r.shape) for n in ["ov-rtn-int4","ov-gptq-int4"]}
g=D["ov-gptq-int4"]; e=(g-r).abs()
idx=torch.topk(e.flatten(),10).indices
print("fp16 |w| max", r.abs().max().item(), "at", divmod(int(r.abs().argmax()), r.shape[1]))
for i in idx:
    o,c=divmod(int(i), r.shape[1])
    print(f"out {o} in {c} group {c//128}: fp16 {r[o,c]:.4f} rtn {D['ov-rtn-int4'][o,c]:.4f} gptq {g[o,c]:.4f}  gptq group absmax {g[o,(c//128)*128:(c//128+1)*128].abs().max():.4f}")
# per-in-channel column error norm
col=(g-r).norm(dim=0); colr=(D['ov-rtn-int4']-r).norm(dim=0)
t=torch.topk(col,5); print("worst input columns gptq:", t.indices.tolist(), [round(v,3) for v in t.values.tolist()], "rtn same cols:", [round(colr[i].item(),3) for i in t.indices])
# across all layers: count elements with abs err > 0.3
for n in ["ov-rtn-int4","ov-awq-int4","ov-gptq-int4"]:
    d=torch.load(f"diag/deq_{n}.pt",mmap=True); big=[]
    for kk,w in d.items():
        rr=ref[kk].float(); m=(w.reshape(rr.shape)-rr).abs().max().item()
        if m>0.2: big.append((kk,round(m,3)))
    print(n, "weights with max abs err > 0.2:", big)
