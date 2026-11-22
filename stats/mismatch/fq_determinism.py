"""CPU check: is the notebook's fake quantizer (dev/cells.py, fake_quant_weight) deterministic and
independent of thread count / call order / float rounding path? Uses Llama-3.2-1B layer shapes on
synthetic N(0, 0.02) weights with heavy-tailed outliers."""
import re, torch, json, sys
src=open("/workspace/lora/dev/cells.py").read()
m=re.search(r"_EPS32 = .*?\ndef quant_plan", src, re.S).group(0).rsplit("def quant_plan",1)[0]
ns={"torch":torch}; exec(m, ns); fq=ns["fake_quant_weight"]
g=torch.Generator().manual_seed(0)
shapes={"q_proj":(2048,2048),"k_proj":(512,2048),"down_proj":(2048,8192),"gate_proj":(8192,2048),"embed(rows subset)":(16384,2048)}
res={}
for name,shp in shapes.items():
    w=(torch.randn(shp,generator=g)*0.02); w[torch.rand(shp,generator=g)<1e-4]*=40   # outliers
    w=w.to(torch.bfloat16).float()   # checkpoint is bf16 -> exact fp32 copies, as in load_base
    for mode,gs in ([("int8_sym",-1)] if "embed" not in name else [("int8_sym",-1),("int8_asym",-1)])+([("int4_sym",128)] if "embed" not in name else []):
        outs=[]
        for nt in (1,4,16):
            torch.set_num_threads(nt); outs.append(fq(w.clone(),mode,gs,True))
        outs.append(fq(w.clone().contiguous().t().contiguous().t(),mode,gs,True))   # non-contiguous-origin copy
        # float64 reference of the same formula, rounded to fp32 where the notebook is fp32
        same=all(torch.equal(outs[0],o) for o in outs[1:])
        half=outs[0].half().float(); fp16_exact=torch.equal(half,outs[0])
        res[f"{name}/{mode}"]=dict(bit_identical_across_threads_and_layouts=same, dq_values_exact_in_fp16=fp16_exact)
        print(f"{name:20s} {mode:9s} identical={same}  dequantized values exactly representable in fp16={fp16_exact}")
json.dump(dict(torch=torch.__version__,results=res),open("fq_determinism.json","w"),indent=1)
