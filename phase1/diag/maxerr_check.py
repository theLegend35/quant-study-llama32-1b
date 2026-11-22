import openvino as ov, numpy as np, torch, re, sys, glob
from safetensors.torch import load_file
ref={}
for f in glob.glob("models/fp16-src/*.safetensors"): ref.update(load_file(f))
core=ov.Core()
for md in sys.argv[1:]:
    m=core.read_model(md+"/openvino_model.xml"); worst=[]
    for op in m.get_ordered_ops():
        if op.get_type_name()!="MatMul": continue
        mm=re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)", op.get_friendly_name())
        if not mm: continue
        k=f"model.layers.{mm.group(1)}.{mm.group(2)}.{mm.group(3)}.weight"
        w=list(core.compile_model(ov.Model([op.input_value(1)],[]),"CPU").create_infer_request().infer({}).values())[0]
        r=ref[k].float().numpy(); w=np.asarray(w,np.float32).reshape(r.shape); e=w-r
        worst.append((float(np.abs(e).max()), k, float(np.linalg.norm(e)/np.linalg.norm(r)), int(np.linalg.norm(e,axis=0).argmax())))
    worst.sort(reverse=True)
    print(md, "top-3 max-abs-err matrices:", [(k.replace("model.",""),round(a,3),round(rel,3),"worstcol",c) for a,k,rel,c in worst[:3]], flush=True)
