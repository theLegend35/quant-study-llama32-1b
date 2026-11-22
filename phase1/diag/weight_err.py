import openvino as ov, numpy as np, torch, re, json, sys
from safetensors.torch import load_file
import glob
ref = {}
for f in glob.glob("models/fp16-src/*.safetensors"): ref.update(load_file(f))
core = ov.Core()
def deq(model_dir):
    m = core.read_model(model_dir+"/openvino_model.xml")
    out = {}; sc = {}
    for op in m.get_ordered_ops():
        if op.get_type_name()!="MatMul": continue
        mm = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)", op.get_friendly_name())
        if not mm: continue
        key = f"model.layers.{mm.group(1)}.{mm.group(2)}.{mm.group(3)}.weight"
        wout = op.input_value(1)
        sub = ov.Model([wout], [])
        w = core.compile_model(sub, "CPU")({})[0] if False else core.compile_model(sub,"CPU").create_infer_request().infer({})
        w = list(w.values())[0]
        if op.get_attributes().get("transpose_b") is False: w = w.T
        out[key] = torch.from_numpy(np.array(w, dtype=np.float32))
        # scale stats
        n = wout.get_node()
        while n.get_type_name()!="Multiply": n = n.input_value(0).get_node()
        s = n.input_value(1).get_node()
        if s.get_type_name()=="Convert": s = s.input_value(0).get_node()
        sd = np.abs(s.get_data().astype(np.float32)); sc[key]=(float(sd.max()), float(sd.min()), int((sd==0).sum()), int((~np.isfinite(sd)).sum()))
    return out, sc
if __name__=="__main__":
    rows = {}
    for name in ["ov-rtn-int4","ov-awq-int4","ov-gptq-int4"]:
        d, sc = deq("models/"+name)
        torch.save(d, f"diag/deq_{name}.pt")
        for k,w in d.items():
            r = ref[k].float()
            if w.shape!=r.shape: w = w.T
            e = (w-r)
            rows.setdefault(k,{})[name] = dict(rel=float(e.norm()/r.norm()), maxabs=float(e.abs().max()), smax=sc[k][0], smin=sc[k][1], szero=sc[k][2], snonfin=sc[k][3])
        print(name, len(d), flush=True)
    json.dump(rows, open("diag/weight_err.json","w"), indent=1)
    print(f"{'layer':45s} {'rtn':>7s} {'awq':>7s} {'gptq':>7s} {'g/rtn':>6s} gptq_maxabs  gptq_scale_max/min/zero/nonfin")
    for k,v in rows.items():
        g=v["ov-gptq-int4"]; print(f"{k:45s} {v['ov-rtn-int4']['rel']:.4f} {v['ov-awq-int4']['rel']:.4f} {g['rel']:.4f} {g['rel']/v['ov-rtn-int4']['rel']:6.2f} {g['maxabs']:.4f} {g['smax']:.4g}/{g['smin']:.3g}/{g['szero']}/{g['snonfin']}")
