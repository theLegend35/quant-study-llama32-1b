import json, shutil
src=open("diag/se_bisect.py",encoding="utf-8-sig").read().split("R={}; J=")[0]; exec(src)
R={}; n=[0]
def t(name,s,keys):
    global SRC; SRC=s; n[0]+=1; d=f"models/tmp-x{n[0]}"; patch(keys,d); r=ev(d); shutil.rmtree(d,ignore_errors=True); R[name]=r; print(name,json.dumps(r),flush=True); json.dump(R,open("diag/se_extra.json","w"),indent=1)
SE="models/ov-gptq-int4-se"; G="models/ov-gptq-int4"
t("se+RTN[layers.1.mlp.down_proj]",SE,["layers.1.mlp.down_proj"])
t("se+RTN[layer1 except down]",SE,[k for k in ALL if k.startswith("layers.1.") and "down" not in k])
t("se+RTN[layer0 attn]",SE,[k for k in ALL if k.startswith("layers.0.self_attn")])
t("gptq+RTN[layers.0.mlp.down_proj]",G,["layers.0.mlp.down_proj"])
t("se+RTN[L0down+L1down]",SE,["layers.0.mlp.down_proj","layers.1.mlp.down_proj"])
