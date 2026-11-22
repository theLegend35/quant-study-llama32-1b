# Sanity check 2 (reference side): FP16 source weights in PyTorch fp32 on CPU, with the PEFT adapter applied by hand
# (y = W x + (lora_alpha/r) * B A x on q/k/v/o, exactly PEFT's unmerged LoRA forward, dropout off). Greedy 40 tokens.
# Compared against check_ov.json to confirm OpenVINO GenAI matched every key and applied the PEFT scale (alpha=2).
import json, torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
P = json.load(open("speed-lora/check_ov.json"))["prompts"]
tok = AutoTokenizer.from_pretrained("models/fp16-src")
model = AutoModelForCausalLM.from_pretrained("models/fp16-src", torch_dtype=torch.float32).eval()
sd = load_file("lora-results/fp16-seed0/adapter_model.safetensors")
mods = dict(model.named_modules()); hooks = []; matched = 0
def make_hook(A, B, s):
    def h(mod, inp, out): return out + s * (inp[0] @ A.t() @ B.t())
    return h
def gen():
    outs = []
    for p in P:
        enc = tok(p, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False)
        outs.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return outs
res = {"ref_off": gen()}
for scale, key in [(2.0, "ref_a2"), (1.0, "ref_a1")]:
    for h in hooks: h.remove()
    hooks = []; matched = 0
    for k in sd:
        if k.endswith("lora_A.weight"):
            name = k[len("base_model.model."):-len(".lora_A.weight")]
            A = sd[k].float(); B = sd[k.replace("lora_A", "lora_B")].float()
            hooks.append(mods[name].register_forward_hook(make_hook(A, B, scale))); matched += 1
    res[key] = gen(); res[key + "_matched_modules"] = matched
for h in hooks: h.remove()
ov = json.load(open("speed-lora/check_ov.json"))["outputs"]["fp16"]
for i in range(len(P)):
    for a in ("off", "on_a2", "on_a1"):
        r = {"off": "ref_off", "on_a2": "ref_a2", "on_a1": "ref_a1"}[a]
        o, t = ov[a][i], res[r][i]
        n = next((j for j in range(min(len(o), len(t))) if o[j] != t[j]), min(len(o), len(t)))
        print(f"prompt{i} OV {a} vs {r}: identical={o == t} common_prefix_chars={n}/{max(len(o),len(t))}")
    print(f"prompt{i} OV on_a2 vs ref_a1: common_prefix_chars="
          f"{next((j for j in range(min(len(ov['on_a2'][i]), len(res['ref_a1'][i]))) if ov['on_a2'][i][j] != res['ref_a1'][i][j]), -1)}")
json.dump(res, open("speed-lora/check_ref.json", "w"), indent=1)
print(json.dumps(res, indent=1))
