# Sanity check 2: is the GPU mismatch vs. the PyTorch reference just fp16 rounding, or a wrong scale / missed keys?
# Runs the fp16 base + fp16-seed0 adapter in OpenVINO GenAI with fp32 inference precision (GPU and CPU), at alpha 1 and 2,
# and the PyTorch fp32 reference (manual PEFT-style LoRA hooks) at scales 0, 1, 2. 8 prompts, greedy, 40 new tokens.
import json, sys, gc
P = ["The capital of France is",
     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n",
     "Robert Boulter is an English film and television actor. He had a guest-starring role on",
     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain why the sky is blue.\n\n### Response:\n",
     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a short poem about autumn.\n\n### Response:\n",
     "The three primary colors are",
     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the difference between a virus and a bacterium?\n\n### Response:\n",
     "In 1969, Neil Armstrong"]
which = sys.argv[1]
res = {}
if which == "ov":
    import openvino_genai as og
    ad = og.Adapter("lora-results/fp16-seed0/adapter_model.safetensors")
    for dev in ["GPU", "CPU"]:
        pipe = og.LLMPipeline("models/ov-fp16", dev, adapters=og.AdapterConfig(ad, 2.0), INFERENCE_PRECISION_HINT="f32")
        for name, ac in [("off", og.AdapterConfig()), ("a1", og.AdapterConfig(ad, 1.0)), ("a2", og.AdapterConfig(ad, 2.0))]:
            c = og.GenerationConfig(); c.max_new_tokens = 40; c.do_sample = False; c.adapters = ac
            res[f"ov_{dev}_f32_{name}"] = [pipe.generate(p, c) for p in P]
        del pipe; gc.collect()
else:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/fp16-src")
    model = AutoModelForCausalLM.from_pretrained("models/fp16-src", dtype=torch.float32).eval()
    sd = load_file("lora-results/fp16-seed0/adapter_model.safetensors"); mods = dict(model.named_modules())
    for scale in [0.0, 1.0, 2.0]:
        hooks = []
        for k in sd:
            if k.endswith("lora_A.weight"):
                A = sd[k].float(); B = sd[k.replace("lora_A", "lora_B")].float()
                hooks.append(mods[k[len("base_model.model."):-len(".lora_A.weight")]].register_forward_hook(
                    (lambda A, B: (lambda m, i, o: o + scale * (i[0] @ A.t() @ B.t())))(A, B)))
        outs = []
        for p in P:
            enc = tok(p, return_tensors="pt")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        res[f"ref_s{scale:g}"] = outs
        for h in hooks: h.remove()
json.dump(dict(prompts=P, outputs=res), open(f"speed-lora/check2_{which}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
