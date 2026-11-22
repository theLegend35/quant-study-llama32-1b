# Sanity check 1 (OpenVINO side): greedy outputs with vs without each seed-0 adapter, plus alpha=1 vs alpha=2 for fp16.
import sys, json, openvino_genai as og
PROMPTS = ["The capital of France is",
           "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n",
           "Robert Boulter is an English film and television actor. He had a guest-starring role on"]
cfg = og.GenerationConfig(); cfg.max_new_tokens = 40; cfg.do_sample = False
res = {}
for m, ln in [("fp16","fp16"),("rtn-int8","int8-rtn"),("rtn-int4","int4-rtn")]:
    ad = og.Adapter(f"lora-results/{ln}-seed0/adapter_model.safetensors")
    pipe = og.LLMPipeline(f"models/ov-{m}", "GPU", adapters=og.AdapterConfig(ad, 2.0))
    variants = {"off": og.AdapterConfig(), "on_a2": og.AdapterConfig(ad, 2.0)}
    if m == "fp16":
        variants["on_a1"] = og.AdapterConfig(ad, 1.0)
    res[m] = {}
    for vn, ac in variants.items():
        c = og.GenerationConfig(); c.max_new_tokens = 40; c.do_sample = False; c.adapters = ac
        res[m][vn] = [pipe.generate(p, c) for p in PROMPTS]
    # plain pipeline (no adapter machinery at all) for the "off" reference
    del pipe
    plain = og.LLMPipeline(f"models/ov-{m}", "GPU")
    res[m]["plain"] = [plain.generate(p, cfg) for p in PROMPTS]
    del plain
    for i in range(len(PROMPTS)):
        print(f"[{m}] prompt{i}: on!=off: {res[m]['on_a2'][i] != res[m]['off'][i]} | off==plain: {res[m]['off'][i] == res[m]['plain'][i]}")
json.dump(dict(prompts=PROMPTS, outputs=res), open("speed-lora/check_ov.json", "w"), indent=1)
print(json.dumps(res, indent=1))
