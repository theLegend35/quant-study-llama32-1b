import openvino_genai as og
cfg = og.GenerationConfig(); cfg.max_new_tokens = 60; cfg.do_sample = False
prompts = ["The capital of France is", "Robert Boulter is an English film and television actor. He had a guest-starring role on"]
for m in ["awq-int4","gptq-int4"]:
    p = og.LLMPipeline(f"models/ov-{m}", "GPU")
    for q in prompts:
        print(f"[{m}] {q} >>> {p.generate(q, cfg)!r}")
    del p
