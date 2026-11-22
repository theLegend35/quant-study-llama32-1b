import time, openvino_genai as og
p = og.LLMPipeline(r"models\ov-fp16", "GPU")
t=time.time(); out = p.generate("The capital of France is", max_new_tokens=20); print(repr(out)); print("secs", round(time.time()-t,2))
