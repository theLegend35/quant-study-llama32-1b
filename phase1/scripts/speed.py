import sys, json, statistics as st, psutil, openvino_genai as og
m = sys.argv[1]; path = f"models/ov-{m}"
pipe = og.LLMPipeline(path, "GPU")
tok = pipe.get_tokenizer()
base = "The history of science is full of surprising discoveries made by careful observers. " * 40
ids = tok.encode(base).input_ids.data[0][:128].tolist()
prompt = tok.decode(ids)
cfg = og.GenerationConfig(); cfg.max_new_tokens = 128; cfg.min_new_tokens = 128; cfg.do_sample = False
ttft, tps = [], []
for i in range(35):
    r = pipe.generate([prompt], cfg)
    pm = r.perf_metrics
    if i >= 5:
        ttft.append(pm.get_ttft().mean); tps.append(pm.get_throughput().mean)
def q(a): s=sorted(a); return s[int(0.025*len(s))], s[min(len(s)-1,int(0.975*len(s)))]
out = dict(model=m, prompt_tokens=len(ids), ttft_ms_median=st.median(ttft), ttft_ms_range95=q(ttft),
  tok_per_s_median=st.median(tps), tok_per_s_range95=q(tps),
  peak_rss_mb=psutil.Process().memory_info().peak_wset/2**20)
print(json.dumps(out)); open(f"results/speed-{m}.json","w").write(json.dumps(out, indent=1))
