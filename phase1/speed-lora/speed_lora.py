# LoRA-on/off speed + memory benchmark. Mirrors Phase 1 speed.py EXACTLY (same prompt, 128 prompt tokens,
# max/min_new_tokens=128, greedy, GPU, 35 runs with the first 5 discarded, peak working set via psutil).
# Only difference for mode "on": the PEFT adapter is attached unmerged at runtime via openvino_genai.Adapter.
# Usage: python speed_lora.py <fp16|rtn-int8|rtn-int4> <off|on> <round> [alpha]
import sys, json, time, statistics as st, psutil, openvino_genai as og
from pathlib import Path
m, mode, rnd = sys.argv[1], sys.argv[2], sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0      # PEFT scale = lora_alpha / r = 32 / 16
path = f"models/ov-{m}"
lora_name = {"fp16": "fp16", "rtn-int8": "int8-rtn", "rtn-int4": "int4-rtn"}[m]
adapter_file = f"lora-results/{lora_name}-seed0/adapter_model.safetensors"
avail_start = psutil.virtual_memory().available/2**20
t0 = time.perf_counter()
if mode == "on":
    adapter = og.Adapter(adapter_file)
    ac = og.AdapterConfig(adapter, alpha)
    pipe = og.LLMPipeline(path, "GPU", adapters=ac)
else:
    pipe = og.LLMPipeline(path, "GPU")
load_s = time.perf_counter() - t0
tok = pipe.get_tokenizer()
base = "The history of science is full of surprising discoveries made by careful observers. " * 40
ids = tok.encode(base).input_ids.data[0][:128].tolist()
prompt = tok.decode(ids)
cfg = og.GenerationConfig(); cfg.max_new_tokens = 128; cfg.min_new_tokens = 128; cfg.do_sample = False
if mode == "on":
    cfg.adapters = ac
ttft, tps, tpot, first_text = [], [], [], None
me = psutil.Process(); psutil.cpu_percent(None); c0 = sum(me.cpu_times()[:2]); w0 = time.perf_counter()
for i in range(35):
    r = pipe.generate([prompt], cfg)
    pm = r.perf_metrics
    if i == 0:
        first_text = r.texts[0]
    if i >= 5:
        ttft.append(pm.get_ttft().mean); tps.append(pm.get_throughput().mean); tpot.append(pm.get_tpot().mean)
cpu_sys = psutil.cpu_percent(None); cpu_self = 100*(sum(me.cpu_times()[:2])-c0)/(time.perf_counter()-w0)/psutil.cpu_count()
def q(a): s=sorted(a); return s[int(0.025*len(s))], s[min(len(s)-1,int(0.975*len(s)))]
out = dict(model=m, mode=mode, round=int(rnd), alpha=alpha if mode == "on" else None,
  adapter=adapter_file if mode == "on" else None, prompt_tokens=len(ids), n_timed=len(tps), load_s=load_s,
  ttft_ms_median=st.median(ttft), ttft_ms_mean=st.mean(ttft), ttft_ms_std=st.stdev(ttft), ttft_ms_range95=q(ttft),
  tok_per_s_median=st.median(tps), tok_per_s_mean=st.mean(tps), tok_per_s_std=st.stdev(tps), tok_per_s_range95=q(tps),
  tpot_ms_mean=st.mean(tpot),
  peak_rss_mb=psutil.Process().memory_info().peak_wset/2**20,
  peak_commit_mb=psutil.Process().memory_info().peak_pagefile/2**20,
  avail_mb_at_start=avail_start, cpu_sys_pct_during=cpu_sys, cpu_self_pct_during=cpu_self,
  ttft_ms_all=ttft, tok_per_s_all=tps, first_output=first_text,
  sys_free_mb_at_end=psutil.virtual_memory().available/2**20,
  og_version=og.__version__)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_all")}))
Path("speed-lora").mkdir(exist_ok=True)
open(f"speed-lora/speed-{m}-{mode}-r{rnd}.json", "w").write(json.dumps(out, indent=1))
