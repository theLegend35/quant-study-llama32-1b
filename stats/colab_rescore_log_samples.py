# ==== Re-score with per-example outputs (for the paired bootstrap) ====
# Paste into a new cell AFTER the cells that define CONFIG, DRIVE, load_base, make_lm,
# load_tokenizer, eval_jobs, is_done, TASK_METRIC (i.e. run the notebook top-down, skip training/eval).
# Re-runs lm-eval on all 12 models (3 no-LoRA + 9 adapters) with log_samples=True and saves ONLY
# compact per-question correctness to <drive_root>/eval_samples/<name>.json.gz (~100 KB each).
# Resumable: finished models are skipped. Also checks the re-scored accuracy matches eval/<name>.json.
# Time: ~16-18 min per model on a T4 (same as the original evals) -> ~3.5-4 h for all 12.
import gzip, json, time
import lm_eval
from peft import PeftModel

SAMPLE_DIR = DRIVE / "eval_samples"; SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
tok = load_tokenizer()
for name, base, adir in eval_jobs():
    dest = SAMPLE_DIR / f"{name}.json.gz"
    if dest.exists():
        print(f"[skip] {name}"); continue
    if adir is not None and not is_done(adir):
        print(f"[wait] {name}: adapter missing"); continue
    print(f"[rescore] {name} ...", flush=True); t0 = time.time()
    model, _ = load_base(base)
    if adir is not None:
        model = PeftModel.from_pretrained(model, str(adir), is_trainable=False); model.eval()
    lm = make_lm(model, tok)
    out = lm_eval.simple_evaluate(model=lm, tasks=CONFIG["eval_tasks"], num_fewshot=CONFIG["eval_num_fewshot"],
                                  limit=CONFIG["eval_limit"], log_samples=True, bootstrap_iters=0)
    per = {}
    for task, metric in TASK_METRIC.items():
        rows = sorted(out["samples"][task], key=lambda s: s["doc_id"])
        per[task] = {"metric": metric, "doc_id": [int(s["doc_id"]) for s in rows],
                     "correct": [int(round(float(s[metric]))) for s in rows]}
        acc = sum(per[task]["correct"]) / len(rows)
        ref = json.load(open(EVAL_DIR / f"{name}.json"))["metrics"][task]["value"]
        flag = "OK" if abs(acc - ref) < 1e-6 else f"MISMATCH (saved {ref:.4f})"
        print(f"   {task:13s} n={len(rows):5d} acc={acc:.4f} {flag}")
    with gzip.open(dest, "wt") as f:
        json.dump({"name": name, "base": base, "adapter": str(adir) if adir else None,
                   "lm_eval_version": lm_eval.__version__, "per_task": per}, f)
    del lm, model; free_memory()
    print(f"[done] {name} in {(time.time()-t0)/60:.1f} min", flush=True)
print("All done. Download the eval_samples folder (or it is already on Drive).")
