# ==== One-cell Colab check: is the re-score mismatch caused by the auto batch size? (~35 min on a T4) ====
# Run after the notebook's setup/helper cells (same as the re-score cell). Nothing on Drive is overwritten.
# Prediction if batch size is the cause:
#   (1) int4-rtn-nolora at batch_size=11 reproduces the ORIGINAL eval/int4-rtn-nolora.json exactly
#       (0.3191 / 0.5749 / 0.6000 / 0.7329 / 0.5991 / 0.5560), not the re-score (0.3200 / 0.5741 / 0.5996 / ... / 0.5581).
#   (2) int4-rtn-nolora at batch_size=13 reproduces the re-score file eval_samples/int4-rtn-nolora.json.gz exactly.
import gzip, json, lm_eval
tok = load_tokenizer()
saved = json.load(open(EVAL_DIR / "int4-rtn-nolora.json"))["metrics"]
resc = json.load(gzip.open(DRIVE / "eval_samples" / "int4-rtn-nolora.json.gz", "rt"))["per_task"]
for bs in (11, 13):
    model, _ = load_base("int4-rtn")
    lm = make_lm(model, tok, batch_size=bs)
    out = lm_eval.simple_evaluate(model=lm, tasks=CONFIG["eval_tasks"], num_fewshot=0, limit=None, log_samples=True, bootstrap_iters=0)
    print(f"--- batch_size={bs}")
    for task, metric in TASK_METRIC.items():
        rows = sorted(out["samples"][task], key=lambda s: s["doc_id"])
        c = [int(round(float(s[metric]))) for s in rows]; acc = sum(c) / len(c)
        same_resc = c == resc[task]["correct"]
        print(f"   {task:13s} acc={acc:.4f}  saved={saved[task]['value']:.4f} {'MATCH' if abs(acc-saved[task]['value'])<1e-9 else 'diff'}"
              f"  | per-question identical to re-score: {same_resc}")
    del lm, model; free_memory()
