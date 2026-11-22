# Results

- `results_table.csv`, `results_long.csv`, `training_summary.csv`, `results_table.md` — first-run Phase 2 task scores and training summaries (the paper's Tables 10–13 use these first-run scores).
- `eval/*.json` — per-model first-run lm-eval summaries.
- `eval_samples/*.json.gz` — per-question re-score files for all 12 Phase 2 cells (used by the paired tests).
- `paired_results.json` — paired bootstrap and McNemar results (Table 14 and Appendix B).
- `mismatch_evidence.json`, `fq_determinism.json` — evidence for the re-score mismatch and the check that the fake-quantizer is deterministic.
- `data_subset.json` — the fixed Dolly subset (data seed 1234; 5,000 train + 200 held out).
- `sanity_wikitext.json` — Colab WikiText sanity check.
- `train/<run>/` — per-seed training logs and configs. Adapter weights (`adapter_model.safetensors`) are **not** included.
