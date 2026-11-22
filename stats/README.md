# Statistical tests

- `paired_bootstrap.py` — stratified paired bootstrap (B = 10,000) and exact McNemar tests used in the paper. The saved results are in `../results/paired_results.json`.

  ```bash
  python3 stats/paired_bootstrap.py results/eval_samples results/paired_results.json results/eval
  ```

  Needs `numpy` and `scipy`.

- `mismatch/` — scripts used to investigate why two no-LoRA re-scores differed from the first evaluation. `colab_check_batchsize.py` is a Colab cell (run after the notebook's setup/helper cells) that re-evaluated INT4 no-LoRA at batch sizes 11 and 13 and confirmed the cause. `collect_evidence.py` and `fq_determinism.py` still contain box paths (`/workspace/lora/...`) from the original analysis; the JSON results they produced are already in `../results/`.

- `colab_rescore_log_samples.py` — Colab helper used to re-score models with `log_samples` enabled. Run it after the notebook's setup cells.

- `archive/` — earlier drafts (`paired_bootstrap_v0_synthetic.py`, `interim_unpaired.py`), kept only for history. Do not use them for the paper numbers.
