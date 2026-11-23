# Phase 1 paired significance tests

`p1_paired.py` runs a stratified paired bootstrap (B = 10,000, seed 20261119) and exact McNemar tests on the six-task average and per task, with Holm correction over the seven Phase 1 comparisons (RTN INT8, RTN INT4, GPTQ INT4, and AWQ INT4 each versus FP16, plus AWQ vs RTN INT4, GPTQ vs RTN INT4, and AWQ vs GPTQ). The method matches `stats/paired_bootstrap.py` from Phase 2.

`p1_paired_results.json` is the script output used in Appendix B (Tables 22 and 23). `p1_correctness.csv` is a compact per-question 0/1 table (19,965 questions: ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande, BoolQ, PIQA) aligned by `doc_id` across the five models. Metrics match the paper: `acc_norm` for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; `acc` for WinoGrande and BoolQ. These sample means reproduce Tables 5 and 6 exactly.

The raw lm-eval `--log_samples` logs from the laptop re-run (OpenVINO, batch size 1, seed 42, 22–23 November 2026) are not in this repository (they are about 46 MB zipped). If you have that zip or a directory of `<model>-full` and `<model>-qa` folders, you can regenerate the JSON and CSV with:

```
python p1_paired.py /path/to/p1_samples.zip p1_paired_results.json p1_correctness.csv
```
