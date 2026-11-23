# Phase 1 (laptop / OpenVINO)

The main Phase 1 compression, accuracy-evaluation and original speed scripts were run on the author's laptop and are **not in this repository yet**. They live at:

`C:\Users\dhruv\quant-study`

and will be added by the author (export/compression commands, the original `speed.py` protocol, OpenVINO IRs, and Phase 1 lm-eval logs).

What is included here, from copies already on the project box:

- `inspect_phase1.py` and `verify_phase1.py` — read-only checks of the laptop OpenVINO IRs (NNCF settings, and a bit-exact comparison of the notebook's fake-quantizer against the stored integer codes and scales). Point them at the laptop study root, e.g. `python inspect_phase1.py C:\Users\dhruv\quant-study`.
- `speed-lora/` — laptop scripts that measured unmerged LoRA adapters with OpenVINO GenAI on the Arc GPU (`speed_lora.py`, PowerShell drivers, and small reference checks). These expect models under `models/ov-*` and the seed-0 adapter under `lora-results/` on the laptop.
- `speed-results/` — raw JSON timing logs from the Phase 1 no-LoRA speed runs and the later unmerged-LoRA speed pass.
- `stats/` — paired bootstrap and McNemar tests on a later laptop re-run with per-question logs (`p1_paired.py`, `p1_paired_results.json`, and a compact per-question correctness CSV). The raw 46 MB sample-log zip is not in this repository.

Hardware used for Phase 1: Dell Inspiron, Intel Core Ultra 7 155H, Intel Arc integrated GPU, 16 GB RAM, Windows 11 (build 10.0.26300). The Arc GPU driver version was not recorded. See the top-level `requirements-laptop.txt` and the paper's Table 4.

## Laptop scripts and results (copied from C:\Users\dhruv\quant-study)

- `scripts/`: the run scripts (`run_full.ps1`, `run_ext.ps1`, `run_gptq_v2.ps1`, `run_pplsweep.ps1`, and `run_samples.ps1`, the Nov 2026 re-run with `--log_samples` used for the paired tests), the speed benchmarks (`speed.py`, `speed_lora.py`), and the sanity checks.
- `results/`: the lm-eval result JSONs for each OpenVINO model, plus the speed JSONs. The raw progress logs are left out.
- `diag/`: the INT4 GPTQ diagnosis scripts and notes (see `diag/FINDINGS.md`).
- `requirements-lock.txt`: the exact package versions in the laptop venv.

The models were exported with `optimum-cli export openvino` (see `scripts/run_gptq_v2.ps1` for an example). The OpenVINO model folders are not included.
