# A controlled benchmark of weight-only quantization and LoRA on Llama-3.2-1B

This repository holds the code, evaluation logs and paper draft for a **controlled benchmark** of weight-only quantization of [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), with and without LoRA. It is **not a new method**. The study compares FP16, RTN INT8 and RTN INT4, each with LoRA off and on, and reports accuracy, peak memory, latency and throughput. QLoRA already showed that LoRA on a frozen 4-bit base can recover much of the accuracy lost to quantization; this work checks that finding under matched settings on a 1B model, and also measures speed and memory on a consumer Intel laptop.

## The six-cell grid (Phase 2 accuracy)

All six cells use the same LoRA recipe when LoRA is on, and the same Colab evaluation when scores are compared.

| Base (fake-quant RTN, matching the laptop OpenVINO IRs) | LoRA off | LoRA on (r = 16, α = 32, q/k/v/o; 3 seeds) |
|---|---|---|
| FP16 | yes | yes |
| RTN INT8 (INT8_SYM, per channel; embedding included) | yes | yes |
| RTN INT4 (INT4_SYM, group size 128; embedding and lm_head INT8_ASYM) | yes | yes |

GPTQ and AWQ were compressed and evaluated in Phase 1 (laptop) but do not yet have LoRA runs.

## Key paired results (six-task average, percentage points)

Numbers are from `results/paired_results.json` only (stratified paired bootstrap, B = 10,000, over the re-scored per-question files). LoRA cells use the per-question share of three seeds correct. Holm correction is over the seven main comparisons. The last row is a secondary comparison and is not Holm-corrected. First-run task scores (paper Tables 10–13) are in `results/results_table.csv`.

| Comparison | Diff. (points) | 95% CI | p (bootstrap) | Holm p |
|---|---|---|---|---|
| RTN INT8 vs FP16 (no LoRA) | −0.06 | [−0.29, +0.17] | 0.60 | 1.00 |
| RTN INT4 vs FP16 (no LoRA) | −4.03 | [−4.73, −3.33] | < 0.001 | < 0.01 |
| LoRA effect, FP16 | +1.05 | [+0.64, +1.46] | < 0.001 | < 0.01 |
| LoRA effect, RTN INT8 | +1.11 | [+0.70, +1.52] | < 0.001 | < 0.01 |
| LoRA effect, RTN INT4 | +3.97 | [+3.37, +4.56] | < 0.001 | < 0.01 |
| INT8+LoRA vs FP16+LoRA | −0.01 | [−0.14, +0.13] | 0.92 | 1.00 |
| INT4+LoRA vs FP16+LoRA | −1.11 | [−1.63, −0.59] | < 0.001 | < 0.01 |
| INT4+LoRA vs plain FP16 (secondary) | −0.05 | [−0.66, +0.53] | 0.85 | — |

In short: INT8 is not significantly different from FP16, with or without LoRA. INT4 loses about 4 points without LoRA. LoRA recovers most of that loss, but INT4+LoRA remains about 1.11 points behind FP16+LoRA (95% CI 0.59 to 1.63 points).

## How to reproduce

### Phase 1 (laptop, OpenVINO / NNCF 3.4.0)

Phase 1 compressed Llama-3.2-1B with NNCF 3.4.0 (OpenVINO 2026.4.1, optimum-intel) into FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4, then measured zero-shot accuracy with lm-evaluation-harness 0.4.13 (OpenVINO backend) and speed/memory with OpenVINO GenAI on the Arc GPU (batch size 1).

The main laptop compression and evaluation scripts are **not in this repository yet**. They live at `C:\Users\dhruv\quant-study` and will be added by the author. What *is* here is under `phase1/`: IR inspection and fake-quant verification scripts, the later unmerged-LoRA speed scripts, and the speed JSON logs. See `phase1/README.md` and `requirements-laptop.txt`. You may replace `requirements-laptop.txt` with `pip freeze` output from the laptop venv.

### Phase 2 (Colab T4)

1. Open `phase2/lora_phase2.ipynb` in Colab and select a T4 GPU.
2. Accept the Llama 3.2 licence on the Hugging Face model page with **your** account. Add a Colab secret named `HF_TOKEN` (a Hugging Face read token) and enable notebook access. Do not paste a token into the notebook or this repository. If no secret is found, the notebook shows an interactive login box.
3. Install the Phase 2 libraries (`requirements-colab.txt`, or the notebook's install cell). The reported runs used PyTorch 2.14.1+cu130, transformers 4.57.6, PEFT 0.21.2 and lm-eval 0.4.13.
4. Run the cells from top to bottom. The notebook:
   - rebuilds frozen PyTorch copies of the FP16, RTN INT8 and RTN INT4 bases with fake quantization that matches NNCF 3.4.0 RTN (INT8_SYM per channel; INT4_SYM group size 128, embedding and lm_head INT8_ASYM);
   - trains LoRA (r = 16, α = 32, dropout 0.05, targets q/k/v/o, unmerged) for one epoch on a fixed 5,000-example Dolly-15k subset (data seed 1234; 200 held out), with training seeds 0, 1 and 2;
   - evaluates every cell zero-shot with lm-eval 0.4.13 on ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande, BoolQ and PIQA;
   - can re-score with per-question logs (Section 12) for the paired tests.
5. To recompute the paired tests locally:

   ```bash
   python3 stats/paired_bootstrap.py results/eval_samples results/paired_results.json results/eval
   ```

### Fix the evaluation batch size

The reported Phase 2 runs used lm-eval `eval_batch_size="auto"`. The automatically chosen batch depends on free GPU memory. A different batch changes padding and FP16 matmul rounding and can flip near-tie questions. This was confirmed on the INT4 no-LoRA base: batch size 11 reproduced the original saved scores exactly, and batch size 13 reproduced the re-score question for question (paper, Section 4.3 and Appendix Table 19). The effect was at most 7 questions per task and did not change any conclusion, but **future evaluations should fix the batch size** (for example 11 or 13) instead of using `"auto"`. The check cell is `stats/mismatch/colab_check_batchsize.py`.

## Hardware

- **Phase 1 (and laptop LoRA speed):** Dell Inspiron, Intel Core Ultra 7 155H, Intel Arc integrated GPU, 16 GB RAM, Windows 11 (build 10.0.26300). Arc GPU driver version was not recorded.
- **Phase 2 training and accuracy:** Google Colab, NVIDIA T4.
- Phase 1 used transformers 5.5.4; Phase 2 used transformers 4.57.6. Accuracy results are compared only within each phase, except for a check that Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task.

## Repository layout

```
phase1/       Laptop inspection, LoRA speed scripts, and speed logs
phase2/       Colab notebook (Section 12 re-score included; outputs stripped)
stats/        Paired bootstrap and mismatch / batch-size check scripts
results/      CSVs, eval JSONs, 12 per-question files, paired_results.json
paper/        Draft PDF/DOCX, build_paper.py, patch scripts, figures
```

## What is not included

- Llama 3.2 weights, OpenVINO IRs, and LoRA adapter weights (`*.safetensors`).
- The main Phase 1 laptop compression and evaluation scripts (see `phase1/README.md`).
- Intermediate Colab session notebooks with outputs.

## Licence

The code and paper source in this repository are released under the MIT Licence (see `LICENSE`).

**Llama 3.2 weights are not included.** They are distributed by Meta under a separate licence. You must accept that licence on the Hugging Face model page to download `meta-llama/Llama-3.2-1B`. Dolly-15k and the evaluation datasets have their own licences as well.
