"""Fill Phase 1 (laptop) software-version blanks in Table 4 and related wording.

Apply to the current (already bscheck-patched) build_paper.py:
    python3 /workspace/paper/tools/patch_versions.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_versions.py. Exact one-occurrence replacements.

Laptop versions are from a 2026-11-21 probe in the Phase 1 venv
(torch/transformers/openvino/nncf/lm_eval/datasets/numpy), plus the Windows
build string. optimum.__version__ was not reported by that probe; the table
keeps optimum 2.3.0 / optimum-intel 2.2.0 as previously recorded from the
laptop inspection (lora/README.md and Section 4.1). The Arc GPU driver was
not recorded and is left as such, not invented.
"""
P = "/workspace/paper/build_paper.py"
src = open(P).read()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:160])
    src = src.replace(old, new)

# ---------------- 4.1 machine line: Dell Inspiron + Windows 11 build ----------------
rep(
    "Compression and all efficiency measurements ran on a single laptop: Intel Core Ultra 7 155H, integrated Intel Arc GPU sharing 16 GB of system RAM, Windows, no discrete NVIDIA GPU.",
    "Compression and all efficiency measurements ran on a single Dell Inspiron laptop: Intel Core Ultra 7 155H, integrated Intel Arc GPU sharing 16 GB of system RAM, Windows 11 (build 10.0.26300), no discrete NVIDIA GPU.",
)

# ---------------- 4.6: transformers version note ----------------
# The paper already says Phase 2 accuracy is compared only within Colab, not
# with Phase 1 averages, but also reports a no-LoRA Colab-vs-laptop agreement
# check (within 0.6 points). The sentence below matches that, rather than
# claiming there is never any cross-phase comparison.
rep(
    "We will release scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs [[TBD: repository link]]. Table 4 records versions and fixed settings.",
    "We will release scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs [[TBD: repository link]]. Table 4 records versions and fixed settings. Phase 1 (laptop) and Phase 2 (Colab) used different transformers versions (5.5.4 and 4.57.6 respectively). Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages; the only cross-phase accuracy statement is a check that the Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Sections 4.3 and 5.5).",
)

# ---------------- Table 4 ----------------
rep(
    """table([
 ["Component","Version / setting"],
 ["Python (laptop / Colab)","[TBD] / 3.13"],["OpenVINO / OpenVINO GenAI","2026.4.1 / 2026.4.1"],["NNCF","3.4.0"],
 ["optimum / optimum-intel / transformers","2.3.0 / 2.2.0 / [TBD]"],["lm-evaluation-harness (laptop / Colab)","[TBD] (OpenVINO backend) / 0.4.13 (Hugging Face backend)"],["PyTorch / transformers / PEFT (Colab)","2.14.1+cu130 / 4.57.6 / 0.21.2"],["Colab evaluation batch size","automatic (11 or 13; see Section 4.3)"],
 ["Intel Arc GPU driver / Windows build","[TBD] / [TBD]"],["Colab GPU","NVIDIA T4 (all 9 runs)"],
 ["LoRA","r = 16, α = 32, dropout 0.05; q/k/v/o projections; not merged"],
 ["Training","1 epoch; AdamW, lr 2 × 10⁻⁴, cosine, 3% warm-up; batch 4 × 4 accumulation = 16; max length 512 (longer dropped); FP16 autocast; gradient checkpointing"],
 ["Dolly subset; seeds","5,000 train + 200 held out, data seed 1234; training seeds 0, 1, 2"],
],[3.0,3.5],cap="Software versions and fixed settings ([TBD] entries to be completed).")""",
    """table([
 ["Component","Version / setting"],
 ["Python (laptop / Colab)","3.11.9 / 3.13"],
 ["PyTorch (laptop / Colab)","2.14.1+cpu / 2.14.1+cu130"],
 ["transformers (laptop / Colab)","5.5.4 / 4.57.6"],
 ["PEFT (Colab)","0.21.2"],
 ["OpenVINO / OpenVINO GenAI","2026.4.1 / 2026.4.1"],
 ["NNCF","3.4.0"],
 ["optimum / optimum-intel","2.3.0 / 2.2.0"],
 ["lm-evaluation-harness (laptop / Colab)","0.4.13 (OpenVINO backend) / 0.4.13 (Hugging Face backend)"],
 ["datasets / numpy (laptop)","5.0.1 / 2.4.6"],
 ["Colab evaluation batch size","automatic (11 or 13; see Section 4.3)"],
 ["Windows (laptop)","Windows 11, build 10.0.26300"],
 ["Intel Arc GPU driver","not recorded"],
 ["Colab GPU","NVIDIA T4 (all 9 runs)"],
 ["LoRA","r = 16, α = 32, dropout 0.05; q/k/v/o projections; not merged"],
 ["Training","1 epoch; AdamW, lr 2 × 10⁻⁴, cosine, 3% warm-up; batch 4 × 4 accumulation = 16; max length 512 (longer dropped); FP16 autocast; gradient checkpointing"],
 ["Dolly subset; seeds","5,000 train + 200 held out, data seed 1234; training seeds 0, 1, 2"],
],[3.0,3.5],cap="Software versions and fixed settings. Phase 1 ran on the laptop; Phase 2 training and accuracy evaluation ran on Colab. The Intel Arc GPU driver version was not recorded.")""",
)

open(P, "w").write(src)
print("patched")
