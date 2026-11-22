"""Patch build_paper.py so the re-score mismatch cause is stated as confirmed
by a direct Colab re-run of INT4 no-LoRA at batch sizes 11 and 13.

Apply to the current (already paired-patched) build_paper.py:
    python3 /workspace/paper/tools/patch_bscheck.py
    python3 /workspace/paper/build_paper.py

This script is written against archive/build_paper_pre_bscheck.py (the source
after tools/patch_paired.py). It uses exact one-occurrence replacements.
"""
P = "/workspace/paper/build_paper.py"
src = open(P).read()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:120])
    src = src.replace(old, new)

# ---------------- data block: Table 19 rows ----------------
# INT4 no-LoRA confirmatory re-run (same Colab T4 session, same versions;
# 67,106 log-likelihood requests per run). Accuracies are as printed by the
# check cell (4 d.p.). "Flags = re-score" is a question-for-question match
# against the saved re-score per-question file.
P2_APP_BSCHECK = [
    ["Task", "Original (batch 11)", "Check, batch 11", "Check, batch 13", "Acc. = original?", "Flags = re-score?"],
    ["ARC-C", "0.3191", "0.3191", "0.3200", "yes", "yes"],
    ["ARC-E", "0.5749", "0.5749", "0.5741", "yes", "yes"],
    ["HellaSwag", "0.6000", "0.6000", "0.5996", "yes", "yes"],
    ["WinoGrande", "0.5991", "0.5991", "0.5983", "yes", "yes"],
    ["BoolQ", "0.5560", "0.5560", "0.5581", "yes", "yes"],
    ["PIQA", "0.7329", "0.7329", "0.7329", "yes", "yes"],
]
rep("P2_APP_RESCORE=", "P2_APP_BSCHECK=" + repr(P2_APP_BSCHECK) + "\nP2_APP_RESCORE=")

# ---------------- 4.3 Re-scoring paragraph ----------------
rep(
    "We have not yet re-run a quantized base at the original batch size to confirm this directly. ",
    "A later check, in the same Colab session and with the same library versions, confirmed this directly on the INT4 no-LoRA base (all six zero-shot tasks, 67,106 log-likelihood requests per run). Re-running at batch size 11 reproduced the original saved scores exactly on all six tasks. Re-running at batch size 13 reproduced the re-score question for question. PIQA had the same accuracy at both batch sizes, but the per-question correctness flags differed, so some flips cancelled out (Appendix Table 19). ",
)

# ---------------- 7 Limitations ----------------
rep(
    "Re-scoring the Phase 2 models showed that results depend slightly on the automatically chosen evaluation batch size (up to 7 questions per task; Section 4.3), which bounds how exactly Phase 2 scores can be reproduced unless the batch size is fixed.",
    "Re-scoring the Phase 2 models showed that results depend slightly on the automatically chosen evaluation batch size (up to 7 questions per task; Section 4.3). A later re-run of the INT4 no-LoRA base confirmed the cause: batch size 11 reproduced the original scores exactly, and batch size 13 reproduced the re-score question for question (Appendix Table 19). Future evaluations should fix the batch size. Separately, an accidental retrain of the FP16 LoRA adapter at seed 0 produced slightly different scores (for example, ARC-E 0.6322 rather than 0.6313), so LoRA training is not bit-for-bit reproducible across sessions, while no-LoRA evaluations at a fixed batch size are.",
)

# ---------------- Appendix B: re-score check + Table 19 ----------------
rep(
    'para("**Re-score check.** Table 18 compares each re-scored task accuracy with the first evaluation, as a net count of questions. The cause of the two non-zero rows (a change of the automatically chosen batch size) is discussed in Section 4.3. Code: lora/stats/paired_bootstrap.py; results: lora/stats/paired_results.json; mismatch evidence: lora/stats/mismatch/.")',
    'para("**Re-score check.** Table 18 compares each re-scored task accuracy with the first evaluation, as a net count of questions. The two non-zero rows are the quantized bases without LoRA, whose automatically chosen batch size changed from 11 to 13. A later check on the INT4 no-LoRA base confirmed the cause: at batch size 11 the original scores were reproduced exactly, and at batch size 13 the re-score was reproduced question for question (Table 19). Code: lora/stats/paired_bootstrap.py; results: lora/stats/paired_results.json; mismatch evidence: lora/stats/mismatch/.")',
)

rep(
    'table(P2_APP_RESCORE,[1.15,0.95,0.5,0.5,0.65,0.7,0.5,0.5,0.8],cap="Re-score check: net change in the number of correctly answered questions (re-score minus first evaluation) per task, and change in the six-task average. Batch size: automatically chosen evaluation batch size in the first evaluation and in the re-score, from the saved Colab logs (n/l: first-run log not saved; these seven models were first evaluated in a later session that ran no training). All 12 first and re-score runs used the same library versions and a T4 GPU.",fs=7)',
    'table(P2_APP_RESCORE,[1.15,0.95,0.5,0.5,0.65,0.7,0.5,0.5,0.8],cap="Re-score check: net change in the number of correctly answered questions (re-score minus first evaluation) per task, and change in the six-task average. Batch size: automatically chosen evaluation batch size in the first evaluation and in the re-score, from the saved Colab logs (n/l: first-run log not saved; these seven models were first evaluated in a later session that ran no training). All 12 first and re-score runs used the same library versions and a T4 GPU. A confirmatory re-run of the INT4 no-LoRA base at both batch sizes is in Table 19.",fs=7)\n'
    'table(P2_APP_BSCHECK,[0.9,1.15,1.05,1.05,1.15,1.2],cap="Confirmation of the evaluation-batch-size effect on the INT4 no-LoRA model. Both checks were run in the same Colab T4 session, with the same library versions as the first evaluation and the re-score (PyTorch 2.14.1+cu130, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13) and 67,106 log-likelihood requests per run. At batch size 11, every task score matched the original saved accuracy exactly. At batch size 13, per-question correctness was identical to the re-score file on all six tasks. PIQA had the same accuracy at both batch sizes, but the per-question flags at batch 11 differed from the re-score, so some flips cancelled.",fs=7)',
)

open(P, "w").write(src)
print("patched")
