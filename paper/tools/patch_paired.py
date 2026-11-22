"""Replace the interim unpaired Phase 2 tests with the paired bootstrap + McNemar results
(lora/stats/paired_results.json), fill Appendix B, revise wording on ~1-point effects, and add the
re-score mismatch explanation (lora/stats/mismatch/mismatch_evidence.json)."""
import json
P="/workspace/paper/build_paper.py"
src=open(P).read()
def rep(old,new):
    global src
    n=src.count(old); assert n==1,(n,old[:90]); src=src.replace(old,new)
def repl_line(prefix,new):
    global src
    lines=src.split("\n"); idx=[i for i,l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx)==1,(prefix[:80],idx); lines[idx[0]]=new; src="\n".join(lines)

R=json.load(open("/workspace/lora/stats/paired_results.json"))
E=json.load(open("/workspace/lora/stats/mismatch/mismatch_evidence.json"))
M={r["name"]:r for r in R["main"]}
def f(x,d=2): return f"{x:+.{d}f}".replace("-","−")
def ci(r): return f"[{f(r['lo'])}, {f(r['hi'])}]"
def fp(p,floor=False):
    if p is None: return "—"
    if floor or p<0.001: return "< 0.001"
    return f"{p:.2f}" if p>=0.01 else f"{p:.3f}"
def fsci(p):
    if p>=0.01: return f"{p:.2f}"
    if p>=0.001: return f"{p:.3f}"
    e=int(f"{p:.0e}".split("e")[1]); return f"< 10^{e+1}".replace("^","").replace("10","10",1) if False else "< 10⁻"+"".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(c)] for c in str(-(e+1)))
def holm_b(r):
    if r["p_holm"] is None: return "—"
    return "< 0.01" if r["p_holm"]<0.01 else f"{r['p_holm']:.2f}"
NQ=sum(R["settings"]["n_questions"].values())

# ---------------- data block: Table 14 rows ----------------
rows=[["Comparison (six-task average)","Diff. (points)","95% CI (paired bootstrap)","p (boot.)","Holm p","McNemar p","Per-seed diffs"]]
for r in R["main"]:
    sd="—" if not r["seed_diffs"] else ", ".join(f(x) for x in r["seed_diffs"])
    name=r["name"].replace(" (secondary)","")+(" †" if r["p_holm"] is None else "")
    rows.append([name,f(r["diff"]),ci(r),fp(r["p"],r["p_at_floor"]),holm_b(r),fsci(r["mcnemar_p"]),sd])
i0=src.index("# Interim significance tests (lora/stats/interim_unpaired.py)")
i1=src.index("\n",src.index("P2_STATS_ROWS=",i0))
src=src[:i0]+("# Paired significance tests (lora/stats/paired_bootstrap.py -> lora/stats/paired_results.json): stratified paired bootstrap\n"
              "# (B=10,000) on the six-task average + exact McNemar (pooled over tasks; LoRA cells: max p over seeds). Re-scored per-question data.\n"
              "P2_STATS_ROWS="+repr(rows))+src[i1:]

# Appendix tables
TN={"arc_challenge":"ARC-C","arc_easy":"ARC-E","hellaswag":"HellaSwag","winogrande":"WinoGrande","boolq":"BoolQ","piqa":"PIQA"}
appA=[["Comparison","Diff.","95% CI","p (boot.)","Holm p (boot.)","McNemar: first-only / second-only questions (per seed)","McNemar p (max)","Holm p (McNemar)"]]
for r in R["main"]:
    disc="; ".join(f"{d['a_only']}/{d['b_only']}" for d in r["mcnemar_pooled"])
    hm="—" if r["mcnemar_p_holm"] is None else fsci(r["mcnemar_p_holm"])
    appA.append([r["name"].replace(" (secondary)"," †"),f(r["diff"]),ci(r),fp(r["p"],r["p_at_floor"]),holm_b(r),disc,fsci(r["mcnemar_p"]),hm])
appB=[["Comparison","Task","Diff.","95% CI","p (boot.)","McNemar p (max over seeds)"]]
for r in R["main"]:
    for t in R["settings"]["tasks"]:
        v=R["per_task"][r["name"]][t]; mp=max(d["p"] for d in R["mcnemar_per_task"][r["name"]][t])
        appB.append([r["name"].replace(" (secondary)"," †") if t=="arc_challenge" else " ",TN[t],f(v["diff"]),f"[{f(v['lo'])}, {f(v['hi'])}]",
                     "< 0.001" if v["p"]<=R["settings"]["p_floor"]+1e-12 else fp(v["p"]),fsci(mp)])
appC=[["Model","Batch size (first / re-score)","ARC-C","ARC-E","HellaSwag","WinoGrande","BoolQ","PIQA","Avg. change (points)"]]
order=[f"{b}-{k}" for b in ["fp16","int8-rtn","int4-rtn"] for k in ["nolora","seed0","seed1","seed2"]]
for n in order:
    v=E["models"][n]; bo="n/l" if v["bs_orig"] is None else str(v["bs_orig"])
    appC.append([n,f"{bo} / {v['bs_rescore']}"]+[("0" if v["net_questions"][t]==0 else f"{v['net_questions'][t]:+d}".replace("-","−")) for t in ["arc_challenge","arc_easy","hellaswag","winogrande","boolq","piqa"]]+["0" if v["exact"] else f(v["avg_change_points"],3)])
rep("P2_STATS_ROWS=", "P2_APP_MAIN="+repr(appA)+"\nP2_APP_TASK="+repr(appB)+"\nP2_APP_RESCORE="+repr(appC)+"\nP2_STATS_ROWS=")

m=lambda n:M[n]
i8,i4,lf,l8,l4,g8,g4,sec=[m(n) for n in ["RTN INT8 vs FP16 (no LoRA)","RTN INT4 vs FP16 (no LoRA)","LoRA effect, FP16","LoRA effect, RTN INT8","LoRA effect, RTN INT4","INT8+LoRA vs FP16+LoRA","INT4+LoRA vs FP16+LoRA","INT4+LoRA vs plain FP16 (secondary)"]]
bq=R["per_task"]["LoRA effect, RTN INT4"]["boolq"]; hs=R["per_task"]["INT4+LoRA vs FP16+LoRA"]["hellaswag"]
bq2=R["per_task"]["INT4+LoRA vs FP16+LoRA"]["boolq"]; bq2m=max(d["p"] for d in R["mcnemar_per_task"]["INT4+LoRA vs FP16+LoRA"]["boolq"])
hsm=max(d["p"] for d in R["mcnemar_per_task"]["INT4+LoRA vs FP16+LoRA"]["hellaswag"])
agree_lf=1-max(d["a_only"]+d["b_only"] for d in lf["mcnemar_pooled"])/NQ
agree_g4=1-max(d["a_only"]+d["b_only"] for d in g4["mcnemar_pooled"])/NQ

# ---------------- Abstract ----------------
rep("RTN INT8 is not significantly different from FP16 (60.35% vs 60.41%; conservative 95% CI for the difference −1.26 to +1.15 points).",
    f"RTN INT8 is not significantly different from FP16 (60.35% vs 60.41%; paired 95% CI for the difference {f(i8['lo'])} to {f(i8['hi'])} points).")
rep("The 4.04-point loss and INT4’s LoRA gain are significant after Holm correction; the remaining 1.10-point gap is consistent across seeds and tasks but not yet significant in our conservative interim test (p = 0.08), pending a paired test.",
    f"In paired tests over questions with Holm correction, the 4.04-point loss, the LoRA gains on all three bases and the remaining 1.10-point gap (95% CI {f(-g4['hi'])[1:]} to {f(-g4['lo'])[1:]} points) are all significant; INT8 vs FP16, with or without LoRA, is not.")

# ---------------- 4.3 Evaluation: re-score paragraph ----------------
v8=E["models"]["int8-rtn-nolora"]; v4=E["models"]["int4-rtn-nolora"]
rescore_para=("para(\"**Re-scoring for paired tests.** The first Colab evaluations saved only task-level scores. To run paired tests, all 12 Phase 2 models were evaluated a second time with per-question logging, using the same notebook, adapters, GPU type (T4) and library versions (PyTorch 2.14.1, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13; recorded in every result file). "
 "Ten of the 12 models reproduced every task score exactly. The two quantized bases without LoRA did not: per task they changed by at most 7 of 1,172–10,042 questions (at most 0.21 points, INT4 BoolQ), and their six-task averages changed by "
 f"{f(v8['avg_change_points'],3)} (INT8) and {f(v4['avg_change_points'],3)} (INT4) points (Appendix Table 18). "
 "The cause is the evaluation batch size. We let lm-evaluation-harness pick the largest batch that fits in free GPU memory, and that choice depends on the session. The saved Colab logs show batch 11 for all four models whose first-run logs survive. In the re-score, the batch stayed at 11 for the two FP16 models, which matched exactly, and became 13 for the two quantized no-LoRA bases, which are the only two that changed. "
 "A different batch size changes how requests are grouped and padded, and so the shapes of the FP16 matrix multiplications and their rounding. This can flip questions whose answer options have almost equal log-likelihoods. "
 "The quantized weights themselves are reproducible: the fake-quantization code is deterministic (bit-identical across CPU thread counts and repeated runs), and all six adapters on quantized bases, which load the same quantized weights, reproduced exactly. "
 "We have not yet re-run a quantized base at the original batch size to confirm this directly. "
 "Tables 10–13 keep the first-run scores. The paired tests (Table 14, Appendix B) use the re-scored per-question data, so their differences can differ from Table 10 by up to 0.016 points (e.g., the INT4 LoRA gain is "
 f"{f(l4['diff'])} rather than +3.99). The change only affects comparisons that involve the INT8 or INT4 base without LoRA, where it is under 3% of the half-width of the 95% CI, and it does not change any conclusion. Future runs should fix the evaluation batch size.\")")
rep('para("**Laptop speed and memory.** Speed and memory of the unmerged adapters',
    rescore_para+'\npara("**Laptop speed and memory.** Speed and memory of the unmerged adapters')

# ---------------- 4.5 statistics paragraph ----------------
old_start="The planned paired bootstrap over questions (10,000 resamples) could not yet be run because per-question outputs were not saved in either phase."
old_end="For Phase 1 we interpret differences against the per-task SEs only."
a=src.index(old_start); b=src.index(old_end)+len(old_end)
src=src[:a]+("For Phase 2 we test differences between models with paired tests over questions (Section 5.7; details in Appendix B), because every model answers the same questions. "
 "The primary test is a paired bootstrap of the difference in six-task average accuracy (the unweighted mean of the six task accuracies, as in Table 10) with 10,000 resamples. Each resample draws questions with replacement within each task, and the same questions are used for both models. "
 "For a LoRA cell, each question is scored as the share of the three seeds that answered it correctly. We report the percentile 95% CI and a two-sided bootstrap p-value, and we apply the Holm correction across the seven main Phase 2 comparisons. "
 "As a secondary check we run exact McNemar tests on the questions that exactly one of the two models answered correctly. These need one model per side, so for LoRA cells they are run once per seed and we report the largest of the three p-values. "
 "These tests treat the test questions as the random sample. They do not cover training randomness beyond our three seeds, so we also list per-seed differences. "
 "Phase 1 per-question outputs were not saved, so Phase 1 differences are interpreted against the per-task SEs only.")+src[b:]

# ---------------- 5.5 ----------------
rep("which is not significantly different from zero (conservative 95% CI −1.26 to +1.15, p = 0.93; Table 14). This interval does not rule out an INT8 loss of up to about one point; a paired test should narrow it. With LoRA the two are identical (61.46).",
    f"which is not significantly different from zero (paired 95% CI {f(i8['lo'])} to {f(i8['hi'])}, p = {i8['p']:.2f}; Table 14). The interval rules out an average INT8 loss larger than about 0.3 points on these tasks. With LoRA the two are identical (61.46; paired 95% CI {f(g8['lo'])} to {f(g8['hi'])}).")
rep("LoRA simply raises INT8 by the same amount as FP16 (+1.10 vs +1.05).",
    "LoRA simply raises INT8 by the same amount as FP16 (+1.10 vs +1.05), and both gains are significant in the paired test (Holm p < 0.01; McNemar p < 10⁻⁸ in every seed).")
rep("In the conservative interim test (Table 14), the 4.04-point loss and the +3.99-point INT4 LoRA gain are both significant after Holm correction (p < 0.001), but the remaining 1.10-point gap is not (95% CI −2.33 to +0.11; unadjusted p = 0.08, Holm p = 0.37). Its consistency across all three seeds (−0.82 to −1.29) and all six tasks suggests a real gap, but a paired test is needed to confirm it.",
    f"In the paired tests (Table 14), the loss, the INT4 LoRA gain and the remaining gap are all significant after Holm correction: {f(i4['diff'])} points (95% CI {f(i4['lo'])} to {f(i4['hi'])}), {f(l4['diff'])} ({f(l4['lo'])} to {f(l4['hi'])}) and {f(g4['diff'])} ({f(g4['lo'])} to {f(g4['hi'])}). The remaining gap is also significant by McNemar in each seed separately (p {fsci(g4['mcnemar_p'])}), and its direction is the same in all three seeds ({', '.join(f(x) for x in g4['seed_diffs'])}).")
rep("The BoolQ gain itself is far outside sampling error (conservative 95% CI +7.19 to +12.06 points), which shows the change is real but not what caused it.",
    f"The BoolQ gain itself is far outside sampling error (paired 95% CI {f(bq['lo'])} to {f(bq['hi'])} points on the re-scored data, where the INT4 base scores 55.81 rather than 55.60), which shows the change is real but not what caused it. "
    f"The remaining gap between INT4 + LoRA and FP16 + LoRA is clearest on HellaSwag ({f(hs['diff'])} points, 95% CI {f(hs['lo'])} to {f(hs['hi'])}; McNemar p {fsci(hsm)} in every seed). On BoolQ it is {f(bq2['diff'])} points (95% CI {f(bq2['lo'])} to {f(bq2['hi'])}, uncorrected bootstrap p = {bq2['p']:.2f}), but McNemar does not confirm it in every seed (largest p = {bq2m:.2f}). On the other four tasks it is negative but not individually significant (Appendix Table 17).")
rep("The seed spread does not include the sampling error of the test questions, which a paired bootstrap would add (not yet run).",
    "The seed spread does not include the sampling error of the test questions; that is covered by the paired tests in Section 5.7. For every comparison that is significant there, all three seeds show a difference in the same direction.")

# ---------------- 5.7 ----------------
rep('H2("5.7 Statistical tests (interim)")','H2("5.7 Statistical tests (paired)")')
T57=(f"Table 14 gives the paired tests for Phase 2 described in Section 4.5 (full results in Appendix B). They replace an earlier interim test that used only the task-level SEs saved by lm-evaluation-harness and ignored pairing. Pairing matters here because two models scored on the same questions mostly agree. For example, FP16 with and without LoRA are both right or both wrong on at least {100*agree_lf:.0f}% of the {NQ:,} questions, so the variance of their difference is much smaller than the sum of their separate variances. "
 f"As a result, three effects of about one point that the interim test could not resolve (Holm p = 0.37 for each) are now significant: the LoRA gain on FP16 ({f(lf['diff'])} points, 95% CI {f(lf['lo'])} to {f(lf['hi'])}), the LoRA gain on INT8 ({f(l8['diff'])}, {f(l8['lo'])} to {f(l8['hi'])}) and the remaining gap between INT4 + LoRA and FP16 + LoRA ({f(g4['diff'])}, {f(g4['lo'])} to {f(g4['hi'])}). "
 f"The two INT8-vs-FP16 comparisons remain non-significant, but their intervals are now narrow (half-widths 0.23 and 0.13 points), so they bound any INT8 effect rather than leaving it open. The large effects (INT4 loss and INT4 LoRA gain) were already significant and remain so. "
 f"Plain FP16 and INT4 + LoRA are not distinguishable ({f(sec['diff'])}, 95% CI {f(sec['lo'])} to {f(sec['hi'])}). "
 f"Two limits apply. First, these tests treat the test questions as the random sample and condition on our three trained adapters; the per-seed differences show that all three seeds agree in direction for every significant comparison, but three seeds cannot estimate training variance precisely. Second, 10,000 resamples cannot resolve bootstrap p-values below about 0.0002 (shown as < 0.001), so the Holm-adjusted values for these comparisons are upper bounds; the McNemar p-values are exact. Phase 1 per-question outputs were not saved, so Tables 5 and 6 still have no paired test.")
repl_line('para("Table 14 gives the interim tests for Phase 2', 'para('+repr(T57)+')')
repl_line('table(P2_STATS_ROWS,',
 'table(P2_STATS_ROWS,[1.6,0.5,1.0,0.6,0.5,0.75,1.15],cap="Paired significance tests for Phase 2 (Colab, six-task average accuracy, percentage points; first model minus second), on the re-scored per-question data. LoRA cells use per-question correctness averaged over seeds 0–2. 95% CI and p from a stratified paired bootstrap (10,000 resamples; smallest resolvable p ≈ 0.0002). Holm correction over the seven main comparisons; values marked < 0.01 are bounds limited by bootstrap resolution. McNemar: exact test pooled over all six tasks; for LoRA cells, the largest p of the three seeds. † Secondary comparison, not in the Holm family. Per-seed diffs: the same difference for each seed separately (the no-LoRA side is a single evaluation). Differences can differ from Table 10 by up to 0.02 points because of the re-score (Section 4.3).",fs=7.5)')

# ---------------- Discussion / H7a / limitations / conclusion ----------------
rep("The INT4 gain is significant in the interim test; the FP16 and INT8 gains and the remaining 1.10-point gap are not yet resolved (Table 14). GPTQ and AWQ pending.",
    "All three gains and the remaining 1.10-point gap are significant in the paired test (Holm p < 0.01; Table 14). GPTQ and AWQ pending.")
repl_line('para("**Statistics.** Per-question outputs were not saved',
 'para("**Statistics.** Phase 2 differences are tested with paired tests over questions (Table 14, Appendix B). These tests take the test questions as the random sample and condition on three trained adapters per base, so they do not fully capture training variance; a different recipe, dataset or more seeds could shift the ~1-point effects. Phase 1 per-question outputs were not saved, so Phase 1 conclusions rely on per-task SEs and consistency across tasks. The Holm correction covers only the seven main Phase 2 comparisons; per-task and Phase 1 comparisons are uncorrected, so some apparent differences there may arise by chance. Re-scoring the Phase 2 models showed that results depend slightly on the automatically chosen evaluation batch size (up to 7 questions per task; Section 4.3), which bounds how exactly Phase 2 scores can be reproduced unless the batch size is fixed.")')
rep("remained 1.10 points behind (consistent across seeds and tasks, though not yet significant in a conservative interim test), with a higher held-out loss.",
    f"remained 1.10 points behind (paired 95% CI {f(-g4['hi'])[1:]} to {f(-g4['lo'])[1:]} points; significant after Holm correction), with a higher held-out loss.")

# ---------------- Reproducibility ----------------
rep('["lm-evaluation-harness (laptop / Colab)","[TBD] (OpenVINO backend) / 0.4.x (Hugging Face backend)"],["PyTorch / PEFT (Colab)","[TBD] / [TBD]"],',
    '["lm-evaluation-harness (laptop / Colab)","[TBD] (OpenVINO backend) / 0.4.13 (Hugging Face backend)"],["PyTorch / transformers / PEFT (Colab)","2.14.1+cu130 / 4.57.6 / 0.21.2"],["Colab evaluation batch size","automatic (11 or 13; see Section 4.3)"],')

# ---------------- Appendix B ----------------
rep('H1("Appendix (placeholder)")','H1("Appendix")')
rep('"B. Per-task accuracies with standard errors and paired bootstrap intervals for all configurations [[TBD]]",',
    '"B. Paired significance tests for Phase 2 (below). Phase 1 per-task accuracies with SEs [[TBD]]",')
rep("# footer page numbers",
r'''H2("Appendix B. Paired significance tests (Phase 2)")
para("**Data.** All 12 Phase 2 models (three bases, each without LoRA and with the seed-0, 1 and 2 adapters) were re-scored on Colab with lm-evaluation-harness 0.4.13 and log_samples enabled, which records for every question whether the model was correct (acc_norm for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; acc for WinoGrande and BoolQ). The full test sets were used: 1,172, 2,376, 10,042, 1,267, 3,270 and 1,838 questions (19,965 in total). Question IDs were checked to be identical and in the same order for all 12 models.")
para("**Paired bootstrap.** For two cells A and B, let d be the per-question difference in correctness (for a LoRA cell, correctness is the mean over the three seeds, so d can take fractional values). The statistic is the six-task average of the per-task mean of d, in percentage points. Each of 10,000 resamples draws questions with replacement within each task (stratified), using the same draws for both cells and for every comparison (random seed 20261119). The 95% CI is the 2.5th–97.5th percentile of the resampled statistic. The two-sided p-value is 2 × min(P(d* ≤ 0), P(d* ≥ 0)) with add-one smoothing, so the smallest value is 2/10,001 ≈ 0.0002. The Holm step-down correction is applied across the seven main comparisons; the comparison of INT4 + LoRA with plain FP16 is secondary and not corrected.")
para("**McNemar.** For each comparison and task, and pooled over all six tasks, we count questions answered correctly by only the first model and by only the second, and run an exact two-sided binomial test on these discordant counts. Because McNemar needs a single model on each side, LoRA cells are tested once per seed (seed k with LoRA vs the same base without LoRA, or seed k vs seed k across bases), and we report the largest of the three p-values. The pooled test weights questions equally, so HellaSwag dominates it, while the bootstrap weights the six tasks equally; the two therefore answer slightly different questions and are reported side by side. Holm is applied to the pooled McNemar p-values as well.")
para("**Re-score check.** Table 18 compares each re-scored task accuracy with the first evaluation, as a net count of questions. The cause of the two non-zero rows (a change of the automatically chosen batch size) is discussed in Section 4.3. Code: lora/stats/paired_bootstrap.py; results: lora/stats/paired_results.json; mismatch evidence: lora/stats/mismatch/.")
table(P2_APP_MAIN,[1.45,0.45,0.95,0.5,0.55,1.45,0.6,0.6],cap="Full results for the eight Phase 2 comparisons (six-task average, percentage points; first model minus second). McNemar columns use all 19,965 questions pooled; for LoRA cells, the discordant counts are listed for seeds 0, 1 and 2, and the p-value is the largest of the three. † Secondary comparison, outside the Holm family.",fs=7)
table(P2_APP_TASK,[1.6,0.8,0.55,1.15,0.65,1.0],cap="Per-task paired results (percentage points; first model minus second). Bootstrap CIs and p-values are per task and not corrected for multiple comparisons. McNemar p is the largest over seeds for LoRA cells. † Secondary comparison.",fs=7)
table(P2_APP_RESCORE,[1.15,0.95,0.5,0.5,0.65,0.7,0.5,0.5,0.8],cap="Re-score check: net change in the number of correctly answered questions (re-score minus first evaluation) per task, and change in the six-task average. Batch size: automatically chosen evaluation batch size in the first evaluation and in the re-score, from the saved Colab logs (n/l: first-run log not saved; these seven models were first evaluated in a later session that ran no training). All 12 first and re-score runs used the same library versions and a T4 GPU.",fs=7)

# footer page numbers''')

open(P,"w").write(src); print("patched")
