"""Add interim significance tests (Phase 2, unpaired/conservative from saved lm-eval SEs) and speed per-run spread.
Paired bootstrap is blocked: no per-example outputs were saved (Colab log_samples=False; no samples_*.jsonl on laptop)."""
import json
P="/workspace/paper/build_paper.py"
src=open(P).read()
def rep(old,new):
    global src
    n=src.count(old); assert n==1,(n,old[:80]); src=src.replace(old,new)

S=json.load(open("/workspace/lora/stats/interim_unpaired.json"))
def f(x): return f"{x:+.2f}".replace("-","−")
def fp(p):
    if p is None: return "—"
    return "< 0.001" if p<0.001 else f"{p:.2f}" if p>=0.01 else f"{p:.3f}"
rows=[["Comparison (six-task average)","Diff. (points)","95% CI (conservative)","p","Holm p","Per-seed diffs"]]
for r in S["main"]:
    sd="—" if not r["seed_diffs"] else ", ".join(f(x) for x in r["seed_diffs"])
    name=r["name"]+(" †" if r["p_holm"] is None else "")
    rows.append([name,f(r["diff"]),f"[{f(r['lo'])}, {f(r['hi'])}]",fp(r["p"]),fp(r["p_holm"]),sd])
pt=S["per_task"]["LoRA effect, RTN INT4"]["boolq"]; hs=S["per_task"]["INT4+LoRA vs FP16+LoRA"]["hellaswag"]

# ---- data block ----
rep('P2_SPEED_DELTA={"FP16":"−8%","RTN INT8":"−15%","RTN INT4":"−18%"}',
    'P2_SPEED_DELTA={"FP16":"−8%","RTN INT8":"−15%","RTN INT4":"−18%"}\n'
    '# Interim significance tests (lora/stats/interim_unpaired.py): unpaired normal approximation from lm-eval SEs.\n'
    'P2_STATS_ROWS='+repr(rows))

# ---- Methods: statistics paragraph ----
rep('The planned paired bootstrap over questions has not yet been run; until it is, we interpret differences against these unpaired SEs, which overstate the uncertainty of paired differences between models evaluated on the same questions, and we emphasize patterns that are consistent across tasks.',
    'The planned paired bootstrap over questions (10,000 resamples) could not yet be run because per-question outputs were not saved in either phase. For Phase 2 we therefore report an interim, conservative test (Section 5.7). For each comparison, the difference in six-task average accuracy (the unweighted mean of the six task accuracies, as in Table 10) is tested with a normal approximation. Its variance is the sum of both models’ squared per-task SEs divided by 36, treating tasks as independent. For a LoRA cell, which is the mean of three seeds evaluated on the same questions, we use the mean of the three per-seed SEs (an upper bound) plus the seed variance divided by three. Because this ignores the positive correlation between two models scored on the same questions, its intervals are wider and its p-values larger than a paired test would give. A non-significant result is therefore inconclusive, not evidence of equality. We apply the Holm correction across the seven main Phase 2 comparisons and also list per-seed differences. For Phase 1 we interpret differences against the per-task SEs only.')

# ---- Key results ----
rep('RTN INT8 shows no measurable loss (60.35% vs 60.41% for FP16).',
    'RTN INT8 is not significantly different from FP16 (60.35% vs 60.41%; conservative 95% CI for the difference −1.26 to +1.15 points).')
rep('there, the INT4 gap shrinks from 4.04 to 1.10 points (about 73% closed), and a held-out training-loss gap remains.',
    'there, the INT4 gap shrinks from 4.04 to 1.10 points (about 73% closed), and a held-out training-loss gap remains. The 4.04-point loss and INT4’s LoRA gain are significant after Holm correction; the remaining 1.10-point gap is consistent across seeds and tasks but not yet significant in our conservative interim test (p = 0.08), pending a paired test.')

# ---- 5.5 INT8 / INT4 paragraphs ----
rep('(60.35 vs 60.41), which is within noise. With LoRA the two are identical (61.46). Because there is no measurable gap,',
    '(60.35 vs 60.41), which is not significantly different from zero (conservative 95% CI −1.26 to +1.15, p = 0.93; Table 14). This interval does not rule out an INT8 loss of up to about one point; a paired test should narrow it. With LoRA the two are identical (61.46). Because no gap is detectable,')
rep('INT4 + LoRA is still below FP16 + LoRA on all six tasks (by 0.4 to 2.0 points).")',
    'INT4 + LoRA is still below FP16 + LoRA on all six tasks (by 0.4 to 2.0 points). In the conservative interim test (Table 14), the 4.04-point loss and the +3.99-point INT4 LoRA gain are both significant after Holm correction (p < 0.001), but the remaining 1.10-point gap is not (95% CI −2.33 to +0.11; unadjusted p = 0.08, Holm p = 0.37). Its consistency across all three seeds (−0.82 to −1.29) and all six tasks suggests a real gap, but a paired test is needed to confirm it.")')
rep('"not meaningful (no measurable gap)"','"not meaningful (no detectable gap)"')
rep('so the recovery is not limited to BoolQ.")',
    f'so the recovery is not limited to BoolQ. The BoolQ gain itself is far outside sampling error (conservative 95% CI {f(pt["lo"])} to {f(pt["hi"])} points), which shows the change is real but not what caused it.")')

# ---- 5.6 speed: per-run spread ----
rep('In every kept round the six cells kept the same speed order: INT4, INT4 + LoRA, INT8, INT8 + LoRA, FP16, FP16 + LoRA.")',
    'In every kept round the six cells kept the same speed order: INT4, INT4 + LoRA, INT8, INT8 + LoRA, FP16, FP16 + LoRA. Within a round, the standard deviation over the 30 timed runs was 0.1–0.3 tokens/s for FP16, 0.4–0.9 for INT8 and 0.9–2.0 for INT4, smaller than the round-to-round differences. With only two or three rounds per cell we do not run a formal test on throughput. Descriptively, every LoRA round was slower than every no-LoRA round of the same base (highest LoRA round vs lowest no-LoRA round: 23.06 vs 24.81, 36.11 vs 40.82 and 52.89 vs 60.75 tokens/s).")')

# ---- new 5.7 ----
rep('# ---------- 6 Discussion ----------',
r'''H2("5.7 Statistical tests (interim)")
para("Table 14 gives the interim tests for Phase 2 described in Section 4.5. They come from the per-task SEs saved by lm-evaluation-harness and ignore pairing, so they are conservative. Effects of about four points are clearly significant. Effects of about one point, including the 1.05-point LoRA gain on FP16 and the 1.10-point gap between INT4 + LoRA and FP16 + LoRA, are not resolved at this sensitivity, even though their direction is the same in every seed. A paired bootstrap over questions, with per-task McNemar tests, will replace this table once the models are re-scored with per-question logging. Phase 1 per-question outputs were not saved either, so the Phase 1 comparisons (Tables 5 and 6) have no paired test yet.")
table(P2_STATS_ROWS,[1.95,0.7,1.25,0.6,0.6,1.3],cap="Interim significance tests for Phase 2 (Colab, six-task average accuracy, percentage points; first model minus second). Unpaired normal approximation from lm-evaluation-harness per-task SEs; LoRA cells are three-seed means (SE: mean of per-seed SEs plus seed variance / 3). Conservative, because the positive correlation between models scored on the same questions is ignored. Holm correction over the seven main comparisons. † Secondary comparison, not in the Holm family. Per-seed diffs: the same difference computed for each seed separately (the no-LoRA side is the single deterministic evaluation).",fs=8)

# ---------- 6 Discussion ----------''')

# ---- discussion / H7a / limitations / conclusion ----
rep('INT8 had nothing to recover and gained the same as FP16.',
    'INT8 had no detectable loss to recover and gained the same as FP16.')
rep('and its gap to FP16 + LoRA fell from 4.04 to 1.10 points. GPTQ and AWQ pending."]',
    'and its gap to FP16 + LoRA fell from 4.04 to 1.10 points. The INT4 gain is significant in the interim test; the FP16 and INT8 gains and the remaining 1.10-point gap are not yet resolved (Table 14). GPTQ and AWQ pending."]')
rep('para("**Statistics.** Paired significance tests have not yet been run; conclusions rely on unpaired SEs and on consistency across tasks. Many comparisons are made, so some apparent differences may arise by chance.")',
    'para("**Statistics.** Per-question outputs were not saved, so the planned paired bootstrap has not been run. The interim Phase 2 tests (Table 14) are unpaired and conservative, and cannot resolve differences of about one point. Phase 1 conclusions rely on per-task SEs and consistency across tasks. The Holm correction covers only the seven main Phase 2 comparisons; per-task and Phase 1 comparisons are uncorrected, so some apparent differences there may arise by chance.")')
rep('INT8 had no measurable loss to recover,','INT8 showed no detectable loss to recover,')
rep('remained 1.10 points behind, with a higher held-out loss.',
    'remained 1.10 points behind (consistent across seeds and tasks, though not yet significant in a conservative interim test), with a higher held-out loss.')
open(P,"w").write(src); print("patched")
