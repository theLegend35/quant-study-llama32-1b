#!/usr/bin/env python3
"""Add Phase 1 paired significance tests (Appendix B + §5.1).

    python3 /workspace/paper/tools/patch_p1paired.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_p1paired.py. Numbers are from
phase1/stats/p1_paired_results.json (laptop OpenVINO re-run, batch 1,
seed 42, --log_samples), which reproduces Tables 5 and 6 exactly.
"""
from pathlib import Path

P = Path("/workspace/paper/build_paper.py")
src = P.read_text()

def replace_span(start, end_inclusive, new):
    global src
    a = src.find(start)
    b = src.find(end_inclusive, a)
    assert a >= 0 and b >= 0 and src.count(start) == 1, (
        a, b, start[:90], src.count(start)
    )
    b = b + len(end_inclusive)
    src = src[:a] + new + src[b:]

P1_APP_MAIN = [['Comparison', 'Diff.', '95% CI', 'p (boot.)', 'Holm p (boot.)', 'McNemar: first-only / second-only questions', 'McNemar p', 'Holm p (McNemar)'], ['RTN INT8 vs FP16', '−0.42', '[−0.69, −0.15]', '0.003', '0.006', '143/196', '0.005', '0.005'], ['RTN INT4 vs FP16', '−4.19', '[−4.89, −3.49]', '< 0.001', '< 0.01', '881/1795', '< 10⁻⁷⁰', '< 10⁻⁶⁹'], ['GPTQ INT4 vs FP16', '−5.01', '[−5.75, −4.29]', '< 0.001', '< 0.01', '921/2101', '< 10⁻¹⁰⁴', '< 10⁻¹⁰³'], ['AWQ INT4 vs FP16', '−2.34', '[−2.99, −1.70]', '< 0.001', '< 0.01', '800/1320', '< 10⁻²⁹', '< 10⁻²⁸'], ['AWQ INT4 vs RTN INT4', '+1.85', '[+1.29, +2.41]', '< 0.001', '< 0.01', '1060/666', '< 10⁻²⁰', '< 10⁻²⁰'], ['GPTQ INT4 vs RTN INT4', '−0.82', '[−1.57, −0.10]', '0.03', '0.03', '1271/1537', '< 10⁻⁶', '< 10⁻⁵'], ['AWQ INT4 vs GPTQ INT4', '+2.67', '[+1.94, +3.42]', '< 0.001', '< 0.01', '1747/1087', '< 10⁻³⁴', '< 10⁻³⁴']]
P1_APP_TASK = [['Comparison', 'Task', 'Diff.', '95% CI', 'p (boot.)', 'McNemar p'], ['RTN INT8 vs FP16', 'ARC-C', '−0.43', '[−1.28, +0.43]', '0.38', '0.44'], [' ', 'ARC-E', '+0.00', '[−0.55, +0.55]', '1.00', '1.00'], [' ', 'HellaSwag', '−0.02', '[−0.21, +0.16]', '0.88', '0.92'], [' ', 'WinoGrande', '−0.79', '[−1.82, +0.24]', '0.14', '0.16'], [' ', 'BoolQ', '−0.86', '[−1.50, −0.21]', '0.009', '0.01'], [' ', 'PIQA', '−0.44', '[−0.92, +0.05]', '0.10', '0.13'], ['RTN INT4 vs FP16', 'ARC-C', '−4.35', '[−6.48, −2.30]', '< 0.001', '< 10⁻⁴'], [' ', 'ARC-E', '−4.50', '[−5.93, −3.07]', '< 0.001', '< 10⁻⁹'], [' ', 'HellaSwag', '−4.33', '[−4.90, −3.76]', '< 0.001', '< 10⁻⁵²'], [' ', 'WinoGrande', '−2.21', '[−4.50, +0.08]', '0.07', '0.07'], [' ', 'BoolQ', '−7.95', '[−9.82, −6.08]', '< 0.001', '< 10⁻¹⁶'], [' ', 'PIQA', '−1.80', '[−3.16, −0.44]', '0.01', '0.01'], ['GPTQ INT4 vs FP16', 'ARC-C', '−3.41', '[−5.63, −1.28]', '0.003', '0.003'], [' ', 'ARC-E', '−5.35', '[−6.82, −3.87]', '< 0.001', '< 10⁻¹¹'], [' ', 'HellaSwag', '−6.10', '[−6.71, −5.51]', '< 0.001', '< 10⁻⁸⁸'], [' ', 'WinoGrande', '−2.53', '[−5.05, −0.08]', '0.04', '0.05'], [' ', 'BoolQ', '−9.42', '[−11.35, −7.49]', '< 0.001', '< 10⁻²⁰'], [' ', 'PIQA', '−3.26', '[−4.68, −1.85]', '< 0.001', '< 10⁻⁵'], ['AWQ INT4 vs FP16', 'ARC-C', '−2.82', '[−4.78, −0.94]', '0.006', '0.006'], [' ', 'ARC-E', '−2.02', '[−3.37, −0.67]', '0.004', '0.003'], [' ', 'HellaSwag', '−2.60', '[−3.09, −2.11]', '< 0.001', '< 10⁻²⁴'], [' ', 'WinoGrande', '−0.87', '[−3.08, +1.26]', '0.44', '0.48'], [' ', 'BoolQ', '−4.28', '[−5.93, −2.63]', '< 0.001', '< 10⁻⁶'], [' ', 'PIQA', '−1.47', '[−2.67, −0.27]', '0.02', '0.02'], ['AWQ INT4 vs RTN INT4', 'ARC-C', '+1.54', '[−0.26, +3.24]', '0.10', '0.11'], [' ', 'ARC-E', '+2.48', '[+1.35, +3.66]', '0.001', '< 10⁻⁴'], [' ', 'HellaSwag', '+1.73', '[+1.27, +2.20]', '< 0.001', '< 10⁻¹²'], [' ', 'WinoGrande', '+1.34', '[−0.47, +3.16]', '0.16', '0.17'], [' ', 'BoolQ', '+3.67', '[+2.20, +5.11]', '< 0.001', '< 10⁻⁵'], [' ', 'PIQA', '+0.33', '[−0.76, +1.41]', '0.59', '0.63'], ['GPTQ INT4 vs RTN INT4', 'ARC-C', '+0.94', '[−1.19, +3.16]', '0.42', '0.44'], [' ', 'ARC-E', '−0.84', '[−2.31, +0.63]', '0.28', '0.29'], [' ', 'HellaSwag', '−1.77', '[−2.35, −1.19]', '< 0.001', '< 10⁻⁸'], [' ', 'WinoGrande', '−0.32', '[−2.84, +2.21]', '0.82', '0.86'], [' ', 'BoolQ', '−1.47', '[−3.36, +0.43]', '0.13', '0.14'], [' ', 'PIQA', '−1.47', '[−2.88, −0.05]', '0.04', '0.05'], ['AWQ INT4 vs GPTQ INT4', 'ARC-C', '+0.60', '[−1.71, +2.82]', '0.63', '0.65'], [' ', 'ARC-E', '+3.32', '[+1.81, +4.88]', '< 0.001', '< 10⁻⁴'], [' ', 'HellaSwag', '+3.51', '[+2.91, +4.09]', '< 0.001', '< 10⁻³⁰'], [' ', 'WinoGrande', '+1.66', '[−0.95, +4.34]', '0.23', '0.24'], [' ', 'BoolQ', '+5.14', '[+3.36, +6.94]', '< 0.001', '< 10⁻⁷'], [' ', 'PIQA', '+1.80', '[+0.44, +3.21]', '0.02', '0.02']]

# ----- table literals, next to the Phase 1 SE rows -----
replace_span(
    "P1_SE_ROWS=",
    "P1_SE_ROWS=",
    "P1_APP_MAIN=" + repr(P1_APP_MAIN) + "\n"
    "P1_APP_TASK=" + repr(P1_APP_TASK) + "\n"
    "P1_SE_ROWS=",
)

# ----- §4.5 -----
replace_span(
    "Phase 1 per-question outputs were not saved, so Phase 1 differences are interpreted against the per-task SEs only.",
    "Phase 1 per-question outputs were not saved, so Phase 1 differences are interpreted against the per-task SEs only.",
    "Phase 1 differences are tested with the same paired bootstrap and exact "
    "McNemar procedure, on a later laptop re-run (batch size 1, seed 42) that "
    "reproduced Tables 5 and 6 exactly (Appendix B). The Holm family for Phase 1 "
    "is the four methods versus FP16 and the three INT4-method contrasts "
    "(AWQ vs RTN INT4, GPTQ vs RTN INT4, and AWQ vs GPTQ).",
)

# ----- §5.1: one/two sentences after Table 6 -----
replace_span(
    "PIQA changes little (at most 3.3 points).",
    "PIQA changes little (at most 3.3 points).",
    "PIQA changes little (at most 3.3 points).\")\n"
    "para(\""
    "A later re-run of the same five models with per-question logs (OpenVINO, "
    "batch size 1, seed 42) reproduced Tables 5 and 6 exactly. On the six-task "
    "average (Appendix B), RTN INT8 is 0.42 points below FP16 (paired 95% CI "
    "\u22120.69 to \u22120.15, Holm p = 0.006), a gap that is not significant on the "
    "four Table 5 tasks alone (\u22120.31, 95% CI \u22120.67 to +0.05, p = 0.09) and "
    "is driven mainly by BoolQ; all three INT4 methods are significantly below "
    "FP16, AWQ is significantly above both RTN INT4 (+1.85 points) and GPTQ INT4 "
    "(+2.67), and GPTQ is 0.82 points below RTN INT4 (95% CI \u22121.57 to \u22120.10, "
    "Holm p = 0.03).",
)

# ----- Figure 3 paragraph -----
replace_span(
    "INT8 moves the model to substantially lower memory and higher throughput at no detectable accuracy cost.",
    "INT8 moves the model to substantially lower memory and higher throughput at no detectable accuracy cost.",
    "INT8 moves the model to substantially lower memory and higher throughput at "
    "little accuracy cost (0.3 points on the four Table 5 tasks, not significant "
    "in the paired test; 0.42 points and significant on the six-task average).",
)

# ----- §5.7 -----
replace_span(
    "Phase 1 per-question outputs were not saved, so Tables 5 and 6 still have no paired test.",
    "Phase 1 per-question outputs were not saved, so Tables 5 and 6 still have no paired test.",
    "The same tests for the Phase 1 laptop models, from a later re-run that "
    "reproduced Tables 5 and 6, are in Appendix B.",
)

# ----- Appendix B heading -----
replace_span(
    'H2("Appendix B. Paired significance tests (Phase 2) and Phase 1 per-task standard errors")',
    'H2("Appendix B. Paired significance tests (Phase 2) and Phase 1 per-task standard errors")',
    'H2("Appendix B. Paired significance tests (Phase 2 and Phase 1) and Phase 1 per-task standard errors")',
)

# ----- Appendix B: drop "cannot be run", add Phase 1 paired section after the QA table -----
replace_span(
    " Phase 1 per-question logs were not saved, so a paired test like Table 14 cannot be run for Phase 1.",
    " Phase 1 per-question logs were not saved, so a paired test like Table 14 cannot be run for Phase 1.",
    "",
)

replace_span(
    'These columns match Tables 6 and 8 (standard-protocol GPTQ is shown here to two decimals, 423.66, and as 423.7 in Table 8).",fs=7.5)',
    'These columns match Tables 6 and 8 (standard-protocol GPTQ is shown here to two decimals, 423.66, and as 423.7 in Table 8).",fs=7.5)',
    'These columns match Tables 6 and 8 (standard-protocol GPTQ is shown here to two decimals, 423.66, and as 423.7 in Table 8).",fs=7.5)\n'
    'para("'
    "**Phase 1 paired tests.** The original laptop evaluations did not keep "
    "per-question logs. The five Phase 1 models were later re-run on the same "
    "laptop with lm-evaluation-harness 0.4.13, the OpenVINO backend, batch size 1 "
    "and seed 42, and with log_samples enabled. The per-question metrics "
    "(acc_norm for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; acc for "
    "WinoGrande and BoolQ) reproduce the accuracies in Tables 5 and 6 exactly "
    "(19,965 questions; document IDs were identical and in the same order for "
    "all five models). The tests below use the same stratified paired bootstrap "
    "(B = 10,000, seed 20261119) and exact McNemar procedure as Phase 2. The "
    "statistic is the six-task average, not the four-task average of Table 5. "
    "The Holm correction is applied to all seven comparisons: RTN INT8, RTN INT4, "
    "GPTQ INT4 and AWQ INT4 each versus FP16, and the three INT4-method contrasts "
    "AWQ vs RTN INT4, GPTQ vs RTN INT4, and AWQ vs GPTQ. No comparison is left "
    "outside the family. Per-task tests are uncorrected. Code: "
    "phase1/stats/p1_paired.py; results: phase1/stats/p1_paired_results.json.\")\n"
    "para(\""
    "On the four Table 5 tasks alone, the RTN INT8 versus FP16 interval includes "
    "zero (\u22120.31 points, 95% CI \u22120.67 to +0.05, p = 0.09), and GPTQ versus "
    "RTN INT4 is also not significant (\u22120.50, \u22121.44 to +0.43, p = 0.29). The "
    "six-task average, which adds BoolQ and PIQA, makes both contrasts "
    "significant; the INT8 gap is driven mainly by BoolQ (\u22120.86 points, p = 0.009), "
    "where 73 questions are unique errors for INT8 and 45 the other way, while "
    "ARC-Easy is tied at 1,472/2,376 with 21 cancelling flips. The GPTQ versus "
    "RTN gap is concentrated in HellaSwag (\u22121.77 points) together with smaller "
    "losses on BoolQ and PIQA. These paired tests do not isolate a mechanism for "
    "the HellaSwag drop; they are consistent with GPTQ already being the weakest "
    "INT4 method in Tables 5 and 6.\")\n"
    "table(P1_APP_MAIN,[1.45,0.45,0.95,0.5,0.5,1.5,0.6,0.65],cap=\""
    "Paired significance tests for Phase 1 (laptop, OpenVINO, six-task average "
    "accuracy, percentage points; first model minus second), on the later "
    "per-question re-run that reproduces Tables 5 and 6. 95% CI and p from a "
    "stratified paired bootstrap (10,000 resamples; smallest resolvable p "
    "\u2248 0.0002). Holm correction over all seven comparisons; values marked "
    "< 0.01 are bounds limited by bootstrap resolution. McNemar: exact test "
    "pooled over all 19,965 questions (question-weighted, so HellaSwag dominates). "
    "Source: phase1/stats/p1_paired_results.json.\",fs=7)\n"
    "table(P1_APP_TASK,[1.55,0.8,0.55,1.15,0.65,0.75],cap=\""
    "Per-task paired results for Phase 1 (percentage points; first model minus "
    "second). Bootstrap CIs and p-values are per task and not corrected for "
    "multiple comparisons. Source: phase1/stats/p1_paired_results.json.\",fs=7)",
)

# ----- limitations -----
replace_span(
    "Phase 1 per-question outputs were not saved, so Phase 1 conclusions rely on per-task SEs and consistency across tasks. The Holm correction covers only the seven main Phase 2 comparisons; per-task and Phase 1 comparisons are uncorrected, so some apparent differences there may arise by chance.",
    "Phase 1 per-question outputs were not saved, so Phase 1 conclusions rely on per-task SEs and consistency across tasks. The Holm correction covers only the seven main Phase 2 comparisons; per-task and Phase 1 comparisons are uncorrected, so some apparent differences there may arise by chance.",
    "Phase 1 differences are tested with the same paired method, on a later "
    "per-question re-run that reproduced Tables 5 and 6 (Appendix B). The Holm "
    "correction is applied separately to the seven main Phase 2 comparisons and "
    "to the seven Phase 1 comparisons; per-task tests are uncorrected, so some "
    "apparent per-task differences may arise by chance.",
)

# ----- H1 status -----
replace_span(
    '["H1","Supported. All per-task differences are within about one SE; four-task average \u22120.003."]',
    '["H1","Supported. All per-task differences are within about one SE; four-task average \u22120.003."]',
    '["H1","Mostly supported on the four Table 5 tasks (paired 95% CI includes 0; average \u22120.003). On the six-task average the 0.42-point loss is significant (Holm p = 0.006), driven mainly by BoolQ."]',
)

# ----- practical guidance -----
replace_span(
    "INT8 is a low-risk choice that yields 1.7\u00d7 decode throughput and a third less peak memory at no detectable accuracy cost.",
    "INT8 is a low-risk choice that yields 1.7\u00d7 decode throughput and a third less peak memory at no detectable accuracy cost.",
    "INT8 is a low-risk choice that yields 1.7\u00d7 decode throughput and a third less peak memory at little accuracy cost (0.42 points on the six-task average; not significant on the four Table 5 tasks).",
)

P.write_text(src)
print("patched", P, "chars", len(src))

# catch a duplicated four-task-average sentence if the first replace stacked it
if src.count("The four-task average is copied from Table 5; the result files do not include an SE for the average.") > 1:
    print("WARNING: duplicated four-task average sentence; count=",
          src.count("The four-task average is copied from Table 5; the result files do not include an SE for the average."))

for s in [
    "cannot be run for Phase 1",
    "still have no paired test",
    "Phase 1 conclusions rely on per-task SEs",
    "interpreted against the per-task SEs only",
    "at no detectable accuracy cost",
]:
    print("remaining", s, src.count(s))
assert "P1_APP_MAIN=" in src and "P1_APP_TASK=" in src
assert "Holm p = 0.006" in src
assert "phase1/stats/p1_paired.py" in src
print("sanity ok")
