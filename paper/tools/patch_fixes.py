#!/usr/bin/env python3
"""Fix Table 5 averages / Table 13 bookkeeping, and add the AWQ scale-folding note.

    python3 /workspace/paper/tools/patch_fixes.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_fixes.py.
"""
from pathlib import Path

P = Path("/workspace/paper/build_paper.py")
src = P.read_text()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:180])
    src = src.replace(old, new)

# ---- Table 5 stored averages (display) ----
rep(
    '"RTN INT4":(".3242",".5745",".6000",".5951",".524","730"),\n'
    ' "GPTQ INT4":(".3336",".5661",".5823",".5919",".519","730"),',
    '"RTN INT4":(".3242",".5745",".6000",".5951",".523","730"),\n'
    ' "GPTQ INT4":(".3336",".5661",".5823",".5919",".518","730"),',
)

# Deltas must come from unrounded means, rounded once, not from the already-rounded averages.
# Unrounded four-task means (mean of the four official *-full accuracies):
#   FP16 0.5619450515, RTN INT8 0.5588555428, RTN INT4 0.5234534165,
#   GPTQ INT4 0.5184748016, AWQ INT4 0.5411871062
# Deltas vs FP16, rounded once to 3 decimals: -0.003, -0.038, -0.043, -0.021
rep(
    '    a=ACC[m]; d=num(a[4])-num(ACC["FP16"][4])\n'
    '    rows.append([m,a[0],a[1],a[2],a[3],a[4],"—" if m=="FP16" else f"{d:+.3f}".replace("-","−"),a[5]])',
    '    a=ACC[m]; d=ACC_AVG_EXACT[m]-ACC_AVG_EXACT["FP16"]\n'
    '    rows.append([m,a[0],a[1],a[2],a[3],a[4],"—" if m=="FP16" else f"{d:+.3f}".replace("-","−"),a[5]])',
)

# Insert ACC_AVG_EXACT next to ACC
rep(
    'ACC={  # ARC-C acc_norm, ARC-E acc_norm, HellaSwag acc_norm, WinoGrande acc, average of the four, disk MB\n',
    'ACC_AVG_EXACT={  # unrounded mean of the four official *-full accuracies; Table 5 averages are these rounded once to 3 d.p.\n'
    ' "FP16":0.561945051500613,"RTN INT8":0.5588555427896876,"RTN INT4":0.5234534165132109,\n'
    ' "GPTQ INT4":0.5184748016161879,"AWQ INT4":0.5411871062315032}\n'
    'ACC={  # ARC-C acc_norm, ARC-E acc_norm, HellaSwag acc_norm, WinoGrande acc, average of the four, disk MB\n',
)

# ---- Table 13 P2_SPEED ----
rep(
    '"FP16":    {"off":(24.9,41.8,6253,5578),"on":(22.9,44.3,6287,5595)},',
    '"FP16":    {"off":(24.9,41.7,6252,5578),"on":(22.9,44.3,6287,5595)},  # on WS 5595 excludes trimmed r5; see Table 13 note',
)

# ---- Prose: Table 5 quotes ----
rep(
    "AWQ recovers about half of RTN INT4’s loss (0.541 vs 0.524), whereas NNCF GPTQ does not improve on RTN (0.519).",
    "AWQ recovers about half of RTN INT4’s loss (0.541 vs 0.523), whereas NNCF GPTQ does not improve on RTN (0.518).",
)
rep(
    "AWQ is above RTN INT4 on all four tasks and closes about 45% of RTN’s average gap to FP16 (1.7 of 3.8 points);",
    "AWQ is above RTN INT4 on all four tasks and closes about 46% of RTN’s average gap to FP16 (1.8 of 3.8 points);",
)

# ---- Table 13 / memory / TTFT claims ----
rep(
    "Time to first token rose by 2.5, 4.0 and 4.7 ms respectively.",
    "Time to first token rose by 2.6, 4.0 and 4.7 ms respectively.",
)
rep(
    "The adapter added little memory: 34 MB for FP16, 22 MB for INT8 and 11 MB for INT4 (peak commit). Peak working set showed the same pattern, rising by about 17, 23 and 19 MB (about 5,578 to 5,595; 3,590 to 3,613; 2,433 to 2,452).",
    "The adapter added little memory: 35 MB for FP16, 22 MB for INT8 and 11 MB for INT4 (peak commit). Peak working set showed the same pattern, rising by about 17, 23 and 19 MB (about 5,578 to 5,595; 3,590 to 3,613; 2,433 to 2,452). The FP16 + LoRA working-set mean of 5,595 MB excludes round 5 (5,313 MB): that round’s peak commit stayed at 6,287 MB, its throughput matched the other LoRA rounds, and system CPU was normal (9.6%), so the drop is working-set trimming by the OS, not a smaller model footprint. The pass-2 script recorded commit for this reason, after pass 1 had already shown the same trimming. Tok/s, TTFT and commit for round 5 are kept in the averages.",
)
rep(
    "at about half the peak memory (3,108 vs 6,253–6,287 MB).",
    "at about half the peak memory (3,108 vs 6,252–6,287 MB).",
)
rep(
    'Unmerged LoRA added only 11–34 MB of peak memory on the laptop but reduced decode throughput',
    'Unmerged LoRA added only 11–35 MB of peak memory on the laptop but reduced decode throughput',
)
rep(
    "The unmerged adapter costs almost no memory (11–34 MB) but some speed",
    "The unmerged adapter costs almost no memory (11–35 MB) but some speed",
)
rep(
    "the unmerged adapter added only 11–34 MB of peak memory but reduced decode throughput",
    "the unmerged adapter added only 11–35 MB of peak memory but reduced decode throughput",
)

# Table 13 caption note
rep(
    'Peak memory is peak commit (private bytes), not the working-set metric of Table 7.",fs=8.5)',
    'Peak memory is peak commit (private bytes), not the working-set metric of Table 7. The working-set figures in the text (5,578 to 5,595 MB for FP16) exclude FP16 + LoRA round 5 from the working-set mean only: that round’s working set was 5,313 MB while peak commit stayed at 6,287 MB, consistent with OS working-set trimming (Appendix G).",fs=8.5)',
)

# ---- §5.4 AWQ sentence ----
rep(
    "although we did not re-run the accuracy tasks on the patched model.\")",
    "although we did not re-run the accuracy tasks on the patched model. The same stored-weight comparison also shows large raw differences for AWQ on some MLP up and down matrices (maximum 4.97, on the same down_proj). Those are not a defect of the same kind: AWQ folds per-channel smoothing scales into those matrices, the comparison does not undo the scales, every attention and gate matrix matches RTN exactly, and AWQ\u2019s standard-protocol perplexity remains healthy (14.89). Details are in Appendix F.\")",
)

# ---- Appendix B copy of Table 5 averages ----
rep(
    ' ["RTN INT4","0.3242 (0.0137)","0.5745 (0.0101)","0.6000 (0.0049)","0.5951 (0.0138)","0.524"],\n'
    ' ["GPTQ INT4","0.3336 (0.0138)","0.5661 (0.0102)","0.5823 (0.0049)","0.5919 (0.0138)","0.519"],',
    ' ["RTN INT4","0.3242 (0.0137)","0.5745 (0.0101)","0.6000 (0.0049)","0.5951 (0.0138)","0.523"],\n'
    ' ["GPTQ INT4","0.3336 (0.0138)","0.5661 (0.0102)","0.5823 (0.0049)","0.5919 (0.0138)","0.518"],',
)
rep(
    "The four-task average is the value published in Table 5.",
    "The four-task average is the mean of the four unrounded accuracies, rounded once to three decimals, and matches Table 5.",
)

# ---- Appendix F: AWQ explanation ----
rep(
    'table(GPTQ_ABL_ROWS,[3.2,2.0,1.7],cap="GPTQ ablation perplexities. The standard-protocol column is lm-eval WikiText-2 word perplexity at max_length = 2048 (BOS on the first window only): default models from the ppl runs; patched, C4 and scale-estimation models from diag/FINDINGS.md. The last column is a shorter diagnostic on four 2048-token WikiText-2 passages (diag/se_bisect.json and se_extra.json). A dash means that measurement was not in the files.",fs=7.5)\n',
    'table(GPTQ_ABL_ROWS,[3.2,2.0,1.7],cap="GPTQ ablation perplexities. The standard-protocol column is lm-eval WikiText-2 word perplexity at max_length = 2048 (BOS on the first window only): default models from the ppl runs; patched, C4 and scale-estimation models from diag/FINDINGS.md. The last column is a shorter diagnostic on four 2048-token WikiText-2 passages (diag/se_bisect.json and se_extra.json). A dash means that measurement was not in the files.",fs=7.5)\n'
    'para("The same weight-error file also lists a large maximum absolute difference for AWQ on layers.1.mlp.down_proj (4.97) and on other MLP up and down matrices (20 of 32 have relative Frobenius error above 0.3; the largest is 4.42, on layers.0.mlp.up_proj). This is not a GPTQ-style defect. weight_err.py compares each MatMul’s dequantized stored weights with the original FP16 matrix; it does not undo AWQ’s per-channel smoothing scales, which NNCF folds into up_proj and down_proj (diag/FINDINGS.md). All 80 attention and gate-projection matrices have exactly the same error statistics as RTN. AWQ’s standard-protocol WikiText-2 word perplexity is 14.89, next to RTN’s 15.59 and far from the GPTQ collapse. We therefore treat those large differences as the scale transform, not as quantization error. The files on hand do not identify which input column holds the 4.97 entry.")\n',
)

# ---- Appendix G: no longer a discrepancy; state the corrected Table 13 and the WS note ----
rep(
    "The tok/s means match Table 13 to the reported one decimal. Three small bookkeeping differences are worth recording, and are not applied back to Table 13. First, Table 13 lists TTFT for FP16 without LoRA as 41.8 ms; the mean of the two kept per-round means is 41.7 ms (41.96 and 41.52). Second, Table 13\\u2019s peak commit for FP16 without LoRA is 6,253 MB; the mean of the two kept rounds is 6,252 MB (6,251 and 6,253). Third, Table 13\\u2019s peak working set for FP16 with LoRA is 5,595 MB, which is the mean of rounds 4 and 8 only (5,597.6 and 5,594.1 MB; the unrounded mean is 5,595.9 MB). Round 5\\u2019s working set was 5,313 MB (trimming) and was omitted from that average, while its tok/s, TTFT and commit were kept. Decode throughput falls by 8% (FP16), 15% (INT8) and 18% (INT4) when the unmerged adapter is attached, as in the main text.",
    "The tok/s, TTFT and peak-commit means match Table 13 (FP16 without LoRA: 24.9 tokens/s, 41.7 ms, 6,252 MB). The 5,595 MB working-set figure used in Section 5.6 is the mean of rounds 4 and 8 only (5,597.6 and 5,594.1 MB; unrounded mean 5,595.9 MB). Round 5 is kept in the tok/s, TTFT and commit averages but omitted from the working-set mean: its working set was 5,313 MB, while peak commit stayed at 6,287 MB (6,291 and 6,283 in the other two LoRA rounds), throughput matched round 4 (22.86 tokens/s) and system CPU was 9.6%. That pattern is working-set trimming by the OS, which the pass-2 script was written to detect (it added a commit counter after pass 1 had already shown the same drop, for example FP16 + LoRA round 2 at 3,218 MB). Including round 5 would pull the working-set mean down to 5,501 MB, which would understate the memory the process actually reserved. Decode throughput falls by 8% (FP16), 15% (INT8) and 18% (INT4) when the unmerged adapter is attached, as in the main text.",
)
rep(
    "For FP16 with LoRA, the working-set mean of all three kept rounds is 5,501 MB; 5,596 MB is the unrounded mean of rounds 4 and 8 (5,595.9 MB) rounded to the nearest megabyte. Table 13 reports 5,595 MB for that same pair of rounds.",
    "For FP16 with LoRA, the working-set mean of all three kept rounds is 5,501 MB; 5,595 MB (Section 5.6 and the Table 13 note) is the mean of rounds 4 and 8 only (unrounded 5,595.9 MB), after excluding the trimmed round-5 working set.",
)

P.write_text(src)
print("patched", P, "bytes", len(src))
