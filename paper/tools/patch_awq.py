#!/usr/bin/env python3
"""Confirm the AWQ max-abs 'error' as folded channel scales, from awq_scale_check.txt.

    python3 /workspace/paper/tools/patch_awq.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_awq.py. Numbers are from
/workspace/laptop/awq_scale_check.txt (dequantized from the ov-awq-int4 IR).
"""
from pathlib import Path

P = Path("/workspace/paper/build_paper.py")
src = P.read_text()

def replace_span(start, end_inclusive, new):
    global src
    a = src.find(start)
    b = src.find(end_inclusive, a)
    assert a >= 0 and b >= 0 and src.count(start) == 1, (a, b, start[:80], src.count(start))
    b = b + len(end_inclusive)
    src = src[:a] + new + src[b:]

# ----- §5.4 -----
replace_span(
    "The same stored-weight comparison also shows large raw differences for AWQ",
    "Details are in Appendix F.",
    "The same stored-weight comparison also shows large raw differences for AWQ "
    "on some MLP up and down matrices (maximum 4.97, on the same down_proj). "
    "A later check on the OpenVINO IR confirmed the cause: those differences "
    "are AWQ\u2019s per-channel smoothing scales, folded into down_proj input "
    "columns and up_proj output rows, not a quantization defect. The 4.97 "
    "entry is in input column 1417, not column 2558; dividing out the estimated "
    "scales brings the residual relative error back into the usual INT4 range, "
    "and AWQ\u2019s standard-protocol perplexity is 14.89. Details are in Appendix F.",
)

# ----- Appendix F: replace the hedge paragraph and add a results table -----
AWQ_TAB = [
    ["Matrix and site of max. raw difference", "Raw max-abs (rel. err.)", "Estimated scale (median AWQ/FP16 ratio)", "After undoing, max-abs (rel. err.)"],
    ["layers.1.mlp.down_proj, row 400, input col 1417", "4.97 (0.391)", "15.67 on that input column", "0.184 (0.156)"],
    ["layers.15.mlp.down_proj, row 1159, input col 7258", "1.52 (0.314)", "4.79 on that input column", "0.125 (0.140)"],
    ["layers.0.mlp.up_proj, row 348, input col 1224", "2.19 (4.42)", "per output row (max 32)", "0.021 (0.108)"],
    ["layers.1.mlp.up_proj, row 1462, input col 658", "2.30 (4.37)", "per output row (max 32.24)", "2.29 (0.157)"],
]

# Insert the table literal just before the (replaced) paragraph is awkward;
# put the literal immediately before the new para, as with other appendix tables.
new_f = (
    "AWQ_SCALE_ROWS=" + repr(AWQ_TAB) + "\n"
    'para("The same weight-error file also lists a large maximum absolute difference for AWQ on layers.1.mlp.down_proj (4.97) and on other MLP up and down matrices (20 of 32 have relative Frobenius error above 0.3; the largest relative error is 4.42, on layers.0.mlp.up_proj). A check that dequantized the AWQ OpenVINO IR the same way as weight_err.py, then compared it with the FP16 weights (diag/awq_scale_check.txt), confirms the cause. weight_err.py does not undo AWQ\u2019s per-channel smoothing scales, which NNCF folds into up_proj and down_proj (diag/FINDINGS.md). All 80 attention and gate-projection matrices have exactly the same error statistics as RTN.")\n'
    'para("On layers.1.mlp.down_proj the 4.97 maximum sits at output row 400, input column 1417 (FP16 \\u22120.338, AWQ \\u22125.3125). The median AWQ/FP16 ratio on that column is 15.67, so AWQ scaled the channel up about 15.7 times; the column\u2019s L2 error is 9.62, against at most 0.81 on every other column. Column 2558 is not involved (rank 1,584 of 8,192, maximum absolute difference 0.053). Dividing each input column by its median ratio drops the maximum absolute difference from 4.97 to 0.184 and the relative error from 0.391 to 0.156. The same per-column undo on layers.15.mlp.down_proj (maximum 1.52 at column 7258, ratio 4.79) leaves 0.125 (relative error 0.140).")\n'
    'para("For the two up_proj matrices the scales are on output rows, not input columns, with row-scale maxima of 32 and 32.24. Dividing out the per-row median ratio drops the relative error of layers.0.mlp.up_proj from 4.42 to 0.108 and the maximum absolute difference from 2.19 to 0.021. On layers.1.mlp.up_proj the relative error falls from 4.37 to 0.157, in the same range, but the maximum absolute residual stays 2.29 (from 2.30). A simple per-row median is therefore not a perfect inverse for every entry of that matrix; the check does not report which entry remains large. The drop in relative error still shows that most of the matrix is accounted for by the row scales.")\n'
    'para("This is the expected pattern of AWQ\u2019s equivalent channel scaling: a factor on a down_proj input column and on the matching up_proj output row cancels in the MLP product, leaving the computed result unchanged. We did not multiply the recovered per-channel factors to check that they are exact reciprocals. After the scales are divided out, the residual relative error is 0.108\\u20130.157, next to RTN\u2019s range of 0.105\\u20130.148. That matches AWQ\u2019s healthy standard-protocol WikiText-2 word perplexity of 14.89 (RTN 15.59; GPTQ 423.66) and is unrelated to GPTQ\u2019s column-2558 defect.")\n'
    'table(AWQ_SCALE_ROWS,[2.3,1.15,1.7,1.4],cap="AWQ stored-weight differences versus FP16, before and after dividing out a median per-column (down_proj) or per-row (up_proj) scale. Values are from diag/awq_scale_check.txt, which dequantized models/ov-awq-int4 the same way as weight_err.py. Relative error is the Frobenius norm of the difference divided by the Frobenius norm of the FP16 matrix.",fs=7.5)\n'
)

replace_span(
    'para("The same weight-error file also lists a large maximum absolute difference for AWQ',
    "The files on hand do not identify which input column holds the 4.97 entry.\")",
    new_f,
)

P.write_text(src)
print("patched", P, "bytes", len(src))
# sanity: no leftover hedge
assert "do not identify which input column" not in src
assert "column 1417" in src
assert "AWQ_SCALE_ROWS" in src
print("ok")
