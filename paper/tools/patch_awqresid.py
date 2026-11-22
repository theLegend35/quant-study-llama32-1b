#!/usr/bin/env python3
"""Update Appendix F and §5.4 with the leftover 2.29 residual diagnosis.

    python3 /workspace/paper/tools/patch_awqresid.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_awqresid.py. Numbers are from
/workspace/laptop/awq_residual_check.txt (and the earlier scale-check
values already in the table). Arithmetic from logged values:
-2.5/12.9655 = -0.192819, error vs -0.208008 is 0.015189;
0.208008*12.9655 = 2.69686 (clip to 8*0.3125 = 2.5);
0.5*0.3125/12.9655 = 0.01205.
"""
from pathlib import Path

P = Path("/workspace/paper/build_paper.py")
src = P.read_text()


def replace_span(start, end_inclusive, new):
    global src
    a = src.find(start)
    b = src.find(end_inclusive, a)
    assert a >= 0 and b >= 0 and src.count(start) == 1, (
        a, b, start[:80], src.count(start)
    )
    b = b + len(end_inclusive)
    src = src[:a] + new + src[b:]


# ----- §5.4 -----
replace_span(
    "The 4.97 entry is in input column 1417",
    "Details are in Appendix F.",
    "The 4.97 entry is in input column 1417, not column 2558; dividing "
    "out the estimated scales brings the residual relative error back "
    "into the usual INT4 range. A leftover 2.29 maximum after a per-row "
    "median undo on layers.1.mlp.up_proj was an artifact of that "
    "estimator on a heavily scaled row, not an abnormal weight. "
    "AWQ\u2019s standard-protocol perplexity is 14.89. Details are in Appendix F.",
)

# ----- Table 30: do not treat the 2.29 leftover as the raw-max site -----
replace_span(
    "AWQ_SCALE_ROWS=",
    "['layers.1.mlp.up_proj, row 1462, input col 658', '2.30 (4.37)', 'per output row (max 32.24)', '2.29 (0.157)']]",
    "AWQ_SCALE_ROWS="
    "[['Matrix and site of max. raw difference', 'Raw max-abs (rel. err.)', "
    "'Estimated scale (median AWQ/FP16 ratio)', 'After undoing, max-abs (rel. err.)'], "
    "['layers.1.mlp.down_proj, row 400, input col 1417', '4.97 (0.391)', "
    "'15.67 on that input column', '0.184 (0.156)'], "
    "['layers.15.mlp.down_proj, row 1159, input col 7258', '1.52 (0.314)', "
    "'4.79 on that input column', '0.125 (0.140)'], "
    "['layers.0.mlp.up_proj, row 348, input col 1224', '2.19 (4.42)', "
    "'per output row (max 32)', '0.021 (0.108)'], "
    "['layers.1.mlp.up_proj, row 1462, col 658 (raw max)', '2.30 (4.37)', "
    "'per output row (max 32.24)', '2.29 is at row 1473'], "
    "['layers.1.mlp.up_proj, row 1473, col 923 (leftover)', "
    "'2.29 (raw difference)', '12.965 from paired down_proj 1473', '0.015']]",
)

# ----- Appendix F prose: leftover diagnosis + pairing -----
replace_span(
    "On layers.1.mlp.up_proj the relative error falls",
    "column-2558 defect.",
    "On layers.1.mlp.up_proj the relative error after a per-row median "
    "undo falls from 4.37 to 0.157, in the usual INT4 range, but the "
    "maximum absolute residual stays 2.29 (from 2.30). A follow-up check "
    "(diag/awq_residual_check.txt) locates that leftover 2.29 at output "
    "row 1473, input column 923 (FP16 \u22120.208008, stored AWQ \u22122.500000), "
    "not at the raw-max site (row 1462, column 658). The median of that "
    "row\u2019s per-element AWQ/FP16 ratios is 0: at least half of the "
    "dequantized AWQ weights in the row are exactly 0 (every FP16 entry "
    "in the row is nonzero). The per-row estimator therefore returns no "
    "usable scale (s is recorded as NaN), and the 2.29 figure is just the "
    "raw, unscaled difference. The paired down_proj input column 1473 "
    "gives s_down = 0.0771277, so 1/s_down = 12.9655: AWQ scaled that "
    "intermediate channel by about 13 times. With a group step of 0.3125, "
    "original weights smaller than about half a step after scaling "
    "(roughly 0.012) quantize to zero, which accounts for the mass of "
    "exact zeros. Dividing the stored value by 12.9655 gives \u22120.1928 "
    "against FP16 \u22120.2080, an error of 0.015. That is ordinary INT4 "
    "size: the RTN error at this exact element is 0, and the RTN error at "
    "the leftover residual on layers.0.mlp.up_proj (row 4548, column 890, "
    "residual 0.0207) is 0.018. The inferred INT4 code, "
    "round(W_awq / group scale), is exactly \u22128.0000; the group scale is "
    "0.3125, so 8 \u00d7 0.3125 = 2.5 equals both the stored |W_awq| and the "
    "largest |W_awq| in the group (columns 896\u20131023). Using 1/s_down, the "
    "pre-quant value would be 0.208008 \u00d7 12.9655 = 2.697, which exceeds "
    "the symmetric INT4 maximum of 2.5, so the 0.015 residual is a slight "
    "clip of the group maximum, not an abnormal weight. The leftover 2.29 "
    "after the per-row median is an artifact of that estimator on a "
    "heavily scaled row. Layer 0\u2019s leftover of 0.0207 is about half a "
    "step (|qerr|/step = 0.516) and is ordinary rounding.\")\n"
    "para(\""
    "The recovered scales are approximately reciprocal, not exact. Across "
    "paired up_proj rows and down_proj columns, the correlation of s_up "
    "with 1/s_down is 0.9996 (layer 0) and 0.9960 (layer 1), but the "
    "median of the product s_up \u00d7 s_down is 0.966 in both layers, about "
    "3.4% below 1 (median |product \u2212 1| is 0.0338 and 0.0337; 88.4% and "
    "90.5% of the finite products lie within 5% of 1; on layer 1, 73 of "
    "8,192 paired products are undefined because a median was unusable, "
    "including row 1473). The offset is systematic rather than a few "
    "outlying channels. A plausible contributor is that the median of "
    "W_awq/W_fp16 is taken after INT4 rounding, which sends many small "
    "scaled values to zero and can pull the estimated scale down a "
    "little; we did not measure that bias against an unquantized AWQ "
    "scale. The gate_proj matrices look like ordinary INT4 (relative "
    "error 0.106 on layers 0 and 1), not a scale transform. After a "
    "usable scale is applied, the leftover element errors sit in the "
    "usual INT4 range, which matches AWQ\u2019s healthy standard-protocol "
    "WikiText-2 word perplexity of 14.89 (RTN 15.59; GPTQ 423.66) and is "
    "unrelated to GPTQ\u2019s column-2558 defect.",
)

# ----- caption: cite the residual-check log -----
replace_span(
    "Values are from diag/awq_scale_check.txt, which dequantized models/ov-awq-int4 the same way as weight_err.py.",
    "Values are from diag/awq_scale_check.txt, which dequantized models/ov-awq-int4 the same way as weight_err.py.",
    "On layers.1.mlp.up_proj the leftover 2.29 after the per-row median "
    "is at row 1473, not at the raw-max site (row 1462); that row\u2019s "
    "median was 0, so the scale in the last row is 1/s_down from the "
    "paired down_proj column. Values are from diag/awq_scale_check.txt "
    "and diag/awq_residual_check.txt, which dequantized models/ov-awq-int4 "
    "the same way as weight_err.py.",
)

P.write_text(src)
print("patched", P, "chars", len(src))

# sanity
assert "not a perfect inverse" not in src
assert "does not report which entry remains large" not in src
assert "We did not multiply the recovered per-channel factors" not in src
assert "row 1473" in src
assert "12.9655" in src
assert "0.0771277" in src
assert "0.9996" in src
assert "0.9960" in src
assert "0.966" in src
assert "artifact of that estimator" in src
assert "['layers.1.mlp.up_proj, row 1462, input col 658', '2.30 (4.37)', 'per output row (max 32.24)', '2.29 (0.157)']" not in src
print("sanity ok")
