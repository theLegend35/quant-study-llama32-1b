#!/usr/bin/env python3
"""Update Appendix F with the confirmed median-of-ratios bias (H1).

    python3 /workspace/paper/tools/patch_awqoffset.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_awqoffset.py. Numbers are from
/workspace/laptop/awq_offset_check.txt. The script's
'constant scale on activation path: True' flag is a false positive
(post_attention_layernorm weight and fq_weights dequant nodes) and is
not cited.
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


# ----- data table, next to the existing Table 30 rows -----
replace_span(
    "AWQ_SCALE_ROWS=",
    "AWQ_SCALE_ROWS=",
    "AWQ_OFFSET_ROWS="
    "[['Estimator', 'Layer 0 product p50', 'Layer 1 product p50'], "
    "['Median of ratios, real AWQ vs FP16', '0.9662', '0.9663'], "
    "['Least-squares, real AWQ vs FP16', '1.0010', '1.0007'], "
    "['Median of ratios, simulation (true product = 1)', '0.9668', '0.9668'], "
    "['Least-squares, simulation (true product = 1)', '1.0013', '1.0011']]\n"
    "AWQ_SCALE_ROWS=",
)

# ----- Appendix F pairing paragraph -----
replace_span(
    "The recovered scales are approximately reciprocal, not exact.",
    "column-2558 defect.",
    "The recovered scales are approximately reciprocal. Across paired "
    "up_proj rows and down_proj columns, the correlation of s_up with "
    "1/s_down is 0.9996 (layer 0) and 0.9960 (layer 1), and the median of "
    "the product s_up \u00d7 s_down, computed with the same per-row/column "
    "median of AWQ/FP16 ratios, is 0.9662 and 0.9663, about 3.4% below 1 "
    "(median |product \u2212 1| is 0.0338 and 0.0337; 88.4% and 90.5% of the "
    "finite products lie within 5% of 1; on layer 1, 73 of 8,192 paired "
    "products are undefined because a median was unusable, including row "
    "1473). A follow-up check (diag/awq_offset_check.txt) shows that this "
    "offset is an artifact of that median-of-ratios estimator after INT4 "
    "rounding and zeros, not a failure of the folded scales to be "
    "reciprocal. Applying exact-reciprocal scales to the real FP16 "
    "weights, then quantizing with the same INT4_SYM group-128 RTN rule, "
    "reproduces a median product of 0.9668 in both layers, even though "
    "the true product is 1. A least-squares estimator "
    "s = \u27e8W_q, W_fp16\u27e9 / \u27e8W_fp16, W_fp16\u27e9, restricted to nonzero "
    "codes, recovers a product of 1.0010 (layer 0) and 1.0007 (layer 1) "
    "on the real AWQ weights, and 1.0013 and 1.0011 on the simulation. A "
    "median that simply drops the exact-zero dequantized values "
    "overshoots to 1.0301 and 1.0298. The least-squares product can sit "
    "a little off 1 because of ordinary INT4 error; the check does not "
    "claim that every channel is an exact reciprocal.\")\n"
    "para(\""
    "No extra scale was found on the activation path between up_proj and "
    "down_proj. The IR chain\u2019s only Constant is the "
    "post_attention_layernorm weight (shape [1, 1, 2048]), which is "
    "applied to the residual stream before up_proj and gate_proj, not "
    "between them. The MLP Multiply that implements silu(gate)\u00d7up has "
    "no Constant inputs. Apparent extra Multiplies in a name-scoped scan "
    "are the weight-dequantization nodes (integer codes times the group "
    "scale). The export log shows data-aware AWQ followed by weight "
    "compression, with no Scale Estimation apply step and no "
    "LoRA-correction mention (the GPTQ scale-estimation ablation log, by "
    "contrast, records \u201cApplying GPTQ with Scale Estimation\u201d). The "
    "gate_proj matrices look like ordinary INT4, not a second "
    "smooth-scale: relative error 0.1061 and 0.1063, least-squares "
    "row-scale median 1.000 on both layers, and essentially no "
    "correlation with the up_proj scales (\u22120.0054 and 0.0025). The "
    "gate_proj median-of-ratios scale itself is 0.984, the same downward "
    "bias as on the paired up and down matrices. After a usable scale is "
    "applied, the leftover element errors sit in the usual INT4 range, "
    "which matches AWQ\u2019s healthy standard-protocol WikiText-2 word "
    "perplexity of 14.89 (RTN 15.59; GPTQ 423.66) and is unrelated to "
    "GPTQ\u2019s column-2558 defect.",
)

# ----- new table after Table 30; keep Table 30's number -----
replace_span(
    'Relative error is the Frobenius norm of the difference divided by the Frobenius norm of the FP16 matrix.",fs=7.5)',
    'Relative error is the Frobenius norm of the difference divided by the Frobenius norm of the FP16 matrix.",fs=7.5)',
    'Relative error is the Frobenius norm of the difference divided by the Frobenius norm of the FP16 matrix.",fs=7.5)\n'
    'table(AWQ_OFFSET_ROWS,[3.5,1.4,1.4],cap="Median product s_up \u00d7 s_down on paired MLP channels in layers 0 and 1. The first two rows are estimated from the stored AWQ weights versus FP16. The last two apply exact-reciprocal scales to the real FP16 weights, then quantize with INT4_SYM group size 128 RTN and re-estimate; the true product is 1. Values are from diag/awq_offset_check.txt.",fs=8)',
)

P.write_text(src)
print("patched", P, "chars", len(src))

assert "approximately reciprocal, not exact" not in src
assert "plausible contributor" not in src
assert "we did not measure that bias" not in src
assert "0.9668" in src
assert "1.0010" in src
assert "1.0007" in src
assert "1.0013" in src
assert "1.0011" in src
assert "AWQ_OFFSET_ROWS=" in src
assert "post_attention_layernorm" in src
assert "constant scale on activation path" not in src
assert "false positive" not in src.lower()  # don't use that phrase in the paper
print("sanity ok")
