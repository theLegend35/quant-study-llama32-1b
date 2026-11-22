# GPTQ-INT4 perplexity diagnosis (Llama-3.2-1B, NNCF 3.4.0 / OV 2026.4.1)
All scripts/logs in diag\. Numbers below are copied from the logs named.

## Step 1 - per-token analysis (pertoken_analysis.py -> pertoken_analysis.log, prefix_test.log)
4 WikiText-2 test passages x 2048 tokens (offsets 0/50k/120k/200k), OVModelForCausalLM on GPU, full-sequence logits.
Two conditions: "bos" = window starts with <|begin_of_text|>; "none" = window starts mid-text with no BOS
(this is what lm-eval's rolling-loglikelihood windows after the first one look like).

| model | ppl (bos) | ppl (none) | entropy bos/none | mean max-logit bos/none |
|---|---|---|---|---|
| FP16 | 9.58 | 9.58 | 2.29/2.30 | 22.23/22.20 |
| AWQ  | 11.04 | 11.14 | 2.44/2.45 | 21.84/21.78 |
| GPTQ | 12.11 | **3043.9** | 2.52/**6.16** | 22.03/**15.66** |

- No NaN/inf logits in any model. GPTQ is not overconfident; when it fails it is *under*-confident (high entropy, low max logit).
- With BOS: GPTQ is only mildly worse than AWQ (mean NLL diff +0.093, median +0.012); top 1% tokens = 42% of excess. Worst tokens are rare names/subword pieces. No category effect (newline/digit/punct/word all similar).
- Without BOS: degradation is uniform, not a few catastrophic tokens: median diff +5.1 nats, 82% of tokens >2 nats worse, top 1% tokens = only 2.9% of excess. All token categories are hit.
- Position profile (none): GPTQ is fine for positions 0-32 (3.26 vs AWQ 3.20), then collapses: 4.42 at [32,128), ~8.3 from 256 on.
- prefix_test.log (2 passages): GPTQ BOS 14.22 / no prefix 2543.7 / EOS prefix 16088; FP16 11.10/11.11/162.6; AWQ 12.84/12.98/224.4.
- This explains the lm-eval symptoms: only the first window of each doc has BOS; shorter max_length = more non-BOS windows = worse ppl (1258 @256 vs 424 @2048). MC tasks are short, BOS-prefixed contexts, so they are barely affected.

## Step 2 - per-layer analysis (weight_err.py/.log/.json, outlier_check.log, hidden_err2.py/.log)
Weights dequantized by constant-folding each MatMul's weight subgraph from the OV IR and compared with HF fp16 weights.
- Relative Frobenius error: RTN 0.105-0.148, GPTQ 0.115-0.177 (uniformly ~1.06-1.26x RTN, expected since GPTQ minimizes output error, not weight error). No zero/inf/NaN scales anywhere; scales from 4e-4 to 0.156.
- AWQ weight errors are not comparable as-is (AWQ folds per-channel scales into up_proj/down_proj, rel err 0.2-4.4 by design); its attention projections equal RTN.
- **Single outlier: layers.1.mlp.down_proj in GPTQ.** Max abs error 0.833 vs <=0.12 in every other GPTQ matrix, and no RTN matrix exceeds 0.2. It is concentrated in input (intermediate) column 2558: column error norm 2.68 vs RTN 0.08 (next columns 4616/6540/8089/2937: 1.1-1.5 vs ~0.08). Example: W[2023,2558] fp16 0.052 -> GPTQ -0.781 (RTN 0.053). GPTQ's error-compensation updates pushed small weights to group-max values.
- Hidden states (HF torch fp32 with dequantized weights, 512-token passage; embeddings/lm_head kept fp16 so ppl differs from OV):
  - GPTQ none: ppl 1424.6; hidden rel err of positions 1+ jumps from 0.36 after layer 1 to 2.98 after layer 2 (RTN: 0.29 -> 0.34); position 0 stays 0.05.
  - GPTQ bos: ppl 16.10 (RTN bos 16.48, RTN none 15.79, FP16 none/bos 13.03/13.59).
  - Single-layer swap (GPTQ weights in one layer only, rest fp16, no BOS): every layer 12.96-13.19, except **layer 1: 844.1**.
  - Replace only GPTQ layers.1.mlp.down_proj: with RTN -> none 14.72 / bos 15.52; with FP16 -> 13.92 / 14.90. That is, the remaining 111 GPTQ matrices are healthy (better than RTN).
- Mechanism (lower confidence): the corrupted columns are only harmful when there is no BOS. A likely story is that those intermediate channels were barely excited in the BOS-prefixed calibration data, so the Hessian columns are near-singular and GPTQ dumped error into them. I did not measure Hessian/calibration activation statistics, so this is unverified. Channel 2558 is not the massive-activation channel (that is 1417, |act| 1261 at pos0).

## Step 3 - ablation rebuilds (run_ablations.ps1, ablation-status.txt)
(filled in below as runs complete)

### (a) --dataset c4 (models\ov-gptq-int4-c4; c4 was available, allenai/c4 downloaded)
- lm-eval wikitext word_perplexity @2048: **305.69** (byte_ppl 2.9158, bits_per_byte 1.5439). Still broken (baseline 423.7).
- maxerr_check.log: same defect, layers.1.mlp.down_proj max abs err 0.883, worst column again 2558. So the corruption is NOT specific to wikitext2 calibration data.

### Extra targeted test: surgical patch (models\ov-gptq-int4-l1down-rtn, diag\patch_l1down.py)
This is not a rebuild. It is the baseline GPTQ IR with only layers.1.mlp.down_proj replaced by the RTN-INT4 dequantized weight, stored as an fp16 constant: 1 of 112 matrices, about 0.3% extra bin size. Why: steps 1-2 pinned the failure to that one matrix.
- lm-eval wikitext word_perplexity @2048: **14.86** (byte_ppl 1.6564, bpb 0.7280), vs GPTQ 423.7, RTN 15.59, AWQ 14.89, FP16 12.89.
- So the other 111 GPTQ matrices are fine, and GPTQ minus this defect is at least as good as AWQ on this metric.
- Caveat: in the patched matrix the weight is fp16 holding RTN-quantized values, not packed int4. Accuracy is equivalent to RTN for that matrix, but the bin is not int4-packed there.

## Step 4 - why ov-gptq-int4-se fails without BOS (se_bisect.py/.log/.json, se_extra.py/.json, act_check.py/.log/.json, patch_se_l1down.py)
Fast metric: 4 WikiText-2 test passages x 2048 tokens (offsets 0/50k/120k/200k), OV GPU, same as pertoken_analysis. Patched variants = se IR with listed matrices replaced by RTN-INT4 dequantized fp16 constants (patch_l1down.py method).
- Metric check against known numbers: GPTQ none 3043.9 / bos 12.04 (matches Step 1), GPTQ-l1down-rtn none 11.01, RTN 11.71, FP16 9.58.
- se: bos 10.90, none **562.1**. Mean NLL by position bin [0,32)/[32,128)/[128,512)/[512,2047): none 3.81/3.32/5.68/6.73 vs bos 3.65/2.41/2.47/2.34. Same pattern as default GPTQ: fine for about 32 tokens, then degrades.
- Layer-15 hypothesis rejected: RTN into all of L15 555.6, L15 mlp 563.1, L15 attn 557.8. Groups: L0-3 **11.25**, L4-7 637.8, L8-11 580.7, L12-14 563.1.
- Bisection inside L0-3: layer 0 alone 11.32, layer 1 alone 11.00, layers 2-3 452.9. Layer 0: mlp 11.29 (attn 479.1). layers.0.mlp.down_proj alone **11.20**; gate+up 641.4.
- se_extra: se+RTN[layers.1.mlp.down_proj] alone **10.97**; layer 1 without down_proj 223.8; L0down+L1down 11.05. Default GPTQ+RTN[layers.0.mlp.down_proj] alone: 3043.9 -> **15.14**.
- **So se has the same layers.1.mlp.down_proj defect as default GPTQ, just less concentrated.** Max-abs-err ranking misses it, but in se that matrix's input column 2558 has error norm 2.08 vs RTN 0.08 (26x). That is the largest excess column-error ratio among all 112 se matrices (default GPTQ: 2.68, 33.5x). The L15 max-abs-err matrices are a red herring.
- Patching either L0 down_proj (upstream) or L1 down_proj fixes both builds. So the failure needs both pieces: the corrupted L1 down column, plus something in L0 down_proj's quantized output that drives the input to that column in no-BOS context. This interaction is inferred from the patch results; the intermediate activations of the quantized models were not measured.
- lm-eval wikitext word_perplexity @2048 (no-BOS rolling windows): se 202.6 -> se+RTN[L1down] (models\ov-gptq-int4-se-l1down-rtn) **14.78** (byte 1.6547, bpb 0.7266); se+RTN[L0down] (models\ov-gptq-int4-se-patched) **14.91** (byte 1.6574, bpb 0.7289). For reference: FP16 12.89, RTN 15.59, AWQ 14.89, default GPTQ+L1down 14.86.

### Attention-sink / calibration-activation hypothesis (act_check.py, FP16 HF model, same 4 passages, stats exclude position 0)
- layers.1.mlp.down_proj input channel 2558: mean |x| bos 0.0180 vs none 0.0175 (ratio 0.98). Rank 678/8192 under BOS, so it is NOT barely activated in BOS-prefixed text. What differs is the tail: max |x| bos 0.131 vs none 0.777, and position 0 |x| 1.32 when position 0 is a normal token vs 0.47 when it is BOS. The other high-error columns show the same thing (4616: max 0.11 vs 1.13; 8089: 0.11 vs 0.83; 2937: 0.14 vs 7.43). Their means are unchanged, but they spike only without BOS.
- se's L15 columns (up_proj 26, o_proj 1023, gate) have identical stats with and without BOS (ratio ~1.0, similar max), consistent with L15 being harmless.
- Across all matrices, excess column error (GPTQ/RTN) is not correlated with the none/bos activation ratio (median Spearman 0.002 GPTQ, -0.006 se). It is weakly negatively correlated with BOS mean-square activation (-0.03 / -0.17).
- Verdict: the strict hypothesis ("error is dumped into channels that are barely activated in BOS calibration") is not supported, because channel 2558 is moderately active on average. A weaker version fits: the bad columns are ones whose activations are near-zero-variance and bounded under BOS-prefixed text but get large rare spikes without BOS (mainly at the early sink-like positions). GPTQ's objective, weighted by second moments, tolerates large error there because those spikes never appear in calibration. This is not a general pattern across matrices, and Hessians/calibration activations from the actual NNCF calibration set were not measured.

### What can / cannot be claimed
- Can claim: in both GPTQ builds (default and scale-estimation; c4 calibration also has the L1 col-2558 defect), the no-BOS collapse is caused by the GPTQ-quantized layers.1.mlp.down_proj, specifically input column 2558 and a few siblings. Replacing that one matrix with RTN (or the upstream L0 down_proj) restores no-BOS word ppl to about 14.8-14.9, on par with AWQ. Max-abs-error screening alone misses it in the SE build; column-error ratio vs RTN finds it. The remaining 111 GPTQ matrices are fine.
- Cannot claim: a verified mechanism. The attention-sink explanation is suggestive (channel spikes only without BOS), but not shown in the actual calibration Hessian, and there is no global correlation. Also unclaimable: generality beyond Llama-3.2-1B/NNCF 3.4.0, and that this is an NNCF bug as opposed to a GPTQ property. Patched models store that matrix as fp16-holding-RTN values, not packed int4.
- Cleanup: no deq_*.pt files existed anywhere under the project; temp patch dirs models\tmp-* removed.
