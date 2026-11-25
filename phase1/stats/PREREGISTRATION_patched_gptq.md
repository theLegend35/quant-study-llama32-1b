# Pre-registration: patched GPTQ INT4 accuracy run and Holm family 7 -> 10

Written 2026-11-25 17:04 -06:00 (America/Chicago), before the accuracy run on the patched model started and before any accuracy result for it existed.

## Model
`models/ov-gptq-int4-l1down-rtn`: the default NNCF 3.4.0 GPTQ INT4 OpenVINO IR with only `layers.1.mlp.down_proj` replaced by its RTN INT4 values (made by `diag/patch_l1down.py`; the replaced matrix is an FP16 constant holding RTN INT4 values, so its weight values equal those of the RTN INT4 model for that matrix). Standard-protocol WikiText-2 word perplexity of this IR was already known (14.86); no accuracy task had been run on it.

## Run (identical to the Phase 1 GPTQ re-run)
- `lm_eval --model openvino --model_args pretrained=models/ov-gptq-int4-l1down-rtn --device gpu --tasks arc_easy,arc_challenge,hellaswag,winogrande --batch_size 1 --seed 42 --log_samples`
- `lm_eval --model openvino --model_args pretrained=models/ov-gptq-int4-l1down-rtn,max_length=2048 --device gpu --tasks piqa,boolq --batch_size 1 --seed 42 --log_samples`
- Same laptop, OpenVINO 2026.4.1, lm-eval 0.4.13. Metrics unchanged: acc_norm for ARC-C, ARC-E, HellaSwag and PIQA; acc for WinoGrande and BoolQ.

## Analysis plan
`p1_paired.py` unchanged in method (stratified paired bootstrap, B = 10,000, seed 20261119, percentile 95% CI, +1 smoothing; exact McNemar). The Holm family grows from 7 to 10 comparisons by adding exactly these three, all on the six-task average:
8. patched GPTQ INT4 vs RTN INT4
9. AWQ INT4 vs patched GPTQ INT4
10. patched GPTQ INT4 vs GPTQ INT4 (unpatched)

The original seven comparisons are re-reported with Holm over all ten. Patched GPTQ vs FP16 is reported descriptively only (outside the family). Per-task tests stay uncorrected. The patched model is a diagnostic, not an extra cell of the compression grid.
