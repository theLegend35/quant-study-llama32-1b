P1_SE_ROWS=[["Version","ARC-C","ARC-E","HellaSwag","WinoGrande","4-task average"],
 ["FP16","0.3677 (0.0141)","0.6195 (0.0100)","0.6433 (0.0048)","0.6172 (0.0137)","0.562"],
 ["RTN INT8","0.3635 (0.0141)","0.6195 (0.0100)","0.6431 (0.0048)","0.6093 (0.0137)","0.559"],
 ["RTN INT4","0.3242 (0.0137)","0.5745 (0.0101)","0.6000 (0.0049)","0.5951 (0.0138)","0.524"],
 ["GPTQ INT4","0.3336 (0.0138)","0.5661 (0.0102)","0.5823 (0.0049)","0.5919 (0.0138)","0.519"],
 ["AWQ INT4","0.3396 (0.0138)","0.5993 (0.0101)","0.6173 (0.0049)","0.6085 (0.0137)","0.541"]]
P1_QA_ROWS=[["Version","BoolQ acc (SE)","PIQA acc_norm (SE)","WikiText-2 word ppl, standard protocol","WikiText-2 word ppl, BOS on every window"],
 ["FP16","0.6391 (0.0084)","0.7497 (0.0101)","12.89","15.46"],
 ["RTN INT8","0.6306 (0.0084)","0.7454 (0.0102)","12.92","15.50"],
 ["RTN INT4","0.5596 (0.0087)","0.7318 (0.0103)","15.59","19.98"],
 ["GPTQ INT4","0.5450 (0.0087)","0.7171 (0.0105)","423.66","20.68"],
 ["AWQ INT4","0.5963 (0.0086)","0.7350 (0.0103)","14.89","18.22"]]
P2_TRAIN_ROWS=[["Run","Seed","Mean train loss","Final logged loss","Holdout loss","Train time (s)","Peak GPU memory (GB)"],
 ["fp16-seed0","0","1.8544","1.8011","1.6962","658.4","5.84"],
 ["fp16-seed1","1","1.8654","1.7627","1.6942","655.5","5.84"],
 ["fp16-seed2","2","1.8586","1.8912","1.6953","652.4","5.84"],
 ["int8-rtn-seed0","0","1.8554","1.8021","1.6972","661.1","5.84"],
 ["int8-rtn-seed1","1","1.8665","1.7647","1.6954","661.2","5.84"],
 ["int8-rtn-seed2","2","1.8597","1.8916","1.6967","659.1","5.84"],
 ["int4-rtn-seed0","0","1.9380","1.8635","1.7687","653.1","5.84"],
 ["int4-rtn-seed1","1","1.9489","1.8476","1.7678","654.4","5.84"],
 ["int4-rtn-seed2","2","1.9422","1.9706","1.7683","652.0","5.84"]]
FQ_ROWS=[["Weight / mode","Bit-identical across thread counts and layouts?","Dequantized values exact in FP16?"],
 ["q_proj, INT8_SYM","yes","no"],
 ["q_proj, INT4_SYM","yes","yes"],
 ["k_proj, INT8_SYM","yes","no"],
 ["k_proj, INT4_SYM","yes","yes"],
 ["down_proj, INT8_SYM","yes","no"],
 ["down_proj, INT4_SYM","yes","yes"],
 ["gate_proj, INT8_SYM","yes","no"],
 ["gate_proj, INT4_SYM","yes","yes"],
 ["embedding (row subset), INT8_SYM","yes","no"],
 ["embedding (row subset), INT8_ASYM","yes","no"]]
CL_ROWS=[["Version","Task","Laptop (%)","Colab (%)","Colab \u2212 laptop (points)"],
 ["FP16","ARC-C","36.77","36.69","\u22120.09"],
 ["","ARC-E","61.95","61.78","\u22120.17"],
 ["","HellaSwag","64.33","64.21","\u22120.12"],
 ["","WinoGrande","61.72","61.17","\u22120.55"],
 ["","BoolQ","63.91","63.79","\u22120.12"],
 ["","PIQA","74.97","74.81","\u22120.16"],
 ["RTN INT8","ARC-C","36.35","36.52","+0.17"],
 ["","ARC-E","61.95","61.95","0.00"],
 ["","HellaSwag","64.31","64.31","0.00"],
 ["","WinoGrande","60.93","61.17","+0.24"],
 ["","BoolQ","63.06","63.52","+0.46"],
 ["","PIQA","74.54","74.65","+0.11"],
 ["RTN INT4","ARC-C","32.42","31.91","\u22120.51"],
 ["","ARC-E","57.45","57.49","+0.04"],
 ["","HellaSwag","60.00","60.00","0.00"],
 ["","WinoGrande","59.51","59.91","+0.39"],
 ["","BoolQ","55.96","55.60","\u22120.37"],
 ["","PIQA","73.18","73.29","+0.11"]]
NNCF_INT4_ROWS=[["Weight compression mode","% all parameters (layers)","% ratio-defining parameters (layers)"],
 ["float","0% (1 / 114)","0% (0 / 112)"],
 ["int8_asym, per-channel","21% (1 / 114)","0% (0 / 112)"],
 ["int4_sym, group size 128","79% (112 / 114)","100% (112 / 112)"]]
NNCF_INT8_ROWS=[["Weight compression mode","% all parameters (layers)","% ratio-defining parameters (layers)"],
 ["float","0% (1 / 114)","0% (0 / 113)"],
 ["int8_sym, per-channel","100% (113 / 114)","100% (113 / 113)"]]
GPTQ_MAT_ROWS=[["Matrix","GPTQ max. abs. error","GPTQ rel. Frobenius error","RTN max. abs. error"],
 ["layers.1.mlp.down_proj","0.833","0.125","0.042"],
 ["layers.1.mlp.up_proj (next-highest GPTQ max. abs.)","0.119","0.128","0.069"],
 ["Largest of the remaining 110 GPTQ matrices","0.117","\u2014","\u2014"],
 ["Range over all 112 RTN matrices","\u2264 0.096","0.105\u20130.148","\u2264 0.096"]]
GPTQ_COL_ROWS=[["Input column of layers.1.mlp.down_proj","GPTQ column-error norm","RTN column-error norm","BOS max |x|","no-BOS max |x|"],
 ["2558","2.681","0.080","0.131","0.777"],
 ["4616","1.457","0.082","0.110","1.127"],
 ["6540","1.291","0.082","0.212","1.602"],
 ["8089","1.106","0.085","0.114","0.828"],
 ["2937","1.075","0.082","0.138","7.432"]]
GPTQ_ABL_ROWS=[["Build / patch","Standard-protocol word ppl (lm-eval, 2048)","Four-passage no-BOS ppl"],
 ["GPTQ INT4 (default, WikiText-2 calibration)","423.66","3043.9"],
 ["GPTQ INT4, C4 calibration","305.69","\u2014"],
 ["GPTQ + RTN on layers.1.mlp.down_proj","14.86","11.01"],
 ["GPTQ + RTN on layers.0.mlp.down_proj","\u2014","15.14"],
 ["GPTQ with scale estimation (SE)","202.6","562.1"],
 ["SE + RTN on layers.1.mlp.down_proj","14.78","10.97"],
 ["SE + RTN on layers.0.mlp.down_proj","14.91","11.20"],
 ["RTN INT4","15.59","11.71"],
 ["AWQ INT4","14.89","\u2014"],
 ["FP16","12.89","9.58"]]
SPD_ROWS=[["Base","LoRA","Round","tok/s (mean of 30)","TTFT ms (mean of 30)","Peak commit (MB)","Peak working set (MB)","System CPU %","In Table 13?"],
 ["FP16","off","3","20.36","54.93","6,257","5,578","59.1","no (CPU 59%)"],
 ["FP16","off","6","24.98","41.96","6,251","5,577","9.3","yes"],
 ["FP16","off","7","24.81","41.52","6,253","5,578","8.9","yes"],
 ["FP16","on","4","22.86","44.20","6,291","5,598","9.6","yes"],
 ["FP16","on","5","22.86","44.63","6,287","5,313","9.6","yes (WS trimmed)"],
 ["FP16","on","8","23.06","44.06","6,283","5,594","8.2","yes"],
 ["RTN INT8","off","3","41.47","25.13","4,266","3,592","7.7","yes"],
 ["RTN INT8","off","6","41.30","25.66","4,253","3,588","8.0","yes"],
 ["RTN INT8","off","7","40.82","25.95","4,255","3,591","9.4","yes"],
 ["RTN INT8","on","4","32.73","31.14","4,282","3,611","7.4","yes"],
 ["RTN INT8","on","5","36.11","28.80","4,280","3,614","9.6","yes"],
 ["RTN INT8","on","8","36.03","28.97","4,276","3,613","10.6","yes"],
 ["RTN INT4","off","3","63.68","17.29","3,095","2,436","9.1","yes"],
 ["RTN INT4","off","6","60.75","18.12","3,102","2,430","9.6","yes"],
 ["RTN INT4","off","7","63.94","17.00","3,095","2,432","9.9","yes"],
 ["RTN INT4","on","4","52.89","21.46","3,116","2,452","8.3","yes"],
 ["RTN INT4","on","5","49.34","23.09","3,101","2,453","9.1","yes"],
 ["RTN INT4","on","8","52.09","21.97","3,108","2,452","7.8","yes"]]
SPD_SUM_ROWS=[["Base","LoRA","Kept rounds","tok/s mean (SD)","TTFT ms mean (SD)","Peak commit MB (mean)","Peak working set MB (mean)"],
 ["FP16","off","6, 7","24.9 (0.12)","41.7 (0.32)","6,252","5,578"],
 ["FP16","on","4, 5, 8","22.9 (0.12)","44.3 (0.30)","6,287","5,501 (5,596 without r5)"],
 ["RTN INT8","off","3, 6, 7","41.2 (0.34)","25.6 (0.41)","4,258","3,590"],
 ["RTN INT8","on","4, 5, 8","35.0 (1.93)","29.6 (1.31)","4,280","3,613"],
 ["RTN INT4","off","3, 6, 7","62.8 (1.77)","17.5 (0.58)","3,097","2,433"],
 ["RTN INT4","on","4, 5, 8","51.4 (1.87)","22.2 (0.83)","3,108","2,452"]]

para("Phase 1 per-task accuracies and standard errors. The next table reports the Phase 1 laptop scores from the official full lm-evaluation-harness runs (directories named fp16-full, and so on), with the standard errors stored by the harness as acc_norm_stderr (ARC-Challenge, ARC-Easy, HellaSwag) or acc_stderr (WinoGrande). These are the same runs as Table 5. The approximate SEs quoted in the main text (\u00b10.014, \u00b10.010, \u00b10.005) are these values rounded. The four-task average is copied from Table 5; the result files do not include an SE for the average. BoolQ and PIQA were scored in later qa runs, not in the full runs; they appear in the table after that, with WikiText-2 word perplexity. lm-eval reports the perplexity standard error as N/A, so none is given. The BOS-on-every-window perplexities are from a separate diagnostic (diag/bos_window_ppl.json). Phase 1 per-question logs were not saved, so a paired test like Table 14 cannot be run for Phase 1.")
table(P1_SE_ROWS,[1.1,1.15,1.15,1.2,1.2,0.85],cap="Phase 1 zero-shot accuracy on the laptop (OpenVINO backend, full runs). Each cell is accuracy (standard error). ARC-C, ARC-E and HellaSwag: acc_norm; WinoGrande: acc. The four-task average is the value published in Table 5. Sources: results/fp16-full, rtn-int8-full, rtn-int4-full, gptq-int4-full and awq-int4-full.",fs=8)
table(P1_QA_ROWS,[0.95,1.15,1.25,1.6,1.55],cap="Phase 1 BoolQ, PIQA and WikiText-2 word perplexity. BoolQ (acc) and PIQA (acc_norm) are from the later qa runs, not the full runs, and include lm-eval standard errors. The standard-protocol perplexities (BOS only on the first window of each document) are from the ppl runs; the harness records those SEs as N/A. The last column is word perplexity with BOS prepended to every 2048-token window (diag/bos_window_ppl.json; no SE in that file). These columns match Tables 6 and 8 (standard-protocol GPTQ is shown here to two decimals, 423.66, and as 423.7 in Table 8).",fs=7.5)

H2("Appendix C. LoRA training curves")
para("All nine Phase 2 runs used the same recipe: LoRA r = 16, \u03b1 = 32, dropout 0.05, on q_proj, k_proj, v_proj and o_proj, adapters unmerged; one epoch on the 5,000-example Dolly subset of Appendix A; AdamW, learning rate 2 \u00d7 10\u207b\u2074, cosine schedule with 3% warm-up, effective batch 16 (4 \u00d7 4 accumulation), gradient clipping 1.0, gradient checkpointing, FP16 autocast, LoRA weights in FP32. Training seeds were 0, 1 and 2. GPTQ and AWQ LoRA runs have not been done.")
figure(G+"fig_app_lora_loss.png",6.3,"Training loss versus optimizer step for the nine Phase 2 runs. Each line is the mean over the three seeds of one base; the band is the min\u2013max range across those seeds. Loss was logged every 10 steps (the per-run train_log.csv files).")
para("The table below lists the summary written at the end of each run (results/training_summary.csv). INT4 holds a higher training and held-out loss than FP16 and INT8, which stay close to each other. Wall time was about 11 minutes per run on the T4 (652\u2013661 s), and peak GPU memory was 5.84 GB in every run.")
table(P2_TRAIN_ROWS,[1.3,0.5,1.15,1.2,1.0,1.0,1.15],cap="Phase 2 training summaries (results/training_summary.csv). Mean train loss is the average of the logged step losses; final logged loss is the last of those (every 10 steps). Holdout loss is the loss on the 200 held-out Dolly examples after training. Train time is wall time in seconds, excluding the later evaluation.",fs=8)

H2("Appendix D. Phase 2 base checks")
para("Fake-quantizer determinism. The next table is the box-side check in lora/stats/mismatch/fq_determinism.json (recorded under PyTorch 2.15.0+cpu). For every tested weight and mode, the fake-quantizer produced bit-identical integer codes and scales across CPU thread counts and memory layouts. Dequantized values (integer \u00d7 scale) were exactly representable in FP16 for INT4_SYM, but not for INT8_SYM or INT8_ASYM. A laptop inspection note records a further bit-exact match of integer codes, FP16 scales and zero points against the OpenVINO IRs on selected layers (layer-0 q_proj and k_proj, layer-7 down_proj, layer-15 gate_proj, and the embedding, for both RTN models). The raw comparison dump from that laptop check was not in the appendix zip.")
table(FQ_ROWS,[2.2,2.2,2.1],cap="Fake-quantizer determinism on the box (fq_determinism.json). \u201cDequantized values exact in FP16\u201d means the integer\u00d7scale result equals its FP16 rounding.",fs=8)
para("Colab versus laptop no-LoRA scores. The table below compares the Phase 2 Colab evaluations of the frozen bases (the no-LoRA eval JSON files under results/eval/) with the Phase 1 laptop scores (the full runs for ARC and WinoGrande; the qa runs for BoolQ and PIQA). Differences are Colab minus laptop, in percentage points, computed from the unrounded accuracies. Every task is within 0.6 points, as stated in Sections 4.3 and 5.5. The largest gap is WinoGrande on FP16 (0.55 points). HellaSwag on INT8 and INT4, and ARC-Easy on INT8, match to the reported precision.")
table(CL_ROWS,[1.0,1.1,1.1,1.1,1.6],cap="No-LoRA accuracy, laptop versus Colab (percentage points). Laptop ARC and WinoGrande: full runs; laptop BoolQ and PIQA: qa runs. Colab: first-run eval JSON files. Diff. = Colab minus laptop, from unrounded values.",fs=8)

H2("Appendix E. NNCF compression logs")
para("The Phase 1 export logs (results/export-rtn-int8.log, export-rtn-int4.log, export-gptq-int4.log, export-awq-int4.log) each contain NNCF\u2019s \u201cStatistics of the bitwidth distribution\u201d table. The same INT4 table appears in the GPTQ rebuild log (results/gptq-v2-build.log). The two tables below copy those rows. The logs do not name the single float layer (0% of parameters). For the INT4 models, the 1 of 114 layers in int8_asym is the backup-precision embedding (and tied lm_head), as in Section 4.2; it is 21% of all parameters and 0% of the ratio-defining (decoder) layers. All 112 decoder linear layers are int4_sym with group size 128. For RTN INT8, 113 of 114 layers are int8_sym per channel (100% of parameters); there is no int8_asym backup row. The logs also record that GPTQ itself took 1:08:36 on the default build and 1:13:20 on the rebuild; data-aware AWQ took 0:01:35; the final weight-compression step took 12 s (INT8), 13 s (RTN INT4 and AWQ), 15 s (default GPTQ) and 11 s (GPTQ rebuild).")
table(NNCF_INT4_ROWS,[2.2,2.0,2.3],cap="NNCF bitwidth distribution for RTN INT4, GPTQ INT4, AWQ INT4 and the GPTQ rebuild (identical table in all four of those export logs).",fs=8)
table(NNCF_INT8_ROWS,[2.2,2.0,2.3],cap="NNCF bitwidth distribution for RTN INT8 (results/export-rtn-int8.log).",fs=8)

H2("Appendix F. GPTQ diagnostics")
para("This appendix summarises the laptop diagnosis in diag/FINDINGS.md and the accompanying JSON and log files. It is consistent with Section 5.4: the no-BOS WikiText collapse of NNCF GPTQ INT4 on this model is isolated to layers.1.mlp.down_proj, and within that matrix to input column 2558 (with a few smaller sibling columns). The remaining 111 GPTQ matrices are healthy. The proposed mechanism was not verified, and the patched models are not pure INT4, because the replaced matrix is stored as FP16 holding RTN values.")
para("The first of the two weight-error tables is from diag/weight_err.json (112 decoder linear matrices compared with the FP16 weights after dequantizing the OpenVINO IR). Relative Frobenius error is 0.105\u20130.148 for RTN and 0.115\u20130.177 for GPTQ (ratio 1.06\u20131.26). One outlier stands out: layers.1.mlp.down_proj has maximum absolute error 0.833 under GPTQ, against 0.119 in the next-highest GPTQ matrix and 0.096 in the worst RTN matrix. The figure below plots these maxima. The column table lists the worst input columns of that matrix (diag/outlier_check.log, diag/act_check.json). Column 2558 has GPTQ column-error norm 2.681 versus 0.080 for RTN. One entry in that column, output row 2023, is 0.0518 in FP16, 0.0531 after RTN, and \u22120.7812 after GPTQ. The main text\u2019s 0.83, 0.052 and \u22120.781 are these measurements rounded.")
figure(G+"fig_app_gptq_maxabs.png",6.3,"Maximum absolute weight error versus the FP16 matrix, for all 112 decoder linear layers (diag/weight_err.json). The highlighted point is layers.1.mlp.down_proj under GPTQ (0.833). The dashed line is 0.12; every other GPTQ matrix, and every RTN matrix, lies at or below 0.119.")
table(GPTQ_MAT_ROWS,[2.5,1.4,1.5,1.3],cap="Per-matrix weight error for GPTQ INT4 versus RTN INT4 (diag/weight_err.json). The second row is layers.1.mlp.up_proj, which has the second-largest GPTQ maximum absolute error. The third row is layers.15.mlp.up_proj (0.117).",fs=8)
table(GPTQ_COL_ROWS,[2.2,1.4,1.4,1.0,1.1],cap="Worst input columns of GPTQ layers.1.mlp.down_proj. Column-error norms are from diag/outlier_check.log; BOS and no-BOS max |activation| are from diag/act_check.json on an FP16 model (same four WikiText-2 passages as the per-token analysis). Column 2558 is only moderately active on average under BOS (mean |x| 0.0180, rank 678 of 8,192) but its no-BOS tail is larger (max 0.777 versus 0.131).",fs=7.5)
para("The ablation table collects perplexities from the diagnosis. Replacing only layers.1.mlp.down_proj with its RTN INT4 values restores the standard-protocol (lm-eval) word perplexity from 423.66 to 14.86, next to AWQ\u2019s 14.89 and RTN\u2019s 15.59; the BOS-on-every-window diagnostic for that patched model is 18.67, versus 20.68 for unpatched GPTQ (diag/bos_window_ppl.json). The same matrix is the defect in a C4-calibrated rebuild (305.69, worst column still 2558, max abs error 0.883) and in a scale-estimation build (202.6, restored to 14.78). Patching the upstream layers.0.mlp.down_proj also removes the four-passage collapse (15.14). A four-passage diagnostic (diag/se_bisect.json) shows the same pattern: GPTQ is fine for about the first 32 tokens without BOS and then degrades. What the files do not support is a verified mechanism, or any claim that the defect is general beyond Llama-3.2-1B and NNCF 3.4.0.")
table(GPTQ_ABL_ROWS,[3.2,2.0,1.7],cap="GPTQ ablation perplexities. The standard-protocol column is lm-eval WikiText-2 word perplexity at max_length = 2048 (BOS on the first window only): default models from the ppl runs; patched, C4 and scale-estimation models from diag/FINDINGS.md. The last column is a shorter diagnostic on four 2048-token WikiText-2 passages (diag/se_bisect.json and se_extra.json). A dash means that measurement was not in the files.",fs=7.5)

H2("Appendix G. Laptop LoRA timing logs")
para("Unmerged LoRA speed was measured on the same laptop as Phase 1, with OpenVINO GenAI 2026.4.1 on the Arc GPU, the seed-0 adapter, and scale 2.0. An earlier pass (speed-lora/status.txt, 18 November 2026, 21:44:54 to 22:18:42 as recorded) ran two rounds of each cell: first FP16, INT8, INT4, each off then on (rounds 1), then the same cells in reverse (rounds 2). Pass-1 JSON files do not record peak commit or system CPU. The numbers in Table 13 and in this appendix are from a second, balanced interleaved pass (speed-lora/run_pass2.ps1 and status2.txt, 22:20:09 to 23:05:31 the same day; the log records the Balanced power scheme). The script waited until a check2.status file contained ALLDONE, then, for each of FP16, RTN INT8 and RTN INT4, used the order off, on, on, off, off, on, stored as rounds 3\u20138. The pass recorded peak commit (private bytes) as well as peak working set, and system CPU load. One FP16 no-LoRA round (round 3) had system CPU 59.1% and was dropped, as stated in Section 5.6; the other kept rounds had system CPU between 7.4% and 10.6%.")
para("The first timing table lists every pass-2 round. The tok/s and TTFT columns are the mean of the 30 timed runs in that round (the tok_per_s_mean and ttft_ms_mean fields in the JSON), which is what Table 13 uses. The second table then gives the mean and sample standard deviation of those kept rounds. The tok/s means match Table 13 to the reported one decimal. Three small bookkeeping differences are worth recording, and are not applied back to Table 13. First, Table 13 lists TTFT for FP16 without LoRA as 41.8 ms; the mean of the two kept per-round means is 41.7 ms (41.96 and 41.52). Second, Table 13\u2019s peak commit for FP16 without LoRA is 6,253 MB; the mean of the two kept rounds is 6,252 MB (6,251 and 6,253). Third, Table 13\u2019s peak working set for FP16 with LoRA is 5,595 MB, which is the mean of rounds 4 and 8 only (5,597.6 and 5,594.1 MB; the unrounded mean is 5,595.9 MB). Round 5\u2019s working set was 5,313 MB (trimming) and was omitted from that average, while its tok/s, TTFT and commit were kept. Decode throughput falls by 8% (FP16), 15% (INT8) and 18% (INT4) when the unmerged adapter is attached, as in the main text.")
table(SPD_ROWS,[0.7,0.45,0.45,0.85,0.9,0.85,0.95,0.6,0.85],cap="Pass-2 unmerged-LoRA timing, one row per round (the per-round speed-lora JSON files). tok/s and TTFT are the mean of 30 timed runs after 5 warm-ups, with a 128-token prompt and 128 new tokens, greedy, batch size 1. Memory is the peak process commit and working set in that round, in megabytes. System CPU is cpu_sys_pct_during (not recorded in pass 1). Round 3 of FP16 without LoRA is listed but was not used in Table 13.",fs=6.5)
table(SPD_SUM_ROWS,[0.85,0.5,0.75,1.15,1.15,1.1,1.3],cap="Means of the kept pass-2 rounds in the previous table. Standard deviations are the sample SD of the per-round means (n = 2 for FP16 without LoRA, n = 3 otherwise). Peak memory means are of the per-round peaks. For FP16 with LoRA, the working-set mean of all three kept rounds is 5,501 MB; 5,596 MB is the unrounded mean of rounds 4 and 8 (5,595.9 MB) rounded to the nearest megabyte. Table 13 reports 5,595 MB for that same pair of rounds.",fs=8)
