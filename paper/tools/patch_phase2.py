import re
P="/workspace/paper/build_paper.py"
src=open(P).read()
lines=src.split("\n")
def repl_line(prefix,new):
    idx=[i for i,l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx)==1,(prefix,idx)
    lines[idx[0]]=new
def repl_block(start_prefix,end_prefix,new):
    s=[i for i,l in enumerate(lines) if l.startswith(start_prefix)]; assert len(s)==1,start_prefix
    e=[i for i,l in enumerate(lines) if l.startswith(end_prefix) and i>s[0]]; assert e,end_prefix
    lines[s[0]:e[0]+1]=new.split("\n")
def insert_after(prefix,new):
    idx=[i for i,l in enumerate(lines) if l.startswith(prefix)]; assert len(idx)==1,prefix
    lines[idx[0]+1:idx[0]+1]=new.split("\n")

# ---- Phase 2 data ----
insert_after('num=lambda s:', r'''
# ---------- Phase 2 data (Colab T4, lm-eval HF backend, zero-shot; % accuracy; LoRA = mean ± std over seeds 0,1,2) ----------
P2_TASKS=["ARC-C","ARC-E","Hella- Swag","Wino- Grande","BoolQ","PIQA"]
P2_BASES=["FP16","RTN INT8","RTN INT4"]
P2_NO={"FP16":[36.69,61.78,64.21,61.17,63.79,74.81,60.41],
       "RTN INT8":[36.52,61.95,64.31,61.17,63.52,74.65,60.35],
       "RTN INT4":[31.91,57.49,60.00,59.91,55.60,73.29,56.36]}
P2_LORA={"FP16":[(37.06,.47),(63.10,.09),(64.96,.11),(61.43,.24),(66.44,.42),(75.77,.11),(61.46,.18)],
         "RTN INT8":[(36.97,.49),(63.05,.46),(64.99,.10),(61.56,.34),(66.45,.44),(75.72,.13),(61.46,.25)],
         "RTN INT4":[(35.55,.13),(62.53,.21),(62.93,.28),(61.06,.50),(65.22,.53),(74.85,.35),(60.36,.07)]}
P2_DELTA={"FP16":["+0.37","+1.32","+0.75","+0.26","+2.65","+0.96","+1.05"],   # computed from unrounded per-seed values
          "RTN INT8":["+0.46","+1.09","+0.68","+0.39","+2.94","+1.07","+1.10"],
          "RTN INT4":["+3.64","+5.04","+2.93","+1.16","+9.62","+1.56","+3.99"]}
P2_TRAIN={  # train seconds (min–max), peak GPU GB, mean train loss per seed, held-out Dolly loss per seed
 "FP16":("652–658","5.84",(1.854,1.865,1.859),(1.696,1.694,1.695)),
 "RTN INT8":("659–661","5.84",(1.855,1.866,1.860),(1.697,1.695,1.697)),
 "RTN INT4":("652–654","5.84",(1.938,1.949,1.942),(1.769,1.768,1.768))}''')

# ---- figures: replace the planned Phase 2 diagram with a method diagram + results bar chart ----
repl_block('    # Fig: Phase 2 (planned)','    plt.tight_layout(); plt.savefig(G+"fig_phase2.png",dpi=200); plt.close()', r'''    # Fig: Phase 2 method (as run)
    fig,ax=plt.subplots(figsize=(9,3.1)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3.4)
    ax.add_patch(FancyBboxPatch((1.95,0.15),7.95,2.85,boxstyle="round,pad=0.05",fc="#f4f9ef",ec="#6aaa64",lw=1.2,ls="--"))
    ax.text(5.9,2.8,"Google Colab, NVIDIA T4 GPU (PyTorch + PEFT)",fontsize=8.5,ha="center",color="#3d7a37",weight="bold")
    box(0.1,1.2,1.6,1.1,"Phase 1 laptop\nmodels: FP16,\nRTN INT8, RTN INT4\n(NNCF schemes)","#dbe9f6",7.5)
    box(2.15,1.2,1.75,1.1,"Frozen PyTorch base,\nfake-quantized to\nbit-match the\nlaptop weights","#ffffff",7.5)
    box(4.3,1.65,1.75,0.95,"Train LoRA ×3 seeds\nr=16, α=32, q/k/v/o\n5,000 Dolly examples","#e3f2e1",7.5)
    box(4.3,0.35,1.75,0.95,"No adapter\n(same base,\nsame route)","#eeeeee",7.5)
    box(6.45,1.65,1.4,0.95,"Adapter kept\nseparate\n(not merged)","#fff3c4",7.5)
    box(8.25,0.9,1.55,1.2,"lm-eval, zero-shot,\n6 tasks\n(HF backend)","#ece4f5",7.5)
    arr(1.7,1.75,2.15,1.75); arr(3.9,1.95,4.3,2.1); arr(3.9,1.55,4.3,0.85)
    arr(6.05,2.1,6.45,2.1); arr(7.85,2.1,8.25,1.7); arr(6.05,0.82,8.25,1.2)
    plt.tight_layout(); plt.savefig(G+"fig_phase2_method.png",dpi=200); plt.close()
    # Fig: Phase 2 results, 6-task average with and without LoRA
    import numpy as np
    fig,ax=plt.subplots(figsize=(7,3.6)); x=np.arange(len(P2_BASES)); w=0.36
    no=[P2_NO[b][6] for b in P2_BASES]; lo=[P2_LORA[b][6][0] for b in P2_BASES]; sd=[P2_LORA[b][6][1] for b in P2_BASES]
    b1=ax.bar(x-w/2,no,w,color="#b8c7dc",ec="#333",lw=0.6,label="No LoRA (single evaluation)")
    b2=ax.bar(x+w/2,lo,w,yerr=sd,capsize=4,color="#4a7fb5",ec="#333",lw=0.6,error_kw=dict(lw=1),label="LoRA (mean ± std, 3 seeds)")
    ax.axhline(P2_NO["FP16"][6],ls="--",color="#c0504d",lw=1.1,label=f"FP16 without LoRA ({P2_NO['FP16'][6]:.2f})")
    for r,v in zip(b1,no): ax.text(r.get_x()+r.get_width()/2,v+0.12,f"{v:.2f}",ha="center",va="bottom",fontsize=7.5)
    for r,v,s in zip(b2,lo,sd): ax.text(r.get_x()+r.get_width()/2,v+s+0.12,f"{v:.2f}",ha="center",va="bottom",fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(P2_BASES,fontsize=9); ax.set_ylim(54,63)
    ax.set_ylabel("Average accuracy, 6 tasks (%)",fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5,loc="lower left",frameon=True)
    ax.text(0.99,0.02,"y-axis starts at 54%",transform=ax.transAxes,ha="right",fontsize=6.5,color="#666")
    plt.tight_layout(); plt.savefig(G+"fig_phase2.png",dpi=200); plt.close()''')
repl_line('    ax.text(4.2,0.75,"NNCF 3.4.0 weight compression', '    ax.text(4.2,0.75,"NNCF 3.4.0 weight compression\\nINT8: symmetric, per channel; INT4: symmetric, group size 128",fontsize=7,color="#555")')

# ---- title status / abstract ----
repl_line('p=para("[[Author information withheld]]', 'p=para("[[Author information withheld]] · Draft, November 2026 · *Phase 1 (compression benchmark) complete; Phase 2 (LoRA) complete for FP16, RTN INT8 and RTN INT4; LoRA on GPTQ and AWQ pending.*",align=WD_ALIGN_PARAGRAPH.CENTER)')
repl_line('para("**What we did.**', 'para("**What we did.** This is a controlled benchmark, not a new method. We compress Llama-3.2-1B with OpenVINO NNCF 3.4.0 weight compression into four variants, round-to-nearest (RTN) INT8 and RTN, GPTQ and AWQ INT4 (group size 128, one shared calibration set), and compare them with the FP16 reference. Accuracy is measured zero-shot with lm-evaluation-harness on six tasks and WikiText-2 perplexity; latency, throughput and peak memory are measured with OpenVINO GenAI on the integrated Arc GPU of an Intel Core Ultra 7 laptop that shares 16 GB of system RAM. In a second phase, we train the same LoRA adapter recipe (three seeds) on FP16, RTN INT8 and RTN INT4 bases on a Colab T4 GPU, using frozen PyTorch copies whose weights bit-match the laptop models, and evaluate with and without the adapter on the same six tasks. LoRA on the GPTQ and AWQ bases has not yet been run.")')
repl_line('para("**Takeaway.**', '''para("**Key results (Phase 2).** QLoRA {qlora} already showed that LoRA on a 4-bit base recovers much of the lost accuracy; we check this under controlled conditions on a 1B model. On the six-task average (Colab evaluation), RTN INT8 shows no measurable loss (60.35% vs 60.41% for FP16). RTN INT4 loses 4.04 points (56.36%); with LoRA it reaches 60.36 ± 0.07%, about the level of FP16 without LoRA. Because the same LoRA recipe also raises FP16 by 1.05 points, the fair comparison is with FP16 + LoRA: there, the INT4 gap shrinks from 4.04 to 1.10 points (about 73% closed), and a held-out training-loss gap remains.")
para("**Takeaway.** On this laptop, INT8 is a low-risk default, and AWQ was the most accurate of the three INT4 methods at an identical footprint. A short, cheap LoRA run closes most, but not all, of the RTN INT4 accuracy gap. Whether recovery differs for GPTQ and AWQ bases is still open.")''')

# ---- intro ----
repl_line(' "**RQ3.**', ' "**RQ3.** Does a fixed LoRA recipe recover accuracy differently depending on the compression applied to the frozen base model? (Phase 2: answered here for RTN INT8 and RTN INT4; GPTQ and AWQ bases pending.)",')
repl_line('para("**Contribution.**', 'para("**Contribution.** This paper is a controlled benchmark rather than a new method. On a single 1B model, we provide a matched comparison of RTN, GPTQ and AWQ that pairs zero-shot accuracy with measured on-device efficiency on a consumer Intel laptop (OpenVINO/NNCF), and we report a reproducible robustness failure of one GPTQ-compressed model when inputs lack a BOS token. Phase 2 measures how much a fixed LoRA recipe recovers on bases whose weights bit-match the laptop models, with FP16 + LoRA as the control; it does not test whether LoRA can recover quantization loss in general, which QLoRA {qlora} has already shown. We do not claim this is the first such comparison; our literature review covers the closest work we identified (Section 3) rather than the full literature.")')
repl_line('para("Section 2 gives background,', 'para("Section 2 gives background, Section 3 reviews related work, Section 4 describes the method, Section 5 reports results (Phase 1 in Sections 5.1–5.4, Phase 2 in Section 5.5), and Sections 6–8 discuss, state limitations and conclude.")')

# ---- method ----
repl_line('para("The study has two phases.', 'para("The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. Phase 2 (LoRA recovery) ran on Google Colab for the FP16, RTN INT8 and RTN INT4 bases; its settings were fixed before training. LoRA on the GPTQ and AWQ bases is pending.")')
lines[:]=[l.replace("and compressed with NNCF 3.4.0 {nncfdoc}. LoRA training (Section 4.3) will run on Google Colab with an NVIDIA GPU.","and compressed with NNCF 3.4.0 {nncfdoc} (OpenVINO 2026.4.1, exported through optimum-intel 2.2.0). LoRA training (Section 4.3) ran on Google Colab with an NVIDIA T4 GPU.") for l in lines]
assert any("optimum-intel 2.2.0). LoRA" in l for l in lines)
repl_line('para("Table 1 lists the five versions.', 'para("Table 1 lists the five versions. All compressed versions use NNCF weight-only compression. RTN INT8 uses symmetric quantization with one scale per output channel (NNCF mode INT8_SYM) for every weight matrix, including the embedding. The INT4 versions use symmetric, group-wise quantization with group size 128 (INT4_SYM, ratio 1.0) for all decoder linear layers, while the embedding and the tied output layer (lm_head) are kept at a backup precision of INT8_ASYM, one scale and zero point per channel. GPTQ and AWQ use the same 128 calibration samples drawn once from WikiText-2 {wikitext}. Optional NNCF refinements (e.g., scale estimation) were disabled for all methods so that each algorithm is evaluated in isolation, and the same backup-precision setting was used for all INT4 versions, which have identical on-disk sizes. GPTQ and AWQ were not evaluated at INT8 because prior work reports near-lossless 8-bit quantization across methods {huang, evalq}; RTN INT8 serves as the near-lossless reference point. Calibration data, symmetry and scale estimation were varied only in the diagnostic ablations of Section 5.4.")')
repl_line(' ["2","RTN INT8","Round-to-nearest","INT8","—"', ' ["2","RTN INT8","Round-to-nearest","INT8, per channel","—","Near-lossless reference point"],')
lines[:]=[l.replace('cap="Model versions. All compressed versions: NNCF 3.4.0 weight-only compression; INT4 versions are symmetric with group size 128 (g128).")','cap="Model versions. All compressed versions: NNCF 3.4.0 weight-only compression. INT8: symmetric, per output channel (INT8_SYM), all weights including the embedding. INT4: symmetric with group size 128 (g128, INT4_SYM, ratio 1.0); embedding and tied lm_head at INT8_ASYM per channel.")') for l in lines]
assert any("INT8_SYM), all weights" in l for l in lines)
repl_line('H2("4.3 LoRA training on Colab', 'H2("4.3 LoRA training on Colab (Phase 2)")')
idx=[i for i,l in enumerate(lines) if l.startswith('para("For each of the five versions, we will construct')]; assert len(idx)==1
lines[idx[0]:idx[0]+3]=r'''para("**Bases.** LoRA needs a PyTorch model on an NVIDIA GPU, so for FP16, RTN INT8 and RTN INT4 we built frozen PyTorch copies on Colab. For the quantized bases, each weight is quantized and immediately dequantized (integer × scale) with the same scheme as the laptop model (Section 4.2), and scales are rounded to FP16 as OpenVINO stores them. The resulting integer codes and scales bit-match those in the laptop models. This is the usual QLoRA-style setup {qlora}: the quantized weights stay frozen, they are stored in FP16 on the GPU (fake quantization), and no integer kernels are used. As a further check, the Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Section 5.5).")
para("**LoRA recipe.** Adapters were trained with PEFT {peft}: rank *r* = 16, α = 32, dropout 0.05, on the query, key, value and output projections of every attention layer. Adapters are kept separate and never merged into the base. The recipe is identical for every base; only the frozen base differs. FP16 + LoRA is the control that separates recovery of quantization loss from the general effect of fine-tuning on Dolly.")
para("**Data and training.** We used databricks-dolly-15k {dolly} with an Alpaca-style prompt (instruction, optional context, response). Examples longer than 512 tokens were dropped (not truncated); the rest were shuffled with data seed 1234, giving 200 held-out and 5,000 training examples. Every run used the same 5,000 examples, and loss was computed on response tokens only. Training ran for one epoch (about 310 optimizer steps) with AdamW, learning rate 2 × 10⁻⁴, a cosine schedule with 3% warm-up, an effective batch of 16 (4 per step × 4 accumulation steps), gradient clipping at 1.0, gradient checkpointing and FP16 autocast, with LoRA weights kept in FP32. Each base was trained with seeds 0, 1 and 2 (9 runs); the seed changes only the adapter initialization, dropout and batch order.")
para("**Evaluation.** All Phase 2 models, with and without adapters, were evaluated on Colab with lm-evaluation-harness 0.4.x {lmeval} (Hugging Face backend), zero-shot, with BOS, on the full sets of the six tasks and with the same metrics as Phase 1 (Section 4.4). The no-LoRA bases were re-evaluated on Colab rather than taken from Phase 1, so every with/without comparison uses the same software. Because the runtime and numerics differ from OpenVINO, Phase 2 scores are compared only with each other, not with the Phase 1 averages. We did not run adapters on the laptop or merge them and re-compress, so Phase 2 reports accuracy only, not speed or memory.")
figure(G+"fig_phase2_method.png",6.3,"Phase 2 as run: frozen bases that bit-match the laptop models, LoRA trained with three seeds on a Colab T4, and every base evaluated with and without its adapter by the same route.")'''.split("\n")
repl_line('para("All laptop measurements used the same machine,', 'para("All laptop measurements used the same machine, plugged in, in the same power mode, on the Arc GPU, with batch size 1, the same prompts, identical software versions and no other workloads. Compression settings and calibration data are matched across methods (Section 4.2), and the LoRA recipe, data and seeds are matched across bases (Section 4.3).")')
lines[:]=[l.replace("LoRA results will be reported as the mean across three seeds with the seed range; effects smaller than the seed-to-seed spread will be treated as noise.","LoRA results are reported as the mean ± standard deviation across three seeds; effects smaller than the seed-to-seed spread are treated as noise. The seed spread reflects training randomness only, not the sampling error of the test questions.") for l in lines]
assert any("training randomness only" in l for l in lines)
# reproducibility table
repl_line('para("We will release scripts,', 'para("We will release scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs [[TBD: repository link]]. Table 4 records versions and fixed settings.")')
idx=[i for i,l in enumerate(lines) if l.startswith(' ["Python","[TBD]"]')]; assert len(idx)==1
lines[idx[0]:idx[0]+4]=r''' ["Python (laptop / Colab)","[TBD] / 3.13"],["OpenVINO / OpenVINO GenAI","2026.4.1 / [TBD]"],["NNCF","3.4.0"],
 ["optimum / optimum-intel / transformers","2.3.0 / 2.2.0 / [TBD]"],["lm-evaluation-harness (laptop / Colab)","[TBD] (OpenVINO backend) / 0.4.x (Hugging Face backend)"],["PyTorch / PEFT (Colab)","[TBD] / [TBD]"],
 ["Intel Arc GPU driver / Windows build","[TBD] / [TBD]"],["Colab GPU","NVIDIA T4 (all 9 runs)"],
 ["LoRA","r = 16, α = 32, dropout 0.05; q/k/v/o projections; not merged"],
 ["Training","1 epoch; AdamW, lr 2 × 10⁻⁴, cosine, 3% warm-up; batch 4 × 4 accumulation = 16; max length 512 (longer dropped); FP16 autocast; gradient checkpointing"],
 ["Dolly subset; seeds","5,000 train + 200 held out, data seed 1234; training seeds 0, 1, 2"],'''.split("\n")
lines[:]=[l.replace('cap="Software versions and fixed settings (remaining entries to be completed).")','cap="Software versions and fixed settings ([TBD] entries to be completed).")') for l in lines]

# ---- results intro ----
repl_line('para("This section reports Phase 1.', 'para("Sections 5.1–5.4 report Phase 1 (laptop, OpenVINO). Section 5.5 reports Phase 2 (LoRA, Colab) for FP16, RTN INT8 and RTN INT4.")')

# ---- 5.5 ----
idx=[i for i,l in enumerate(lines) if l.startswith('H2("5.5 LoRA recovery')]; assert len(idx)==1
end=[i for i,l in enumerate(lines) if l.startswith('table(rows,[0.95,0.62,0.55,0.55,0.6,0.62,0.5,0.55,0.95,0.5]')]; assert len(end)==1
lines[idx[0]:end[0]+1]=r'''H2("5.5 LoRA recovery (Phase 2)")
para("Table 10 gives Phase 2 accuracy on Colab. All numbers in this section are percentages and come from the same Colab evaluation, so they can be compared with each other but not with the Phase 1 tables. As a check on the fake-quantized bases, the Colab no-LoRA scores agree with the laptop scores (Tables 5 and 6) within 0.6 points on every task for all three bases.")
fmt=lambda v: f"{v:.2f}"
rows=[["Base","LoRA"]+P2_TASKS+["Avg. (6 tasks)"]]
for b in P2_BASES:
    rows.append([b,"no"]+[fmt(v) for v in P2_NO[b]])
    rows.append([b,"yes"]+[f"{m:.2f} ± {s:.2f}" for m,s in P2_LORA[b]])
    rows.append(["","*Δ*"]+["*"+d+"*" for d in P2_DELTA[b]])
table(rows,[0.8,0.45,0.72,0.72,0.72,0.72,0.72,0.72,0.82],cap="Phase 2 zero-shot accuracy (%) on Colab (lm-evaluation-harness, Hugging Face backend). “yes” rows are mean ± standard deviation over seeds 0, 1 and 2; “no” rows are a single evaluation (log-likelihood scoring is deterministic). Δ = LoRA minus no LoRA, computed before rounding. ARC-C, ARC-E, HellaSwag and PIQA: acc_norm; WinoGrande and BoolQ: acc.",fs=8)
para("**INT8.** Without LoRA, RTN INT8 is 0.06 points below FP16 on the six-task average (60.35 vs 60.41), which is within noise. With LoRA the two are identical (61.46). Because there is no measurable gap, we do not report a recovery percentage for INT8; LoRA simply raises INT8 by the same amount as FP16 (+1.10 vs +1.05).")
para("**INT4.** RTN INT4 loses 4.04 points without LoRA (56.36 vs 60.41). With LoRA it reaches 60.36 ± 0.07, about the level of plain FP16. That comparison flatters INT4, however, because the same recipe also improves FP16 by 1.05 points, so part of the INT4 gain is the general benefit of Dolly fine-tuning. The fair comparison is with FP16 + LoRA (Table 11): the INT4 gap shrinks from 4.04 to 1.10 points, so LoRA closes about 73% of it. The remaining gap is about six times the largest seed standard deviation, and INT4 + LoRA is still below FP16 + LoRA on all six tasks (by 0.4 to 2.0 points).")
table([
 ["Base","Gap to FP16, no LoRA (points)","Gap to FP16 + LoRA, with LoRA (points)","Share of gap closed"],
 ["RTN INT8","0.06","0.00","not meaningful (no measurable gap)"],
 ["RTN INT4","4.04","1.10","≈ 73%"],
],[1.1,1.7,1.9,1.8],cap="Six-task average gap to FP16, comparing like with like (no LoRA vs no LoRA; LoRA vs LoRA). Gaps computed before rounding.",fs=8.5)
figure(G+"fig_phase2.png",5.6,"Six-task average accuracy without and with LoRA for each base (Colab evaluation). Error bars: standard deviation over three seeds. Dashed line: FP16 without LoRA. The y-axis starts at 54%.")
para("**Per task.** The largest INT4 gain is on BoolQ (+9.62 points, from 55.60 to 65.22). Without LoRA, INT4 was below BoolQ’s majority-class rate of about 62%, so part of this gain may come from LoRA restoring the model’s yes/no answer calibration or format rather than restoring knowledge. This is a hypothesis we have not tested; one check would be to compare the share of “yes” answers before and after LoRA. The other five tasks gain between +1.2 (WinoGrande) and +5.0 (ARC-Easy) points, so the recovery is not limited to BoolQ.")
table([
 ["Base","Train time per run (s)","Peak GPU memory (GB)","Mean train loss","Held-out Dolly loss"],
]+[[b,P2_TRAIN[b][0],P2_TRAIN[b][1],f"{sum(P2_TRAIN[b][2])/3:.2f}",f"{sum(P2_TRAIN[b][3])/3:.3f} ({min(P2_TRAIN[b][3]):.3f}–{max(P2_TRAIN[b][3]):.3f})"] for b in P2_BASES],
 [1.1,1.3,1.3,1.2,1.6],cap="Phase 2 training (three seeds per base, Colab T4). Held-out loss is on the 200 held-out Dolly examples, mean (range) over seeds. Peak GPU memory is the same for every base because the fake-quantized weights are stored in FP16; it is not deployment memory (see Table 7 for measured laptop memory).",fs=8.5)
para("**Training loss and cost.** The held-out loss tells the same story as accuracy (Table 12). FP16 and INT8 reach the same held-out loss (1.695 and 1.696), while INT4 stays higher (1.768). LoRA therefore does not fully undo INT4 rounding even on text like its own training data. Each run took about 11 minutes and 5.84 GB of GPU memory for every base. The equal cost is expected in our setup, which stores every base in FP16; with true 4-bit storage, as in QLoRA {qlora}, the INT4 base would need less memory.")
para("**Seed variation.** Across seeds, the six-task average varies by at most 0.25 points (standard deviation), and per-task standard deviations are at most 0.53 points. The differences discussed above are larger than this training noise. The seed spread does not include the sampling error of the test questions, which a paired bootstrap would add (not yet run).")'''.split("\n")

# ---- discussion ----
repl_line('para("**RQ3 (LoRA).**', 'para("**RQ3 (LoRA).** For RTN bases, the answer so far is that LoRA helps most where quantization hurt most. INT8 had nothing to recover and gained the same as FP16. INT4 gained about four times as much, closing about 73% of its gap to FP16 + LoRA. This agrees with QLoRA {qlora}, which already showed that LoRA on a 4-bit base recovers much of the lost accuracy; our addition is a controlled measurement on a 1B model with bases that bit-match a laptop deployment. It contrasts with Huang et al. {huang}, who found that general instruction data did not restore zero-shot accuracy for LLaMA3-8B; differences in model size, quantizer, recipe and tasks could explain this, and we did not test which. Two cautions apply. First, comparing INT4 + LoRA with plain FP16 makes recovery look complete, which it is not. Second, the large BoolQ gain may partly reflect answer calibration rather than knowledge. Whether GPTQ and AWQ bases recover differently, the main question behind RQ3, awaits their LoRA runs.")')
repl_line('para("**Practical guidance.**', 'para("**Practical guidance.** For this model and laptop, INT8 is a low-risk choice that yields 1.7× decode throughput and a third less peak memory at no detectable accuracy cost. When memory or speed requires INT4, AWQ gave the best accuracy of the methods tested at no cost in size or speed relative to RTN. If fine-tuning is planned anyway, a short LoRA run on an RTN INT4 base brings it close to, but still about one point below, FP16 with the same adapter. We did not measure the speed or memory of adapters on the laptop. These recommendations rest on one model, one machine and one toolchain version.")')
lines[:]=[l.replace(' ["H7a, H7b, H8","Pending (Phase 2)."],',' ["H7a","Supported for RTN bases: INT4 gained +3.99 vs +1.05 (FP16) and +1.10 (INT8), and its gap to FP16 + LoRA fell from 4.04 to 1.10 points. GPTQ and AWQ pending."],\n ["H7b","Not supported for RTN bases: LoRA on Dolly raised zero-shot accuracy on all six tasks for all three bases."],\n ["H8","Not tested: adapters were evaluated unmerged in PyTorch only (no merge-and-re-compress)."],') for l in lines]
assert any('["H7b","Not supported' in l for l in lines)
lines[:]=[l.replace('cap="Status of the pre-specified hypotheses after Phase 1.")','cap="Status of the pre-specified hypotheses after Phase 1 and the RTN part of Phase 2.")').replace(' ["#","Phase 1 outcome"],',' ["#","Outcome"],') for l in lines]

# ---- limitations ----
repl_line('para("**LoRA design (Phase 2).**', 'para("**LoRA design (Phase 2).** We tested one adapter configuration (r = 16, attention projections only), one general-purpose dataset (5,000 Dolly examples, one epoch) and three seeds; other ranks, target layers, data or training lengths could change how much is recovered. Training used fake quantization in PyTorch, not true integer kernels: the weights match the laptop models exactly, but the arithmetic during training and evaluation is FP16, so Phase 2 accuracy is not a measurement of the laptop runtime (the no-LoRA scores agree within 0.6 points per task). Adapters were not run on the laptop or merged and re-compressed, so their speed and memory cost and any loss from merging are unmeasured. LoRA has not yet been run on the GPTQ and AWQ bases, so whether recovery depends on the compression method is still open.")')

# ---- conclusion ----
lines[:]=[l.replace("Phase 2, which tests whether a fixed LoRA recipe (r = 16, α = 32, dropout 0.05, q/k/v/o, Dolly-15k subset, three seeds) recovers accuracy differently across compression methods, is pending.",'In Phase 2, a fixed LoRA recipe (r = 16, α = 32, q/k/v/o, 5,000 Dolly examples, three seeds) trained on bases that bit-match the laptop models confirmed, at 1B scale, what QLoRA {qlora} showed for 4-bit models: INT8 had no measurable loss to recover, and RTN INT4 + LoRA (60.36%) reached about the six-task average of plain FP16 (60.41%). Against FP16 + LoRA, the fair control, INT4 closed about 73% of its 4.04-point gap and remained 1.10 points behind, with a higher held-out loss. The large BoolQ gain may partly reflect answer calibration rather than knowledge.') for l in lines]
assert any("fair control, INT4 closed" in l for l in lines)
lines[:]=[l.replace("Future work, after the INT4 grid and Phase 2 are complete:","Next, LoRA on the GPTQ and AWQ bases will test whether recovery depends on the compression method, and running adapters on the laptop will measure their speed and memory cost. Future work after that:") for l in lines]
assert any("Next, LoRA on the GPTQ" in l for l in lines)

# ---- appendix ----
lines[:]=[l.replace('"C. Full LoRA hyperparameters and training curves for all 15 runs [[TBD]]"','"C. LoRA training curves for the 9 completed runs (GPTQ and AWQ runs pending) [[TBD]]"').replace('"D. Evaluation-route log for every Phase 2 configuration [[TBD]]"','"D. Phase 2 base checks: fake-quantization code/scale comparison and Colab vs laptop scores per task [[TBD]]"') for l in lines]
assert any('9 completed runs' in l for l in lines)
open(P,"w").write("\n".join(lines))
print("patched")
