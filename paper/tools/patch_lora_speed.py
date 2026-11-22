"""Add laptop LoRA speed/memory results to the paper (Section 5.6 and related updates)."""
P="/workspace/paper/build_paper.py"
src=open(P).read()
lines=src.split("\n")
def repl_line(prefix,new):
    idx=[i for i,l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx)==1,(prefix,idx)
    lines[idx[0]]=new
def insert_after(prefix,new):
    idx=[i for i,l in enumerate(lines) if l.startswith(prefix)]; assert len(idx)==1,prefix
    lines[idx[0]+1:idx[0]+1]=new.split("\n")

# ---- data ----
insert_after(' "RTN INT4":("652–654"', r'''
# ---------- Laptop LoRA speed/memory (OpenVINO GenAI 2026.4.1, Arc iGPU; seed-0 adapter, unmerged, scale 2.0) ----------
# Means of kept balanced-interleaved rounds. Memory: peak commit (private bytes) and peak working set, MB.
# FP16 no-LoRA: 2 rounds (r3 dropped, CPU 59%). Others: 3 rounds.
P2_SPEED={  # tok/s, TTFT ms, peak commit MB, peak working set MB
 "FP16":    {"off":(24.9,41.8,6253,5578),"on":(22.9,44.3,6287,5595)},
 "RTN INT8":{"off":(41.2,25.6,4258,3590),"on":(35.0,29.6,4280,3613)},
 "RTN INT4":{"off":(62.8,17.5,3097,2433),"on":(51.4,22.2,3108,2452)}}
P2_SPEED_ROUNDS={  # raw tok/s per kept round
 "FP16":    {"off":[24.98,24.81],"on":[22.86,22.86,23.06]},
 "RTN INT8":{"off":[41.47,41.30,40.82],"on":[32.73,36.11,36.03]},
 "RTN INT4":{"off":[63.68,60.75,63.94],"on":[52.89,49.34,52.09]}}
P2_SPEED_DELTA={"FP16":"−8%","RTN INT8":"−15%","RTN INT4":"−18%"}''')
assert any("P2_SPEED={" in l for l in lines)

# ---- figure: accuracy vs tok/s scatter for the six cells ----
insert_after('    plt.tight_layout(); plt.savefig(G+"fig_phase2.png",dpi=200); plt.close()', r'''
    # Fig: Phase 2 accuracy vs laptop decode throughput (unmerged LoRA)
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    cols={"FP16":"#4a7fb5","RTN INT8":"#6aaa64","RTN INT4":"#2e7d32"}
    mks={"FP16":"o","RTN INT8":"s","RTN INT4":"D"}
    laboff={"FP16":{"off":(-46,-14),"on":(-58,8)},
            "RTN INT8":{"off":(8,-14),"on":(-4,8)},
            "RTN INT4":{"off":(8,-14),"on":(8,6)}}
    labl={"FP16":{"off":"FP16","on":"FP16 + LoRA"},
          "RTN INT8":{"off":"INT8","on":"INT8 + LoRA"},
          "RTN INT4":{"off":"INT4","on":"INT4 + LoRA"}}
    for b in P2_BASES:
        x0,y0=P2_SPEED[b]["off"][0],P2_NO[b][6]
        x1,y1=P2_SPEED[b]["on"][0],P2_LORA[b][6][0]
        c,mk=cols[b],mks[b]
        ax.plot([x0,x1],[y0,y1],color=c,lw=1.1,zorder=2)
        ax.scatter(x0,y0,facecolors="white",edgecolors=c,marker=mk,s=80,lw=1.5,zorder=3)
        ax.scatter(x1,y1,color=c,marker=mk,s=80,ec="#333",lw=0.5,zorder=3)
        ax.annotate(labl[b]["off"],(x0,y0),xytext=laboff[b]["off"],textcoords="offset points",fontsize=7.5,color=c)
        ax.annotate(labl[b]["on"],(x1,y1),xytext=laboff[b]["on"],textcoords="offset points",fontsize=7.5,color=c)
    ax.scatter([],[],facecolors="white",edgecolors="#333",marker="o",s=55,lw=1.3,label="No LoRA")
    ax.scatter([],[],color="#555",marker="o",s=55,ec="#333",lw=0.5,label="LoRA (seed 0, unmerged)")
    ax.set_xlabel("Decode throughput (tokens/s)",fontsize=8.5)
    ax.set_ylabel("Average accuracy, 6 tasks (%)",fontsize=8.5)
    ax.set_xlim(16,72); ax.set_ylim(55.2,62.6)
    ax.tick_params(labelsize=8); ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5,loc="lower left",frameon=False)
    plt.tight_layout(); plt.savefig(G+"fig_lora_tradeoff.png",dpi=200); plt.close()''')
assert any("fig_lora_tradeoff.png" in l for l in lines)

# ---- title status line ----
repl_line('p=para("[[Author information withheld]]',
         'p=para("[[Author information withheld]] · Draft, November 2026 · *Phase 1 (compression benchmark) complete; Phase 2 (LoRA accuracy and laptop efficiency) complete for FP16, RTN INT8 and RTN INT4; LoRA on GPTQ and AWQ pending.*",align=WD_ALIGN_PARAGRAPH.CENTER)')

# ---- abstract ----
repl_line('para("**What we did.**',
         'para("**What we did.** This is a controlled benchmark, not a new method. We compress Llama-3.2-1B with OpenVINO NNCF 3.4.0 weight compression into four variants, round-to-nearest (RTN) INT8 and RTN, GPTQ and AWQ INT4 (group size 128, one shared calibration set), and compare them with the FP16 reference. Accuracy is measured zero-shot with lm-evaluation-harness on six tasks and WikiText-2 perplexity; latency, throughput and peak memory are measured with OpenVINO GenAI on the integrated Arc GPU of an Intel Core Ultra 7 laptop that shares 16 GB of system RAM. In a second phase, we train the same LoRA adapter recipe (three seeds) on FP16, RTN INT8 and RTN INT4 bases on a Colab T4 GPU, using frozen PyTorch copies whose weights bit-match the laptop models, and evaluate with and without the adapter on the same six tasks. We then measure the seed-0 adapter of each base, left unmerged, on the same laptop and with the same OpenVINO GenAI timing protocol as Phase 1. LoRA on the GPTQ and AWQ bases has not yet been run.")')
repl_line('para("**Key results (Phase 2).**',
         'para("**Key results (Phase 2).** QLoRA {qlora} already showed that LoRA on a 4-bit base recovers much of the lost accuracy; we check this under controlled conditions on a 1B model. On the six-task average (Colab evaluation), RTN INT8 shows no measurable loss (60.35% vs 60.41% for FP16). RTN INT4 loses 4.04 points (56.36%); with LoRA it reaches 60.36 ± 0.07%, about the level of FP16 without LoRA. Because the same LoRA recipe also raises FP16 by 1.05 points, the fair comparison is with FP16 + LoRA: there, the INT4 gap shrinks from 4.04 to 1.10 points (about 73% closed), and a held-out training-loss gap remains. Unmerged LoRA added only 11–34 MB of peak memory on the laptop but reduced decode throughput by 8% (FP16), 15% (INT8) and 18% (INT4). Even so, INT4 + LoRA remained about 2.2× as fast as FP16 + LoRA at about half the memory, with accuracy 1.10 points lower and about equal to plain FP16.")')
repl_line('para("**Takeaway.**',
         'para("**Takeaway.** On this laptop, INT8 is a low-risk default, and AWQ was the most accurate of the three INT4 methods at an identical footprint. A short, cheap LoRA run closes most, but not all, of the RTN INT4 accuracy gap, at a modest speed cost and almost no extra memory. Whether recovery differs for GPTQ and AWQ bases is still open.")')

# ---- intro roadmap and contribution ----
repl_line('para("**Contribution.**',
         'para("**Contribution.** This paper is a controlled benchmark rather than a new method. On a single 1B model, we provide a matched comparison of RTN, GPTQ and AWQ that pairs zero-shot accuracy with measured on-device efficiency on a consumer Intel laptop (OpenVINO/NNCF), and we report a reproducible robustness failure of one GPTQ-compressed model when inputs lack a BOS token. Phase 2 measures how much a fixed LoRA recipe recovers on bases whose weights bit-match the laptop models, with FP16 + LoRA as the control, and reports the speed and memory of the unmerged adapters on the same laptop; it does not test whether LoRA can recover quantization loss in general, which QLoRA {qlora} has already shown. We do not claim this is the first such comparison; our literature review covers the closest work we identified (Section 3) rather than the full literature.")')
repl_line('para("Section 2 gives background',
         'para("Section 2 gives background, Section 3 reviews related work, Section 4 describes the method, Section 5 reports results (Phase 1 in Sections 5.1–5.4, Phase 2 in Sections 5.5–5.6), and Sections 6–8 discuss, state limitations and conclude.")')

# ---- method opening, evaluation, laptop protocol, 4.4, table 4, results intro ----
repl_line('para("The study has two phases.',
         'para("The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. Phase 2 trained LoRA adapters on Google Colab for the FP16, RTN INT8 and RTN INT4 bases, with settings fixed before training, and then measured those unmerged adapters on the same laptop. LoRA on the GPTQ and AWQ bases is pending.")')
repl_line('para("**Evaluation.** All Phase 2',
         'para("**Evaluation.** All Phase 2 models, with and without adapters, were evaluated on Colab with lm-evaluation-harness 0.4.x {lmeval} (Hugging Face backend), zero-shot, with BOS, on the full sets of the six tasks and with the same metrics as Phase 1 (Section 4.4). The no-LoRA bases were re-evaluated on Colab rather than taken from Phase 1, so every with/without comparison uses the same software. Because the runtime and numerics differ from OpenVINO, Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages.")')
insert_after('para("**Evaluation.** All Phase 2',
         r'''para("**Laptop speed and memory.** Speed and memory of the unmerged adapters were measured on the same laptop as Phase 1, with OpenVINO GenAI 2026.4.1 (LLMPipeline) on the Arc GPU. Each seed-0 adapter was applied unmerged at runtime through openvino_genai.Adapter and AdapterConfig, with a scaling factor of 2.0, equal to the training ratio α / *r* = 32 / 16. We used the seed-0 adapter of each base because speed is not expected to depend on the seed. The prompt and timing protocol matched the Phase 1 speed test (Table 2): a 128-token prompt, 5 warm-up runs, then 30 timed runs per round, greedy decoding of 128 new tokens. For each base we used a balanced interleaved order (off, on, on, off, off, on), giving three rounds per cell. The reported memory figure is peak commit (private bytes), which is not affected by Windows trimming the working set; peak working set is also recorded. One FP16 no-LoRA round was excluded because system CPU load during it was 59% (versus about 8–10% in the other rounds). An earlier non-interleaved pass showed run-to-run drift and working-set trimming, which is why we adopted the balanced order and the commit-memory metric. Raw logs are under quant-study/speed-lora on the laptop.")''')
# The *r* above will be italicized by add_runs. That's OK (rank r). But "α / *r* = 32 / 16" is fine.
repl_line('para("Efficiency was measured on the laptop only',
         'para("Efficiency was measured on the laptop only (Table 2), with OpenVINO GenAI {genai} on the Arc GPU, including a later pass with unmerged LoRA adapters (Section 4.3).")')
# Fill OpenVINO GenAI version now that this pass used 2026.4.1
lines[:]=[l.replace('["OpenVINO / OpenVINO GenAI","2026.4.1 / [TBD]"]','["OpenVINO / OpenVINO GenAI","2026.4.1 / 2026.4.1"]') for l in lines]
assert any("2026.4.1 / 2026.4.1" in l for l in lines)
repl_line('para("Sections 5.1–5.4 report',
         'para("Sections 5.1–5.4 report Phase 1 (laptop, OpenVINO). Section 5.5 reports Phase 2 accuracy (LoRA, Colab) for FP16, RTN INT8 and RTN INT4. Section 5.6 reports the speed and memory of those unmerged adapters on the laptop.")')

# Table 12 caption: point also to the new efficiency table
lines[:]=[l.replace("see Table 7 for measured laptop memory).","see Table 7 for Phase 1 laptop memory and Table 13 for memory with LoRA).") for l in lines]
assert any("Table 13 for memory with LoRA" in l for l in lines)

# ---- Section 5.6 ----
insert_after('para("**Seed variation.**', r'''
H2("5.6 LoRA speed and memory on the laptop")
para("Table 13 puts the six cells in one place: Colab six-task accuracy (Table 10) and laptop speed and memory with the seed-0 adapter left unmerged. The timing protocol matched Phase 1 (Table 2), except that memory here is peak commit (private bytes) rather than peak working set, so the no-LoRA memory figures are not directly comparable with Table 7. The no-LoRA throughputs, however, agree with the Phase 1 values (25.0, 42.3 and 65.7 tokens/s) to within about 3–5%. Each cell is the mean of three interleaved rounds, except FP16 without LoRA, where one round was dropped because system CPU load during it was 59% (versus about 8–10% otherwise).")
rows=[["Base","LoRA","Avg. acc. (%)","Decode (tok/s)","vs no LoRA","TTFT (ms)","Peak mem. (MB)"]]
for b in P2_BASES:
    off,on=P2_SPEED[b]["off"],P2_SPEED[b]["on"]
    rows.append([b,"no",f"{P2_NO[b][6]:.2f}",f"{off[0]:.1f}","—",f"{off[1]:.1f}","{:,}".format(off[2])])
    rows.append([b,"yes",f"{P2_LORA[b][6][0]:.2f}",f"{on[0]:.1f}",P2_SPEED_DELTA[b],f"{on[1]:.1f}","{:,}".format(on[2])])
table(rows,[0.95,0.5,1.0,1.05,0.85,0.8,1.05],cap="Six-cell grid: Colab six-task average accuracy (Table 10) and laptop efficiency with the seed-0 adapter applied unmerged (OpenVINO GenAI 2026.4.1, Arc GPU; same prompt and run counts as Table 7). LoRA accuracy is the three-seed mean from Table 10; speed and memory used only the seed-0 adapter. Decode and TTFT are means of the kept rounds (two for FP16 without LoRA; three otherwise). Peak memory is peak commit (private bytes), not the working-set metric of Table 7.",fs=8.5)
para("**Speed.** Unmerged LoRA slowed decoding by 8% for FP16 (24.9 to 22.9 tokens/s), 15% for INT8 (41.2 to 35.0) and 18% for INT4 (62.8 to 51.4). Time to first token rose by 2.5, 4.0 and 4.7 ms respectively. The relative cost therefore grew as the base became cheaper. A likely reason, which we did not measure directly, is that the adapter matrix multiplies run at higher precision and add a roughly fixed cost, which is a larger share of a faster model. Round-to-round spread was small for FP16 (kept no-LoRA rounds 24.98 and 24.81 tokens/s; LoRA 22.86–23.06). It was larger for INT8 with LoRA (32.73, 36.11 and 36.03) and for INT4 (no LoRA 60.75–63.94; with LoRA 49.34–52.89). One INT8 + LoRA round (32.73) was well below the other two (about 36); we did not drop it. In every kept round the six cells kept the same speed order: INT4, INT4 + LoRA, INT8, INT8 + LoRA, FP16, FP16 + LoRA.")
para("**Memory.** The adapter added little memory: 34 MB for FP16, 22 MB for INT8 and 11 MB for INT4 (peak commit). Peak working set showed the same pattern, rising by about 17, 23 and 19 MB (about 5,578 to 5,595; 3,590 to 3,613; 2,433 to 2,452). Phase 1 reported working set without LoRA of 5,579, 3,693 and 2,545 MB, so the absolute working-set levels in this pass differ somewhat for INT8 and INT4; we therefore treat Table 7 and Table 13 as separate measurements and do not mix the two memory metrics.")
para("**The trade-off.** Figure 5 plots accuracy against throughput for the six cells. INT4 + LoRA is still about 2.2× as fast as FP16 + LoRA (51.4 vs 22.9 tokens/s) and about 2.1× as fast as plain FP16, at about half the peak memory (3,108 vs 6,253–6,287 MB). Its six-task average (60.36%) is 1.10 points below FP16 + LoRA and about equal to plain FP16 (60.41%). The unmerged adapter therefore recovers most of the INT4 accuracy at a noticeable speed cost but almost no extra memory. Merging the adapter into the base and re-quantizing could in principle remove that speed cost, but it would change the INT4 weights and would require a new accuracy check; we did not test this (see Section 8).")
figure(G+"fig_lora_tradeoff.png",5.8,"Six-task average accuracy (Colab) versus decode throughput on the laptop for the six cells. Hollow markers: no LoRA. Filled markers: unmerged seed-0 LoRA. Lines connect the same base. Throughput is the mean of the kept rounds (Table 13).")
''')
assert any('H2("5.6 LoRA speed and memory on the laptop")' in l for l in lines)

# ---- discussion, hypotheses, limitations, conclusion ----
insert_after('para("**RQ3 (LoRA).**',
         r'''para("**LoRA on the laptop.** The unmerged adapter costs almost no memory (11–34 MB) but some speed, and the relative cost is larger for the cheaper bases (8%, 15% and 18% for FP16, INT8 and INT4). We take the most likely cause to be that the adapter’s extra matrix multiplies stay at higher precision and are a roughly fixed cost—an explanation, not a measurement. The practical result is that INT4 + LoRA remains more than twice as fast as FP16 + LoRA at about half the memory, while remaining 1.10 points below it in accuracy.")''')
repl_line('para("**Practical guidance.**',
         'para("**Practical guidance.** For this model and laptop, INT8 is a low-risk choice that yields 1.7× decode throughput and a third less peak memory at no detectable accuracy cost. When memory or speed requires INT4, AWQ gave the best accuracy of the methods tested at no cost in size or speed relative to RTN. If fine-tuning is planned anyway, a short LoRA run on an RTN INT4 base brings it close to, but still 1.10 points below, FP16 with the same adapter. On this laptop the unmerged adapter added only 11 MB of peak memory to INT4 and left it about 2.2× as fast as FP16 + LoRA (51.4 vs 22.9 tokens/s), though 18% slower than INT4 without LoRA. These recommendations rest on one model, one machine, one prompt length, one adapter rank and unmerged adapters only.")')
repl_line(' ["H8","Not tested:',
         ' ["H8","Not tested: adapters were run unmerged on the laptop; merging and re-compressing was not."],')
repl_line('],[1.0,5.5],cap="Status of the pre-specified hypotheses',
         '],[1.0,5.5],cap="Status of the pre-specified hypotheses after Phase 1, the RTN part of Phase 2, and the laptop LoRA speed measurements.")')
repl_line('para("**LoRA design (Phase 2).**',
         'para("**LoRA design (Phase 2).** We tested one adapter configuration (r = 16, attention projections only), one general-purpose dataset (5,000 Dolly examples, one epoch) and three seeds; other ranks, target layers, data or training lengths could change how much is recovered, and the laptop speed test used only the seed-0 adapter and this single rank. Training used fake quantization in PyTorch, not true integer kernels: the weights match the laptop models exactly, but the arithmetic during training and evaluation is FP16, so Phase 2 accuracy is not a measurement of the laptop runtime (the no-LoRA scores agree within 0.6 points per task). Speed and memory were measured with the adapter left unmerged, on the integrated Arc GPU only, and with a single 128-token prompt; a longer or shorter prompt, a discrete GPU, or a different runtime could change the speed cost. Merging the adapter and re-compressing was not tested, so any accuracy loss or speed gain from that route is unknown. LoRA has not yet been run on the GPTQ and AWQ bases, so whether recovery depends on the compression method is still open.")')
repl_line('para("We benchmarked RTN, GPTQ and AWQ',
         'para("We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 on an Intel Core Ultra 7 laptop with an integrated Arc GPU. RTN INT8 was nearly lossless (four-task average 0.559 vs 0.562) with 1.7× decode throughput and 34% lower peak memory. INT4 delivered about 2.6× throughput and 54–56% lower peak memory at a 2–4 point accuracy cost; AWQ recovered about half of RTN’s loss, while NNCF GPTQ did not improve on RTN. GPTQ INT4 also failed on inputs without BOS (perplexity 424), a reproducible defect localized to one column of one weight matrix. In Phase 2, a fixed LoRA recipe (r = 16, α = 32, q/k/v/o, 5,000 Dolly examples, three seeds) trained on bases that bit-match the laptop models confirmed, at 1B scale, what QLoRA {qlora} showed for 4-bit models: INT8 had no measurable loss to recover, and RTN INT4 + LoRA (60.36%) reached about the six-task average of plain FP16 (60.41%). Against FP16 + LoRA, the fair control, INT4 closed about 73% of its 4.04-point gap and remained 1.10 points behind, with a higher held-out loss. The large BoolQ gain may partly reflect answer calibration rather than knowledge. On the laptop, the unmerged adapter added only 11–34 MB of peak memory but reduced decode throughput by 8% (FP16), 15% (INT8) and 18% (INT4). INT4 + LoRA remained about 2.2× as fast as FP16 + LoRA at about half the memory.")')
repl_line('para("Next, LoRA on the GPTQ and AWQ bases',
         'para("Next, LoRA on the GPTQ and AWQ bases will test whether recovery depends on the compression method. Merging the adapters and re-quantizing, which might remove the unmerged speed cost but would change the INT4 weights, is untested. Future work after that: INT3 quantization, which NNCF supports from version 3.3.0 {nncf}; pruning, including SparseGPT and Wanda {sparsegpt, wanda}; mixed-precision formats; a test of whether the GPTQ defect occurs in other models or GPTQ implementations; additional models and sizes; task-aligned LoRA data; quantization-aware adapters (QA-LoRA, LoftQ); and other consumer hardware such as NPUs.")')

# ---- appendix placeholder for timing logs ----
lines[:]=[l.replace(
    '"F. GPTQ diagnostics: per-matrix and per-column error statistics and column-2558 activation traces [[TBD]]"',
    '"F. GPTQ diagnostics: per-matrix and per-column error statistics and column-2558 activation traces [[TBD]]",\n          "G. Laptop LoRA timing logs (balanced interleaved pass) [[TBD]]"'
) for l in lines]
assert any("Laptop LoRA timing logs" in l for l in lines)

# ---- sanity: forbidden leftover claims ----
joined="\n".join(lines)
for bad in [
    "Phase 2 reports accuracy only",
    "Phase 2 measures accuracy only",
    "We did not run adapters on the laptop",
    "We did not measure the speed or memory of adapters on the laptop",
    "running adapters on the laptop will measure",
    "their speed and memory cost and any loss from merging are unmeasured",
]:
    assert bad not in joined, bad

open(P,"w").write("\n".join(lines))
print("patched")
