import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from pathlib import Path as _Path
_HERE=_Path(__file__).resolve().parent
G=str(_HERE/"figs")+"/"  # figures regenerated below by make_figures()
REFS={
 "gptq":'Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv:2210.17323. https://arxiv.org/abs/2210.17323',
 "awq":'Lin, J., et al. (2023). AWQ: Activation-aware weight quantization for LLM compression and acceleration. arXiv:2306.00978. https://arxiv.org/abs/2306.00978',
 "sparsegpt":'Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive language models can be accurately pruned in one-shot. arXiv:2301.00774. https://arxiv.org/abs/2301.00774',
 "wanda":'Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. arXiv:2306.11695. https://arxiv.org/abs/2306.11695',
 "lora":'Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685',
 "qlora":'Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. arXiv:2305.14314. https://arxiv.org/abs/2305.14314',
 "huang":'Huang, W., et al. (2024). An empirical study of LLaMA3 quantization: From LLMs to MLLMs. arXiv:2404.14047. https://arxiv.org/abs/2404.14047',
 "llmc":'Gong, R., et al. (2024). LLMC: Benchmarking large language model quantization with a versatile compression toolkit. arXiv:2405.06001. https://arxiv.org/abs/2405.06001',
 "evalq":'Li, S., et al. (2024). Evaluating quantized large language models. arXiv:2402.18158. https://arxiv.org/abs/2402.18158',
 "zhou":'Zhou, Z., Kurz, S., & Zhao, Z. (2025). Revisiting pruning vs quantization for small language models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 12055–12070. https://aclanthology.org/2025.findings-emnlp.645/',
 "kick":'Jaiswal, A., et al. (2023). Compressing LLMs: The truth is rarely pure and never simple (LLM-KICK). arXiv:2310.01382. https://arxiv.org/abs/2310.01382',
 "loftq":'Li, Y., et al. (2023). LoftQ: LoRA-fine-tuning-aware quantization for large language models. arXiv:2310.08659. https://arxiv.org/abs/2310.08659',
 "irqlora":'Qin, H., et al. (2024). Accurate LoRA-finetuning quantization of LLMs via information retention (IR-QLoRA). arXiv:2402.05445. https://arxiv.org/abs/2402.05445',
 "qalora":'Xu, Y., et al. (2023). QA-LoRA: Quantization-aware low-rank adaptation of large language models. arXiv:2309.14717. https://arxiv.org/abs/2309.14717',
 "pruner":'Ma, X., Fang, G., & Wang, X. (2023). LLM-Pruner: On the structural pruning of large language models. arXiv:2305.11627. https://arxiv.org/abs/2305.11627',
 "llama":'Meta. Llama-3.2-1B model card. Hugging Face. https://huggingface.co/meta-llama/Llama-3.2-1B',
 "qwen":'Qwen Team. Qwen2.5-0.5B model card. Hugging Face. https://huggingface.co/Qwen/Qwen2.5-0.5B',
 "openvino":'Intel. OpenVINO documentation. https://docs.openvino.ai/',
 "nncfdoc":'Intel. LLM weight compression (OpenVINO / NNCF documentation). https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html',
 "nncf":'OpenVINO Toolkit. NNCF: Neural Network Compression Framework, release notes (INT3/INT2 modes added in 3.3.0). https://github.com/openvinotoolkit/nncf/blob/develop/ReleaseNotes.md',
 "genai":'OpenVINO Toolkit. OpenVINO GenAI: Text generation and LoRA adapters. https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-generation/',
 "lmeval":'EleutherAI. lm-evaluation-harness. GitHub. https://github.com/EleutherAI/lm-evaluation-harness',
 "peft":'Hugging Face. PEFT: Parameter-efficient fine-tuning. GitHub. https://github.com/huggingface/peft',
 "dolly":'Databricks. databricks-dolly-15k. Hugging Face dataset. https://huggingface.co/datasets/databricks/databricks-dolly-15k',
 "wikitext":'Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models (WikiText). arXiv:1609.07843. https://arxiv.org/abs/1609.07843',
 "arc":'Clark, P., et al. (2018). Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457. https://arxiv.org/abs/1803.05457',
 "hellaswag":'Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019). HellaSwag: Can a machine really finish your sentence? arXiv:1905.07830. https://arxiv.org/abs/1905.07830',
 "winogrande":'Sakaguchi, K., Le Bras, R., Bhagavatula, C., & Choi, Y. (2019). WinoGrande: An adversarial Winograd schema challenge at scale. arXiv:1907.10641. https://arxiv.org/abs/1907.10641',
}
order=[]
def cites(text):
    def rep(m):
        keys=[k.strip() for k in m.group(1).split(",")]
        nums=[]
        for k in keys:
            assert k in REFS,k
            if k not in order: order.append(k)
            nums.append(order.index(k)+1)
        return "["+", ".join(str(n) for n in sorted(nums))+"]"
    return re.sub(r"\{([a-z0-9_, ]+)\}",rep,text)

doc=Document()
sec=doc.sections[0]
for side in ("left_margin","right_margin","top_margin","bottom_margin"): setattr(sec,side,Inches(1))
st=doc.styles["Normal"]; st.font.name="Times New Roman"; st.font.size=Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"),"Times New Roman")
st.paragraph_format.space_after=Pt(6); st.paragraph_format.line_spacing=1.15
for name,size in (("Heading 1",14),("Heading 2",12),("Title",18)):
    h=doc.styles[name]; h.font.name="Times New Roman"; h.font.size=Pt(size); h.font.bold=True; h.font.color.rgb=RGBColor(0x1F,0x2A,0x44)
    h.element.rPr.rFonts.set(qn("w:asciiTheme"),"") if False else None
    rf=h.element.rPr.rFonts; rf.set(qn("w:ascii"),"Times New Roman"); rf.set(qn("w:hAnsi"),"Times New Roman")
    for a in ("w:asciiTheme","w:hAnsiTheme","w:eastAsiaTheme","w:cstheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]

def add_runs(p,text):
    # **bold**, *italic*, [TBD...] grey italic placeholders
    text=cites(text)
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|\[\[.+?\]\])",text):
        if not part: continue
        if part.startswith("**"): r=p.add_run(part[2:-2]); r.bold=True
        elif part.startswith("[["): r=p.add_run("["+part[2:-2]+"]"); r.italic=True; r.font.color.rgb=RGBColor(0x80,0x80,0x80)
        elif part.startswith("*"): r=p.add_run(part[1:-1]); r.italic=True
        else: p.add_run(part)
    return p
def para(text,style=None,align=None):
    p=doc.add_paragraph(style=style); add_runs(p,text)
    if align: p.alignment=align
    return p
def H1(t): doc.add_heading(t,level=1)
def H2(t): doc.add_heading(t,level=2)
def pre(text, fs=8.5):
    """Monospaced lines for templates and printed examples."""
    for line in text.split("\n"):
        p=doc.add_paragraph()
        r=p.add_run(line if line else "\u00a0")
        r.font.name="Courier New"
        r._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
        r.font.size=Pt(fs)
        p.paragraph_format.space_after=Pt(0)
        p.paragraph_format.space_before=Pt(0)
        p.paragraph_format.line_spacing=1.05
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
def bullets(items,style="List Bullet"):
    for i in items: add_runs(doc.add_paragraph(style=style),i)
def shade(cell,hexc):
    tcPr=cell._tc.get_or_add_tcPr(); s=OxmlElement("w:shd"); s.set(qn("w:val"),"clear"); s.set(qn("w:color"),"auto"); s.set(qn("w:fill"),hexc); tcPr.append(s)
tcount=[0]; fcount=[0]
def caption(kind,text):
    if kind=="Table": tcount[0]+=1; n=tcount[0]
    else: fcount[0]+=1; n=fcount[0]
    p=doc.add_paragraph(); r=p.add_run(f"{kind} {n}. "); r.bold=True; r.font.size=Pt(9.5)
    add_runs(p,text); 
    for r in p.runs: r.font.size=Pt(9.5)
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER if kind=="Figure" else WD_ALIGN_PARAGRAPH.LEFT
    if kind=="Table": p.paragraph_format.keep_with_next=True
    return n
def table(rows,widths=None,cap=None,fs=9):
    if cap: caption("Table",cap)
    t=doc.add_table(rows=len(rows),cols=len(rows[0])); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            c=t.cell(i,j); c.text=""; p=c.paragraphs[0]; add_runs(p,str(val))
            p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.0
            for r in p.runs:
                r.font.size=Pt(fs)
                if i==0: r.bold=True
            if i==0: shade(c,"D9E2F3")
            if widths: c.width=Inches(widths[j])
    t.autofit=False
    if widths:
        for j,wd in enumerate(widths): t.columns[j].width=Inches(wd)
        grid=t._tbl.tblGrid
        for j,gc in enumerate(grid.findall(qn("w:gridCol"))): gc.set(qn("w:w"),str(int(widths[j]*1440)))
    for i,row in enumerate(t.rows):
        trPr=row._tr.get_or_add_trPr(); cs=OxmlElement("w:cantSplit"); trPr.append(cs)
        if len(rows)<=20 and i<len(rows)-1:
            for c in row.cells:
                for p in c.paragraphs: p.paragraph_format.keep_with_next=True
    trPr=t.rows[0]._tr.get_or_add_trPr(); h=OxmlElement("w:tblHeader"); h.set(qn("w:val"),"true"); trPr.append(h)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return t
def figure(path,width,cap):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(path,width=Inches(width)); p.paragraph_format.keep_with_next=True
    caption("Figure",cap)
def fig_placeholder(text,cap):
    t=doc.add_table(rows=1,cols=1); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    c=t.cell(0,0); c.width=Inches(6); shade(c,"F2F2F2"); p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for _ in range(3): p.add_run("\n")
    r=p.add_run(text); r.italic=True; r.font.color.rgb=RGBColor(0x60,0x60,0x60)
    for _ in range(3): p.add_run("\n")
    caption("Figure",cap)

# ---------- Phase 1 data (measured; do not edit without new runs) ----------
MODELS=["FP16","RTN INT8","RTN INT4","GPTQ INT4","AWQ INT4"]
ACC_AVG_EXACT={  # unrounded mean of the four official *-full accuracies; Table 2 averages are these rounded once to 3 d.p.
 "FP16":0.561945051500613,"RTN INT8":0.5588555427896876,"RTN INT4":0.5234534165132109,
 "GPTQ INT4":0.5184748016161879,"AWQ INT4":0.5411871062315032}
ACC={  # ARC-C acc_norm, ARC-E acc_norm, HellaSwag acc_norm, WinoGrande acc, average of the four, disk MB
 "FP16":(".3677",".6195",".6433",".6172",".562","2,357"),
 "RTN INT8":(".3635",".6195",".6431",".6093",".559","1,180"),
 "RTN INT4":(".3242",".5745",".6000",".5951",".523","730"),
 "GPTQ INT4":(".3336",".5661",".5823",".5919",".518","730"),
 "AWQ INT4":(".3396",".5993",".6173",".6085",".541","730")}
EXT={  # BoolQ acc, PIQA acc_norm, WikiText-2 word ppl (BOS at every 2048-token window)
 "FP16":(".639",".750","15.46"),"RTN INT8":(".631",".745","15.50"),"RTN INT4":(".560",".732","19.98"),
 "GPTQ INT4":(".545",".717","20.68"),"AWQ INT4":(".596",".735","18.22")}
SPEED={  # TTFT ms, decode tok/s, peak working set MB
 "FP16":("41.2","25.0","5,579"),"RTN INT8":("24.7","42.3","3,693"),"RTN INT4":("16.5","65.7","2,545"),
 "GPTQ INT4":("17.9","59.9","2,434"),"AWQ INT4":("16.9","65.3","2,433")}
NOBOS={"FP16":"12.89","RTN INT8":"12.92","RTN INT4":"15.59","GPTQ INT4":"423.7","AWQ INT4":"14.89"}
num=lambda s: float(s.replace(",",""))

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
 "RTN INT4":("652–654","5.84",(1.938,1.949,1.942),(1.769,1.768,1.768))}

# ---------- Laptop LoRA speed/memory (OpenVINO GenAI 2026.4.1, Arc iGPU; seed-0 adapter, unmerged, scale 2.0) ----------
# Means of kept balanced-interleaved rounds. Memory: peak commit (private bytes) and peak working set, MB.
# FP16 no-LoRA: 2 rounds (r3 dropped, CPU 59%). Others: 3 rounds.
P2_SPEED={  # tok/s, TTFT ms, peak commit MB, peak working set MB
 "FP16":    {"off":(24.9,41.7,6252,5578),"on":(22.9,44.3,6287,5595)},  # on WS 5595 excludes trimmed r5; see Table 6 note
 "RTN INT8":{"off":(41.2,25.6,4258,3590),"on":(35.0,29.6,4280,3613)},
 "RTN INT4":{"off":(62.8,17.5,3097,2433),"on":(51.4,22.2,3108,2452)}}
P2_SPEED_ROUNDS={  # raw tok/s per kept round
 "FP16":    {"off":[24.98,24.81],"on":[22.86,22.86,23.06]},
 "RTN INT8":{"off":[41.47,41.30,40.82],"on":[32.73,36.11,36.03]},
 "RTN INT4":{"off":[63.68,60.75,63.94],"on":[52.89,49.34,52.09]}}
P2_SPEED_DELTA={"FP16":"−8%","RTN INT8":"−15%","RTN INT4":"−18%"}
# Paired significance tests (lora/stats/paired_bootstrap.py -> lora/stats/paired_results.json): stratified paired bootstrap
# (B=10,000) on the six-task average + exact McNemar (pooled over tasks; LoRA cells: max p over seeds). Re-scored per-question data.
P2_APP_MAIN=[['Comparison', 'Diff.', '95% CI', 'p (boot.)', 'Holm p (boot.)', 'McNemar: first-only / second-only questions (per seed)', 'McNemar p (max)', 'Holm p (McNemar)'], ['RTN INT8 vs FP16 (no LoRA)', '−0.06', '[−0.29, +0.17]', '0.60', '1.00', '121/115', '0.74', '1.00'], ['RTN INT4 vs FP16 (no LoRA)', '−4.03', '[−4.73, −3.33]', '< 0.001', '< 0.01', '895/1787', '< 10⁻⁶⁶', '< 10⁻⁶⁵'], ['LoRA effect, FP16', '+1.05', '[+0.64, +1.46]', '< 0.001', '< 0.01', '614/386; 615/412; 630/404', '< 10⁻⁹', '< 10⁻⁸'], ['LoRA effect, RTN INT8', '+1.11', '[+0.70, +1.52]', '< 0.001', '< 0.01', '627/381; 618/426; 611/407', '< 10⁻⁸', '< 10⁻⁸'], ['LoRA effect, RTN INT4', '+3.97', '[+3.37, +4.56]', '< 0.001', '< 0.01', '1565/730; 1561/755; 1534/735', '< 10⁻⁶³', '< 10⁻⁶²'], ['INT8+LoRA vs FP16+LoRA', '−0.01', '[−0.14, +0.13]', '0.92', '1.00', '114/90; 113/118; 88/104', '0.79', '1.00'], ['INT4+LoRA vs FP16+LoRA', '−1.11', '[−1.63, −0.59]', '< 0.001', '< 0.01', '713/998; 709/998; 687/1006', '< 10⁻¹¹', '< 10⁻¹⁰'], ['INT4+LoRA vs plain FP16 †', '−0.05', '[−0.66, +0.53]', '0.85', '—', '979/1036; 980/1066; 962/1055', '0.21', '—']]
P2_APP_TASK=[['Comparison', 'Task', 'Diff.', '95% CI', 'p (boot.)', 'McNemar p (max over seeds)'], ['RTN INT8 vs FP16 (no LoRA)', 'ARC-C', '−0.17', '[−0.94, +0.60]', '0.74', '0.83'], [' ', 'ARC-E', '+0.17', '[−0.34, +0.67]', '0.57', '0.62'], [' ', 'HellaSwag', '+0.15', '[−0.01, +0.30]', '0.07', '0.07'], [' ', 'WinoGrande', '−0.16', '[−0.95, +0.63]', '0.77', '0.84'], [' ', 'BoolQ', '−0.18', '[−0.73, +0.34]', '0.53', '0.57'], [' ', 'PIQA', '−0.16', '[−0.60, +0.22]', '0.50', '0.61'], ['RTN INT4 vs FP16 (no LoRA)', 'ARC-C', '−4.69', '[−6.83, −2.65]', '< 0.001', '< 10⁻⁴'], [' ', 'ARC-E', '−4.38', '[−5.77, −2.95]', '< 0.001', '< 10⁻⁸'], [' ', 'HellaSwag', '−4.25', '[−4.82, −3.70]', '< 0.001', '< 10⁻⁵⁰'], [' ', 'WinoGrande', '−1.34', '[−3.71, +1.03]', '0.27', '0.29'], [' ', 'BoolQ', '−7.98', '[−9.85, −6.15]', '< 0.001', '< 10⁻¹⁶'], [' ', 'PIQA', '−1.52', '[−2.88, −0.22]', '0.03', '0.03'], ['LoRA effect, FP16', 'ARC-C', '+0.37', '[−1.17, +1.85]', '0.65', '1.00'], [' ', 'ARC-E', '+1.32', '[+0.28, +2.36]', '0.02', '0.04'], [' ', 'HellaSwag', '+0.75', '[+0.44, +1.06]', '< 0.001', '< 10⁻³'], [' ', 'WinoGrande', '+0.26', '[−0.68, +1.24]', '0.61', '1.00'], [' ', 'BoolQ', '+2.65', '[+1.70, +3.62]', '< 0.001', '< 10⁻⁴'], [' ', 'PIQA', '+0.96', '[+0.22, +1.74]', '0.01', '0.05'], ['LoRA effect, RTN INT8', 'ARC-C', '+0.46', '[−1.08, +1.99]', '0.58', '1.00'], [' ', 'ARC-E', '+1.09', '[+0.06, +2.12]', '0.04', '0.32'], [' ', 'HellaSwag', '+0.63', '[+0.33, +0.94]', '< 0.001', '0.002'], [' ', 'WinoGrande', '+0.55', '[−0.53, +1.63]', '0.33', '0.70'], [' ', 'BoolQ', '+2.84', '[+1.89, +3.83]', '< 0.001', '< 10⁻⁵'], [' ', 'PIQA', '+1.07', '[+0.38, +1.80]', '0.003', '0.03'], ['LoRA effect, RTN INT4', 'ARC-C', '+3.56', '[+1.71, +5.40]', '< 0.001', '< 10⁻³'], [' ', 'ARC-E', '+5.12', '[+3.82, +6.41]', '< 0.001', '< 10⁻¹¹'], [' ', 'HellaSwag', '+2.97', '[+2.47, +3.46]', '< 0.001', '< 10⁻²³'], [' ', 'WinoGrande', '+1.24', '[−0.61, +3.08]', '0.21', '0.39'], [' ', 'BoolQ', '+9.41', '[+7.77, +11.08]', '< 0.001', '< 10⁻²²'], [' ', 'PIQA', '+1.56', '[+0.49, +2.67]', '0.005', '0.04'], ['INT8+LoRA vs FP16+LoRA', 'ARC-C', '−0.09', '[−0.54, +0.34]', '0.73', '1.00'], [' ', 'ARC-E', '−0.06', '[−0.36, +0.25]', '0.72', '0.71'], [' ', 'HellaSwag', '+0.02', '[−0.09, +0.14]', '0.69', '1.00'], [' ', 'WinoGrande', '+0.13', '[−0.29, +0.58]', '0.59', '0.83'], [' ', 'BoolQ', '+0.01', '[−0.29, +0.30]', '0.97', '0.90'], [' ', 'PIQA', '−0.05', '[−0.29, +0.18]', '0.63', '0.79'], ['INT4+LoRA vs FP16+LoRA', 'ARC-C', '−1.51', '[−3.19, +0.09]', '0.07', '0.37'], [' ', 'ARC-E', '−0.58', '[−1.67, +0.51]', '0.30', '0.56'], [' ', 'HellaSwag', '−2.04', '[−2.47, −1.61]', '< 0.001', '< 10⁻¹¹'], [' ', 'WinoGrande', '−0.37', '[−2.29, +1.50]', '0.69', '0.70'], [' ', 'BoolQ', '−1.22', '[−2.31, −0.12]', '0.03', '0.10'], [' ', 'PIQA', '−0.92', '[−1.96, +0.11]', '0.08', '0.51'], ['INT4+LoRA vs plain FP16 †', 'ARC-C', '−1.14', '[−3.07, +0.74]', '0.23', '0.36'], [' ', 'ARC-E', '+0.74', '[−0.63, +2.09]', '0.30', '0.49'], [' ', 'HellaSwag', '−1.28', '[−1.78, −0.80]', '< 0.001', '< 10⁻³'], [' ', 'WinoGrande', '−0.11', '[−2.18, +1.92]', '0.90', '0.78'], [' ', 'BoolQ', '+1.43', '[+0.07, +2.79]', '0.04', '0.27'], [' ', 'PIQA', '+0.04', '[−1.12, +1.20]', '0.96', '0.93']]
P2_APP_BSCHECK=[['Task', 'Original (batch 11)', 'Check, batch 11', 'Check, batch 13', 'Acc. = original?', 'Flags = re-score?'], ['ARC-C', '0.3191', '0.3191', '0.3200', 'yes', 'yes'], ['ARC-E', '0.5749', '0.5749', '0.5741', 'yes', 'yes'], ['HellaSwag', '0.6000', '0.6000', '0.5996', 'yes', 'yes'], ['WinoGrande', '0.5991', '0.5991', '0.5983', 'yes', 'yes'], ['BoolQ', '0.5560', '0.5560', '0.5581', 'yes', 'yes'], ['PIQA', '0.7329', '0.7329', '0.7329', 'yes', 'yes']]
P2_APP_RESCORE=[['Model', 'Batch size (first / re-score)', 'ARC-C', 'ARC-E', 'HellaSwag', 'WinoGrande', 'BoolQ', 'PIQA', 'Avg. change (points)'], ['fp16-nolora', '11 / 11', '0', '0', '0', '0', '0', '0', '0'], ['fp16-seed0', '11 / 11', '0', '0', '0', '0', '0', '0', '0'], ['fp16-seed1', 'n/l / 11', '0', '0', '0', '0', '0', '0', '0'], ['fp16-seed2', 'n/l / 11', '0', '0', '0', '0', '0', '0', '0'], ['int8-rtn-nolora', '11 / 13', '0', '0', '+5', '−2', '+3', '0', '−0.003'], ['int8-rtn-seed0', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0'], ['int8-rtn-seed1', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0'], ['int8-rtn-seed2', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0'], ['int4-rtn-nolora', '11 / 13', '+1', '−2', '−4', '−1', '+7', '0', '+0.016'], ['int4-rtn-seed0', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0'], ['int4-rtn-seed1', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0'], ['int4-rtn-seed2', 'n/l / 13', '0', '0', '0', '0', '0', '0', '0']]
P2_STATS_ROWS=[['Comparison (six-task average)', 'Diff. (points)', '95% CI (paired bootstrap)', 'p (boot.)', 'Holm p', 'McNemar p', 'Per-seed diffs'], ['RTN INT8 vs FP16 (no LoRA)', '−0.06', '[−0.29, +0.17]', '0.60', '1.00', '0.74', '—'], ['RTN INT4 vs FP16 (no LoRA)', '−4.03', '[−4.73, −3.33]', '< 0.001', '< 0.01', '< 10⁻⁶⁶', '—'], ['LoRA effect, FP16', '+1.05', '[+0.64, +1.46]', '< 0.001', '< 0.01', '< 10⁻⁹', '+1.17, +0.84, +1.15'], ['LoRA effect, RTN INT8', '+1.11', '[+0.70, +1.52]', '< 0.001', '< 0.01', '< 10⁻⁸', '+1.37, +0.86, +1.09'], ['LoRA effect, RTN INT4', '+3.97', '[+3.37, +4.56]', '< 0.001', '< 0.01', '< 10⁻⁶³', '+3.91, +4.05, +3.97'], ['INT8+LoRA vs FP16+LoRA', '−0.01', '[−0.14, +0.13]', '0.92', '1.00', '0.79', '+0.14, −0.04, −0.11'], ['INT4+LoRA vs FP16+LoRA', '−1.11', '[−1.63, −0.59]', '< 0.001', '< 0.01', '< 10⁻¹¹', '−1.29, −0.82, −1.21'], ['INT4+LoRA vs plain FP16 †', '−0.05', '[−0.66, +0.53]', '0.85', '—', '0.21', '−0.12, +0.02, −0.06']]

# ---------- Figures ----------
import os
def make_figures():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams["font.family"]="DejaVu Sans"
    os.makedirs(G,exist_ok=True)
    # Fig: Phase 1 pipeline (five versions)
    fig,ax=plt.subplots(figsize=(9,4.0)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,4.6)
    def box(x,y,w,h,t,c,fs=8.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",fc=c,ec="#333",lw=1))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
    def arr(x1,y1,x2,y2,c="#333",ls="-"): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",lw=1.1,color=c,linestyle=ls))
    box(0.1,2.3,1.5,0.8,"Download\nLlama-3.2-1B\n(FP16)","#dbe9f6")
    box(2.0,2.3,1.5,0.8,"Model 1\nFP16 reference","#dbe9f6")
    box(2.0,0.45,1.5,0.75,"128 WikiText-2\ncalibration\nsamples","#fdf0d5",8)
    labels=[("Model 2: RTN INT8","#e3f2e1"),("Model 3: RTN INT4","#e3f2e1"),("Model 4: GPTQ INT4","#f6e0e0"),("Model 5: AWQ INT4","#fff3c4")]
    ys=[3.75,3.05,2.35,1.65]
    for (l,c),y in zip(labels,ys): box(4.2,y,2.2,0.48,l,c,8.5)
    arr(1.6,2.7,2.0,2.7)
    for y in ys: arr(3.5,2.7,4.2,y+0.24)
    for y in ys[2:]: arr(3.5,0.85,4.2,y+0.24,"#b5651d","--")
    ax.text(0.9,0.1,"same calibration set: GPTQ and AWQ only (dashed)",fontsize=7,color="#b5651d")
    ax.plot([6.6,6.75,6.75,6.6],[4.25,4.25,1.65,1.65],color="#333",lw=1)
    ax.text(6.9,4.3,"all 5 models, incl. FP16",fontsize=7,style="italic")
    arr(6.75,2.95,7.15,2.95)
    box(7.2,3.15,1.45,0.95,"Accuracy\n(lm-eval, 6 tasks\n+ WikiText-2 ppl)","#ece4f5",8)
    box(7.2,1.75,1.45,0.95,"Size, peak memory,\nTTFT, tokens/s","#ece4f5",8)
    ax.text(7.92,1.35,"on the laptop\n(OpenVINO, Arc GPU)",fontsize=7,ha="center",style="italic")
    box(8.9,2.5,1.05,0.8,"Phase 1\nresults\ntables","#eeeeee")
    arr(8.65,3.6,8.9,3.1); arr(8.65,2.2,8.9,2.7)
    ax.text(4.2,0.75,"NNCF 3.4.0 weight compression\nINT8: symmetric, per channel; INT4: symmetric, group size 128",fontsize=7,color="#555")
    plt.tight_layout(); plt.savefig(G+"fig_phase1.png",dpi=200); plt.close()
    # Fig: Phase 2 method (as run)
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
    fig,ax=plt.subplots(figsize=(7,3.9)); x=np.arange(len(P2_BASES)); w=0.36
    no=[P2_NO[b][6] for b in P2_BASES]; lo=[P2_LORA[b][6][0] for b in P2_BASES]; sd=[P2_LORA[b][6][1] for b in P2_BASES]
    b1=ax.bar(x-w/2,no,w,color="#b8c7dc",ec="#333",lw=0.6,label="No LoRA (single evaluation)")
    b2=ax.bar(x+w/2,lo,w,yerr=sd,capsize=4,color="#4a7fb5",ec="#333",lw=0.6,error_kw=dict(lw=1),label="LoRA (mean ± std, 3 seeds)")
    ax.axhline(P2_NO["FP16"][6],ls="--",color="#c0504d",lw=1.1,label=f"FP16 without LoRA ({P2_NO['FP16'][6]:.2f})")
    for r,v in zip(b1,no): ax.text(r.get_x()+r.get_width()/2,v-0.25,f"{v:.2f}",ha="center",va="top",fontsize=7.5,color="#1F2A44")
    for r,v,s in zip(b2,lo,sd): ax.text(r.get_x()+r.get_width()/2,v-0.25,f"{v:.2f}\n±{s:.2f}",ha="center",va="top",fontsize=7.5,color="white")
    ax.set_xticks(x); ax.set_xticklabels(P2_BASES,fontsize=9); ax.set_ylim(54,63)
    ax.set_ylabel("Average accuracy, 6 tasks (%)",fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5,loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False)
    plt.tight_layout(); plt.savefig(G+"fig_phase2.png",dpi=200); plt.close()

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
    plt.tight_layout(); plt.savefig(G+"fig_lora_tradeoff.png",dpi=200); plt.close()
    # Fig: accuracy vs peak memory and vs decode throughput (measured)
    col={"FP16":"#4a7fb5","RTN INT8":"#6aaa64","RTN INT4":"#2e7d32","GPTQ INT4":"#c0504d","AWQ INT4":"#d08c3a"}
    mk={"FP16":"o","RTN INT8":"s","RTN INT4":"s","GPTQ INT4":"^","AWQ INT4":"D"}
    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    off={"RTN INT4":(6,-11),"GPTQ INT4":(6,-11),"AWQ INT4":(6,4),"FP16":(-10,-13),"RTN INT8":(6,4)}
    off2={"RTN INT4":(-62,-4),"GPTQ INT4":(6,-4),"AWQ INT4":(-58,5),"FP16":(6,-4),"RTN INT8":(6,-4)}
    for m in MODELS:
        a=num(ACC[m][4])
        axs[0].scatter(num(SPEED[m][2])/1000,a,color=col[m],marker=mk[m],s=45,zorder=3)
        axs[0].annotate(m,(num(SPEED[m][2])/1000,a),xytext=off[m],textcoords="offset points",fontsize=7.5)
        axs[1].scatter(num(SPEED[m][1]),a,color=col[m],marker=mk[m],s=45,zorder=3)
        axs[1].annotate(m,(num(SPEED[m][1]),a),xytext=off2[m],textcoords="offset points",fontsize=7.5)
    axs[0].set_xlabel("Peak process memory (GB)",fontsize=8.5); axs[1].set_xlabel("Decode throughput (tokens/s)",fontsize=8.5)
    axs[0].set_xlim(2.0,6.0); axs[1].set_xlim(20,72)
    for a_ in axs:
        a_.set_ylabel("Average accuracy (4 tasks)",fontsize=8.5); a_.set_ylim(0.50,0.575)
        a_.tick_params(labelsize=8); a_.spines[["top","right"]].set_visible(False); a_.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(G+"fig_tradeoff.png",dpi=200); plt.close()
make_figures()

# ---------- Title ----------
tp=doc.add_paragraph(style="Title"); tp.add_run("Shrinking a Small Language Model on a Laptop: A Controlled Comparison of Rounding, GPTQ, and AWQ Weight Compression, With and Without LoRA"); tp.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=para("Dhruv Vinesh, Cypress Ranch High School, Cypress, Texas",align=WD_ALIGN_PARAGRAPH.CENTER)
p=para("November 2026",align=WD_ALIGN_PARAGRAPH.CENTER)

H1("Abstract")
para("Small language models on consumer laptops are limited by memory capacity and bandwidth. This controlled benchmark, not a new method, studies weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 on an Intel Core Ultra 7 laptop (Arc GPU, 16 GB shared RAM). We compare FP16 with RTN INT8 and with RTN, GPTQ and AWQ INT4 (group size 128, shared calibration set). Accuracy is measured zero-shot on six tasks plus WikiText-2 perplexity; speed and peak memory are measured with OpenVINO GenAI. Phase 2 trains the same LoRA recipe (three seeds) on bit-matched FP16, RTN INT8 and RTN INT4 bases. QLoRA showed that 4-bit LoRA can recover accuracy; LoRA on GPTQ and AWQ is pending.")
para("On the four-task average (ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande), RTN INT8 is 0.559 versus 0.562 for FP16 (not significant), with 1.7× decode throughput and 34% lower peak memory. On the six-task average the INT8 gap is 0.42 points and significant, driven by BoolQ. INT4 raises throughput about 2.6× (RTN, AWQ) and cuts peak memory by 54–56%. AWQ is the most accurate INT4 method (0.541, versus 0.523 for RTN and 0.518 for GPTQ). GPTQ INT4 WikiText-2 perplexity is 424 under the standard protocol, from one matrix column. On Colab, RTN INT4 loses 4.04 six-task points; LoRA closes about 73% of the gap to FP16 + LoRA, leaving a significant 1.10-point remainder. Unmerged LoRA adds 11–35 MB and slows decode by 8–18%; INT4 + LoRA remains about 2.2× as fast as FP16 + LoRA at about half the memory.")

# ---------- 1 Introduction ----------
H1("1 Introduction")
para("Running a language model locally on a laptop keeps data on the device, works offline and avoids per-query cost. The obstacle is memory. A model with about 1.24 billion weights, such as Llama-3.2-1B {llama}, occupies roughly 2.5 GB in 16-bit floating point (FP16). On a consumer laptop whose integrated GPU shares 16 GB of system memory with the operating system and other applications, that footprint matters. It also matters for speed: generating each token requires reading nearly all weights from memory, so autoregressive decoding is typically limited by memory bandwidth, and storing weights in fewer bits can, in principle, make generation faster as well as smaller.")
para("Post-training weight-only methods such as GPTQ {gptq} and AWQ {awq} reduce weights to 4 bits with modest accuracy loss on large models, and 8-bit weight quantization is typically close to lossless {huang, evalq}. Degradation grows at lower bit widths, and the LLaMA3 family has been reported as more sensitive than earlier models {huang}. Configuration choices, including calibration data, can confound comparisons {llmc}. Low-rank adaptation (LoRA) {lora} offers a way to recover lost accuracy: QLoRA {qlora} already showed that LoRA trained on a frozen 4-bit model recovers much of the accuracy lost to quantization. Recovery is not guaranteed on every setting: LoRA fine-tuning of quantized LLaMA3-8B on general instruction data did not restore, and sometimes lowered, zero-shot accuracy {huang}, and merging an adapter into a quantized model and re-quantizing costs accuracy {qalora}.")
para("Three practical questions remain for a ~1B model on a laptop. Comparisons of compression methods are usually run on NVIDIA GPUs; when efficiency is reported at all, it is rarely *measured* on consumer non-NVIDIA hardware with the toolchain a laptop user would actually use. Small models at 4 bits are less studied than 7B+ models, even though they are the models most likely to run on laptops. And LoRA recovery is typically examined for one compression scheme at a time. This paper is a controlled benchmark rather than a new method. It does not test whether LoRA can recover quantization loss in general, which QLoRA {qlora} has already shown. We asked:")
bullets([
 "**RQ1.** Which weight-only method (RTN, GPTQ or AWQ) best preserves the zero-shot accuracy of a ~1B model at INT4, relative to FP16 and an RTN INT8 reference?",
 "**RQ2.** What memory and speed benefits are actually realized on a consumer Intel laptop (Core Ultra 7 155H, integrated Arc GPU) running OpenVINO, as opposed to those implied by bit width?",
 "**RQ3.** Does a fixed LoRA recipe recover accuracy differently depending on the compression applied to the frozen base model? (Answered here for RTN INT8 and RTN INT4; GPTQ and AWQ bases pending.)",
])
para("**What was measured.**")
bullets([
 "Zero-shot accuracy of FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 on six tasks and WikiText-2, with a matched group size and calibration set, on the laptop (OpenVINO).",
 "Time-to-first-token, decode throughput and peak memory of those five versions, and of the unmerged seed-0 LoRA adapters, on the same laptop (OpenVINO GenAI).",
 "The same LoRA recipe (*r* = 16, α = 32, q/k/v/o, 5,000 Dolly examples, three seeds) trained on frozen PyTorch bases that bit-match the laptop FP16, RTN INT8 and RTN INT4 models, evaluated on Colab with and without the adapter.",
 "Paired significance tests on per-question logs for both phases, and a confirmation of the Colab evaluation-batch-size effect.",
 "Diagnostics of the GPTQ INT4 WikiText failure (one input column of layers.1.mlp.down_proj) and of AWQ’s folded scales and 0.966 median scale product.",
])
para("Scripts, configuration, calibration and Dolly indices, per-question logs, training logs and raw timing logs are at https://github.com/theLegend35/quant-study-llama32-1b. Section 2 reviews related work, Section 3 describes the setup, Section 4 reports results, and Sections 5 and 6 discuss limitations and conclude. Background on the compression methods and further experimental detail are in Appendices H and I.")

# ---------- 2 Related Work ----------
H1("2 Related Work")
para("In *weight-only* quantization, weights are stored as integers with a scale (real value ≈ integer × scale) and dequantized during computation, while activations remain in higher precision. This study uses symmetric grouped quantization with group size 128. Round-to-nearest (RTN) sets each group’s scale from its largest absolute weight and rounds every weight independently; it is data-free. GPTQ {gptq} uses calibration inputs to quantize a layer column by column and updates the remaining weights to compensate for the error just introduced. AWQ {awq} scales up weight channels that multiply large activations, then folds the inverse scale into the preceding computation so that salient channels have smaller relative rounding error. LoRA {lora} freezes a weight matrix *W* and learns a low-rank update *W* + (α/*r*)·*BA*. QLoRA {qlora} applies LoRA to a frozen 4-bit base. An adapter can be kept separate at inference or merged; merging into a quantized model requires re-quantization, which can reduce accuracy {qalora}. Fuller definitions are in Appendix H.")
para("**Quantization benchmarks.** Li et al. {evalq} evaluate weight, activation and KV-cache quantization across 11 model families from 125M to 180B parameters. Huang et al. {huang} evaluate ten post-training quantization and LoRA fine-tuning methods on LLaMA3 at 1–8 bits and report non-negligible degradation, particularly at ultra-low bit widths. LLMC {llmc} shows that calibration data, algorithm choice and data format all affect results. Our study is narrower, one 1B model and weight-only formats, but adopts LLMC’s lesson by fixing calibration data and group size across methods, and adds measured efficiency on a consumer Intel laptop rather than reporting accuracy alone.")
para("**Pruning versus quantization.** Jaiswal et al. {kick} argue that perplexity can hide capability loss, and that quantization often holds up better than pruning on knowledge-intensive tasks. Focusing on small language models (0.5–3.8B), Zhou, Kurz and Zhao {zhou} find that quantization consistently outperforms pruning and recommend AWQ in particular. Structural pruning with brief LoRA-based recovery is explored by LLM-Pruner {pruner}. Pruning is outside the scope of this study (Section 5).")
para("**LoRA-based recovery of compressed models.** QLoRA {qlora} established that LoRA on a frozen 4-bit base can be effective. Subsequent work modifies the quantization–adaptation interface: LoftQ {loftq} jointly quantizes the model and initializes the adapter to compensate quantization error; IR-QLoRA {irqlora} retains more information during quantization; QA-LoRA {qalora} uses group-wise operators so that the merged model remains quantized. Huang et al. {huang} report that QLoRA-style fine-tuning of 4-bit LLaMA3-8B on general instruction data did not recover zero-shot accuracy. Rather than proposing a new adaptation method, we hold the LoRA recipe fixed and vary only the compression method of the frozen base, with FP16 + LoRA as the control.")

# ---------- 3 Experimental Setup ----------
H1("3 Experimental Setup")
para("The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. Phase 2 trained LoRA adapters on Google Colab for the FP16, RTN INT8 and RTN INT4 bases, with settings fixed before training, and then measured those unmerged adapters on the same laptop. LoRA on the GPTQ and AWQ bases is pending. Training details, software versions, metric definitions and the pre-specified hypotheses are in Appendix I.")

H2("3.1 Model and hardware")
para("We use the base (not instruction-tuned) Llama-3.2-1B {llama} (meta-llama/Llama-3.2-1B), with approximately 1.24 billion parameters (≈2.5 GB in FP16), obtained from Hugging Face under its gated license; the planned fallback to Qwen2.5-0.5B {qwen} was not needed. Compression and all efficiency measurements ran on a single Dell Inspiron laptop: Intel Core Ultra 7 155H, integrated Intel Arc GPU sharing 16 GB of system RAM, Windows 11 (build 10.0.26300), no discrete NVIDIA GPU. Models were converted to and executed with OpenVINO {openvino} on the Arc GPU, and compressed with NNCF 3.4.0 {nncfdoc} (OpenVINO 2026.4.1, exported through optimum-intel 2.2.0). LoRA training ran on Google Colab with an NVIDIA T4 GPU.")

H2("3.2 Five model versions")
para("Table 1 lists the five versions. All compressed versions use NNCF weight-only compression. RTN INT8 uses symmetric quantization with one scale per output channel (NNCF mode INT8_SYM) for every weight matrix, including the embedding. The INT4 versions use symmetric, group-wise quantization with group size 128 (INT4_SYM, ratio 1.0) for all decoder linear layers, while the embedding and the tied output layer (lm_head) are kept at a backup precision of INT8_ASYM, one scale and zero point per channel. GPTQ and AWQ use the same 128 calibration samples drawn once from WikiText-2 {wikitext}. Optional NNCF refinements (e.g., scale estimation) were disabled for all methods so that each algorithm is evaluated in isolation, and the same backup-precision setting was used for all INT4 versions, which have identical on-disk sizes. GPTQ and AWQ were not evaluated at INT8 because prior work reports near-lossless 8-bit quantization across methods {huang, evalq}; RTN INT8 serves as the near-lossless reference point. Calibration data, symmetry and scale estimation were varied only in the diagnostic ablations in Appendix F.")
table([
 ["#","Version","Method","Precision","Calibration","Notes"],
 ["1","FP16","none","FP16","—","Uncompressed reference"],
 ["2","RTN INT8","Round-to-nearest","INT8, per channel","—","Near-lossless reference point"],
 ["3","RTN INT4","Round-to-nearest","INT4, g128","—","Data-free baseline"],
 ["4","GPTQ INT4","GPTQ","INT4, g128","128 × WikiText-2","Error-compensating"],
 ["5","AWQ INT4","AWQ","INT4, g128","same 128","Activation-aware scaling"],
],[0.3,1.0,1.4,1.0,1.2,1.6],cap="Model versions. All compressed versions: NNCF 3.4.0 weight-only compression. INT8: symmetric, per output channel (INT8_SYM), all weights including the embedding. INT4: symmetric with group size 128 (g128, INT4_SYM, ratio 1.0); embedding and tied lm_head at INT8_ASYM per channel.")
para("INT3 variants, unstructured pruning and mixed-precision formats were in the original plan and remain deferred until this INT4 grid and the RTN part of Phase 2 are complete (Section 5).")

H2("3.3 Benchmarks, metrics and statistics")
para("Phase 1 accuracy was measured zero-shot with lm-evaluation-harness {lmeval} using its OpenVINO backend, on the full evaluation sets of ARC-Challenge and ARC-Easy {arc} (length-normalized accuracy, acc_norm), HellaSwag {hellaswag} (acc_norm) and WinoGrande {winogrande} (accuracy). The unweighted mean of these four tasks is the primary Phase 1 accuracy summary. As an extended set, we report BoolQ (accuracy), PIQA (acc_norm) and WikiText-2 {wikitext} word perplexity with max_length = 2048. Approximate chance levels are 25% (ARC, HellaSwag) and 50% (WinoGrande, PIQA); BoolQ’s majority-class rate is about 62%. Because log-likelihood scoring is deterministic, each model–task pair was evaluated once. The headline WikiText perplexity prepends a beginning-of-sequence (BOS) token to every 2048-token window, matching the BOS-prefixed inputs used in all accuracy evaluations. The standard lm-evaluation-harness protocol, which places BOS only on each document’s first window, is the robustness check in Section 4.3.")
para("Efficiency was measured on the laptop only, with OpenVINO GenAI {genai} on the Arc GPU, batch size 1, a 128-token prompt, 128 new tokens, greedy decoding, and the median of 30 runs after 5 warm-ups. Metric definitions are in Appendix I.")
para("For each task we report the standard error (SE) from lm-evaluation-harness; approximate SEs are ±0.014 for ARC-Challenge and WinoGrande, ±0.010 for ARC-Easy and ±0.005 for HellaSwag. Differences between models are tested with paired tests over questions (Section 4.6; details in Appendix B), because every model answers the same questions. The primary test is a paired bootstrap of the difference in six-task average accuracy (the unweighted mean of the six task accuracies, as in Table 5) with 10,000 resamples, a percentile 95% CI and a two-sided bootstrap p-value. Holm correction is applied separately to the seven main Phase 2 comparisons and to the seven Phase 1 comparisons (each method versus FP16, plus AWQ vs RTN INT4, GPTQ vs RTN INT4, and AWQ vs GPTQ). Exact McNemar tests are a secondary check. Per-task tests are uncorrected. LoRA results are the mean ± standard deviation across three seeds.")

H2("3.4 Phase 2 LoRA setup")
para("LoRA needs a PyTorch model on an NVIDIA GPU, so for FP16, RTN INT8 and RTN INT4 we built frozen PyTorch copies on Colab. For the quantized bases, each weight is quantized and immediately dequantized (integer × scale) with the same scheme as the laptop model (Section 3.2), and scales are rounded to FP16 as OpenVINO stores them. The resulting integer codes and scales bit-match those in the laptop models. This is the usual QLoRA-style setup {qlora}: the quantized weights stay frozen and are stored in FP16 on the GPU (fake quantization). The Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Section 4.4 and Appendix D).")
para("Adapters were trained with PEFT {peft}: rank *r* = 16, α = 32, dropout 0.05, on the query, key, value and output projections of every attention layer, and were kept unmerged. The recipe is identical for every base; only the frozen base differs. FP16 + LoRA is the control that separates recovery of quantization loss from the general effect of fine-tuning on Dolly. Training used 5,000 examples from databricks-dolly-15k {dolly} and three seeds (0, 1, 2). Phase 2 models were evaluated on Colab with lm-evaluation-harness 0.4.13 (Hugging Face backend), zero-shot, with BOS, on the same six tasks and metrics. Because the runtime and numerics differ from OpenVINO, Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages. The prompt template, optimizer settings and training curves are in Appendices A and C; the full recipe is in Appendix I.")
para("The first Colab evaluations saved only task-level scores. A second evaluation with per-question logging was used for the paired tests. Ten of the 12 models reproduced every task score exactly. The two quantized bases without LoRA changed by at most 0.21 points when the automatically chosen evaluation batch size went from 11 to 13; a later check in the same session confirmed the cause (Appendix B). Tables 5 and 6 keep the first-run scores. The paired tests (Table 7, Appendix B) use the re-scored data, so some differences can differ from Table 5 by up to 0.016 points. The change does not alter any conclusion. Unmerged seed-0 adapters were then timed on the laptop with the same OpenVINO GenAI protocol as Phase 1, at scale α/*r* = 2.0 (Appendix G).")

H2("3.5 Reproducibility")
para("Scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs are at https://github.com/theLegend35/quant-study-llama32-1b. Software versions and fixed settings are recorded in Appendix I. Phase 1 (laptop) and Phase 2 (Colab) used different transformers versions (5.5.4 and 4.57.6 respectively). The Intel Arc GPU driver version was not recorded.")

# ---------- 4 Results ----------
H1("4 Results")
para("Sections 4.1–4.3 report Phase 1 (laptop, OpenVINO). Sections 4.4–4.6 report Phase 2 accuracy (Colab) and the speed and memory of the unmerged adapters on the laptop.")

H2("4.1 Phase 1 accuracy")
rows=[["Version","ARC-C","ARC-E","Hella- Swag","Wino- Grande","Average (4 tasks)","Δ avg vs FP16","Size on disk (MB)"]]
for m in MODELS:
    a=ACC[m]; d=ACC_AVG_EXACT[m]-ACC_AVG_EXACT["FP16"]
    rows.append([m,a[0],a[1],a[2],a[3],a[4],"—" if m=="FP16" else f"{d:+.3f}".replace("-","−"),a[5]])
table(rows,[1.05,0.65,0.65,0.7,0.7,0.85,0.85,0.95],cap="Zero-shot accuracy (fraction correct). ARC-C, ARC-E and HellaSwag: acc_norm; WinoGrande: acc. Approximate SEs: ±0.014 (ARC-C, WinoGrande), ±0.010 (ARC-E), ±0.005 (HellaSwag).",fs=8.5)
para("RTN INT8 is within one SE of FP16 on every task in Table 2, and its four-task average is 0.3 points lower (0.559 vs 0.562), while its size on disk is halved (1,180 vs 2,357 MB). All INT4 versions lose accuracy: the average falls by 3.8 points for RTN, 4.3 for GPTQ and 2.1 for AWQ. The RTN INT4 losses on ARC-Challenge, ARC-Easy and HellaSwag (4.4, 4.5 and 4.3 points) are each more than three SEs. AWQ is above RTN INT4 on all four tasks and closes about 46% of RTN’s average gap to FP16 (1.8 of 3.8 points). GPTQ is not better than RTN: it is slightly higher on ARC-Challenge (within one SE) and lower on the other three tasks.")
rows=[["Version","BoolQ (acc)","PIQA (acc_norm)","WikiText-2 word ppl ↓"]]+[[m]+list(EXT[m]) for m in MODELS]
table(rows,[1.4,1.4,1.4,1.9],cap="Extended evaluation. WikiText-2 word perplexity uses max_length = 2048 with BOS at the start of every 2048-token window (headline protocol; the standard protocol is discussed in Section 4.3).",fs=8.5)
para("The extended tasks show the same ordering (Table 3). INT8 remains close to FP16 on BoolQ, PIQA and perplexity (15.50 vs 15.46). Among INT4 versions, AWQ has the lowest perplexity (18.22, vs 19.98 for RTN and 20.68 for GPTQ) and the highest BoolQ and PIQA scores. BoolQ is the most sensitive task: INT4 costs 4.3 (AWQ) to 9.4 (GPTQ) points, and RTN and GPTQ INT4 fall below the task’s majority-class rate of about 62%. PIQA changes little (at most 3.3 points).")
para("A later re-run of the same five models with per-question logs (OpenVINO, batch size 1, seed 42) reproduced Tables 2 and 3 exactly. On the six-task average (Appendix B, Tables 14 and 15), RTN INT8 is 0.42 points below FP16 (paired 95% CI −0.69 to −0.15, Holm p = 0.006), a gap that is not significant on the four Table 2 tasks alone (−0.31, 95% CI −0.67 to +0.05, p = 0.09) and is driven mainly by BoolQ; all three INT4 methods are significantly below FP16, AWQ is significantly above both RTN INT4 (+1.85 points) and GPTQ INT4 (+2.67), and GPTQ is 0.82 points below RTN INT4 (95% CI −1.57 to −0.10, Holm p = 0.03).")

H2("4.2 Memory, speed and trade-off")
rows=[["Version","TTFT, median (ms)","Decode (tokens/s)","Decode vs FP16","Peak memory (MB)","Peak memory vs FP16"]]
for m in MODELS:
    s=SPEED[m]; r=num(s[1])/num(SPEED["FP16"][1]); pm=num(s[2])/num(SPEED["FP16"][2])-1
    rows.append([m,s[0],s[1],f"{r:.2f}×","{:,}".format(int(num(s[2]))),"—" if m=="FP16" else f"{pm*100:+.0f}%".replace("-","−")])
table(rows,[1.1,1.05,1.0,0.95,1.1,1.1],cap="Measured efficiency on the Intel Core Ultra 7 155H laptop (Arc GPU, OpenVINO GenAI, batch size 1; 128-token prompt, 128 new tokens, greedy; median of 30 runs after 5 warm-ups). Peak memory is the process working set, which includes the GPU’s shared memory. Ratio columns are computed from the measured values.",fs=8.5)
para("Lower precision made generation both faster and lighter (Table 4). RTN INT8 raised decode throughput from 25.0 to 42.3 tokens/s (1.7×), cut TTFT from 41.2 to 24.7 ms and reduced peak memory by 34%. RTN and AWQ INT4 reached about 65 tokens/s (2.6×) with TTFT of 16.5–16.9 ms, and all INT4 versions reduced peak memory by 54–56%. GPTQ INT4 decoded more slowly (59.9 tokens/s, 2.4×) despite the same format and size; we did not investigate the cause. The realized gains are smaller than the bit-width ratios: size on disk fell to 50% of FP16 at INT8 and 31% at INT4 (ideal: 50% and about 26% including group scales), while peak memory fell only to 66% and 44–46%, as the process also holds components that do not shrink with weight precision, such as the KV cache, activations and runtime buffers.")
figure(G+"fig_tradeoff.png",5.6,"Four-task average accuracy versus measured peak memory (left) and decode throughput (right). Data from Tables 2 and 4. The approximate SE of a four-task average is about ±0.006 (derived from the per-task SEs, assuming independence).")
para("Figure 1 summarizes the trade-off. INT8 moves the model to substantially lower memory and higher throughput at little accuracy cost on the four Table 2 tasks (0.3 points, not significant in the paired test) and a significant 0.42-point cost on the six-task average. INT4 roughly halves memory again and adds further throughput, at an accuracy cost that depends on the method; AWQ lies closest to the FP16 accuracy level among the INT4 versions.")

H2("4.3 Diagnostics")
para("Under the standard lm-evaluation-harness WikiText protocol, in which only the first window of each document begins with BOS, GPTQ INT4 word perplexity is 423.7, against 12.89–15.59 for the other four models; an identical rebuild gave 423.66. The failure is tied to BOS and is isolated to input column 2558 of layers.1.mlp.down_proj. Replacing only this matrix with its RTN INT4 version restores the standard-protocol perplexity to 14.86. Ablations that change the calibration data, the quantization grid and NNCF’s scale estimation do not remove the defect. Details are in Appendix F.")
para("The same stored-weight comparison shows large raw differences for AWQ on some MLP up and down matrices (maximum 4.97). Those differences are AWQ’s per-channel smoothing scales, folded into down_proj input columns and up_proj output rows, not a quantization defect. The 4.97 entry is in input column 1417, not column 2558. A leftover 2.29 maximum after a per-row median undo on layers.1.mlp.up_proj was an artifact of that estimator on a heavily scaled row. The median product of paired up- and down-projection scales is 0.966 rather than 1, which a simulation attributes to median-of-ratios bias (0.9668; least-squares on nonzero codes ≈ 1.001). AWQ’s standard-protocol perplexity is 14.89. Details are in Appendix F.")
para("On Colab, automatically chosen evaluation batch size changed the two quantized no-LoRA bases by at most 0.21 points (at most 7 of 1,172–10,042 questions) when the batch went from 11 to 13, because grouping and padding change the shapes of the FP16 matrix multiplications. A later run in the same session reproduced the original scores exactly at batch 11 and the re-score question for question at batch 13. The paired tests use the re-scored data; first-run scores remain in Tables 5 and 6. Details are in Appendix B.")

H2("4.4 LoRA recovery (Phase 2)")
para("Table 5 gives Phase 2 accuracy on Colab. All numbers in this section are percentages and come from the same Colab evaluation, so they can be compared with each other but not with the Phase 1 tables. As a check on the fake-quantized bases, the Colab no-LoRA scores agree with the laptop scores (Tables 2 and 3) within 0.6 points on every task for all three bases.")
fmt=lambda v: f"{v:.2f}"
rows=[["Base","LoRA"]+P2_TASKS+["Avg. (6 tasks)"]]
for b in P2_BASES:
    rows.append([b,"no"]+[fmt(v) for v in P2_NO[b]])
    rows.append([b,"yes"]+[f"{m:.2f} ± {s:.2f}" for m,s in P2_LORA[b]])
    rows.append(["","*Δ*"]+["*"+d+"*" for d in P2_DELTA[b]])
table(rows,[0.8,0.45,0.72,0.72,0.72,0.72,0.72,0.72,0.82],cap="Phase 2 zero-shot accuracy (%) on Colab (lm-evaluation-harness, Hugging Face backend). “yes” rows are mean ± standard deviation over seeds 0, 1 and 2; “no” rows are a single evaluation (log-likelihood scoring is deterministic). Δ = LoRA minus no LoRA, computed before rounding. ARC-C, ARC-E, HellaSwag and PIQA: acc_norm; WinoGrande and BoolQ: acc.",fs=8)
para("Without LoRA, RTN INT8 is 0.06 points below FP16 on the six-task average (60.35 vs 60.41), which is not significantly different from zero (paired 95% CI −0.29 to +0.17, p = 0.60; Table 7). The interval rules out an average INT8 loss larger than about 0.3 points on these tasks. With LoRA the two are identical (61.46; paired 95% CI −0.14 to +0.13). Because no gap is detectable on this Colab evaluation, we do not report a recovery percentage for INT8; LoRA raises INT8 by the same amount as FP16 (+1.10 vs +1.05), and both gains are significant (Holm p < 0.01).")
para("RTN INT4 loses 4.04 points without LoRA (56.36 vs 60.41). With LoRA it reaches 60.36 ± 0.07, about the level of plain FP16. That comparison flatters INT4, because the same recipe also improves FP16 by 1.05 points. Against FP16 + LoRA, the INT4 gap shrinks from 4.04 to 1.10 points, so LoRA closes about 73% of it (the gap table is in Appendix I). The remaining gap is about six times the largest seed standard deviation, and INT4 + LoRA is still below FP16 + LoRA on all six tasks (by 0.4 to 2.0 points). In the paired tests (Table 7), the loss, the INT4 LoRA gain and the remaining gap are all significant after Holm correction: −4.03 points (95% CI −4.73 to −3.33), +3.97 (+3.37 to +4.56) and −1.11 (−1.63 to −0.59). The remaining gap has the same direction in all three seeds (−1.29, −0.82, −1.21).")
figure(G+"fig_phase2.png",5.1,"Six-task average accuracy without and with LoRA for each base (Colab evaluation). Error bars: standard deviation over three seeds. Dashed line: FP16 without LoRA. The y-axis starts at 54%.")
para("The largest INT4 LoRA gain is on BoolQ (+9.62 points, from 55.60 to 65.22). Without LoRA, INT4 was below the majority-class rate on that task; some of the gain may be answer calibration rather than added knowledge. Held-out training loss tells the same story as accuracy, and the six-task average varies by at most 0.25 points across seeds (Appendix C and Appendix I).")

H2("4.5 LoRA speed and memory on the laptop")
para("Table 6 puts the six cells in one place: Colab six-task accuracy (Table 5) and laptop speed and memory with the seed-0 adapter applied unmerged. Raw per-round logs are in Appendix G.")
rows=[["Base","LoRA","Avg. acc. (%)","Decode (tok/s)","vs no LoRA","TTFT (ms)","Peak mem. (MB)"]]
for b in P2_BASES:
    off,on=P2_SPEED[b]["off"],P2_SPEED[b]["on"]
    rows.append([b,"no",f"{P2_NO[b][6]:.2f}",f"{off[0]:.1f}","—",f"{off[1]:.1f}","{:,}".format(off[2])])
    rows.append([b,"yes",f"{P2_LORA[b][6][0]:.2f}",f"{on[0]:.1f}",P2_SPEED_DELTA[b],f"{on[1]:.1f}","{:,}".format(on[2])])
table(rows,[0.95,0.5,1.0,1.05,0.85,0.8,1.05],cap="Six-cell grid: Colab six-task average accuracy (Table 5) and laptop efficiency with the seed-0 adapter applied unmerged (OpenVINO GenAI 2026.4.1, Arc GPU; same prompt and run counts as Table 4). LoRA accuracy is the three-seed mean from Table 5; speed and memory used only the seed-0 adapter. Decode and TTFT are means of the kept rounds (two for FP16 without LoRA; three otherwise). Peak memory is peak commit (private bytes), not the working-set metric of Table 4. The working-set figures in the text (5,578 to 5,595 MB for FP16) exclude FP16 + LoRA round 5 from the working-set mean only: that round’s working set was 5,313 MB while peak commit stayed at 6,287 MB, consistent with OS working-set trimming (Appendix G).",fs=8.5)
para("Unmerged LoRA slowed decoding by 8% for FP16 (24.9 to 22.9 tokens/s), 15% for INT8 (41.2 to 35.0) and 18% for INT4 (62.8 to 51.4). The adapter added little memory: 35 MB for FP16, 22 MB for INT8 and 11 MB for INT4 (peak commit). Peak working set showed the same pattern. The FP16 + LoRA working-set mean of 5,595 MB excludes a trimmed round, as noted under Table 6; tok/s, TTFT and commit for that round are kept in the averages. Phase 1 reported working set without LoRA, so we treat Tables 4 and 6 as separate measurements and do not mix the two memory metrics.")
para("Figure 3 plots accuracy against throughput for the six cells. INT4 + LoRA is still about 2.2× as fast as FP16 + LoRA (51.4 vs 22.9 tokens/s) and about 2.1× as fast as plain FP16, at about half the peak memory (3,108 vs 6,252–6,287 MB). Its six-task average (60.36%) is 1.10 points below FP16 + LoRA and about equal to plain FP16 (60.41%). The unmerged adapter therefore recovers most of the INT4 accuracy at a noticeable speed cost but almost no extra memory. Merging the adapter into the base and re-quantizing could in principle remove that speed cost, but it would change the INT4 weights; we did not test this.")
figure(G+"fig_lora_tradeoff.png",5.2,"Six-task average accuracy (Colab) versus decode throughput on the laptop for the six cells. Hollow markers: no LoRA. Filled markers: unmerged seed-0 LoRA. Lines connect the same base. Throughput is the mean of the kept rounds (Table 6).")

H2("4.6 Statistical tests")
para("Table 7 gives the paired tests for Phase 2 described in Section 3.3 (full results in Appendix B). Pairing matters because two models scored on the same questions mostly agree. Three effects of about one point are significant after Holm correction: the LoRA gain on FP16 (+1.05 points, 95% CI +0.64 to +1.46), the LoRA gain on INT8 (+1.11, +0.70 to +1.52) and the remaining gap between INT4 + LoRA and FP16 + LoRA (−1.11, −1.63 to −0.59). The two INT8-versus-FP16 comparisons remain non-significant, with narrow intervals (half-widths 0.23 and 0.13 points). Plain FP16 and INT4 + LoRA are not distinguishable (−0.05, 95% CI −0.66 to +0.53). These tests treat the test questions as the random sample and condition on our three trained adapters; three seeds cannot estimate training variance precisely. The same tests for the Phase 1 laptop models are in Appendix B (Tables 14 and 15).")
table(P2_STATS_ROWS,[1.6,0.5,1.0,0.6,0.5,0.75,1.15],cap="Paired significance tests for Phase 2 (Colab, six-task average accuracy, percentage points; first model minus second), on the re-scored per-question data. LoRA cells use per-question correctness averaged over seeds 0–2. 95% CI and p from a stratified paired bootstrap (10,000 resamples; smallest resolvable p ≈ 0.0002). Holm correction over the seven main comparisons; values marked < 0.01 are bounds limited by bootstrap resolution. McNemar: exact test pooled over all six tasks; for LoRA cells, the largest p of the three seeds. † Secondary comparison, not in the Holm family. Per-seed diffs: the same difference for each seed separately (the no-LoRA side is a single evaluation). Differences can differ from Table 5 by up to 0.02 points because of the re-score (Section 3.4).",fs=7.5)

# ---------- 5 Discussion and Limitations ----------
H1("5 Discussion and Limitations")
para("**RQ1 (accuracy).** At INT8, the four-task laptop average is 0.3 points below FP16 and is not significant in the paired test, consistent with prior reports that 8-bit weight quantization is close to lossless {huang, evalq}. On the six-task average the 0.42-point INT8 gap is significant (Holm p = 0.006), driven mainly by BoolQ. At INT4, AWQ loses less than RTN (2.1 vs 3.8 four-task points), and GPTQ does not improve on RTN. That GPTQ result is for NNCF 3.4.0 on this 1B model and should not be read as a general ranking of GPTQ against AWQ.")
para("**RQ2 (efficiency).** On the Arc integrated GPU, lower precision translated into real speed and memory gains, but smaller than the bit-width ratios: 1.7× decode throughput and 34% less peak memory at INT8, and about 2.6× throughput and 54–56% less memory at INT4 (RTN, AWQ). GPTQ INT4 was slower (2.4×) at the same size. TTFT also fell, by 40–60%.")
para("**RQ3 (LoRA).** For RTN bases, LoRA helps most where quantization hurt most. On the Colab evaluation, INT8 had no detectable loss to recover and gained the same as FP16. INT4 gained about four times as much, closing about 73% of its gap to FP16 + LoRA. This agrees with QLoRA {qlora}, which showed that LoRA on a 4-bit base can recover a large share of quantization loss; the remaining 1.10-point gap is significant. Whether recovery differs for GPTQ and AWQ bases is still open.")
para("**Practical guidance.** For this model and laptop, INT8 is a low-risk choice that yields 1.7× decode throughput and a third less peak memory at little accuracy cost (0.42 points on the six-task average; not significant on the four Table 2 tasks). When memory or speed requires INT4, AWQ gave the best accuracy of the methods tested at no cost in size or speed relative to RTN. If fine-tuning is planned anyway, a short LoRA run on an RTN INT4 base brings it close to, but still 1.10 points below, FP16 with the same adapter. On this laptop the unmerged adapter added only 11 MB of peak memory to INT4 and left it about 2.2× as fast as FP16 + LoRA, though 18% slower than INT4 without LoRA. These recommendations rest on one model, one machine, one prompt length, one adapter rank and unmerged adapters only. The status of the pre-specified hypotheses is in Appendix I.")
para("**Limitations.** All conclusions concern one ~1B model on one laptop with a specific driver and runtime, and one Colab T4 training setup. The Arc GPU shares memory and bandwidth with the CPU. Only weight-only quantization with one group size and one calibration set was tested; INT3 (supported in NNCF from version 3.3.0 {nncf}), pruning, including SparseGPT and Wanda {sparsegpt, wanda}, and mixed precision remain deferred, as do LoRA on the GPTQ and AWQ bases and merging-plus-re-quantization. Six zero-shot tasks and WikiText-2 perplexity do not capture open-ended generation quality, and the large BoolQ LoRA gain may partly reflect answer calibration. The GPTQ diagnosis is specific to Llama-3.2-1B and NNCF 3.4.0; the proposed mechanism was not verified, and the patched models are not pure INT4. Phase 2 paired tests condition on three trained adapters and treat the test questions as the random sample. Per-task tests are uncorrected. Future runs should fix the evaluation batch size.")

# ---------- 6 Conclusion ----------
H1("6 Conclusion")
para("We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 on an Intel Core Ultra 7 laptop with an integrated Arc GPU. RTN INT8 was close to FP16 on the four-task average (0.559 vs 0.562; not significant) with 1.7× decode throughput and 34% lower peak memory; on the six-task average the 0.42-point INT8 gap is significant and is driven mainly by BoolQ. INT4 delivered about 2.6× throughput and 54–56% lower peak memory at a 2–4 point accuracy cost; AWQ recovered about half of RTN’s loss, while NNCF GPTQ did not improve on RTN. GPTQ INT4 also failed on inputs without BOS (perplexity 424), a reproducible defect localized to one column of one weight matrix. In Phase 2, a fixed LoRA recipe trained on bases that bit-match the laptop models confirmed, at 1B scale, what QLoRA {qlora} showed for 4-bit models: on the Colab evaluation INT8 had no detectable loss to recover, and RTN INT4 + LoRA (60.36%) reached about the six-task average of plain FP16 (60.41%). Against FP16 + LoRA, INT4 closed about 73% of its 4.04-point gap and remained 1.10 points behind (significant after Holm correction). On the laptop, the unmerged adapter added only 11–35 MB of peak memory but reduced decode throughput by 8–18%. INT4 + LoRA remained about 2.2× as fast as FP16 + LoRA at about half the memory. LoRA on the GPTQ and AWQ bases is the natural next measurement.")

# ---------- References ----------
H1("References")
for i,k in enumerate(order,1):
    p=doc.add_paragraph(); r=p.add_run(f"[{i}] "+REFS[k]); r.font.size=Pt(9.5)
    p.paragraph_format.left_indent=Inches(0.3); p.paragraph_format.first_line_indent=Inches(-0.3); p.paragraph_format.space_after=Pt(3)
unused=[k for k in REFS if k not in order]
assert not unused, unused

# ---------- Appendix ----------
H1("Appendix")
H2("Appendix A. Prompt template for Dolly training")
para("Phase 2 LoRA training used databricks-dolly-15k with the Alpaca-style templates below, copied from the training notebook (lora_phase2.ipynb). The tokenizer prepends a beginning-of-sequence token (add_special_tokens = True). Loss is computed only on the response tokens and a final EOS token, not on the prompt.")
pre("""PROMPT_WITH_CONTEXT:
### Instruction:
{instruction}

### Context:
{context}

### Response:

PROMPT_NO_CONTEXT:
### Instruction:
{instruction}

### Response:
""")
para("If the Dolly context field is empty after stripping, the second template is used. Examples longer than 512 tokens are dropped, not truncated. The remaining examples are shuffled with data seed 1234; the first 200 form a held-out set and the next 5,000 are the training set (684,954 tokens; SHA-1 of the training original-index list 4b704cd207a56ebbe34b26fe8ca83fd42d13b35b, from results/data_subset.json). Every run uses this same subset.")
para("The notebook printed the first training example after this shuffle, truncated to 600 characters. That example has no context (Dolly original index 115). Item 5 of the response is cut off by the 600-character limit. A with-context example was not saved in the logs.")
pre("""<|begin_of_text|>### Instruction:
What are 5 potential uses for an empty La Croix can?

### Response:
1. Make your bike sound like it has an engine by wrapping the can around the back tire.
2. A percussion instrument to play along with friends in a jam session.  You could just crinkle it, hit it with a pen, or fill it with sand or debris and shake it.  
3. A baking tool.  You can used the bottom of the can to stamp out perfectly circle, uniform cookies from a rolled-out sheet of cookie dough.
4. A field razor.  Tear the can carefully to create an effective cutting device for emergency use.
5. """)

H2("Appendix B. Paired significance tests (Phase 2 and Phase 1) and Phase 1 per-task standard errors")
para("**Data.** All 12 Phase 2 models (three bases, each without LoRA and with the seed-0, 1 and 2 adapters) were re-scored on Colab with lm-evaluation-harness 0.4.13 and log_samples enabled, which records for every question whether the model was correct (acc_norm for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; acc for WinoGrande and BoolQ). The full test sets were used: 1,172, 2,376, 10,042, 1,267, 3,270 and 1,838 questions (19,965 in total). Question IDs were checked to be identical and in the same order for all 12 models.")
para("**Paired bootstrap.** For two cells A and B, let d be the per-question difference in correctness (for a LoRA cell, correctness is the mean over the three seeds, so d can take fractional values). The statistic is the six-task average of the per-task mean of d, in percentage points. Each of 10,000 resamples draws questions with replacement within each task (stratified), using the same draws for both cells and for every comparison (random seed 20261119). The 95% CI is the 2.5th–97.5th percentile of the resampled statistic. The two-sided p-value is 2 × min(P(d* ≤ 0), P(d* ≥ 0)) with add-one smoothing, so the smallest value is 2/10,001 ≈ 0.0002. The Holm step-down correction is applied across the seven main comparisons; the comparison of INT4 + LoRA with plain FP16 is secondary and not corrected.")
para("**McNemar.** For each comparison and task, and pooled over all six tasks, we count questions answered correctly by only the first model and by only the second, and run an exact two-sided binomial test on these discordant counts. Because McNemar needs a single model on each side, LoRA cells are tested once per seed (seed k with LoRA vs the same base without LoRA, or seed k vs seed k across bases), and we report the largest of the three p-values. The pooled test weights questions equally, so HellaSwag dominates it, while the bootstrap weights the six tasks equally; the two therefore answer slightly different questions and are reported side by side. Holm is applied to the pooled McNemar p-values as well.")
para("**Re-score check.** Table 10 compares each re-scored task accuracy with the first evaluation, as a net count of questions. The two non-zero rows are the quantized bases without LoRA, whose automatically chosen batch size changed from 11 to 13. A later check on the INT4 no-LoRA base confirmed the cause: at batch size 11 the original scores were reproduced exactly, and at batch size 13 the re-score was reproduced question for question (Table 11). Code: lora/stats/paired_bootstrap.py; results: lora/stats/paired_results.json; mismatch evidence: lora/stats/mismatch/.")
table(P2_APP_MAIN,[1.45,0.45,0.95,0.5,0.55,1.45,0.6,0.6],cap="Full results for the eight Phase 2 comparisons (six-task average, percentage points; first model minus second). McNemar columns use all 19,965 questions pooled; for LoRA cells, the discordant counts are listed for seeds 0, 1 and 2, and the p-value is the largest of the three. † Secondary comparison, outside the Holm family.",fs=7)
table(P2_APP_TASK,[1.6,0.8,0.55,1.15,0.65,1.0],cap="Per-task paired results (percentage points; first model minus second). Bootstrap CIs and p-values are per task and not corrected for multiple comparisons. McNemar p is the largest over seeds for LoRA cells. † Secondary comparison.",fs=7)
table(P2_APP_RESCORE,[1.15,0.95,0.5,0.5,0.65,0.7,0.5,0.5,0.8],cap="Re-score check: net change in the number of correctly answered questions (re-score minus first evaluation) per task, and change in the six-task average. Batch size: automatically chosen evaluation batch size in the first evaluation and in the re-score, from the saved Colab logs (n/l: first-run log not saved; these seven models were first evaluated in a later session that ran no training). All 12 first and re-score runs used the same library versions and a T4 GPU. A confirmatory re-run of the INT4 no-LoRA base at both batch sizes is in Table 11.",fs=7)
table(P2_APP_BSCHECK,[0.9,1.15,1.05,1.05,1.15,1.2],cap="Confirmation of the evaluation-batch-size effect on the INT4 no-LoRA model. Both checks were run in the same Colab T4 session, with the same library versions as the first evaluation and the re-score (PyTorch 2.14.1+cu130, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13) and 67,106 log-likelihood requests per run. At batch size 11, every task score matched the original saved accuracy exactly. At batch size 13, per-question correctness was identical to the re-score file on all six tasks. PIQA had the same accuracy at both batch sizes, but the per-question flags at batch 11 differed from the re-score, so some flips cancelled.",fs=7)

P1_APP_MAIN=[['Comparison', 'Diff.', '95% CI', 'p (boot.)', 'Holm p (boot.)', 'McNemar: first-only / second-only questions', 'McNemar p', 'Holm p (McNemar)'], ['RTN INT8 vs FP16', '−0.42', '[−0.69, −0.15]', '0.003', '0.006', '143/196', '0.005', '0.005'], ['RTN INT4 vs FP16', '−4.19', '[−4.89, −3.49]', '< 0.001', '< 0.01', '881/1795', '< 10⁻⁷⁰', '< 10⁻⁶⁹'], ['GPTQ INT4 vs FP16', '−5.01', '[−5.75, −4.29]', '< 0.001', '< 0.01', '921/2101', '< 10⁻¹⁰⁴', '< 10⁻¹⁰³'], ['AWQ INT4 vs FP16', '−2.34', '[−2.99, −1.70]', '< 0.001', '< 0.01', '800/1320', '< 10⁻²⁹', '< 10⁻²⁸'], ['AWQ INT4 vs RTN INT4', '+1.85', '[+1.29, +2.41]', '< 0.001', '< 0.01', '1060/666', '< 10⁻²⁰', '< 10⁻²⁰'], ['GPTQ INT4 vs RTN INT4', '−0.82', '[−1.57, −0.10]', '0.03', '0.03', '1271/1537', '< 10⁻⁶', '< 10⁻⁵'], ['AWQ INT4 vs GPTQ INT4', '+2.67', '[+1.94, +3.42]', '< 0.001', '< 0.01', '1747/1087', '< 10⁻³⁴', '< 10⁻³⁴']]
P1_APP_TASK=[['Comparison', 'Task', 'Diff.', '95% CI', 'p (boot.)', 'McNemar p'], ['RTN INT8 vs FP16', 'ARC-C', '−0.43', '[−1.28, +0.43]', '0.38', '0.44'], [' ', 'ARC-E', '+0.00', '[−0.55, +0.55]', '1.00', '1.00'], [' ', 'HellaSwag', '−0.02', '[−0.21, +0.16]', '0.88', '0.92'], [' ', 'WinoGrande', '−0.79', '[−1.82, +0.24]', '0.14', '0.16'], [' ', 'BoolQ', '−0.86', '[−1.50, −0.21]', '0.009', '0.01'], [' ', 'PIQA', '−0.44', '[−0.92, +0.05]', '0.10', '0.13'], ['RTN INT4 vs FP16', 'ARC-C', '−4.35', '[−6.48, −2.30]', '< 0.001', '< 10⁻⁴'], [' ', 'ARC-E', '−4.50', '[−5.93, −3.07]', '< 0.001', '< 10⁻⁹'], [' ', 'HellaSwag', '−4.33', '[−4.90, −3.76]', '< 0.001', '< 10⁻⁵²'], [' ', 'WinoGrande', '−2.21', '[−4.50, +0.08]', '0.07', '0.07'], [' ', 'BoolQ', '−7.95', '[−9.82, −6.08]', '< 0.001', '< 10⁻¹⁶'], [' ', 'PIQA', '−1.80', '[−3.16, −0.44]', '0.01', '0.01'], ['GPTQ INT4 vs FP16', 'ARC-C', '−3.41', '[−5.63, −1.28]', '0.003', '0.003'], [' ', 'ARC-E', '−5.35', '[−6.82, −3.87]', '< 0.001', '< 10⁻¹¹'], [' ', 'HellaSwag', '−6.10', '[−6.71, −5.51]', '< 0.001', '< 10⁻⁸⁸'], [' ', 'WinoGrande', '−2.53', '[−5.05, −0.08]', '0.04', '0.05'], [' ', 'BoolQ', '−9.42', '[−11.35, −7.49]', '< 0.001', '< 10⁻²⁰'], [' ', 'PIQA', '−3.26', '[−4.68, −1.85]', '< 0.001', '< 10⁻⁵'], ['AWQ INT4 vs FP16', 'ARC-C', '−2.82', '[−4.78, −0.94]', '0.006', '0.006'], [' ', 'ARC-E', '−2.02', '[−3.37, −0.67]', '0.004', '0.003'], [' ', 'HellaSwag', '−2.60', '[−3.09, −2.11]', '< 0.001', '< 10⁻²⁴'], [' ', 'WinoGrande', '−0.87', '[−3.08, +1.26]', '0.44', '0.48'], [' ', 'BoolQ', '−4.28', '[−5.93, −2.63]', '< 0.001', '< 10⁻⁶'], [' ', 'PIQA', '−1.47', '[−2.67, −0.27]', '0.02', '0.02'], ['AWQ INT4 vs RTN INT4', 'ARC-C', '+1.54', '[−0.26, +3.24]', '0.10', '0.11'], [' ', 'ARC-E', '+2.48', '[+1.35, +3.66]', '0.001', '< 10⁻⁴'], [' ', 'HellaSwag', '+1.73', '[+1.27, +2.20]', '< 0.001', '< 10⁻¹²'], [' ', 'WinoGrande', '+1.34', '[−0.47, +3.16]', '0.16', '0.17'], [' ', 'BoolQ', '+3.67', '[+2.20, +5.11]', '< 0.001', '< 10⁻⁵'], [' ', 'PIQA', '+0.33', '[−0.76, +1.41]', '0.59', '0.63'], ['GPTQ INT4 vs RTN INT4', 'ARC-C', '+0.94', '[−1.19, +3.16]', '0.42', '0.44'], [' ', 'ARC-E', '−0.84', '[−2.31, +0.63]', '0.28', '0.29'], [' ', 'HellaSwag', '−1.77', '[−2.35, −1.19]', '< 0.001', '< 10⁻⁸'], [' ', 'WinoGrande', '−0.32', '[−2.84, +2.21]', '0.82', '0.86'], [' ', 'BoolQ', '−1.47', '[−3.36, +0.43]', '0.13', '0.14'], [' ', 'PIQA', '−1.47', '[−2.88, −0.05]', '0.04', '0.05'], ['AWQ INT4 vs GPTQ INT4', 'ARC-C', '+0.60', '[−1.71, +2.82]', '0.63', '0.65'], [' ', 'ARC-E', '+3.32', '[+1.81, +4.88]', '< 0.001', '< 10⁻⁴'], [' ', 'HellaSwag', '+3.51', '[+2.91, +4.09]', '< 0.001', '< 10⁻³⁰'], [' ', 'WinoGrande', '+1.66', '[−0.95, +4.34]', '0.23', '0.24'], [' ', 'BoolQ', '+5.14', '[+3.36, +6.94]', '< 0.001', '< 10⁻⁷'], [' ', 'PIQA', '+1.80', '[+0.44, +3.21]', '0.02', '0.02']]
P1_SE_ROWS=[["Version","ARC-C","ARC-E","HellaSwag","WinoGrande","4-task average"],
 ["FP16","0.3677 (0.0141)","0.6195 (0.0100)","0.6433 (0.0048)","0.6172 (0.0137)","0.562"],
 ["RTN INT8","0.3635 (0.0141)","0.6195 (0.0100)","0.6431 (0.0048)","0.6093 (0.0137)","0.559"],
 ["RTN INT4","0.3242 (0.0137)","0.5745 (0.0101)","0.6000 (0.0049)","0.5951 (0.0138)","0.523"],
 ["GPTQ INT4","0.3336 (0.0138)","0.5661 (0.0102)","0.5823 (0.0049)","0.5919 (0.0138)","0.518"],
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
SPD_ROWS=[["Base","LoRA","Round","tok/s (mean of 30)","TTFT ms (mean of 30)","Peak commit (MB)","Peak working set (MB)","System CPU %","In Table 6?"],
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

para("Phase 1 per-task accuracies and standard errors. The next table reports the Phase 1 laptop scores from the official full lm-evaluation-harness runs (directories named fp16-full, and so on), with the standard errors stored by the harness as acc_norm_stderr (ARC-Challenge, ARC-Easy, HellaSwag) or acc_stderr (WinoGrande). These are the same runs as Table 2. The approximate SEs quoted in the main text (\u00b10.014, \u00b10.010, \u00b10.005) are these values rounded. The four-task average is copied from Table 2; the result files do not include an SE for the average. BoolQ and PIQA were scored in later qa runs, not in the full runs; they appear in the table after that, with WikiText-2 word perplexity. lm-eval reports the perplexity standard error as N/A, so none is given. The BOS-on-every-window perplexities are from a separate diagnostic (diag/bos_window_ppl.json).")
table(P1_SE_ROWS,[1.1,1.15,1.15,1.2,1.2,0.85],cap="Phase 1 zero-shot accuracy on the laptop (OpenVINO backend, full runs). Each cell is accuracy (standard error). ARC-C, ARC-E and HellaSwag: acc_norm; WinoGrande: acc. The four-task average is the mean of the four unrounded accuracies, rounded once to three decimals, and matches Table 2. Sources: results/fp16-full, rtn-int8-full, rtn-int4-full, gptq-int4-full and awq-int4-full.",fs=8)
table(P1_QA_ROWS,[0.95,1.15,1.25,1.6,1.55],cap="Phase 1 BoolQ, PIQA and WikiText-2 word perplexity. BoolQ (acc) and PIQA (acc_norm) are from the later qa runs, not the full runs, and include lm-eval standard errors. The standard-protocol perplexities (BOS only on the first window of each document) are from the ppl runs; the harness records those SEs as N/A. The last column is word perplexity with BOS prepended to every 2048-token window (diag/bos_window_ppl.json; no SE in that file). These columns match Table 3 and Table 31 (standard-protocol GPTQ is shown here to two decimals, 423.66, and as 423.7 in Table 31).",fs=7.5)
para("**Phase 1 paired tests.** The original laptop evaluations did not keep per-question logs. The five Phase 1 models were later re-run on the same laptop with lm-evaluation-harness 0.4.13, the OpenVINO backend, batch size 1 and seed 42, and with log_samples enabled. The per-question metrics (acc_norm for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; acc for WinoGrande and BoolQ) reproduce the accuracies in Tables 2 and 3 exactly (19,965 questions; document IDs were identical and in the same order for all five models). The tests below use the same stratified paired bootstrap (B = 10,000, seed 20261119) and exact McNemar procedure as Phase 2. The statistic is the six-task average, not the four-task average of Table 2. The Holm correction is applied to all seven comparisons: RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 each versus FP16, and the three INT4-method contrasts AWQ vs RTN INT4, GPTQ vs RTN INT4, and AWQ vs GPTQ. No comparison is left outside the family. Per-task tests are uncorrected. Code: phase1/stats/p1_paired.py; results: phase1/stats/p1_paired_results.json.")
para("On the four Table 2 tasks alone, the RTN INT8 versus FP16 interval includes zero (−0.31 points, 95% CI −0.67 to +0.05, p = 0.09), and GPTQ versus RTN INT4 is also not significant (−0.50, −1.44 to +0.43, p = 0.29). The six-task average, which adds BoolQ and PIQA, makes both contrasts significant; the INT8 gap is driven mainly by BoolQ (−0.86 points, p = 0.009), where 73 questions are unique errors for INT8 and 45 the other way, while ARC-Easy is tied at 1,472/2,376 with 21 cancelling flips. The GPTQ versus RTN gap is concentrated in HellaSwag (−1.77 points) together with smaller losses on BoolQ and PIQA. These paired tests do not isolate a mechanism for the HellaSwag drop; they are consistent with GPTQ already being the weakest INT4 method in Tables 2 and 3.")
table(P1_APP_MAIN,[1.45,0.45,0.95,0.5,0.5,1.5,0.6,0.65],cap="Paired significance tests for Phase 1 (laptop, OpenVINO, six-task average accuracy, percentage points; first model minus second), on the later per-question re-run that reproduces Tables 2 and 3. 95% CI and p from a stratified paired bootstrap (10,000 resamples; smallest resolvable p ≈ 0.0002). Holm correction over all seven comparisons; values marked < 0.01 are bounds limited by bootstrap resolution. McNemar: exact test pooled over all 19,965 questions (question-weighted, so HellaSwag dominates). Source: phase1/stats/p1_paired_results.json.",fs=7)
table(P1_APP_TASK,[1.55,0.8,0.55,1.15,0.65,0.75],cap="Per-task paired results for Phase 1 (percentage points; first model minus second). Bootstrap CIs and p-values are per task and not corrected for multiple comparisons. Source: phase1/stats/p1_paired_results.json.",fs=7)

H2("Appendix C. LoRA training curves")
para("All nine Phase 2 runs used the same recipe: LoRA r = 16, \u03b1 = 32, dropout 0.05, on q_proj, k_proj, v_proj and o_proj, adapters unmerged; one epoch on the 5,000-example Dolly subset of Appendix A; AdamW, learning rate 2 \u00d7 10\u207b\u2074, cosine schedule with 3% warm-up, effective batch 16 (4 \u00d7 4 accumulation), gradient clipping 1.0, gradient checkpointing, FP16 autocast, LoRA weights in FP32. Training seeds were 0, 1 and 2. GPTQ and AWQ LoRA runs have not been done.")
figure(G+"fig_app_lora_loss.png",6.3,"Training loss versus optimizer step for the nine Phase 2 runs. Each line is the mean over the three seeds of one base; the band is the min\u2013max range across those seeds. Loss was logged every 10 steps (the per-run train_log.csv files).")
para("The table below lists the summary written at the end of each run (results/training_summary.csv). INT4 holds a higher training and held-out loss than FP16 and INT8, which stay close to each other. Wall time was about 11 minutes per run on the T4 (652\u2013661 s), and peak GPU memory was 5.84 GB in every run.")
table(P2_TRAIN_ROWS,[1.3,0.5,1.15,1.2,1.0,1.0,1.15],cap="Phase 2 training summaries (results/training_summary.csv). Mean train loss is the average of the logged step losses; final logged loss is the last of those (every 10 steps). Holdout loss is the loss on the 200 held-out Dolly examples after training. Train time is wall time in seconds, excluding the later evaluation.",fs=8)

H2("Appendix D. Phase 2 base checks")
para("Fake-quantizer determinism. The next table is the box-side check in lora/stats/mismatch/fq_determinism.json (recorded under PyTorch 2.15.0+cpu). For every tested weight and mode, the fake-quantizer produced bit-identical integer codes and scales across CPU thread counts and memory layouts. Dequantized values (integer \u00d7 scale) were exactly representable in FP16 for INT4_SYM, but not for INT8_SYM or INT8_ASYM. A laptop inspection note records a further bit-exact match of integer codes, FP16 scales and zero points against the OpenVINO IRs on selected layers (layer-0 q_proj and k_proj, layer-7 down_proj, layer-15 gate_proj, and the embedding, for both RTN models). The raw comparison dump from that laptop check was not in the appendix zip.")
table(FQ_ROWS,[2.2,2.2,2.1],cap="Fake-quantizer determinism on the box (fq_determinism.json). \u201cDequantized values exact in FP16\u201d means the integer\u00d7scale result equals its FP16 rounding.",fs=8)
para("Colab versus laptop no-LoRA scores. The table below compares the Phase 2 Colab evaluations of the frozen bases (the no-LoRA eval JSON files under results/eval/) with the Phase 1 laptop scores (the full runs for ARC and WinoGrande; the qa runs for BoolQ and PIQA). Differences are Colab minus laptop, in percentage points, computed from the unrounded accuracies. Every task is within 0.6 points, as stated in Sections 3.4 and 4.4. The largest gap is WinoGrande on FP16 (0.55 points). HellaSwag on INT8 and INT4, and ARC-Easy on INT8, match to the reported precision.")
table(CL_ROWS,[1.0,1.1,1.1,1.1,1.6],cap="No-LoRA accuracy, laptop versus Colab (percentage points). Laptop ARC and WinoGrande: full runs; laptop BoolQ and PIQA: qa runs. Colab: first-run eval JSON files. Diff. = Colab minus laptop, from unrounded values.",fs=8)

H2("Appendix E. NNCF compression logs")
para("The Phase 1 export logs (results/export-rtn-int8.log, export-rtn-int4.log, export-gptq-int4.log, export-awq-int4.log) each contain NNCF\u2019s \u201cStatistics of the bitwidth distribution\u201d table. The same INT4 table appears in the GPTQ rebuild log (results/gptq-v2-build.log). The two tables below copy those rows. The logs do not name the single float layer (0% of parameters). For the INT4 models, the 1 of 114 layers in int8_asym is the backup-precision embedding (and tied lm_head), as in Section 3.2; it is 21% of all parameters and 0% of the ratio-defining (decoder) layers. All 112 decoder linear layers are int4_sym with group size 128. For RTN INT8, 113 of 114 layers are int8_sym per channel (100% of parameters); there is no int8_asym backup row. The logs also record that GPTQ itself took 1:08:36 on the default build and 1:13:20 on the rebuild; data-aware AWQ took 0:01:35; the final weight-compression step took 12 s (INT8), 13 s (RTN INT4 and AWQ), 15 s (default GPTQ) and 11 s (GPTQ rebuild).")
table(NNCF_INT4_ROWS,[2.2,2.0,2.3],cap="NNCF bitwidth distribution for RTN INT4, GPTQ INT4, AWQ INT4 and the GPTQ rebuild (identical table in all four of those export logs).",fs=8)
table(NNCF_INT8_ROWS,[2.2,2.0,2.3],cap="NNCF bitwidth distribution for RTN INT8 (results/export-rtn-int8.log).",fs=8)

H2("Appendix F. GPTQ and AWQ diagnostics")
para("The following paragraphs were in the main text and are collected here so that the main text can summarize them. Table and section numbers have been updated.")
para("Under the standard lm-evaluation-harness WikiText protocol, in which only the first window of each document begins with BOS, four models behave as expected, but GPTQ INT4 does not (Table 31): its word perplexity is 423.7, against 12.89–15.59 for the others. An identical rebuild of the GPTQ model gave 423.66, so the result is reproducible rather than a one-off. A token-level test on four passages of 2048 tokens confirms that the failure is tied to BOS: perplexity with vs without BOS was 9.58 vs 9.58 for FP16, 11.0 vs 11.1 for AWQ, and about 12 vs 3,044 for GPTQ. Without BOS, GPTQ predicts normally for about the first 32 tokens and then degrades. In free generation it still produces coherent, though more repetitive, text.")
para("**Localization.** Comparing each GPTQ weight matrix with its FP16 original isolates the defect to a single matrix, layers.1.mlp.down_proj, and within it to input column 2558. Weights in this column are grossly altered (for example, 0.052 became −0.781); the maximum absolute error there is 0.83, against at most 0.12 elsewhere. Replacing only this matrix with its RTN INT4 version restores the standard-protocol perplexity to 14.86 and lowers the headline perplexity from 20.68 to 18.67, below RTN INT4’s 19.98. This suggests that the defect also contributes to GPTQ’s weaker headline results, although we did not re-run the accuracy tasks on the patched model. The same stored-weight comparison also shows large raw differences for AWQ on some MLP up and down matrices (maximum 4.97, on the same down_proj). A later check on the OpenVINO IR confirmed the cause: those differences are AWQ’s per-channel smoothing scales, folded into down_proj input columns and up_proj output rows, not a quantization defect. The 4.97 entry is in input column 1417, not column 2558; dividing out the estimated scales brings the residual relative error back into the usual INT4 range. A leftover 2.29 maximum after a per-row median undo on layers.1.mlp.up_proj was an artifact of that estimator on a heavily scaled row, not an abnormal weight. AWQ’s standard-protocol perplexity is 14.89. The tables of the two-protocol comparison and the ablations are Tables 31 and 32.")
para("**Ablations.** The defect survives changes to the calibration data, the quantization grid and NNCF’s scale estimation (Table 9). With scale estimation, the error is no longer concentrated in a few weights but spread along column 2558, whose error norm is 26 times that of RTN; screening matrices by maximum absolute error therefore misses it, while replacing the same matrix again restores perplexity (14.78). Replacing layer 0’s down_proj instead, leaving the defective matrix in place, also removes the failure (15.14 for the default build in a fast check; 14.91 in a full evaluation of the scale-estimation build), suggesting that layer 0 feeds a large signal into the defective column.")
para("**A partial explanation.** In the FP16 model, the activation entering column 2558 has the same mean with and without BOS, but without BOS it spikes (maximum 0.78 vs 0.13 with BOS), mostly at early positions that behave like attention sinks. A plausible account is that GPTQ, calibrated on BOS-prefixed text in which this channel is small and steady, assigned large compensating errors to a column that appeared unimportant, which are then amplified when the channel spikes. This link is suggestive, not verified: we found no global correlation between per-column quantization error and the without/with-BOS activation ratio, and we did not measure the channel’s statistics on the calibration data itself.")
para("**Scope of the finding.** The finding holds for Llama-3.2-1B compressed with NNCF 3.4.0; we have not shown that it generalizes to GPTQ as an algorithm, nor that it constitutes a bug in NNCF. Patched models store the replaced matrix in FP16 and are not pure INT4 models. All accuracy evaluations in Sections 4.1 and 4.2 supplied BOS to every model, so the main comparison is unaffected; we therefore treat with-BOS perplexity as the headline metric and this as a secondary robustness result.")

para("This appendix summarises the laptop diagnosis in diag/FINDINGS.md and the accompanying JSON and log files. It is consistent with Section 4.3: the no-BOS WikiText collapse of NNCF GPTQ INT4 on this model is isolated to layers.1.mlp.down_proj, and within that matrix to input column 2558 (with a few smaller sibling columns). The remaining 111 GPTQ matrices are healthy. The proposed mechanism was not verified, and the patched models are not pure INT4, because the replaced matrix is stored as FP16 holding RTN values.")
para("The first of the two weight-error tables is from diag/weight_err.json (112 decoder linear matrices compared with the FP16 weights after dequantizing the OpenVINO IR). Relative Frobenius error is 0.105\u20130.148 for RTN and 0.115\u20130.177 for GPTQ (ratio 1.06\u20131.26). One outlier stands out: layers.1.mlp.down_proj has maximum absolute error 0.833 under GPTQ, against 0.119 in the next-highest GPTQ matrix and 0.096 in the worst RTN matrix. The figure below plots these maxima. The column table lists the worst input columns of that matrix (diag/outlier_check.log, diag/act_check.json). Column 2558 has GPTQ column-error norm 2.681 versus 0.080 for RTN. One entry in that column, output row 2023, is 0.0518 in FP16, 0.0531 after RTN, and \u22120.7812 after GPTQ. The main text\u2019s 0.83, 0.052 and \u22120.781 are these measurements rounded.")
figure(G+"fig_app_gptq_maxabs.png",6.3,"Maximum absolute weight error versus the FP16 matrix, for all 112 decoder linear layers (diag/weight_err.json). The highlighted point is layers.1.mlp.down_proj under GPTQ (0.833). The dashed line is 0.12; every other GPTQ matrix, and every RTN matrix, lies at or below 0.119.")
table(GPTQ_MAT_ROWS,[2.5,1.4,1.5,1.3],cap="Per-matrix weight error for GPTQ INT4 versus RTN INT4 (diag/weight_err.json). The second row is layers.1.mlp.up_proj, which has the second-largest GPTQ maximum absolute error. The third row is layers.15.mlp.up_proj (0.117).",fs=8)
table(GPTQ_COL_ROWS,[2.2,1.4,1.4,1.0,1.1],cap="Worst input columns of GPTQ layers.1.mlp.down_proj. Column-error norms are from diag/outlier_check.log; BOS and no-BOS max |activation| are from diag/act_check.json on an FP16 model (same four WikiText-2 passages as the per-token analysis). Column 2558 is only moderately active on average under BOS (mean |x| 0.0180, rank 678 of 8,192) but its no-BOS tail is larger (max 0.777 versus 0.131).",fs=7.5)
para("The ablation table collects perplexities from the diagnosis. Replacing only layers.1.mlp.down_proj with its RTN INT4 values restores the standard-protocol (lm-eval) word perplexity from 423.66 to 14.86, next to AWQ\u2019s 14.89 and RTN\u2019s 15.59; the BOS-on-every-window diagnostic for that patched model is 18.67, versus 20.68 for unpatched GPTQ (diag/bos_window_ppl.json). The same matrix is the defect in a C4-calibrated rebuild (305.69, worst column still 2558, max abs error 0.883) and in a scale-estimation build (202.6, restored to 14.78). Patching the upstream layers.0.mlp.down_proj also removes the four-passage collapse (15.14). A four-passage diagnostic (diag/se_bisect.json) shows the same pattern: GPTQ is fine for about the first 32 tokens without BOS and then degrades. What the files do not support is a verified mechanism, or any claim that the defect is general beyond Llama-3.2-1B and NNCF 3.4.0.")
table(GPTQ_ABL_ROWS,[3.2,2.0,1.7],cap="GPTQ ablation perplexities. The standard-protocol column is lm-eval WikiText-2 word perplexity at max_length = 2048 (BOS on the first window only): default models from the ppl runs; patched, C4 and scale-estimation models from diag/FINDINGS.md. The last column is a shorter diagnostic on four 2048-token WikiText-2 passages (diag/se_bisect.json and se_extra.json). A dash means that measurement was not in the files.",fs=7.5)
AWQ_OFFSET_ROWS=[['Estimator', 'Layer 0 product p50', 'Layer 1 product p50'], ['Median of ratios, real AWQ vs FP16', '0.9662', '0.9663'], ['Least-squares, real AWQ vs FP16', '1.0010', '1.0007'], ['Median of ratios, simulation (true product = 1)', '0.9668', '0.9668'], ['Least-squares, simulation (true product = 1)', '1.0013', '1.0011']]
AWQ_SCALE_ROWS=[['Matrix and site of max. raw difference', 'Raw max-abs (rel. err.)', 'Estimated scale (median AWQ/FP16 ratio)', 'After undoing, max-abs (rel. err.)'], ['layers.1.mlp.down_proj, row 400, input col 1417', '4.97 (0.391)', '15.67 on that input column', '0.184 (0.156)'], ['layers.15.mlp.down_proj, row 1159, input col 7258', '1.52 (0.314)', '4.79 on that input column', '0.125 (0.140)'], ['layers.0.mlp.up_proj, row 348, input col 1224', '2.19 (4.42)', 'per output row (max 32)', '0.021 (0.108)'], ['layers.1.mlp.up_proj, row 1462, col 658 (raw max)', '2.30 (4.37)', 'per output row (max 32.24)', '2.29 is at row 1473'], ['layers.1.mlp.up_proj, row 1473, col 923 (leftover)', '2.29 (raw difference)', '12.965 from paired down_proj 1473', '0.015']]
para("The same weight-error file also lists a large maximum absolute difference for AWQ on layers.1.mlp.down_proj (4.97) and on other MLP up and down matrices (20 of 32 have relative Frobenius error above 0.3; the largest relative error is 4.42, on layers.0.mlp.up_proj). A check that dequantized the AWQ OpenVINO IR the same way as weight_err.py, then compared it with the FP16 weights (diag/awq_scale_check.txt), confirms the cause. weight_err.py does not undo AWQ’s per-channel smoothing scales, which NNCF folds into up_proj and down_proj (diag/FINDINGS.md). All 80 attention and gate-projection matrices have exactly the same error statistics as RTN.")
para("On layers.1.mlp.down_proj the 4.97 maximum sits at output row 400, input column 1417 (FP16 \u22120.338, AWQ \u22125.3125). The median AWQ/FP16 ratio on that column is 15.67, so AWQ scaled the channel up about 15.7 times; the column’s L2 error is 9.62, against at most 0.81 on every other column. Column 2558 is not involved (rank 1,584 of 8,192, maximum absolute difference 0.053). Dividing each input column by its median ratio drops the maximum absolute difference from 4.97 to 0.184 and the relative error from 0.391 to 0.156. The same per-column undo on layers.15.mlp.down_proj (maximum 1.52 at column 7258, ratio 4.79) leaves 0.125 (relative error 0.140).")
para("For the two up_proj matrices the scales are on output rows, not input columns, with row-scale maxima of 32 and 32.24. Dividing out the per-row median ratio drops the relative error of layers.0.mlp.up_proj from 4.42 to 0.108 and the maximum absolute difference from 2.19 to 0.021. On layers.1.mlp.up_proj the relative error after a per-row median undo falls from 4.37 to 0.157, in the usual INT4 range, but the maximum absolute residual stays 2.29 (from 2.30). A follow-up check (diag/awq_residual_check.txt) locates that leftover 2.29 at output row 1473, input column 923 (FP16 −0.208008, stored AWQ −2.500000), not at the raw-max site (row 1462, column 658). The median of that row’s per-element AWQ/FP16 ratios is 0: at least half of the dequantized AWQ weights in the row are exactly 0 (every FP16 entry in the row is nonzero). The per-row estimator therefore returns no usable scale (s is recorded as NaN), and the 2.29 figure is just the raw, unscaled difference. The paired down_proj input column 1473 gives s_down = 0.0771277, so 1/s_down = 12.9655: AWQ scaled that intermediate channel by about 13 times. With a group step of 0.3125, original weights smaller than about half a step after scaling (roughly 0.012) quantize to zero, which accounts for the mass of exact zeros. Dividing the stored value by 12.9655 gives −0.1928 against FP16 −0.2080, an error of 0.015. That is ordinary INT4 size: the RTN error at this exact element is 0, and the RTN error at the leftover residual on layers.0.mlp.up_proj (row 4548, column 890, residual 0.0207) is 0.018. The inferred INT4 code, round(W_awq / group scale), is exactly −8.0000; the group scale is 0.3125, so 8 × 0.3125 = 2.5 equals both the stored |W_awq| and the largest |W_awq| in the group (columns 896–1023). Using 1/s_down, the pre-quant value would be 0.208008 × 12.9655 = 2.697, which exceeds the symmetric INT4 maximum of 2.5, so the 0.015 residual is a slight clip of the group maximum, not an abnormal weight. The leftover 2.29 after the per-row median is an artifact of that estimator on a heavily scaled row. Layer 0’s leftover of 0.0207 is about half a step (|qerr|/step = 0.516) and is ordinary rounding.")
para("The recovered scales are approximately reciprocal. Across paired up_proj rows and down_proj columns, the correlation of s_up with 1/s_down is 0.9996 (layer 0) and 0.9960 (layer 1), and the median of the product s_up × s_down, computed with the same per-row/column median of AWQ/FP16 ratios, is 0.9662 and 0.9663, about 3.4% below 1 (median |product − 1| is 0.0338 and 0.0337; 88.4% and 90.5% of the finite products lie within 5% of 1; on layer 1, 73 of 8,192 paired products are undefined because a median was unusable, including row 1473). A follow-up check (diag/awq_offset_check.txt) shows that this offset is an artifact of that median-of-ratios estimator after INT4 rounding and zeros, not a failure of the folded scales to be reciprocal. Applying exact-reciprocal scales to the real FP16 weights, then quantizing with the same INT4_SYM group-128 RTN rule, reproduces a median product of 0.9668 in both layers, even though the true product is 1. A least-squares estimator s = ⟨W_q, W_fp16⟩ / ⟨W_fp16, W_fp16⟩, restricted to nonzero codes, recovers a product of 1.0010 (layer 0) and 1.0007 (layer 1) on the real AWQ weights, and 1.0013 and 1.0011 on the simulation. A median that simply drops the exact-zero dequantized values overshoots to 1.0301 and 1.0298. The least-squares product can sit a little off 1 because of ordinary INT4 error; the check does not claim that every channel is an exact reciprocal.")
para("No extra scale was found on the activation path between up_proj and down_proj. The IR chain’s only Constant is the post_attention_layernorm weight (shape [1, 1, 2048]), which is applied to the residual stream before up_proj and gate_proj, not between them. The MLP Multiply that implements silu(gate)×up has no Constant inputs. Apparent extra Multiplies in a name-scoped scan are the weight-dequantization nodes (integer codes times the group scale). The export log shows data-aware AWQ followed by weight compression, with no Scale Estimation apply step and no LoRA-correction mention (the GPTQ scale-estimation ablation log, by contrast, records “Applying GPTQ with Scale Estimation”). The gate_proj matrices look like ordinary INT4, not a second smooth-scale: relative error 0.1061 and 0.1063, least-squares row-scale median 1.000 on both layers, and essentially no correlation with the up_proj scales (−0.0054 and 0.0025). The gate_proj median-of-ratios scale itself is 0.984, the same downward bias as on the paired up and down matrices. After a usable scale is applied, the leftover element errors sit in the usual INT4 range, which matches AWQ’s healthy standard-protocol WikiText-2 word perplexity of 14.89 (RTN 15.59; GPTQ 423.66) and is unrelated to GPTQ’s column-2558 defect.")
table(AWQ_SCALE_ROWS,[2.3,1.15,1.7,1.4],cap="AWQ stored-weight differences versus FP16, before and after dividing out a median per-column (down_proj) or per-row (up_proj) scale. On layers.1.mlp.up_proj the leftover 2.29 after the per-row median is at row 1473, not at the raw-max site (row 1462); that row’s median was 0, so the scale in the last row is 1/s_down from the paired down_proj column. Values are from diag/awq_scale_check.txt and diag/awq_residual_check.txt, which dequantized models/ov-awq-int4 the same way as weight_err.py. Relative error is the Frobenius norm of the difference divided by the Frobenius norm of the FP16 matrix.",fs=7.5)
table(AWQ_OFFSET_ROWS,[3.5,1.4,1.4],cap="Median product s_up × s_down on paired MLP channels in layers 0 and 1. The first two rows are estimated from the stored AWQ weights versus FP16. The last two apply exact-reciprocal scales to the real FP16 weights, then quantize with INT4_SYM group size 128 RTN and re-estimate; the true product is 1. Values are from diag/awq_offset_check.txt.",fs=8)


H2("Appendix G. Laptop LoRA timing logs")
para("Unmerged LoRA speed was measured on the same laptop as Phase 1, with OpenVINO GenAI 2026.4.1 on the Arc GPU, the seed-0 adapter, and scale 2.0. An earlier pass (speed-lora/status.txt, 18 November 2026, 21:44:54 to 22:18:42 as recorded) ran two rounds of each cell: first FP16, INT8, INT4, each off then on (rounds 1), then the same cells in reverse (rounds 2). Pass-1 JSON files do not record peak commit or system CPU. The numbers in Table 6 and in this appendix are from a second, balanced interleaved pass (speed-lora/run_pass2.ps1 and status2.txt, 22:20:09 to 23:05:31 the same day; the log records the Balanced power scheme). The script waited until a check2.status file contained ALLDONE, then, for each of FP16, RTN INT8 and RTN INT4, used the order off, on, on, off, off, on, stored as rounds 3\u20138. The pass recorded peak commit (private bytes) as well as peak working set, and system CPU load. One FP16 no-LoRA round (round 3) had system CPU 59.1% and was dropped, as stated in Section 4.5; the other kept rounds had system CPU between 7.4% and 10.6%.")
para("The first timing table lists every pass-2 round. The tok/s and TTFT columns are the mean of the 30 timed runs in that round (the tok_per_s_mean and ttft_ms_mean fields in the JSON), which is what Table 6 uses. The second table then gives the mean and sample standard deviation of those kept rounds. The tok/s, TTFT and peak-commit means match Table 6 (FP16 without LoRA: 24.9 tokens/s, 41.7 ms, 6,252 MB). The 5,595 MB working-set figure used in Section 4.5 is the mean of rounds 4 and 8 only (5,597.6 and 5,594.1 MB; unrounded mean 5,595.9 MB). Round 5 is kept in the tok/s, TTFT and commit averages but omitted from the working-set mean: its working set was 5,313 MB, while peak commit stayed at 6,287 MB (6,291 and 6,283 in the other two LoRA rounds), throughput matched round 4 (22.86 tokens/s) and system CPU was 9.6%. That pattern is working-set trimming by the OS, which the pass-2 script was written to detect (it added a commit counter after pass 1 had already shown the same drop, for example FP16 + LoRA round 2 at 3,218 MB). Including round 5 would pull the working-set mean down to 5,501 MB, which would understate the memory the process actually reserved. Decode throughput falls by 8% (FP16), 15% (INT8) and 18% (INT4) when the unmerged adapter is attached, as in the main text.")
table(SPD_ROWS,[0.7,0.45,0.45,0.85,0.9,0.85,0.95,0.6,0.85],cap="Pass-2 unmerged-LoRA timing, one row per round (the per-round speed-lora JSON files). tok/s and TTFT are the mean of 30 timed runs after 5 warm-ups, with a 128-token prompt and 128 new tokens, greedy, batch size 1. Memory is the peak process commit and working set in that round, in megabytes. System CPU is cpu_sys_pct_during (not recorded in pass 1). Round 3 of FP16 without LoRA is listed but was not used in Table 6.",fs=6.5)
table(SPD_SUM_ROWS,[0.85,0.5,0.75,1.15,1.15,1.1,1.3],cap="Means of the kept pass-2 rounds in the previous table. Standard deviations are the sample SD of the per-round means (n = 2 for FP16 without LoRA, n = 3 otherwise). Peak memory means are of the per-round peaks. For FP16 with LoRA, the working-set mean of all three kept rounds is 5,501 MB; 5,595 MB (Section 4.5 and the Table 6 note) is the mean of rounds 4 and 8 only (unrounded 5,595.9 MB), after excluding the trimmed round-5 working set.",fs=8)


# footer page numbers
fp=sec.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=fp.add_run(); f1=OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"),"begin"); r._r.append(f1)
r2=fp.add_run(); it=OxmlElement("w:instrText"); it.set(qn("xml:space"),"preserve"); it.text="PAGE"; r2._r.append(it)
r3=fp.add_run(); f2=OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"),"end"); r3._r.append(f2)
doc.core_properties.title="Shrinking a Small Language Model on a Laptop"; doc.core_properties.author="Dhruv Vinesh"


H2("Appendix H. Background on weight-only quantization, RTN, GPTQ, AWQ and LoRA")
para("This appendix records the background that was previously in the main text, so that the body can stay short. Nothing below changes the measurements in the main tables.")
H2("Numerical precision and weight-only quantization")
para("A model’s weights are stored as numbers of fixed width. FP16 uses 16 bits per weight; INT8 and INT4 integers represent 256 and 16 distinct values respectively. Storage is approximately (number of weights × bits) / 8 bytes, giving roughly 2.5, 1.24 and 0.62 GB for a 1.24B-weight model at 16, 8 and 4 bits, before overheads. In *weight-only* quantization, weights are stored as integers with a scale factor (real value ≈ integer × scale) and are dequantized during computation, while activations remain in higher precision. With *symmetric, grouped* quantization, each group of consecutive weights (here 128) shares one scale and the integer grid is centered on zero; one 16-bit scale per 128 weights adds 0.125 bits per weight. In practice, toolkits often keep embeddings and the output layer at a higher backup precision, so measured sizes differ from this arithmetic.")
H2("Round-to-nearest (RTN)")
para("RTN computes each group’s scale from its largest absolute weight and rounds every weight independently to the nearest grid point. It is data-free and fast, but it does not account for how many small rounding errors combine to shift a layer’s output.")
H2("GPTQ")
para("GPTQ {gptq} uses a small calibration set to record the inputs each linear layer receives. It quantizes the layer’s weights column by column and, after each step, updates the remaining unquantized weights to compensate for the error just introduced, using approximate second-order information (an estimate of the output’s sensitivity to each weight derived from the calibration inputs). The result lies on the same integer grid as RTN but minimizes the layer’s output error on the calibration inputs. Because the correction is fitted to those inputs, inputs whose statistics differ from the calibration data are not directly constrained.")
H2("AWQ")
para("AWQ {awq} observes that a small fraction of weight channels, those multiplying input channels with large activations, dominate quantization error. Rather than storing these channels at higher precision, AWQ scales them up by per-channel factors before rounding and folds the inverse scale into the preceding computation, leaving the full-precision function unchanged while reducing the relative rounding error of salient weights. Scale factors are chosen by a search that minimizes output error on calibration data; no gradient-based training is involved.")
H2("LoRA and QLoRA")
para("LoRA {lora} freezes a weight matrix *W* and learns a low-rank update, so the adapted layer computes with *W* + (α/*r*)·*BA*, where *B* and *A* have inner dimension *r* (the rank) and α scales the update. Only *A* and *B* are trained, typically a fraction of a percent of the model’s parameters. QLoRA {qlora} applies LoRA to a frozen 4-bit base model, showing that fine-tuning through a quantized backbone can match 16-bit fine-tuning on their evaluations. An adapter can be kept separate at inference or merged into the weights; merging into a quantized model requires re-quantization, which can reduce accuracy {qalora}.")



H2("Appendix I. Experimental details")
para("This appendix holds the experimental detail moved out of Section 3 and the extra result tables that the main text now summarizes: metric definitions, pre-specified hypotheses, software versions, the Phase 2 training protocol and re-score, pipeline figures, the GPTQ two-protocol and ablation tables, the Phase 2 gap table, the training-loss table, and the hypothesis-status table.")
figure(G+"fig_phase1.png",6.3,"Phase 1 pipeline: one FP16 reference and four compressed versions, all evaluated identically on the laptop.")
figure(G+"fig_phase2_method.png",6.3,"Phase 2 as run: frozen bases that bit-match the laptop models, LoRA trained with three seeds on a Colab T4, and every base evaluated with and without its adapter by the same route.")

para("**Efficiency metrics.** The definitions below are those used for all laptop timing runs (Phase 1 and the later unmerged-LoRA pass).")
table([
 ["Metric","Definition","Protocol"],
 ["Size on disk","Total bytes of OpenVINO model files (plus adapter, if any)","Measured once per version"],
 ["Peak memory","Peak working set of the inference process (psutil)","Includes GPU allocations, because the Arc GPU uses shared system RAM"],
 ["Time to first token (TTFT)","Latency from request to first generated token (prefill)","128-token prompt; 5 warm-up runs discarded; median of 30 timed runs"],
 ["Decode throughput","Tokens per second while generating 128 new tokens","Greedy decoding; same runs as TTFT"],
],[1.4,2.5,2.6],cap="Efficiency metrics.")

para("**Pre-specified hypotheses** (expectations, not findings):")
table([
 ["#","Hypothesis"],
 ["H1","RTN INT8 is within noise of FP16."],
 ["H2","RTN loses accuracy at INT4 (originally also: more at INT3; deferred)."],
 ["H3","GPTQ and AWQ lose less than RTN at INT4 (no prior ordering of GPTQ vs AWQ is assumed)."],
 ["H4","50% magnitude pruning may approach chance accuracy before LoRA, with little or no speed gain (deferred with pruning)."],
 ["H5","Disk size and peak memory decrease with bit width, by less than the ideal ratios."],
 ["H6","Decode throughput increases at lower bit widths, by less than the size reduction; TTFT changes less."],
 ["H7a","LoRA narrows the gap to FP16 + LoRA, most for versions that lost the most."],
 ["H7b","Competing: LoRA trained on general instruction data recovers little or no zero-shot accuracy."],
 ["H8","Merging and re-compressing (Route B) retains less of any LoRA gain than unmerged evaluation."],
],[0.5,6.0],cap="Pre-specified hypotheses (expectations, not findings).")

para("**Software versions and fixed settings.** Phase 1 ran on the laptop; Phase 2 training and accuracy evaluation ran on Colab. The Intel Arc GPU driver version was not recorded.")
table([
 ["Component","Version / setting"],
 ["Python (laptop / Colab)","3.11.9 / 3.13"],
 ["PyTorch (laptop / Colab)","2.14.1+cpu / 2.14.1+cu130"],
 ["transformers (laptop / Colab)","5.5.4 / 4.57.6"],
 ["PEFT (Colab)","0.21.2"],
 ["OpenVINO / OpenVINO GenAI","2026.4.1 / 2026.4.1"],
 ["NNCF","3.4.0"],
 ["optimum / optimum-intel","2.3.0 / 2.2.0"],
 ["lm-evaluation-harness (laptop / Colab)","0.4.13 (OpenVINO backend) / 0.4.13 (Hugging Face backend)"],
 ["datasets / numpy (laptop)","5.0.1 / 2.4.6"],
 ["Colab evaluation batch size","automatic (11 or 13; see Section 3.4 and Appendix B)"],
 ["Windows (laptop)","Windows 11, build 10.0.26300"],
 ["Intel Arc GPU driver","not recorded"],
 ["Colab GPU","NVIDIA T4 (all 9 runs)"],
 ["LoRA","r = 16, α = 32, dropout 0.05; q/k/v/o projections; not merged"],
 ["Training","1 epoch; AdamW, lr 2 × 10⁻⁴, cosine, 3% warm-up; batch 4 × 4 accumulation = 16; max length 512 (longer dropped); FP16 autocast; gradient checkpointing"],
 ["Dolly subset; seeds","5,000 train + 200 held out, data seed 1234; training seeds 0, 1, 2"],
],[3.0,3.5],cap="Software versions and fixed settings. Phase 1 ran on the laptop; Phase 2 training and accuracy evaluation ran on Colab. The Intel Arc GPU driver version was not recorded.")


para("**Phase 2 protocol (full).** The shorter account in Section 3.4 is the official summary; the paragraphs below are the original method text, with cross-references updated.")
para("**Bases.** LoRA needs a PyTorch model on an NVIDIA GPU, so for FP16, RTN INT8 and RTN INT4 we built frozen PyTorch copies on Colab. For the quantized bases, each weight is quantized and immediately dequantized (integer × scale) with the same scheme as the laptop model (Section 3.2), and scales are rounded to FP16 as OpenVINO stores them. The resulting integer codes and scales bit-match those in the laptop models. This is the usual QLoRA-style setup {qlora}: the quantized weights stay frozen, they are stored in FP16 on the GPU (fake quantization), and no integer kernels are used. As a further check, the Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Section 4.4).")
para("**LoRA recipe.** Adapters were trained with PEFT {peft}: rank *r* = 16, α = 32, dropout 0.05, on the query, key, value and output projections of every attention layer. Adapters are kept separate and never merged into the base. The recipe is identical for every base; only the frozen base differs. FP16 + LoRA is the control that separates recovery of quantization loss from the general effect of fine-tuning on Dolly.")
para("**Data and training.** We used databricks-dolly-15k {dolly} with an Alpaca-style prompt (instruction, optional context, response). Examples longer than 512 tokens were dropped (not truncated); the rest were shuffled with data seed 1234, giving 200 held-out and 5,000 training examples. Every run used the same 5,000 examples, and loss was computed on response tokens only. Training ran for one epoch (about 310 optimizer steps) with AdamW, learning rate 2 × 10⁻⁴, a cosine schedule with 3% warm-up, an effective batch of 16 (4 per step × 4 accumulation steps), gradient clipping at 1.0, gradient checkpointing and FP16 autocast, with LoRA weights kept in FP32. Each base was trained with seeds 0, 1 and 2 (9 runs); the seed changes only the adapter initialization, dropout and batch order.")
para("**Evaluation.** All Phase 2 models, with and without adapters, were evaluated on Colab with lm-evaluation-harness 0.4.x {lmeval} (Hugging Face backend), zero-shot, with BOS, on the full sets of the six tasks and with the same metrics as Phase 1 (Section 3.3). The no-LoRA bases were re-evaluated on Colab rather than taken from Phase 1, so every with/without comparison uses the same software. Because the runtime and numerics differ from OpenVINO, Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages.")
para("**Re-scoring for paired tests.** The first Colab evaluations saved only task-level scores. To run paired tests, all 12 Phase 2 models were evaluated a second time with per-question logging, using the same notebook, adapters, GPU type (T4) and library versions (PyTorch 2.14.1, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13; recorded in every result file). Ten of the 12 models reproduced every task score exactly. The two quantized bases without LoRA did not: per task they changed by at most 7 of 1,172–10,042 questions (at most 0.21 points, INT4 BoolQ), and their six-task averages changed by −0.003 (INT8) and +0.016 (INT4) points (Table 10). The cause is the evaluation batch size. We let lm-evaluation-harness pick the largest batch that fits in free GPU memory, and that choice depends on the session. The saved Colab logs show batch 11 for all four models whose first-run logs survive. In the re-score, the batch stayed at 11 for the two FP16 models, which matched exactly, and became 13 for the two quantized no-LoRA bases, which are the only two that changed. A different batch size changes how requests are grouped and padded, and so the shapes of the FP16 matrix multiplications and their rounding. This can flip questions whose answer options have almost equal log-likelihoods. The quantized weights themselves are reproducible: the fake-quantization code is deterministic (bit-identical across CPU thread counts and repeated runs), and all six adapters on quantized bases, which load the same quantized weights, reproduced exactly. A later check, in the same Colab session and with the same library versions, confirmed this directly on the INT4 no-LoRA base (all six zero-shot tasks, 67,106 log-likelihood requests per run). Re-running at batch size 11 reproduced the original saved scores exactly on all six tasks. Re-running at batch size 13 reproduced the re-score question for question. PIQA had the same accuracy at both batch sizes, but the per-question correctness flags differed, so some flips cancelled out (Table 11). Tables 5 and 6 keep the first-run scores. The paired tests (Table 7, Appendix B) use the re-scored per-question data, so their differences can differ from Table 5 by up to 0.016 points (e.g., the INT4 LoRA gain is +3.97 rather than +3.99). The change only affects comparisons that involve the INT8 or INT4 base without LoRA, where it is under 3% of the half-width of the 95% CI, and it does not change any conclusion. Future runs should fix the evaluation batch size.")
para("**Laptop speed and memory.** Speed and memory of the unmerged adapters were measured on the same laptop as Phase 1, with OpenVINO GenAI 2026.4.1 (LLMPipeline) on the Arc GPU. Each seed-0 adapter was applied unmerged at runtime through openvino_genai.Adapter and AdapterConfig, with a scaling factor of 2.0, equal to the training ratio α / *r* = 32 / 16. We used the seed-0 adapter of each base because speed is not expected to depend on the seed. The prompt and timing protocol matched the Phase 1 speed test (Appendix I): a 128-token prompt, 5 warm-up runs, then 30 timed runs per round, greedy decoding of 128 new tokens. For each base we used a balanced interleaved order (off, on, on, off, off, on), giving three rounds per cell. The reported memory figure is peak commit (private bytes), which is not affected by Windows trimming the working set; peak working set is also recorded. One FP16 no-LoRA round was excluded because system CPU load during it was 59% (versus about 8–10% in the other rounds). An earlier non-interleaved pass showed run-to-run drift and working-set trimming, which is why we adopted the balanced order and the commit-memory metric. Raw logs are under quant-study/speed-lora on the laptop.")

para("**GPTQ two-protocol perplexity and ablations.** These tables accompany the diagnosis in Appendix F.")
table([
 ["Version","Word ppl, BOS at every window (headline)","Word ppl, standard protocol (BOS at document start only)"],
 ["FP16","15.46","12.89"],["RTN INT8","15.50","12.92"],["RTN INT4","19.98","15.59"],
 ["GPTQ INT4","20.68","423.7 (rebuild: 423.66)"],["AWQ INT4","18.22","14.89"],
 ["GPTQ INT4, layers.1.mlp.down_proj replaced by RTN†","18.67","14.86"],
],[2.4,2.0,2.1],cap="WikiText-2 word perplexity (max_length = 2048) under the two BOS protocols. Values should be compared within a column. † Diagnostic model; the replaced matrix is stored in FP16, so this model is not purely INT4.",fs=8.5)
table([
 ["GPTQ INT4 build","Standard-protocol word ppl","Defect","Word ppl after replacing layers.1.mlp.down_proj by RTN"],
 ["Default (WikiText-2 calibration, symmetric)","423.7","Column 2558; max abs error 0.83","14.86"],
 ["C4 calibration","305.7","Same column","not measured"],
 ["Asymmetric quantization","229.6","Same column","not measured"],
 ["Scale estimation enabled","202.6","Error spread along column 2558 (column error norm 26× RTN); missed by max-error screening","14.78"],
],[1.75,1.1,2.2,1.45],cap="Ablations of the GPTQ INT4 build (standard WikiText-2 protocol; lower is better).",fs=8.5)

para("**Phase 2 gap to FP16**, comparing like with like (no LoRA vs no LoRA; LoRA vs LoRA). Gaps computed before rounding.")
table([
 ["Base","Gap to FP16, no LoRA (points)","Gap to FP16 + LoRA, with LoRA (points)","Share of gap closed"],
 ["RTN INT8","0.06","0.00","not meaningful (no detectable gap)"],
 ["RTN INT4","4.04","1.10","≈ 73%"],
],[1.1,1.7,1.9,1.8],cap="Six-task average gap to FP16, comparing like with like (no LoRA vs no LoRA; LoRA vs LoRA). Gaps computed before rounding.",fs=8.5)

para("**Per task.** The largest INT4 gain is on BoolQ (+9.62 points, from 55.60 to 65.22). Without LoRA, INT4 was below BoolQ’s majority-class rate of about 62%, so part of this gain may come from LoRA restoring the model’s yes/no answer calibration or format rather than restoring knowledge. This is a hypothesis we have not tested; one check would be to compare the share of “yes” answers before and after LoRA. The other five tasks gain between +1.2 (WinoGrande) and +5.0 (ARC-Easy) points, so the recovery is not limited to BoolQ. The BoolQ gain itself is far outside sampling error (paired 95% CI +7.77 to +11.08 points on the re-scored data, where the INT4 base scores 55.81 rather than 55.60), which shows the change is real but not what caused it. The remaining gap between INT4 + LoRA and FP16 + LoRA is clearest on HellaSwag (−2.04 points, 95% CI −2.47 to −1.61; McNemar p < 10⁻¹¹ in every seed). On BoolQ it is −1.22 points (95% CI −2.31 to −0.12, uncorrected bootstrap p = 0.03), but McNemar does not confirm it in every seed (largest p = 0.10). On the other four tasks it is negative but not individually significant (Table 9).")
table([
 ["Base","Train time per run (s)","Peak GPU memory (GB)","Mean train loss","Held-out Dolly loss"],
]+[[b,P2_TRAIN[b][0],P2_TRAIN[b][1],f"{sum(P2_TRAIN[b][2])/3:.2f}",f"{sum(P2_TRAIN[b][3])/3:.3f} ({min(P2_TRAIN[b][3]):.3f}–{max(P2_TRAIN[b][3]):.3f})"] for b in P2_BASES],
 [1.1,1.3,1.3,1.2,1.6],cap="Phase 2 training (three seeds per base, Colab T4). Held-out loss is on the 200 held-out Dolly examples, mean (range) over seeds. Peak GPU memory is the same for every base because the fake-quantized weights are stored in FP16; it is not deployment memory (see Table 4 for Phase 1 laptop memory and Table 6 for memory with LoRA).",fs=8.5)
para("**Training loss and cost.** The held-out loss tells the same story as accuracy (Table 34). FP16 and INT8 reach the same held-out loss (1.695 and 1.696), while INT4 stays higher (1.768). LoRA therefore does not fully undo INT4 rounding even on text like its own training data. Each run took about 11 minutes and 5.84 GB of GPU memory for every base. The equal cost is expected in our setup, which stores every base in FP16; with true 4-bit storage, as in QLoRA {qlora}, the INT4 base would need less memory.")
para("**Seed variation.** Across seeds, the six-task average varies by at most 0.25 points (standard deviation), and per-task standard deviations are at most 0.53 points. The differences discussed above are larger than this training noise. The seed spread does not include the sampling error of the test questions; that is covered by the paired tests in Section 4.6. For every comparison that is significant there, all three seeds show a difference in the same direction.")


para("**Status of the pre-specified hypotheses** after Phase 1, the RTN part of Phase 2, and the laptop LoRA speed measurements.")
table([
 ["#","Outcome"],
 ["H1","Mostly supported on the four Table 2 tasks (paired 95% CI includes 0; average −0.003). On the six-task average the 0.42-point loss is significant (Holm p = 0.006), driven mainly by BoolQ."],
 ["H2","Supported for INT4 (average −0.038; INT3 deferred)."],
 ["H3","Partly supported. AWQ lost less than RTN (average −0.021 vs −0.038, consistent across tasks); GPTQ did not (−0.043)."],
 ["H4","Deferred (pruning out of scope)."],
 ["H5","Supported for peak memory and INT4 disk size; INT8 disk size was essentially ideal (50% of FP16)."],
 ["H6","Decode part supported (1.7× and 2.4–2.6× vs 2× and ~3.2× smaller files); TTFT part not supported (TTFT fell 40–60%)."],
 ["H7a","Supported for RTN bases: INT4 gained +3.99 vs +1.05 (FP16) and +1.10 (INT8), and its gap to FP16 + LoRA fell from 4.04 to 1.10 points. All three gains and the remaining 1.10-point gap are significant in the paired test (Holm p < 0.01; Table 7). GPTQ and AWQ pending."],
 ["H7b","Not supported for RTN bases: LoRA on Dolly raised zero-shot accuracy on all six tasks for all three bases."],
 ["H8","Not tested: adapters were run unmerged on the laptop; merging and re-compressing was not."],
],[1.0,5.5],cap="Status of the pre-specified hypotheses after Phase 1, the RTN part of Phase 2, and the laptop LoRA speed measurements.")



# ---------- PDF ----------
OUT=str(_HERE/"paper_draft.docx")
doc.save(OUT)
print("refs",len(order),"tables",tcount[0],"figures",fcount[0])
import subprocess
subprocess.run(["soffice","--headless","--convert-to","pdf","--outdir",str(_HERE),OUT],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("pdf written")
