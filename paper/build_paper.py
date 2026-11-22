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
ACC={  # ARC-C acc_norm, ARC-E acc_norm, HellaSwag acc_norm, WinoGrande acc, average of the four, disk MB
 "FP16":(".3677",".6195",".6433",".6172",".562","2,357"),
 "RTN INT8":(".3635",".6195",".6431",".6093",".559","1,180"),
 "RTN INT4":(".3242",".5745",".6000",".5951",".524","730"),
 "GPTQ INT4":(".3336",".5661",".5823",".5919",".519","730"),
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
 "FP16":    {"off":(24.9,41.8,6253,5578),"on":(22.9,44.3,6287,5595)},
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
p=para("Dhruv Vinesh, Cypress Ranch High School, Cypress, Texas · Draft, November 2026 · *Phase 1 (compression benchmark) complete; Phase 2 (LoRA accuracy and laptop efficiency) complete for FP16, RTN INT8 and RTN INT4; LoRA on GPTQ and AWQ pending.*",align=WD_ALIGN_PARAGRAPH.CENTER)

H1("Abstract")
para("**Problem.** Small language models are increasingly run locally on consumer laptops, where memory capacity and memory bandwidth, not raw compute, limit what can be deployed. Post-training weight compression reduces both but can cost accuracy, and most published comparisons rely on NVIDIA GPUs, larger models, or perplexity alone. It is therefore unclear what standard methods deliver for a ~1B-parameter model on an ordinary laptop without a discrete GPU.")
para("**What we did.** This is a controlled benchmark, not a new method. We compress Llama-3.2-1B with OpenVINO NNCF 3.4.0 weight compression into four variants, round-to-nearest (RTN) INT8 and RTN, GPTQ and AWQ INT4 (group size 128, one shared calibration set), and compare them with the FP16 reference. Accuracy is measured zero-shot with lm-evaluation-harness on six tasks and WikiText-2 perplexity; latency, throughput and peak memory are measured with OpenVINO GenAI on the integrated Arc GPU of an Intel Core Ultra 7 laptop that shares 16 GB of system RAM. In a second phase, we train the same LoRA adapter recipe (three seeds) on FP16, RTN INT8 and RTN INT4 bases on a Colab T4 GPU, using frozen PyTorch copies whose weights bit-match the laptop models, and evaluate with and without the adapter on the same six tasks. We then measure the seed-0 adapter of each base, left unmerged, on the same laptop and with the same OpenVINO GenAI timing protocol as Phase 1. LoRA on the GPTQ and AWQ bases has not yet been run.")
para("**Key results (Phase 1).** RTN INT8 is nearly lossless (mean accuracy over ARC-Challenge, ARC-Easy, HellaSwag and WinoGrande 0.559 vs 0.562 for FP16) while raising decode throughput 1.7× and lowering peak memory by 34%. INT4 raises throughput about 2.6× (RTN, AWQ) and lowers peak memory by 54–56%, at a cost of 2–4 accuracy points. AWQ recovers about half of RTN INT4’s loss (0.541 vs 0.524), whereas NNCF GPTQ does not improve on RTN (0.519). As a secondary robustness finding, the GPTQ INT4 model’s WikiText-2 word perplexity rises from 12.9–15.6 (other models) to 424 when evaluation windows lack a beginning-of-sequence (BOS) token; we localize this to one input column of a single weight matrix.")
para("**Key results (Phase 2).** QLoRA {qlora} already showed that LoRA on a 4-bit base recovers much of the lost accuracy; we check this under controlled conditions on a 1B model. On the six-task average (Colab evaluation), RTN INT8 is not significantly different from FP16 (60.35% vs 60.41%; paired 95% CI for the difference −0.29 to +0.17 points). RTN INT4 loses 4.04 points (56.36%); with LoRA it reaches 60.36 ± 0.07%, about the level of FP16 without LoRA. Because the same LoRA recipe also raises FP16 by 1.05 points, the fair comparison is with FP16 + LoRA: there, the INT4 gap shrinks from 4.04 to 1.10 points (about 73% closed), and a held-out training-loss gap remains. In paired tests over questions with Holm correction, the 4.04-point loss, the LoRA gains on all three bases and the remaining 1.10-point gap (95% CI 0.59 to 1.63 points) are all significant; INT8 vs FP16, with or without LoRA, is not. Unmerged LoRA added only 11–34 MB of peak memory on the laptop but reduced decode throughput by 8% (FP16), 15% (INT8) and 18% (INT4). Even so, INT4 + LoRA remained about 2.2× as fast as FP16 + LoRA at about half the memory, with accuracy 1.10 points lower and about equal to plain FP16.")
para("**Takeaway.** On this laptop, INT8 is a low-risk default, and AWQ was the most accurate of the three INT4 methods at an identical footprint. A short, cheap LoRA run closes most, but not all, of the RTN INT4 accuracy gap, at a modest speed cost and almost no extra memory. Whether recovery differs for GPTQ and AWQ bases is still open.")

# ---------- 1 Introduction ----------
H1("1 Introduction")
para("Running a language model locally on a laptop keeps data on the device, works offline and avoids per-query cost. The obstacle is memory. A model with about 1.24 billion weights, such as Llama-3.2-1B {llama}, occupies roughly 2.5 GB in 16-bit floating point (FP16). On a consumer laptop whose integrated GPU shares 16 GB of system memory with the operating system and other applications, that footprint matters. It also matters for speed. Generating each token requires reading nearly all weights from memory, so autoregressive decoding is typically limited by memory bandwidth, and storing weights in fewer bits can, in principle, make generation faster as well as smaller.")
para("Post-training compression is the standard response. Weight-only quantization methods such as GPTQ {gptq} and AWQ {awq} reduce weights to 4 bits with modest accuracy loss on large models, and benchmark studies report that 8-bit weight quantization is close to lossless across methods {huang, evalq}. Degradation grows at lower bit widths, and Huang et al. {huang} found the LLaMA3 family noticeably more sensitive to low-bit quantization than earlier models. Benchmarking work also shows that configuration choices, including calibration data, change outcomes enough to confound comparisons between methods {llmc}. Low-rank adaptation (LoRA) {lora} offers a way to recover lost accuracy: QLoRA {qlora} already showed that LoRA trained on top of a frozen 4-bit model recovers much of the accuracy lost to quantization and can match 16-bit fine-tuning on some benchmarks. Recovery is not guaranteed, however: LoRA fine-tuning of quantized LLaMA3-8B on general instruction data did not restore, and sometimes lowered, zero-shot accuracy {huang}, and merging an adapter into a quantized model and re-quantizing costs accuracy {qalora}.")
para("Three gaps motivate this study. First, comparisons of compression methods are usually run on NVIDIA GPUs with CUDA kernels; when efficiency is reported at all, it is rarely *measured* on consumer non-NVIDIA hardware with the toolchain a laptop user would actually use. Second, small models (around 1B parameters) at 4 bits are less studied than 7B+ models, even though they are the models most likely to run on laptops. Third, LoRA recovery is typically examined for one compression scheme at a time, so it is unknown whether the same adaptation recipe repairs a rounded, an error-compensated and an activation-aware model to the same degree. We therefore ask:")
bullets([
 "**RQ1.** Which weight-only method (RTN, GPTQ or AWQ) best preserves the zero-shot accuracy of a ~1B model at INT4, relative to FP16 and an RTN INT8 reference?",
 "**RQ2.** What memory and speed benefits are actually realized on a consumer Intel laptop (Core Ultra 7 155H, integrated Arc GPU) running OpenVINO, as opposed to those implied by bit width?",
 "**RQ3.** Does a fixed LoRA recipe recover accuracy differently depending on the compression applied to the frozen base model? (Phase 2: answered here for RTN INT8 and RTN INT4; GPTQ and AWQ bases pending.)",
])
para("**Contribution.** This paper is a controlled benchmark rather than a new method. On a single 1B model, we provide a matched comparison of RTN, GPTQ and AWQ that pairs zero-shot accuracy with measured on-device efficiency on a consumer Intel laptop (OpenVINO/NNCF), and we report a reproducible robustness failure of one GPTQ-compressed model when inputs lack a BOS token. Phase 2 measures how much a fixed LoRA recipe recovers on bases whose weights bit-match the laptop models, with FP16 + LoRA as the control, and reports the speed and memory of the unmerged adapters on the same laptop; it does not test whether LoRA can recover quantization loss in general, which QLoRA {qlora} has already shown. We do not claim this is the first such comparison; our literature review covers the closest work we identified (Section 3) rather than the full literature.")
para("Section 2 gives background, Section 3 reviews related work, Section 4 describes the method, Section 5 reports results (Phase 1 in Sections 5.1–5.4, Phase 2 in Sections 5.5–5.6), and Sections 6–8 discuss, state limitations and conclude.")

# ---------- 2 Background ----------
H1("2 Background")
H2("2.1 Numerical precision and weight-only quantization")
para("A model’s weights are stored as numbers of fixed width. FP16 uses 16 bits per weight; INT8 and INT4 integers represent 256 and 16 distinct values respectively. Storage is approximately (number of weights × bits) / 8 bytes, giving roughly 2.5, 1.24 and 0.62 GB for a 1.24B-weight model at 16, 8 and 4 bits, before overheads. In *weight-only* quantization, weights are stored as integers with a scale factor (real value ≈ integer × scale) and are dequantized during computation, while activations remain in higher precision. With *symmetric, grouped* quantization, each group of consecutive weights (here 128) shares one scale and the integer grid is centered on zero; one 16-bit scale per 128 weights adds 0.125 bits per weight. In practice, toolkits often keep embeddings and the output layer at a higher backup precision, so measured sizes differ from this arithmetic.")
H2("2.2 Round-to-nearest (RTN)")
para("RTN computes each group’s scale from its largest absolute weight and rounds every weight independently to the nearest grid point. It is data-free and fast, but it does not account for how many small rounding errors combine to shift a layer’s output.")
H2("2.3 GPTQ")
para("GPTQ {gptq} uses a small calibration set to record the inputs each linear layer receives. It quantizes the layer’s weights column by column and, after each step, updates the remaining unquantized weights to compensate for the error just introduced, using approximate second-order information (an estimate of the output’s sensitivity to each weight derived from the calibration inputs). The result lies on the same integer grid as RTN but minimizes the layer’s output error on the calibration inputs. Because the correction is fitted to those inputs, inputs whose statistics differ from the calibration data are not directly constrained.")
H2("2.4 AWQ")
para("AWQ {awq} observes that a small fraction of weight channels, those multiplying input channels with large activations, dominate quantization error. Rather than storing these channels at higher precision, AWQ scales them up by per-channel factors before rounding and folds the inverse scale into the preceding computation, leaving the full-precision function unchanged while reducing the relative rounding error of salient weights. Scale factors are chosen by a search that minimizes output error on calibration data; no gradient-based training is involved.")
H2("2.5 LoRA and QLoRA")
para("LoRA {lora} freezes a weight matrix *W* and learns a low-rank update, so the adapted layer computes with *W* + (α/*r*)·*BA*, where *B* and *A* have inner dimension *r* (the rank) and α scales the update. Only *A* and *B* are trained, typically a fraction of a percent of the model’s parameters. QLoRA {qlora} applies LoRA to a frozen 4-bit base model, showing that fine-tuning through a quantized backbone can match 16-bit fine-tuning on their evaluations. An adapter can be kept separate at inference or merged into the weights; merging into a quantized model requires re-quantization, which can reduce accuracy {qalora}.")

# ---------- 3 Related Work ----------
H1("3 Related Work")
para("**Quantization benchmarks.** Several studies benchmark post-training quantization systematically. Li et al. {evalq} evaluate weight, activation and KV-cache quantization across 11 model families from 125M to 180B parameters and five categories of tasks, providing broad guidance on which components tolerate low precision. Huang et al. {huang} evaluate ten post-training quantization and LoRA fine-tuning methods on LLaMA3 at 1–8 bits and report non-negligible degradation, particularly at ultra-low bit widths. LLMC {llmc} introduces a toolkit for fair comparison and shows that calibration data, algorithm choice and data format all affect results. Our study is narrower in scope, one 1B model and weight-only formats, but adopts LLMC’s lesson by fixing calibration data and group size across methods, and adds measured efficiency on a consumer Intel laptop rather than reporting accuracy alone.")
para("**Pruning versus quantization.** Jaiswal et al. {kick} introduce LLM-KICK and argue that perplexity hides real capability loss: on knowledge-intensive tasks, pruning methods degrade substantially, sometimes at 25–30% sparsity, whereas quantization holds up better. Focusing on small language models (0.5–3.8B), Zhou, Kurz and Zhao {zhou} compare SparseGPT, Wanda, GPTQ and AWQ and find that quantization consistently outperforms pruning, recommending AWQ in particular. Structural pruning followed by brief LoRA-based recovery training is explored by LLM-Pruner {pruner}. Pruning is outside the scope of the present study and is left to future work (Section 8).")
para("**LoRA-based recovery of compressed models.** QLoRA {qlora} established that LoRA on a frozen 4-bit base can be effective, and subsequent work modifies the quantization–adaptation interface. LoftQ {loftq} jointly quantizes the model and initializes the low-rank adapter so that it starts by compensating quantization error; IR-QLoRA {irqlora} improves low-bit LoRA fine-tuning by retaining more information during quantization; QA-LoRA {qalora} uses group-wise operators so that the merged model remains quantized, avoiding the accuracy loss of post-merge re-quantization. Huang et al. {huang} report that QLoRA-style fine-tuning of 4-bit LLaMA3-8B on general instruction data did not recover zero-shot accuracy. Rather than proposing a new adaptation method, we hold the LoRA recipe fixed and vary only the compression method of the frozen base, asking whether recovery depends on how the model was compressed, with FP16 + LoRA as the control.")

# ---------- 4 Method ----------
H1("4 Method")
para("The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. Phase 2 trained LoRA adapters on Google Colab for the FP16, RTN INT8 and RTN INT4 bases, with settings fixed before training, and then measured those unmerged adapters on the same laptop. LoRA on the GPTQ and AWQ bases is pending.")
H2("4.1 Model and hardware")
para("We use the base (not instruction-tuned) Llama-3.2-1B {llama} (meta-llama/Llama-3.2-1B), with approximately 1.24 billion parameters (≈2.5 GB in FP16), obtained from Hugging Face under its gated license; the planned fallback to Qwen2.5-0.5B {qwen} was not needed. Compression and all efficiency measurements ran on a single Dell Inspiron laptop: Intel Core Ultra 7 155H, integrated Intel Arc GPU sharing 16 GB of system RAM, Windows 11 (build 10.0.26300), no discrete NVIDIA GPU. Because CUDA-dependent tooling is unavailable on this machine, models were converted to and executed with OpenVINO {openvino} on the Arc GPU, and compressed with NNCF 3.4.0 {nncfdoc} (OpenVINO 2026.4.1, exported through optimum-intel 2.2.0). LoRA training (Section 4.3) ran on Google Colab with an NVIDIA T4 GPU.")
H2("4.2 Five model versions")
para("Table 1 lists the five versions. All compressed versions use NNCF weight-only compression. RTN INT8 uses symmetric quantization with one scale per output channel (NNCF mode INT8_SYM) for every weight matrix, including the embedding. The INT4 versions use symmetric, group-wise quantization with group size 128 (INT4_SYM, ratio 1.0) for all decoder linear layers, while the embedding and the tied output layer (lm_head) are kept at a backup precision of INT8_ASYM, one scale and zero point per channel. GPTQ and AWQ use the same 128 calibration samples drawn once from WikiText-2 {wikitext}. Optional NNCF refinements (e.g., scale estimation) were disabled for all methods so that each algorithm is evaluated in isolation, and the same backup-precision setting was used for all INT4 versions, which have identical on-disk sizes. GPTQ and AWQ were not evaluated at INT8 because prior work reports near-lossless 8-bit quantization across methods {huang, evalq}; RTN INT8 serves as the near-lossless reference point. Calibration data, symmetry and scale estimation were varied only in the diagnostic ablations of Section 5.4.")
table([
 ["#","Version","Method","Precision","Calibration","Notes"],
 ["1","FP16","none","FP16","—","Uncompressed reference"],
 ["2","RTN INT8","Round-to-nearest","INT8, per channel","—","Near-lossless reference point"],
 ["3","RTN INT4","Round-to-nearest","INT4, g128","—","Data-free baseline"],
 ["4","GPTQ INT4","GPTQ","INT4, g128","128 × WikiText-2","Error-compensating"],
 ["5","AWQ INT4","AWQ","INT4, g128","same 128","Activation-aware scaling"],
],[0.3,1.0,1.4,1.0,1.2,1.6],cap="Model versions. All compressed versions: NNCF 3.4.0 weight-only compression. INT8: symmetric, per output channel (INT8_SYM), all weights including the embedding. INT4: symmetric with group size 128 (g128, INT4_SYM, ratio 1.0); embedding and tied lm_head at INT8_ASYM per channel.")
para("**Scope.** The original plan also included INT3 variants of all three methods and a 50% unstructured magnitude-pruned model. These, together with mixed-precision formats, are deferred until the INT4 grid and Phase 2 are complete, so that the core comparison is finished before the design is widened (Section 8).")
figure(G+"fig_phase1.png",6.3,"Phase 1 pipeline: one FP16 reference and four compressed versions, all evaluated identically on the laptop.")
H2("4.3 LoRA training on Colab (Phase 2)")
para("**Bases.** LoRA needs a PyTorch model on an NVIDIA GPU, so for FP16, RTN INT8 and RTN INT4 we built frozen PyTorch copies on Colab. For the quantized bases, each weight is quantized and immediately dequantized (integer × scale) with the same scheme as the laptop model (Section 4.2), and scales are rounded to FP16 as OpenVINO stores them. The resulting integer codes and scales bit-match those in the laptop models. This is the usual QLoRA-style setup {qlora}: the quantized weights stay frozen, they are stored in FP16 on the GPU (fake quantization), and no integer kernels are used. As a further check, the Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Section 5.5).")
para("**LoRA recipe.** Adapters were trained with PEFT {peft}: rank *r* = 16, α = 32, dropout 0.05, on the query, key, value and output projections of every attention layer. Adapters are kept separate and never merged into the base. The recipe is identical for every base; only the frozen base differs. FP16 + LoRA is the control that separates recovery of quantization loss from the general effect of fine-tuning on Dolly.")
para("**Data and training.** We used databricks-dolly-15k {dolly} with an Alpaca-style prompt (instruction, optional context, response). Examples longer than 512 tokens were dropped (not truncated); the rest were shuffled with data seed 1234, giving 200 held-out and 5,000 training examples. Every run used the same 5,000 examples, and loss was computed on response tokens only. Training ran for one epoch (about 310 optimizer steps) with AdamW, learning rate 2 × 10⁻⁴, a cosine schedule with 3% warm-up, an effective batch of 16 (4 per step × 4 accumulation steps), gradient clipping at 1.0, gradient checkpointing and FP16 autocast, with LoRA weights kept in FP32. Each base was trained with seeds 0, 1 and 2 (9 runs); the seed changes only the adapter initialization, dropout and batch order.")
para("**Evaluation.** All Phase 2 models, with and without adapters, were evaluated on Colab with lm-evaluation-harness 0.4.x {lmeval} (Hugging Face backend), zero-shot, with BOS, on the full sets of the six tasks and with the same metrics as Phase 1 (Section 4.4). The no-LoRA bases were re-evaluated on Colab rather than taken from Phase 1, so every with/without comparison uses the same software. Because the runtime and numerics differ from OpenVINO, Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages.")
para("**Re-scoring for paired tests.** The first Colab evaluations saved only task-level scores. To run paired tests, all 12 Phase 2 models were evaluated a second time with per-question logging, using the same notebook, adapters, GPU type (T4) and library versions (PyTorch 2.14.1, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13; recorded in every result file). Ten of the 12 models reproduced every task score exactly. The two quantized bases without LoRA did not: per task they changed by at most 7 of 1,172–10,042 questions (at most 0.21 points, INT4 BoolQ), and their six-task averages changed by −0.003 (INT8) and +0.016 (INT4) points (Appendix Table 18). The cause is the evaluation batch size. We let lm-evaluation-harness pick the largest batch that fits in free GPU memory, and that choice depends on the session. The saved Colab logs show batch 11 for all four models whose first-run logs survive. In the re-score, the batch stayed at 11 for the two FP16 models, which matched exactly, and became 13 for the two quantized no-LoRA bases, which are the only two that changed. A different batch size changes how requests are grouped and padded, and so the shapes of the FP16 matrix multiplications and their rounding. This can flip questions whose answer options have almost equal log-likelihoods. The quantized weights themselves are reproducible: the fake-quantization code is deterministic (bit-identical across CPU thread counts and repeated runs), and all six adapters on quantized bases, which load the same quantized weights, reproduced exactly. A later check, in the same Colab session and with the same library versions, confirmed this directly on the INT4 no-LoRA base (all six zero-shot tasks, 67,106 log-likelihood requests per run). Re-running at batch size 11 reproduced the original saved scores exactly on all six tasks. Re-running at batch size 13 reproduced the re-score question for question. PIQA had the same accuracy at both batch sizes, but the per-question correctness flags differed, so some flips cancelled out (Appendix Table 19). Tables 10–13 keep the first-run scores. The paired tests (Table 14, Appendix B) use the re-scored per-question data, so their differences can differ from Table 10 by up to 0.016 points (e.g., the INT4 LoRA gain is +3.97 rather than +3.99). The change only affects comparisons that involve the INT8 or INT4 base without LoRA, where it is under 3% of the half-width of the 95% CI, and it does not change any conclusion. Future runs should fix the evaluation batch size.")
para("**Laptop speed and memory.** Speed and memory of the unmerged adapters were measured on the same laptop as Phase 1, with OpenVINO GenAI 2026.4.1 (LLMPipeline) on the Arc GPU. Each seed-0 adapter was applied unmerged at runtime through openvino_genai.Adapter and AdapterConfig, with a scaling factor of 2.0, equal to the training ratio α / *r* = 32 / 16. We used the seed-0 adapter of each base because speed is not expected to depend on the seed. The prompt and timing protocol matched the Phase 1 speed test (Table 2): a 128-token prompt, 5 warm-up runs, then 30 timed runs per round, greedy decoding of 128 new tokens. For each base we used a balanced interleaved order (off, on, on, off, off, on), giving three rounds per cell. The reported memory figure is peak commit (private bytes), which is not affected by Windows trimming the working set; peak working set is also recorded. One FP16 no-LoRA round was excluded because system CPU load during it was 59% (versus about 8–10% in the other rounds). An earlier non-interleaved pass showed run-to-run drift and working-set trimming, which is why we adopted the balanced order and the commit-memory metric. Raw logs are under quant-study/speed-lora on the laptop.")
figure(G+"fig_phase2_method.png",6.3,"Phase 2 as run: frozen bases that bit-match the laptop models, LoRA trained with three seeds on a Colab T4, and every base evaluated with and without its adapter by the same route.")
H2("4.4 Benchmarks and metrics")
para("Accuracy was measured zero-shot with lm-evaluation-harness {lmeval} using its OpenVINO backend, on the full evaluation sets of ARC-Challenge and ARC-Easy {arc} (length-normalized accuracy, acc_norm), HellaSwag {hellaswag} (acc_norm) and WinoGrande {winogrande} (accuracy). The unweighted mean of these four tasks is the primary accuracy summary. As an extended set, we report BoolQ (accuracy), PIQA (acc_norm) and WikiText-2 {wikitext} word perplexity with max_length = 2048. Approximate chance levels are 25% (ARC, HellaSwag) and 50% (WinoGrande, PIQA); BoolQ’s majority-class rate is about 62%. Because log-likelihood scoring is deterministic, each model–task pair was evaluated once. MMLU, considered initially, was dropped: a 1B model scores near chance on it, and evaluation was very costly on this hardware.")
para("**BOS handling in perplexity.** The standard lm-evaluation-harness WikiText task scores each document in rolling 2048-token windows and places a beginning-of-sequence (BOS) token only before the document’s first window. Our headline perplexity instead prepends BOS to every 2048-token window, matching the BOS-prefixed inputs used in all accuracy evaluations. The standard protocol, in which most windows lack BOS, is reported as a robustness check in Section 5.4.")
para("Efficiency was measured on the laptop only (Table 2), with OpenVINO GenAI {genai} on the Arc GPU, including a later pass with unmerged LoRA adapters (Section 4.3).")
table([
 ["Metric","Definition","Protocol"],
 ["Size on disk","Total bytes of OpenVINO model files (plus adapter, if any)","Measured once per version"],
 ["Peak memory","Peak working set of the inference process (psutil)","Includes GPU allocations, because the Arc GPU uses shared system RAM"],
 ["Time to first token (TTFT)","Latency from request to first generated token (prefill)","128-token prompt; 5 warm-up runs discarded; median of 30 timed runs"],
 ["Decode throughput","Tokens per second while generating 128 new tokens","Greedy decoding; same runs as TTFT"],
],[1.4,2.5,2.6],cap="Efficiency metrics.")
H2("4.5 Fairness controls and statistics")
para("All laptop measurements used the same machine, plugged in, in the same power mode, on the Arc GPU, with batch size 1, the same prompts, identical software versions and no other workloads. Compression settings and calibration data are matched across methods (Section 4.2), and the LoRA recipe, data and seeds are matched across bases (Section 4.3).")
para("For each task we report accuracy with the standard error (SE) from lm-evaluation-harness; approximate SEs are ±0.014 for ARC-Challenge and WinoGrande, ±0.010 for ARC-Easy and ±0.005 for HellaSwag. For Phase 2 we test differences between models with paired tests over questions (Section 5.7; details in Appendix B), because every model answers the same questions. The primary test is a paired bootstrap of the difference in six-task average accuracy (the unweighted mean of the six task accuracies, as in Table 10) with 10,000 resamples. Each resample draws questions with replacement within each task, and the same questions are used for both models. For a LoRA cell, each question is scored as the share of the three seeds that answered it correctly. We report the percentile 95% CI and a two-sided bootstrap p-value, and we apply the Holm correction across the seven main Phase 2 comparisons. As a secondary check we run exact McNemar tests on the questions that exactly one of the two models answered correctly. These need one model per side, so for LoRA cells they are run once per seed and we report the largest of the three p-values. These tests treat the test questions as the random sample. They do not cover training randomness beyond our three seeds, so we also list per-seed differences. Phase 1 per-question outputs were not saved, so Phase 1 differences are interpreted against the per-task SEs only. LoRA results are reported as the mean ± standard deviation across three seeds; effects smaller than the seed-to-seed spread are treated as noise. The seed spread reflects training randomness only, not the sampling error of the test questions. Table 3 lists the pre-specified hypotheses; parts concerning INT3 and pruning are deferred with those conditions, and outcomes are summarized in Section 6.")
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
H2("4.6 Reproducibility")
para("Scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs are at https://github.com/theLegend35/quant-study-llama32-1b. Table 4 records versions and fixed settings. Phase 1 (laptop) and Phase 2 (Colab) used different transformers versions (5.5.4 and 4.57.6 respectively). Phase 2 accuracy scores are compared only with each other, not with the Phase 1 averages; the only cross-phase accuracy statement is a check that the Colab no-LoRA scores agree with the laptop scores within 0.6 points on every task (Sections 4.3 and 5.5).")
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
 ["Colab evaluation batch size","automatic (11 or 13; see Section 4.3)"],
 ["Windows (laptop)","Windows 11, build 10.0.26300"],
 ["Intel Arc GPU driver","not recorded"],
 ["Colab GPU","NVIDIA T4 (all 9 runs)"],
 ["LoRA","r = 16, α = 32, dropout 0.05; q/k/v/o projections; not merged"],
 ["Training","1 epoch; AdamW, lr 2 × 10⁻⁴, cosine, 3% warm-up; batch 4 × 4 accumulation = 16; max length 512 (longer dropped); FP16 autocast; gradient checkpointing"],
 ["Dolly subset; seeds","5,000 train + 200 held out, data seed 1234; training seeds 0, 1, 2"],
],[3.0,3.5],cap="Software versions and fixed settings. Phase 1 ran on the laptop; Phase 2 training and accuracy evaluation ran on Colab. The Intel Arc GPU driver version was not recorded.")

# ---------- 5 Results ----------
H1("5 Results")
para("Sections 5.1–5.4 report Phase 1 (laptop, OpenVINO). Section 5.5 reports Phase 2 accuracy (LoRA, Colab) for FP16, RTN INT8 and RTN INT4. Section 5.6 reports the speed and memory of those unmerged adapters on the laptop.")
H2("5.1 Accuracy")
rows=[["Version","ARC-C","ARC-E","Hella- Swag","Wino- Grande","Average (4 tasks)","Δ avg vs FP16","Size on disk (MB)"]]
for m in MODELS:
    a=ACC[m]; d=num(a[4])-num(ACC["FP16"][4])
    rows.append([m,a[0],a[1],a[2],a[3],a[4],"—" if m=="FP16" else f"{d:+.3f}".replace("-","−"),a[5]])
table(rows,[1.05,0.65,0.65,0.7,0.7,0.85,0.85,0.95],cap="Zero-shot accuracy (fraction correct). ARC-C, ARC-E and HellaSwag: acc_norm; WinoGrande: acc. Approximate SEs: ±0.014 (ARC-C, WinoGrande), ±0.010 (ARC-E), ±0.005 (HellaSwag).",fs=8.5)
para("RTN INT8 is within one SE of FP16 on every task in Table 5, and its four-task average is 0.3 points lower (0.559 vs 0.562), while its size on disk is halved (1,180 vs 2,357 MB). All INT4 versions lose accuracy: the average falls by 3.8 points for RTN, 4.3 for GPTQ and 2.1 for AWQ. The RTN INT4 losses on ARC-Challenge, ARC-Easy and HellaSwag (4.4, 4.5 and 4.3 points) are each more than three SEs. AWQ is above RTN INT4 on all four tasks and closes about 45% of RTN’s average gap to FP16 (1.7 of 3.8 points); its per-task advantages are individually modest relative to the SEs (largest on HellaSwag and ARC-Easy) but consistent in direction. GPTQ is not better than RTN: it is slightly higher on ARC-Challenge (within one SE) and lower on the other three tasks.")
rows=[["Version","BoolQ (acc)","PIQA (acc_norm)","WikiText-2 word ppl ↓"]]+[[m]+list(EXT[m]) for m in MODELS]
table(rows,[1.4,1.4,1.4,1.9],cap="Extended evaluation. WikiText-2 word perplexity uses max_length = 2048 with BOS at the start of every 2048-token window (headline protocol; see Section 5.4 for the standard protocol).",fs=8.5)
para("The extended tasks show the same ordering (Table 6). INT8 remains close to FP16 on BoolQ, PIQA and perplexity (15.50 vs 15.46). Among INT4 versions, AWQ has the lowest perplexity (18.22, vs 19.98 for RTN and 20.68 for GPTQ) and the highest BoolQ and PIQA scores. BoolQ is the most sensitive task: INT4 costs 4.3 (AWQ) to 9.4 (GPTQ) points, and RTN and GPTQ INT4 fall below the task’s majority-class rate of about 62%, as FP16 itself is only slightly above it. PIQA changes little (at most 3.3 points).")
H2("5.2 Memory and speed")
rows=[["Version","TTFT, median (ms)","Decode (tokens/s)","Decode vs FP16","Peak memory (MB)","Peak memory vs FP16"]]
for m in MODELS:
    s=SPEED[m]; r=num(s[1])/num(SPEED["FP16"][1]); pm=num(s[2])/num(SPEED["FP16"][2])-1
    rows.append([m,s[0],s[1],f"{r:.2f}×","{:,}".format(int(num(s[2]))),"—" if m=="FP16" else f"{pm*100:+.0f}%".replace("-","−")])
table(rows,[1.1,1.05,1.0,0.95,1.1,1.1],cap="Measured efficiency on the Intel Core Ultra 7 155H laptop (Arc GPU, OpenVINO GenAI, batch size 1; 128-token prompt, 128 new tokens, greedy; median of 30 runs after 5 warm-ups). Peak memory is the process working set, which includes the GPU’s shared memory. Ratio columns are computed from the measured values.",fs=8.5)
para("Lower precision made generation both faster and lighter (Table 7). RTN INT8 raised decode throughput from 25.0 to 42.3 tokens/s (1.7×), cut TTFT from 41.2 to 24.7 ms and reduced peak memory by 34%. RTN and AWQ INT4 reached about 65 tokens/s (2.6×) with TTFT of 16.5–16.9 ms, and all INT4 versions reduced peak memory by 54–56%. GPTQ INT4 decoded more slowly (59.9 tokens/s, 2.4×) despite the same format and size; we did not investigate the cause. The realized gains are smaller than the bit-width ratios: size on disk fell to 50% of FP16 at INT8 and 31% at INT4 (ideal: 50% and about 26% including group scales), while peak memory fell only to 66% and 44–46%, as the process also holds components that do not shrink with weight precision, such as the KV cache, activations and runtime buffers.")
H2("5.3 Accuracy–efficiency trade-off")
figure(G+"fig_tradeoff.png",6.3,"Four-task average accuracy versus measured peak memory (left) and decode throughput (right). Data from Tables 5 and 7. The approximate SE of a four-task average is about ±0.006 (derived from the per-task SEs, assuming independence).")
para("Figure 3 summarizes the trade-off. INT8 moves the model to substantially lower memory and higher throughput at no detectable accuracy cost. INT4 roughly halves memory again and adds further throughput, at an accuracy cost that depends on the method; AWQ lies closest to the FP16 accuracy level among the INT4 versions.")
H2("5.4 Robustness finding: GPTQ perplexity without BOS")
para("Under the standard lm-evaluation-harness WikiText protocol, in which only the first window of each document begins with BOS, four models behave as expected, but GPTQ INT4 does not (Table 8): its word perplexity is 423.7, against 12.89–15.59 for the others. An identical rebuild of the GPTQ model gave 423.66, so the result is reproducible rather than a one-off. A token-level test on four passages of 2048 tokens confirms that the failure is tied to BOS: perplexity with vs without BOS was 9.58 vs 9.58 for FP16, 11.0 vs 11.1 for AWQ, and about 12 vs 3,044 for GPTQ. Without BOS, GPTQ predicts normally for about the first 32 tokens and then degrades. In free generation it still produces coherent, though more repetitive, text.")
table([
 ["Version","Word ppl, BOS at every window (headline)","Word ppl, standard protocol (BOS at document start only)"],
 ["FP16","15.46","12.89"],["RTN INT8","15.50","12.92"],["RTN INT4","19.98","15.59"],
 ["GPTQ INT4","20.68","423.7 (rebuild: 423.66)"],["AWQ INT4","18.22","14.89"],
 ["GPTQ INT4, layers.1.mlp.down_proj replaced by RTN†","18.67","14.86"],
],[2.4,2.0,2.1],cap="WikiText-2 word perplexity (max_length = 2048) under the two BOS protocols. Values should be compared within a column. † Diagnostic model; the replaced matrix is stored in FP16, so this model is not purely INT4.",fs=8.5)
para("**Localization.** Comparing each GPTQ weight matrix with its FP16 original isolates the defect to a single matrix, layers.1.mlp.down_proj, and within it to input column 2558. Weights in this column are grossly altered (for example, 0.052 became −0.781); the maximum absolute error there is 0.83, against at most 0.12 elsewhere. Replacing only this matrix with its RTN INT4 version restores the standard-protocol perplexity to 14.86 and lowers the headline perplexity from 20.68 to 18.67, below RTN INT4’s 19.98. This suggests that the defect also contributes to GPTQ’s weaker headline results, although we did not re-run the accuracy tasks on the patched model.")
table([
 ["GPTQ INT4 build","Standard-protocol word ppl","Defect","Word ppl after replacing layers.1.mlp.down_proj by RTN"],
 ["Default (WikiText-2 calibration, symmetric)","423.7","Column 2558; max abs error 0.83","14.86"],
 ["C4 calibration","305.7","Same column","not measured"],
 ["Asymmetric quantization","229.6","Same column","not measured"],
 ["Scale estimation enabled","202.6","Error spread along column 2558 (column error norm 26× RTN); missed by max-error screening","14.78"],
],[1.75,1.1,2.2,1.45],cap="Ablations of the GPTQ INT4 build (standard WikiText-2 protocol; lower is better).",fs=8.5)
para("**Ablations.** The defect survives changes to the calibration data, the quantization grid and NNCF’s scale estimation (Table 9). With scale estimation, the error is no longer concentrated in a few weights but spread along column 2558, whose error norm is 26 times that of RTN; screening matrices by maximum absolute error therefore misses it, while replacing the same matrix again restores perplexity (14.78). Replacing layer 0’s down_proj instead, leaving the defective matrix in place, also removes the failure (15.14 for the default build in a fast check; 14.91 in a full evaluation of the scale-estimation build), suggesting that layer 0 feeds a large signal into the defective column.")
para("**A partial explanation.** In the FP16 model, the activation entering column 2558 has the same mean with and without BOS, but without BOS it spikes (maximum 0.78 vs 0.13 with BOS), mostly at early positions that behave like attention sinks. A plausible account is that GPTQ, calibrated on BOS-prefixed text in which this channel is small and steady, assigned large compensating errors to a column that appeared unimportant, which are then amplified when the channel spikes. This link is suggestive, not verified: we found no global correlation between per-column quantization error and the without/with-BOS activation ratio, and we did not measure the channel’s statistics on the calibration data itself.")
para("**Scope of the finding.** The finding holds for Llama-3.2-1B compressed with NNCF 3.4.0; we have not shown that it generalizes to GPTQ as an algorithm, nor that it constitutes a bug in NNCF. Patched models store the replaced matrix in FP16 and are not pure INT4 models. All accuracy evaluations in Sections 5.1–5.3 supplied BOS to every model, so the main comparison is unaffected; we therefore treat with-BOS perplexity as the headline metric and this as a secondary robustness result.")
H2("5.5 LoRA recovery (Phase 2)")
para("Table 10 gives Phase 2 accuracy on Colab. All numbers in this section are percentages and come from the same Colab evaluation, so they can be compared with each other but not with the Phase 1 tables. As a check on the fake-quantized bases, the Colab no-LoRA scores agree with the laptop scores (Tables 5 and 6) within 0.6 points on every task for all three bases.")
fmt=lambda v: f"{v:.2f}"
rows=[["Base","LoRA"]+P2_TASKS+["Avg. (6 tasks)"]]
for b in P2_BASES:
    rows.append([b,"no"]+[fmt(v) for v in P2_NO[b]])
    rows.append([b,"yes"]+[f"{m:.2f} ± {s:.2f}" for m,s in P2_LORA[b]])
    rows.append(["","*Δ*"]+["*"+d+"*" for d in P2_DELTA[b]])
table(rows,[0.8,0.45,0.72,0.72,0.72,0.72,0.72,0.72,0.82],cap="Phase 2 zero-shot accuracy (%) on Colab (lm-evaluation-harness, Hugging Face backend). “yes” rows are mean ± standard deviation over seeds 0, 1 and 2; “no” rows are a single evaluation (log-likelihood scoring is deterministic). Δ = LoRA minus no LoRA, computed before rounding. ARC-C, ARC-E, HellaSwag and PIQA: acc_norm; WinoGrande and BoolQ: acc.",fs=8)
para("**INT8.** Without LoRA, RTN INT8 is 0.06 points below FP16 on the six-task average (60.35 vs 60.41), which is not significantly different from zero (paired 95% CI −0.29 to +0.17, p = 0.60; Table 14). The interval rules out an average INT8 loss larger than about 0.3 points on these tasks. With LoRA the two are identical (61.46; paired 95% CI −0.14 to +0.13). Because no gap is detectable, we do not report a recovery percentage for INT8; LoRA simply raises INT8 by the same amount as FP16 (+1.10 vs +1.05), and both gains are significant in the paired test (Holm p < 0.01; McNemar p < 10⁻⁸ in every seed).")
para("**INT4.** RTN INT4 loses 4.04 points without LoRA (56.36 vs 60.41). With LoRA it reaches 60.36 ± 0.07, about the level of plain FP16. That comparison flatters INT4, however, because the same recipe also improves FP16 by 1.05 points, so part of the INT4 gain is the general benefit of Dolly fine-tuning. The fair comparison is with FP16 + LoRA (Table 11): the INT4 gap shrinks from 4.04 to 1.10 points, so LoRA closes about 73% of it. The remaining gap is about six times the largest seed standard deviation, and INT4 + LoRA is still below FP16 + LoRA on all six tasks (by 0.4 to 2.0 points). In the paired tests (Table 14), the loss, the INT4 LoRA gain and the remaining gap are all significant after Holm correction: −4.03 points (95% CI −4.73 to −3.33), +3.97 (+3.37 to +4.56) and −1.11 (−1.63 to −0.59). The remaining gap is also significant by McNemar in each seed separately (p < 10⁻¹¹), and its direction is the same in all three seeds (−1.29, −0.82, −1.21).")
table([
 ["Base","Gap to FP16, no LoRA (points)","Gap to FP16 + LoRA, with LoRA (points)","Share of gap closed"],
 ["RTN INT8","0.06","0.00","not meaningful (no detectable gap)"],
 ["RTN INT4","4.04","1.10","≈ 73%"],
],[1.1,1.7,1.9,1.8],cap="Six-task average gap to FP16, comparing like with like (no LoRA vs no LoRA; LoRA vs LoRA). Gaps computed before rounding.",fs=8.5)
figure(G+"fig_phase2.png",5.6,"Six-task average accuracy without and with LoRA for each base (Colab evaluation). Error bars: standard deviation over three seeds. Dashed line: FP16 without LoRA. The y-axis starts at 54%.")
para("**Per task.** The largest INT4 gain is on BoolQ (+9.62 points, from 55.60 to 65.22). Without LoRA, INT4 was below BoolQ’s majority-class rate of about 62%, so part of this gain may come from LoRA restoring the model’s yes/no answer calibration or format rather than restoring knowledge. This is a hypothesis we have not tested; one check would be to compare the share of “yes” answers before and after LoRA. The other five tasks gain between +1.2 (WinoGrande) and +5.0 (ARC-Easy) points, so the recovery is not limited to BoolQ. The BoolQ gain itself is far outside sampling error (paired 95% CI +7.77 to +11.08 points on the re-scored data, where the INT4 base scores 55.81 rather than 55.60), which shows the change is real but not what caused it. The remaining gap between INT4 + LoRA and FP16 + LoRA is clearest on HellaSwag (−2.04 points, 95% CI −2.47 to −1.61; McNemar p < 10⁻¹¹ in every seed). On BoolQ it is −1.22 points (95% CI −2.31 to −0.12, uncorrected bootstrap p = 0.03), but McNemar does not confirm it in every seed (largest p = 0.10). On the other four tasks it is negative but not individually significant (Appendix Table 17).")
table([
 ["Base","Train time per run (s)","Peak GPU memory (GB)","Mean train loss","Held-out Dolly loss"],
]+[[b,P2_TRAIN[b][0],P2_TRAIN[b][1],f"{sum(P2_TRAIN[b][2])/3:.2f}",f"{sum(P2_TRAIN[b][3])/3:.3f} ({min(P2_TRAIN[b][3]):.3f}–{max(P2_TRAIN[b][3]):.3f})"] for b in P2_BASES],
 [1.1,1.3,1.3,1.2,1.6],cap="Phase 2 training (three seeds per base, Colab T4). Held-out loss is on the 200 held-out Dolly examples, mean (range) over seeds. Peak GPU memory is the same for every base because the fake-quantized weights are stored in FP16; it is not deployment memory (see Table 7 for Phase 1 laptop memory and Table 13 for memory with LoRA).",fs=8.5)
para("**Training loss and cost.** The held-out loss tells the same story as accuracy (Table 12). FP16 and INT8 reach the same held-out loss (1.695 and 1.696), while INT4 stays higher (1.768). LoRA therefore does not fully undo INT4 rounding even on text like its own training data. Each run took about 11 minutes and 5.84 GB of GPU memory for every base. The equal cost is expected in our setup, which stores every base in FP16; with true 4-bit storage, as in QLoRA {qlora}, the INT4 base would need less memory.")
para("**Seed variation.** Across seeds, the six-task average varies by at most 0.25 points (standard deviation), and per-task standard deviations are at most 0.53 points. The differences discussed above are larger than this training noise. The seed spread does not include the sampling error of the test questions; that is covered by the paired tests in Section 5.7. For every comparison that is significant there, all three seeds show a difference in the same direction.")

H2("5.6 LoRA speed and memory on the laptop")
para("Table 13 puts the six cells in one place: Colab six-task accuracy (Table 10) and laptop speed and memory with the seed-0 adapter left unmerged. The timing protocol matched Phase 1 (Table 2), except that memory here is peak commit (private bytes) rather than peak working set, so the no-LoRA memory figures are not directly comparable with Table 7. The no-LoRA throughputs, however, agree with the Phase 1 values (25.0, 42.3 and 65.7 tokens/s) to within about 3–5%. Each cell is the mean of three interleaved rounds, except FP16 without LoRA, where one round was dropped because system CPU load during it was 59% (versus about 8–10% otherwise).")
rows=[["Base","LoRA","Avg. acc. (%)","Decode (tok/s)","vs no LoRA","TTFT (ms)","Peak mem. (MB)"]]
for b in P2_BASES:
    off,on=P2_SPEED[b]["off"],P2_SPEED[b]["on"]
    rows.append([b,"no",f"{P2_NO[b][6]:.2f}",f"{off[0]:.1f}","—",f"{off[1]:.1f}","{:,}".format(off[2])])
    rows.append([b,"yes",f"{P2_LORA[b][6][0]:.2f}",f"{on[0]:.1f}",P2_SPEED_DELTA[b],f"{on[1]:.1f}","{:,}".format(on[2])])
table(rows,[0.95,0.5,1.0,1.05,0.85,0.8,1.05],cap="Six-cell grid: Colab six-task average accuracy (Table 10) and laptop efficiency with the seed-0 adapter applied unmerged (OpenVINO GenAI 2026.4.1, Arc GPU; same prompt and run counts as Table 7). LoRA accuracy is the three-seed mean from Table 10; speed and memory used only the seed-0 adapter. Decode and TTFT are means of the kept rounds (two for FP16 without LoRA; three otherwise). Peak memory is peak commit (private bytes), not the working-set metric of Table 7.",fs=8.5)
para("**Speed.** Unmerged LoRA slowed decoding by 8% for FP16 (24.9 to 22.9 tokens/s), 15% for INT8 (41.2 to 35.0) and 18% for INT4 (62.8 to 51.4). Time to first token rose by 2.5, 4.0 and 4.7 ms respectively. The relative cost therefore grew as the base became cheaper. A likely reason, which we did not measure directly, is that the adapter matrix multiplies run at higher precision and add a roughly fixed cost, which is a larger share of a faster model. Round-to-round spread was small for FP16 (kept no-LoRA rounds 24.98 and 24.81 tokens/s; LoRA 22.86–23.06). It was larger for INT8 with LoRA (32.73, 36.11 and 36.03) and for INT4 (no LoRA 60.75–63.94; with LoRA 49.34–52.89). One INT8 + LoRA round (32.73) was well below the other two (about 36); we did not drop it. In every kept round the six cells kept the same speed order: INT4, INT4 + LoRA, INT8, INT8 + LoRA, FP16, FP16 + LoRA. Within a round, the standard deviation over the 30 timed runs was 0.1–0.3 tokens/s for FP16, 0.4–0.9 for INT8 and 0.9–2.0 for INT4, smaller than the round-to-round differences. With only two or three rounds per cell we do not run a formal test on throughput. Descriptively, every LoRA round was slower than every no-LoRA round of the same base (highest LoRA round vs lowest no-LoRA round: 23.06 vs 24.81, 36.11 vs 40.82 and 52.89 vs 60.75 tokens/s).")
para("**Memory.** The adapter added little memory: 34 MB for FP16, 22 MB for INT8 and 11 MB for INT4 (peak commit). Peak working set showed the same pattern, rising by about 17, 23 and 19 MB (about 5,578 to 5,595; 3,590 to 3,613; 2,433 to 2,452). Phase 1 reported working set without LoRA of 5,579, 3,693 and 2,545 MB, so the absolute working-set levels in this pass differ somewhat for INT8 and INT4; we therefore treat Table 7 and Table 13 as separate measurements and do not mix the two memory metrics.")
para("**The trade-off.** Figure 5 plots accuracy against throughput for the six cells. INT4 + LoRA is still about 2.2× as fast as FP16 + LoRA (51.4 vs 22.9 tokens/s) and about 2.1× as fast as plain FP16, at about half the peak memory (3,108 vs 6,253–6,287 MB). Its six-task average (60.36%) is 1.10 points below FP16 + LoRA and about equal to plain FP16 (60.41%). The unmerged adapter therefore recovers most of the INT4 accuracy at a noticeable speed cost but almost no extra memory. Merging the adapter into the base and re-quantizing could in principle remove that speed cost, but it would change the INT4 weights and would require a new accuracy check; we did not test this (see Section 8).")
figure(G+"fig_lora_tradeoff.png",5.8,"Six-task average accuracy (Colab) versus decode throughput on the laptop for the six cells. Hollow markers: no LoRA. Filled markers: unmerged seed-0 LoRA. Lines connect the same base. Throughput is the mean of the kept rounds (Table 13).")


H2("5.7 Statistical tests (paired)")
para('Table 14 gives the paired tests for Phase 2 described in Section 4.5 (full results in Appendix B). They replace an earlier interim test that used only the task-level SEs saved by lm-evaluation-harness and ignored pairing. Pairing matters here because two models scored on the same questions mostly agree. For example, FP16 with and without LoRA are both right or both wrong on at least 95% of the 19,965 questions, so the variance of their difference is much smaller than the sum of their separate variances. As a result, three effects of about one point that the interim test could not resolve (Holm p = 0.37 for each) are now significant: the LoRA gain on FP16 (+1.05 points, 95% CI +0.64 to +1.46), the LoRA gain on INT8 (+1.11, +0.70 to +1.52) and the remaining gap between INT4 + LoRA and FP16 + LoRA (−1.11, −1.63 to −0.59). The two INT8-vs-FP16 comparisons remain non-significant, but their intervals are now narrow (half-widths 0.23 and 0.13 points), so they bound any INT8 effect rather than leaving it open. The large effects (INT4 loss and INT4 LoRA gain) were already significant and remain so. Plain FP16 and INT4 + LoRA are not distinguishable (−0.05, 95% CI −0.66 to +0.53). Two limits apply. First, these tests treat the test questions as the random sample and condition on our three trained adapters; the per-seed differences show that all three seeds agree in direction for every significant comparison, but three seeds cannot estimate training variance precisely. Second, 10,000 resamples cannot resolve bootstrap p-values below about 0.0002 (shown as < 0.001), so the Holm-adjusted values for these comparisons are upper bounds; the McNemar p-values are exact. Phase 1 per-question outputs were not saved, so Tables 5 and 6 still have no paired test.')
table(P2_STATS_ROWS,[1.6,0.5,1.0,0.6,0.5,0.75,1.15],cap="Paired significance tests for Phase 2 (Colab, six-task average accuracy, percentage points; first model minus second), on the re-scored per-question data. LoRA cells use per-question correctness averaged over seeds 0–2. 95% CI and p from a stratified paired bootstrap (10,000 resamples; smallest resolvable p ≈ 0.0002). Holm correction over the seven main comparisons; values marked < 0.01 are bounds limited by bootstrap resolution. McNemar: exact test pooled over all six tasks; for LoRA cells, the largest p of the three seeds. † Secondary comparison, not in the Holm family. Per-seed diffs: the same difference for each seed separately (the no-LoRA side is a single evaluation). Differences can differ from Table 10 by up to 0.02 points because of the re-score (Section 4.3).",fs=7.5)

# ---------- 6 Discussion ----------
H1("6 Discussion")
para("**RQ1 (accuracy).** At INT8, RTN is effectively lossless on this model, consistent with prior reports for larger models {huang, evalq}. At INT4, all three methods cost accuracy, and AWQ preserved it best: it was ahead of RTN on all six tasks and on perplexity, recovering roughly half of RTN’s average loss. This agrees with Zhou et al.’s recommendation of AWQ for small language models {zhou}. GPTQ did not improve on RTN in our setting. We do not read this as a general verdict on GPTQ: the result concerns NNCF 3.4.0 with default settings on one model, and Section 5.4 identifies a localized defect whose removal improves GPTQ’s headline perplexity beyond RTN’s. The four-task average also understates INT4 damage on some tasks; on BoolQ, RTN and GPTQ INT4 fell below the majority-class rate.")
para("**RQ2 (efficiency).** On the Arc integrated GPU, lower precision translated into real speed and memory gains, but smaller than bit width alone would suggest: 1.7× and about 2.6× decode throughput for 2× and roughly 3.2× smaller files, and 34% and 54–56% lower peak memory. This pattern is consistent with decoding being largely memory-bound, with fixed costs (KV cache, activations, runtime buffers and GPU allocations counted in the shared working set) diluting the savings. Contrary to our expectation (H6), TTFT improved by a similar factor to decode throughput; with a 128-token prompt and a 1B model, prefill may also be limited by weight reads on this device, though we did not test this. GPTQ’s lower throughput at identical size is unexplained.")
para("**RQ3 (LoRA).** For RTN bases, the answer so far is that LoRA helps most where quantization hurt most. INT8 had no detectable loss to recover and gained the same as FP16. INT4 gained about four times as much, closing about 73% of its gap to FP16 + LoRA. This agrees with QLoRA {qlora}, which already showed that LoRA on a 4-bit base recovers much of the lost accuracy; our addition is a controlled measurement on a 1B model with bases that bit-match a laptop deployment. It contrasts with Huang et al. {huang}, who found that general instruction data did not restore zero-shot accuracy for LLaMA3-8B; differences in model size, quantizer, recipe and tasks could explain this, and we did not test which. Two cautions apply. First, comparing INT4 + LoRA with plain FP16 makes recovery look complete, which it is not. Second, the large BoolQ gain may partly reflect answer calibration rather than knowledge. Whether GPTQ and AWQ bases recover differently, the main question behind RQ3, awaits their LoRA runs.")
para("**LoRA on the laptop.** The unmerged adapter costs almost no memory (11–34 MB) but some speed, and the relative cost is larger for the cheaper bases (8%, 15% and 18% for FP16, INT8 and INT4). We take the most likely cause to be that the adapter’s extra matrix multiplies stay at higher precision and are a roughly fixed cost—an explanation, not a measurement. The practical result is that INT4 + LoRA remains more than twice as fast as FP16 + LoRA at about half the memory, while remaining 1.10 points below it in accuracy.")
para("**Evaluation practice.** The GPTQ result shows that a compressed model can match its peers on standard BOS-prefixed benchmarks yet fail when inputs lack BOS, a condition that can arise when long inputs are processed in windows. Reporting the BOS protocol used for perplexity, and spot-checking compressed models without BOS, is a low-cost safeguard.")
para("**Practical guidance.** For this model and laptop, INT8 is a low-risk choice that yields 1.7× decode throughput and a third less peak memory at no detectable accuracy cost. When memory or speed requires INT4, AWQ gave the best accuracy of the methods tested at no cost in size or speed relative to RTN. If fine-tuning is planned anyway, a short LoRA run on an RTN INT4 base brings it close to, but still 1.10 points below, FP16 with the same adapter. On this laptop the unmerged adapter added only 11 MB of peak memory to INT4 and left it about 2.2× as fast as FP16 + LoRA (51.4 vs 22.9 tokens/s), though 18% slower than INT4 without LoRA. These recommendations rest on one model, one machine, one prompt length, one adapter rank and unmerged adapters only.")
table([
 ["#","Outcome"],
 ["H1","Supported. All per-task differences are within about one SE; four-task average −0.003."],
 ["H2","Supported for INT4 (average −0.038; INT3 deferred)."],
 ["H3","Partly supported. AWQ lost less than RTN (average −0.021 vs −0.038, consistent across tasks); GPTQ did not (−0.043)."],
 ["H4","Deferred (pruning out of scope)."],
 ["H5","Supported for peak memory and INT4 disk size; INT8 disk size was essentially ideal (50% of FP16)."],
 ["H6","Decode part supported (1.7× and 2.4–2.6× vs 2× and ~3.2× smaller files); TTFT part not supported (TTFT fell 40–60%)."],
 ["H7a","Supported for RTN bases: INT4 gained +3.99 vs +1.05 (FP16) and +1.10 (INT8), and its gap to FP16 + LoRA fell from 4.04 to 1.10 points. All three gains and the remaining 1.10-point gap are significant in the paired test (Holm p < 0.01; Table 14). GPTQ and AWQ pending."],
 ["H7b","Not supported for RTN bases: LoRA on Dolly raised zero-shot accuracy on all six tasks for all three bases."],
 ["H8","Not tested: adapters were run unmerged on the laptop; merging and re-compressing was not."],
],[1.0,5.5],cap="Status of the pre-specified hypotheses after Phase 1, the RTN part of Phase 2, and the laptop LoRA speed measurements.")

# ---------- 7 Limitations ----------
H1("7 Limitations")
para("**Single model and single machine.** All conclusions concern one ~1B model on one laptop with specific drivers and software versions. Larger models may tolerate compression differently, and other hardware or runtimes may realize different efficiency gains.")
para("**Integrated GPU and runtime support.** The Arc GPU shares memory and bandwidth with the CPU, and peak working set includes GPU allocations in shared memory, so memory figures are not directly comparable with dedicated-GPU measurements. Measured speed depends on how well OpenVINO kernels support each format.")
para("**Restricted compression space.** Only weight-only quantization with one group size and one calibration set (outside the ablations) is studied; INT3, pruning and mixed precision are not covered. Each configuration was compressed once (the GPTQ model was rebuilt once and reproduced), so variability due to calibration sampling is not measured. GPTQ and AWQ results reflect the NNCF 3.4.0 implementations and may differ in other toolkits.")
para("**Statistics.** Phase 2 differences are tested with paired tests over questions (Table 14, Appendix B). These tests take the test questions as the random sample and condition on three trained adapters per base, so they do not fully capture training variance; a different recipe, dataset or more seeds could shift the ~1-point effects. Phase 1 per-question outputs were not saved, so Phase 1 conclusions rely on per-task SEs and consistency across tasks. The Holm correction covers only the seven main Phase 2 comparisons; per-task and Phase 1 comparisons are uncorrected, so some apparent differences there may arise by chance. Re-scoring the Phase 2 models showed that results depend slightly on the automatically chosen evaluation batch size (up to 7 questions per task; Section 4.3). A later re-run of the INT4 no-LoRA base confirmed the cause: batch size 11 reproduced the original scores exactly, and batch size 13 reproduced the re-score question for question (Appendix Table 19). Future evaluations should fix the batch size. Separately, an accidental retrain of the FP16 LoRA adapter at seed 0 produced slightly different scores (for example, ARC-E 0.6322 rather than 0.6313), so LoRA training is not bit-for-bit reproducible across sessions, while no-LoRA evaluations at a fixed batch size are.")
para("**Benchmark scope.** Six zero-shot tasks and WikiText-2 perplexity do not capture open-ended generation quality, multi-step reasoning, long-context behavior or multilingual ability. MMLU was excluded, and scores near chance or majority-class rates (e.g., BoolQ) are difficult to interpret.")
para("**GPTQ robustness finding.** The diagnosis is specific to Llama-3.2-1B and NNCF 3.4.0, the proposed mechanism is not verified, and the patched diagnostic models are not pure INT4.")
para("**LoRA design (Phase 2).** We tested one adapter configuration (r = 16, attention projections only), one general-purpose dataset (5,000 Dolly examples, one epoch) and three seeds; other ranks, target layers, data or training lengths could change how much is recovered, and the laptop speed test used only the seed-0 adapter and this single rank. Training used fake quantization in PyTorch, not true integer kernels: the weights match the laptop models exactly, but the arithmetic during training and evaluation is FP16, so Phase 2 accuracy is not a measurement of the laptop runtime (the no-LoRA scores agree within 0.6 points per task). Speed and memory were measured with the adapter left unmerged, on the integrated Arc GPU only, and with a single 128-token prompt; a longer or shorter prompt, a discrete GPU, or a different runtime could change the speed cost. Merging the adapter and re-compressing was not tested, so any accuracy loss or speed gain from that route is unknown. LoRA has not yet been run on the GPTQ and AWQ bases, so whether recovery depends on the compression method is still open.")

# ---------- 8 Conclusion ----------
H1("8 Conclusion and Future Work")
para("We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 on an Intel Core Ultra 7 laptop with an integrated Arc GPU. RTN INT8 was nearly lossless (four-task average 0.559 vs 0.562) with 1.7× decode throughput and 34% lower peak memory. INT4 delivered about 2.6× throughput and 54–56% lower peak memory at a 2–4 point accuracy cost; AWQ recovered about half of RTN’s loss, while NNCF GPTQ did not improve on RTN. GPTQ INT4 also failed on inputs without BOS (perplexity 424), a reproducible defect localized to one column of one weight matrix. In Phase 2, a fixed LoRA recipe (r = 16, α = 32, q/k/v/o, 5,000 Dolly examples, three seeds) trained on bases that bit-match the laptop models confirmed, at 1B scale, what QLoRA {qlora} showed for 4-bit models: INT8 showed no detectable loss to recover, and RTN INT4 + LoRA (60.36%) reached about the six-task average of plain FP16 (60.41%). Against FP16 + LoRA, the fair control, INT4 closed about 73% of its 4.04-point gap and remained 1.10 points behind (paired 95% CI 0.59 to 1.63 points; significant after Holm correction), with a higher held-out loss. The large BoolQ gain may partly reflect answer calibration rather than knowledge. On the laptop, the unmerged adapter added only 11–34 MB of peak memory but reduced decode throughput by 8% (FP16), 15% (INT8) and 18% (INT4). INT4 + LoRA remained about 2.2× as fast as FP16 + LoRA at about half the memory.")
para("Next, LoRA on the GPTQ and AWQ bases will test whether recovery depends on the compression method. Merging the adapters and re-quantizing, which might remove the unmerged speed cost but would change the INT4 weights, is untested. Future work after that: INT3 quantization, which NNCF supports from version 3.3.0 {nncf}; pruning, including SparseGPT and Wanda {sparsegpt, wanda}; mixed-precision formats; a test of whether the GPTQ defect occurs in other models or GPTQ implementations; additional models and sizes; task-aligned LoRA data; quantization-aware adapters (QA-LoRA, LoftQ); and other consumer hardware such as NPUs.")

# ---------- References ----------
H1("References")
for i,k in enumerate(order,1):
    p=doc.add_paragraph(); r=p.add_run(f"[{i}] "+REFS[k]); r.font.size=Pt(9.5)
    p.paragraph_format.left_indent=Inches(0.3); p.paragraph_format.first_line_indent=Inches(-0.3); p.paragraph_format.space_after=Pt(3)
unused=[k for k in REFS if k not in order]
assert not unused, unused

# ---------- Appendix ----------
H1("Appendix")
for t in ["A. Prompt template for Dolly training and example formatted inputs [[TBD]]",
          "B. Paired significance tests for Phase 2 (below). Phase 1 per-task accuracies with SEs [[TBD]]",
          "C. LoRA training curves for the 9 completed runs (GPTQ and AWQ runs pending) [[TBD]]",
          "D. Phase 2 base checks: fake-quantization code/scale comparison and Colab vs laptop scores per task [[TBD]]",
          "E. NNCF compression logs, including backup-precision layers [[TBD]]",
          "F. GPTQ diagnostics: per-matrix and per-column error statistics and column-2558 activation traces [[TBD]]",
          "G. Laptop LoRA timing logs (balanced interleaved pass) [[TBD]]"]:
    para(t)

H2("Appendix B. Paired significance tests (Phase 2)")
para("**Data.** All 12 Phase 2 models (three bases, each without LoRA and with the seed-0, 1 and 2 adapters) were re-scored on Colab with lm-evaluation-harness 0.4.13 and log_samples enabled, which records for every question whether the model was correct (acc_norm for ARC-Challenge, ARC-Easy, HellaSwag and PIQA; acc for WinoGrande and BoolQ). The full test sets were used: 1,172, 2,376, 10,042, 1,267, 3,270 and 1,838 questions (19,965 in total). Question IDs were checked to be identical and in the same order for all 12 models.")
para("**Paired bootstrap.** For two cells A and B, let d be the per-question difference in correctness (for a LoRA cell, correctness is the mean over the three seeds, so d can take fractional values). The statistic is the six-task average of the per-task mean of d, in percentage points. Each of 10,000 resamples draws questions with replacement within each task (stratified), using the same draws for both cells and for every comparison (random seed 20261119). The 95% CI is the 2.5th–97.5th percentile of the resampled statistic. The two-sided p-value is 2 × min(P(d* ≤ 0), P(d* ≥ 0)) with add-one smoothing, so the smallest value is 2/10,001 ≈ 0.0002. The Holm step-down correction is applied across the seven main comparisons; the comparison of INT4 + LoRA with plain FP16 is secondary and not corrected.")
para("**McNemar.** For each comparison and task, and pooled over all six tasks, we count questions answered correctly by only the first model and by only the second, and run an exact two-sided binomial test on these discordant counts. Because McNemar needs a single model on each side, LoRA cells are tested once per seed (seed k with LoRA vs the same base without LoRA, or seed k vs seed k across bases), and we report the largest of the three p-values. The pooled test weights questions equally, so HellaSwag dominates it, while the bootstrap weights the six tasks equally; the two therefore answer slightly different questions and are reported side by side. Holm is applied to the pooled McNemar p-values as well.")
para("**Re-score check.** Table 18 compares each re-scored task accuracy with the first evaluation, as a net count of questions. The two non-zero rows are the quantized bases without LoRA, whose automatically chosen batch size changed from 11 to 13. A later check on the INT4 no-LoRA base confirmed the cause: at batch size 11 the original scores were reproduced exactly, and at batch size 13 the re-score was reproduced question for question (Table 19). Code: lora/stats/paired_bootstrap.py; results: lora/stats/paired_results.json; mismatch evidence: lora/stats/mismatch/.")
table(P2_APP_MAIN,[1.45,0.45,0.95,0.5,0.55,1.45,0.6,0.6],cap="Full results for the eight Phase 2 comparisons (six-task average, percentage points; first model minus second). McNemar columns use all 19,965 questions pooled; for LoRA cells, the discordant counts are listed for seeds 0, 1 and 2, and the p-value is the largest of the three. † Secondary comparison, outside the Holm family.",fs=7)
table(P2_APP_TASK,[1.6,0.8,0.55,1.15,0.65,1.0],cap="Per-task paired results (percentage points; first model minus second). Bootstrap CIs and p-values are per task and not corrected for multiple comparisons. McNemar p is the largest over seeds for LoRA cells. † Secondary comparison.",fs=7)
table(P2_APP_RESCORE,[1.15,0.95,0.5,0.5,0.65,0.7,0.5,0.5,0.8],cap="Re-score check: net change in the number of correctly answered questions (re-score minus first evaluation) per task, and change in the six-task average. Batch size: automatically chosen evaluation batch size in the first evaluation and in the re-score, from the saved Colab logs (n/l: first-run log not saved; these seven models were first evaluated in a later session that ran no training). All 12 first and re-score runs used the same library versions and a T4 GPU. A confirmatory re-run of the INT4 no-LoRA base at both batch sizes is in Table 19.",fs=7)
table(P2_APP_BSCHECK,[0.9,1.15,1.05,1.05,1.15,1.2],cap="Confirmation of the evaluation-batch-size effect on the INT4 no-LoRA model. Both checks were run in the same Colab T4 session, with the same library versions as the first evaluation and the re-score (PyTorch 2.14.1+cu130, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13) and 67,106 log-likelihood requests per run. At batch size 11, every task score matched the original saved accuracy exactly. At batch size 13, per-question correctness was identical to the re-score file on all six tasks. PIQA had the same accuracy at both batch sizes, but the per-question flags at batch 11 differed from the re-score, so some flips cancelled.",fs=7)

# footer page numbers
fp=sec.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=fp.add_run(); f1=OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"),"begin"); r._r.append(f1)
r2=fp.add_run(); it=OxmlElement("w:instrText"); it.set(qn("xml:space"),"preserve"); it.text="PAGE"; r2._r.append(it)
r3=fp.add_run(); f2=OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"),"end"); r3._r.append(f2)
doc.core_properties.title="Shrinking a Small Language Model on a Laptop (draft)"; doc.core_properties.author="Dhruv Vinesh"
OUT=str(_HERE/"paper_draft.docx")
doc.save(OUT)
print("refs",len(order),"tables",tcount[0],"figures",fcount[0])

# ---------- PDF ----------
import subprocess
subprocess.run(["soffice","--headless","--convert-to","pdf","--outdir",str(_HERE),OUT],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("pdf written")
