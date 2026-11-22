#!/usr/bin/env python3
"""Fill Appendix A–G with content taken from real result files.

    python3 /workspace/paper/tools/patch_appendix.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_appendix.py. Figures
fig_app_lora_loss.png and fig_app_gptq_maxabs.png should already be
in paper/figs/ (this script regenerates them if the source CSVs/JSONs
are present). Every number in the inserted appendix comes from a named
result file.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
P = PAPER / "build_paper.py"
SECTION = Path("/tmp/appendix_section.py")
# keep a copy next to the patcher so a later rerun does not depend on /tmp
LOCAL_SECTION = HERE / "appendix_section.py"
if SECTION.exists() and not LOCAL_SECTION.exists():
    LOCAL_SECTION.write_text(SECTION.read_text())
elif LOCAL_SECTION.exists() and not SECTION.exists():
    SECTION = LOCAL_SECTION
elif SECTION.exists():
    LOCAL_SECTION.write_text(SECTION.read_text())
    SECTION = LOCAL_SECTION
else:
    SECTION = LOCAL_SECTION

src = P.read_text()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:200])
    src = src.replace(old, new)

# helper: monospaced block
rep(
    'def H2(t): doc.add_heading(t,level=2)',
    'def H2(t): doc.add_heading(t,level=2)\n'
    'def pre(text, fs=8.5):\n'
    '    """Monospaced lines for templates and printed examples."""\n'
    '    for line in text.split("\\n"):\n'
    '        p=doc.add_paragraph()\n'
    '        r=p.add_run(line if line else "\\u00a0")\n'
    '        r.font.name="Courier New"\n'
    '        r._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")\n'
    '        r.font.size=Pt(fs)\n'
    '        p.paragraph_format.space_after=Pt(0)\n'
    '        p.paragraph_format.space_before=Pt(0)\n'
    '        p.paragraph_format.line_spacing=1.05\n'
    '    doc.add_paragraph().paragraph_format.space_after=Pt(2)'
)

OLD_OUTLINE = '''H1("Appendix")
for t in ["A. Prompt template for Dolly training and example formatted inputs [[TBD]]",
          "B. Paired significance tests for Phase 2 (below). Phase 1 per-task accuracies with SEs [[TBD]]",
          "C. LoRA training curves for the 9 completed runs (GPTQ and AWQ runs pending) [[TBD]]",
          "D. Phase 2 base checks: fake-quantization code/scale comparison and Colab vs laptop scores per task [[TBD]]",
          "E. NNCF compression logs, including backup-precision layers [[TBD]]",
          "F. GPTQ diagnostics: per-matrix and per-column error statistics and column-2558 activation traces [[TBD]]",
          "G. Laptop LoRA timing logs (balanced interleaved pass) [[TBD]]"]:
    para(t)

H2("Appendix B. Paired significance tests (Phase 2)")
'''

NEW_A = r'''H1("Appendix")
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

H2("Appendix B. Paired significance tests (Phase 2) and Phase 1 per-task standard errors")
'''

rep(OLD_OUTLINE, NEW_A)

TAIL = '''table(P2_APP_BSCHECK,[0.9,1.15,1.05,1.05,1.15,1.2],cap="Confirmation of the evaluation-batch-size effect on the INT4 no-LoRA model. Both checks were run in the same Colab T4 session, with the same library versions as the first evaluation and the re-score (PyTorch 2.14.1+cu130, transformers 4.57.6, PEFT 0.21.2, lm-evaluation-harness 0.4.13) and 67,106 log-likelihood requests per run. At batch size 11, every task score matched the original saved accuracy exactly. At batch size 13, per-question correctness was identical to the re-score file on all six tasks. PIQA had the same accuracy at both batch sizes, but the per-question flags at batch 11 differed from the re-score, so some flips cancelled.",fs=7)
'''

section = SECTION.read_text()
rep(TAIL, TAIL + "\n" + section + "\n")

P.write_text(src)
print("patched", P, "bytes", len(src))
