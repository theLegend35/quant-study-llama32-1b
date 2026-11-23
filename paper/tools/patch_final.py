#!/usr/bin/env python3
"""Restructure the paper into a short main body plus appendices.

    python3 /workspace/paper/tools/patch_final.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_final.py. Does not change measured
results; it moves detail out of the main text and updates cross-references.
"""
from pathlib import Path
import re

PAPER = Path("/workspace/paper")
SRC0_PATH = PAPER / "archive" / "build_paper_pre_final.py"
OUT = PAPER / "build_paper.py"
BODY = PAPER / "tools" / "final_body_insert.py"

src0 = SRC0_PATH.read_text()
body = BODY.read_text()
# Drop the header comments in the insert file
body = body[body.find("# ---------- Title ----------"):]
assert body.startswith("# ---------- Title ----------")

def grab(start, end, src=src0):
    a = src.find(start)
    b = src.find(end, a)
    assert a >= 0 and b >= 0, (start[:90], a, b)
    return src[a:b]

def grab_inclusive_line(start, end_after, src=src0):
    """Return start..end_after inclusive of end_after's line."""
    a = src.find(start)
    b = src.find(end_after, a)
    assert a >= 0 and b >= 0, (start[:90], a, b)
    b = src.find("\n", b)
    return src[a:b+1] if b >= 0 else src[a:]

# --- pieces to splice into the new body ---
sixcell = grab(
    'rows=[["Base","LoRA","Avg. acc. (%)","Decode (tok/s)","vs no LoRA","TTFT (ms)","Peak mem. (MB)"]]',
    'para("**Speed.** Unmerged LoRA',
)
# update captions/refs inside the six-cell table() call
sixcell = sixcell.replace("Colab six-task average accuracy (Table 10)", "Colab six-task average accuracy (Table 5)")
sixcell = sixcell.replace("same prompt and run counts as Table 7)", "same prompt and run counts as Table 4)")
sixcell = sixcell.replace("the working-set metric of Table 7.", "the working-set metric of Table 4.")
sixcell = sixcell.replace("three-seed mean from Table 10;", "three-seed mean from Table 5;")

lora_fig = grab_inclusive_line(
    'figure(G+"fig_lora_tradeoff.png"',
    'figure(G+"fig_lora_tradeoff.png"',
).rstrip() + "\n"
lora_fig = lora_fig.replace("(Table 13).", "(Table 6).")

p2stats = grab(
    'table(P2_STATS_ROWS,',
    '\n\n# ---------- 6 Discussion ----------',
)
p2stats = p2stats.replace("differ from Table 10 by", "differ from Table 5 by")
p2stats = p2stats.replace("because of the re-score (Section 4.3).", "because of the re-score (Section 3.4).")
if not p2stats.endswith("\n"):
    p2stats += "\n"

assert body.count("# SIXCELL_TABLE") == 1
assert body.count("# LORA_TRADEOFF_FIG") == 1
assert body.count("# P2_STATS_TABLE") == 1
body = body.replace("# SIXCELL_TABLE\n", sixcell)
body = body.replace("# LORA_TRADEOFF_FIG\n", lora_fig)
body = body.replace("# P2_STATS_TABLE\n", p2stats)

# --- preamble and tail ---
preamble = src0[:src0.find("# ---------- Title ----------")]
# data comments
preamble = preamble.replace(
    "Table 5 averages are these rounded once to 3 d.p.",
    "Table 2 averages are these rounded once to 3 d.p.",
)
preamble = preamble.replace("see Table 13 note", "see Table 6 note")

refs = src0[src0.find("# ---------- References ----------"):src0.find("# ---------- PDF ----------")]
pdf_block = src0[src0.find("# ---------- PDF ----------"):]

# --- reference rewrites in existing appendices (A–G) ---
def rewrite_app(t):
    repls = [
        # compound first
        ("Sections 4.3 and 5.5", "Sections 3.4 and 4.4"),
        ("Tables 5 and 6", "Tables 2 and 3"),
        ("Tables 6 and 8", "Table 3 and Table 31"),
        ("as 423.7 in Table 8)", "as 423.7 in Table 31)"),
        ("Appendix Table 18", "Table 10"),
        ("Appendix Table 19", "Table 11"),
        ("a confirmatory re-run of the INT4 no-LoRA base at both batch sizes is in Table 19.",
         "a confirmatory re-run of the INT4 no-LoRA base at both batch sizes is in Table 11."),
        ("question for question (Table 19).", "question for question (Table 11)."),
        ("Table 18 compares", "Table 10 compares"),
        ("In Table 13?", "In Table 6?"),
        ("not used in Table 13.", "not used in Table 6."),
        ("Section 5.6 and the Table 13 note", "Section 4.5 and the Table 6 note"),
        ("The numbers in Table 13 and in this appendix", "The numbers in Table 6 and in this appendix"),
        ("which is what Table 13 uses.", "which is what Table 6 uses."),
        ("The tok/s, TTFT and peak-commit means match Table 13", "The tok/s, TTFT and peak-commit means match Table 6"),
        ("as stated in Section 5.6;", "as stated in Section 4.5;"),
        ("used in Section 5.6 is the mean", "used in Section 4.5 is the mean"),
        ("It is consistent with Section 5.4:", "It is consistent with Section 4.3:"),
        ("as in Section 4.2;", "as in Section 3.2;"),
        ("as stated in Sections 3.4 and 4.4.", "as stated in Sections 3.4 and 4.4."),  # no-op if already done
        ("Every task is within 0.6 points, as stated in Sections 4.3 and 5.5.",
         "Every task is within 0.6 points, as stated in Sections 3.4 and 4.4."),
        ("the four-task average of Table 5.", "the four-task average of Table 2."),
        ("the same runs as Table 5.", "the same runs as Table 2."),
        ("are these values rounded. The four-task average is copied from Table 5;",
         "are these values rounded. The four-task average is copied from Table 2;"),
        ("and matches Table 5.", "and matches Table 2."),
        ("On the four Table 5 tasks alone", "On the four Table 2 tasks alone"),
        ("weakest INT4 method in Tables 2 and 3.", "weakest INT4 method in Tables 2 and 3."),
        ("weakest INT4 method in Tables 5 and 6.", "weakest INT4 method in Tables 2 and 3."),
        ("Appendix F. GPTQ diagnostics", "Appendix F. GPTQ and AWQ diagnostics"),
    ]
    for a, b in repls:
        t = t.replace(a, b)
    return t

refs = rewrite_app(refs)

# leftover Table 5 / 14 / 13 / 10 in appendix region?
# After the specific replacements, fix remaining unique old citations.
# Do not globally replace Table 10 (now used for rescore) etc.

# --- moved content for new appendices ---
background = grab(
    'H2("2.1 Numerical precision and weight-only quantization")',
    '# ---------- 3 Related Work ----------',
)
# retitle the H2s without the old 2.x numbers? Keep them as topical headings without numbers
background = background.replace('H2("2.1 Numerical precision and weight-only quantization")',
                                'H2("Numerical precision and weight-only quantization")')
background = background.replace('H2("2.2 Round-to-nearest (RTN)")', 'H2("Round-to-nearest (RTN)")')
background = background.replace('H2("2.3 GPTQ")', 'H2("GPTQ")')
background = background.replace('H2("2.4 AWQ")', 'H2("AWQ")')
background = background.replace('H2("2.5 LoRA and QLoRA")', 'H2("LoRA and QLoRA")')

# long method / training / re-score / speed protocol
method_figs = (
    grab_inclusive_line('figure(G+"fig_phase1.png"', 'figure(G+"fig_phase1.png"')
    + grab_inclusive_line('figure(G+"fig_phase2_method.png"', 'figure(G+"fig_phase2_method.png"')
)
lora_long = grab(
    'para("**Bases.** LoRA needs a PyTorch model',
    'H2("4.4 Benchmarks and metrics")',
)
# drop the second method figure from this span if present
# The span includes fig_phase2_method at the end; we already add both figures at the start of App I,
# so strip the trailing figure from lora_long.
k = lora_long.find('figure(G+"fig_phase2_method.png"')
if k >= 0:
    lora_long = lora_long[:k]
# update internal refs in the moved method text
lora_long = lora_long.replace("(Section 4.2)", "(Section 3.2)")
lora_long = lora_long.replace("(Section 5.5)", "(Section 4.4)")
lora_long = lora_long.replace("(Section 4.4)", "(Section 3.3)")  # "same metrics as Phase 1 (Section 4.4)"
# careful: we may have already changed 5.5 to 4.4, then that 4.4 got changed to 3.3 wrongly!
# Re-do this more carefully from a fresh grab if needed. Check below.

eff_table = grab(
    'table([\n ["Metric","Definition","Protocol"],',
    'H2("4.5 Fairness controls and statistics")',
)
# may include trailing newlines and a following para? grab ends at H2 so includes text before it.
# There's a short intro para before the table; get just the table.
# Find table start
hyp_table = grab(
    'table([\n ["#","Hypothesis"],',
    'H2("4.6 Reproducibility")',
)
soft_table = grab(
    'table([\n ["Component","Version / setting"],',
    '# ---------- 5 Results ----------',
)

# GPTQ body section (prose + both tables)
gptq_sec = grab(
    'para("Under the standard lm-evaluation-harness WikiText protocol',
    'H2("5.5 LoRA recovery (Phase 2)")',
)
gptq_sec = gptq_sec.replace("(Table 8)", "(Table 31)")
gptq_sec = gptq_sec.replace("Details are in Appendix F.", "The tables of the two-protocol comparison and the ablations are Tables 31 and 32.")
# "see Section 5.4" won't be in this span.

gap_table = grab(
    'table([\n ["Base","Gap to FP16, no LoRA (points)"',
    'figure(G+"fig_phase2.png"',
)

# training table + surrounding training/seed paras
train_block = grab(
    'para("**Per task.** The largest INT4 gain',
    'H2("5.6 LoRA speed and memory on the laptop")',
)
train_block = (
    train_block
    .replace("(Appendix Table 17).", "(Table 9).")
    .replace("(see Table 7 for Phase 1 laptop memory and Table 13 for memory with LoRA).",
             "(see Table 4 for Phase 1 laptop memory and Table 6 for memory with LoRA).")
    .replace("(Table 12).", "(Table 34).")
)

hyp_status = grab(
    'table([\n ["#","Outcome"],',
    '# ---------- 7 Limitations ----------',
)
hyp_status = hyp_status.replace("four Table 5 tasks", "four Table 2 tasks")
hyp_status = hyp_status.replace("Holm p < 0.01; Table 14).", "Holm p < 0.01; Table 7).")

# --- fix lora_long refs carefully from original ---
lora_long_orig = grab(
    'para("**Bases.** LoRA needs a PyTorch model',
    'H2("4.4 Benchmarks and metrics")',
)
k = lora_long_orig.find('figure(G+"fig_phase2_method.png"')
if k >= 0:
    lora_long_orig = lora_long_orig[:k]
lora_long = (
    lora_long_orig
    .replace("with the same scheme as the laptop model (Section 4.2)",
             "with the same scheme as the laptop model (Section 3.2)")
    .replace("within 0.6 points on every task (Section 5.5).",
             "within 0.6 points on every task (Section 4.4).")
    .replace("the same metrics as Phase 1 (Section 4.4).",
             "the same metrics as Phase 1 (Section 3.3).")
    .replace("(Appendix Table 18).", "(Table 10).")
    .replace("(Appendix Table 19).", "(Table 11).")
    .replace("Tables 10–13 keep the first-run scores.",
             "Tables 5 and 6 keep the first-run scores.")
    .replace("The paired tests (Table 14, Appendix B)",
             "The paired tests (Table 7, Appendix B)")
    .replace("can differ from Table 10 by up to",
             "can differ from Table 5 by up to")
    .replace("matched the Phase 1 speed test (Table 2):",
             "matched the Phase 1 speed test (Appendix I):")
)

# efficiency table only (not preceding paras)
# grab already starts at table(
# hyp_table similarly
# soft_table similarly

# If train_block start failed we need a fallback. Detect:

# Prepend GPTQ 5.4 prose+tables note into Appendix F, but the tables themselves
# are emitted in Appendix I so that A–G table numbers stay a uniform shift.
# We will insert the 5.4 *prose* (without the two table() calls) after the F heading.

def strip_table_calls(text):
    """Remove top-level table(...) calls from a source snippet."""
    out = []
    i = 0
    while i < len(text):
        j = text.find("table(", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        # skip balanced parens
        depth = 0
        k = j
        while k < len(text):
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        # drop following newline
        if k < len(text) and text[k] == "\n":
            k += 1
        i = k
    return "".join(out)

gptq_prose = strip_table_calls(gptq_sec)
# Extract the two table calls to place in Appendix I
gptq_tables = []
i = 0
s = gptq_sec
while True:
    j = s.find("table(", i)
    if j < 0:
        break
    depth = 0
    k = j
    while k < len(s):
        if s[k] == "(":
            depth += 1
        elif s[k] == ")":
            depth -= 1
            if depth == 0:
                k += 1
                break
        k += 1
    gptq_tables.append(s[j:k] + "\n")
    i = k
assert len(gptq_tables) == 2, len(gptq_tables)

# Insert gptq prose after Appendix F heading paragraph start
f_head = 'H2("Appendix F. GPTQ and AWQ diagnostics")'
assert f_head in refs
# The original first para of F should follow the moved 5.4 prose
refs = refs.replace(
    f_head,
    f_head + "\n"
    "para(\"The following paragraphs were in the main text and are collected here so that the main text can summarize them. Table and section numbers have been updated.\")\n"
    + gptq_prose,
)

# Appendix I moved tables: need a short metrics intro? The efficiency table
# may have been grabbed with a trailing newline only.

new_apps = f'''
H2("Appendix H. Background on weight-only quantization, RTN, GPTQ, AWQ and LoRA")
para("This appendix records the background that was previously in the main text, so that the body can stay short. Nothing below changes the measurements in the main tables.")
{background}

H2("Appendix I. Experimental details")
para("This appendix holds the experimental detail moved out of Section 3 and the extra result tables that the main text now summarizes: metric definitions, pre-specified hypotheses, software versions, the Phase 2 training protocol and re-score, pipeline figures, the GPTQ two-protocol and ablation tables, the Phase 2 gap table, the training-loss table, and the hypothesis-status table.")
{method_figs}
para("**Efficiency metrics.** The definitions below are those used for all laptop timing runs (Phase 1 and the later unmerged-LoRA pass).")
{eff_table}
para("**Pre-specified hypotheses** (expectations, not findings):")
{hyp_table}
para("**Software versions and fixed settings.** Phase 1 ran on the laptop; Phase 2 training and accuracy evaluation ran on Colab. The Intel Arc GPU driver version was not recorded.")
{soft_table}
para("**Phase 2 protocol (full).** The shorter account in Section 3.4 is the official summary; the paragraphs below are the original method text, with cross-references updated.")
{lora_long}
para("**GPTQ two-protocol perplexity and ablations.** These tables accompany the diagnosis in Appendix F.")
{gptq_tables[0]}{gptq_tables[1]}
para("**Phase 2 gap to FP16**, comparing like with like (no LoRA vs no LoRA; LoRA vs LoRA). Gaps computed before rounding.")
{gap_table}
{train_block}
para("**Status of the pre-specified hypotheses** after Phase 1, the RTN part of Phase 2, and the laptop LoRA speed measurements.")
{hyp_status}
'''

# Assemble
# refs currently ends just before PDF, which starts after Appendix G tables.
# Insert new apps before the end of refs (refs includes through last G table).
# refs starts at "# ---------- References ----------" and ends at last of G.
new_src = preamble + body + "\n" + refs + new_apps + "\n" + pdf_block

# The original file saved the docx at the end of Appendix G, before the PDF
# comment. New appendices H and I were inserted after that save. Move the save
# to just before the soffice conversion.
_save = 'OUT="/workspace/paper/paper_draft.docx"\ndoc.save(OUT)\nprint("refs",len(order),"tables",tcount[0],"figures",fcount[0])\n'
if new_src.count(_save) != 1:
    raise SystemExit(f"save block count {new_src.count(_save)}")
new_src = new_src.replace(_save, "", 1)
_mark = '# ---------- PDF ----------\nimport subprocess\n'
if _mark not in new_src:
    raise SystemExit('pdf mark missing')
new_src = new_src.replace(_mark, '# ---------- PDF ----------\n' + _save + 'import subprocess\n', 1)


# sanity
for s in ["H1(\"Abstract\")", "H1(\"1 Introduction\")", "H1(\"2 Related Work\")",
          "H1(\"3 Experimental Setup\")", "H1(\"4 Results\")",
          "H1(\"5 Discussion and Limitations\")", "H1(\"6 Conclusion\")",
          "H1(\"References\")", "Appendix H.", "Appendix I.",
          "phase1/stats/p1_paired.py", "https://github.com/theLegend35/quant-study-llama32-1b",
          "Dhruv Vinesh, Cypress Ranch High School, Cypress, Texas",
          "P2_STATS_ROWS", "fig_tradeoff.png", "fig_lora_tradeoff.png",
          "[[TBD]]"]:
    if s == "[[TBD]]":
        assert new_src.count(s) == 0, "TBD leaked in"
    else:
        assert s in new_src, f"missing {s}"

# leftover stale claims in the NEW body (not historical patch scripts)
body_only = body
assert "no detectable accuracy cost" not in body_only
assert "cannot be run for Phase 1" not in new_src[new_src.find('H1("Abstract")'):new_src.find('H1("References")')]
assert "still have no paired test" not in new_src


# leftover section numbers that lived only in moved appendix prose
new_src = new_src.replace(
    "All accuracy evaluations in Sections 5.1–5.3 supplied BOS to every model",
    "All accuracy evaluations in Sections 4.1 and 4.2 supplied BOS to every model",
)
new_src = new_src.replace(
    '["Colab evaluation batch size","automatic (11 or 13; see Section 4.3)"]',
    '["Colab evaluation batch size","automatic (11 or 13; see Section 3.4 and Appendix B)"]',
)
new_src = new_src.replace(
    "that is covered by the paired tests in Section 5.7.",
    "that is covered by the paired tests in Section 4.6.",
)

OUT.write_text(new_src)
print("wrote", OUT, "chars", len(new_src), "lines", new_src.count("\n"))

# report remaining old section/table phrases in the assembled file after Title
rest = new_src[new_src.find("# ---------- Title ----------"):]
for s in ["Section 5.4", "Section 5.5", "Section 5.6", "Section 5.7",
          "Section 4.2", "Section 4.3", "Section 4.4", "Section 4.5",
          "Table 8", "Table 13", "Table 14", "Table 16", "Table 18",
          "Appendix Table", "H1(\"2 Background\")", "H1(\"7 Limitations\")",
          "H1(\"8 Conclusion"]:
    c = rest.count(s)
    if c:
        print(f"  leftover {c:2d}  {s}")
