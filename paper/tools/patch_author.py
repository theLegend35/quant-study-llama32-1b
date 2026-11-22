"""Fill the author line and the Section 4.6 repository-link placeholder.

Apply to the current (already versions-patched) build_paper.py:
    python3 /workspace/paper/tools/patch_author.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_author.py. Exact one-occurrence replacements.

Author: Dhruv Vinesh, Cypress Ranch High School, Cypress, Texas.
Repo (expected URL; the user is pushing the study repo now):
    https://github.com/theLegend35/quant-study-llama32-1b
"""
P = "/workspace/paper/build_paper.py"
src = open(P).read()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:160])
    src = src.replace(old, new)

rep(
    'para("[[Author information withheld]] · Draft, November 2026 · *Phase 1 (compression benchmark) complete; Phase 2 (LoRA accuracy and laptop efficiency) complete for FP16, RTN INT8 and RTN INT4; LoRA on GPTQ and AWQ pending.*",align=WD_ALIGN_PARAGRAPH.CENTER)',
    'para("Dhruv Vinesh, Cypress Ranch High School, Cypress, Texas · Draft, November 2026 · *Phase 1 (compression benchmark) complete; Phase 2 (LoRA accuracy and laptop efficiency) complete for FP16, RTN INT8 and RTN INT4; LoRA on GPTQ and AWQ pending.*",align=WD_ALIGN_PARAGRAPH.CENTER)',
)

rep(
    "We will release scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs [[TBD: repository link]].",
    "Scripts, configuration files, the calibration sample indices, the Dolly subset indices, per-question evaluation logs, training logs and raw timing logs are at https://github.com/theLegend35/quant-study-llama32-1b.",
)

rep(
    'doc.core_properties.title="Shrinking a Small Language Model on a Laptop (draft)"; doc.core_properties.author=""',
    'doc.core_properties.title="Shrinking a Small Language Model on a Laptop (draft)"; doc.core_properties.author="Dhruv Vinesh"',
)

open(P, "w").write(src)
print("patched")
