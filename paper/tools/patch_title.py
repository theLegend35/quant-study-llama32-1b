#!/usr/bin/env python3
"""Set the new paper title and make both hardware setups explicit.

    python3 /workspace/paper/tools/patch_title.py
    python3 /workspace/paper/build_paper.py

Written against archive/build_paper_pre_title.py. Changes no results.
"""
from pathlib import Path
import re, ast

P = Path("/workspace/paper/build_paper.py")
INSERT = Path("/workspace/paper/tools/final_body_insert.py")
src = P.read_text()

OLD_TITLE = (
    "Shrinking a Small Language Model on a Laptop: A Controlled Comparison "
    "of Rounding, GPTQ, and AWQ Weight Compression, With and Without LoRA"
)
OLD_CORE = "Shrinking a Small Language Model on a Laptop"
NEW_TITLE = (
    "A Controlled Comparison of RTN, GPTQ, and AWQ Weight Compression "
    "for Llama-3.2-1B, With and Without LoRA Recovery"
)

def replace_once(old, new, label):
    global src
    n = src.count(old)
    assert n == 1, f"{label}: expected 1, found {n}: {old[:80]}"
    src = src.replace(old, new)

# ----- titles -----
replace_once(
    f'tp.add_run("{OLD_TITLE}")',
    f'tp.add_run("{NEW_TITLE}")',
    "visible title",
)
replace_once(
    f'doc.core_properties.title="{OLD_CORE}"',
    f'doc.core_properties.title="{NEW_TITLE}"',
    "core metadata title",
)

# ----- abstract (keep under 250 words) -----
old_abs = (
    "Small language models on consumer laptops are limited by memory capacity and bandwidth. "
    "This controlled benchmark, not a new method, studies weight compression of Llama-3.2-1B "
    "with OpenVINO NNCF 3.4.0 on an Intel Core Ultra 7 laptop (Arc GPU, 16 GB shared RAM). "
    "We compare FP16 with RTN INT8 and with RTN, GPTQ and AWQ INT4 (group size 128, shared calibration set). "
    "Accuracy is measured zero-shot on six tasks plus WikiText-2 perplexity; speed and peak memory are measured with OpenVINO GenAI. "
    "Phase 2 trains the same LoRA recipe (three seeds) on bit-matched FP16, RTN INT8 and RTN INT4 bases. "
    "QLoRA showed that 4-bit LoRA can recover accuracy; LoRA on GPTQ and AWQ is pending."
)
new_abs = (
    "Small language models on consumer laptops are limited by memory capacity and bandwidth. "
    "This controlled benchmark, not a new method, studies weight compression of Llama-3.2-1B "
    "with OpenVINO NNCF 3.4.0. We compare FP16 with RTN INT8 and with RTN, GPTQ and AWQ INT4 "
    "(group size 128, shared calibration set). Phase 1 compression, zero-shot accuracy "
    "(six tasks and WikiText-2) and speed ran on an Intel Core Ultra 7 laptop (Arc iGPU); "
    "Phase 2 LoRA training and evaluation ran on an NVIDIA T4 (Colab), using the same LoRA recipe "
    "(three seeds) on bit-matched FP16, RTN INT8 and RTN INT4 bases. QLoRA showed that 4-bit LoRA "
    "can recover accuracy; LoRA on GPTQ and AWQ is pending."
)
replace_once(old_abs, new_abs, "abstract")

# ----- intro -----
replace_once(
    "Three practical questions remain for a ~1B model on a laptop.",
    "Three practical questions remain for a ~1B model intended to run on a laptop.",
    "intro laptop framing",
)
replace_once(
    "which QLoRA {qlora} has already shown. We asked:",
    "which QLoRA {qlora} has already shown. Phase 1 compression and on-device measurements "
    "ran on an Intel Core Ultra 7 laptop (Arc iGPU); Phase 2 LoRA training and evaluation "
    "ran on an NVIDIA T4 (Colab). We asked:",
    "intro hardware sentence",
)

# ----- contributions -----
replace_once(
    "Zero-shot accuracy of FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 on six tasks and WikiText-2, with a matched group size and calibration set, on the laptop (OpenVINO).",
    "Phase 1 zero-shot accuracy of FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 on six tasks and WikiText-2, with a matched group size and calibration set, on the laptop (OpenVINO).",
    "contrib bullet 1",
)
replace_once(
    "trained on frozen PyTorch bases that bit-match the laptop FP16, RTN INT8 and RTN INT4 models, evaluated on Colab with and without the adapter.",
    "trained on frozen PyTorch bases that bit-match the laptop FP16, RTN INT8 and RTN INT4 models, evaluated on a Colab NVIDIA T4 with and without the adapter.",
    "contrib bullet 3",
)

# ----- setup -----
replace_once(
    "The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. "
    "Phase 2 trained LoRA adapters on Google Colab for the FP16, RTN INT8 and RTN INT4 bases, "
    "with settings fixed before training, and then measured those unmerged adapters on the same laptop.",
    "The study has two phases. Phase 1 compression, accuracy and speed ran on an Intel Core Ultra 7 laptop (Arc iGPU). "
    "Phase 2 LoRA training and accuracy evaluation ran on an NVIDIA T4 (Colab) for the FP16, RTN INT8 and RTN INT4 bases, "
    "with settings fixed before training; the unmerged adapters were then measured on the same laptop.",
    "setup opening",
)
replace_once(
    "LoRA training ran on Google Colab with an NVIDIA T4 GPU.",
    "LoRA training and the Phase 2 accuracy evaluations ran on Google Colab with an NVIDIA T4 GPU.",
    "setup 3.1",
)

# ----- conclusion -----
replace_once(
    "We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 "
    "on an Intel Core Ultra 7 laptop with an integrated Arc GPU.",
    "We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0. "
    "Phase 1 compression and speed ran on an Intel Core Ultra 7 laptop with an integrated Arc iGPU; "
    "Phase 2 LoRA training and evaluation ran on an NVIDIA T4 (Colab).",
    "conclusion opening",
)

# abstract word count
i = src.find('H1("Abstract")')
chunk = src[i:src.find('H1("1 Introduction")')]
words = []
for m in re.finditer(r'para\("((?:[^"\\]|\\.)*)"\)', chunk):
    words.extend(ast.literal_eval('"' + m.group(1) + '"').split())
assert len(words) < 250, f"abstract has {len(words)} words"
assert "[[TBD]]" not in src
assert NEW_TITLE in src
assert OLD_TITLE not in src
assert src.count(OLD_CORE) == 0, f"old core title still present: {src.count(OLD_CORE)}"

P.write_text(src)
print(f"patched {P}; abstract words={len(words)}")

# keep the body insert in sync (used by patch_final; not the live source)
if INSERT.exists():
    ins = INSERT.read_text()
    n = 0
    for a, b in [
        (f'tp.add_run("{OLD_TITLE}")', f'tp.add_run("{NEW_TITLE}")'),
        (old_abs, new_abs),
        ("Three practical questions remain for a ~1B model on a laptop.",
         "Three practical questions remain for a ~1B model intended to run on a laptop."),
        ("which QLoRA {qlora} has already shown. We asked:",
         "which QLoRA {qlora} has already shown. Phase 1 compression and on-device measurements "
         "ran on an Intel Core Ultra 7 laptop (Arc iGPU); Phase 2 LoRA training and evaluation "
         "ran on an NVIDIA T4 (Colab). We asked:"),
        ("Zero-shot accuracy of FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 on six tasks and WikiText-2, with a matched group size and calibration set, on the laptop (OpenVINO).",
         "Phase 1 zero-shot accuracy of FP16, RTN INT8, RTN INT4, GPTQ INT4 and AWQ INT4 on six tasks and WikiText-2, with a matched group size and calibration set, on the laptop (OpenVINO)."),
        ("trained on frozen PyTorch bases that bit-match the laptop FP16, RTN INT8 and RTN INT4 models, evaluated on Colab with and without the adapter.",
         "trained on frozen PyTorch bases that bit-match the laptop FP16, RTN INT8 and RTN INT4 models, evaluated on a Colab NVIDIA T4 with and without the adapter."),
        ("The study has two phases. Phase 1 (compression, accuracy and efficiency) ran on a laptop. "
         "Phase 2 trained LoRA adapters on Google Colab for the FP16, RTN INT8 and RTN INT4 bases, "
         "with settings fixed before training, and then measured those unmerged adapters on the same laptop.",
         "The study has two phases. Phase 1 compression, accuracy and speed ran on an Intel Core Ultra 7 laptop (Arc iGPU). "
         "Phase 2 LoRA training and accuracy evaluation ran on an NVIDIA T4 (Colab) for the FP16, RTN INT8 and RTN INT4 bases, "
         "with settings fixed before training; the unmerged adapters were then measured on the same laptop."),
        ("LoRA training ran on Google Colab with an NVIDIA T4 GPU.",
         "LoRA training and the Phase 2 accuracy evaluations ran on Google Colab with an NVIDIA T4 GPU."),
        ("We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0 "
         "on an Intel Core Ultra 7 laptop with an integrated Arc GPU.",
         "We benchmarked RTN, GPTQ and AWQ weight compression of Llama-3.2-1B with OpenVINO NNCF 3.4.0. "
         "Phase 1 compression and speed ran on an Intel Core Ultra 7 laptop with an integrated Arc iGPU; "
         "Phase 2 LoRA training and evaluation ran on an NVIDIA T4 (Colab)."),
    ]:
        if a in ins:
            ins = ins.replace(a, b)
            n += 1
        else:
            print("insert skip (already changed or missing):", a[:70])
    INSERT.write_text(ins)
    print(f"updated {INSERT} ({n} replacements)")
