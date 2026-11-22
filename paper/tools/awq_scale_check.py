#!/usr/bin/env python3
"""Check whether the large AWQ vs FP16 weight differences are the folded
per-channel smoothing scales, and report which column holds the 4.97
max-abs on layers.1.mlp.down_proj.

Designed for the laptop workspace:

    cd C:\\Users\\dhruv\\quant-study
    .\\.venv\\Scripts\\python.exe diag\\awq_scale_check.py

Expects the FP16 Hugging Face weights in models\\fp16-src\\*.safetensors
and the OpenVINO IR in models\\ov-awq-int4\\ (and optionally the already
dequantized dict diag\\deq_ov-awq-int4.pt written by diag\\weight_err.py).
Writes diag\\awq_scale_check.txt and prints the same report.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
DIAG = ROOT / "diag"
IR_DIR = ROOT / "models" / "ov-awq-int4"
FP16_GLOB = str(ROOT / "models" / "fp16-src" / "*.safetensors")
DEQ_PT = DIAG / "deq_ov-awq-int4.pt"
OUT = DIAG / "awq_scale_check.txt"

KEYS = [
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.15.mlp.down_proj.weight",
]


def log(lines, msg=""):
    lines.append(msg)
    print(msg, flush=True)


def load_fp16():
    try:
        from safetensors.torch import load_file
        import torch
    except ImportError as e:
        sys.exit(f"need safetensors + torch: {e}")
    ref = {}
    files = glob.glob(FP16_GLOB)
    if not files:
        sys.exit(f"no FP16 safetensors at {FP16_GLOB}")
    for f in files:
        ref.update(load_file(f))
    return {k: ref[k].float().numpy() for k in KEYS if k in ref}


def dequant_from_ir(keys):
    import openvino as ov
    import re
    import torch

    core = ov.Core()
    xml = IR_DIR / "openvino_model.xml"
    if not xml.exists():
        sys.exit(f"missing IR {xml}")
    m = core.read_model(str(xml))
    out = {}
    for op in m.get_ordered_ops():
        if op.get_type_name() != "MatMul":
            continue
        mm = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)", op.get_friendly_name())
        if not mm:
            continue
        key = f"model.layers.{mm.group(1)}.{mm.group(2)}.{mm.group(3)}.weight"
        if key not in keys:
            continue
        w = list(
            core.compile_model(ov.Model([op.input_value(1)], []), "CPU")
            .create_infer_request()
            .infer({})
            .values()
        )[0]
        out[key] = np.asarray(w, np.float32)
        # brief IR chain on the weight input, and on the activation input
        def chain(node, depth=8):
            parts = []
            cur = node
            for _ in range(depth):
                parts.append(f"{cur.get_type_name()}:{cur.get_friendly_name()}")
                if cur.get_input_size() == 0:
                    break
                # follow data input 0
                cur = cur.input_value(0).get_node()
            return " <- ".join(parts)

        out[key + ".ir_weight"] = chain(op.input_value(1).get_node())
        out[key + ".ir_act"] = chain(op.input_value(0).get_node())
    return out


def load_awq(keys, lines):
    if DEQ_PT.exists():
        import torch
        log(lines, f"Loading dequantized AWQ weights from {DEQ_PT}")
        d = torch.load(DEQ_PT, map_location="cpu", weights_only=False)
        out = {}
        for k in keys:
            w = d[k]
            if hasattr(w, "numpy"):
                w = w.numpy()
            out[k] = np.asarray(w, np.float32)
        return out, "deq_pt"
    log(lines, f"{DEQ_PT} not found; dequantizing from {IR_DIR}")
    raw = dequant_from_ir(keys)
    out = {k: raw[k] for k in keys if k in raw}
    for k in keys:
        if k + ".ir_weight" in raw:
            log(lines, f"  IR weight chain {k}: {raw[k + '.ir_weight']}")
            log(lines, f"  IR act    chain {k}: {raw[k + '.ir_act']}")
    return out, "ir"


def align(w, r):
    w = np.asarray(w, np.float32)
    if w.shape != r.shape:
        if w.T.shape == r.shape:
            w = w.T
        else:
            raise ValueError(f"shape mismatch {w.shape} vs {r.shape}")
    return w


def column_report(name, w, r, lines, top=8, col_axis=1):
    """col_axis=1: error over output dim, one value per input column (down_proj).
    col_axis=0: one value per output row (useful for up_proj if scales are on rows).
    """
    e = w - r
    maxabs = float(np.abs(e).max())
    rel = float(np.linalg.norm(e) / np.linalg.norm(r))
    flat = int(np.abs(e).argmax())
    out_i, in_i = divmod(flat, e.shape[1])
    log(lines, f"{name}: shape {tuple(w.shape)} maxabs {maxabs:.6f} rel {rel:.4f}")
    log(lines, f"  argmax |W_awq - W_fp16| at output row {out_i}, input col {in_i}: "
        f"fp16={r[out_i, in_i]:.6f} awq={w[out_i, in_i]:.6f} diff={e[out_i, in_i]:.6f}")
    col_norm = np.linalg.norm(e, axis=0)
    order = np.argsort(col_norm)[::-1]
    log(lines, "  worst input columns by L2 error (col, norm, maxabs, median W_awq/W_fp16):")
    for c in order[:top]:
        rr, ww = r[:, c], w[:, c]
        mask = np.abs(rr) > 1e-6
        ratio = np.median(ww[mask] / rr[mask]) if mask.any() else float("nan")
        log(lines, f"    col {int(c):5d}  ||err||={col_norm[c]:.4f}  maxabs={np.abs(e[:, c]).max():.4f}  med_ratio={ratio:.4f}")
    # highlight 2558 if present
    if e.shape[1] > 2558:
        c = 2558
        rr, ww = r[:, c], w[:, c]
        mask = np.abs(rr) > 1e-6
        ratio = np.median(ww[mask] / rr[mask]) if mask.any() else float("nan")
        rank = int(np.where(order == c)[0][0]) + 1
        log(lines, f"  column 2558: rank {rank}/{e.shape[1]}  ||err||={col_norm[c]:.4f}  "
            f"maxabs={np.abs(e[:, c]).max():.4f}  med_ratio={ratio:.4f}")
    return out_i, in_i, maxabs


def undo_scales(name, w, r, lines, axis):
    """Estimate a per-position scale as the median of W_awq/W_fp16 along the other axis,
    divide it out, and report residual error. axis=1: one scale per input column.
    """
    scales = np.ones(w.shape[axis], dtype=np.float32)
    residual = np.empty_like(w)
    if axis == 1:
        for c in range(w.shape[1]):
            mask = np.abs(r[:, c]) > 1e-6
            s = np.median(w[mask, c] / r[mask, c]) if mask.any() else 1.0
            if not np.isfinite(s) or s == 0:
                s = 1.0
            scales[c] = s
            residual[:, c] = w[:, c] / s - r[:, c]
    else:
        for i in range(w.shape[0]):
            mask = np.abs(r[i, :]) > 1e-6
            s = np.median(w[i, mask] / r[i, mask]) if mask.any() else 1.0
            if not np.isfinite(s) or s == 0:
                s = 1.0
            scales[i] = s
            residual[i, :] = w[i, :] / s - r[i, :]
    raw = float(np.abs(w - r).max())
    res = float(np.abs(residual).max())
    rel = float(np.linalg.norm(residual) / np.linalg.norm(r))
    log(lines, f"  after dividing out per-{'column' if axis==1 else 'row'} median ratio on {name}: "
        f"maxabs {raw:.4f} -> {res:.4f}, rel {rel:.4f}; scale min/median/max "
        f"{scales.min():.4g}/{np.median(scales):.4g}/{scales.max():.4g}")
    return res


def main():
    DIAG.mkdir(exist_ok=True)
    lines = []
    log(lines, "AWQ scale-folding check")
    log(lines, f"cwd={ROOT}")
    fp16 = load_fp16()
    awq, src = load_awq(set(KEYS), lines)
    log(lines, f"AWQ source: {src}")
    log(lines, "")
    for key in KEYS:
        if key not in fp16 or key not in awq:
            log(lines, f"MISSING {key}  fp16={key in fp16} awq={key in awq}")
            continue
        r = fp16[key]
        w = align(awq[key], r)
        log(lines, "=" * 72)
        column_report(key, w, r, lines)
        undo_scales(key, w, r, lines, axis=1)
        undo_scales(key, w, r, lines, axis=0)
        log(lines, "")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(lines, f"Wrote {OUT}")


if __name__ == "__main__":
    main()
