#!/usr/bin/env python3
"""Root-cause the systematic median product s_up * s_down ~ 0.966
(not 1) in the AWQ MLP pairing.

Designed for the laptop workspace (same paths as diag/awq_residual_check.py):

    cd C:\\Users\\dhruv\\quant-study
    .\\.venv\\Scripts\\python.exe diag\\awq_offset_check.py

Expects:
  models\\fp16-src\\*.safetensors
  models\\ov-awq-int4\\openvino_model.xml

Read-only on model files. Writes diag\\awq_offset_check.txt and prints the
same report. Completes in well under 5 minutes (dequant of 6 matrices plus
numpy simulations).

Hypotheses
  H1  Estimator bias: the median of W_q/W_fp16 is pulled by INT4 rounding
      and exact zeros. Test by applying known exact-reciprocal scales to
      FP16 (or synthetic) weights, quantizing with the same INT4_SYM
      group-128 RTN rule as NNCF 3.4.0, and re-estimating. Also recompute
      the product with a least-squares estimator
      s = <W_q, W_fp16> / <W_fp16, W_fp16> on nonzero codes.
  H2  AWQ in NNCF is not exactly reciprocal (extra scale elsewhere, or
      scale-estimation / LoRA-correction / an activation-path Multiply).
      Inspect the IR graph around layers.0/1 MLP and the export logs.
  H3  gate_proj rows were scaled too.

Box-side logic check without models:

    python3 awq_offset_check.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
DIAG = ROOT / "diag"
AWQ_IR = ROOT / "models" / "ov-awq-int4"
FP16_GLOB = str(ROOT / "models" / "fp16-src" / "*.safetensors")
OUT = DIAG / "awq_offset_check.txt"

LAYERS = (0, 1)
GROUP = 128
EPS = 1e-6
EPS32 = np.finfo(np.float32).eps
CLIP_LO, CLIP_HI = -8, 7  # INT4_SYM


def log(lines, msg=""):
    lines.append(msg)
    print(msg, flush=True)


def safe_section(lines, title, fn):
    log(lines, "")
    log(lines, "=" * 72)
    log(lines, title)
    log(lines, "=" * 72)
    try:
        return fn()
    except Exception as e:
        log(lines, f"FAILED: {type(e).__name__}: {e}")
        for row in traceback.format_exc().splitlines():
            log(lines, "  " + row)
        return None


def align(w, r):
    w = np.asarray(w, np.float32)
    r = np.asarray(r, np.float32)
    if w.shape != r.shape:
        if w.T.shape == r.shape:
            w = np.ascontiguousarray(w.T)
        else:
            raise ValueError(f"shape mismatch {w.shape} vs {r.shape}")
    return w


def percentiles(x, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}


def fmt_pct(d):
    return "  ".join(f"p{p}={d[p]:.4g}" for p in d)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def median_ratio(num, den, axis, eps=EPS, nonzero_num=False):
    """Median of num/den along `axis`, ignoring |den| <= eps.

    If nonzero_num is True, also drop entries where num is (near) zero,
    i.e. skip exact-zero dequantized weights / zero codes.
    A zero median is treated as unusable (NaN), same as awq_residual_check.py.
    """
    num = np.asarray(num, np.float32)
    den = np.asarray(den, np.float32)
    other = num.shape[0] if axis == 1 else num.shape[1]
    out = np.empty(other, dtype=np.float32)
    if axis == 1:
        for i in range(num.shape[0]):
            mask = np.abs(den[i, :]) > eps
            if nonzero_num:
                mask &= np.abs(num[i, :]) > eps
            if not mask.any():
                out[i] = np.nan
                continue
            s = float(np.median(num[i, mask] / den[i, mask]))
            out[i] = s if np.isfinite(s) and s != 0 else np.nan
        return out
    if axis == 0:
        for j in range(num.shape[1]):
            mask = np.abs(den[:, j]) > eps
            if nonzero_num:
                mask &= np.abs(num[:, j]) > eps
            if not mask.any():
                out[j] = np.nan
                continue
            s = float(np.median(num[mask, j] / den[mask, j]))
            out[j] = s if np.isfinite(s) and s != 0 else np.nan
        return out
    raise ValueError(axis)


def ls_scale(num, den, axis, codes=None, eps=EPS):
    """Least-squares s = <num, den> / <den, den> along `axis`.

    Restricted to nonzero codes when `codes` is given; otherwise to
    entries with |num| > eps and |den| > eps (for SYM, |W_q| > 0 iff code != 0).
    """
    num = np.asarray(num, np.float64)
    den = np.asarray(den, np.float64)
    mask = np.abs(den) > eps
    if codes is not None:
        mask &= np.asarray(codes) != 0
    else:
        mask &= np.abs(num) > eps
    num_m = np.where(mask, num, 0.0)
    den_m = np.where(mask, den, 0.0)
    nd = np.sum(num_m * den_m, axis=axis)
    dd = np.sum(den_m * den_m, axis=axis)
    n = np.sum(mask, axis=axis)
    out = np.full(nd.shape, np.nan, dtype=np.float64)
    ok = (n > 0) & (dd > 0)
    out[ok] = nd[ok] / dd[ok]
    return out


def pairing_stats(s_up, s_down):
    s_up = np.asarray(s_up, np.float64)
    s_down = np.asarray(s_down, np.float64)
    prod = s_up * s_down
    finite = np.isfinite(prod)
    n_fin = int(finite.sum())
    n_all = int(prod.size)
    out = {
        "n_finite": n_fin,
        "n_all": n_all,
        "product": prod,
        "p50": float("nan"),
        "mean": float("nan"),
        "med_abs": float("nan"),
        "frac_5": float("nan"),
        "corr": float("nan"),
        "pct": percentiles(prod),
    }
    if n_fin == 0:
        return out
    pf = prod[finite]
    out["p50"] = float(np.median(pf))
    out["mean"] = float(np.mean(pf))
    out["med_abs"] = float(np.median(np.abs(pf - 1.0)))
    out["frac_5"] = float(np.mean(np.abs(pf - 1.0) < 0.05))
    inv = np.full_like(s_down, np.nan)
    ok = np.isfinite(s_down) & (s_down != 0) & np.isfinite(s_up)
    if ok.sum() >= 3:
        inv[ok] = 1.0 / s_down[ok]
        a, b = s_up[ok], inv[ok]
        if np.std(a) > 0 and np.std(b) > 0:
            out["corr"] = float(np.corrcoef(a, b)[0, 1])
    return out


def log_pairing(lines, label, st):
    log(lines,
        f"  {label}: n={st['n_finite']}/{st['n_all']} finite products  "
        f"p50={st['p50']:.4f}  mean={st['mean']:.4f}  "
        f"median |product-1|={st['med_abs']:.4f}  "
        f"frac within 5% of 1={st['frac_5']:.3f}  "
        f"corr(s_up, 1/s_down)={st['corr']:.4f}")
    log(lines, f"    product percentiles: {fmt_pct(st['pct'])}")


# ---------------------------------------------------------------------------
# INT4_SYM group-128 RTN (NNCF 3.4.0 / verify_phase1.fake_quant_weight)
# ---------------------------------------------------------------------------

def torch_like_round(x):
    """Match torch.round when torch is available (laptop); else half-away-from-zero."""
    try:
        import torch
        return torch.round(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy()
    except ImportError:
        x = np.asarray(x, np.float32)
        return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def int4_sym_rtn(w, group_size=GROUP, fp16_scale=True):
    """Symmetric INT4, groups along the input dim (axis 1).

    Scale rule (from the study's NNCF 3.4.0 replica in verify_phase1.py):
        amin = abs(group min), mx = group max
        scale = (amin if amin >= mx else -mx) / 8
        scale rounded to FP16 when fp16_scale is True
        code = clip(round(w / scale), -8, 7)
        dequant = code * scale
    The scale can be negative, which is also what the AWQ IR stores.
    Returns (dequant, codes, per-group scale of shape (out, n_groups)).
    """
    w = np.asarray(w, np.float32)
    o, i = w.shape
    if i % group_size != 0:
        raise ValueError(f"input dim {i} is not a multiple of group size {group_size}")
    wg = w.reshape(o, i // group_size, group_size)
    amin = np.abs(wg.min(axis=-1, keepdims=True))
    mx = wg.max(axis=-1, keepdims=True)
    scale = np.where(amin >= mx, amin, -mx) / np.float32(8.0)
    scale = np.where(np.abs(scale) < EPS32, np.full_like(scale, EPS32), scale)
    if fp16_scale:
        scale = scale.astype(np.float16).astype(np.float32)
    q = np.clip(torch_like_round(wg / scale), CLIP_LO, CLIP_HI)
    deq = (q * scale).reshape(o, i).astype(np.float32)
    codes = q.reshape(o, i).astype(np.int16)
    return deq, codes, scale.reshape(o, i // group_size)


# ---------------------------------------------------------------------------
# Loading (same as awq_residual_check.py)
# ---------------------------------------------------------------------------

def load_fp16(keys):
    from safetensors.torch import load_file

    ref = {}
    files = glob.glob(FP16_GLOB)
    if not files:
        raise FileNotFoundError(f"no FP16 safetensors at {FP16_GLOB}")
    for f in files:
        ref.update(load_file(f))
    out = {}
    for k in keys:
        if k not in ref:
            continue
        w = ref[k]
        out[k] = w.float().numpy() if hasattr(w, "float") else np.asarray(w, np.float32)
    return out


def _chain(node, depth=12):
    parts = []
    cur = node
    for _ in range(depth):
        try:
            et = cur.get_output_element_type(0)
            sh = cur.get_output_partial_shape(0)
        except Exception:
            et, sh = "?", "?"
        parts.append(f"{cur.get_type_name()}:{cur.get_friendly_name()}:{et}:{sh}")
        if cur.get_input_size() == 0:
            break
        cur = cur.input_value(0).get_node()
    return " <- ".join(parts)


def _const_data(node):
    cur = node
    for _ in range(6):
        if cur.get_type_name() in ("Convert", "Reshape", "Squeeze", "Unsqueeze") and cur.get_input_size() >= 1:
            cur = cur.input_value(0).get_node()
            continue
        break
    try:
        if cur.get_type_name() != "Constant":
            return None, f"not a Constant ({cur.get_type_name()}:{cur.get_friendly_name()})"
        data = cur.get_data()
        arr = np.array(data)
        return arr, f"{cur.get_friendly_name()} dtype={arr.dtype} shape={tuple(arr.shape)}"
    except Exception as e:
        return None, f"get_data failed on {cur.get_type_name()}:{cur.get_friendly_name()}: {e}"


def find_matmul(model, key):
    parts = key.replace(".weight", "").split(".")
    pat = rf"layers\.{parts[2]}\.{parts[3]}\.{parts[4]}"
    for op in model.get_ordered_ops():
        if op.get_type_name() != "MatMul":
            continue
        if re.search(pat, op.get_friendly_name()):
            return op
    return None


def dequant_weight(core, ov, op):
    w = list(
        core.compile_model(ov.Model([op.input_value(1)], []), "CPU")
        .create_infer_request()
        .infer({})
        .values()
    )[0]
    return np.asarray(w, np.float32)


def key_for(layer, proj):
    return f"model.layers.{layer}.mlp.{proj}.weight"


# ---------------------------------------------------------------------------
# H1 / H3 helpers
# ---------------------------------------------------------------------------

def fill_reciprocal(s_up, s_down):
    """Build an exact-reciprocal pair, preferring whichever side is finite."""
    s_up = np.array(s_up, dtype=np.float64, copy=True)
    s_down = np.array(s_down, dtype=np.float64, copy=True)
    su = np.array(s_up, dtype=np.float64, copy=True)
    for i in range(su.size):
        u, d = s_up[i], s_down[i]
        u_ok = np.isfinite(u) and u != 0
        d_ok = np.isfinite(d) and d != 0
        if u_ok:
            su[i] = u
        elif d_ok:
            su[i] = 1.0 / d
        else:
            su[i] = 1.0
    sd = 1.0 / su
    return su.astype(np.float32), sd.astype(np.float32)


def rel_fro(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def describe_scales(lines, name, s, s_ref=None):
    s = np.asarray(s, np.float64)
    fin = s[np.isfinite(s)]
    log(lines, f"  {name}: n_finite={fin.size}/{s.size}  {fmt_pct(percentiles(s))}")
    if fin.size:
        log(lines, f"    |s-1|: median={np.median(np.abs(fin-1)):.4f}  "
            f"frac |s-1|<0.05={np.mean(np.abs(fin-1)<0.05):.3f}  "
            f"frac |s-1|<0.20={np.mean(np.abs(fin-1)<0.20):.3f}")
    if s_ref is not None:
        a = np.asarray(s_ref, np.float64)
        ok = np.isfinite(s) & np.isfinite(a) & (a != 0)
        if ok.sum() >= 3:
            corr = float(np.corrcoef(s[ok], a[ok])[0, 1]) if (np.std(s[ok]) > 0 and np.std(a[ok]) > 0) else float("nan")
            log(lines, f"    corr with reference scale: {corr:.4f}  n={int(ok.sum())}")


# ---------------------------------------------------------------------------
# H2: IR activation-path inspection
# ---------------------------------------------------------------------------

def _input_nodes(op):
    out = []
    try:
        n = op.get_input_size()
    except Exception:
        return out
    for i in range(n):
        try:
            out.append((i, op.input_value(i).get_node()))
        except Exception:
            pass
    return out


def _const_inputs_of(op):
    """Return (index, array or None, note) for each input that is a folded Constant."""
    found = []
    for i, node in _input_nodes(op):
        arr, note = _const_data(node)
        if arr is not None or "Constant" in note:
            found.append((i, arr, note))
    return found


def _summarize_array(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    flat = arr.astype(np.float64, copy=False).ravel()
    uniq = np.unique(flat)
    msg = (f"shape={tuple(arr.shape)} dtype={arr.dtype} min={flat.min():.6g} "
           f"max={flat.max():.6g} median={np.median(flat):.6g} n_unique={uniq.size}")
    if uniq.size <= 8:
        msg += f" values={uniq.tolist()}"
    return msg


def inspect_mlp_activation_path(model, layer, lines, s_up=None, s_down=None):
    """Walk from down_proj's activation input back toward up_proj / gate_proj.

    Report every Multiply/Divide on that path, and any Constant factors.
    Also scan other ops in the layer MLP whose names mention layers.N.mlp,
    looking for stray Constant scales on the activation path.
    """
    flags = {
        "constant_scale_on_act_path": False,
        "multiplies": [],
        "chain": None,
        "notes": [],
    }
    ups = {
        "up": find_matmul(model, key_for(layer, "up_proj")),
        "gate": find_matmul(model, key_for(layer, "gate_proj")),
        "down": find_matmul(model, key_for(layer, "down_proj")),
    }
    for name, op in ups.items():
        if op is None:
            log(lines, f"  layer {layer}: no MatMul for {name}_proj")
        else:
            log(lines, f"  layer {layer} {name}_proj MatMul: {op.get_friendly_name()}")

    down = ups["down"]
    if down is None:
        flags["notes"].append("no down_proj MatMul")
        return flags

    # Backward chain along activation input 0 of down_proj.
    try:
        start = down.input_value(0).get_node()
    except Exception as e:
        log(lines, f"  could not read down_proj input 0: {e}")
        flags["notes"].append(str(e))
        return flags

    log(lines, f"  down_proj activation input chain:")
    log(lines, f"    {_chain(start, depth=16)}")
    flags["chain"] = _chain(start, depth=16)

    # Walk back, recording Multiply/Divide and stopping at a MatMul (up/gate).
    cur = start
    visited = set()
    act_multiplies = []
    for step in range(20):
        if id(cur) in visited:
            break
        visited.add(id(cur))
        tname = cur.get_type_name()
        fname = cur.get_friendly_name()
        if tname in ("Multiply", "Divide", "Add", "Subtract"):
            consts = _const_inputs_of(cur)
            n_in = cur.get_input_size()
            log(lines, f"    [{tname}] {fname}  n_inputs={n_in}  constant_inputs={len(consts)}")
            rec = {"type": tname, "name": fname, "n_inputs": n_in, "constants": []}
            for idx, arr, note in consts:
                rec["constants"].append((idx, note, None if arr is None else arr.shape))
                if arr is not None:
                    log(lines, f"      input {idx} Constant: {note}  {_summarize_array(arr)}")
                    flags["constant_scale_on_act_path"] = True
                    # Compare to recovered scales if the length matches the intermediate dim.
                    flat = np.asarray(arr, np.float64).ravel()
                    for label, ref in (("s_up", s_up), ("s_down", s_down), ("1/s_up", None if s_up is None else 1.0 / np.asarray(s_up, np.float64))):
                        if ref is None:
                            continue
                        ref = np.asarray(ref, np.float64).ravel()
                        if ref.size != flat.size:
                            continue
                        ok = np.isfinite(ref) & np.isfinite(flat) & (ref != 0)
                        if ok.sum() >= 3 and np.std(ref[ok]) > 0 and np.std(flat[ok]) > 0:
                            corr = float(np.corrcoef(flat[ok], ref[ok])[0, 1])
                            med_ratio = float(np.median(flat[ok] / ref[ok]))
                            log(lines, f"        vs {label}: corr={corr:.4f}  median(const/{label})={med_ratio:.4f}")
                else:
                    log(lines, f"      input {idx}: {note}")
            if not consts:
                log(lines, f"      no Constant input (dynamic * dynamic, e.g. silu(gate)*up, is expected)")
            act_multiplies.append(rec)
            flags["multiplies"].append(rec)
        if tname == "MatMul":
            log(lines, f"    reached MatMul {fname}; stopping this branch")
            break
        # Prefer walking input 0; if a node has several dynamic inputs (gate * up),
        # note the other inputs too.
        inputs = _input_nodes(cur)
        if not inputs:
            break
        # If more than one input, mention them, then follow all non-constant ones
        # one level (shallow) so we don't miss a scale on a sibling.
        dynamic = []
        for i, n in inputs:
            arr, note = _const_data(n)
            if arr is None and "Constant" not in note:
                dynamic.append((i, n))
        if not dynamic:
            break
        if len(dynamic) > 1:
            for i, n in dynamic[1:]:
                log(lines, f"    also input {i} of {tname}:{fname}: {n.get_type_name()}:{n.get_friendly_name()}")
                if n.get_type_name() == "MatMul":
                    log(lines, f"      (MatMul, probably the other of up/gate)")
                else:
                    log(lines, f"      chain: {_chain(n, depth=8)}")
                    # one-level constant check on this sibling
                    if n.get_type_name() in ("Multiply", "Divide"):
                        for idx, arr, note in _const_inputs_of(n):
                            if arr is not None:
                                flags["constant_scale_on_act_path"] = True
                                log(lines, f"      sibling Constant input {idx}: {note}  {_summarize_array(arr)}")
        cur = dynamic[0][1]

    # Name-scoped scan: any Multiply/Divide under layers.N.mlp that has a Constant
    # and is not obviously a weight-dequant Multiply sitting on a MatMul weight input.
    pat = re.compile(rf"layers\.{layer}\.mlp")
    weight_muls = set()
    for name, op in ups.items():
        if op is None:
            continue
        try:
            n = op.input_value(1).get_node()
        except Exception:
            continue
        for _ in range(12):
            if n.get_type_name() == "Multiply":
                weight_muls.add(id(n))
                break
            if n.get_input_size() == 0:
                break
            n = n.input_value(0).get_node()

    log(lines, f"  name-scoped scan of Multiply/Divide under layers.{layer}.mlp "
        f"(excluding known weight-dequant Multiplies):")
    n_extra = 0
    for op in model.get_ordered_ops():
        tname = op.get_type_name()
        if tname not in ("Multiply", "Divide"):
            continue
        fname = op.get_friendly_name()
        if not pat.search(fname):
            continue
        if id(op) in weight_muls:
            continue
        consts = [(i, arr, note) for i, arr, note in _const_inputs_of(op) if arr is not None]
        if not consts:
            continue
        n_extra += 1
        flags["constant_scale_on_act_path"] = True
        log(lines, f"    EXTRA {tname} {fname}")
        for i, arr, note in consts:
            log(lines, f"      input {i}: {note}  {_summarize_array(arr)}")
    if n_extra == 0:
        log(lines, "    none (no extra Constant-scaled Multiply/Divide in the named MLP subgraph)")
    flags["n_extra_named"] = n_extra
    return flags


# ---------------------------------------------------------------------------
# H2: export logs / run scripts
# ---------------------------------------------------------------------------

FLAG_PATTERNS = (
    "scale-estimation",
    "scale_estimation",
    "scale estimation",
    "lora-correction",
    "lora_correction",
    "lora correction",
    "awq",
    "gptq",
    "--sym",
    "--dataset",
    "--awq",
    "--gptq",
    "--ratio",
    "--group-size",
    "--weight-format",
    "--num-samples",
    "applying data-aware awq",
    "applying gptq",
    "applying weight compression",
    "export openvino",
)


def read_text_auto(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace")


def parse_export_log(text):
    """Pull NNCF progress lines and flag mentions out of an export log or script."""
    info = {
        "has_awq_step": False,
        "has_gptq_step": False,
        "has_scale_estimation_step": False,
        "has_weight_compression_step": False,
        "has_lora_correction": False,
        "flag_mentions": [],
        "applying_lines": [],
        "cli_lines": [],
    }
    for i, line in enumerate(text.splitlines()):
        low = line.lower()
        if "applying" in low and ("awq" in low or "gptq" in low or "weight compression" in low or "scale" in low or "lora" in low):
            info["applying_lines"].append(line.strip())
        if re.search(r"applying\s+data-aware\s+awq", low):
            info["has_awq_step"] = True
        if re.search(r"applying\s+gptq", low):
            info["has_gptq_step"] = True
        if "applying" in low and "weight compression" in low:
            info["has_weight_compression_step"] = True
        if "scale estimation" in low or "scale-estimation" in low or "scale_estimation" in low:
            # "Applying GPTQ with Scale Estimation" counts as a step;
            # a comment in a script mentioning the flag also gets recorded.
            if "applying" in low:
                info["has_scale_estimation_step"] = True
            info["flag_mentions"].append(("scale-estimation", i + 1, line.strip()))
        if "lora correction" in low or "lora-correction" in low or "lora_correction" in low:
            info["has_lora_correction"] = True
            info["flag_mentions"].append(("lora-correction", i + 1, line.strip()))
        if "export openvino" in low or "--awq" in low or "--scale-estimation" in low:
            info["cli_lines"].append((i + 1, line.strip()))
    return info


def find_log_candidates():
    names = [
        "export-awq-int4.log",
        "export-gptq-int4.log",
        "build-ov-gptq-int4-se.log",
        "run_ablations.ps1",
    ]
    dirs = [ROOT, ROOT / "results", DIAG, DIAG / "results"]
    found = []
    seen = set()
    for d in dirs:
        for name in names:
            p = d / name
            if p.is_file() and p.resolve() not in seen:
                seen.add(p.resolve())
                found.append(p)
    return found


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest(verbose=True):
    failures = []

    def check(name, cond, detail=""):
        if cond:
            if verbose:
                print(f"  PASS {name}", flush=True)
        else:
            failures.append(f"{name}: {detail}")
            print(f"  FAIL {name}: {detail}", flush=True)

    # INT4_SYM scale: mixed group, |min| dominates -> positive scale = |min|/8
    g = np.zeros((1, 128), np.float32)
    g[0, 0] = -16.0
    g[0, 1] = 4.0
    deq, codes, sc = int4_sym_rtn(g, fp16_scale=False)
    check("scale when |min| dominates", abs(float(sc[0, 0]) - 2.0) < 1e-6, f"scale={sc[0,0]}")
    check("code of -16 is -8", int(codes[0, 0]) == -8, f"{codes[0,0]}")
    check("dequant of -16 is exact", abs(float(deq[0, 0]) + 16.0) < 1e-5, f"{deq[0,0]}")

    # Positive max dominates -> negative scale = -max/8
    g = np.zeros((1, 128), np.float32)
    g[0, 0] = 8.0
    g[0, 1] = -4.0
    deq, codes, sc = int4_sym_rtn(g, fp16_scale=False)
    check("scale when max dominates is -1", abs(float(sc[0, 0]) + 1.0) < 1e-6, f"scale={sc[0,0]}")
    check("code of +8 is -8 (negative scale)", int(codes[0, 0]) == -8, f"{codes[0,0]}")
    check("dequant of +8 is exact", abs(float(deq[0, 0]) - 8.0) < 1e-5, f"{deq[0,0]}")

    # FP16 scale rounding is applied when requested
    g = np.zeros((1, 128), np.float32)
    g[0, 0] = 1.0  # scale would be -1/8 = -0.125, already a f16 value
    _, _, sc16 = int4_sym_rtn(g, fp16_scale=True)
    _, _, sc32 = int4_sym_rtn(g, fp16_scale=False)
    check("fp16_scale leaves -0.125 unchanged", abs(float(sc16[0, 0]) + 0.125) < 1e-8)

    # LS recovers a known scale, including after some zeros
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, size=(4, 128)).astype(np.float32)
    s_true = np.array([1.5, 2.0, 0.5, 4.0], dtype=np.float32)
    y = x * s_true[:, None]
    y[0, :10] = 0  # zero codes
    hat = ls_scale(y, x, axis=1)
    check("LS recovers known row scales", np.allclose(hat, s_true, rtol=1e-5, atol=1e-5), f"{hat}")

    # Median + LS without quantization, exact reciprocals -> product 1
    out, inn, hid = 32, 256, 16
    fp_u = rng.normal(0, 0.05, size=(out, inn)).astype(np.float32)
    fp_d = rng.normal(0, 0.05, size=(hid, out)).astype(np.float32)
    s = np.linspace(0.5, 8.0, out).astype(np.float32)
    aw_u = fp_u * s[:, None]
    aw_d = fp_d * (1.0 / s)[None, :]
    su = median_ratio(aw_u, fp_u, 1)
    sd = median_ratio(aw_d, fp_d, 0)
    st = pairing_stats(su, sd)
    check("no-quant median product is 1", abs(st["p50"] - 1.0) < 1e-5, f"p50={st['p50']}")
    sul = ls_scale(aw_u, fp_u, 1)
    sdl = ls_scale(aw_d, fp_d, 0)
    stl = pairing_stats(sul, sdl)
    check("no-quant LS product is 1", abs(stl["p50"] - 1.0) < 1e-5, f"p50={stl['p50']}")

    # With INT4, median product should drop below 1 (zeros / rounding bias).
    # Use small weights and a spread of scales so many values round to zero.
    # down_proj input dim is the up_proj row count, so it must be a multiple of 128.
    out, inn, hid = 128, 256, 64
    fp_u = rng.normal(0, 0.02, size=(out, inn)).astype(np.float32)
    fp_d = rng.normal(0, 0.02, size=(hid, out)).astype(np.float32)
    s = np.exp(rng.normal(0, 0.6, size=out)).astype(np.float32)
    aw_u, cu, _ = int4_sym_rtn(fp_u * s[:, None])
    aw_d, cd, _ = int4_sym_rtn(fp_d * (1.0 / s)[None, :])
    st_m = pairing_stats(median_ratio(aw_u, fp_u, 1), median_ratio(aw_d, fp_d, 0))
    st_l = pairing_stats(ls_scale(aw_u, fp_u, 1, cu), ls_scale(aw_d, fp_d, 0, cd))
    check("INT4 median product is below 1 (estimator bias direction)",
          st_m["n_finite"] >= 8 and st_m["p50"] < 0.99,
          f"p50={st_m['p50']} n={st_m['n_finite']}")
    check("INT4 LS product is closer to 1 than the median product",
          st_l["n_finite"] >= 8 and abs(st_l["p50"] - 1.0) < abs(st_m["p50"] - 1.0),
          f"LS p50={st_l['p50']} median p50={st_m['p50']}")

    # fill_reciprocal
    a, b = fill_reciprocal([2.0, np.nan, 0.0], [np.nan, 0.25, 0.0])
    check("fill_reciprocal uses up when finite", abs(a[0] - 2.0) < 1e-6 and abs(b[0] - 0.5) < 1e-6, f"{a[0]},{b[0]}")
    check("fill_reciprocal uses 1/down when up is nan", abs(a[1] - 4.0) < 1e-6 and abs(b[1] - 0.25) < 1e-6, f"{a[1]},{b[1]}")
    check("fill_reciprocal falls back to 1", abs(a[2] - 1.0) < 1e-6, f"{a[2]}")

    # Log parser
    awq_txt = (
        "Applying data-aware AWQ ------------------------- 100% 16/16\n"
        "Applying Weight Compression --------------------------- 100%\n"
    )
    se_txt = "Applying GPTQ with Scale Estimation ----------- 100% 112/112\n"
    pa, ps = parse_export_log(awq_txt), parse_export_log(se_txt)
    check("parser: AWQ step, no scale-estimation step",
          pa["has_awq_step"] and pa["has_weight_compression_step"] and not pa["has_scale_estimation_step"])
    check("parser: SE log has scale-estimation step",
          ps["has_scale_estimation_step"] and ps["has_gptq_step"])

    if failures:
        raise AssertionError("selftest failed: " + "; ".join(failures))
    if verbose:
        print("selftest: all checks passed", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def section_h1_h3(lines, fp16, awq):
    results = {}
    for layer in LAYERS:
        log(lines, f"--- layer {layer} ---")
        ku, kd, kg = key_for(layer, "up_proj"), key_for(layer, "down_proj"), key_for(layer, "gate_proj")
        if ku not in fp16 or kd not in fp16 or ku not in awq or kd not in awq:
            log(lines, f"  missing weights for layer {layer}; skip")
            continue
        wu = align(awq[ku], fp16[ku])
        ru = np.asarray(fp16[ku], np.float32)
        wd = align(awq[kd], fp16[kd])
        rd = np.asarray(fp16[kd], np.float32)

        su_med = median_ratio(wu, ru, 1)
        sd_med = median_ratio(wd, rd, 0)
        su_nz = median_ratio(wu, ru, 1, nonzero_num=True)
        sd_nz = median_ratio(wd, rd, 0, nonzero_num=True)
        su_ls = ls_scale(wu, ru, 1)
        sd_ls = ls_scale(wd, rd, 0)

        st_med = pairing_stats(su_med, sd_med)
        st_nz = pairing_stats(su_nz, sd_nz)
        st_ls = pairing_stats(su_ls, sd_ls)
        log(lines, "  Real AWQ vs FP16 (no simulation):")
        log_pairing(lines, "median (includes zero dequant values)", st_med)
        log_pairing(lines, "median on nonzero dequant values only", st_nz)
        log_pairing(lines, "least-squares on nonzero codes", st_ls)

        # H1 simulation on this layer's real FP16 weights, exact-reciprocal scales.
        su_true, sd_true = fill_reciprocal(su_ls, sd_ls)
        n_from_up = int(np.isfinite(su_ls).sum())
        log(lines, f"  Simulation true scales: exact reciprocals filled from the LS estimates "
            f"({n_from_up}/{su_ls.size} from s_up_LS; others from 1/s_down or 1). "
            "True product is exactly 1 before quantization.")
        sim_u, cu, _ = int4_sym_rtn(ru * su_true[:, None])
        sim_d, cd, _ = int4_sym_rtn(rd * sd_true[None, :])
        st_sim_med = pairing_stats(median_ratio(sim_u, ru, 1), median_ratio(sim_d, rd, 0))
        st_sim_nz = pairing_stats(
            median_ratio(sim_u, ru, 1, nonzero_num=True),
            median_ratio(sim_d, rd, 0, nonzero_num=True),
        )
        st_sim_ls = pairing_stats(ls_scale(sim_u, ru, 1, cu), ls_scale(sim_d, rd, 0, cd))
        log(lines, "  After applying those exact-reciprocal scales and INT4_SYM g128 RTN:")
        log_pairing(lines, "sim median (includes zeros)", st_sim_med)
        log_pairing(lines, "sim median on nonzero only", st_sim_nz)
        log_pairing(lines, "sim least-squares on nonzero codes", st_sim_ls)

        # Bias of each side vs the known true scale
        def bias_line(label, hat, true):
            r = np.asarray(hat, np.float64) / np.asarray(true, np.float64)
            f = np.isfinite(r)
            if not f.any():
                log(lines, f"    {label}: no finite ratios")
                return
            log(lines, f"    {label}: median(s_hat/s_true)={np.median(r[f]):.4f}  "
                f"mean={np.mean(r[f]):.4f}  n={int(f.sum())}")

        bias_line("sim median s_up  / true", median_ratio(sim_u, ru, 1), su_true)
        bias_line("sim median s_down / true", median_ratio(sim_d, rd, 0), sd_true)
        bias_line("sim LS     s_up  / true", ls_scale(sim_u, ru, 1, cu), su_true)
        bias_line("sim LS     s_down / true", ls_scale(sim_d, rd, 0, cd), sd_true)

        # Also a synthetic-weight control at this layer shape, independent of FP16.
        rng = np.random.default_rng(1000 + layer)
        syn_u = rng.normal(0, float(np.std(ru)), size=ru.shape).astype(np.float32)
        syn_d = rng.normal(0, float(np.std(rd)), size=rd.shape).astype(np.float32)
        q_u, csu, _ = int4_sym_rtn(syn_u * su_true[:, None])
        q_d, csd, _ = int4_sym_rtn(syn_d * sd_true[None, :])
        st_syn_med = pairing_stats(median_ratio(q_u, syn_u, 1), median_ratio(q_d, syn_d, 0))
        st_syn_ls = pairing_stats(ls_scale(q_u, syn_u, 1, csu), ls_scale(q_d, syn_d, 0, csd))
        log(lines, "  Synthetic Gaussian weights (same shapes and std, same exact-reciprocal scales, then INT4):")
        log_pairing(lines, "syn median (includes zeros)", st_syn_med)
        log_pairing(lines, "syn least-squares on nonzero codes", st_syn_ls)

        # H3 gate_proj
        if kg in fp16 and kg in awq:
            wg = align(awq[kg], fp16[kg])
            rg = np.asarray(fp16[kg], np.float32)
            sg_med = median_ratio(wg, rg, 1)
            sg_ls = ls_scale(wg, rg, 1)
            log(lines, f"  H3 gate_proj vs FP16: maxabs={np.max(np.abs(wg-rg)):.4f}  "
                f"rel={rel_fro(wg, rg):.4f}  (ordinary INT4 in the residual check was ~0.106)")
            describe_scales(lines, "gate row median s", sg_med, su_med)
            describe_scales(lines, "gate row LS s", sg_ls, su_ls)
            results.setdefault(layer, {})["gate_rel"] = rel_fro(wg, rg)
            results[layer]["gate_med_p50"] = float(np.nanmedian(sg_med))
            results[layer]["gate_ls_p50"] = float(np.nanmedian(sg_ls))
        else:
            log(lines, "  H3: gate_proj weights missing")

        results[layer] = {
            **results.get(layer, {}),
            "real_median": st_med,
            "real_median_nz": st_nz,
            "real_ls": st_ls,
            "sim_median": st_sim_med,
            "sim_median_nz": st_sim_nz,
            "sim_ls": st_sim_ls,
            "syn_median": st_syn_med,
            "syn_ls": st_syn_ls,
            "su_ls": su_ls,
            "sd_ls": sd_ls,
            "su_med": su_med,
            "sd_med": sd_med,
        }
    return results


def section_h2_ir(lines, awq_model, h1):
    flags = {}
    for layer in LAYERS:
        log(lines, f"--- layer {layer} IR activation path ---")
        s_up = h1.get(layer, {}).get("su_ls") if h1 else None
        s_down = h1.get(layer, {}).get("sd_ls") if h1 else None
        flags[layer] = inspect_mlp_activation_path(awq_model, layer, lines, s_up=s_up, s_down=s_down)
        log(lines, f"  constant scale on activation path: "
            f"{flags[layer]['constant_scale_on_act_path']}")
    return flags


def section_h2_logs(lines):
    found = find_log_candidates()
    if not found:
        log(lines, "No export-awq-int4.log / related scripts found under cwd, results/, or diag/. "
            "Looked for export-awq-int4.log, export-gptq-int4.log, "
            "build-ov-gptq-int4-se.log, run_ablations.ps1.")
        return {"files": []}
    info_by = {}
    for p in found:
        log(lines, f"--- {p} ---")
        try:
            text = read_text_auto(p)
        except Exception as e:
            log(lines, f"  could not read: {e}")
            continue
        info = parse_export_log(text)
        info_by[str(p)] = info
        log(lines, f"  has data-aware AWQ step:          {info['has_awq_step']}")
        log(lines, f"  has GPTQ step:                    {info['has_gptq_step']}")
        log(lines, f"  has 'Scale Estimation' apply step:{info['has_scale_estimation_step']}")
        log(lines, f"  has Weight Compression step:      {info['has_weight_compression_step']}")
        log(lines, f"  mentions LoRA correction:         {info['has_lora_correction']}")
        if info["applying_lines"]:
            log(lines, "  Applying-lines:")
            for row in info["applying_lines"][:12]:
                log(lines, "    " + row)
        if info["cli_lines"]:
            log(lines, "  CLI / flag lines (export logs often truncate the full command):")
            for lineno, row in info["cli_lines"][:20]:
                log(lines, f"    L{lineno}: {row}")
        if info["flag_mentions"]:
            log(lines, "  scale-estimation / lora-correction mentions:")
            for kind, lineno, row in info["flag_mentions"][:20]:
                log(lines, f"    L{lineno} [{kind}]: {row}")
        if p.name == "run_ablations.ps1":
            # Quote the SE flag line if present, for contrast with the default AWQ build.
            for i, row in enumerate(text.splitlines(), 1):
                if "scale-estimation" in row.lower() or "scale_estimation" in row.lower():
                    log(lines, f"  script L{i}: {row.strip()}")
    return {"files": found, "info": info_by}


def section_verdict(lines, h1, ir_flags, log_info):
    log(lines, "Numbers below are from this run, not from memory.")
    any_layer = [L for L in LAYERS if L in (h1 or {})]
    if not any_layer:
        log(lines, "No layer results; cannot form a data verdict on the real model.")
        return

    def p50(block, key):
        st = (block or {}).get(key)
        return None if st is None else st.get("p50")

    for layer in any_layer:
        b = h1[layer]
        log(lines,
            f"  layer {layer}: real median product p50={p50(b,'real_median'):.4f}, "
            f"real nonzero-median p50={p50(b,'real_median_nz'):.4f}, "
            f"real LS p50={p50(b,'real_ls'):.4f}")
        log(lines,
            f"           sim  median product p50={p50(b,'sim_median'):.4f} "
            f"(true product was 1), sim LS p50={p50(b,'sim_ls'):.4f}")

    # Decision rules, stated cautiously.
    real_meds = [p50(h1[L], "real_median") for L in any_layer]
    real_ls = [p50(h1[L], "real_ls") for L in any_layer]
    sim_meds = [p50(h1[L], "sim_median") for L in any_layer]
    sim_ls = [p50(h1[L], "sim_ls") for L in any_layer]
    gate_rels = [h1[L].get("gate_rel") for L in any_layer if h1[L].get("gate_rel") is not None]
    gate_p50 = [h1[L].get("gate_med_p50") for L in any_layer if h1[L].get("gate_med_p50") is not None]
    ir_extra = any((ir_flags or {}).get(L, {}).get("constant_scale_on_act_path") for L in any_layer)

    se_in_awq = False
    se_in_se_log = False
    awq_step = False
    lora_corr = False
    n_logs = 0
    if log_info and log_info.get("info"):
        n_logs = len(log_info["info"])
        for path, info in log_info["info"].items():
            name = Path(path).name.lower()
            if "awq" in name and "gptq" not in name:
                awq_step = awq_step or info["has_awq_step"]
                se_in_awq = se_in_awq or info["has_scale_estimation_step"]
                lora_corr = lora_corr or info["has_lora_correction"]
            if "se" in name or "scale" in name:
                se_in_se_log = se_in_se_log or info["has_scale_estimation_step"]
            # run_ablations is not the default AWQ export
            if name.startswith("export-awq"):
                se_in_awq = se_in_awq or info["has_scale_estimation_step"]
                awq_step = awq_step or info["has_awq_step"]

    def near(x, target, tol):
        return x is not None and np.isfinite(x) and abs(x - target) <= tol

    def all_near(xs, target, tol):
        return xs and all(near(x, target, tol) for x in xs)

    log(lines, "")
    log(lines, "Interpretation:")
    # H1: simulation with true product 1 reproduces a median product near the real 0.966,
    # and/or LS on real data is clearly closer to 1 than the median.
    h1_sim_reproduces = False
    if real_meds and sim_meds and all(v is not None for v in real_meds + sim_meds):
        # Within 0.02 of the observed real median, and clearly below 0.99.
        h1_sim_reproduces = all(
            np.isfinite(sm) and sm < 0.99 and abs(sm - rm) <= 0.025
            for sm, rm in zip(sim_meds, real_meds)
        )
        h1_sim_direction = all(np.isfinite(sm) and sm < 0.99 for sm in sim_meds)
    else:
        h1_sim_direction = False

    ls_real_close = all_near(real_ls, 1.0, 0.02)
    ls_real_closer = (
        real_ls and real_meds
        and all(l is not None and m is not None and abs(l - 1) < abs(m - 1) - 0.005
                for l, m in zip(real_ls, real_meds))
    )
    ls_sim_close = all_near(sim_ls, 1.0, 0.02)

    if h1_sim_reproduces and (ls_real_close or ls_real_closer) and ls_sim_close:
        log(lines, "  H1 is supported: applying exact-reciprocal scales to the real FP16 "
            "weights, then quantizing with INT4_SYM g128 RTN, reproduces a median "
            "product close to the observed 0.966 even though the true product is 1. "
            "The least-squares estimator, restricted to nonzero codes, brings the "
            "product back near 1 on both the simulation and the real AWQ weights. "
            "The offset is therefore an artifact of the per-row/column median of "
            "ratios after INT4 rounding and zeros, not evidence that the folded "
            "AWQ scales fail to be reciprocal.")
    elif h1_sim_direction and (ls_real_close or ls_real_closer):
        log(lines, "  H1 is partly supported: the INT4 simulation moves the median product "
            "below 1, and least-squares on nonzero codes is closer to 1 than the "
            "median on the real weights. The simulated median product does not "
            "match the observed 0.966 as closely as one would want for a full "
            "reproduction, so treat the size of the offset as only partly explained.")
    elif h1_sim_direction and not ls_real_closer:
        log(lines, "  H1 is incomplete: the simulation can bias the median product downward, "
            "but the least-squares product on the real AWQ weights is not clearly "
            "closer to 1. A genuine non-reciprocal residual (H2) may still be present.")
    else:
        log(lines, "  H1 is not supported by this run (the simulation did not produce a "
            "downward median-product bias, or results were missing).")

    if ir_extra:
        log(lines, "  H2 IR: a Constant-scaled Multiply/Divide was found on the activation "
            "path between up_proj/gate and down_proj (see the chain above). That "
            "could hold a leftover scale that the weight-only pairing would miss. "
            "Compare those Constant values to the recovered s_up / s_down.")
    else:
        log(lines, "  H2 IR: no Constant-scaled Multiply/Divide was found on the activation "
            "path in layers.0/1 MLP. The only Multiply that is expected there is "
            "the dynamic silu(gate)*up product. This does not support an "
            "un-folded residual scale sitting between up_proj and down_proj.")

    if n_logs == 0:
        log(lines, "  H2 logs: no export log was found in cwd / results / diag, so flags "
            "could not be checked here.")
    else:
        log(lines, f"  H2 logs: data-aware AWQ step in the AWQ export log: {awq_step}; "
            f"a 'Scale Estimation' apply step in that same AWQ log: {se_in_awq}; "
            f"LoRA-correction mention in the AWQ log: {lora_corr}. "
            "The default GPTQ-with-scale-estimation ablation log is expected to "
            "contain 'Applying GPTQ with Scale Estimation'; the default AWQ log "
            "should not, if scale estimation was left off.")
        if awq_step and not se_in_awq:
            log(lines, "    The AWQ export log shows data-aware AWQ followed by weight "
                "compression, and does not show a Scale Estimation apply step. "
                "That is evidence against scale-estimation being the source of the "
                "0.966 offset, but the full optimum-cli command is often truncated "
                "in these PowerShell transcripts, so missing flags are reported "
                "from the NNCF progress lines rather than from a complete argv.")
        elif se_in_awq:
            log(lines, "    A Scale Estimation apply step WAS present in the AWQ export log. "
                "That would support a genuine extra scaling stage (H2), and the "
                "LS / IR checks above should be weighed against that.")

    if gate_rels:
        rel_s = ", ".join(f"layer {L} rel={h1[L]['gate_rel']:.4f}, median s={h1[L].get('gate_med_p50', float('nan')):.3f}"
                          for L in any_layer if "gate_rel" in h1[L])
        log(lines, f"  H3 gate_proj: {rel_s}.")
        if all(r is not None and 0.05 < r < 0.2 for r in gate_rels) and all(
            p is not None and abs(p - 1.0) < 0.1 for p in gate_p50
        ):
            log(lines, "    gate_proj looks like ordinary INT4 (relative error near 0.106, "
                "row-scale median near 1), not like a second AWQ smooth-scale. "
                "H3 is not supported.")
        elif any(p is not None and abs(p - 1.0) > 0.2 for p in gate_p50):
            log(lines, "    gate_proj row scales are not clustered at 1; H3 may contribute. "
                "See the percentiles above.")
        else:
            log(lines, "    gate_proj numbers are mixed; see the percentiles above rather "
                "than a hard H3 call.")

    log(lines, "  Do not treat a supported H1 as a claim that every channel is an exact "
        "reciprocal: the LS product can still sit a little off 1 because of "
        "ordinary INT4 error. Report the measured LS and simulated median "
        "products, not a rounded slogan.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true",
                        help="run synthetic-tensor checks and exit")
    args = parser.parse_args()
    if args.selftest:
        selftest(verbose=True)
        return

    t0 = time.time()
    DIAG.mkdir(exist_ok=True)
    lines = []
    log(lines, "AWQ MLP pairing offset check (why median s_up * s_down is ~0.966, not 1)")
    log(lines, f"cwd={ROOT}")
    log(lines, f"FP16 glob: {FP16_GLOB}")
    log(lines, f"AWQ IR:    {AWQ_IR}")
    log(lines, "INT4 rule: NNCF 3.4.0 INT4_SYM group 128 RTN, same math as "
        "phase1/verify_phase1.py fake_quant_weight (signed scale, codes [-8, 7], "
        "FP16 group scales). Model files are not modified.")

    try:
        selftest(verbose=False)
        log(lines, "internal selftest: passed")
    except Exception as e:
        log(lines, f"internal selftest FAILED (continuing with the real IRs): {e}")

    keys = [key_for(layer, proj) for layer in LAYERS for proj in ("up_proj", "down_proj", "gate_proj")]
    fp16, awq_w, awq_model, core, ov = {}, {}, None, None, None

    def _load():
        nonlocal fp16, awq_w, awq_model, core, ov
        fp16.update(load_fp16(keys))
        log(lines, f"loaded FP16 keys: {sorted(fp16)}")
        import openvino as ov_mod
        ov = ov_mod
        core = ov.Core()
        xml = AWQ_IR / "openvino_model.xml"
        if not xml.exists():
            raise FileNotFoundError(f"missing {xml}")
        awq_model = core.read_model(str(xml))
        log(lines, f"read AWQ IR {xml}")
        for key in keys:
            op = find_matmul(awq_model, key)
            if op is None:
                log(lines, f"  AWQ: no MatMul for {key}")
                continue
            awq_w[key] = dequant_weight(core, ov, op)
            log(lines, f"  AWQ dequant {key} {awq_w[key].shape}")

    safe_section(lines, "Load FP16 weights and dequantize the AWQ IR", _load)

    h1 = safe_section(
        lines,
        "H1. Median vs least-squares, and an exact-reciprocal INT4 simulation",
        lambda: section_h1_h3(lines, fp16, awq_w),
    ) or {}

    ir_flags = {}
    if awq_model is not None:
        ir_flags = safe_section(
            lines,
            "H2. IR graph around layers.0/1 MLP activation path",
            lambda: section_h2_ir(lines, awq_model, h1),
        ) or {}
    else:
        log(lines, "Skipping H2 IR inspection (AWQ model was not loaded).")

    log_info = safe_section(
        lines,
        "H2. NNCF export logs and run-script flags",
        lambda: section_h2_logs(lines),
    ) or {}

    safe_section(lines, "VERDICT", lambda: section_verdict(lines, h1, ir_flags, log_info))

    elapsed = time.time() - t0
    log(lines, f"elapsed {elapsed:.1f} s")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
