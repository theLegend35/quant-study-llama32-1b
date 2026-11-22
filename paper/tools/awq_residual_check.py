#!/usr/bin/env python3
"""Root-cause the leftover ~2.29 max-abs residual in layers.1.mlp.up_proj
after per-row median AWQ/FP16 scale division.

Designed for the laptop workspace (same paths as diag/awq_scale_check.py):

    cd C:\\Users\\dhruv\\quant-study
    .\\.venv\\Scripts\\python.exe diag\\awq_residual_check.py

Expects:
  models\\fp16-src\\*.safetensors     (HF FP16 source)
  models\\ov-awq-int4\\openvino_model.xml
  models\\ov-rtn-int4\\openvino_model.xml

Dequantizes MatMul weight subgraphs the same way as weight_err.py /
awq_scale_check.py (constant-fold the weight input; do not modify any
model files). Writes diag\\awq_residual_check.txt and prints the same
report. Optional activation probe is skipped if no cheap dump is present.

Run a box-side logic check without models:

    python3 awq_residual_check.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
DIAG = ROOT / "diag"
AWQ_IR = ROOT / "models" / "ov-awq-int4"
RTN_IR = ROOT / "models" / "ov-rtn-int4"
FP16_GLOB = str(ROOT / "models" / "fp16-src" / "*.safetensors")
OUT = DIAG / "awq_residual_check.txt"

# up_proj is (intermediate, hidden) = (8192, 2048); down_proj is (hidden, intermediate).
# Pairing: up_proj output row r <-> down_proj input column r.
LAYERS = (0, 1)
GROUP = 128
EPS = 1e-6
# INT4_SYM codes are typically in [-8, 7].
CLIP_LO, CLIP_HI = -8, 7


def log(lines, msg=""):
    lines.append(msg)
    print(msg, flush=True)


def safe_section(lines, title, fn):
    log(lines, "")
    log(lines, "=" * 72)
    log(lines, title)
    log(lines, "=" * 72)
    try:
        fn()
    except Exception as e:
        log(lines, f"FAILED: {type(e).__name__}: {e}")
        for row in traceback.format_exc().splitlines():
            log(lines, "  " + row)


def align(w, r):
    w = np.asarray(w, np.float32)
    r = np.asarray(r, np.float32)
    if w.shape != r.shape:
        if w.T.shape == r.shape:
            w = np.ascontiguousarray(w.T)
        else:
            raise ValueError(f"shape mismatch {w.shape} vs {r.shape}")
    return w


def median_ratio(num, den, axis, eps=EPS):
    """Median of num/den along `axis`, ignoring |den| <= eps. Shape of the other axis."""
    num = np.asarray(num, np.float32)
    den = np.asarray(den, np.float32)
    if axis == 1:
        out = np.empty(num.shape[0], dtype=np.float32)
        for i in range(num.shape[0]):
            mask = np.abs(den[i, :]) > eps
            if not mask.any():
                out[i] = np.nan
                continue
            s = float(np.median(num[i, mask] / den[i, mask]))
            out[i] = s if np.isfinite(s) and s != 0 else np.nan
        return out
    if axis == 0:
        out = np.empty(num.shape[1], dtype=np.float32)
        for j in range(num.shape[1]):
            mask = np.abs(den[:, j]) > eps
            if not mask.any():
                out[j] = np.nan
                continue
            s = float(np.median(num[mask, j] / den[mask, j]))
            out[j] = s if np.isfinite(s) and s != 0 else np.nan
        return out
    raise ValueError(axis)


def apply_row_scales(w, r, s_row):
    """residual = W / s_row[:, None] - W_fp16, with s=1 where the median is unusable."""
    s = np.array(s_row, dtype=np.float32, copy=True)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return w / s[:, None] - r, s


def percentiles(x, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}


def fmt_pct(d):
    return "  ".join(f"p{p}={d[p]:.4g}" for p in d)


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


def _chain(node, depth=10):
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
    """Best-effort Constant payload. Returns (array or None, note)."""
    cur = node
    # unwrap Convert
    for _ in range(4):
        if cur.get_type_name() == "Convert" and cur.get_input_size() >= 1:
            cur = cur.input_value(0).get_node()
            continue
        break
    try:
        if cur.get_type_name() != "Constant":
            return None, f"not a Constant ({cur.get_type_name()}:{cur.get_friendly_name()})"
        data = cur.get_data()
        return np.array(data), f"{cur.get_friendly_name()} dtype={data.dtype} shape={tuple(np.array(data).shape)}"
    except Exception as e:
        return None, f"get_data failed on {cur.get_type_name()}:{cur.get_friendly_name()}: {e}"


def find_matmul(model, key):
    """key like model.layers.1.mlp.up_proj.weight"""
    import re

    parts = key.replace(".weight", "").split(".")
    # model, layers, N, mlp, up_proj
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


def inspect_weight_ir(core, ov, model, key, lines):
    """Walk the MatMul weight subgraph; return dequant, scale array, codes, notes."""
    import openvino as ov_mod  # noqa: F401 — kept for callers that pass ov

    info = {
        "dequant": None,
        "scale": None,
        "scale_note": None,
        "codes": None,
        "codes_note": None,
        "zp": None,
        "chain": None,
        "multiply_name": None,
    }
    op = find_matmul(model, key)
    if op is None:
        info["scale_note"] = f"no MatMul found for {key}"
        log(lines, f"  {info['scale_note']}")
        return info
    info["chain"] = _chain(op.input_value(1).get_node())
    log(lines, f"  IR weight chain {key}: {info['chain']}")
    try:
        info["dequant"] = dequant_weight(core, ov, op)
        log(lines, f"  dequant shape={info['dequant'].shape} dtype={info['dequant'].dtype}")
    except Exception as e:
        log(lines, f"  dequant failed: {e}")

    # Find the first Multiply walking up input 0 (same as weight_err.py).
    n = op.input_value(1).get_node()
    mul = None
    for _ in range(12):
        if n.get_type_name() == "Multiply":
            mul = n
            break
        if n.get_input_size() == 0:
            break
        n = n.input_value(0).get_node()
    if mul is None:
        info["scale_note"] = "no Multiply in weight chain"
        log(lines, f"  {info['scale_note']}")
        return info
    info["multiply_name"] = mul.get_friendly_name()
    log(lines, f"  Multiply: {info['multiply_name']}  n_inputs={mul.get_input_size()}")

    # Scale is conventionally input 1; also inspect input 0 as codes.
    try:
        scale_node = mul.input_value(1).get_node()
        scale, note = _const_data(scale_node)
        info["scale"] = scale
        info["scale_note"] = note
        if scale is not None:
            log(lines, f"  group/channel scale: {note}  min={scale.min():.6g} max={scale.max():.6g}")
        else:
            log(lines, f"  group/channel scale: {note}")
    except Exception as e:
        info["scale_note"] = f"scale extract failed: {e}"
        log(lines, f"  {info['scale_note']}")

    try:
        code_node = mul.input_value(0).get_node()
        codes, note = _const_data(code_node)
        info["codes"] = codes
        info["codes_note"] = note
        if codes is not None:
            log(lines, f"  stored codes: {note}  min={codes.min()} max={codes.max()}")
        else:
            log(lines, f"  stored codes: {note}")
    except Exception as e:
        info["codes_note"] = f"code extract failed: {e}"
        log(lines, f"  {info['codes_note']}")

    # Optional zero point: look for a Subtract/Add next to the Multiply.
    try:
        for inp_i in range(mul.get_input_size()):
            # already handled 0/1
            pass
        # Walk a few neighbours by name
        parent = mul
        for _ in range(3):
            if parent.get_input_size() == 0:
                break
            parent = parent.input_value(0).get_node()
            if parent.get_type_name() in ("Subtract", "Add"):
                zp, note = _const_data(parent.input_value(1).get_node()) if parent.get_input_size() > 1 else (None, "no zp input")
                info["zp"] = zp
                log(lines, f"  zero-point candidate via {parent.get_type_name()}: {note}")
                break
    except Exception as e:
        log(lines, f"  zero-point walk failed (continuing): {e}")
    return info


def broadcast_scale(scale, shape):
    """Broadcast an NNCF per-group scale to a full weight shape (out, in)."""
    s = np.asarray(scale, np.float32)
    if s.shape == shape:
        return s
    # Common layouts: (out, n_groups), (out, 1, n_groups), (1, out, n_groups), (n_groups, out), (out,)
    out, inn = shape
    n_g = inn // GROUP if inn % GROUP == 0 else None
    candidates = []
    if s.ndim == 2 and n_g and s.shape == (out, n_g):
        candidates.append(np.repeat(s, GROUP, axis=1))
    if s.ndim == 2 and n_g and s.shape == (n_g, out):
        candidates.append(np.repeat(s.T, GROUP, axis=1))
    if s.ndim == 3 and n_g:
        s2 = s.reshape(s.shape[0], -1) if s.shape[-1] == n_g else s.reshape(-1, n_g)
        if s2.shape == (out, n_g):
            candidates.append(np.repeat(s2, GROUP, axis=1))
    if s.ndim == 1 and s.size == out:
        candidates.append(np.repeat(s[:, None], inn, axis=1))
    if s.ndim == 1 and n_g and s.size == n_g:
        candidates.append(np.tile(np.repeat(s, GROUP), (out, 1)))
    if s.ndim == 1 and s.size == out * (n_g or 0):
        candidates.append(np.repeat(s.reshape(out, n_g), GROUP, axis=1))
    if not candidates:
        raise ValueError(f"cannot broadcast scale shape {s.shape} to weight {shape} (group={GROUP})")
    return candidates[0]


def unpack_codes(codes, shape):
    """Try to present integer codes as an array broadcastable to `shape`."""
    c = np.asarray(codes)
    if c.shape == shape:
        return c
    out, inn = shape
    # packed u4/i4 as uint8 with two nibbles per byte along the input dim
    if c.dtype == np.uint8 and c.ndim == 2 and c.shape[0] == out and c.shape[1] * 2 == inn:
        lo = (c & 0x0F).astype(np.int16)
        hi = ((c >> 4) & 0x0F).astype(np.int16)
        # map unsigned nibble to signed i4 if values look like 0..15
        def to_i4(x):
            return np.where(x >= 8, x - 16, x)

        packed = np.empty((out, inn), dtype=np.int16)
        packed[:, 0::2] = to_i4(lo)
        packed[:, 1::2] = to_i4(hi)
        return packed
    if c.size == out * inn:
        return c.reshape(shape)
    raise ValueError(f"cannot reshape codes {c.shape} {c.dtype} to {shape}")


def group_index(col, group=GROUP):
    return int(col) // group, (int(col) // group) * group, (int(col) // group) * group + group


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_residual(lines, fp16, awq):
    """(1) (row, col) of the largest residual after per-row median division.
    (2) per-element ratio distribution on that row."""
    results = {}
    for layer in LAYERS:
        key = f"model.layers.{layer}.mlp.up_proj.weight"
        if key not in fp16 or key not in awq:
            log(lines, f"SKIP {key}: missing fp16={key in fp16} awq={key in awq}")
            continue
        r = fp16[key]
        w = align(awq[key], r)
        s_row = median_ratio(w, r, axis=1)
        residual, s_used = apply_row_scales(w, r, s_row)
        rel = float(np.linalg.norm(residual) / np.linalg.norm(r))
        flat = int(np.abs(residual).argmax())
        i, j = divmod(flat, residual.shape[1])
        fp, aw = float(r[i, j]), float(w[i, j])
        s_med = float(s_row[i])
        elem_ratio = aw / fp if abs(fp) > EPS else float("nan")
        res = float(residual[i, j])
        results[layer] = dict(i=i, j=j, fp=fp, aw=aw, s_med=s_med,
                              elem_ratio=elem_ratio, residual=res,
                              residual_abs=abs(res), rel=rel,
                              s_row=s_row, residual_mat=residual, w=w, r=r)

        log(lines, f"{key}: shape {tuple(w.shape)}")
        log(lines, f"  after per-row median division: maxabs={float(np.abs(residual).max()):.6f}  rel={rel:.4f}")
        log(lines, f"  largest residual at row {i}, col {j}:")
        log(lines, f"    W_fp16          = {fp:.6f}")
        log(lines, f"    W_awq           = {aw:.6f}")
        log(lines, f"    row median s    = {s_med:.6g}")
        log(lines, f"    element ratio   = {elem_ratio:.6g}   (W_awq/W_fp16)")
        log(lines, f"    W_awq/s - W_fp16 = {res:.6f}")
        if np.isfinite(s_med) and s_med != 0:
            log(lines, f"    residual * s    = {res * s_med:.6f}   (error in the scaled/stored-weight domain, if s is right)")

        # (2) distribution of per-element ratios on that row
        mask = np.abs(r[i, :]) > EPS
        ratios = w[i, mask] / r[i, mask]
        log(lines, f"  row {i}: {int(mask.sum())}/{r.shape[1]} elements with |W_fp16| > {EPS:g}")
        log(lines, f"  row {i} per-element ratios W_awq/W_fp16: {fmt_pct(percentiles(ratios))}")
        # how many are close to the median vs outliers
        if np.isfinite(s_med) and s_med != 0:
            close = np.abs(ratios - s_med) / max(abs(s_med), 1e-12)
            log(lines, f"  fraction of those ratios within 5%/10%/25% of the row median: "
                f"{float((close < 0.05).mean()):.3f} / {float((close < 0.10).mean()):.3f} / {float((close < 0.25).mean()):.3f}")
            # extremes
            order = np.argsort(np.abs(residual[i, :]))[::-1]
            log(lines, f"  row {i} top-5 |residual| entries (col, W_fp16, W_awq, ratio, residual):")
            for col in order[:5]:
                if abs(r[i, col]) > EPS:
                    rat = w[i, col] / r[i, col]
                else:
                    rat = float("nan")
                log(lines, f"    col {int(col):5d}  fp16={r[i, col]:.6f}  awq={w[i, col]:.6f}  "
                    f"ratio={rat:.6g}  resid={residual[i, col]:.6f}")
        # sanity: is the leftover residual in the same row as the raw max-abs (row 1462 on layer 1)?
        raw_flat = int(np.abs(w - r).argmax())
        ri, rj = divmod(raw_flat, w.shape[1])
        log(lines, f"  raw |W_awq-W_fp16| argmax is at row {ri}, col {rj} "
            f"(same row as residual argmax: {ri == i})")
    return results


def section_pairing(lines, fp16, awq, residual_info):
    """(3) up row scale vs 1/down column scale; product, correlation, residual row."""
    for layer in LAYERS:
        up_k = f"model.layers.{layer}.mlp.up_proj.weight"
        dn_k = f"model.layers.{layer}.mlp.down_proj.weight"
        gate_k = f"model.layers.{layer}.mlp.gate_proj.weight"
        missing = [k for k in (up_k, dn_k) if k not in fp16 or k not in awq]
        if missing:
            log(lines, f"layer {layer}: SKIP pairing, missing {missing}")
            continue
        up_r, up_w = fp16[up_k], align(awq[up_k], fp16[up_k])
        dn_r, dn_w = fp16[dn_k], align(awq[dn_k], fp16[dn_k])
        s_up = median_ratio(up_w, up_r, axis=1)          # per output row
        s_dn = median_ratio(dn_w, dn_r, axis=0)          # per input column
        # down input dim must match up output dim
        if s_up.size != s_dn.size:
            log(lines, f"layer {layer}: pairing size mismatch up_rows={s_up.size} down_cols={s_dn.size}")
            continue
        inv_dn = np.where((np.abs(s_dn) > EPS) & np.isfinite(s_dn), 1.0 / s_dn, np.nan)
        product = s_up * s_dn  # should be ~1 if they are reciprocal
        ok = np.isfinite(product)
        log(lines, f"layer {layer} MLP pairing (up row r <-> down col r); n={int(ok.sum())}/{product.size} finite products")
        if ok.any():
            log(lines, f"  product s_up * s_down percentiles: {fmt_pct(percentiles(product[ok]))}")
            # correlation between s_up and 1/s_down
            a, b = s_up[ok], inv_dn[ok]
            if a.size >= 3 and np.std(a) > 0 and np.std(b) > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
                log(lines, f"  corr(s_up, 1/s_down) = {corr:.4f}")
            log(lines, f"  median |product - 1| = {float(np.median(np.abs(product[ok] - 1))):.4g};  "
                f"fraction within 5% of 1: {float((np.abs(product[ok] - 1) < 0.05).mean()):.3f}")
        if layer in residual_info:
            i = residual_info[layer]["i"]
            su, sd, pr = float(s_up[i]), float(s_dn[i]), float(product[i])
            log(lines, f"  residual row {i}: s_up={su:.6g}  s_down={sd:.6g}  1/s_down={1/sd if sd else float('nan'):.6g}  "
                f"product={pr:.6g}")
        # gate_proj should be essentially unscaled vs FP16 relative to up/down
        if gate_k in fp16 and gate_k in awq:
            g_r, g_w = fp16[gate_k], align(awq[gate_k], fp16[gate_k])
            e = g_w - g_r
            log(lines, f"  gate_proj vs FP16: maxabs={float(np.abs(e).max()):.4f}  "
                f"rel={float(np.linalg.norm(e)/np.linalg.norm(g_r)):.4f}  "
                "(expected to look like ordinary INT4, not a large scale transform)")


def section_group(lines, residual_info, awq_ir, rtn_deq, fp16, ov_info_up):
    """(4) INT4 group of the residual element, IR scale, code, clip, RTN, a/b/c."""
    for layer in LAYERS:
        if layer not in residual_info:
            continue
        info = residual_info[layer]
        i, j = info["i"], info["j"]
        key = f"model.layers.{layer}.mlp.up_proj.weight"
        g, g0, g1 = group_index(j)
        log(lines, f"{key} residual element row {i}, col {j}:")
        log(lines, f"  input-dim group size {GROUP}: group {g} covers columns {g0}..{g1 - 1}")

        scale_full = None
        ir = ov_info_up.get(layer, {})
        if ir.get("scale") is not None:
            try:
                scale_full = broadcast_scale(ir["scale"], info["w"].shape)
                log(lines, f"  IR scale broadcast to {scale_full.shape}; "
                    f"this element scale={float(scale_full[i, j]):.6g}; "
                    f"group {g} unique values in this row: "
                    f"{sorted(set(np.round(scale_full[i, g0:g1], 8).tolist()))}")
            except Exception as e:
                log(lines, f"  could not broadcast IR scale ({ir.get('scale_note')}): {e}")
                scale_full = None
        else:
            log(lines, f"  no IR scale array ({ir.get('scale_note')})")

        if ir.get("zp") is not None:
            log(lines, f"  zero point present, shape={np.asarray(ir['zp']).shape}  "
                f"min={np.asarray(ir['zp']).min()} max={np.asarray(ir['zp']).max()}")
        else:
            log(lines, "  no zero point found in the weight Multiply chain (consistent with INT4_SYM)")

        # Stored int code: prefer IR payload, else infer from dequant / scale.
        code = None
        code_src = None
        if ir.get("codes") is not None:
            try:
                codes = unpack_codes(ir["codes"], info["w"].shape)
                code = int(codes[i, j])
                code_src = "IR Constant"
                log(lines, f"  stored code from IR: {code}  (codes dtype range {codes.min()}..{codes.max()})")
            except Exception as e:
                log(lines, f"  could not unpack IR codes ({ir.get('codes_note')}): {e}")
        if code is None and scale_full is not None and abs(float(scale_full[i, j])) > 0:
            raw = float(info["w"][i, j] / scale_full[i, j])
            code = int(np.round(raw))
            code_src = f"inferred as round(W_awq/scale) = round({raw:.4f})"
            log(lines, f"  inferred code: {code}  ({code_src})")

        if code is not None:
            at_clip = (code <= CLIP_LO) or (code >= CLIP_HI)
            log(lines, f"  clip window [{CLIP_LO}, {CLIP_HI}]: hits a clip limit? {at_clip}")

        # Group max-abs in the scaled FP16 domain vs the dequant
        s_row = info["s_med"]
        # Prefer the recovered AWQ row scale for the "fp16 * s" comparison, falling
        # back to the element's own ratio if the median is unusable.
        s_for_scaled = s_row if (np.isfinite(s_row) and s_row != 0) else info["elem_ratio"]
        if np.isfinite(s_for_scaled) and s_for_scaled != 0:
            scaled_fp = info["r"][i, g0:g1] * s_for_scaled
            deq_g = info["w"][i, g0:g1]
            log(lines, f"  group {g} on row {i}, using s={s_for_scaled:.6g}:")
            log(lines, f"    max |W_fp16 * s| = {float(np.abs(scaled_fp).max()):.6f}")
            log(lines, f"    max |W_awq|      = {float(np.abs(deq_g).max()):.6f}")

        # RTN at the same element
        rtn_key_w = None
        if rtn_deq is not None and key in rtn_deq:
            rtn_w = align(rtn_deq[key], info["r"])
            rtn_err = float(rtn_w[i, j] - info["r"][i, j])
            log(lines, f"  same element under RTN INT4: W_rtn={float(rtn_w[i, j]):.6f}  "
                f"W_rtn-W_fp16={rtn_err:.6f}  |err|={abs(rtn_err):.6f}")
            rtn_key_w = rtn_w
        else:
            log(lines, "  RTN dequant not available for this element")

        # (a)/(b)/(c)
        log(lines, "  reading against (a) normal INT4 rounding/clip of a large scaled value,")
        log(lines, "                 (b) per-row median is the wrong inverse for this element,")
        log(lines, "                 (c) something abnormal (sign flip, bad code, bad group scale):")
        s_med, elem_r, res = info["s_med"], info["elem_ratio"], info["residual"]
        ratio_disagreement = (
            np.isfinite(s_med) and np.isfinite(elem_r) and abs(s_med) > 0
            and abs(elem_r - s_med) / max(abs(s_med), 1e-12) > 0.25
        )
        log(lines, f"    row-median s={s_med:.6g}, element ratio={elem_r:.6g}, "
            f"relative disagreement={abs(elem_r - s_med)/max(abs(s_med), 1e-12) if np.isfinite(s_med) and np.isfinite(elem_r) else float('nan'):.3f}")
        if scale_full is not None and np.isfinite(s_med) and s_med != 0:
            qerr = res * s_med  # stored-domain error IF the median scale is correct
            step = float(abs(scale_full[i, j]))
            log(lines, f"    residual * s_median = {qerr:.6f}   IR group step (scale) = {step:.6g}   "
                f"|qerr|/step = {abs(qerr)/step if step else float('nan'):.3f}")
            # If the element's own ratio were the true scale, the residual after
            # that scale would be ~0 by construction for this one entry, but we
            # can still compare W_awq to quantize(W_fp16 * s_med).
            if step > 0 and abs(qerr) <= 0.6 * step:
                log(lines, "    |qerr| is within about half a step: consistent with (a) ordinary rounding, "
                    "provided the row median is the right scale.")
            elif step > 0:
                log(lines, "    |qerr| is larger than a rounding bin. That can still be (a) if the element "
                    "clipped, or (b) if s_median is the wrong inverse so residual*s is not the real qerr.")
        if ratio_disagreement:
            log(lines, "    the element's own ratio is well off the row median: this favours (b) "
                "(the leftover max-abs is from applying the wrong inverse to this entry), "
                "not a huge quantization error on a correctly-unscaled weight.")
        if np.isfinite(elem_r) and np.isfinite(s_med) and elem_r * s_med < 0:
            log(lines, "    sign of the element ratio disagrees with the row median: that would be (c).")
        if code is not None and not (CLIP_LO - 1 <= code <= CLIP_HI + 1):
            log(lines, f"    code {code} is outside the INT4 range: that would be (c).")
        if rtn_key_w is not None:
            log(lines, f"    RTN error at this element is {abs(float(rtn_key_w[i, j] - info['r'][i, j])):.6f}, "
                "which is the baseline INT4 error without an AWQ scale transform.")


def section_activation(lines):
    """(5) optional: residual times a typical activation on that input column."""
    # Cheap dumps from the earlier diagnosis are for down_proj *intermediate*
    # activations (act_check.json), not up_proj's hidden-state inputs. A forward
    # pass would need the full model and calibration text; skip rather than
    # invent a magnitude.
    candidates = [
        DIAG / "act_check.json",
        DIAG / "pertoken.npz",
        DIAG / "pertoken_full.npz",
    ]
    present = [str(p) for p in candidates if p.exists()]
    log(lines, "No cheap calibration dump of up_proj input activations (hidden states, "
        "dim 2048) is used here. Existing diagnosis dumps record down_proj / residual-stream "
        "stats, which are the wrong axis for this element.")
    if present:
        log(lines, f"  dumps present but not applied: {present}")
    else:
        log(lines, "  no act_check / pertoken dumps in diag/ either.")
    log(lines, "  Skipped rather than run a full-model forward pass.")


def load_ir_weights(ir_dir, keys, lines, label):
    import openvino as ov

    xml = ir_dir / "openvino_model.xml"
    if not xml.exists():
        raise FileNotFoundError(f"missing IR {xml}")
    core = ov.Core()
    model = core.read_model(str(xml))
    log(lines, f"{label} IR: {xml}")
    weights = {}
    details = {}
    for key in keys:
        try:
            op = find_matmul(model, key)
            if op is None:
                log(lines, f"  no MatMul for {key}")
                continue
            weights[key] = dequant_weight(core, ov, op)
        except Exception as e:
            log(lines, f"  dequant {key} failed: {e}")
    return core, ov, model, weights, details


# ---------------------------------------------------------------------------
# Self-test (synthetic tensors; no models required)
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

    rng = np.random.default_rng(0)
    # up: (8, 16) with group 4 for a miniature case, but group_index uses GROUP=128
    # so use realistic shapes reduced: 4 rows, 256 cols (2 groups).
    out, inn = 4, 256
    fp = rng.normal(0, 0.05, size=(out, inn)).astype(np.float32)
    fp[0, 0] = 0.0  # a near-zero to test the mask
    s_true = np.array([1.0, 2.0, 32.0, 0.5], dtype=np.float32)
    aw = fp * s_true[:, None]
    # Plant a leftover residual: row 2 (s=32) is clean after undo, but row 1 has
    # one element with a very different ratio, so the per-row median (≈2) is the
    # wrong inverse for that entry and leaves a large residual.
    aw[1, 10] = fp[1, 10] * 40.0  # element ratio 40 vs row median 2

    s_hat = median_ratio(aw, fp, axis=1)
    resid, _ = apply_row_scales(aw, fp, s_hat)
    i, j = divmod(int(np.abs(resid).argmax()), inn)
    check("planted residual is at row 1, col 10", i == 1 and j == 10, f"got row {i} col {j}")
    check("row-1 median stays near 2", abs(float(s_hat[1]) - 2.0) < 0.05, f"s={s_hat[1]}")
    check("element ratio at plant is 40", abs(aw[1, 10] / fp[1, 10] - 40) < 1e-4, f"{aw[1,10]/fp[1,10]}")
    check("row-2 (scale 32) residual is tiny", float(np.abs(resid[2]).max()) < 1e-5, f"{np.abs(resid[2]).max()}")

    # Pairing: down columns scaled by 1/s_true (down is (inn_down=out, ...) wait)
    # down shape (hidden, intermediate) = (inn_up, out_up) if we pair up rows with down cols.
    # Here up rows=4, so down has 4 input columns. Use down (8, 4).
    fp_d = rng.normal(0, 0.05, size=(8, out)).astype(np.float32)
    aw_d = fp_d * (1.0 / s_true)[None, :]
    s_up = median_ratio(aw, fp, axis=1)
    s_dn = median_ratio(aw_d, fp_d, axis=0)
    prod = s_up * s_dn
    check("reciprocal product ~ 1", np.allclose(prod, 1.0, rtol=1e-4, atol=1e-4), f"product={prod}")

    # Group index for col 658 with group 128
    g, g0, g1 = group_index(658, 128)
    check("group of col 658", (g, g0, g1) == (5, 640, 768), f"{g, g0, g1}")

    # broadcast scale (out, n_groups) with GROUP 128
    sc = np.arange(out * (inn // 128), dtype=np.float32).reshape(out, inn // 128) + 1
    br = broadcast_scale(sc, (out, inn))
    check("broadcast scale shape", br.shape == (out, inn), str(br.shape))
    check("broadcast repeats group", np.allclose(br[:, 0], br[:, 127]) and not np.allclose(br[:, 0], br[:, 128]))

    # packing unpack of u4 nibbles
    raw = np.array([[0x1F, 0x07]], dtype=np.uint8)  # lo=15, hi=1  then lo=7, hi=0 -> i4: -1, 1, 7, 0
    unpacked = unpack_codes(raw, (1, 4))
    check("u4 unpack", np.array_equal(unpacked, np.array([[-1, 1, 7, 0]], dtype=np.int16)), str(unpacked))

    # residual*s vs step: if median is correct, qerr = residual*s should be small
    # For row 2, residual ~ 0, qerr ~ 0. For planted row 1, qerr is large vs a tiny step.
    qerr = resid[1, 10] * float(s_hat[1])
    check("planted qerr is large vs a 0.05 step (favours b, not a)",
          abs(qerr) > 0.6 * 0.05, f"qerr={qerr}")

    # a vs b disagreement flag
    elem_ratio = aw[1, 10] / fp[1, 10]
    rel_dis = abs(elem_ratio - s_hat[1]) / abs(s_hat[1])
    check("ratio disagreement > 25% at the plant", rel_dis > 0.25, f"{rel_dis}")

    if failures:
        raise AssertionError("selftest failed: " + "; ".join(failures))
    if verbose:
        print("selftest: all checks passed", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run synthetic-tensor checks and exit")
    args = parser.parse_args()
    if args.selftest:
        selftest(verbose=True)
        return

    DIAG.mkdir(exist_ok=True)
    lines = []
    log(lines, "AWQ leftover-residual check (layers.0/1 mlp.up_proj after per-row median undo)")
    log(lines, f"cwd={ROOT}")
    log(lines, f"FP16 glob: {FP16_GLOB}")
    log(lines, f"AWQ IR:    {AWQ_IR}")
    log(lines, f"RTN IR:    {RTN_IR}")

    # Always run the synthetic checks first; they are cheap and fail loudly.
    try:
        selftest(verbose=False)
        log(lines, "internal selftest: passed")
    except Exception as e:
        log(lines, f"internal selftest FAILED (continuing with the real IRs): {e}")

    keys = []
    for layer in LAYERS:
        for proj in ("up_proj", "down_proj", "gate_proj"):
            keys.append(f"model.layers.{layer}.mlp.{proj}.weight")

    residual_info = {}
    awq_w = {}
    fp16 = {}
    rtn_w = None
    ov_info_up = {}

    def _load():
        nonlocal fp16, awq_w, rtn_w
        fp16.update(load_fp16(keys))
        log(lines, f"loaded FP16 keys: {sorted(fp16)}")
        import openvino as ov

        core = ov.Core()
        awq_xml = AWQ_IR / "openvino_model.xml"
        rtn_xml = RTN_IR / "openvino_model.xml"
        if not awq_xml.exists():
            raise FileNotFoundError(f"missing {awq_xml}")
        awq_model = core.read_model(str(awq_xml))
        log(lines, f"read AWQ IR {awq_xml}")
        for key in keys:
            try:
                op = find_matmul(awq_model, key)
                if op is None:
                    log(lines, f"  AWQ: no MatMul for {key}")
                    continue
                awq_w[key] = dequant_weight(core, ov, op)
                log(lines, f"  AWQ dequant {key} {awq_w[key].shape}")
            except Exception as e:
                log(lines, f"  AWQ dequant {key} failed: {e}")
        # IR details for the two up_proj matrices (codes / scales)
        for layer in LAYERS:
            key = f"model.layers.{layer}.mlp.up_proj.weight"
            log(lines, f"inspecting AWQ weight subgraph for {key}")
            ov_info_up[layer] = inspect_weight_ir(core, ov, awq_model, key, lines)

        if rtn_xml.exists():
            rtn_model = core.read_model(str(rtn_xml))
            log(lines, f"read RTN IR {rtn_xml}")
            rtn_w = {}
            for layer in LAYERS:
                key = f"model.layers.{layer}.mlp.up_proj.weight"
                try:
                    op = find_matmul(rtn_model, key)
                    rtn_w[key] = dequant_weight(core, ov, op)
                    log(lines, f"  RTN dequant {key} {rtn_w[key].shape}")
                except Exception as e:
                    log(lines, f"  RTN dequant {key} failed: {e}")
        else:
            log(lines, f"RTN IR not found at {rtn_xml} (section 4 will skip the RTN comparison)")

    safe_section(lines, "Load FP16 weights and dequantize AWQ / RTN IRs", _load)

    def _s1():
        nonlocal residual_info
        residual_info = section_residual(lines, fp16, awq_w)

    safe_section(lines, "1–2. Largest per-row residual and ratio distribution on that row", _s1)
    safe_section(lines, "3. up_proj row scale vs 1/down_proj column scale (MLP pairing)",
                 lambda: section_pairing(lines, fp16, awq_w, residual_info))
    safe_section(lines, "4. INT4 group, IR scale/code, RTN, and (a)/(b)/(c)",
                 lambda: section_group(lines, residual_info, awq_w, rtn_w, fp16, ov_info_up))
    safe_section(lines, "5. Contribution to the layer output (optional activation probe)",
                 lambda: section_activation(lines))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
