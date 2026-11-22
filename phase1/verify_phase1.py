"""Read-only: (1) dump RTN openvino_config.json + optimum-intel int8 sym default logic,
(2) decompress selected weights from the Phase 1 OV IRs with OpenVINO itself and compare them to the
notebook's PyTorch fake-quant applied to the FP16 source weights."""
import sys, os, re, glob, json, importlib.util
import numpy as np, torch, openvino as ov
ROOT = os.path.abspath(sys.argv[1]); SRC = os.path.join(ROOT, sys.argv[2] if len(sys.argv) > 2 else "models/fp16-src")
def hdr(t): print("\n" + "=" * 90 + "\n## " + t + "\n" + "=" * 90)

hdr("RTN openvino_config.json (full)")
for d in ["ov-rtn-int8", "ov-rtn-int4"]:
    p = os.path.join(ROOT, "models", d, "openvino_config.json")
    if os.path.exists(p): print(d, open(p).read())

hdr("optimum-intel: how --weight-format int8 / sym map to NNCF mode")
try:
    base = os.path.dirname(importlib.util.find_spec("optimum.intel").origin)
    for f in [os.path.join(base, "openvino", "configuration.py"), os.path.join(base, "openvino", "quantization.py")] + glob.glob(os.path.join(base, "..", "commands", "export", "openvino.py")):
        if not os.path.exists(f): continue
        L = open(f, encoding="utf-8", errors="replace").read().splitlines()
        for i, l in enumerate(L):
            if re.search(r"INT8_SYM|INT8_ASYM|int8_sym|\"--sym\"|args\.sym|sym=.*weight_format|def _int8|\.sym is None|sym: bool|if self\.sym|sym = ", l):
                print(f"{os.path.basename(f)}:{i+1}: {l.strip()[:200]}")
except Exception as e: print("failed", e)

# ---- the notebook's quantizer (copied verbatim from cells.py) ----
_EPS32 = torch.finfo(torch.float32).eps
def _grouped(w, group_size):
    if group_size in (-1, None): return w.unsqueeze(1)
    o, i = w.shape; return w.reshape(o, i // group_size, group_size)
def fake_quant_weight(w, mode, group_size=-1, fp16_scale=True):
    wg = _grouped(w.detach().to(torch.float32), group_size)
    if mode in ("int8_asym", "int4_asym"):
        bits = 8 if mode == "int8_asym" else 4; lo, hi = 0, 2 ** bits - 1
        mn = wg.amin(-1, keepdim=True).clamp(max=0.0); mx = wg.amax(-1, keepdim=True).clamp(min=0.0)
        scale = (mx - mn) / (hi - lo); scale = torch.where(scale.abs() < _EPS32, torch.full_like(scale, _EPS32), scale)
        zp = (lo - torch.round(mn / scale)).clamp(lo, hi); q = torch.round(wg / scale + zp).clamp(lo, hi)
    else:
        bits = 8 if mode == "int8_sym" else 4; lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amin = wg.amin(-1, keepdim=True).abs(); mx = wg.amax(-1, keepdim=True)
        scale = torch.where(amin >= mx, amin, -mx) / (2 ** (bits - 1))
        scale = torch.where(scale.abs() < _EPS32, torch.full_like(scale, _EPS32), scale); zp = None
        q = torch.round(wg / scale).clamp(lo, hi)
    if fp16_scale: scale = scale.half().float()
    return ((q - zp) * scale if zp is not None else q * scale).reshape(w.shape)

hdr("source weights: " + SRC)
from safetensors import safe_open
files = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
idx = {}
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys(): idx[k] = f
def src_w(name):
    with safe_open(idx[name], "pt") as h: return h.get_tensor(name)
print("files:", [os.path.basename(f) for f in files], "| dtype:", src_w(next(iter(idx))).dtype, "| tensors:", len(idx))

core = ov.Core()
def components(w, mode, group_size):
    """Same math as fake_quant_weight, but returns integer codes, zero point, and the f16 scale OV would store."""
    wg = _grouped(w.detach().to(torch.float32), group_size)
    if mode.endswith("asym"):
        bits = 8 if mode == "int8_asym" else 4; lo, hi = 0, 2 ** bits - 1
        mn = wg.amin(-1, keepdim=True).clamp(max=0.0); mx = wg.amax(-1, keepdim=True).clamp(min=0.0)
        scale = (mx - mn) / (hi - lo); scale = torch.where(scale.abs() < _EPS32, torch.full_like(scale, _EPS32), scale)
        zp = (lo - torch.round(mn / scale)).clamp(lo, hi); q = torch.round(wg / scale + zp).clamp(lo, hi)
    else:
        bits = 8 if mode == "int8_sym" else 4; lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amin = wg.amin(-1, keepdim=True).abs(); mx = wg.amax(-1, keepdim=True)
        scale = torch.where(amin >= mx, amin, -mx) / (2 ** (bits - 1))
        scale = torch.where(scale.abs() < _EPS32, torch.full_like(scale, _EPS32), scale); zp = None
        q = torch.round(wg / scale).clamp(lo, hi)
    return q.numpy(), (None if zp is None else zp.numpy()), scale.half().numpy()

PASS = {"Convert", "Reshape", "Multiply", "Subtract"}
def walk(node, role=None, acc=None, depth=0):
    acc = [] if acc is None else acc
    t = node.get_type_name()
    if t == "Constant": acc.append((role, node)); return acc
    if t in PASS and depth < 8:
        for k, inp in enumerate(node.inputs()):
            r = "zp" if (t == "Subtract" and k == 1) else ("mul" if (t == "Multiply" and role != "zp") else role)
            walk(inp.get_source_output().get_node(), r, acc, depth + 1)
    return acc
def unpack4(raw, signed, n):
    b = raw.reshape(-1).view(np.uint8)
    v = np.empty(b.size * 2, np.int16); v[0::2] = b & 0x0F; v[1::2] = b >> 4
    if signed: v = np.where(v >= 8, v - 16, v)
    return v[:n]
def ov_components(model, target):
    for op in model.get_ordered_ops():
        t = op.get_type_name()
        if t not in ("MatMul", "Gather"): continue
        if not re.search(re.escape(target) + r"(/|$)", op.get_friendly_name().replace("__module.", "")): continue
        consts = walk(op.input(1 if t == "MatMul" else 0).get_source_output().get_node())
        wc = [n for r, n in consts if r != "zp" and n.get_element_type().get_type_name() in ("u8", "i8", "u4", "i4")][0]
        et = wc.get_element_type().get_type_name(); shape = list(wc.get_output_shape(0))
        raw = wc.get_data()
        q = unpack4(raw, et == "i4", int(np.prod(shape))).reshape(shape) if et in ("u4", "i4") else raw.astype(np.int16)
        zc = [n for r, n in consts if r == "zp"]
        zp = None
        if zc:
            z = zc[0]; zet = z.get_element_type().get_type_name(); zs = list(z.get_output_shape(0))
            zp = unpack4(z.get_data(), zet == "i4", int(np.prod(zs))).reshape(zs) if zet in ("u4", "i4") else z.get_data().astype(np.int16)
        sc = [n for r, n in consts if r == "mul" and n.get_element_type().get_type_name() == "f16"][0].get_data()
        return et, q, zp, sc
    raise KeyError(target)

LAYERS = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.7.mlp.down_proj",
          "model.layers.15.mlp.gate_proj", "model.embed_tokens"]
for mdir, plan in [("ov-rtn-int8", {"*": ("int8_sym", -1)}),
                   ("ov-rtn-int4", {"*": ("int4_sym", 128), "model.embed_tokens": ("int8_asym", -1)})]:
    xml = os.path.join(ROOT, "models", mdir, "openvino_model.xml")
    if not os.path.exists(xml): print("missing", xml); continue
    hdr(f"{mdir}: stored OV integer codes / zero points / f16 scales vs notebook fake-quant of fp16-src")
    model = core.read_model(xml)
    for L in LAYERS + ["__control__"]:
        if L == "__control__":
            L = LAYERS[0]; mode, gs = ("int8_asym", -1) if mdir == "ov-rtn-int8" else ("int4_asym", 128); tag = "CONTROL(wrong mode)"
        else:
            mode, gs = plan.get(L, plan["*"]); tag = ""
        name = L + ".weight"
        if name not in idx: print("  not in source:", name); continue
        w = src_w(name).float()
        try: et, q_ov, zp_ov, sc_ov = ov_components(model, L)
        except Exception as e: print(f"  {L}: failed {e!r}"); continue
        q, zp, sc = components(w, mode, gs)
        q = q.reshape(q_ov.shape) if q.size == q_ov.size else q
        sc = sc.reshape(sc_ov.shape) if sc.size == sc_ov.size else sc
        codes = (q.astype(np.int16) == q_ov).mean() * 100 if q.size == q_ov.size else float("nan")
        scales = (sc == sc_ov).mean() * 100 if sc.size == sc_ov.size else float("nan")
        zps = "n/a" if zp is None and zp_ov is None else (f"{(zp.reshape(zp_ov.shape).astype(np.int16) == zp_ov).mean()*100:.4f}%" if (zp is not None and zp_ov is not None) else f"MISMATCH(ours={zp is not None}, ov={zp_ov is not None})")
        print(f"  {tag}{L:34s} ov={et:3s} ours={mode:9s} gs={gs:4d}  codes equal={codes:9.5f}%  scales(f16) equal={scales:9.5f}%  zero-points equal={zps}")
print("\nVERIFY DONE")
