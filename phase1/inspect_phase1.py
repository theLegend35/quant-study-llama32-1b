"""Read-only inspection of Phase 1 OpenVINO models, build logs, shell history and optimum-intel defaults."""
import sys, os, re, json, glob, importlib, importlib.util, collections
import xml.etree.ElementTree as ET
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
HIST = sys.argv[2] if len(sys.argv) > 2 else ""
PAT = re.compile(r"optimum-cli|compress_weights|weight[-_]format|CompressWeightsMode|OVWeightQuantizationConfig|nncf\.", re.I)
def hdr(t): print("\n" + "=" * 100 + "\n## " + t + "\n" + "=" * 100)

# ---------------------------------------------------------------- versions
hdr("versions")
for mod in ["openvino", "nncf", "optimum.intel", "transformers"]:
    try:
        m = importlib.import_module(mod); print(mod, getattr(m, "__version__", "?"))
    except Exception as e: print(mod, "IMPORT FAILED", e)
try:
    from importlib.metadata import version; print("optimum-intel (dist)", version("optimum-intel"))
except Exception as e: print("optimum-intel dist?", e)

# ---------------------------------------------------------------- logs / scripts / history
hdr("build logs, scripts, notes mentioning optimum-cli / compress_weights")
skip = ("\\.venv", "/.venv", "site-packages", "\\.git", "node_modules")
cands = set(glob.glob(os.path.join(ROOT, "results", "*build*.log")))
for ext in ("*.ps1", "*.txt", "*.log", "*.md", "*.py", "*.bat", "*.sh", "*.json", "*.ipynb"):
    cands.update(glob.glob(os.path.join(ROOT, "**", ext), recursive=True))
cands = sorted(p for p in cands if not any(s in p for s in skip) and os.path.getsize(p) < 20_000_000)
print("files scanned:", len(cands))
for p in cands:
    if p.endswith(".bin"): continue
    try: lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    except Exception: continue
    hits = [(i, l) for i, l in enumerate(lines) if PAT.search(l)]
    if hits:
        print(f"\n--- {os.path.relpath(p, ROOT)} ({len(hits)} hits)")
        for i, l in hits[:40]: print(f"  {i+1}: {l.strip()[:300]}")
hdr("PowerShell history: " + HIST)
if HIST and os.path.exists(HIST):
    for i, l in enumerate(open(HIST, encoding="utf-8", errors="replace").read().splitlines()):
        if PAT.search(l) or re.search(r"rtn|ov-|int8|int4", l, re.I): print(f"  {i+1}: {l.strip()[:400]}")
else: print("history file not found")

# ---------------------------------------------------------------- optimum-intel defaults
hdr("optimum-intel default 4-bit configs")
try:
    spec = importlib.util.find_spec("optimum.intel")
    base = os.path.dirname(spec.origin)
    for f in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
        txt = open(f, encoding="utf-8", errors="replace").read()
        if "_DEFAULT_4BIT" in txt or re.search(r"llama-3\.2-1b", txt, re.I):
            L = txt.splitlines()
            for i, l in enumerate(L):
                if re.search(r"llama-3\.2-1b|_DEFAULT_4BIT_WQ_CONFIG\b *=|def get_default_quantization_config|_DEFAULT_INT8|def _int8|int8_asym|INT8_ASYM|INT8_SYM", l, re.I):
                    print(f"\n{os.path.relpath(f, base)}:{i+1}")
                    for j in range(max(0, i - 2), min(len(L), i + 14)): print("   ", L[j])
    from optimum.intel.openvino import configuration as cfgmod
    for name in dir(cfgmod):
        if "DEFAULT" in name and "4BIT" in name:
            obj = getattr(cfgmod, name)
            if isinstance(obj, dict):
                keys = [k for k in obj if "llama-3.2" in str(k).lower()]
                print(f"{name}: {len(obj)} entries; Llama-3.2 keys: {keys}")
                for k in keys: print("   ", k, "->", obj[k])
            else: print(f"{name} = {obj}")
    if hasattr(cfgmod, "get_default_quantization_config"):
        import inspect
        for mid in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"]:
            try:
                sig = inspect.signature(cfgmod.get_default_quantization_config); print("signature", sig)
                print(mid, "->", cfgmod.get_default_quantization_config(mid, "int4") if len(sig.parameters) > 1 else cfgmod.get_default_quantization_config(mid))
            except Exception as e: print(mid, "call failed:", e)
except Exception as e: print("optimum-intel inspection failed:", repr(e))

# ---------------------------------------------------------------- OV model inspection
import openvino as ov
core = ov.Core()
PASS = {"Convert", "Reshape", "Multiply", "Subtract", "Transpose", "Unsqueeze", "Squeeze"}
def walk(node, depth=0, role=None, acc=None):
    """Collect (role, Constant) feeding a weight input. role: 'zp' = under Subtract input 1, 'mul' = Multiply operand."""
    acc = acc if acc is not None else []
    t = node.get_type_name()
    if t == "Constant":
        acc.append((role, node)); return acc
    if t in PASS and depth < 8:
        for k, inp in enumerate(node.inputs()):
            r = role
            if t == "Subtract" and k == 1: r = "zp"
            elif t == "Multiply" and role != "zp": r = "mul"
            walk(inp.get_source_output().get_node(), depth + 1, r, acc)
    else:
        acc.append((role, node))
    return acc
def short(name):
    m = re.search(r"(model\.layers\.(\d+)\.[\w.]+?|model\.embed_tokens|lm_head)(?=/|$|\.weight)", name)
    return m.group(1) if m else name[:90]
for xml in sorted(glob.glob(os.path.join(ROOT, "models", "**", "openvino_model.xml"), recursive=True)):
    mdir = os.path.dirname(xml)
    hdr("MODEL " + os.path.relpath(mdir, ROOT))
    for jf in ["openvino_config.json"]:
        p = os.path.join(mdir, jf)
        if os.path.exists(p): print(jf, "=", open(p).read()[:3000])
    # model-level rt_info (NNCF writes its weight-compression args here)
    try:
        root = ET.parse(xml).getroot(); rt = root.find("rt_info")
        def dump(e, ind=0):
            v = e.attrib.get("value"); print("  " * ind + e.tag + (f" = {v}" if v is not None else ""))
            for c in e: dump(c, ind + 1)
        if rt is not None:
            for c in rt:
                if c.tag in ("nncf", "optimum", "conversion_parameters", "Runtime_version", "framework") or "nncf" in c.tag.lower(): dump(c, 1)
            print("  rt_info top-level keys:", [c.tag for c in rt])
    except Exception as e: print("rt_info parse failed", e)
    model = core.read_model(xml)
    rows = []
    for op in model.get_ordered_ops():
        t = op.get_type_name()
        if t == "MatMul": w_in = op.input(1).get_source_output().get_node()
        elif t == "Gather": w_in = op.input(0).get_source_output().get_node()
        else: continue
        consts = walk(w_in)
        lowbit = [(c, n) for c, n in consts if n.get_type_name() == "Constant" and n.get_element_type().get_type_name() in ("u8", "i8", "u4", "i4", "u2", "nf4", "f8e4m3")]
        if t == "Gather" and not lowbit and not any(n.get_type_name() == "Constant" and len(n.get_output_shape(0)) == 2 and n.get_output_shape(0)[0] > 1000 for c, n in consts):
            continue  # gathers over indices, not embeddings
        if t == "MatMul" and not any(n.get_type_name() == "Constant" for c, n in consts): continue  # activation x activation
        if not lowbit:
            fc = [n for c, n in consts if n.get_type_name() == "Constant"]
            big = [n for n in fc if len(n.get_output_shape(0)) >= 2]
            if not big: continue
            n = big[0]; rows.append(dict(op=t, layer=short(op.get_friendly_name()), wtype=n.get_element_type().get_type_name(), wshape=list(n.get_output_shape(0)), zp=None, scale=None, wconst=n.get_friendly_name())); continue
        wc = lowbit[0][1]
        zp = [n for c, n in consts if c == "zp" and n.get_type_name() == "Constant"]
        sc = [n for c, n in consts if c == "mul" and n.get_type_name() == "Constant" and n.get_element_type().get_type_name() in ("f16", "f32", "bf16", "f8e8m0")]
        zpi = None
        if zp:
            z = zp[0]; zpi = f"{z.get_element_type().get_type_name()}{list(z.get_output_shape(0))}"
            try:
                import numpy as np; d = z.get_data(); zpi += f" uniq={len(np.unique(d))} min={d.min()} max={d.max()}"
            except Exception: pass
        rows.append(dict(op=t, layer=short(op.get_friendly_name()), wtype=wc.get_element_type().get_type_name(), wshape=list(wc.get_output_shape(0)),
                         zp=zpi, scale=(f"{sc[0].get_element_type().get_type_name()}{list(sc[0].get_output_shape(0))}" if sc else None), wconst=wc.get_friendly_name()))
    # print per-layer table + summary
    for r in rows:
        gs = r["wshape"][-1] if len(r["wshape"]) == 3 else ("per-channel" if r["scale"] else "-")
        print(f"  {r['op']:6s} {r['layer']:42s} {r['wtype']:4s} w{str(r['wshape']):18s} gs={gs!s:11s} zp={r['zp']} scale={r['scale']}")
    summ = collections.Counter()
    for r in rows:
        kind = re.sub(r"model\.layers\.\d+\.", "layers.*.", r["layer"])
        gs = r["wshape"][-1] if len(r["wshape"]) == 3 else -1
        summ[(kind, r["op"], r["wtype"], gs, "asym(zp)" if r["zp"] else "sym/no-zp")] += 1
    print("\n  SUMMARY (layer kind, op, weight dtype, group size(-1=per-channel), zero point) : count")
    for k, v in sorted(summ.items()): print("   ", k, ":", v)
    consts_by_name = collections.Counter(r["wconst"] for r in rows)
    shared = [k for k, v in consts_by_name.items() if v > 1]
    print("  shared weight constants (tied embed/lm_head):", shared[:5])
    dec = [r for r in rows if r["layer"].startswith("model.layers.")]
    print(f"  decoder linears: {len(dec)}; dtype counts: {collections.Counter(r['wtype'] for r in dec)}")
print("\nINSPECTION DONE")
