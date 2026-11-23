#!/usr/bin/env python3
"""Paired bootstrap + exact McNemar for Phase 1 (laptop, OpenVINO).

    python3 p1_paired.py /path/to/p1_samples.zip [out.json] [out.csv]

Input: a zip (or a directory) of lm-eval --log_samples reruns, with folders
  <model>-full  (arc_easy, arc_challenge, hellaswag, winogrande)
  <model>-qa    (piqa, boolq)
for model in fp16, rtn-int8, rtn-int4, gptq-int4, awq-int4.

The per-question metric matches the paper: acc_norm for ARC-Challenge,
ARC-Easy, HellaSwag and PIQA; acc for WinoGrande and BoolQ. Aggregates
are checked against the paper's Phase 1 tables (and against the rerun
results JSONs in the same zip). If a displayed four-decimal accuracy
differs, the script exits before testing.

Method matches lora/stats/paired_bootstrap.py (Phase 2 appendix):
  stratified paired bootstrap, B=10,000, rng seed 20261119; each resample
  draws questions with replacement within each task; the same indices are
  used for both models and for every comparison. The main statistic is the
  six-task average of the per-task means, in percentage points (first minus
  second). Two-sided p = 2*min(P(d*<=0), P(d*>=0)) with +1 smoothing
  (smallest p = 2/10,001 ≈ 0.0002). 95% CI = percentile interval.
  Exact two-sided McNemar (binomial on discordant pairs), per task and
  pooled over all questions (pooled is question-weighted, so HellaSwag
  dominates). Holm step-down over the seven Phase 1 comparisons listed
  below (all seven are in the family; there is no secondary comparison).

Comparisons (Holm family of 7):
  RTN INT8 vs FP16
  RTN INT4 vs FP16
  GPTQ INT4 vs FP16
  AWQ INT4 vs FP16
  AWQ INT4 vs RTN INT4
  GPTQ INT4 vs RTN INT4
  AWQ INT4 vs GPTQ INT4

Does not write the 46 MB sample zip; optionally writes a compact
per-question correctness CSV.
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

TASKS = [
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "winogrande",
    "boolq",
    "piqa",
]
EXPECT_METRIC = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
}
EXPECT_N = {
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "hellaswag": 10042,
    "winogrande": 1267,
    "boolq": 3270,
    "piqa": 1838,
}
# Displayed values in paper Tables 5/6 and the Phase 1 appendix SE tables.
PAPER_DISPLAY = {
    "fp16": dict(
        arc_challenge="0.3677", arc_easy="0.6195", hellaswag="0.6433",
        winogrande="0.6172", boolq="0.6391", piqa="0.7497",
    ),
    "rtn-int8": dict(
        arc_challenge="0.3635", arc_easy="0.6195", hellaswag="0.6431",
        winogrande="0.6093", boolq="0.6306", piqa="0.7454",
    ),
    "rtn-int4": dict(
        arc_challenge="0.3242", arc_easy="0.5745", hellaswag="0.6000",
        winogrande="0.5951", boolq="0.5596", piqa="0.7318",
    ),
    "gptq-int4": dict(
        arc_challenge="0.3336", arc_easy="0.5661", hellaswag="0.5823",
        winogrande="0.5919", boolq="0.5450", piqa="0.7171",
    ),
    "awq-int4": dict(
        arc_challenge="0.3396", arc_easy="0.5993", hellaswag="0.6173",
        winogrande="0.6085", boolq="0.5963", piqa="0.7350",
    ),
}
MODELS = ["fp16", "rtn-int8", "rtn-int4", "gptq-int4", "awq-int4"]
# Holm family of 7. Order is vs-FP16 first, then INT4-method contrasts.
FAMILY = [
    ("RTN INT8 vs FP16", "rtn-int8", "fp16"),
    ("RTN INT4 vs FP16", "rtn-int4", "fp16"),
    ("GPTQ INT4 vs FP16", "gptq-int4", "fp16"),
    ("AWQ INT4 vs FP16", "awq-int4", "fp16"),
    ("AWQ INT4 vs RTN INT4", "awq-int4", "rtn-int4"),
    ("GPTQ INT4 vs RTN INT4", "gptq-int4", "rtn-int4"),
    ("AWQ INT4 vs GPTQ INT4", "awq-int4", "gptq-int4"),
]
B = 10000
RNG_SEED = 20261119
PMIN = 2 / (B + 1)


def norm(p):
    return str(p).replace("\\", "/")


def list_entries(src: Path):
    """Yield (stored_name, bytes_or_None, is_zip) for each file."""
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                yield n, z.read(n), True
        return
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                yield str(p.relative_to(src)), p.read_bytes(), False
        return
    raise FileNotFoundError(f"not a zip or directory: {src}")


def index_inputs(src: Path):
    files = defaultdict(lambda: {"samples": {}, "results": {}})
    raw = {}
    for stored, blob, _ in list_entries(src):
        nn = norm(stored)
        parts = nn.split("/")
        folder = parts[0]
        if folder.endswith("-full"):
            model, kind = folder[: -len("-full")], "full"
        elif folder.endswith("-qa"):
            model, kind = folder[: -len("-qa")], "qa"
        else:
            continue
        bn = parts[-1]
        raw[(model, kind, bn)] = blob
        if bn.startswith("results_"):
            files[model]["results"][kind] = (stored, blob)
        elif bn.startswith("samples_"):
            for t in TASKS:
                if bn.startswith(f"samples_{t}_"):
                    files[model]["samples"][t] = (stored, blob)
                    break
    return files


def load_results_json(blob):
    d = json.loads(blob)
    out = {}
    for task, mets in d.get("results", {}).items():
        rec = {}
        for k, v in mets.items():
            if isinstance(v, (int, float)) and k.endswith(",none"):
                rec[k.split(",")[0]] = float(v)
        out[task] = rec
    return out


def parse_samples(blob, metric):
    doc_ids, corr = [], []
    for line in blob.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if metric not in rec:
            raise KeyError(f"sample missing {metric}: keys={list(rec)[:20]}")
        v = rec[metric]
        if v not in (0, 1, 0.0, 1.0):
            raise ValueError(f"non-binary {metric}={v!r}")
        doc_ids.append(rec["doc_id"])
        corr.append(float(v))
    return doc_ids, np.asarray(corr, dtype=np.float64)


def round4_half_up(x):
    return format(Decimal(str(x) if False else x).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    ), "f")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [None] * len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


def pval(bs):
    return min(1.0, 2 * min(((bs <= 0).sum() + 1) / (B + 1), ((bs >= 0).sum() + 1) / (B + 1)))


def mcn(x, y):
    n01 = int(((x == 1) & (y == 0)).sum())  # first only
    n10 = int(((x == 0) & (y == 1)).sum())  # second only
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    n11 = int(((x == 1) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    return dict(a_only=n01, b_only=n10, both_right=n11, both_wrong=n00, n_discordant=n, p=float(p))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("p1_paired_results.json")
    out_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else out_json.with_name("p1_correctness.csv")

    files = index_inputs(src)
    missing = [m for m in MODELS if m not in files]
    if missing:
        sys.exit(f"missing models in {src}: {missing}")

    cell = {}  # model -> task -> array aligned to reference doc_ids
    ids = None
    json_metrics = {}
    paper_check = {}
    acc = {}
    mismatches = []

    for model in MODELS:
        rec = files[model]
        jm = {}
        for kind, (_n, blob) in rec["results"].items():
            jm.update(load_results_json(blob))
        json_metrics[model] = jm
        raw_ids, raw_corr = {}, {}
        for t in TASKS:
            if t not in rec["samples"]:
                mismatches.append(f"{model} missing samples for {t}")
                continue
            metric = EXPECT_METRIC[t]
            d, c = parse_samples(rec["samples"][t][1], metric)
            raw_ids[t], raw_corr[t] = d, c
        if ids is None:
            ids = {t: list(raw_ids[t]) for t in TASKS}
            for t in TASKS:
                if len(ids[t]) != EXPECT_N[t]:
                    mismatches.append(f"{t} n={len(ids[t])} expected {EXPECT_N[t]}")
                if len(set(ids[t])) != len(ids[t]):
                    mismatches.append(f"{t} duplicate doc_id")
        else:
            for t in TASKS:
                if list(raw_ids[t]) != ids[t]:
                    # try align by id
                    if set(raw_ids[t]) != set(ids[t]):
                        mismatches.append(f"{model} {t} doc_id set differs")
                    else:
                        mismatches.append(f"{model} {t} doc_id order differs (will align)")
        # align to reference order
        aligned = {}
        for t in TASKS:
            if list(raw_ids[t]) == ids[t]:
                aligned[t] = raw_corr[t]
            else:
                pos = {d: i for i, d in enumerate(raw_ids[t])}
                aligned[t] = np.asarray([raw_corr[t][pos[d]] for d in ids[t]], dtype=np.float64)
        cell[model] = aligned
        acc[model] = {t: float(aligned[t].mean()) for t in TASKS}
        paper_check[model] = {}
        for t in TASKS:
            exact = acc[model][t]
            metric = EXPECT_METRIC[t]
            jv = jm.get(t, {}).get(metric)
            displayed = format(
                Decimal(exact).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), "f"
            )
            # keep 4 places (0.6000 not 0.6)
            if "." in displayed:
                whole, frac = displayed.split(".")
                displayed = whole + "." + frac.ljust(4, "0")
            expected = PAPER_DISPLAY[model][t]
            json_diff = None if jv is None else float(exact - jv)
            paper_check[model][t] = dict(
                metric=metric, n=int(aligned[t].size), n_correct=int(aligned[t].sum()),
                exact=exact, rerun_json=jv, json_diff=json_diff,
                displayed=displayed, paper=expected, match=(displayed == expected),
            )
            if jv is not None and abs(exact - jv) > 1e-12:
                mismatches.append(f"{model} {t}: sample mean {exact} != results JSON {jv}")
            if displayed != expected:
                mismatches.append(
                    f"{model} {t}: displayed {displayed} != paper {expected} (exact {exact})"
                )

    if mismatches:
        print("VERIFICATION FAILED:")
        for row in mismatches:
            print(" ", row)
        sys.exit(1)

    print("verification: sample means match rerun results JSONs and the paper tables")
    for model in MODELS:
        for t in TASKS:
            c = paper_check[model][t]
            print(f"  {model:10s} {t:16s} {c['metric']:8s} {c['n_correct']}/{c['n']} "
                  f"exact={c['exact']:.16f} display={c['displayed']}")

    # six-task and four-task observed averages (unrounded)
    four = TASKS[:4]
    def six_avg(m):
        return float(np.mean([acc[m][t] for t in TASKS]))
    def four_avg(m):
        return float(np.mean([acc[m][t] for t in four]))

    print("six-task averages (unrounded, as fractions):")
    for m in MODELS:
        print(f"  {m:10s} six={six_avg(m):.16f}  four={four_avg(m):.16f}")

    rng = np.random.default_rng(RNG_SEED)
    nq = {t: len(ids[t]) for t in TASKS}
    idx = {t: rng.integers(0, nq[t], (B, nq[t])) for t in TASKS}

    def boot(a, b, tasks=TASKS):
        d = {t: cell[a][t] - cell[b][t] for t in tasks}
        obs = float(np.mean([d[t].mean() for t in tasks]) * 100)
        bs = np.mean([d[t][idx[t]].mean(1) for t in tasks], 0) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(diff=obs, lo=float(lo), hi=float(hi), p=float(pval(bs)),
                    boot_se=float(bs.std(ddof=1)), p_at_floor=bool(pval(bs) <= PMIN + 1e-12))

    def boot_task(a, b, t):
        d = cell[a][t] - cell[b][t]
        bs = d[idx[t]].mean(1) * 100
        return dict(diff=float(d.mean() * 100), lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)), p=float(pval(bs)),
                    p_at_floor=bool(pval(bs) <= PMIN + 1e-12))

    res = [boot(a, b) for _, a, b in FAMILY]
    res4 = [boot(a, b, tasks=four) for _, a, b in FAMILY]
    p_boot = [r["p"] for r in res]
    h_boot = holm(p_boot)

    mc_pooled, mc_task = [], {}
    for name, a, b in FAMILY:
        X = np.concatenate([cell[a][t] for t in TASKS])
        Y = np.concatenate([cell[b][t] for t in TASKS])
        mc_pooled.append(mcn(X, Y))
        mc_task[name] = {t: mcn(cell[a][t], cell[b][t]) for t in TASKS}
    p_mc = [d["p"] for d in mc_pooled]
    h_mc = holm(p_mc)

    main_rows = []
    print("\nMAIN (six-task average, percentage points; first minus second)")
    print(f"Holm family ({len(FAMILY)}): " + "; ".join(n for n, _, _ in FAMILY))
    for i, (name, a, b) in enumerate(FAMILY):
        rec = dict(name=name, a=a, b=b, **res[i],
                   p_holm=h_boot[i],
                   four_task=res4[i],
                   mcnemar_pooled=mc_pooled[i],
                   mcnemar_p=p_mc[i], mcnemar_p_holm=h_mc[i])
        main_rows.append(rec)
        r, m = res[i], mc_pooled[i]
        print(f"  {name:24s} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"p_boot={r['p']:.4f} holm={h_boot[i]:.4f}  "
              f"McN {m['a_only']}/{m['b_only']} p={m['p']:.4g} holm={h_mc[i]:.4g}")

    per_task = {}
    print("\nPER TASK")
    for name, a, b in FAMILY:
        per_task[name] = {}
        for t in TASKS:
            bt = boot_task(a, b, t)
            mt = mc_task[name][t]
            per_task[name][t] = dict(**bt, mcnemar=mt)
            print(f"  {name:24s} {t:16s} {bt['diff']:+.4f} "
                  f"[{bt['lo']:+.4f}, {bt['hi']:+.4f}] p={bt['p']:.4f}  "
                  f"McN {mt['a_only']}/{mt['b_only']} p={mt['p']:.4g}")

    # agreement extras (useful for INT8 vs FP16)
    agreement = {}
    for name, a, b in FAMILY:
        rec = {}
        for t in TASKS:
            mt = mc_task[name][t]
            n = nq[t]
            rec[t] = dict(
                agree=mt["both_right"] + mt["both_wrong"],
                agree_frac=(mt["both_right"] + mt["both_wrong"]) / n,
                **{k: mt[k] for k in ("a_only", "b_only", "both_right", "both_wrong")},
            )
        X = np.concatenate([cell[a][t] for t in TASKS])
        Y = np.concatenate([cell[b][t] for t in TASKS])
        rec["pooled"] = dict(
            agree=int((X == Y).sum()),
            agree_frac=float((X == Y).mean()),
            n=int(X.size),
        )
        agreement[name] = rec

    out = {
        "settings": dict(
            B=B, rng_seed=RNG_SEED, tasks=TASKS, n_questions=nq,
            n_questions_total=int(sum(nq.values())),
            metrics=EXPECT_METRIC,
            p_floor=PMIN, ci="percentile 95%",
            statistic="six-task average of per-task means, percentage points",
            mcnemar="exact binomial, two-sided; pooled is question-weighted",
            holm_family=[n for n, _, _ in FAMILY],
            holm_family_size=len(FAMILY),
            holm_note="Holm step-down over all seven Phase 1 comparisons "
                      "(four methods vs FP16, and the three INT4-method contrasts). "
                      "No comparison is left outside the family.",
            source=str(src),
            method="same as lora/stats/paired_bootstrap.py (Phase 2 appendix)",
        ),
        "accuracy": acc,
        "six_task_avg": {m: six_avg(m) for m in MODELS},
        "four_task_avg": {m: four_avg(m) for m in MODELS},
        "paper_check": paper_check,
        "main": main_rows,
        "per_task": per_task,
        "agreement": agreement,
        "doc_id_aligned": True,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {out_json}")

    # compact correctness CSV (not the 46 MB logs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "doc_id"] + [m.replace("-", "_") for m in MODELS])
        for t in TASKS:
            for i, did in enumerate(ids[t]):
                w.writerow([t, did] + [int(cell[m][t][i]) for m in MODELS])
    print(f"wrote {out_csv} ({out_csv.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
