"""Aggregate Phase 3 (controlled-disclosure) disclosure-eval results across all
grid cells into a CSV + Markdown summary, and emit a Phase 2 vs Phase 3
baseline-vs-controlled comparison.

Phase 3 uses the SAME 18-cell matrix, evaluator, and metrics as Phase 2; only
the patient prompt changed (bucket-gated controlled disclosure). This script
therefore reuses the Phase 2 row schema and additionally joins each Phase 3
cell to its Phase 2 baseline on (patient_id, tier, doctor_strategy) so the
report can read off the leak reduction directly.

Usage (from src/):
    python eval/aggregate_phase3.py
or:
    python eval/aggregate_phase3.py --result_dir results --evaluator gemini-flash-lite-latest
"""
import os
import sys
import csv
import glob
import json
import argparse
import logging
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


METRIC_COLUMNS = [
    "n_patient_turns",
    "sdr",
    "qtdr",
    "pos_score",
    "neg_score",
    "pain_score",
    "med_score",
    "total_30pt",
    "sample_coverage",
    "opqrst_coverage",
]


def derive_tier(cefr, recall, dazed):
    if cefr == "A":
        return "severe"
    if cefr == "B":
        return "typical"
    if cefr == "C":
        return "cooperative"
    return f"{cefr}/{recall}/{dazed}"


def collect_rows(result_dir, evaluator, exp_substr):
    pattern = os.path.join(result_dir, f"*{exp_substr}*", "outputs", f"disclosure_eval_{evaluator.replace('/','_')}.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for path in files:
        records = load_json(path)
        for rec in records:
            base = {
                "exp_dir": os.path.basename(os.path.dirname(os.path.dirname(path))),
                "patient_id": rec.get("patient_id"),
                "diagnosis": rec.get("diagnosis"),
                "doctor_strategy": rec.get("doctor_strategy"),
                "personality_type": rec.get("personality_type"),
                "cefr_type": rec.get("cefr_type"),
                "recall_level_type": rec.get("recall_level_type"),
                "dazed_level_type": rec.get("dazed_level_type"),
                "tier": derive_tier(rec.get("cefr_type"), rec.get("recall_level_type"), rec.get("dazed_level_type")),
            }
            metrics = rec.get("metrics") or {}
            for k in METRIC_COLUMNS:
                base[k] = metrics.get(k)
            rows.append(base)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def pivot_table(rows, group_keys, value_keys):
    bucket = defaultdict(list)
    for r in rows:
        bucket[tuple(r[k] for k in group_keys)].append(r)
    out = []
    for key, items in sorted(bucket.items()):
        row = dict(zip(group_keys, key))
        row["n_cells"] = len(items)
        for vk in value_keys:
            vals = [it[vk] for it in items if it[vk] is not None]
            row[vk] = round(sum(vals) / len(vals), 3) if vals else None
        out.append(row)
    return out


def write_markdown_table(rows, path, headers):
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n")


def cell_key(r):
    return (r["patient_id"], r["tier"], r["doctor_strategy"])


def build_comparison(phase2_rows, phase3_rows, value_keys):
    """Join Phase 2 baseline and Phase 3 controlled cells on (patient, tier, strategy)."""
    p2 = {cell_key(r): r for r in phase2_rows}
    out = []
    for r3 in phase3_rows:
        k = cell_key(r3)
        r2 = p2.get(k)
        row = {"patient_id": k[0], "tier": k[1], "doctor_strategy": k[2]}
        for vk in value_keys:
            b = r2.get(vk) if r2 else None
            c = r3.get(vk)
            row[f"{vk}_base"] = b
            row[f"{vk}_ctrl"] = c
            row[f"{vk}_delta"] = (round(c - b, 3) if (b is not None and c is not None) else None)
        out.append(row)
    return sorted(out, key=lambda x: (x["patient_id"], x["tier"], x["doctor_strategy"]))


def main(args):
    rows = collect_rows(args.result_dir, args.evaluator, args.exp_substr)
    if not rows:
        logging.error(f"no eval files matching {args.exp_substr} under {args.result_dir}")
        return

    out_csv = os.path.join(args.result_dir, "phase3_summary.csv")
    write_csv(rows, out_csv)
    logging.info(f"wrote {out_csv} with {len(rows)} rows")

    value_keys = ["sdr", "qtdr", "total_30pt", "sample_coverage", "opqrst_coverage"]

    main_rows = [
        r for r in rows
        if r["tier"] == "cooperative"
        and r["doctor_strategy"] in ("passive_dots", "freestyle", "sample")
    ]
    pivot_strategy = pivot_table(main_rows, ["doctor_strategy"], value_keys)
    out_md1 = os.path.join(args.result_dir, "phase3_summary_strategy_mode.md")
    write_markdown_table(pivot_strategy, out_md1, headers=["doctor_strategy", "n_cells"] + value_keys)
    logging.info(f"wrote {out_md1}")

    tier_rows = [
        r for r in rows
        if r["patient_id"] == "pneumonia_001"
        and r["doctor_strategy"] in ("passive_dots", "freestyle", "sample")
    ]
    pivot_tier = pivot_table(tier_rows, ["tier", "doctor_strategy"], value_keys)
    out_md2 = os.path.join(args.result_dir, "phase3_summary_tier.md")
    write_markdown_table(pivot_tier, out_md2, headers=["tier", "doctor_strategy", "n_cells"] + value_keys)
    logging.info(f"wrote {out_md2}")

    abl_rows = [
        r for r in rows
        if r["patient_id"] == "pneumonia_001"
        and r["tier"] == "cooperative"
        and r["doctor_strategy"].startswith("passive_")
    ]
    pivot_abl = pivot_table(abl_rows, ["doctor_strategy"], value_keys)
    out_md3 = os.path.join(args.result_dir, "phase3_summary_abl.md")
    write_markdown_table(pivot_abl, out_md3, headers=["doctor_strategy", "n_cells"] + value_keys)
    logging.info(f"wrote {out_md3}")

    with open(os.path.join(args.result_dir, "phase3_summary.json"), "w") as f:
        json.dump({
            "by_strategy_main": pivot_strategy,
            "by_tier_strategy": pivot_tier,
            "by_passive_cue": pivot_abl,
        }, f, indent=2)

    # ---- Phase 2 vs Phase 3 comparison ----
    phase2_rows = collect_rows(args.result_dir, args.evaluator, "phase2")
    if phase2_rows:
        comp = build_comparison(phase2_rows, rows, value_keys)
        write_csv(comp, os.path.join(args.result_dir, "phase2_vs_phase3.csv"))

        # MAIN passive-vs-active comparison (Cooperative tier), the headline table
        def arm_pivot(filtered):
            p2 = pivot_table(filtered["p2"], ["doctor_strategy"], value_keys)
            p3 = pivot_table(filtered["p3"], ["doctor_strategy"], value_keys)
            p3map = {r["doctor_strategy"]: r for r in p3}
            merged = []
            for r in p2:
                s = r["doctor_strategy"]
                m = {"doctor_strategy": s, "n_cells": r["n_cells"]}
                for vk in value_keys:
                    m[f"{vk}_base"] = r[vk]
                    m[f"{vk}_ctrl"] = p3map.get(s, {}).get(vk)
                merged.append(m)
            return merged

        p2_main = [r for r in phase2_rows if r["tier"] == "cooperative"
                   and r["doctor_strategy"] in ("passive_dots", "freestyle", "sample")]
        merged_main = arm_pivot({"p2": p2_main, "p3": main_rows})
        headers = ["doctor_strategy", "n_cells"] + [f"{vk}_{suf}" for vk in ["sdr", "qtdr", "total_30pt", "sample_coverage"] for suf in ("base", "ctrl")]
        write_markdown_table(merged_main, os.path.join(args.result_dir, "phase2_vs_phase3_main.md"), headers=headers)
        logging.info("wrote phase2_vs_phase3.csv and phase2_vs_phase3_main.md")

        print("\n=== MAIN: Phase2 baseline -> Phase3 controlled (Cooperative, 3 patients) ===")
        for r in merged_main:
            print(f"  {r['doctor_strategy']:14s} | "
                  f"Total30 {r['total_30pt_base']} -> {r['total_30pt_ctrl']} | "
                  f"SDR {r['sdr_base']} -> {r['sdr_ctrl']} | "
                  f"QTDR {r['qtdr_base']} -> {r['qtdr_ctrl']} | "
                  f"SAMPLE {r['sample_coverage_base']} -> {r['sample_coverage_ctrl']}")
    else:
        logging.info("no phase2 baseline rows found; skipped comparison")

    print("\n=== PHASE 3 MAIN: by strategy (3 patients × Cooperative) ===")
    for r in pivot_strategy:
        print(f"  {r['doctor_strategy']:14s} | n={r['n_cells']} "
              f"SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']} "
              f"SAMPLE={r['sample_coverage']} OPQRST={r['opqrst_coverage']}")

    print("\n=== PHASE 3 TIER: tier × strategy (pneumonia) ===")
    for r in pivot_tier:
        print(f"  {r['tier']:12s} | {r['doctor_strategy']:14s} | "
              f"SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']} SAMPLE={r['sample_coverage']}")

    print("\n=== PHASE 3 CUE: passive cue surface form (pneumonia, Cooperative) ===")
    for r in pivot_abl:
        print(f"  {r['doctor_strategy']:14s} | SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--evaluator", type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--exp_substr", type=str, default="phase3")
    args = parser.parse_args()
    main(args)
