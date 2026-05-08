"""Aggregate Phase 2 disclosure-eval results across all grid cells into
a CSV + Markdown summary table.

Usage (from src/):
    python eval/aggregate_phase2.py
or:
    python eval/aggregate_phase2.py --result_dir ../results --evaluator gemini-3.1-flash-lite-preview
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
    """Map (CEFR, recall, dazed) tuple to Phase-1 tier labels.
    Each Phase-1 tier has a unique CEFR (Severe=A, Typical=B, Cooperative=C),
    so CEFR alone is enough; we use the (recall, dazed) tuple only for sanity.
    """
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


def main(args):
    rows = collect_rows(args.result_dir, args.evaluator, args.exp_substr)
    if not rows:
        logging.error(f"no eval files matching {args.exp_substr} under {args.result_dir}")
        return

    out_csv = os.path.join(args.result_dir, "phase2_summary.csv")
    write_csv(rows, out_csv)
    logging.info(f"wrote {out_csv} with {len(rows)} rows")

    value_keys = ["sdr", "qtdr", "total_30pt", "sample_coverage", "opqrst_coverage"]

    # MAIN arm: 3 patients × {passive_dots, freestyle, sample} on Cooperative tier
    main_rows = [
        r for r in rows
        if r["tier"] == "cooperative"
        and r["doctor_strategy"] in ("passive_dots", "freestyle", "sample")
    ]
    pivot_strategy = pivot_table(main_rows, ["doctor_strategy"], value_keys)
    out_md1 = os.path.join(args.result_dir, "phase2_summary_strategy_mode.md")
    write_markdown_table(
        pivot_strategy, out_md1,
        headers=["doctor_strategy", "n_cells"] + value_keys,
    )
    logging.info(f"wrote {out_md1}")

    # TIER arm: pneumonia only × {passive_dots, freestyle, sample} × 3 tiers
    tier_rows = [
        r for r in rows
        if r["patient_id"] == "pneumonia_001"
        and r["doctor_strategy"] in ("passive_dots", "freestyle", "sample")
    ]
    pivot_tier = pivot_table(tier_rows, ["tier", "doctor_strategy"], value_keys)
    out_md2 = os.path.join(args.result_dir, "phase2_summary_tier.md")
    write_markdown_table(
        pivot_tier, out_md2,
        headers=["tier", "doctor_strategy", "n_cells"] + value_keys,
    )
    logging.info(f"wrote {out_md2}")

    # ABL arm: pneumonia, Cooperative, 4 passive cues
    abl_rows = [
        r for r in rows
        if r["patient_id"] == "pneumonia_001"
        and r["tier"] == "cooperative"
        and r["doctor_strategy"].startswith("passive_")
    ]
    pivot_abl = pivot_table(abl_rows, ["doctor_strategy"], value_keys)
    out_md3 = os.path.join(args.result_dir, "phase2_summary_abl.md")
    write_markdown_table(
        pivot_abl, out_md3,
        headers=["doctor_strategy", "n_cells"] + value_keys,
    )
    logging.info(f"wrote {out_md3}")

    with open(os.path.join(args.result_dir, "phase2_summary.json"), "w") as f:
        json.dump({
            "by_strategy_main": pivot_strategy,
            "by_tier_strategy": pivot_tier,
            "by_passive_cue": pivot_abl,
        }, f, indent=2)

    print("\n=== MAIN: by strategy (3 patients × Cooperative) ===")
    for r in pivot_strategy:
        print(f"  {r['doctor_strategy']:14s} | n={r['n_cells']} "
              f"SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']} "
              f"SAMPLE={r['sample_coverage']} OPQRST={r['opqrst_coverage']}")

    print("\n=== TIER ABLATION: tier × strategy (pneumonia) ===")
    for r in pivot_tier:
        print(f"  {r['tier']:12s} | {r['doctor_strategy']:14s} | "
              f"SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']} "
              f"SAMPLE={r['sample_coverage']}")

    print("\n=== CUE ABLATION: passive cue surface form (pneumonia, Cooperative) ===")
    for r in pivot_abl:
        print(f"  {r['doctor_strategy']:14s} | "
              f"SDR={r['sdr']} QTDR={r['qtdr']} Total30={r['total_30pt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results",
                        help="results directory (relative to src/, default 'results')")
    parser.add_argument("--evaluator", type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--exp_substr", type=str, default="phase2")
    args = parser.parse_args()
    main(args)
