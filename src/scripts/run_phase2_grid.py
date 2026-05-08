"""Phase 2 grid runner.

Three arms of experiments on the unmodified baseline PatientSim patient:

  MAIN — patient diversity (3 patients × 3 strategies × Cooperative tier = 9 cells):
    - patients : pneumonia_001 (id 0), uti_001 (id 1), chest_pain_001 (id 2)
    - strategies: passive_dots, freestyle, sample
    - tier: Cooperative (CEFR=C, recall=high, dazed=normal, plain)
    Confirms metric behaves consistently across presenting complaints.

  TIER — Phase-1 style tier ablation (1 patient × 3 strategies × 3 tiers = 9 cells;
                                       3 cells shared with MAIN):
    - patient: pneumonia_001 (id 0)
    - strategies: passive_dots, freestyle, sample
    - tiers: Severe (A/low/high/plain), Typical (B/low/normal/plain),
             Cooperative (C/high/normal/plain)
    Confirms metric behaviour across patient-persona axes that Phase 1 used.

  ABLATION — passive-cue surface form (1 patient × 4 passive cues × Cooperative = 4 cells;
                                       1 cell shared with MAIN):
    - patient: pneumonia_001 (id 0)
    - cues: passive_dots, passive_hmm, passive_isee, passive_oh
    Confirms the leak is invariant to the surface form of passive feedback.

Total unique cells = 18.

Usage (from repo root):
    python src/scripts/run_phase2_grid.py             # all 18
    python src/scripts/run_phase2_grid.py --arm main  # 9 cells (patient diversity)
    python src/scripts/run_phase2_grid.py --arm tier  # 9 cells (tier ablation)
    python src/scripts/run_phase2_grid.py --arm abl   # 4 cells (cue ablation)
"""
import os
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Persona tier definitions (CEFR, recall, dazed, personality)
TIERS = {
    "severe":      ("A", "low",  "high",   "plain"),
    "typical":     ("B", "low",  "normal", "plain"),
    "cooperative": ("C", "high", "normal", "plain"),
}

# (patient_id_label, scenario_id, strategy, tier_name)
MAIN_CELLS = [
    ("pneumonia_001",  0, "passive_dots", "cooperative"),
    ("pneumonia_001",  0, "freestyle",    "cooperative"),
    ("pneumonia_001",  0, "sample",       "cooperative"),
    ("uti_001",        1, "passive_dots", "cooperative"),
    ("uti_001",        1, "freestyle",    "cooperative"),
    ("uti_001",        1, "sample",       "cooperative"),
    ("chest_pain_001", 2, "passive_dots", "cooperative"),
    ("chest_pain_001", 2, "freestyle",    "cooperative"),
    ("chest_pain_001", 2, "sample",       "cooperative"),
]
TIER_CELLS = [
    ("pneumonia_001", 0, strat, tier)
    for tier in ["severe", "typical", "cooperative"]
    for strat in ["passive_dots", "freestyle", "sample"]
]
ABL_CELLS = [
    ("pneumonia_001", 0, "passive_dots", "cooperative"),  # shared with MAIN
    ("pneumonia_001", 0, "passive_hmm",  "cooperative"),
    ("pneumonia_001", 0, "passive_isee", "cooperative"),
    ("pneumonia_001", 0, "passive_oh",   "cooperative"),
]

DEFAULT_TOTAL_INFERENCES = 4
PATIENT_BACKEND = "gemini-flash-lite-latest"
PATIENT_API = "genai"
DOCTOR_BACKEND = "gemini-flash-lite-latest"
DOCTOR_API = "genai"


def cell_exp_name(patient_id, strategy, tier):
    if tier == "cooperative":
        # backward-compatible: old cells use no tier suffix (= Cooperative)
        return f"phase2_{patient_id}_{strategy}"
    return f"phase2_{patient_id}_{strategy}_{tier}"


def main(arm="all", skip_existing=False, dry_run=False, total_inferences=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(repo_root, "src")
    sim_script = os.path.join(src_dir, "run_simulation.py")

    total_inf = total_inferences or DEFAULT_TOTAL_INFERENCES

    if arm == "main":
        cells = MAIN_CELLS
    elif arm == "tier":
        cells = TIER_CELLS
    elif arm == "abl":
        cells = ABL_CELLS
    else:
        seen = set()
        cells = []
        for c in MAIN_CELLS + TIER_CELLS + ABL_CELLS:
            if c not in seen:
                seen.add(c)
                cells.append(c)

    logging.info(f"arm={arm}: running {len(cells)} cells with total_inferences={total_inf}")

    for cell_idx, (patient_id, scenario_id, strategy, tier) in enumerate(cells):
        exp_name = cell_exp_name(patient_id, strategy, tier)
        cefr, recall, dazed, personality = TIERS[tier]

        if skip_existing:
            results_dir = os.path.join(src_dir, "results")
            existing = [
                d for d in os.listdir(results_dir)
                if d.endswith(exp_name) and os.path.exists(os.path.join(results_dir, d, "outputs", "dialogue.jsonl"))
            ] if os.path.isdir(results_dir) else []
            if existing:
                logging.info(f"[{cell_idx+1}/{len(cells)}] SKIP {exp_name} (exists)")
                continue

        overrides = [
            f"experiment.exp_name={exp_name}",
            f"experiment.total_inferences={total_inf}",
            f"doctor_agent.max_infs={total_inf}",
            f"data.scenario_id={scenario_id}",
            f"patient_agent.persona.cefr_type={cefr}",
            f"patient_agent.persona.recall_level_option={recall}",
            f"patient_agent.persona.dazed_level_option={dazed}",
            f"patient_agent.persona.personality_type={personality}",
            f"patient_agent.api_type={PATIENT_API}",
            f"patient_agent.backend={PATIENT_BACKEND}",
            f"doctor_agent.strategy={strategy}",
            f"doctor_agent.api_type={DOCTOR_API}",
            f"doctor_agent.backend={DOCTOR_BACKEND}",
        ]

        cmd = ["python", sim_script, "--config-name", "base"] + overrides
        logging.info(f"[{cell_idx+1}/{len(cells)}] {exp_name}")
        if dry_run:
            print("  CMD:", " ".join(cmd))
            continue
        result = subprocess.run(cmd, cwd=src_dir)
        if result.returncode != 0:
            logging.error(f"cell {exp_name} failed (rc={result.returncode}); continuing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["all", "main", "tier", "abl"], default="all",
                        help="which arm to run (default: all unique cells)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--total_inferences", type=int, default=None,
                        help="override turn count per cell (default 4)")
    args = parser.parse_args()
    main(arm=args.arm, skip_existing=args.skip_existing, dry_run=args.dry_run,
         total_inferences=args.total_inferences)
