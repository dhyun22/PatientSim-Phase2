"""Phase 3 grid runner — controlled-disclosure patient.

Identical 18-cell matrix to Phase 2, but the patient agent now uses the
bucket-gated controlled-disclosure prompt
(``initial_system_patient_controlled``) instead of the unmodified PatientSim
prompt. Everything else (doctor strategies, persona tiers, passive cues,
4-turn dialogues, gemini-flash-lite-latest) is held fixed so Phase 3 cells are
directly comparable to the committed Phase 2 baseline cells.

  MAIN — patient diversity (3 patients × 3 strategies × Cooperative tier = 9 cells)
  TIER — Phase-1 style tier ablation (1 patient × 3 strategies × 3 tiers = 9 cells; 3 shared)
  ABLATION — passive-cue surface form (1 patient × 4 passive cues × Cooperative = 4 cells; 1 shared)
Total unique cells = 18.

Usage (from repo root):
    python src/scripts/run_phase3_grid.py                 # all 18
    python src/scripts/run_phase3_grid.py --arm main      # 9 cells
    python src/scripts/run_phase3_grid.py --arm tier      # 9 cells
    python src/scripts/run_phase3_grid.py --arm abl       # 4 cells
    python src/scripts/run_phase3_grid.py --skip_existing  # resume a partial run
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

# Phase 3 contribution: bucket-gated controlled-disclosure patient prompt.
# patient_agent.py auto-appends "_uti" for the UTI scenario.
PATIENT_PROMPT_FILE = "initial_system_patient_controlled"


def cell_exp_name(patient_id, strategy, tier):
    if tier == "cooperative":
        # backward-compatible: Cooperative cells carry no tier suffix
        return f"phase3_{patient_id}_{strategy}"
    return f"phase3_{patient_id}_{strategy}_{tier}"


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

    logging.info(f"[phase3] arm={arm}: running {len(cells)} cells with "
                 f"total_inferences={total_inf}, patient_prompt={PATIENT_PROMPT_FILE}")

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
            f"data.patient_prompt_file={PATIENT_PROMPT_FILE}",
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
