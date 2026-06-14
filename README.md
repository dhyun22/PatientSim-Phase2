# PatientSim Phase 3 — Controlling Spontaneous Disclosure (Team 12)

ML4H 2026 Phase 3 submission, continuing our Phase 2 disclosure-quality metrics. **GitHub**: <https://github.com/dhyun22/PatientSim-Phase2>

**Phase 3 (latest):** a bucket-gated *controlled-disclosure* patient agent that closes the spontaneous-disclosure leak measured in Phase 2 — see [Phase 3 — Controlled-Disclosure Patient Agent](#phase-3--controlled-disclosure-patient-agent) below. **Phase 2 (baseline):** the disclosure-quality evaluator and the 18-cell measurement of the unmodified PatientSim agent, documented in the rest of this README and left untouched for comparison.

This repository builds on the original PatientSim simulator (Kyung et al., NeurIPS 2025 D&B Track) but contains only the artefacts needed to reproduce our Phase 2 and Phase 3 work; it is not a redistribution of the original codebase.

## Project Overview

Phase 1 (RQ1) found that the original PatientSim patient agent exhibits **excessive spontaneous information disclosure**: even when the doctor only emits a passive cue like `...`, the patient still volunteers most of its clinical chart, collapsing the simulator from a diagnostic-reasoning exercise into passive symptom collection. Phase 2 contributes a **disclosure-quality evaluator** (`src/eval/disclosure_eval.py`) — SDR / QTDR / Total_30pt / SAMPLE·OPQRST coverage — that quantifies, for every dialogue, *which* clinical fields are revealed and *whether they were volunteered or asked-for*. We validate the metric on **18 unique cells** in three arms:

- **MAIN — patient diversity** (3 patients × 3 doctor strategies × Cooperative tier = 9 cells): the metric is consistent across presenting complaints (pneumonia / UTI / chest pain) and discriminates passive vs active doctors (SDR Δ = 0.56, QTDR Δ = 0.89 across arms, while the Phase-1 Total_30pt rubric Δ = 1.17 is indistinguishable).
- **TIER ablation — Phase-1 persona axis** (1 patient × 3 strategies × 3 tiers = 9 cells, 3 shared with MAIN): reproduces Phase 1's three-tier persona axis (Severe / Typical / Cooperative). Passive SDR is 1.00 across every tier; passive Total_30pt is 2 / 7 / 7 — a *tier-conditional* leak shape that exactly matches Phase 1.
- **CUE ablation — passive surface form** (1 patient × 4 passive cues × Cooperative = 4 cells, 1 shared with MAIN): leak is invariant to the cue surface form (`...`, `Hmm`, `I see`, `Oh`).

The patient agent that closes the leak — a bucket-gated **controlled-disclosure** prompt — is the Phase 3 roadmap in the *Next Step* section of the proposal PDF.

## Phase 3 — Controlled-Disclosure Patient Agent

Phase 3 implements that fix and re-runs the **identical** 18-cell matrix, evaluator, and metrics, so every controlled cell is paired to its Phase 2 baseline cell. The change is **prompt-only**: the patient chart is reorganized into topic **buckets** (present illness, pain, medication, allergy, past medical history, family history, social history); every bucket except the chief complaint starts *closed* and opens only when the doctor explicitly asks about that topic; passive cues (`...`, `Hmm`, `I see`, `Oh`) open no bucket and yield a content-free reply; and a turn-level self-check strips volunteering connectors (`and also`, `by the way`). No agent / doctor / persona / evaluator code changed.

```bash
# 1. Run the 18-cell controlled-disclosure grid
python src/scripts/run_phase3_grid.py --arm all --skip_existing
# 2. Evaluate every cell with the SAME disclosure-quality judge + aggregate (writes phase3_summary.* and phase2_vs_phase3*)
bash src/scripts/run_phase3_eval.sh
```

**Headline result (MAIN, Cooperative, 3 patients), Phase 2 → Phase 3:**

| Doctor strategy | SDR | QTDR | Total_30pt | SAMPLE |
|---|---|---|---|---|
| **passive (`...`)** | 1.00 → 1.00 | 0.00 → 0.00 | **7.67 → 2.67** | 0.00 → 0.00 |
| freestyle (active) | 0.33 → 0.42 | 0.86 → 1.00 | 7.33 → 7.67 | 0.56 → 0.61 |
| sample (active) | 0.56 → 0.56 | 0.92 → 0.92 | 5.67 → 5.33 | 0.39 → 0.33 |

The passive doctor's disclosed *volume* (Total_30pt) collapses (−5.0) while active interviewing is fully retained (active QTDR ≈ 0.96), reversing the Phase 2 pathology into the intended `active > passive` ordering. Passive **SDR stays ≈1.0** because the always-allowed chief complaint counts as volunteered — so Total_30pt, not SDR, is the metric that captures this fix.

**New / changed Phase 3 files:** `src/prompts/simulation/initial_system_patient_controlled{,_uti}.txt` (the fix), `src/scripts/run_phase3_grid.py`, `src/scripts/run_phase3_eval.sh`, `src/eval/aggregate_phase3.py`, and the `src/results/*phase3_*` artefacts. The Phase 2 baseline prompt and `src/results/phase2_*` artefacts are left untouched for comparison.

## Folder Structure

```
.
├── README.md                        # this file
├── requirements.txt                 # CPU-only Phase 2 deps
├── .env.example                     # required environment variables
└── src/
    ├── run_simulation.py            # dialogue driver (Hydra entry point)
    ├── models.py                    # LLM backend wrappers (Gemini / Azure / OpenAI / vLLM)
    ├── utils.py                     # I/O & seed helpers
    ├── config/base.yaml             # default config (Gemini, 4 turns, passive_dots)
    ├── agent/                       # PatientAgent and DoctorAgent
    ├── data/
    │   ├── cefr_word_dict.json      # CEFR-graded word list (used for patient persona)
    │   └── final_data/
    │       └── patient_profile_phase2.json  # 3 PHI-free synthetic patients
    ├── prompts/
    │   ├── simulation/              # patient + 6 doctor strategy prompts + persona JSONs
    │   └── eval/                    # disclosure_{patient_extract,doctor_question_tag}.txt + fields schema
    ├── eval/
    │   ├── disclosure_eval.py       # two-pass LLM-judge → per-dialogue metrics
    │   └── aggregate_phase2.py      # arm-wise pivot tables (MAIN / TIER / CUE)
    ├── scripts/
    │   ├── run_phase2_grid.py       # 18-cell grid driver (--arm main|tier|abl|all)
    │   └── run_phase2_eval.sh       # run disclosure_eval.py on every cell, then aggregate
    └── results/                     # per-cell dialogue.jsonl + judge JSON + summary tables
```

The original PatientSim patient prompt (`src/prompts/simulation/initial_system_patient_w_persona*.txt`) is **unchanged** — Phase 2 measures the unmodified PatientSim agent.

## Installation

**Requirements**: Python ≥ 3.10 (3.11 recommended), Linux/macOS, **no GPU required** (CPU-only — all LLM work is over the Gemini API).

```bash
git clone https://github.com/dhyun22/PatientSim-Phase2.git
cd PatientSim-Phase2
conda create -n patientsim python=3.11 -y
conda activate patientsim
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GENAI_API_KEY (Vertex AI Express Mode key works with GOOGLE_GENAI_USE_VERTEXAI=False)
export $(grep -v '^#' .env | xargs)
```

## Usage

```bash
# 1. Single smoke test (passive-dots cue, default-setting pneumonia patient)
cd src
python run_simulation.py --config-name base \
    experiment.exp_name=phase2_smoke \
    data.scenario_id=0 \
    doctor_agent.strategy=passive_dots
cd ..

# 2. Full grid (18 unique cells: --arm main | tier | abl | all)
python src/scripts/run_phase2_grid.py --arm all --skip_existing

# 3. Disclosure-quality LLM-judge evaluation on every produced cell
bash src/scripts/run_phase2_eval.sh

# 4. Aggregate to CSV + Markdown pivot tables
cd src
python eval/aggregate_phase2.py
```

There is no separate "training" stage — the patient agent is unmodified and the grid is a pure inference + LLM-as-judge pipeline.

## Expected Runtime / Resources

- **Hardware**: CPU only. Tested on x86_64 Linux (Python 3.10 / 3.11). No GPU, no CUDA, no special OS dependencies.
- **Wall clock**: full 18-cell grid simulation ~ 40–55 min; full 18-cell judge evaluation ~ 45–60 min. End-to-end (simulation + evaluation + aggregation) ~ 90–115 min.
- **API calls per full run**: ~ 144 simulation calls (18 cells × 8 LLM turns) + ~ 250 judge calls (≈ 14 per cell — patient + doctor passes) ≈ **~ 400 Gemini calls**.
- **API cost**: $0 on the Vertex AI Express Mode free tier for `gemini-flash-lite-latest`; ~ 400 calls is well below daily free-quota limits at the time of writing.
- **Backends exercised**: only Gemini (`google-genai`). The codebase still imports `openai` so Azure / direct-OpenAI configurations also load cleanly, but no OpenAI/Azure calls are made on the default Phase 2 path.

## Expected Output Format

- Per-cell dialogue: `src/results/<timestamp>_phase2_<cell>/outputs/dialogue.jsonl`
- Per-cell eval JSON: `src/results/<...>/outputs/disclosure_eval_gemini-flash-lite-latest.json`
- Aggregated summaries: `src/results/phase2_summary.csv`, `phase2_summary_strategy_mode.md`, `phase2_summary_tier.md`, `phase2_summary_abl.md`, `phase2_summary.json`

The shipped `src/results/` directory already contains the 18 cells we ran, so the headline numbers in the proposal PDF can be reproduced from the committed artefacts without re-running any LLM calls.

## Headline Numbers (`src/results/phase2_summary.csv`)

**MAIN — 3 patients × 3 strategies × Cooperative tier (9 cells):**

| Strategy | n | SDR | QTDR | Total_30pt | SAMPLE | OPQRST |
|---|---|---|---|---|---|---|
| **passive_dots** | 3 | **1.00** | — | **7.67** | 0.00 | 0.00 |
| freestyle | 3 | 0.33 | 0.86 | 7.33 | 0.56 | 0.00 |
| sample | 3 | 0.56 | 0.92 | 5.67 | 0.39 | 0.11 |
| **passive avg** | 3 | **1.00** | — | **7.67** | 0.00 | 0.00 |
| **active avg** | 6 | **0.44** | **0.89** | 6.50 | **0.47** | 0.06 |

**TIER — Phase-1 persona axis (pneumonia × 3 strategies × 3 tiers, 9 cells):**

| Tier | passive_dots Total_30pt | freestyle Total_30pt | sample Total_30pt | passive SDR |
|---|---|---|---|---|
| Severe (CEFR A / recall low / dazed high) | **2** | 5 | 2 | 1.00 |
| Typical (CEFR B / recall low / dazed normal) | **7** | 7 | 4 | 1.00 |
| Cooperative (CEFR C / recall high / dazed normal) | **7** | 7 | 5 | 1.00 |

The flag (passive SDR = 1.00) is uniformly on across every tier; the volume (Total_30pt) tracks Phase 1's tier-conditional leak shape.

**CUE ablation — pneumonia × 4 passive cues × Cooperative (4 cells):**

| Doctor cue | SDR | Total_30pt |
|---|---|---|
| `...` (`passive_dots`) | 1.00 | 7 |
| `Hmm` (`passive_hmm`) | 1.00 | 6 |
| `I see` (`passive_isee`) | 1.00 | 8 |
| `Oh` (`passive_oh`) | 1.00 | 8 |

## Proposal–Code Mapping

| Proposal item | File / function |
|---|---|
| Limitation under study (spontaneous disclosure) | `src/prompts/simulation/initial_system_patient_w_persona.txt` (unmodified original) |
| Passive-cue ablation prompts | `src/prompts/simulation/initial_system_doctor_passive_{dots,hmm,isee,oh}.txt` |
| Active-doctor contrast prompts | `src/prompts/simulation/initial_system_doctor_{freestyle,sample}.txt` |
| Doctor strategy config knob | `src/config/base.yaml` (`doctor_agent.strategy`); resolver `src/run_simulation.py:resolve_doctor_prompt_file` |
| **Metric: Spontaneous Disclosure Rate (SDR)** | `src/eval/disclosure_eval.py:aggregate_dialogue` (key `sdr`) |
| **Metric: Question-Triggered Disclosure Rate (QTDR)** | `src/eval/disclosure_eval.py:aggregate_dialogue` (key `qtdr`) |
| **Metric: Total_30pt** (Phase-1 compatible) | `src/eval/disclosure_eval.py:aggregate_dialogue` (keys `pos_score`, `neg_score`, `pain_score`, `med_score`, `total_30pt`) |
| Metric: SAMPLE / OPQRST coverage | `src/eval/disclosure_eval.py:aggregate_dialogue` (keys `sample_coverage`, `opqrst_coverage`) |
| LLM-judge prompts | `src/prompts/eval/disclosure_patient_extract.txt`, `disclosure_doctor_question_tag.txt` |
| Field schema | `src/prompts/eval/disclosure_fields.json` |
| 18-cell grid driver (MAIN 3×3 + TIER 3×3 + CUE 1×4) | `src/scripts/run_phase2_grid.py` (`--arm main\|tier\|abl\|all`) |
| Persona tier definitions (Severe / Typical / Cooperative) | `src/scripts/run_phase2_grid.py:TIERS` |
| Results aggregation (CSV / MD / JSON) | `src/eval/aggregate_phase2.py` |
| Synthetic patient surrogates (3 patients) | `src/data/final_data/patient_profile_phase2.json` (`scenario_id=0,1,2`) |

### Result reproduction — exact commands

| Result row in proposal PDF | Producing command (run from repo root) | Source data file |
|---|---|---|
| MAIN per-strategy averages (passive_dots / freestyle / sample, n = 3) | `python src/scripts/run_phase2_grid.py --arm main && bash src/scripts/run_phase2_eval.sh && (cd src && python eval/aggregate_phase2.py)` | `src/results/phase2_summary_strategy_mode.md` |
| TIER per-tier-strategy table (Severe / Typical / Cooperative × 3 strategies) | `python src/scripts/run_phase2_grid.py --arm tier && bash src/scripts/run_phase2_eval.sh && (cd src && python eval/aggregate_phase2.py)` | `src/results/phase2_summary_tier.md` |
| CUE ablation table (4 passive cues × pneumonia) | `python src/scripts/run_phase2_grid.py --arm abl && bash src/scripts/run_phase2_eval.sh && (cd src && python eval/aggregate_phase2.py)` | `src/results/phase2_summary_abl.md` |
| Per-cell raw rows (verify any single number) | (same as above; see CSV) | `src/results/phase2_summary.csv` |

## Acknowledgements

The patient agent, persona schema, and base simulation harness are from the original [PatientSim](https://github.com/dek924/PatientSim) (Kyung et al., NeurIPS 2025 D&B Track Spotlight). Phase 2 measures the unmodified PatientSim patient; we add only the disclosure-quality evaluator, the doctor-strategy prompts, the synthetic surrogate patients, and the grid runner.
