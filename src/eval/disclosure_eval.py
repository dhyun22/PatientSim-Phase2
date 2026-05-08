"""Phase 2 disclosure & questioning-quality evaluator.

Reads a dialogue.jsonl produced by run_simulation.py, runs two LLM-judge
passes (patient-turn extraction + doctor-turn question tagging), and
aggregates per-dialogue metrics:
  - SDR  (Spontaneous Disclosure Rate)
  - QTDR (Question-Triggered Disclosure Rate)
  - Total_30pt (Phase-1 compatible coverage score)
  - SAMPLE / OPQRST coverage

Usage:
    python eval/disclosure_eval.py \
        --dialogue_jsonl results/<...>/outputs/dialogue.jsonl \
        --evaluator gemini-flash-lite-latest \
        --evaluator_api_type genai
"""
import os
import re
import sys
import ast
import json
import glob
import argparse
import logging

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from models import get_response_method, vllm_model_setup
from utils import load_json, load_jsonl, save_to_json, file_to_string

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------- helpers ----------

def _extract_text(response):
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    if hasattr(response, "text"):
        return response.text.strip()
    raise NotImplementedError("unsupported response type")


def _parse_json_obj(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in: {text[:300]}")
    snippet = match.group()
    try:
        return ast.literal_eval(snippet)
    except Exception:
        return json.loads(snippet)


def _judge_call(client, model, prompt, temperature, max_retries=5):
    msg = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            response = client(msg, model=model, temperature=temperature, seed=None)
            return _parse_json_obj(_extract_text(response))
        except Exception as e:
            logging.warning(f"judge attempt {attempt+1} failed: {e}")
    return {}


# ---------- prompt builders ----------

def build_field_definition_text(fields_schema):
    return "\n".join(f"  - {f['id']}: {f['definition']}" for f in fields_schema["fields"])


def build_sample_letter_text(fields_schema):
    return "\n".join(f"  - {l['letter']}: {l['topic']}" for l in fields_schema["sample_letters"])


def build_opqrst_letter_text(fields_schema):
    return "\n".join(f"  - {l['letter']}: {l['topic']}" for l in fields_schema["opqrst_letters"])


# ---------- field extraction ----------

def label_patient_turn(client, model, doctor_prev, patient_curr,
                       prompt_template, field_text, temperature):
    prompt = prompt_template.format(
        field_definitions=field_text,
        doctor_prev=doctor_prev or "",
        patient_curr=patient_curr or "",
    )
    return _judge_call(client, model, prompt, temperature)


def label_doctor_turn(client, model, doctor_curr,
                      prompt_template, field_text, sample_text, opqrst_text, temperature):
    prompt = prompt_template.format(
        field_definitions=field_text,
        sample_letters=sample_text,
        opqrst_letters=opqrst_text,
        doctor_curr=doctor_curr or "",
    )
    return _judge_call(client, model, prompt, temperature)


# ---------- aggregation ----------

GT_POS_FIELDS = [
    "present_illness_onset", "present_illness_location", "present_illness_character",
    "present_illness_severity", "present_illness_timing", "present_illness_radiation",
    "present_illness_alleviating", "present_illness_aggravating",
    "present_illness_associated_positive",
]


def _count_revealed(field_id, patient_labels, mode_filter=None):
    """How many turns revealed this field; mode_filter='asked'|'volunteered'|None."""
    n = 0
    for lbl in patient_labels:
        if not isinstance(lbl, dict):
            continue
        item = lbl.get(field_id)
        if isinstance(item, dict) and item.get("revealed"):
            if mode_filter is None or item.get("mode") == mode_filter:
                n += 1
    return n


def _any_revealed(field_id, patient_labels):
    return _count_revealed(field_id, patient_labels) > 0


def _gt_count(value):
    """Approximate GT element count by splitting on commas."""
    if value is None:
        return 0
    s = str(value).strip()
    if not s or s.lower() in {"none", "n/a", "no", "denies"}:
        return 0
    return len([p for p in re.split(r"[,;]", s) if p.strip()])


def aggregate_dialogue(dialogue_record, patient_labels, doctor_labels, gt_profile, max_turn_total_30pt=30):
    """Compute disclosure & questioning-quality metrics for a single dialogue."""
    history = dialogue_record["dialog_history"]

    # Volunteered / asked counts (per patient turn)
    n_patient_turns = sum(1 for t in history if t["role"] == "Patient")
    n_volunteered_turns = 0
    for lbl in patient_labels:
        if not isinstance(lbl, dict):
            continue
        if any(isinstance(v, dict) and v.get("revealed") and v.get("mode") == "volunteered" for v in lbl.values()):
            n_volunteered_turns += 1
    sdr = (n_volunteered_turns / n_patient_turns) if n_patient_turns else 0.0

    # QTDR — for each (doctor turn, requested field) pair, did the FOLLOWING patient turn reveal it?
    pair_total = 0
    pair_responded = 0
    # pair patient_labels[i] with doctor_labels[i] (doctor asks at turn i, patient answers at turn i)
    for i, dlbl in enumerate(doctor_labels):
        if not isinstance(dlbl, dict):
            continue
        requested = dlbl.get("fields_requested") or []
        if i >= len(patient_labels):
            continue
        plbl = patient_labels[i] if isinstance(patient_labels[i], dict) else {}
        for fid in requested:
            pair_total += 1
            item = plbl.get(fid)
            if isinstance(item, dict) and item.get("revealed"):
                pair_responded += 1
    qtdr = (pair_responded / pair_total) if pair_total else 0.0

    # Phase-1 30-pt rubric:
    #   Present_Illness(+) : 7 pts  → mapped to # of GT_POS_FIELDS revealed across dialogue (cap at 9 → score = round(7 * x/9))
    #   Present_Illness(−) : 13 pts → counted by GT items denied (cap at GT count)
    #   Pain               : 1 pt   → 1 if pain_level_numeric revealed
    #   Medication         : 9 pts  → fraction of GT meds covered
    pos_revealed_dims = sum(1 for f in GT_POS_FIELDS if _any_revealed(f, patient_labels))
    pos_score = round(7 * min(pos_revealed_dims, 9) / 9)

    n_neg_revealed = _count_revealed("present_illness_associated_negative", patient_labels)
    n_neg_gt = _gt_count(gt_profile.get("present_illness_negative"))
    neg_score = round(13 * min(n_neg_revealed, n_neg_gt) / n_neg_gt) if n_neg_gt else 0

    pain_score = 1 if _any_revealed("pain_level_numeric", patient_labels) else 0

    n_med_revealed = _count_revealed("medication", patient_labels)
    n_med_gt = _gt_count(gt_profile.get("medication"))
    med_score = round(9 * min(n_med_revealed, n_med_gt) / n_med_gt) if n_med_gt else 0

    total_30pt = pos_score + neg_score + pain_score + med_score

    # SAMPLE / OPQRST coverage from doctor labels
    sample_set, opqrst_set = set(), set()
    for dlbl in doctor_labels:
        if not isinstance(dlbl, dict):
            continue
        for s in dlbl.get("sample_letters") or []:
            sample_set.add(s.upper())
        for o in dlbl.get("opqrst_letters") or []:
            opqrst_set.add(o.upper())
    sample_coverage = len(sample_set & {"S", "A", "M", "P", "L", "E"}) / 6.0
    opqrst_coverage = len(opqrst_set & {"O", "P", "Q", "R", "S", "T"}) / 6.0

    return {
        "n_patient_turns": n_patient_turns,
        "sdr": sdr,
        "qtdr": qtdr,
        "pair_total": pair_total,
        "pair_responded": pair_responded,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "pain_score": pain_score,
        "med_score": med_score,
        "total_30pt": total_30pt,
        "sample_coverage": sample_coverage,
        "opqrst_coverage": opqrst_coverage,
        "sample_letters_covered": sorted(sample_set),
        "opqrst_letters_covered": sorted(opqrst_set),
    }


# ---------- main ----------

def main(args):
    set_eval_paths = []
    if args.dialogue_jsonl:
        set_eval_paths = [args.dialogue_jsonl]
    else:
        # locate by experiment-name substring under results/
        candidates = sorted(glob.glob(os.path.join(args.result_dir, f"*{args.trg_exp_name}*", "outputs", "dialogue.jsonl")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(args.result_dir, f"*{args.trg_exp_name}*", "**", "dialogue.jsonl"), recursive=True))
        set_eval_paths = candidates

    if not set_eval_paths:
        logging.error(f"no dialogue.jsonl found for {args.trg_exp_name}")
        return

    fields_schema = load_json(args.fields_schema)
    field_text = build_field_definition_text(fields_schema)
    sample_text = build_sample_letter_text(fields_schema)
    opqrst_text = build_opqrst_letter_text(fields_schema)

    patient_template = file_to_string(args.patient_template)
    doctor_template = file_to_string(args.doctor_template)

    client = get_response_method(args.evaluator_api_type)
    model = vllm_model_setup(args.evaluator) if args.evaluator_api_type == "vllm" else args.evaluator

    profiles = load_json(args.patient_profile)
    profiles_by_hadm = {str(int(p["hadm_id"])): p for p in profiles}

    for dialogue_path in set_eval_paths:
        logging.info(f"Evaluating {dialogue_path}")
        records = load_jsonl(dialogue_path)
        out = []
        for rec in records:
            history = rec["dialog_history"]
            gt = profiles_by_hadm.get(str(int(rec["hadm_id"])))
            if gt is None:
                logging.warning(f"no GT profile for hadm_id={rec['hadm_id']}; skipping")
                continue

            # iterate paired (doctor, patient) turns. Doctor speaks first (greet), then alternates.
            doctor_turns = [t["content"] for t in history if t["role"] == "Doctor"]
            patient_turns = [t["content"] for t in history if t["role"] == "Patient"]

            # patient_labels[i] = label of i-th patient turn given doctor_turns[i] as the doctor's prev message
            patient_labels = []
            for i, p_msg in enumerate(tqdm(patient_turns, desc="patient", leave=False)):
                d_prev = doctor_turns[i] if i < len(doctor_turns) else ""
                lbl = label_patient_turn(client, model, d_prev, p_msg,
                                         patient_template, field_text, args.temperature)
                patient_labels.append(lbl)

            # doctor_labels[i] = label of doctor_turns[i] (skip the very first greet if it's identical to the canned greet)
            doctor_labels = []
            for i, d_msg in enumerate(tqdm(doctor_turns, desc="doctor", leave=False)):
                lbl = label_doctor_turn(client, model, d_msg,
                                        doctor_template, field_text, sample_text, opqrst_text, args.temperature)
                doctor_labels.append(lbl)

            metrics = aggregate_dialogue(rec, patient_labels, doctor_labels, gt)
            out.append({
                "hadm_id": rec["hadm_id"],
                "patient_id": rec.get("patient_id"),
                "doctor_strategy": rec.get("doctor_strategy"),
                "personality_type": rec.get("personality_type"),
                "cefr_type": rec.get("cefr_type"),
                "recall_level_type": rec.get("recall_level_type"),
                "dazed_level_type": rec.get("dazed_level_type"),
                "diagnosis": rec.get("diagnosis"),
                "metrics": metrics,
                "patient_labels": patient_labels,
                "doctor_labels": doctor_labels,
            })  # tier derivable from (cefr/recall/dazed) at aggregation time

        save_path = os.path.join(os.path.dirname(dialogue_path), f"disclosure_eval_{args.evaluator.replace('/','_')}.json")
        save_to_json(out, save_path)
        logging.info(f"wrote {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trg_exp_name", type=str, default="phase2", help="substring to match experiment dirs under result_dir")
    parser.add_argument("--dialogue_jsonl", type=str, default=None, help="explicit path to a single dialogue.jsonl")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--evaluator", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--evaluator_api_type", type=str, default="genai")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fields_schema", type=str, default="prompts/eval/disclosure_fields.json")
    parser.add_argument("--patient_template", type=str, default="prompts/eval/disclosure_patient_extract.txt")
    parser.add_argument("--doctor_template", type=str, default="prompts/eval/disclosure_doctor_question_tag.txt")
    parser.add_argument("--patient_profile", type=str, default="data/final_data/patient_profile_phase2.json")
    args = parser.parse_args()
    main(args)
