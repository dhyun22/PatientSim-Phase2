#!/usr/bin/env bash
# Run disclosure_eval.py on every phase2 dialogue.jsonl produced under src/results/
# Writes one disclosure_eval_<evaluator>.json per cell, then aggregates.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_MODEL="${EVAL_MODEL:-gemini-flash-lite-latest}"
EVAL_API_TYPE="${EVAL_API_TYPE:-genai}"

cd "$SRC"
for d in results/*phase2_*; do
  jsonl="$d/outputs/dialogue.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "skip (no dialogue): $d"
    continue
  fi
  out_pattern="$d/outputs/disclosure_eval_${EVAL_MODEL//\//_}.json"
  if [[ -f "$out_pattern" ]]; then
    echo "skip (already evaluated): $d"
    continue
  fi
  echo "evaluating: $d"
  python eval/disclosure_eval.py \
    --dialogue_jsonl "$jsonl" \
    --evaluator "$EVAL_MODEL" \
    --evaluator_api_type "$EVAL_API_TYPE"
done

echo "aggregating..."
python eval/aggregate_phase2.py --evaluator "$EVAL_MODEL" --result_dir results
