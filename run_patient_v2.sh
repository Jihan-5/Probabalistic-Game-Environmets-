#!/usr/bin/env bash
# LLM-simulated patient, version 2 (PREREG Amendment 5): 100-session pilot,
# the same audit gate as v1 (contradiction lift >= 1.5), then n=600 reusing the
# pilot sessions. Patient and reply labeler: GPT-4o-mini (OpenRouter); agent:
# Llama-3.3-70B (Together). Budget is in real dollars, checked at meter x 1.18
# (Together's observed billing rate; conservative for the OpenRouter part).
#
#   caffeinate -i bash run_patient_v2.sh            # BUDGET=5.5
set -uo pipefail
cd "$(dirname "$0")"
BUDGET=${BUDGET:-5.5}
FACTOR=${FACTOR:-1.18}
LOG=logs/paid
mkdir -p "$LOG"
SPEND="$LOG/patient_v2_spend.tsv"
touch "$SPEND"
read -r -a EXTRA <<< "${PAID_EXTRA_ARGS:-}"   # test hook only (--mock in a scratch copy)
V2=data/trajectories/healthcare_llmpatient_v2
COMMON=(--model 70B --seed 7 --patient-version v2 --patient-knows profile
        --patient-model gpt-4o-mini --extractor-model gpt-4o-mini --resume)

spent() { awk -F'\t' '{s += $2} END {printf "%.4f", s + 0}' "$SPEND"; }
real() { python3 -c "print(f'{$1 * $FACTOR:.2f}')"; }

step() {  # step <id> <meter cap> <done marker> -- <python args...>
  local id="$1" cap="$2" done="$3"; shift 4
  if [ -e "$done" ]; then echo "[skip $id] $done exists"; return 0; fi
  local s; s=$(spent)
  if ! python3 -c "import sys; sys.exit(0 if ($s + $cap) * $FACTOR <= $BUDGET else 1)"; then
    echo "[patient-v2] skip $id: ~\$$(real "$s") spent + up to \$$(real "$cap") would pass \$$BUDGET"; return 1
  fi
  echo
  echo "=== [$id] up to ~\$$(real "$cap") real; ~\$$(real "$s") of \$$BUDGET spent so far"
  local log="$LOG/P$id.log" rc=0 c
  python3 "$@" ${EXTRA[@]+"${EXTRA[@]}"} --max-cost "$cap" 2>&1 | tee "$log" || rc=$?
  c=$(grep -oE '^\[cost\] \$[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$' || true)
  [ -z "$c" ] && c=$cap
  printf "%s\t%s\t%s\n" "$id" "$c" "$(date +%Y-%m-%dT%H:%M:%S)" >> "$SPEND"
  if [ "$rc" -ne 0 ]; then echo "[patient-v2] $id FAILED (exit $rc); log $log"; return 1; fi
  if grep -q '^\[abort\]' "$log"; then echo "[patient-v2] $id stopped early; rerun resumes"; return 1; fi
  return 0
}

step v2pilot 0.85 "$V2/pilot/70B_7_n100.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
  "${COMMON[@]}" --n 100 --out "$V2/pilot/70B_7_n100.jsonl" || exit 1

GATE_LIFT=${GATE_LIFT:-1.5}
rc=0
python3 -m eval.paid.audit_llmpatient "$V2/pilot/70B_7_n100.jsonl" --seed 7 --gate-lift "$GATE_LIFT" \
  --out "$LOG/audit_v2_70B_7_n100.txt" > "$LOG/audit_v2_gate.log" 2>&1 || rc=$?
tail -3 "$LOG/audit_v2_gate.log"
if [ "$rc" -eq 3 ]; then
  echo "[gate v2] FAIL (lift < $GATE_LIFT): v2 reactions also carry too little signal; not running n=600."
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "[patient-v2] audit crashed (exit $rc)"; exit 1
fi
echo "[gate v2] PASS (lift >= $GATE_LIFT)"
if [ ! -e "$V2/70B_7.jsonl" ] && [ ! -e "$V2/70B_7.jsonl.partial.jsonl" ]; then
  cp "$V2/pilot/70B_7_n100.jsonl" "$V2/70B_7.jsonl.partial.jsonl"   # reuse the 100 pilot sessions
fi
step v2full 3.8 "$V2/70B_7.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
  "${COMMON[@]}" --n 600 --out "$V2/70B_7.jsonl"
echo
echo "[patient-v2] done. Real spend ~\$$(real "$(spent)") of \$$BUDGET."
