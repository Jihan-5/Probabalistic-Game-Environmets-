#!/usr/bin/env bash
# Together-only steps (Llama-3.3-70B agent) under a hard REAL-money budget.
#
# Together bills about 18% above the meter in eval/paid (dashboard $7.70 vs
# meter $6.52 for the same 14,735 calls of step 1a), so every check here is
# meter x FACTOR against BUDGET. Steps run in priority order; a step that would
# exceed the budget is skipped, never started.
#
#   caffeinate -i bash run_together.sh              # BUDGET=20 (dollars, real)
#   BUDGET=15 bash run_together.sh                  # tighter
#
# Order: 1a (finish the 70B corpus; needed by 3 and 4) -> 4 CoVe -> 3 measured
# clarification on the first 120 split-seed-42 test sessions (budget deviation
# from 180, reported) -> 2a patient pilot -> gate -> 2c patient n=600 -> 2b
# informed pilot (only if money is left). Spend log: logs/paid/together_spend.tsv
# (meter dollars). Safe to rerun: finished steps are skipped, partial ones resume.
set -uo pipefail
cd "$(dirname "$0")"
BUDGET=${BUDGET:-20}
FACTOR=${FACTOR:-1.18}
MARGIN=0.25
LOG=logs/paid
mkdir -p "$LOG"
SPEND="$LOG/together_spend.tsv"
touch "$SPEND"
read -r -a EXTRA <<< "${PAID_EXTRA_ARGS:-}"   # test hook only (--mock in a scratch copy)
CF=data/trajectories/healthcare_confirm
LP=data/trajectories/healthcare_llmpatient

spent() { awk -F'\t' '{s += $2} END {printf "%.4f", s + 0}' "$SPEND"; }
real() { python3 -c "print(f'{$1 * $FACTOR:.2f}')"; }

step() {  # step <id> <meter cap> <done marker> -- <python args...>; returns 1 if not completed
  local id="$1" cap="$2" done="$3"; shift 4
  if [ -e "$done" ]; then echo "[skip $id] $done exists"; return 0; fi
  local s; s=$(spent)
  if ! python3 -c "import sys; sys.exit(0 if ($s + $cap) * $FACTOR <= $BUDGET - $MARGIN else 1)"; then
    echo "[together] skip $id: ~\$$(real "$s") spent + up to \$$(real "$cap") would pass the \$$BUDGET budget"
    return 1
  fi
  echo
  echo "=== [$id] up to ~\$$(real "$cap") real (meter cap \$$cap); ~\$$(real "$s") of \$$BUDGET spent so far"
  local log="$LOG/T$id.log" rc=0 c
  python3 "$@" ${EXTRA[@]+"${EXTRA[@]}"} --max-cost "$cap" 2>&1 | tee "$log" || rc=$?
  c=$(grep -oE '^\[cost\] \$[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$' || true)
  [ -z "$c" ] && c=$cap  # spend unknown (crashed before its summary): count the whole cap
  printf "%s\t%s\t%s\n" "$id" "$c" "$(date +%Y-%m-%dT%H:%M:%S)" >> "$SPEND"
  if [ "$rc" -ne 0 ]; then echo "[together] $id FAILED (exit $rc); log $log"; return 1; fi
  if grep -q '^\[abort\]' "$log"; then echo "[together] $id stopped early; log $log (rerun resumes)"; return 1; fi
  return 0
}

step 1a 4.6 "$CF/70B_7.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model 70B --protocol v2fb --seed 7 --n 600 --resume \
  || { echo "[together] the 70B corpus is not complete; steps 3 and 4 need it. Fix, then rerun."; exit 1; }

step 4 0.9 "results/cove_baseline/70B_on_70B_7/scores.json" -- -m eval.paid.run_cove \
  --model 70B --corpus "$CF/70B_7.jsonl" --gen-seed 7

step 3 5.2 "results/e8_closed_loop/70B_7/rerun_full.jsonl" -- -m eval.paid.run_clarification_closed_loop \
  --model 70B --cache "$CF/70B_7.jsonl" --seed 7 --picker v6 --limit 120 --resume

if step 2a 0.7 "$LP/pilot/70B_7_n100.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
     --model 70B --seed 7 --n 100 --patient-knows profile --out "$LP/pilot/70B_7_n100.jsonl" --resume; then
  rc=0
  python3 -m eval.paid.audit_llmpatient "$LP/pilot/70B_7_n100.jsonl" --seed 7 --gate-lift 1.5 \
    --out "$LOG/audit_70B_7_n100.txt" > /dev/null || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[gate 2c] PASS (contradiction lift >= 1.5)"
    if [ ! -e "$LP/70B_7.jsonl" ] && [ ! -e "$LP/70B_7.jsonl.partial.jsonl" ]; then
      cp "$LP/pilot/70B_7_n100.jsonl" "$LP/70B_7.jsonl.partial.jsonl"   # reuse the 100 pilot sessions
    fi
    step 2c 3.6 "$LP/70B_7.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
      --model 70B --seed 7 --n 600 --patient-knows profile --out "$LP/70B_7.jsonl" --resume
  elif [ "$rc" -eq 3 ]; then
    echo "[gate 2c] FAIL (lift < 1.5): the patient's reactions carry too little signal; skipping n=600"
  else
    echo "[together] audit crashed (exit $rc)"
  fi
fi

step 2b 0.7 "$LP/pilot/70B_7_n100_informed.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
  --model 70B --seed 7 --n 100 --patient-knows informed --out "$LP/pilot/70B_7_n100_informed.jsonl" --resume

echo
echo "[together] done. Real spend ~\$$(real "$(spent)") of \$$BUDGET (meter \$$(spent), x$FACTOR)."
