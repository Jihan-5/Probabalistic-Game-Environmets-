#!/usr/bin/env bash
# Paid runs, in recommended order (see RUN_PAID.md). Each step has its own
# hard --max-cost. The launcher:
#   - stops on the first error, and on any step that stopped early ([abort]:
#     cost cap, parse-failure guard, repeated HTTP 4xx);
#   - records each step's actual spend in logs/paid/spend.tsv and refuses a
#     step if recorded spend + that step's cap would exceed TOTAL_CAP. This
#     holds ACROSS reruns of this script (a step's --max-cost is per invocation);
#   - skips steps whose output already exists, so rerunning after a fix is safe;
#   - runs the n=600 LLM-patient arms only if the 100-session pilot of that arm
#     passes the audit gate (contradiction lift P(-1|w)/P(-1|r) >= GATE_LIFT).
#
#   bash run_paid.sh               # everything
#   bash run_paid.sh 1 3           # only steps whose id starts with 1 or 3
#   DRY=1 bash run_paid.sh         # print commands + cost estimates, spend $0
#   TOTAL_CAP=25 bash run_paid.sh  # tighter global cap
#
# Keys: TOGETHER_API_KEY and OPENROUTER_API_KEY in .env at the repo root (or exported).
set -euo pipefail
cd "$(dirname "$0")"
LOG=logs/paid
mkdir -p "$LOG"
SPEND="$LOG/spend.tsv"
touch "$SPEND"
TOTAL_CAP=${TOTAL_CAP:-41}  # $50 budget - $7.96 already spent (ledger) - ~$1 margin
GATE_LIFT=${GATE_LIFT:-1.5}
# test hook only: PAID_EXTRA_ARGS="--mock" in a scratch copy of the repo (see RUN_PAID.md)
read -r -a EXTRA <<< "${PAID_EXTRA_ARGS:-}"
SEL=("$@")

selected() {
  [ "${#SEL[@]}" -eq 0 ] && return 0
  local s; for s in "${SEL[@]}"; do [[ "$1" == "$s"* ]] && return 0; done
  return 1
}
spent() { awk -F'\t' '{s += $2} END {printf "%.4f", s + 0}' "$SPEND"; }

step() {  # step <id> <cap> <output that marks the step done> -- <python args...>
  local id="$1" cap="$2" done="$3"; shift 4
  selected "$id" || return 0
  if [ -n "$done" ] && [ -e "$done" ]; then echo "[skip $id] $done exists"; return 0; fi
  local s; s=$(spent)
  if ! python3 -c "import sys; sys.exit(0 if $s + $cap <= $TOTAL_CAP + 1e-9 else 1)"; then
    echo "[run_paid] step $id: recorded spend \$$s + cap \$$cap > TOTAL_CAP \$$TOTAL_CAP; stopping"
    exit 1
  fi
  echo
  echo "=== [$id] cap \$$cap   (recorded spend so far \$$s of TOTAL_CAP \$$TOTAL_CAP)"
  echo "    python3 $* --max-cost $cap"
  if [ "${DRY:-0}" = "1" ]; then
    local est
    if est=$(python3 "$@" ${EXTRA[@]+"${EXTRA[@]}"} --max-cost "$cap" --estimate-only 2>&1); then
      echo "$est" | sed 's/^/    /'
    else
      echo "    (no estimate yet: this step reads the output of an earlier step)"
    fi
    return 0
  fi
  local log="$LOG/$id.log" rc=0 c
  python3 "$@" ${EXTRA[@]+"${EXTRA[@]}"} --max-cost "$cap" 2>&1 | tee "$log" || rc=$?
  c=$(grep -oE '^\[cost\] \$[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$' || true)
  [ -z "$c" ] && c=$cap  # spend unknown (crash before the summary): count the whole cap
  printf "%s\t%s\t%s\n" "$id" "$c" "$(date +%Y-%m-%dT%H:%M:%S)" >> "$SPEND"
  if [ "$rc" -ne 0 ]; then echo "[run_paid] step $id FAILED (exit $rc); stopping. Log: $log"; exit "$rc"; fi
  if grep -q '^\[abort\]' "$log"; then
    echo "[run_paid] step $id stopped early ([abort] in $log); stopping. Fix, then rerun (it resumes)."
    exit 1
  fi
}

gate() {  # gate <id> <pilot jsonl> : 0 = run the full arm
  selected "$1" || return 1
  if [ "${DRY:-0}" = "1" ]; then echo "    (DRY) [$1] runs only if the pilot audit lift >= $GATE_LIFT"; return 0; fi
  [ -e "$2" ] || { echo "[run_paid] $1: pilot $2 missing; run its pilot step first"; exit 1; }
  local rc=0
  python3 -m eval.paid.audit_llmpatient "$2" --seed "${SEED:-42}" --gate-lift "$GATE_LIFT" \
    --out "$LOG/audit_$(basename "$2" .jsonl).txt" > /dev/null || rc=$?
  if [ "$rc" -eq 0 ]; then echo "[gate $1] PASS (lift >= $GATE_LIFT)"; return 0; fi
  if [ "$rc" -eq 3 ]; then echo "[gate $1] FAIL (lift < $GATE_LIFT): skipping the n=600 arm"; return 1; fi
  echo "[run_paid] audit of $2 crashed (exit $rc)"; exit "$rc"
}

seed_from_pilot() {  # seed_from_pilot <pilot> <full output> <n>: reuse pilot sessions (same stream)
  [ "${DRY:-0}" = "1" ] && return 0
  if [ -e "$2" ] && [ "$(wc -l < "$2" | tr -d ' ')" -lt "$3" ]; then
    # a shorter run (e.g. a manual pilot) at the full path: same stream, so resume it
    mv "$2" "$2.partial.jsonl"
    echo "[run_paid] $2 had fewer than $3 sessions; resuming it as $2.partial.jsonl"
  fi
  if [ -e "$1" ] && [ ! -e "$2" ] && [ ! -e "$2.partial.jsonl" ]; then
    cp "$1" "$2.partial.jsonl"
    echo "[run_paid] seeded $2.partial.jsonl with the pilot sessions"
  fi
}

LP=data/trajectories/healthcare_llmpatient
CF=data/trajectories/healthcare_confirm
SEED=7   # confirmatory seed: sessions never used in development (dev corpora are seed 42)

# Steps run in PRIORITY order: if TOTAL_CAP would be exceeded, only the tail
# (optional) steps are refused. Analysis plan: results/PREREG_confirmatory.md
# (committed before any confirmatory data existed).

# 1. Confirmatory corpora: protocol v2fb (agent sees the raw reaction +/-/none),
#    k=3 resamples on EVERY turn (T=0.9, same context as the guess), n=600, seed 7.
step 1a 12.5 "$CF/70B_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model 70B --protocol v2fb --seed $SEED --n 600 --resume
step 1b 1.5 "$CF/8B_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model 8B --protocol v2fb --seed $SEED --n 600 --resume
step 1c 6.0 "$CF/qwen72b_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model qwen72b --protocol v2fb --seed $SEED --n 600 --resume
step 1d 2.5 "$CF/gpt-4o-mini_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model gpt-4o-mini --protocol v2fb --seed $SEED --n 600 --resume
step 1e 5.5 "$CF/mistral-small-24b_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --model mistral-small-24b --protocol v2fb --seed $SEED --n 600 --resume

# 2. LLM-simulated patient (behavioral feedback), seed 7: 100-session pilot per
#    arm; the n=600 profile arm runs only if its pilot passes the audit gate.
step 2a 1.2 "$LP/pilot/70B_${SEED}_n100.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
  --model 70B --seed $SEED --n 100 --patient-knows profile --out "$LP/pilot/70B_${SEED}_n100.jsonl" --resume
step 2b 1.2 "$LP/pilot/70B_${SEED}_n100_informed.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
  --model 70B --seed $SEED --n 100 --patient-knows informed --out "$LP/pilot/70B_${SEED}_n100_informed.jsonl" --resume
if gate 2c "$LP/pilot/70B_${SEED}_n100.jsonl"; then
  seed_from_pilot "$LP/pilot/70B_${SEED}_n100.jsonl" "$LP/70B_$SEED.jsonl" 600
  step 2c 5.0 "$LP/70B_$SEED.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
    --model 70B --seed $SEED --n 600 --patient-knows profile --out "$LP/70B_$SEED.jsonl" --resume
fi

# 3. Measured closed-loop clarification on the confirmatory 70B corpus (test split
#    of split seed 42; V6 picks; both labels). Done marker = a per-condition file,
#    written only when every condition is complete.
step 3 9.0 "results/e8_closed_loop/70B_$SEED/rerun_full.jsonl" -- -m eval.paid.run_clarification_closed_loop \
  --model 70B --cache "$CF/70B_$SEED.jsonl" --seed $SEED --picker v6 --resume

# 4. CoVe (70B self-verification) on the confirmatory 70B test splits.
step 4 1.5 "results/cove_baseline/70B_on_70B_$SEED/scores.json" -- -m eval.paid.run_cove \
  --model 70B --corpus "$CF/70B_$SEED.jsonl" --gen-seed $SEED

# 5. OPTIONAL (refused automatically if TOTAL_CAP would be exceeded):
#    5a informed-patient arm at n=600 (upper-bound channel), if its pilot passes;
#    5b k=10 resamples on user-dependent turns of the 70B split-seed-42 test sessions.
if gate 5a "$LP/pilot/70B_${SEED}_n100_informed.jsonl"; then
  seed_from_pilot "$LP/pilot/70B_${SEED}_n100_informed.jsonl" "$LP/70B_${SEED}_informed.jsonl" 600
  step 5a 5.0 "$LP/70B_${SEED}_informed.jsonl" -- -m eval.paid.generate_healthcare_llmpatient \
    --model 70B --seed $SEED --n 600 --patient-knows informed --out "$LP/70B_${SEED}_informed.jsonl" --resume
fi
step 5b 4.0 "data/trajectories/healthcare_confirm_k10/70B_${SEED}_k10.jsonl" -- -m eval.paid.augment_resamples_k10 \
  --model 70B --input "$CF/70B_$SEED.jsonl" --seed $SEED --k-target 10 --restrict test-42 --turns user \
  --out "data/trajectories/healthcare_confirm_k10/70B_${SEED}_k10.jsonl" --resume

# 6. Second domain (travel booking, pge/benchmarks/travel_v1.py; PREREG Amendment 4):
#    protocol v2fb, k=3 resamples on EVERY turn (T=0.9), n=600, seed 7. Runs after
#    every healthcare step, so it never displaces them (refused if TOTAL_CAP would be
#    exceeded; run it alone with `bash run_paid.sh 6`). Caps ~1.3x the --estimate-only figure.
TC=data/trajectories/travel_confirm
step 6a 2.55 "$TC/gpt-4o-mini_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --domain travel_v1 --model gpt-4o-mini --protocol v2fb --seed $SEED --n 600 --resume
step 6b 0.75 "$TC/8B_$SEED.jsonl" -- -m eval.paid.generate_healthcare_multimodel \
  --domain travel_v1 --model 8B --protocol v2fb --seed $SEED --n 600 --resume

echo
echo "[run_paid] done. Recorded spend this machine: \$$(spent) (logs/paid/spend.tsv). Send back the files listed in RUN_PAID.md."
