#!/usr/bin/env bash
# Pull the user's latest run outputs into paid-scripts, rerun every analysis,
# regenerate the paper's values and figures, and compile. Two jobs at a time.
set -uo pipefail
R=/home/claude/pge-paid-wt
P=/tmp/claude-0/-home-claude/eddd235b-0eaf-52c4-824e-0534a3e36afd/scratchpad/paper
L=/tmp/claude-0/-home-claude/eddd235b-0eaf-52c4-824e-0534a3e36afd/scratchpad/refresh_logs
mkdir -p "$L"
cd "$R"
git fetch -q origin
git -c user.name="Jihan Mankani" -c user.email="jihanmankani2015@gmail.com" merge -q --no-edit origin/e6-posterior-detector \
  || { echo "MERGE CONFLICT"; exit 1; }
echo "[merge] $(git log --oneline -1)"
run() { echo "[start] $1 $(date +%H:%M)"; timeout 6000 python3 -m "$@" > "$L/$1.log" 2>&1; echo "[end] $1 rc=$? $(date +%H:%M)"; }
run eval.e11_confirmatory & run eval.e16_v7_confirm & wait
run eval.e18_joint_bootstrap & run eval.e17_strong_baselines & wait
run eval.e19_percommit & run eval.e20_judge & wait
if [ "${WITH_E15:-0}" = 1 ]; then run eval.e15_regimes --confirm --draws 5; fi
cd "$P"
python3 make_numbers.py "$R"
python3 fig_lc.py "$R" && python3 fig_ladder.py "$R"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|undefined" | head
pdfinfo main.pdf | grep Pages; pdftotext main.pdf - | python3 -c "import sys,re;print('chars',len(re.sub(r'\\s','',sys.stdin.read())))"
echo "[refresh] done $(date +%H:%M)"
