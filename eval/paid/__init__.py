"""Paid-run scripts (LLM API calls). Every script here takes --max-cost
(hard cap, enforced per call), records tokens and cost, writes JSONL via
pge.trajectory.save_jsonl, prints a [ledger] line for results/RESULTS.md,
and supports --mock for an offline dry run. See RUN_PAID.md."""
