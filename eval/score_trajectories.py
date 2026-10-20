"""score_trajectories — offline detector evaluation over cached LLM trajectories.

Loads JSONL trajectory caches (data/trajectories/<domain>/<model>_<seed>.jsonl),
rebuilds each session's reasoning DAG offline (dispatch by traj.domain), splits
BY task_id (hash-based — the same task can appear in multiple caches under
different model/seed, so row-level splits would leak), fits the supervised
detectors and the five graphon variants on the train split, and scores the test
split with every detector: AUROC (+ stratified bootstrap CI) and AUPRC.

Also emits a calibration report: temperature scaling (pge/calibration.py) fit
on train-split per-turn (probability, signal-outcome) pairs, ECE before/after
on the test split's pairs. Signals are detector-visible inputs — non-circular.

Optional baselines (pge/baselines/{semantic_entropy,selfcheckgpt,cove,gcn})
are imported defensively: absent modules and detectors whose constructors
require an llm_fn are skipped with a notice, never a crash. No API calls.

Usage:
    python3 -m eval.score_trajectories --exp e2_dbbench_main \
        --paths data/trajectories/dbbench_v2/*.jsonl \
        [--train-frac 0.7] [--seeds 42 123 456] [--blocks 6]

Output: results/<exp>/{summary.txt, seed<k>.npz}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc, bootstrap_auroc_ci
from pge.baselines.base import TrajectoryScorer
from pge.baselines.confidence import (
    HighConfContradicted,
    MaxConfidence,
    MeanConfidence,
    NegCumulativeEvidence,
    NegativeSignalFraction,
)
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.structural import (
    CascadeLift,
    ConfidentCascadeScore,
    NegNegEdgeFraction,
)
from pge.benchmarks import dbbench_v2, healthcare_llmpatient, healthcare_v2, travel_v1
from pge.calibration import apply_temperature, fit_temperature
from pge.graphon import GraphonEstimator
from pge.trajectory import Trajectory, load_jsonl

REPO = Path(__file__).resolve().parent.parent

DEFAULT_EXP = "e2_dbbench_main"
DEFAULT_PATHS = ["data/trajectories/dbbench_v2/*.jsonl"]
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_BLOCKS = 6

GRAPHON_VARIANTS = [
    ("position", "abs"),
    ("degree", "abs"),
    ("spectral", "abs"),
    ("feature", "abs"),
    ("feature", "signed"),
]
MIN_CLEAN_TRAIN = 5  # below this, graphons fall back to all train DAGs

DAG_BUILDERS = {
    "dbbench_v2": dbbench_v2.dag_from_trajectory,
    "healthcare_v2": healthcare_v2.dag_from_trajectory,
    "healthcare_llmpatient": healthcare_llmpatient.dag_from_trajectory,
    "travel_v1": travel_v1.dag_from_trajectory,
}

# Optional baseline modules another workstream may add concurrently.
OPTIONAL_BASELINE_MODULES = ("semantic_entropy", "selfcheckgpt", "cove", "gcn")


# ------------------------------------------------------------------ loading


def load_trajectories(path_specs: list[str]) -> list[Trajectory]:
    """Load all trajectories from the given paths (glob patterns ok)."""
    files: list[str] = []
    for spec in path_specs:
        hits = sorted(glob.glob(spec))
        if hits:
            files.extend(hits)
        elif Path(spec).exists():
            files.append(spec)
    trajs: list[Trajectory] = []
    for f in files:
        trajs.extend(load_jsonl(f))
    return trajs


def build_dag(traj: Trajectory):
    """Rebuild the reasoning DAG offline, dispatched by domain."""
    builder = DAG_BUILDERS.get(traj.domain)
    if builder is None:
        raise ValueError(f"no DAG builder for domain '{traj.domain}'")
    return builder(traj)


# ------------------------------------------------------------------ splitting


def split_task_ids(
    task_ids, train_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    """Hash-based split over UNIQUE task_ids (never rows): a task appearing in
    several caches lands entirely in one side. Deterministic in (seed, ids)."""
    unique = sorted(set(task_ids))
    keyed = sorted(
        unique,
        key=lambda tid: hashlib.md5(f"{seed}:{tid}".encode()).hexdigest(),
    )
    n_train = int(round(train_frac * len(keyed)))
    return set(keyed[:n_train]), set(keyed[n_train:])


# ------------------------------------------------------------------ metrics


def average_precision(scores, labels) -> float:
    """AUPRC via the rank formula: AP = (1/n_pos) * sum_{k: y_k=1} P@k."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    precision_at_k = tp / (np.arange(len(y)) + 1)
    return float((precision_at_k * y).sum() / n_pos)


def ece_10bin(probs, outcomes, n_bins: int = 10) -> float:
    """Standard 10-bin ECE: |bin confidence - bin accuracy|, count-weighted."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == n_bins - 1:
            mask |= probs == 1.0
        if not mask.any():
            continue
        ece += (mask.sum() / len(probs)) * abs(
            probs[mask].mean() - outcomes[mask].mean()
        )
    return float(ece)


# ------------------------------------------------------------ calibration


def turn_calibration_pairs(trajs) -> tuple[np.ndarray, np.ndarray]:
    """Per-turn (probability, outcome) pairs from the aggregate signal:
    outcome 1 if sum(signals) > 0, 0 if < 0; zero-signal turns skipped.
    Detector-input-only data — never touches eval_label."""
    probs, outcomes = [], []
    for traj in trajs:
        for turn in traj.turns:
            s = sum(turn.signals.values())
            if s == 0:
                continue
            probs.append(turn.probability)
            outcomes.append(1 if s > 0 else 0)
    return np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=int)


def calibration_report(train_trajs, test_trajs) -> dict:
    p_tr, y_tr = turn_calibration_pairs(train_trajs)
    p_te, y_te = turn_calibration_pairs(test_trajs)
    T = fit_temperature(p_tr, y_tr)
    return {
        "T": float(T),
        "n_train_pairs": int(len(p_tr)),
        "n_test_pairs": int(len(p_te)),
        "ece_before": ece_10bin(p_te, y_te),
        "ece_after": ece_10bin(apply_temperature(p_te, T), y_te),
    }


# ------------------------------------------------------------ detector zoo


def _optional_detectors() -> tuple[list[TrajectoryScorer], list[str]]:
    """No-arg-constructible TrajectoryScorers from optional baseline modules.
    Missing modules and llm_fn-requiring constructors are skipped, not fatal."""
    dets: list[TrajectoryScorer] = []
    notes: list[str] = []
    for mod_name in OPTIONAL_BASELINE_MODULES:
        try:
            mod = importlib.import_module(f"pge.baselines.{mod_name}")
        except ImportError:
            notes.append(f"optional baseline module '{mod_name}' absent; skipped")
            continue
        except Exception as e:  # broken concurrent checkout must not kill the run
            notes.append(f"optional module '{mod_name}' failed to import ({e}); skipped")
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                not issubclass(cls, TrajectoryScorer)
                or inspect.isabstract(cls)
                or cls.__module__ != mod.__name__
            ):
                continue
            try:
                dets.append(cls())
            except TypeError:
                notes.append(
                    f"{mod_name}.{cls.__name__} needs constructor args "
                    "(llm_fn?); skipped"
                )
            except Exception as e:
                notes.append(f"{mod_name}.{cls.__name__} failed to init ({e}); skipped")
    return dets, notes


def build_detectors(core_only: bool = False) -> tuple[list[TrajectoryScorer], list[str]]:
    """Zero-shot + supervised detectors. Graphons are handled separately
    (they fit on clean-only DAGs, not the labeled train split)."""
    dets: list[TrajectoryScorer] = [
        MeanConfidence(),
        MaxConfidence(),
        NegCumulativeEvidence(),
        NegativeSignalFraction(),
        HighConfContradicted(),
        NegNegEdgeFraction(),
        CascadeLift(),
        ConfidentCascadeScore(),
        DagFeatureClassifier(),
    ]
    notes: list[str] = []
    if not core_only:
        extra, notes = _optional_detectors()
        dets.extend(extra)
    return dets, notes


# ------------------------------------------------------------------ per seed


def run_seed(
    trajs: list[Trajectory],
    dags: list,
    seed: int,
    train_frac: float,
    blocks: int,
    n_boot: int = 1000,
    core_only: bool = False,
) -> dict:
    task_ids = [t.task_id for t in trajs]
    train_ids, test_ids = split_task_ids(task_ids, train_frac, seed)
    tr = [i for i, t in enumerate(trajs) if t.task_id in train_ids]
    te = [i for i, t in enumerate(trajs) if t.task_id in test_ids]

    labels = np.array([t.eval_label for t in trajs], dtype=int)
    y_tr, y_te = labels[tr], labels[te]
    notes: list[str] = []

    scores: dict[str, np.ndarray] = {}  # detector name -> test scores

    # Graphon variants: fit on CLEAN train DAGs only (fallback: all train).
    clean_tr_dags = [dags[i] for i in tr if labels[i] == 0]
    graphon_fit_dags = clean_tr_dags
    if len(clean_tr_dags) < MIN_CLEAN_TRAIN:
        notes.append(
            f"seed {seed}: only {len(clean_tr_dags)} clean train DAGs "
            f"(<{MIN_CLEAN_TRAIN}); graphons fit on all {len(tr)} train DAGs"
        )
        graphon_fit_dags = [dags[i] for i in tr]
    for method, weighting in GRAPHON_VARIANTS:
        name = f"graphon_{method}|{weighting}"
        try:
            est = GraphonEstimator(
                B=blocks, block_method=method, weighting=weighting, normalize=True
            ).fit(graphon_fit_dags)
            scores[name] = np.array([-est.log_likelihood(dags[i]) for i in te])
        except Exception as e:
            notes.append(f"{name} failed ({e}); skipped")

    # Zero-shot + supervised detectors: uniform fit(train) then score(test)
    # (fit is a no-op for zero-shot methods).
    detectors, det_notes = build_detectors(core_only=core_only)
    notes.extend(det_notes)
    tr_trajs = [trajs[i] for i in tr]
    tr_dags = [dags[i] for i in tr]
    for det in detectors:
        try:
            det.fit(tr_trajs, tr_dags, y_tr)
            scores[det.name] = np.array(
                [det.score(trajs[i], dags[i]) for i in te]
            )
        except Exception as e:
            notes.append(f"{det.name} failed ({e}); skipped")

    # Metrics per detector.
    metrics: dict[str, tuple[float, float, float, float]] = {}
    for name, s in scores.items():
        pt, lo, hi = bootstrap_auroc_ci(s, y_te, n_resamples=n_boot, seed=seed)
        metrics[name] = (pt, lo, hi, average_precision(s, y_te))

    calib = calibration_report(tr_trajs, [trajs[i] for i in te])

    return {
        "seed": seed,
        "train_idx": tr,
        "test_idx": te,
        "labels_test": y_te,
        "task_ids_test": [trajs[i].task_id for i in te],
        "domains_test": [trajs[i].domain for i in te],
        "dag_sizes_test": [len(dags[i]) for i in te],
        "scores": scores,
        "metrics": metrics,
        "calibration": calib,
        "notes": notes,
        "hall_rate_test": float(y_te.mean()) if len(y_te) else float("nan"),
        "hall_rate_all": float(labels.mean()) if len(labels) else float("nan"),
    }


# ------------------------------------------------------------------ report


def _ranked_lines(metrics: dict) -> list[str]:
    lines = []
    for name, (pt, lo, hi, ap) in sorted(
        metrics.items(), key=lambda kv: (-(kv[1][0]) if np.isfinite(kv[1][0]) else 1e9)
    ):
        lines.append(
            f"  {name:28s} AUROC {pt:.3f}  [{lo:.3f}, {hi:.3f}]   AUPRC {ap:.3f}"
        )
    return lines


def summarize(results: list[dict], exp: str, args_str: str) -> str:
    lines = [f"# {exp} — {args_str}", ""]

    for res in results:
        lines.append(
            f"## seed {res['seed']}  (train n={len(res['train_idx'])}, "
            f"test n={len(res['test_idx'])}, hallucination rate "
            f"{res['hall_rate_test']:.2f} test / {res['hall_rate_all']:.2f} overall)"
        )
        lines.extend(_ranked_lines(res["metrics"]))
        c = res["calibration"]
        lines.append(
            f"  calibration: T={c['T']:.3f}  ECE {c['ece_before']:.4f} -> "
            f"{c['ece_after']:.4f}  (train pairs={c['n_train_pairs']}, "
            f"test pairs={c['n_test_pairs']})"
        )
        for note in res["notes"]:
            lines.append(f"  [note] {note}")
        lines.append("")

    # Mean over seeds.
    all_names = sorted({n for r in results for n in r["metrics"]})
    lines.append(f"## mean over seeds ({', '.join(str(r['seed']) for r in results)})")
    means = {}
    for name in all_names:
        pts = [r["metrics"][name][0] for r in results if name in r["metrics"]]
        aps = [r["metrics"][name][3] for r in results if name in r["metrics"]]
        means[name] = (float(np.nanmean(pts)), float(np.nanmean(aps)), len(pts))
    for name, (m_auroc, m_ap, k) in sorted(means.items(), key=lambda kv: -kv[1][0]):
        lines.append(
            f"  {name:28s} AUROC {m_auroc:.3f}   AUPRC {m_ap:.3f}   ({k} seeds)"
        )
    lines.append("")

    # Per-domain breakdown (only when multiple domains present).
    domains = sorted({d for r in results for d in r["domains_test"]})
    if len(domains) > 1:
        lines.append("## per-domain AUROC (mean over seeds)")
        for dom in domains:
            lines.append(f"### {dom}")
            per_det: dict[str, list[float]] = {}
            for r in results:
                mask = np.array([d == dom for d in r["domains_test"]])
                y = r["labels_test"][mask]
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                for name, s in r["scores"].items():
                    per_det.setdefault(name, []).append(auroc(s[mask], y))
            for name, vals in sorted(
                per_det.items(), key=lambda kv: -float(np.nanmean(kv[1]))
            ):
                lines.append(f"  {name:28s} AUROC {float(np.nanmean(vals)):.3f}")
        lines.append("")

    return "\n".join(lines)


def save_seed_npz(out_dir: Path, res: dict) -> None:
    payload = {
        "seed": np.array(res["seed"]),
        "labels": res["labels_test"],
        "task_ids": np.array(res["task_ids_test"], dtype=str),
        "domains": np.array(res["domains_test"], dtype=str),
        "dag_sizes": np.array(res["dag_sizes_test"], dtype=int),
        "ece_before": np.array(res["calibration"]["ece_before"]),
        "ece_after": np.array(res["calibration"]["ece_after"]),
        "temperature": np.array(res["calibration"]["T"]),
    }
    for name, s in res["scores"].items():
        payload[f"score_{name}"] = s
    for name, m in res["metrics"].items():
        payload[f"metric_{name}"] = np.array(m)  # (auroc, lo, hi, auprc)
    np.savez(out_dir / f"seed{res['seed']}.npz", **payload)


# -------------------------------------------------------------------- main


def run_experiment(
    exp: str,
    paths: list[str],
    seeds=DEFAULT_SEEDS,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    blocks: int = DEFAULT_BLOCKS,
    n_boot: int = 1000,
    core_only: bool = False,
    out_root: Path | str | None = None,
) -> str:
    trajs = load_trajectories(paths)
    if not trajs:
        raise SystemExit(f"no trajectories found under: {paths}")
    dags = [build_dag(t) for t in trajs]

    out_dir = (Path(out_root) if out_root else REPO / "results") / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        res = run_seed(
            trajs, dags, seed, train_frac, blocks, n_boot=n_boot, core_only=core_only
        )
        results.append(res)
        save_seed_npz(out_dir, res)

    args_str = (
        f"n={len(trajs)} trajectories, {len(set(t.task_id for t in trajs))} unique "
        f"tasks, seeds={tuple(seeds)}, train_frac={train_frac}, B={blocks}"
    )
    report = summarize(results, exp, args_str)
    (out_dir / "summary.txt").write_text(report + "\n")
    print(report)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default=DEFAULT_EXP)
    ap.add_argument("--paths", nargs="+", default=DEFAULT_PATHS)
    ap.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--core-only",
        action="store_true",
        help="skip optional baseline modules (tests / reproducibility)",
    )
    ap.add_argument("--out-root", default=None, help="override results/ root (tests)")
    args = ap.parse_args(argv)

    run_experiment(
        exp=args.exp,
        paths=args.paths,
        seeds=args.seeds,
        train_frac=args.train_frac,
        blocks=args.blocks,
        n_boot=args.n_boot,
        core_only=args.core_only,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
