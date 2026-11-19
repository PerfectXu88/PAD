"""Report signed rank fidelity and exact tests for five-group removal."""

import argparse
import csv
from itertools import permutations
from pathlib import Path

import numpy as np


def rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rank(left)
    right_rank = rank(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.sqrt((left_rank**2).sum() * (right_rank**2).sum())
    return float((left_rank * right_rank).sum() / denominator) if denominator > 1e-12 else float("nan")


def exact_pvalues(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Enumerate the small-group null; the prespecified alternative is negative."""
    if not 2 <= len(left) <= 8 or len(left) != len(right):
        raise ValueError("Exact enumeration supports 2 to 8 paired groups")
    observed = spearman(left, right)
    if not np.isfinite(observed):
        raise ValueError("Rank correlation is undefined for constant group values")
    null = np.asarray([spearman(left, np.asarray(p)) for p in permutations(right)])
    return (float(np.mean(null <= observed + 1e-12)),
            float(np.mean(np.abs(null) >= abs(observed) - 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--epdms", required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.groups).open(newline="") as stream:
        groups = {int(row["group_id"]): row for row in csv.DictReader(stream)}
    with Path(args.epdms).open(newline="") as stream:
        evaluations = {
            int(row["group_id"]): float(row["epdms"])
            for row in csv.DictReader(stream)
        }
    if -1 not in evaluations:
        raise ValueError("The evaluation file must contain the full-pool baseline with group_id=-1")
    missing = set(groups) - set(evaluations)
    if missing:
        raise ValueError(f"Missing evaluations for groups: {sorted(missing)}")

    group_ids = sorted(groups)
    predicted = np.asarray([float(groups[group_id]["mean_psi"]) for group_id in group_ids])
    baseline = evaluations[-1]
    observed = np.asarray([evaluations[group_id] - baseline for group_id in group_ids])
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("Group values and evaluation scores must be finite")
    correlation = spearman(predicted, observed)
    negative_p, two_sided_p = exact_pvalues(predicted, observed)
    topk = min(max(args.topk, 1), len(group_ids))
    harmful = np.argsort(predicted)[:topk]
    precision = float((observed[harmful] > 0).mean())

    lines = [
        "# Group-removal rank fidelity",
        "",
        f"- Baseline EPDMS: {baseline:.4f}",
        f"- Spearman correlation: {correlation:.3f}",
        f"- Top-{topk} harmful precision: {precision:.2f}",
        f"- Exact p (negative alternative): {negative_p:.4f}",
        f"- Exact p (two-sided): {two_sided_p:.4f}",
        "- Expected direction: negative (removing higher-value data reduces utility).",
        "",
        "| Group | Mean psi | EPDMS after removal | Delta EPDMS |",
        "|---|---:|---:|---:|",
    ]
    for group_id in group_ids:
        lines.append(
            f"| {group_id} | {float(groups[group_id]['mean_psi']):.3e} | "
            f"{evaluations[group_id]:.4f} | {evaluations[group_id] - baseline:+.4f} |"
        )
    report = "\n".join(lines) + "\n"
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
