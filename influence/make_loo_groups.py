"""Create five ascending-score groups for the paper's removal experiment."""

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psi-csv", required=True)
    parser.add_argument("--n-groups", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with Path(args.psi_csv).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.n_groups < 2 or len(rows) < args.n_groups:
        raise ValueError("The number of groups must be between 2 and the number of scores")

    tokens = np.asarray([row["token"] for row in rows])
    scores = np.asarray([float(row["psi"]) for row in rows], dtype=np.float64)
    if len(set(tokens)) != len(tokens) or not np.isfinite(scores).all():
        raise ValueError("Scores must have unique tokens and finite values")
    groups = np.array_split(np.lexsort((tokens, scores)), args.n_groups)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keep_full.txt").write_text("\n".join(tokens) + "\n")

    with (output_dir / "groups.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group_id", "n_removed", "mean_psi", "min_psi", "max_psi"])
        for group_id, indices in enumerate(groups):
            removed = set(tokens[indices])
            kept = [token for token in tokens if token not in removed]
            (output_dir / f"keep_remove_g{group_id}.txt").write_text("\n".join(kept) + "\n")
            writer.writerow(
                [
                    group_id,
                    len(indices),
                    f"{scores[indices].mean():.8e}",
                    f"{scores[indices].min():.8e}",
                    f"{scores[indices].max():.8e}",
                ]
            )


if __name__ == "__main__":
    main()
