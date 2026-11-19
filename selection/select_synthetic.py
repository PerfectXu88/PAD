"""Select a fixed-budget synthetic subset and write a token allowlist."""

import argparse
import csv
import math
import random
from pathlib import Path

from selection.synthetic_pool import enumerate_synthetic_pool, parse_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["random", "psi"], default="random")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experts", default="pdm,recovery")
    parser.add_argument("--rounds", default="0-4")
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--psi-csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    pool = enumerate_synthetic_pool(
        [expert.strip() for expert in args.experts.split(",") if expert.strip()],
        parse_rounds(args.rounds),
        cache_prefix=args.cache_prefix,
    )
    tokens = sorted(pool)
    if not tokens:
        raise ValueError("The synthetic pool is empty")
    budget = max(1, round(args.fraction * len(tokens)))

    if args.method == "random":
        selected = random.Random(args.seed).sample(tokens, budget)
    else:
        if not args.psi_csv:
            raise ValueError("--psi-csv is required for method=psi")
        with Path(args.psi_csv).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
            scores = {row["token"]: float(row["psi"]) for row in rows}
        if len(scores) != len(rows) or not all(math.isfinite(v) for v in scores.values()):
            raise ValueError("Scores must have unique tokens and finite values")
        if set(scores) != set(tokens):
            raise ValueError("Score file must cover exactly the candidate pool")
        selected = sorted(tokens, key=scores.__getitem__, reverse=True)[:budget]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + "\n")
    print(f"[selection] wrote {len(selected)} of {len(tokens)} tokens to {output_path}")


if __name__ == "__main__":
    main()
