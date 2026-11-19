"""Merge per-scene calibration evaluations with their provenance manifest."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

SUMMARY_TOKENS = {
    "average_all_frames",
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
}


def latest_result(directory: Path) -> Path:
    candidates = sorted(directory.glob("*/*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation CSV found under {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames = []
    for source in (value.strip() for value in args.sources.split(",") if value.strip()):
        result_path = latest_result(Path(args.eval_root) / f"r002_calib_C_{source}")
        frame = pd.read_csv(result_path, dtype={"token": str})
        frame = frame[~frame["token"].isin(SUMMARY_TOKENS)].copy()
        if "valid" in frame and not frame["valid"].eq(True).all():
            raise ValueError("Calibration evaluation contains failed scenes; rerun before merging")
        frame["source_split"] = source
        frames.append(frame[["token", "score", "source_split"]])

    scores = pd.concat(frames, ignore_index=True).rename(columns={"score": "epdms"})
    manifest = pd.read_csv(args.manifest, dtype={"token": str})
    if manifest["token"].duplicated().any() or scores["token"].duplicated().any():
        raise ValueError("Duplicate calibration token")
    if not manifest["source_type"].eq("real").all() or not manifest["source_split"].eq("navtrain").all():
        raise ValueError("Calibration requires held-out real navtrain scenes")
    if set(scores.token) != set(manifest.token):
        raise ValueError("Evaluation must cover exactly the fixed calibration manifest")
    if not np.isfinite(scores.epdms.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite calibration outcomes")
    result = manifest.merge(scores[["token", "epdms"]], on="token", validate="one_to_one")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[calibration] wrote {len(result)} scores to {output_path}")


if __name__ == "__main__":
    main()
