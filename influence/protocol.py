"""Calibration protocol from the PAD paper, independent of model dependencies."""
import numpy as np


def normalized_advantage(rows, delta: float = 1e-8) -> np.ndarray:
    """Global population z-score: main paper Eq. (9)."""
    if not rows or delta <= 0:
        raise ValueError("Calibration rows and positive delta are required")
    tokens = [row["token"] for row in rows]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Calibration tokens must be unique")
    for row in rows:
        if row.get("source_type") != "real":
            raise ValueError("Calibration must contain only held-out real scenes")
        if row.get("source_split") not in {"navtrain", "calib_C_navtrain"}:
            raise ValueError("Calibration must come from navtrain, never evaluation splits")
    scores = np.asarray([float(row["epdms"]) for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Calibration outcomes must all be finite")
    return (scores - scores.mean()) / (scores.std(ddof=0) + delta)

