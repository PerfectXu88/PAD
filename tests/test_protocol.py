"""Deterministic regressions for PAD's calibration, selection, and influence."""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from influence.protocol import normalized_advantage
from influence.analyze_spearman import exact_pvalues, spearman
from selection.select_synthetic import main as select_main
from selection.synthetic_pool import enumerate_synthetic_pool


def calibration(scores):
    return [dict(token=str(i), epdms=s, source_type="real", source_split="navtrain")
            for i, s in enumerate(scores)]


class CalibrationTests(unittest.TestCase):
    def test_global_population_normalization(self):
        rows = calibration([0, 20, 80, 100])
        for i, row in enumerate(rows):
            row["round"] = i // 2
        expected = (np.array([0, 20, 80, 100]) - 50) / (np.sqrt(1700) + 1e-8)
        np.testing.assert_allclose(normalized_advantage(rows), expected)

    def test_constant_outcomes(self):
        np.testing.assert_array_equal(normalized_advantage(calibration([9, 9])), [0, 0])

    def test_forbidden_sources(self):
        for field, value in [("source_type", "pdm"), ("source_split", "navtest"),
                             ("source_split", "navhard")]:
            rows = calibration([1, 2])
            rows[0][field] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalized_advantage(rows)

    def test_invalid_outcomes_and_duplicate_tokens(self):
        rows = calibration([1, 2])
        rows[1]["token"] = rows[0]["token"]
        for invalid in [[], rows, calibration([1, float("nan")]), calibration([float("inf")])]:
            with self.assertRaises(ValueError):
                normalized_advantage(invalid)


class RemovalTests(unittest.TestCase):
    def test_paper_direction_and_exact_pvalues(self):
        x, y = np.arange(5), np.array([3, 4, 2, 1, 0])
        self.assertAlmostEqual(spearman(x, y), -0.9)
        negative, two_sided = exact_pvalues(x, y)
        self.assertAlmostEqual(negative, 5 / 120)
        self.assertAlmostEqual(two_sided, 10 / 120)

    def test_opposite_direction(self):
        self.assertEqual(exact_pvalues(np.arange(5), np.arange(5))[0], 1.0)

    def test_ties_and_constant_values(self):
        self.assertAlmostEqual(spearman(np.array([0, 0, 1]), np.array([2, 2, 1])), -1)
        with self.assertRaises(ValueError):
            exact_pvalues(np.zeros(5), np.arange(5))


class CalibrationMergeTests(unittest.TestCase):
    def test_exact_manifest_and_failed_scene_rejection(self):
        import pandas as pd
        from calibration.merge_calibration_epdms import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "r002_calib_C_navtrain/2026.11.19/scores.csv"
            result.parent.mkdir(parents=True)
            manifest = root / "manifest.csv"
            pd.DataFrame(calibration([0, 0])).drop(columns="epdms").to_csv(manifest, index=False)
            argv = ["merge", "--eval-root", directory, "--sources", "navtrain",
                    "--manifest", str(manifest), "--out", str(root / "merged.csv")]
            records = [{"token": "0", "score": 0.8, "valid": True},
                       {"token": "1", "score": 0.9, "valid": True},
                       {"token": "average_all_frames", "score": 0.85, "valid": True}]
            pd.DataFrame(records).to_csv(result, index=False)
            with patch.object(sys, "argv", argv):
                main()
            np.testing.assert_allclose(pd.read_csv(root / "merged.csv").epdms, [0.8, 0.9])
            records[0]["valid"] = False
            pd.DataFrame(records).to_csv(result, index=False)
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                main()
            pd.DataFrame(records[1:]).to_csv(result, index=False)
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                main()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, NAVSIM_DEVKIT_ROOT=str(self.root / "runtime"),
                         NAVSIM_EXP_ROOT=str(self.root / "exp"))
        env.start()
        self.addCleanup(env.stop)
        config = self.root / "runtime/navsim/planning/script/config/training"
        config.mkdir(parents=True)
        (config / "default_log_split_synthetic_reation_pdm_v1.0-0.yaml").write_text(
            yaml.safe_dump({"train_logs": ["log_a"]}))
        self.cache = self.root / "exp/cache/navtrain_reaction_pdm_v1.0-0_cache/log_a"
        for token in ["a", "b", "c", "d"]:
            (self.cache / token).mkdir(parents=True)
        self.scores, self.out = self.root / "psi.csv", self.root / "selected.txt"

    def select(self, rows):
        with self.scores.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["token", "psi"])
            writer.writerows(rows)
        argv = ["select", "--method", "psi", "--experts", "pdm", "--rounds", "0",
                "--fraction", "0.75", "--psi-csv", str(self.scores), "--out", str(self.out)]
        with patch.object(sys, "argv", argv):
            select_main()
        return self.out.read_text().splitlines()

    def test_budget_includes_negative_scores_and_stable_ties(self):
        self.assertEqual(self.select([("d", -9), ("c", -2), ("b", -2), ("a", 3)]),
                         ["a", "b", "c"])

    def test_incomplete_duplicate_nonfinite_scores(self):
        valid = [("a", 4), ("b", 3), ("c", 2), ("d", 1)]
        for rows in [valid[:-1], valid + [("a", 4)], valid[:-1] + [("d", "nan")],
                     valid + [("foreign", 0)]]:
            with self.assertRaises(ValueError):
                self.select(rows)

    def test_missing_log(self):
        self.cache.rename(self.cache.with_name("moved"))
        with self.assertRaises(FileNotFoundError):
            enumerate_synthetic_pool(["pdm"], [0])

    def test_duplicate_candidates(self):
        with self.assertRaises(ValueError):
            enumerate_synthetic_pool(["pdm", "pdm"], [0])


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for gradient and geometry checks")
class InfluenceTests(unittest.TestCase):
    def test_regularized_solve(self):
        from influence.common import solve_influence_scores
        h = torch.tensor([[1., 0.], [0., 2.]])
        c = torch.tensor([[1., 2.], [3., 0.]])
        actual = solve_influence_scores(h, c, torch.eye(2), np.array([1., -1.]), 0.5)
        np.testing.assert_allclose(actual, [-2., 0.8], rtol=1e-6)

    def test_projection_reuses_same_map(self):
        from influence.common import project_gradient
        a, b = torch.tensor([1., 2., 3.]), torch.tensor([3., 1., 0.])
        project = lambda x: project_gradient(x, 8, 0, 2)
        torch.testing.assert_close(project(a), project(a))
        torch.testing.assert_close(project(a + b), project(a) + project(b))

    def test_unused_parameter_gradient_is_zero(self):
        from influence.common import compute_projected_gradient, project_gradient
        a = torch.nn.Parameter(torch.tensor([2.]))
        b = torch.nn.Parameter(torch.tensor([9.]))
        actual = compute_projected_gradient((a * a).sum(), [a, b], 8, 0, 2)
        torch.testing.assert_close(actual, project_gradient(torch.tensor([4., 0.]), 8, 0, 2))
