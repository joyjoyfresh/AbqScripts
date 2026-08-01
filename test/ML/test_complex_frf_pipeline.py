# -*- coding: utf-8 -*-
"""复频响联合分析、代理训练和真实波重构的纯Python闭环测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
GENERAL = REPO / "Postprocess" / "General"
ML_DIR = REPO / "ML"
for directory in (GENERAL, ML_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analyze_complex_frf as analysis
import evaluate_complex_frf_quality as evaluation
import reconstruct_real_wave as reconstruction
import train_complex_frf_surrogate as training


def segments(s_values):
    labels = np.full(s_values.shape, "B", dtype="S1")
    labels[s_values <= 0.0] = b"A"
    labels[s_values >= 1.0] = b"C"
    return labels


def write_case(case_dir, case_id, slope_angle, thickness_ratio, velocity_ratio):
    case_dir.mkdir(parents=True)
    frequency = analysis.regular_grid(0.5, 10.0, 0.1)
    s_values = analysis.regular_grid(-4.0, 4.0, 0.05)
    f_grid, s_grid = np.meshgrid(frequency, s_values, indexing="ij")
    amplitude = (
        1.0
        + 0.004 * slope_angle * np.exp(-((f_grid - 3.0) / 1.5) ** 2) * np.exp(-((s_grid - 0.5) / 1.8) ** 2)
        + 0.20 * thickness_ratio * np.exp(-((f_grid - 5.0 * velocity_ratio) / 1.2) ** 2)
    )
    phase = (
        0.012 * slope_angle * s_grid
        + 0.10 * thickness_ratio * f_grid
        - 0.08 * velocity_ratio * f_grid * s_grid
    )
    G = amplitude * np.exp(1j * phase)
    H_total = G * (1.5 + 0.1j * frequency[:, None])
    record = "g1b_synthetic"
    prefix = "frf_%s_" % record
    np.savez_compressed(
        case_dir / "surface_results.npz",
        **{
            prefix + "frequency": frequency,
            prefix + "sgrid_s": s_values,
            prefix + "sgrid_segment": segments(s_values),
            prefix + "sgrid_H_surface_over_1D_h": G.T,
            prefix + "sgrid_H_surface_over_1D_h_valid_mask": np.ones(G.T.shape, dtype=bool),
            prefix + "sgrid_H_surface_h": H_total.T,
            prefix + "sgrid_H_surface_h_valid_mask": np.ones(H_total.T.shape, dtype=bool),
            prefix + "input_spectrum": np.exp(-((frequency - 5.0) / 4.0) ** 2).astype(complex),
        }
    )
    layers = [] if case_id.startswith("H") else [{
        "name": "surface", "thickness": 100.0 * thickness_ratio,
        "vs": 2000.0 * velocity_ratio, "density": 2125, "poisson_ratio": 0.35,
    }]
    config = {
        "geometry_cfg": {"slope_angle": slope_angle, "slope_height": 100.0},
        "material_cfg": {
            "bedrock": {"vs": 2000.0, "density": 2500, "poisson_ratio": 0.30},
            "layers": layers,
        },
    }
    (case_dir / "case_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    return config


class ComplexFrfPipelineTests(unittest.TestCase):
    def test_evaluation_ignores_nonfinite_frequency_weights(self):
        reference = np.ones((3, 2), dtype=complex)
        candidate = np.exp(1j * np.full((3, 2), 0.1))
        mask = np.ones((3, 2), dtype=bool)
        weight = np.asarray([1.0, np.nan, 0.5])
        self.assertTrue(np.isfinite(evaluation.weighted_complex_error(
            reference, candidate, mask, weight
        )))
        self.assertTrue(np.isfinite(evaluation.circular_phase_rmse_deg(
            reference, candidate, mask, weight
        )))

    def test_analysis_training_and_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            h_root = root / "H"
            p_root = root / "P"
            b_root = root / "B"
            write_case(h_root / "case-001-H001", "H001", 30.0, 0.0, 1.0)
            p_parameters = [
                (15.0, 0.2, 0.30), (15.0, 0.6, 0.45),
                (30.0, 0.2, 0.60), (30.0, 0.6, 0.75),
                (30.0, 1.0, 0.30), (45.0, 0.6, 0.60),
                (45.0, 1.0, 0.75), (45.0, 1.4, 0.45),
                (60.0, 1.0, 0.60), (60.0, 1.4, 0.30),
            ]
            last_config = None
            for index, values in enumerate(p_parameters, start=1):
                last_config = write_case(
                    p_root / ("case-%03d-P%03d" % (index, index)),
                    "P%03d" % index, *values,
                )
            for index, values in enumerate(((22.5, 0.4, 0.375), (52.5, 1.2, 0.675)), start=1):
                write_case(
                    b_root / ("case-%03d-B%03d" % (index, index)),
                    "B%03d" % index, *values,
                )

            analysis_output = root / "analysis"
            self.assertEqual(analysis.main([
                "--input-roots", str(h_root), str(p_root), str(b_root),
                "--output", str(analysis_output), "--figures", "none", "--strict",
            ]), 0)
            dataset = np.load(analysis_output / "complex_frf_dataset.npz", allow_pickle=False)
            try:
                self.assertEqual(dataset["G_h"].shape[1:], (96, 161))
                self.assertTrue(np.all(np.isfinite(dataset["G_h"].real)))
                self.assertIn("phase_unwrapped_rad", dataset.files)
                self.assertIn("group_delay_s", dataset.files)
            finally:
                dataset.close()

            model_output = root / "model"
            self.assertEqual(training.main([
                "--dataset", str(analysis_output / "complex_frf_dataset.npz"),
                "--output", str(model_output), "--folds", "2",
                "--pod-energy", "0.99", "--max-components", "2",
                "--minimum-valid-fraction", "1.0",
            ]), 0)
            model_path = model_output / "complex_frf_surrogate.pkl"
            self.assertTrue(model_path.is_file())
            self.assertTrue((model_output / "unseen_combination_metrics.csv").is_file())

            reference_time = np.arange(0.0, 8.0 + 0.02, 0.02)
            real_wave = np.sin(2.0 * np.pi * 1.2 * reference_time) * np.exp(-0.35 * reference_time)
            reference_path = root / "freefield_reference_real_wave.npz"
            np.savez_compressed(
                reference_path,
                time=reference_time,
                rock_acc_h=real_wave,
                one_d_left_acc_h=1.4 * real_wave,
                one_d_right_acc_h=1.1 * real_wave,
                record=np.asarray("real_wave"),
            )
            config_path = root / "case_config.json"
            config_path.write_text(json.dumps(last_config), encoding="utf-8")
            reconstruction_output = root / "reconstruction"
            self.assertEqual(reconstruction.main([
                "--model", str(model_path), "--reference", str(reference_path),
                "--case-config", str(config_path), "--output", str(reconstruction_output),
                "--tail-seconds", "1.0", "--period-count", "4", "--no-figures",
            ]), 0)
            result_path = reconstruction_output / "real_wave" / "reconstruction.npz"
            result = np.load(result_path, allow_pickle=False)
            try:
                self.assertEqual(result["reconstructed_acc_h"].shape[0], 161)
                self.assertTrue(np.all(np.isfinite(result["pga_reconstructed"])))
                self.assertTrue(np.all(result["pga_reconstructed"] > 0.0))
            finally:
                result.close()


if __name__ == "__main__":
    unittest.main()
