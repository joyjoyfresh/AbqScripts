import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import os

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "Postprocess" / "Postprocess_All_surface_v2.py"
ABAQUS_CONSTANTS = types.ModuleType("abaqusConstants")
ODB_ACCESS = types.ModuleType("odbAccess")
ODB_ACCESS.openOdb = None
import sys
sys.modules["abaqusConstants"] = ABAQUS_CONSTANTS
sys.modules["odbAccess"] = ODB_ACCESS
SPEC = importlib.util.spec_from_file_location("postprocess_all_surface_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PostprocessAllSurfaceTests(unittest.TestCase):
    def test_complex_frf_preserves_amplitude_phase_and_mask(self):
        """复频响必须保留相位，旧 compute_H 只能是同掩码下的幅值视图。"""
        dt = 1.0 / 128.0
        time = np.arange(256, dtype=float) * dt
        frequency = 8.0
        phase = 0.4
        input_acc = np.cos(2.0 * np.pi * frequency * time)
        output_acc = 2.0 * np.cos(2.0 * np.pi * frequency * time + phase)

        freqs, transfer, valid, _ = MODULE.compute_complex_H(output_acc, input_acc, dt, fc=frequency)
        index = int(np.argmin(np.abs(freqs - frequency)))
        self.assertTrue(valid[index])
        self.assertAlmostEqual(abs(transfer[0, index]), 2.0, places=12)
        self.assertAlmostEqual(np.angle(transfer[0, index]), phase, places=12)

        old_freqs, amplitude = MODULE.compute_H(output_acc, input_acc, dt, fc=frequency)
        np.testing.assert_allclose(old_freqs, freqs[valid])
        np.testing.assert_allclose(amplitude, np.abs(transfer[:, valid]))

    def test_complex_frf_rejects_zero_reference_without_epsilon(self):
        freqs, transfer, valid, _ = MODULE.compute_complex_H(
            np.ones((2, 32)), np.zeros(32), 0.01, fc=4.0
        )

        self.assertEqual(freqs.shape, valid.shape)
        self.assertFalse(np.any(valid))
        self.assertTrue(np.all(np.isnan(transfer.real)))
        self.assertTrue(np.all(np.isnan(transfer.imag)))

    def test_explicit_frf_upper_bound_is_independent_from_damping_fc(self):
        """论文频带显式设为 10 Hz 时，不应再被 2 Hz 阻尼主频截到 5 Hz。"""
        dt = 1.0 / 128.0
        time = np.arange(512, dtype=float) * dt
        input_acc = np.cos(2.0 * np.pi * 8.0 * time)
        output_acc = 1.5 * input_acc

        legacy_freqs, _legacy_h, legacy_valid, _ = MODULE.compute_complex_H(
            output_acc, input_acc, dt, fc=2.0
        )
        freqs, transfer, valid, _ = MODULE.compute_complex_H(
            output_acc, input_acc, dt, fc=2.0, fmax_hz=10.0
        )

        self.assertLessEqual(float(np.max(legacy_freqs)), 5.0)
        self.assertGreater(float(np.max(freqs)), 9.5)
        index = int(np.argmin(np.abs(freqs - 8.0)))
        self.assertTrue(valid[index])
        self.assertAlmostEqual(abs(transfer[0, index]), 1.5, places=12)
        self.assertFalse(np.any(legacy_valid[np.abs(legacy_freqs - 8.0) < 0.1]))

    def test_tail_rms_ratio_exposes_unfinished_ringdown(self):
        quiet = np.r_[np.ones(90), np.zeros(10)]
        ringing = np.ones(100)

        quiet_stats = MODULE.tail_rms_ratio_stats(quiet, tail_fraction=0.10)
        ringing_stats = MODULE.tail_rms_ratio_stats(ringing, tail_fraction=0.10)

        self.assertAlmostEqual(quiet_stats["p95"], 0.0)
        self.assertAlmostEqual(ringing_stats["p95"], 1.0)

    def test_complex_frf_resamples_real_and_imaginary_parts(self):
        transfer = np.array([[1.0 + 0.0j], [3.0 + 2.0j]])
        aligned = MODULE.resample_H_matrix(
            transfer, np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0]), ["A", "B", "C"]
        )

        self.assertTrue(np.isnan(aligned[0, 0].real))
        self.assertAlmostEqual(aligned[1, 0].real, 2.0)
        self.assertAlmostEqual(aligned[1, 0].imag, 1.0)
        self.assertTrue(np.isnan(aligned[2, 0].real))

    def test_continuous_frf_fills_only_short_internal_corner_gap(self):
        """连续地表频响应补齐坡脚短缺口，但分段谱比默认仍保留缺口。"""
        transfer = np.array([
            [1.0 + 0.0j],
            [2.0 + 1.0j],
            [3.0 + 2.0j],
            [4.0 + 3.0j],
        ])
        s_nodes = np.array([0.90, 0.98, 1.02, 1.10])
        s_grid = np.array([0.90, 0.98, 1.00, 1.02, 1.10])
        segments = ["B", "B", "C", "C", "C"]

        segmented = MODULE.resample_H_matrix(
            transfer, s_nodes, s_grid, segments,
        )
        continuous = MODULE.resample_H_matrix(
            transfer, s_nodes, s_grid, segments, fill_short_gaps=True,
        )

        self.assertTrue(np.isnan(segmented[2, 0].real))
        self.assertAlmostEqual(continuous[2, 0].real, 2.5)
        self.assertAlmostEqual(continuous[2, 0].imag, 1.5)

    def test_matrix_csv_accepts_windows_nan_markers(self):
        """Windows写出的-nan(ind)等标记应保留为NaN，不能使整张FSAF矩阵读取失败。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "FSAF_1D_h_demo.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["f_Hz", "x=0.0", "x=1.0"])
                writer.writerow([0.5, "1.25", "-nan(ind)"])
                writer.writerow([1.0, "1.#QNAN", "2.5"])

            axis, xs, values = MODULE.read_H_csv_local(str(path))

        np.testing.assert_allclose(axis, [0.5, 1.0])
        np.testing.assert_allclose(xs, [0.0, 1.0])
        self.assertEqual(values.shape, (2, 2))
        self.assertAlmostEqual(values[0, 0], 1.25)
        self.assertTrue(np.isnan(values[0, 1]))
        self.assertTrue(np.isnan(values[1, 0]))
        self.assertAlmostEqual(values[1, 1], 2.5)

    def test_psa_and_rsaf_use_exact_reference_time_histories(self):
        """PSA 应保持线性缩放；RSAF 只使用配置的真实 rock/1D 参考。"""
        dt = 0.005
        time = np.arange(0.0, 4.0 + 0.5 * dt, dt)
        base = np.sin(2.0 * np.pi * 2.0 * time)
        xs = np.array([0.0, 1.0])
        acc_h = np.vstack([2.0 * base, 3.0 * base])
        acc_v = np.vstack([0.5 * base, 0.75 * base])

        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, values in (("rock", base), ("left", 2.0 * base), ("right", 1.5 * base)):
                path = Path(folder) / (name + ".txt")
                np.savetxt(str(path), np.column_stack([time, values]))
                paths[name] = str(path)
            case_cfg = {
                "run_cfg": {
                    "response_spectrum_cfg": {
                        "periods": [0.1, 0.2, 0.5, 1.0, 2.0],
                        "reference_files": {
                            "demo": {
                                "rock": paths["rock"],
                                "one_d_left": paths["left"],
                                "one_d_right": paths["right"],
                            }
                        },
                    }
                }
            }
            payload = MODULE.compute_response_spectrum_payload(
                "demo", xs, 0.5, time, acc_h, acc_v, dt, case_cfg
            )

        self.assertTrue(payload["quality"]["passed"])
        np.testing.assert_allclose(payload["RSAF_rock_h"][0], 2.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(payload["RSAF_rock_h"][1], 3.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(payload["RSAF_1D_h"][0], 1.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(payload["RSAF_1D_h"][1], 2.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(payload["URSAF_z"][0], 0.5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(payload["key_period"], [0.1, 0.2, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(payload["key_RSAF_1D_h"][1], 2.0, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(payload["RSAF_rock_valid_mask"]))

        freqs, _incident_h, incident_valid, _ = MODULE.compute_complex_H(acc_h, base, dt, fc=2.0)
        over_1d, over_1d_valid = MODULE.compute_side_reference_H(
            acc_h, xs, 0.5, 2.0 * base, 1.5 * base, dt, freqs, incident_valid, fc=2.0
        )
        frequency_index = int(np.argmin(np.abs(freqs - 2.0)))
        self.assertTrue(np.all(over_1d_valid[:, frequency_index]))
        np.testing.assert_allclose(np.abs(over_1d[:, frequency_index]), [1.0, 2.0], rtol=1e-12, atol=1e-12)

        missing = MODULE.compute_response_spectrum_payload(
            "demo", xs, 0.5, time, acc_h, acc_v, dt,
            {"run_cfg": {"response_spectrum_cfg": {"periods": [0.2, 1.0]}}},
        )
        self.assertFalse(missing["quality"]["passed"])
        self.assertTrue(np.all(np.isnan(missing["RSAF_rock_h"])))
        self.assertFalse(np.any(missing["RSAF_rock_valid_mask"]))

    def test_surface_metrics_adds_unified_vertical_definitions(self):
        xs = np.array([0.0, 10.0])
        a1 = np.array([[2.0, -1.0], [4.0, -2.0]])
        a2 = np.array([[0.5, -0.25], [1.0, -0.5]])
        taf_lr = {"left": (1.0, 0.0), "right": (2.0, 0.5)}

        rows = MODULE.surface_metrics(xs, a1, a2, 1.0, 2.0, 0.0, taf_lr, 5.0)

        self.assertTrue(np.isnan(rows[0]["TAF_v"]))
        self.assertAlmostEqual(rows[0]["VTR"], 0.25)
        self.assertAlmostEqual(rows[0]["UTAF_v"], 0.25)
        self.assertAlmostEqual(rows[0]["DUTAF_v"], 0.25)
        self.assertAlmostEqual(rows[1]["TAF_v"], 1.0)
        self.assertAlmostEqual(rows[1]["TAF_v_comp"], 1.0)
        self.assertAlmostEqual(rows[1]["UTAF_R"], 1.0)

    def test_raw_response_only_converts_x_to_s(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "surface_response_demo.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "TAF_h"])
                writer.writerows([[-10.0, 1.0], [0.0, 1.1], [20.0, 1.2], [30.0, 1.3]])

            record, data, s_values, suffix = MODULE.prepare_plot_data(
                str(path), 0.0, 20.0, 10.0, raw=True
            )

        self.assertEqual(record, "demo")
        self.assertEqual([row["TAF_h"] for row in data], [1.0, 1.1, 1.2, 1.3])
        np.testing.assert_allclose(s_values, [-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(suffix, "raw_s_")

    def test_nonuniform_components_share_the_raw_time_axis(self):
        """非等时步 ODB 的两个加速度分量必须重采样到同一时间轴。"""
        t_raw = np.array([0.0, 0.1, 0.2, 0.4])
        a1 = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, -1.0, -2.0, -4.0]])
        a2 = np.array([[0.0, 0.5, 1.0, 2.0], [0.0, -0.5, -1.0, -2.0]])

        t, a1_uniform, _ = MODULE.to_uniform(t_raw, a1)
        t_v, a2_uniform, _ = MODULE.to_uniform(t_raw, a2)

        np.testing.assert_allclose(t_v, t)
        self.assertEqual(a1_uniform.shape, a2_uniform.shape)
        rows = MODULE.surface_metrics(np.array([0.0, 1.0]), a1_uniform, a2_uniform,
                                      1.0, 1.0, 0.0, {"left": (1.0, 0.0)}, 2.0)
        self.assertEqual(len(rows), 2)

    def test_to_uniform_rejects_mismatched_time_and_signal_lengths(self):
        """防止重采样后的时间轴被误用于尚未重采样的另一分量。"""
        with self.assertRaisesRegex(ValueError, "时程列数"):
            MODULE.to_uniform(np.array([0.0, 0.1, 0.2]), np.array([[0.0, 1.0]]))

    def test_plot_window_uses_configured_observation_limits(self):
        """观测窗已配置时，绘图范围必须直接采用其实际 s 跨度。"""
        ctx = MODULE._resolve_s_context(
            {"geometry": {"x_crest": 0.0, "x_toe": 20.0, "H_minus_h": 10.0}},
            {"geometry_cfg": {"crest_window": 1.5, "toe_window": 2.5}},
        )

        self.assertEqual(ctx[3:], (1.5, 2.5))

    def test_npz_package_contains_tables_and_removes_temporary_outputs(self):
        """单工况数值产物应收敛为一个无 pickle 的 NPZ 包。"""
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                for name in ("surface_response_demo.csv", "sgrid_response_demo.csv"):
                    with open(name, "w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["s", "PGA_h"])
                        writer.writerow(["0.0", "1.25"])
                Path("surface_summary.json").write_text(json.dumps({"records": [{"record": "demo"}]}), encoding="utf-8")
                Path("sgrid_params.json").write_text(json.dumps({"N_A": 1}), encoding="utf-8")

                self.assertEqual(MODULE.write_surface_npz({"case": "demo"}, {"cfg": 1}), 2)

                package = np.load("surface_results.npz")
                try:
                    self.assertEqual(int(package["schema_version"]), 2)
                    manifest = json.loads(MODULE._npz_bytes(package["manifest_json"].item()).item().decode("utf-8"))
                    self.assertEqual([item["name"] for item in manifest], ["sgrid_response_demo.csv", "surface_response_demo.csv"])
                finally:
                    package.close()
                self.assertFalse(Path("surface_response_demo.csv").exists())
                self.assertFalse(Path("sgrid_response_demo.csv").exists())
                self.assertFalse(Path("surface_summary.json").exists())
                os.chdir(old_cwd)  # Windows 不允许删除仍作为当前目录的临时文件夹
        finally:
            os.chdir(old_cwd)

    def test_npz_package_preserves_complex_frf_and_valid_masks(self):
        """规范 NPZ 必须无损保存复数 H、布尔掩码和反应谱数组。"""
        spectral = {
            "demo": {
                "frf": {
                    "frequency": np.array([0.0, 1.0]),
                    "H_surface_h": np.array([[np.nan + 1j * np.nan, 2.0 + 0.5j]]),
                    "valid_mask": np.array([False, True]),
                    "quality_json": json.dumps({"status": "passed"}),
                },
                "rsa": {
                    "period": np.array([0.2, 1.0]),
                    "RSAF_rock_h": np.array([[1.2, 1.5]]),
                    "RSAF_rock_valid_mask": np.array([[True, True]]),
                    "quality_json": json.dumps({"status": "passed"}),
                },
            }
        }
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                MODULE.write_surface_npz({}, {}, spectral_results=spectral)
                package = np.load("surface_results.npz", allow_pickle=False)
                try:
                    self.assertTrue(np.iscomplexobj(package["frf_demo_H_surface_h"]))
                    self.assertEqual(package["frf_demo_valid_mask"].dtype, np.dtype(bool))
                    self.assertEqual(package["rsa_demo_RSAF_rock_h"].shape, (1, 2))
                    self.assertAlmostEqual(package["frf_demo_H_surface_h"][0, 1].imag, 0.5)
                finally:
                    package.close()
                os.chdir(old_cwd)
        finally:
            os.chdir(old_cwd)

    def test_main_fails_loudly_when_odb_extraction_fails(self):
        """任一 ODB 提取失败必须返回非零，避免批处理误清理唯一的诊断 ODB。"""
        old_open_odb = MODULE.openOdb
        old_load_json = MODULE._load_json
        old_glob = MODULE.glob.glob
        old_process = MODULE.process_one_odb
        old_resample = MODULE.resample_outputs
        old_plot = MODULE.plot_results
        old_log_step = MODULE.log_step
        old_cwd = os.getcwd()
        calls = []
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                MODULE.openOdb = object()
                MODULE._load_json = lambda path: {}
                MODULE.glob.glob = lambda pattern: ["job-demo.odb"] if pattern == "job-*.odb" else []
                MODULE.process_one_odb = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("模拟提取失败"))
                MODULE.resample_outputs = lambda *args, **kwargs: calls.append("resample")
                MODULE.plot_results = lambda *args, **kwargs: calls.append("plot")
                MODULE.log_step = lambda logger=None, message=None, *args: (logger or object())
                with self.assertRaises(SystemExit) as raised:
                    MODULE.main()
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(calls, [])
                self.assertTrue(Path("surface_summary.json").is_file())
                os.chdir(old_cwd)  # Windows 不允许删除仍作为当前目录的临时文件夹
        finally:
            os.chdir(old_cwd)
            MODULE.openOdb = old_open_odb
            MODULE._load_json = old_load_json
            MODULE.glob.glob = old_glob
            MODULE.process_one_odb = old_process
            MODULE.resample_outputs = old_resample
            MODULE.plot_results = old_plot
            MODULE.log_step = old_log_step


if __name__ == "__main__":
    unittest.main()
