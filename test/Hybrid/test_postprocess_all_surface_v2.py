import csv
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import os

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "Postprocess" / "Hybrid" / "Postprocess_All_surface_v2.py"
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
