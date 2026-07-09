import csv
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
