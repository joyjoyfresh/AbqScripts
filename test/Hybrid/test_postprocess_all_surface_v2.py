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
