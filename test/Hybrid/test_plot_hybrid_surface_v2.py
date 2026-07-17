import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SCRIPT = Path(__file__).resolve().parents[2] / "Postprocess" / "Plot_Hybrid_surface_v2.py"
SPEC = importlib.util.spec_from_file_location("plot_hybrid_surface_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlotHybridSurfaceTests(unittest.TestCase):
    def test_smooth_curve_reduces_spatial_zigzag_without_filling_nan(self):
        values = np.array([1.0, 1.2, 0.8, 1.2, 0.8, 1.2, 1.0])
        smoothed = MODULE.smooth_curve(values, 5)

        self.assertLess(np.ptp(smoothed[1:-1]), np.ptp(values[1:-1]))
        self.assertTrue(np.isnan(MODULE.smooth_curve(np.full(7, np.nan), 5)).all())

    def test_draw_panel_accepts_read_only_pandas_array(self):
        df = pd.DataFrame({"s": [-1.0, 0.0, 1.0], "seg": ["A", "B", "C"], "TAF_h": [1.0, 1.1, 1.0]})
        df["TAF_h"].to_numpy().setflags(write=False)
        fig = plt.figure()

        MODULE.draw_single_panel(fig, "TAF_h", "blue", "TAF", df, df["s"].to_numpy(),
                                 1.0, 1.0, 1.0, ("A", "B", "C"), "s")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
