# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Abaqus FEA automation toolkit for seismic slope analysis. Studies how slope topography affects earthquake ground motion distribution using 2D plane-strain finite element models. All scripts are written in Python and run either standalone or inside the Abaqus Python environment.

## Code Style Requirement

**Every line of code must have a Chinese inline comment** explaining its purpose. This is a strict project requirement.

```python
x = 10  # 定义变量x并赋值为10
```

## Running Scripts

Scripts fall into two execution contexts:

**Standard Python** (wave preprocessing, post-processing, visualization):
```bash
python Wave/Seismic/scale_and_plot_v3.py
python Postprocess/Postprocess_PGA_v6.py
python Postprocess/Distribution_PGA_v5.py
```

**Abaqus environment** (FEA modeling and ODB extraction):
```bash
abaqus cae noGUI=Modeling/Single/VAB_oblique_noGUI_v13.py
abaqus python Postprocess/Postprocess_PGA_v6.py
```

**Batch parametric runs:**
```bash
python Batch/TAF_autorun_v1.py
python Batch/mesh_size_autorun_v2.py
```

## Recommended Workflow (5 Steps)

1. **Wave preprocessing** — `Wave/Seismic/scale_and_plot_v3.py`  
   Low-pass filter raw `.txt` seismic records, scale to target PGA (e.g. 0.30g), output FFT and response spectra. Processed files saved as `*_scaled.txt`.

2. **FEA modeling** — `Modeling/Single/VAB_oblique_noGUI_v13.py`  
   Builds 2D plane-strain slope model with viscoelastic artificial boundaries (VAB), applies oblique SV-wave incidence, runs headless via `abaqus cae noGUI=`.

3. **PGA extraction** — `Postprocess/Postprocess_PGA_v6.py`  
   Reads ODB output, extracts max acceleration from `TOP_SURFACE` nodes, normalizes x-coordinates as `x/h`, writes `PGA_job-XXX.csv`.

4. **Observation point extraction** (optional) — `Postprocess/Extract_OBS_v1.py`  
   Extracts time histories at crest (U), mid-slope (M), and toe (D) points.

5. **Visualization** — `Postprocess/Distribution_PGA_v5.py`, `Postprocess/Mesh_Convergence_v5.py`  
   Spatial PGA distribution plots across multiple input waves; mesh convergence evaluation.

## Architecture

**Data flow:**
```
Raw Seismic TXT → Scaled TXT → FEA Model (CAE/ODB) → CSV → Plots
```

**Directory roles:**
- `Wave/` — seismic signal preprocessing and synthetic wave generation
- `Modeling/` — Abaqus CAE model construction scripts
- `Postprocess/` — ODB result extraction and analysis
- `Nodes/` — node set creation utilities for Abaqus CAE
- `Batch/` — parametric batch runners and system utilities

**Versioning:** Scripts use `_v1`, `_v2`, ... suffixes for iterative refinement. Always use the highest version unless experimenting. Current latest: modeling `v13`, PGA extraction `v6`, distribution `v5`.

**Configuration pattern:** Parameters (file paths, geometric values, tolerances) are defined at the top of each script. Key geometric parameters: slope height `h`, slope angle `i` (degrees), model length `L = 8h`, lower substrate height `H_lower = 2h`. Recommended mesh size: `Vs / (10 * f_max)`.

**Batch automation:** `TAF_autorun_v1.py` creates `fuke-TAF-*` output folders and runs the full pipeline (modeling → extraction → visualization) for each parameter combination across 32 cases (h: 50–400 m, i: 30–60°, incident angle: 0–30°).
