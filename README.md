# Probabilistic Digital Twin Framework for 3D Woven and Braided Composites

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation-based digital twin framework linking textile architecture, manufacturing defects, and structural performance of advanced composite materials. This repository contains the code used in the research study:

**Mahin, S. H. (2026).** *Probabilistic Digital Twin Framework for 3D Woven and Braided Composites: Linking Textile Architecture, Manufacturing Defects, and Structural Performance.*

---

## Overview

The framework integrates four core modelling modules to capture the influence of manufacturing variability on composite performance:

1. **Textile Architecture Model**  
   – Fiber volume fraction, binder density, waviness, braid angle, thickness.

2. **Manufacturing Process Model**  
   – Compaction pressure, resin flow rate, cure temperature deviation.

3. **Defect Prediction Model**  
   – Void fraction, resin‑rich regions, waviness amplification, defect severity index.

4. **Structural Performance Model**  
   – Undamaged compressive strength, Compression‑After‑Impact (CAI) strength, fatigue knockdown.

Uncertainty is propagated through a **Monte Carlo simulation** (typically 3000 samples), enabling sensitivity analysis, reliability assessment, and validation against experimental literature.

---

## Repository Structure

```
.
├── digital_twin_simulation.py   # Main simulation script
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── digital_twin_literature_results/   # Auto‑generated output folder
│   ├── figures/                  # PNG figures
│   ├── data/                     # CSV simulation results
│   ├── tables/                    # LaTeX tables
│   └── validation/                # JSON validation metrics
└── example_outputs/               # Sample outputs (optional)
```

---

## Requirements

- Python **3.9 or newer**
- Required libraries: `numpy`, `scipy`, `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

Execute the main script from the terminal:

```bash
python digital_twin_simulation.py
```

The script will:

1. Generate 3000 random architecture and process parameter sets.
2. Predict defect states and structural performance for each configuration.
3. Compute sensitivity (Pearson correlation) and reliability (pass rates).
4. Validate predicted CAI strength and void fraction against literature benchmarks.
5. Export all results (CSV, JSON, LaTeX tables) and save figures as PNG files in `digital_twin_literature_results/`.

---

## Output

After a successful run, the following outputs are available:

| Type          | Description |
|---------------|-------------|
| **Figures**   | `figure_void_vs_cai.png`, `figure_defect_vs_fatigue.png`, `figure_cai_distribution.png`, `figure_sensitivity_ranking.png`, `figure_reliability_distribution.png`, `figure_parameter_distributions.png`, `figure_defect_distributions.png` |
| **Data**      | `simulation_results.csv`, `summary_metrics.csv`, `validation_results.json` |
| **LaTeX tables** | `latex_tables.tex` (ready for inclusion in manuscripts) |

All outputs are placed in the folder `digital_twin_literature_results/`.

---

## Reproducibility

The code is designed to be fully reproducible. By default, the random seed is fixed (`SEED = 42`). Running the script without modifications will generate exactly the same data and figures presented in the associated paper.

---

## Citation

If you use this code in your research, please cite the original work:

```bibtex
@article{mahin2026digitaltwin,
  title   = {Lifecycle Digital Twin Framework for 3D Woven and Braided Composites: Linking Textile Architecture, Manufacturing Defects, and Structural Performance},
  author  = {Mahin, Shadat Hossen},
  year    = {2026},
  note    = {GitHub repository: \url{https://github.com/shadatmahin17/Composite}}
}
```

Direct link to repository: [Digital-twin-3d-composites](https://github.com/shadatmahin17/Digital-twin-3d-composites.git)

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Author

**Shadat Hossen Mahin**  
Department of Textile Engineering  
Research area: Aerospace Composite Structures  
2026
