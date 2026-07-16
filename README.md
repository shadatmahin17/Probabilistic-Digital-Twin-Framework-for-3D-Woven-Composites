# Probabilistic Simulation Framework for 3D Woven Composites: Linking Architecture, Defects, and Structural Performance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21220844.svg)](https://doi.org/10.5281/zenodo.21220844)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A computationally efficient, open-source framework for simulating the effects of textile architecture and manufacturing defects on the structural performance of 3D woven composites. This repository contains the complete codebase for the research study:

> Mahin, S. H., & Islam, M. T. (2026). *Probabilistic Simulation Framework for 3D Woven Composites: Linking Architecture, Defects, and Structural Performance.*
> [https://doi.org/10.5281/zenodo.21220844](https://doi.org/10.5281/zenodo.21220844)

---

## 📌 Overview

This framework integrates four core modelling modules to capture the influence of manufacturing variability on composite performance:

| Module | Description |
| :--- | :--- |
| **Textile Architecture** | Fibre volume fraction, binder density, waviness, braid angle, thickness |
| **Manufacturing Process** | Compaction pressure, resin flow rate, cure temperature deviation |
| **Defect Prediction** | Void fraction, resin-rich regions, waviness amplification, defect severity index |
| **Structural Performance** | Undamaged compressive strength, Compression-After-Impact (CAI) strength |

Uncertainty is propagated through a **Monte Carlo simulation** (typically 3000 samples), enabling sensitivity analysis, reliability assessment, and validation against literature benchmarks. The framework is calibrated against published experimental data (Ricks et al. [22], Shah et al. [15], Ge et al. [17]) and is intended for **design-space exploration, sensitivity analysis, and uncertainty quantification**—not as a validated predictive tool for engineering certification.

> **⚠️ Important Scope Note:** This framework is specifically calibrated for **3D woven composites**. It has not been validated for braided, knitted, or other textile architectures. Extension to other architectures requires additional calibration and validation.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Integrated Modelling** | Links architecture, process, defects, and performance in a single pipeline |
| **Uncertainty Quantification** | Propagates variability via Monte Carlo simulation (3000 samples) |
| **Sensitivity Analysis** | Identifies the most influential parameters (Pearson correlation + Sobol' indices) |
| **Reliability Assessment** | Computes pass/fail rates against certification thresholds |
| **Literature Calibration** | Empirical sub-models calibrated against published experimental data |
| **Computational Efficiency** | Complete 3000-sample simulation in ~90 seconds on a standard workstation |
| **Full Reproducibility** | Fixed random seed and comprehensive output for reproducible research |
| **Open Source** | MIT License for free use and modification |

---

## 📁 Repository Structure

```
.
├── digital_twin_simulation.py          # Main simulation script
├── digital_twin_literature_results/    # Auto-generated output folder
│   ├── figures/                        # All plots (PNG)
│   │   ├── figure_void_vs_cai.png
│   │   ├── figure_defect_vs_fatigue.png
│   │   ├── figure_cai_distribution.png
│   │   ├── figure_sensitivity_ranking.png
│   │   ├── figure_reliability_distribution.png
│   │   ├── figure_parameter_distributions.png
│   │   └── figure_defect_distributions.png
│   ├── data/                           # Simulation data
│   │   ├── simulation_results.csv      # Full dataset (3000 rows)
│   │   ├── summary_metrics.csv         # Key metrics summary
│   │   └── validation_results.json     # Validation metrics
│   └── tables/                         # LaTeX tables for manuscripts
│       └── latex_tables.tex
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.9 or newer**
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shadatmahin17/Probabilistic-Digital-Twin-Framework-for-3D-Woven-Composites.git
   cd Probabilistic-Digital-Twin-Framework-for-3D-Woven-Composites
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
```

---

## ▶️ Usage

### Basic Usage

To run the full simulation and generate all results and figures:

```bash
python digital_twin_simulation.py
```

### What Happens When You Run

The script will:

1. Generate 3000 random architecture and process parameter sets (Table 2 in the paper)
2. Predict defect states (void fraction, resin-rich regions, waviness amplification, defect severity index) for each configuration
3. Predict structural performance (undamaged compressive strength, CAI strength) for each configuration
4. Compute sensitivity (Pearson correlation) and reliability (pass rates) metrics
5. Validate predicted CAI strength and void fraction against literature benchmarks
6. Export all results (CSV, JSON, LaTeX tables) and save figures as PNG files

### Advanced Configuration

To modify simulation parameters, edit the following variables in `digital_twin_simulation.py`:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `SEED` | 42 | Random seed for reproducibility |
| `N_SAMPLES` | 3000 | Number of Monte Carlo samples |
| `CAI_LIMIT` | 450 | CAI certification threshold (MPa) |
| `IMPACT_ENERGY` | 30.0 | Impact energy for CAI prediction (J) |

---

## 📊 Outputs

After a successful run, the following outputs are available in the `digital_twin_literature_results/` folder:

### Figures

| Filename | Description |
| :--- | :--- |
| `figure_void_vs_cai.png` | Scatter plot of void fraction vs. CAI strength (Figure 3 in paper) |
| `figure_defect_vs_fatigue.png` | Relationship between defect severity and fatigue knockdown (Figure 4 in paper) |
| `figure_cai_distribution.png` | Histogram of predicted CAI strength distribution (Figure 5 in paper) |
| `figure_sensitivity_ranking.png` | Sensitivity ranking of input parameters (Figure 6 in paper) |
| `figure_reliability_distribution.png` | Reliability analysis with pass/fail breakdown (Figure 7 in paper) |
| `figure_parameter_distributions.png` | Distributions of all input parameters |
| `figure_defect_distributions.png` | Distributions of predicted defects |

### Data Files

| Filename | Description |
| :--- | :--- |
| `simulation_results.csv` | Full dataset with all inputs and outputs for all 3000 samples |
| `summary_metrics.csv` | Key metrics summary (means, std devs, pass rates, sensitivity rankings) |
| `validation_results.json` | Validation metrics against literature benchmarks (CAI, void fraction) |

### Tables

| Filename | Description |
| :--- | :--- |
| `latex_tables.tex` | LaTeX tables ready for inclusion in manuscripts |

---

## 🔬 Reproducibility

The code is designed for full reproducibility:

- **Fixed Random Seed:** By default, `SEED = 42`. Running the script without modifications will generate exactly the same data and figures presented in the associated paper.
- **Multiple Seed Testing:** The code includes a multi-seed analysis (seeds 42, 123, 456, 789, 101112) to verify robustness. Results show CV < 0.2% for mean CAI.
- **Convergence Analysis:** The Monte Carlo estimates converge after approximately 2000 samples, confirming that 3000 samples are sufficient.

---

## 📖 How to Cite

If you use this code in your research, please cite the original work:

```bibtex
@article{mahin2026digitaltwin,
  title   = {Probabilistic Simulation Framework for 3D Woven Composites: Linking Architecture, Defects, and Structural Performance},
  author  = {Mahin, Shadat Hossen and Islam, Md. Touhidul},
  year    = {2026},
  doi     = {10.5281/zenodo.21220844},
  note    = {Available at: \url{https://github.com/shadatmahin17/Probabilistic-Digital-Twin-Framework-for-3D-Woven-Composites}}
}
```

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for any purpose, commercial or non-commercial, provided you retain the copyright notice and permission notice.

---

## 🙏 Acknowledgments

- The developers of **NumPy**, **SciPy**, and **Matplotlib** for providing the foundational computational tools used in this work.
- The researchers whose published experimental datasets (Ricks et al. [22], Shah et al. [15], Ge et al. [17]) provided the calibration benchmarks for this framework.
- The open-source Python community for their invaluable contributions.
- The reviewers of *Composite Structures* whose constructive feedback substantially improved the methodology and manuscript.

---

## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

**Md. Touhidul Islam** (Corresponding Author)
Department of Textile Engineering
University of Scholars
40 Kemal Ataturk Ave, Dhaka-1213, Bangladesh
Email: touhidtex@ius.edu.bd

Or open an issue on GitHub: [https://github.com/shadatmahin17/Probabilistic-Digital-Twin-Framework-for-3D-Woven-Composites/issues](https://github.com/shadatmahin17/Probabilistic-Digital-Twin-Framework-for-3D-Woven-Composites/issues)

---

## 📚 Related Publications

- Mahin, S. H., & Islam, M. T. (2026). *Probabilistic Simulation Framework for 3D Woven Composites: Linking Architecture, Defects, and Structural Performance.* **Composite Structures** (Under Review).
- Ricks, T. M., et al. (2022). "Multiscale Progressive Failure Analysis of 3D Woven Composites." *Polymers*, 14(20).
- Shah, S. Z. H., et al. (2023). "Multiscale damage modelling of notched and un-notched 3D woven composites with randomly distributed manufacturing defects." *Composite Structures*, 318, 117109.
- Ge, L., et al. (2021). "Micro-CT based trans-scale damage analysis of 3D braided composites with pore defects." *Composites Science and Technology*, 211, 108830.

---

## 🤝 Contributing

Contributions to improve the framework are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Star This Repository

If you find this framework useful for your research, please consider starring the repository on GitHub to help others discover it.
