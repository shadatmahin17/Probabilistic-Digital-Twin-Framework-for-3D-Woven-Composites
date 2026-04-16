
Code Explanation

This section provides a detailed yet understandable explanation of the Python code used in the Probabilistic Digital Twin Framework for 3D Woven and Braided Composites. The code performs a Monte Carlo simulation to predict how variations in material architecture and manufacturing processes affect defects and structural performance. It then produces all figures and data tables presented in the associated research paper.
Purpose of the Code

The code creates a digital twin – a virtual representation – of a 3D woven composite material. By simulating thousands of different combinations of design and manufacturing parameters, it:

    Estimates the likelihood of manufacturing defects (voids, resin-rich areas, fibre waviness).

    Predicts the resulting mechanical properties (compressive strength, compression-after-impact strength, fatigue strength).

    Identifies which parameters most influence performance (sensitivity analysis).

    Determines how many configurations meet certification requirements (reliability analysis).

    Validates the model by comparing its average predictions with experimental data from the literature.

All results are saved in a dedicated folder, ready for inclusion in a scientific paper.## Code Walkthrough: Digital Twin Simulation for 3D Woven Composites

```python
"""
===========================================================================
LITERATURE-CALIBRATED PROBABILISTIC DIGITAL TWIN FOR 3D WOVEN/BRAIDED COMPOSITES
===========================================================================
Author: Shadat Hossen Mahin
...
"""
```
- **Lines 1-20**: Multi-line comment (docstring) describing the script’s purpose, authors, literature sources, and features. This is for documentation only; Python ignores it.

```python
import os
import csv
import json
import numpy as np
import matplotlib
```
- **Import standard and third-party libraries**:
  - `os`: file and directory operations.
  - `csv`: writing CSV files.
  - `json`: writing JSON files.
  - `numpy as np`: numerical operations (arrays, random numbers, math).
  - `matplotlib`: plotting library (used later).

```python
# ============================================================================
# SAFE BACKEND SELECTION FOR ALL ENVIRONMENTS
# ============================================================================
def configure_matplotlib_backend():
    """
    Try interactive backends first. If unavailable, fall back to Agg.
    This avoids crashes on headless servers / mobile / notebook / cloud systems.
    """
    for backend in ( "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt  # noqa: F401
            return backend
        except Exception:
            continue

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401
    return "Agg"
```
- **`configure_matplotlib_backend()`**: Ensures matplotlib works in any environment (local computer, server, cloud). It tries to set an interactive backend (TkAgg) that can show figures; if that fails (e.g., no display), it falls back to the non‑interactive `Agg` backend which only saves files. This prevents crashes.
- `for backend in ( "TkAgg"):` – list of backends to try (currently only one). It could be extended.
- `matplotlib.use(backend, force=True)` – attempt to switch backend.
- If successful, returns backend name. On failure, continues to next.
- If all interactive backends fail, force `Agg` and import `plt`.

```python
SELECTED_BACKEND = configure_matplotlib_backend()
```
- Calls the function and stores the selected backend (for informational purposes).

```python
import matplotlib.pyplot as plt
from scipy import stats
```
- Now import `pyplot` and `stats` after backend is set.

```python
# ============================================================================
# CONFIGURATION – LITERATURE-BASED PARAMETERS
# ============================================================================
SEED = 42
np.random.seed(SEED)
```
- Sets a fixed random seed for reproducibility. Every run will produce the same random numbers.

```python
OUTPUT_DIR = "digital_twin_literature_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
- Creates an output folder. `exist_ok=True` means no error if folder already exists.

```python
# Certification limits (MPa)
CAI_LIMIT = 450.0
FATIGUE_LIMIT = 350.0
```
- Thresholds for pass/fail reliability analysis.

```python
# Architecture bounds
ARCH_BOUNDS = {
    "fiber_volume_fraction": (0.48, 0.60),
    "binder_density": (0.10, 0.20),
    "waviness": (0.02, 0.08),
    "braid_angle_deg": (25.0, 40.0),
    "thickness_mm": (3.0, 6.0),
}
```
- Dictionary defining minimum and maximum values for each architecture parameter, taken from literature.

```python
# Process bounds
PROC_BOUNDS = {
    "compaction_pressure_MPa": (0.2, 1.5),
    "resin_flow_rate": (0.5, 2.0),
    "cure_temp_deviation_C": (-10.0, 15.0),
}
```
- Process parameter ranges.

```python
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def normalize(x, bounds):
    return (x - bounds[0]) / (bounds[1] - bounds[0] + 1e-12)
```
- Normalizes a value to [0,1] using the given bounds. The small `1e-12` prevents division by zero.

```python
def clip01(x):
    return np.clip(x, 0.0, 1.0)
```
- Clips values to be between 0 and 1.

```python
def mean_std_text(x):
    return f"{np.mean(x):.3f} ± {np.std(x):.3f}"
```
- Returns a formatted string showing mean ± standard deviation, used for printing summaries.

```python
def safe_polyfit(x, y, degree=1):
    """Avoid failure if data are nearly constant."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.array([0.0, np.mean(y)])
    return np.polyfit(x, y, degree)
```
- Fits a polynomial (default linear) but handles the case where data are nearly constant (to avoid errors). If constant, returns [0, mean] representing a flat line.

```python
def can_show_figures():
    """
    Returns True only if a real interactive backend is active.
    """
    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "pdf", "svg", "ps", "cairo", "template"}
    return backend not in non_interactive and "agg" not in backend
```
- Checks if the current matplotlib backend is interactive (can display windows). Used later to decide whether to call `plt.show()`.

```python
def finalize_plots():
    """
    Open figure windows only when interactive backend is available.
    Otherwise just print a helpful message.
    """
    if can_show_figures():
        print(f"\nInteractive backend active: {matplotlib.get_backend()}")
        print("Opening figure windows...")
        plt.show(block=True)
    else:
        print(f"\nInteractive figure window is not available in this environment.")
        print(f"Current backend: {matplotlib.get_backend()}")
        print("All figures were saved as PNG files in the output folder.")
```
- Calls `plt.show(block=True)` if interactive backend is active, otherwise prints a message. This ensures figures appear on screen when possible.

```python
# ============================================================================
# SAMPLING
# ============================================================================
def sample_architecture(n):
    fv = np.random.uniform(*ARCH_BOUNDS["fiber_volume_fraction"], n)
    bd = np.random.uniform(*ARCH_BOUNDS["binder_density"], n)
    wav = np.random.uniform(*ARCH_BOUNDS["waviness"], n)
    angle = np.random.uniform(*ARCH_BOUNDS["braid_angle_deg"], n)
    thk = np.random.uniform(*ARCH_BOUNDS["thickness_mm"], n)
    return np.column_stack([fv, bd, wav, angle, thk])
```
- Generates `n` random architecture samples from uniform distributions within the given bounds. Returns a NumPy array with shape `(n, 5)` where each row is one sample.

```python
def sample_process(n):
    cp = np.random.uniform(*PROC_BOUNDS["compaction_pressure_MPa"], n)
    fr = np.random.uniform(*PROC_BOUNDS["resin_flow_rate"], n)
    td = np.random.uniform(*PROC_BOUNDS["cure_temp_deviation_C"], n)
    return np.column_stack([cp, fr, td])
```
- Similarly, generates `n` process samples, returns array shape `(n, 3)`.

```python
# ============================================================================
# DEFECT MODEL
# ============================================================================
def predict_defects(arch, proc):
    bd = arch[:, 1]
    wav = arch[:, 2]
    angle = arch[:, 3]
    thk = arch[:, 4]
    cp, fr, td = proc.T
```
- Extracts individual columns from the architecture and process arrays for easier use.

```python
    bd_n = normalize(bd, ARCH_BOUNDS["binder_density"])
    wav_n = normalize(wav, ARCH_BOUNDS["waviness"])
    angle_n = normalize(angle, ARCH_BOUNDS["braid_angle_deg"])
    thk_n = normalize(thk, ARCH_BOUNDS["thickness_mm"])
    cp_n = normalize(cp, PROC_BOUNDS["compaction_pressure_MPa"])
    fr_n = normalize(fr, PROC_BOUNDS["resin_flow_rate"])
    td_n = normalize(td, PROC_BOUNDS["cure_temp_deviation_C"])
```
- Normalizes each input to [0,1] using the respective bounds.

```python
    n = len(bd)
```
- Number of samples.

```python
    void_fraction = (
        0.017
        + 0.010 * wav_n
        + 0.008 * fr_n
        + 0.004 * thk_n
        + 0.006 * np.maximum(td_n, 0.0)
        - 0.008 * cp_n
        + 0.006 * bd_n * fr_n
        + 0.011 * np.random.randn(n)
    )
```
- Calculates void fraction using a linear model with coefficients from literature. `np.maximum(td_n,0)` ensures only positive temperature deviations increase voids. `np.random.randn(n)` adds random noise with standard deviation 0.011 (1.1% as per Shah et al.).

```python
    void_fraction = np.clip(void_fraction, 0.005, 0.05)
```
- Clips results to plausible range (0.5% to 5%).

```python
    resin_rich_index = (
        0.05
        + 0.08 * bd_n
        + 0.05 * thk_n
        + 0.05 * fr_n
        - 0.04 * cp_n
        + 0.02 * angle_n
        + 0.010 * np.random.randn(n)
    )
    resin_rich_index = np.clip(resin_rich_index, 0.03, 0.25)
```
- Similar linear model for resin‑rich index, with noise.

```python
    waviness_amplification = (
        0.02
        + 0.10 * wav_n
        - 0.03 * cp_n
        + 0.02 * bd_n
        + 0.02 * fr_n
        + 0.01 * np.random.randn(n)
    )
    waviness_amplification = np.clip(waviness_amplification, 0.0, 0.25)
```
- Waviness amplification model.

```python
    defect_severity = clip01(
        4.0 * void_fraction + 1.2 * resin_rich_index + 1.6 * waviness_amplification
    )
```
- Combines the three defect metrics into a single severity index, weighted by literature-based coefficients, then clipped to [0,1].

```python
    return {
        "void_fraction": void_fraction,
        "resin_rich_index": resin_rich_index,
        "waviness_amplification": waviness_amplification,
        "defect_severity": defect_severity,
    }
```
- Returns a dictionary containing all defect predictions.

```python
# ============================================================================
# STRUCTURAL MODELS
# ============================================================================
def predict_undamaged_strength(arch):
    fv, bd, wav, angle, thk = arch.T
    strength = 780.0 + 420.0 * (fv - 0.48) / 0.12 - 900.0 * wav
```
- Extracts columns, computes base strength from fiber volume fraction and waviness.

```python
    binder_effect = 1.0 - 0.40 * ((bd - 0.15) ** 2 / (0.05 ** 2))
    binder_effect = np.clip(binder_effect, 0.82, 1.05)
```
- Quadratic effect of binder density, centered at 0.15 (optimal). Clipped to reasonable range.

```python
    angle_penalty = 0.10 * ((angle - 30.0) / 10.0) ** 2
    angle_penalty = np.clip(angle_penalty, 0.0, 0.20)
```
- Penalty for braid angle deviating from 30°.

```python
    thk_penalty = 70.0 * ((thk - 4.5) / 1.5) ** 2
```
- Thickness penalty (thicker = weaker due to more defects).

```python
    strength = strength * binder_effect * (1.0 - angle_penalty) - thk_penalty
    return np.clip(strength, 600.0, 1200.0)
```
- Combines all effects and clips to realistic bounds.

```python
def predict_cai_strength(arch, proc, defects, impact_energy_J=30.0):
    s0 = predict_undamaged_strength(arch)
    vf = defects["void_fraction"]
    rr = defects["resin_rich_index"]
    wa = defects["waviness_amplification"]
    ds = defects["defect_severity"]
    bd = arch[:, 1]
    angle = arch[:, 3]
    n = len(bd)
```
- Gets undamaged strength and defect arrays.

```python
    impact_severity = impact_energy_J / 40.0
```
- Normalized impact energy (30 J / 40 J = 0.75).

```python
    arch_tough = 0.16 + 0.45 * bd - 0.08 * ((angle - 30.0) / 15.0) ** 2
    arch_tough = np.clip(arch_tough, 0.08, 0.28)
```
- Architecture toughness factor depending on binder and angle.

```python
    damage_index = (
        0.22 * impact_severity
        + 1.50 * vf
        + 0.45 * rr
        + 0.70 * wa
        + 0.65 * ds
        - 0.50 * arch_tough
        + 0.025 * np.random.randn(n)
    )
    damage_index = np.clip(damage_index, 0.08, 0.60)
```
- Damage index combines impact and defects, subtracts toughness, adds noise, clips.

```python
    cai_strength = s0 * (1.0 - damage_index)
    cai_strength = np.clip(cai_strength, 380.0, 560.0)
    return cai_strength, damage_index
```
- CAI strength = undamaged strength × (1‑damage). Clip to plausible range.

```python
def predict_fatigue_knockdown(arch, defects, cycles=5e5, stress_ratio=0.1):
    bd = arch[:, 1]
    wav = arch[:, 2]
    vf = defects["void_fraction"]
    rr = defects["resin_rich_index"]
    ds = defects["defect_severity"]
    n = len(bd)
    logN = np.log10(cycles)
```
- Fatigue knockdown factor depends on log cycles, defects, binder, and stress ratio.

```python
    knockdown = (
        0.88
        - 0.030 * (logN - 5.0)
        - 0.90 * vf
        - 0.10 * rr
        - 0.45 * wav
        - 0.18 * ds
        + 0.06 * bd
        + 0.03 * stress_ratio
        + 0.01 * np.random.randn(n)
    )
    knockdown = np.clip(knockdown, 0.35, 0.90)
```
- Linear combination, clipped to realistic range.

```python
    fatigue_strength = predict_undamaged_strength(arch) * knockdown
    fatigue_strength = np.clip(fatigue_strength, 350.0, 650.0)
    return fatigue_strength, knockdown
```
- Fatigue strength = undamaged × knockdown.

```python
# ============================================================================
# METRICS & ANALYSIS
# ============================================================================
def certification_metrics(cai_strength, fatigue_strength):
    cai_pass = cai_strength >= CAI_LIMIT
    fatigue_pass = fatigue_strength >= FATIGUE_LIMIT
    return {
        "cai_pass_rate": np.mean(cai_pass),
        "fatigue_pass_rate": np.mean(fatigue_pass),
        "joint_pass_rate": np.mean(cai_pass & fatigue_pass),
    }
```
- Computes pass rates (fraction of samples meeting thresholds).

```python
def reliability_analysis(cai_strength, fatigue_strength):
    cai_fail = cai_strength < CAI_LIMIT
    fatigue_fail = fatigue_strength < FATIGUE_LIMIT
    joint_fail = cai_fail | fatigue_fail
    return {
        "prob_fail_cai": np.mean(cai_fail),
        "prob_fail_fatigue": np.mean(fatigue_fail),
        "prob_fail_joint": np.mean(joint_fail),
        "reliability_cai": 1.0 - np.mean(cai_fail),
        "reliability_fatigue": 1.0 - np.mean(fatigue_fail),
        "reliability_joint": 1.0 - np.mean(joint_fail),
    }
```
- Failure probabilities and reliability (1‑failure).

```python
def correlation_sensitivity(inputs, y, names):
    ranking = []
    for i, name in enumerate(names):
        corr = np.corrcoef(inputs[:, i], y)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        ranking.append((name, abs(corr), corr))
    ranking.sort(key=lambda t: t[1], reverse=True)
    return ranking
```
- Computes Pearson correlation between each input column and the target `y` (e.g., CAI strength). Returns list sorted by absolute correlation (descending).

```python
def validate_against_literature(cai_mean, void_mean, und_min, und_max):
    lit = {
        "cai_strength": {"mean": 450, "range": (430, 470)},
        "void_fraction": {"mean": 2.17, "range": (1.0, 4.0)},
        "undamaged_strength": {"range": (600, 1200)},
    }
```
- Literature benchmark values (from Ricks et al., Shah et al.).

```python
    cai_rmse = np.abs(cai_mean - lit["cai_strength"]["mean"])
```
- RMSE (here just absolute difference, since single benchmark).

```python
    validation = {
        "cai_strength": {
            "sim_mean": float(cai_mean),
            "lit_mean": float(lit["cai_strength"]["mean"]),
            "lit_range": [float(lit["cai_strength"]["range"][0]), float(lit["cai_strength"]["range"][1])],
            "rmse": float(cai_rmse),
            "relative_error": float((cai_rmse / lit["cai_strength"]["mean"]) * 100),
            "within_range": bool(lit["cai_strength"]["range"][0] <= cai_mean <= lit["cai_strength"]["range"][1])
        },
        ...
    }
    return validation
```
- Builds validation dictionary with converted Python native types (for JSON serialisation).

```python
# ============================================================================
# MAIN DIGITAL TWIN
# ============================================================================
def run_digital_twin(n_samples=3000, impact_energy_J=30.0, cycles=5e5):
    arch = sample_architecture(n_samples)
    proc = sample_process(n_samples)
    defects = predict_defects(arch, proc)
    undamaged_strength = predict_undamaged_strength(arch)
    cai_strength, damage_index = predict_cai_strength(arch, proc, defects, impact_energy_J)
    fatigue_strength, fatigue_knockdown = predict_fatigue_knockdown(arch, defects, cycles)
    cert = certification_metrics(cai_strength, fatigue_strength)
    return {
        "arch": arch,
        "proc": proc,
        "defects": defects,
        "undamaged_strength": undamaged_strength,
        "cai_strength": cai_strength,
        "damage_index": damage_index,
        "fatigue_strength": fatigue_strength,
        "fatigue_knockdown": fatigue_knockdown,
        "certification": cert,
    }
```
- Runs the full digital twin: samples, predicts defects, predicts strengths, computes certification metrics, returns a dictionary with all results.

```python
# ============================================================================
# EXPORTS
# ============================================================================
def export_results_csv(results, filename="simulation_results.csv"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([...])   # header
        for i in range(len(results["arch"])):
            pass_crit = "PASS" if results["cai_strength"][i] >= CAI_LIMIT else "FAIL"
            writer.writerow([...])   # one row per sample
    print(f"  - Results CSV: {path}")
```
- Writes all raw simulation data to a CSV file.

```python
def export_summary_csv(results, sensitivity, reliability, validation, filename="summary_metrics.csv"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Mean undamaged strength (MPa)", np.mean(results["undamaged_strength"])])
        ...
        writer.writerow([])
        writer.writerow(["Sensitivity ranking", "Abs(correlation)", "Signed correlation"])
        for name, abs_corr, corr in sensitivity:
            writer.writerow([name, f"{abs_corr:.4f}", f"{corr:.4f}"])
    print(f"  - Summary CSV: {path}")
```
- Writes a summary CSV with mean, std, pass rates, sensitivity, etc.

```python
def export_validation_json(validation, filename="validation_results.json"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"  - Validation JSON: {path}")
```
- Writes validation results as JSON.

```python
def export_latex_tables(results, sensitivity, reliability, validation, filename="latex_tables.tex"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% LaTeX tables for manuscript\n")
        f.write("\\begin{table}[h]\n\\centering\n\\caption{Parameter ranges (literature-based)}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\nParameter & Lower & Upper \\\\\n\\hline\n")
        for p, (lo, hi) in {**ARCH_BOUNDS, **PROC_BOUNDS}.items():
            f.write(f"{p.replace('_', ' ')} & {lo} & {hi} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        ...
```
- Generates LaTeX code for four tables (parameter ranges, performance summary, reliability, validation). These can be copied directly into a manuscript.

```python
# ============================================================================
# FIGURES
# ============================================================================
def save_scatter_void_vs_cai(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    void = results["defects"]["void_fraction"] * 100
    cai = results["cai_strength"]
    sev = results["defects"]["defect_severity"]
    sc = ax.scatter(void, cai, c=sev, cmap="viridis", alpha=0.6, s=20)
    z = safe_polyfit(void, cai, 1)
    p = np.poly1d(z)
    xline = np.linspace(np.min(void), np.max(void), 100)
    ax.plot(xline, p(xline), "r--", label=f"Trend: y={z[0]:.1f}x+{z[1]:.1f}")
    ax.axhspan(430, 470, alpha=0.2, color="green", label="Ricks et al. range")
    ax.set_xlabel("Void Fraction (%)")
    ax.set_ylabel("CAI Strength (MPa)")
    ax.set_title("Void Fraction vs. CAI Strength")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Defect Severity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure_void_vs_cai.png")
    fig.savefig(path, dpi=300)
    return fig, path
```
- Creates scatter plot of void vs CAI, colored by defect severity, adds trend line and literature range. Saves PNG.

Similar functions exist for:
- `save_scatter_defect_vs_fatigue`
- `save_histogram_cai`
- `save_sensitivity_bar`
- `save_reliability_histogram`
- `save_parameter_distributions`
- `save_defect_distributions`

Each one creates a matplotlib figure, customizes it, saves to PNG, and returns the figure object and path.

```python
# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("LITERATURE-CALIBRATED DIGITAL TWIN FOR 3D WOVEN COMPOSITES")
    print("=" * 70)
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    n_samples = 3000
    impact_energy = 30.0
    cycles = 5e5
```
- Sets number of samples, impact energy, fatigue cycles.

```python
    results = run_digital_twin(n_samples, impact_energy, cycles)
```
- Runs the full simulation.

```python
    input_names = list(ARCH_BOUNDS.keys()) + list(PROC_BOUNDS.keys())
    full_inputs = np.column_stack([results["arch"], results["proc"]])
    sensitivity = correlation_sensitivity(full_inputs, results["cai_strength"], input_names)
```
- Prepares input array for sensitivity analysis (architecture + process) and computes correlations.

```python
    reliability = reliability_analysis(results["cai_strength"], results["fatigue_strength"])
```
- Computes reliability metrics.

```python
    validation = validate_against_literature(
        np.mean(results["cai_strength"]),
        np.mean(results["defects"]["void_fraction"]) * 100,
        np.min(results["undamaged_strength"]),
        np.max(results["undamaged_strength"]),
    )
```
- Validates simulation means against literature.

```python
    print("\n" + "-" * 50)
    print("SIMULATION SUMMARY")
    print("-" * 50)
    print(f"Undamaged strength: {mean_std_text(results['undamaged_strength'])} MPa")
    ...
```
- Prints summary to console.

```python
    export_results_csv(results)
    export_summary_csv(results, sensitivity, reliability, validation)
    export_validation_json(validation)
    export_latex_tables(results, sensitivity, reliability, validation)
```
- Exports all data files.

```python
    fig_paths = []
    for _, path in [
        save_scatter_void_vs_cai(results),
        save_scatter_defect_vs_fatigue(results),
        save_histogram_cai(results),
        save_sensitivity_bar(sensitivity),
        save_reliability_histogram(results),
        save_parameter_distributions(results),
        save_defect_distributions(results),
    ]:
        fig_paths.append(path)
```
- Calls each figure‑saving function and collects the file paths.

```python
    print("\n" + "-" * 50)
    print("SAVED FIGURES")
    for p in fig_paths:
        print(f"  {os.path.basename(p)}")
```
- Prints list of saved figure filenames.

```python
    print("\n" + "=" * 70)
    print("DIGITAL TWIN SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    finalize_plots()
```
- Final message and calls `finalize_plots()` to show figures if possible.

```python
if __name__ == "__main__":
    main()
```
- Standard Python guard: runs `main()` only if script is executed directly (not imported).

---

## Summary

This code is a complete, self‑contained digital twin simulation for 3D woven composites. It:

- Samples input parameters from literature ranges.
- Predicts manufacturing defects using empirical models.
- Predicts structural performance (undamaged, CAI, fatigue) using reduced‑order models.
- Performs sensitivity and reliability analyses.
- Validates against literature benchmarks.
- Exports all results (CSV, JSON, LaTeX tables) and saves publication‑ready figures.

The code is modular, well‑commented, and designed to run in any environment (local, server, mobile). It enables full reproducibility of the research results.
