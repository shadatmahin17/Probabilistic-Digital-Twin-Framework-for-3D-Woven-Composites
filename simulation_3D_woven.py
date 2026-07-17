"""
===========================================================================
LITERATURE-CALIBRATED PROBABILISTIC SIMULATION FRAMEWORK FOR 3D WOVEN COMPOSITES
===========================================================================
Authors: Shadat Hossen Mahin, Md. Touhidul Islam
Version: 1.0

This script:
- Monte Carlo simulation (3000 samples) using literature parameter ranges
- Defect prediction (voids, resin-rich regions, waviness amplification)
- Structural performance (undamaged strength, CAI, fatigue)
- Sensitivity analysis, reliability assessment, calibration-consistency checks
- Generates all manuscript figures (3–7) and exports CSV/JSON/LaTeX
===========================================================================
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "simulation_results_3D_woven"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAI_LIMIT = 450.0          # Certification threshold (MPa)
FATIGUE_LIMIT = 350.0      # Fatigue threshold (MPa)

# Architecture bounds (Table 2 in manuscript - Updated yarn nomenclature)
ARCH_BOUNDS = {
    "fiber_volume_fraction": (0.48, 0.60),
    "binder_density": (0.10, 0.20),
    "waviness": (0.02, 0.08),
    "yarn_orientation_angle_deg": (25.0, 40.0),
    "thickness_mm": (3.0, 6.0),
}

# Process bounds (Table 2 in manuscript)
PROC_BOUNDS = {
    "compaction_pressure_MPa": (0.2, 1.5),
    "resin_flow_rate": (0.5, 2.0),
    "cure_temp_deviation_C": (-10.0, 15.0),
}

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def normalize(x, bounds):
    return (x - bounds[0]) / (bounds[1] - bounds[0] + 1e-12)

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def mean_std_text(x):
    return f"{np.mean(x):.3f} ± {np.std(x):.3f}"

def safe_polyfit(x, y, degree=1):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.array([0.0, np.mean(y)])
    return np.polyfit(x, y, degree)

# ----------------------------------------------------------------------------
# SAMPLING
# ----------------------------------------------------------------------------
def sample_architecture(n):
    fv = np.random.uniform(*ARCH_BOUNDS["fiber_volume_fraction"], n)
    bd = np.random.uniform(*ARCH_BOUNDS["binder_density"], n)
    wav = np.random.uniform(*ARCH_BOUNDS["waviness"], n)
    angle = np.random.uniform(*ARCH_BOUNDS["yarn_orientation_angle_deg"], n)
    thk = np.random.uniform(*ARCH_BOUNDS["thickness_mm"], n)
    return np.column_stack([fv, bd, wav, angle, thk])

def sample_process(n):
    cp = np.random.uniform(*PROC_BOUNDS["compaction_pressure_MPa"], n)
    fr = np.random.uniform(*PROC_BOUNDS["resin_flow_rate"], n)
    td = np.random.uniform(*PROC_BOUNDS["cure_temp_deviation_C"], n)
    return np.column_stack([cp, fr, td])

# ----------------------------------------------------------------------------
# DEFECT PREDICTION (Eq. 2–5)
# ----------------------------------------------------------------------------
def predict_defects(arch, proc):
    bd = arch[:, 1]
    wav = arch[:, 2]
    angle = arch[:, 3]
    thk = arch[:, 4]
    cp, fr, td = proc.T

    bd_n = normalize(bd, ARCH_BOUNDS["binder_density"])
    wav_n = normalize(wav, ARCH_BOUNDS["waviness"])
    angle_n = normalize(angle, ARCH_BOUNDS["yarn_orientation_angle_deg"])
    thk_n = normalize(thk, ARCH_BOUNDS["thickness_mm"])
    cp_n = normalize(cp, PROC_BOUNDS["compaction_pressure_MPa"])
    fr_n = normalize(fr, PROC_BOUNDS["resin_flow_rate"])
    td_n = normalize(td, PROC_BOUNDS["cure_temp_deviation_C"])

    n = len(bd)

    # Eq. (2) Void fraction
    void_fraction = (
        0.017 + 0.010 * wav_n + 0.008 * fr_n + 0.004 * thk_n
        + 0.006 * np.maximum(td_n, 0.0) - 0.008 * cp_n
        + 0.006 * bd_n * fr_n + 0.011 * np.random.randn(n)
    )
    void_fraction = np.clip(void_fraction, 0.005, 0.05)

    # Eq. (3) Resin-rich regions
    resin_rich_index = (
        0.05 + 0.08 * bd_n + 0.05 * thk_n + 0.05 * fr_n
        - 0.04 * cp_n + 0.02 * angle_n + 0.010 * np.random.randn(n)
    )
    resin_rich_index = np.clip(resin_rich_index, 0.03, 0.25)

    # Eq. (4) Waviness amplification
    waviness_amplification = (
        0.02 + 0.10 * wav_n - 0.03 * cp_n
        + 0.02 * bd_n + 0.02 * fr_n + 0.01 * np.random.randn(n)
    )
    waviness_amplification = np.clip(waviness_amplification, 0.0, 0.25)

    # Eq. (5) Defect severity index
    defect_severity = clip01(
        4.0 * void_fraction + 1.2 * resin_rich_index + 1.6 * waviness_amplification
    )

    return {
        "void_fraction": void_fraction,
        "resin_rich_index": resin_rich_index,
        "waviness_amplification": waviness_amplification,
        "defect_severity": defect_severity,
    }

# ----------------------------------------------------------------------------
# STRUCTURAL MODELS (Eq. 6–14)
# ----------------------------------------------------------------------------
def predict_undamaged_strength(arch):
    fv, bd, wav, angle, thk = arch.T

    # Eq. (6) baseline
    strength = 780.0 + 420.0 * ((fv - 0.48) / 0.12) - 900.0 * wav

    # Eq. (7) binder effect
    binder_effect = 1.0 - 0.40 * ((bd - 0.15) ** 2 / (0.05 ** 2))
    binder_effect = np.clip(binder_effect, 0.82, 1.05)

    # Eq. (8) angle penalty
    angle_penalty = 0.10 * ((angle - 30.0) / 10.0) ** 2
    angle_penalty = np.clip(angle_penalty, 0.0, 0.20)

    # Eq. (9) thickness penalty
    thk_penalty = 70.0 * ((thk - 4.5) / 1.5) ** 2

    # Eq. (10) final strength
    strength = strength * binder_effect * (1.0 - angle_penalty) - thk_penalty
    return np.clip(strength, 600.0, 1200.0)

def predict_cai_strength(arch, proc, defects, impact_energy_J=30.0):
    s0 = predict_undamaged_strength(arch)

    vf = defects["void_fraction"]
    rr = defects["resin_rich_index"]
    wa = defects["waviness_amplification"]
    ds = defects["defect_severity"]
    bd = arch[:, 1]
    angle = arch[:, 3]
    n = len(bd)

    impact_severity = impact_energy_J / 40.0

    # Architecture toughness term
    arch_tough = 0.16 + 0.45 * bd - 0.08 * ((angle - 30.0) / 15.0) ** 2
    arch_tough = np.clip(arch_tough, 0.08, 0.28)

    # Eq. (12) damage index
    damage_index = (
        0.22 * impact_severity + 1.50 * vf + 0.45 * rr
        + 0.70 * wa + 0.65 * ds - 0.50 * arch_tough
        + 0.025 * np.random.randn(n)
    )
    damage_index = np.clip(damage_index, 0.08, 0.60)

    # Eq. (11) CAI strength
    cai_strength = s0 * (1.0 - damage_index)
    cai_strength = np.clip(cai_strength, 380.0, 560.0)
    return cai_strength, damage_index

def predict_fatigue_knockdown(arch, defects, cycles=5e5, stress_ratio=0.1):
    bd = arch[:, 1]
    wav = arch[:, 2]
    vf = defects["void_fraction"]
    rr = defects["resin_rich_index"]
    ds = defects["defect_severity"]
    n = len(bd)

    logN = np.log10(cycles)

    # Eq. (14) knockdown factor
    knockdown = (
        0.88 - 0.030 * (logN - 5.0) - 0.90 * vf - 0.10 * rr
        - 0.45 * wav - 0.18 * ds + 0.06 * bd + 0.03 * stress_ratio
        + 0.01 * np.random.randn(n)
    )
    knockdown = np.clip(knockdown, 0.35, 0.90)

    # Eq. (13) fatigue strength
    fatigue_strength = predict_undamaged_strength(arch) * knockdown
    fatigue_strength = np.clip(fatigue_strength, 350.0, 650.0)
    return fatigue_strength, knockdown

# ----------------------------------------------------------------------------
# METRICS AND ANALYSIS
# ----------------------------------------------------------------------------
def certification_metrics(cai_strength, fatigue_strength):
    cai_pass = cai_strength >= CAI_LIMIT
    fatigue_pass = fatigue_strength >= FATIGUE_LIMIT
    return {
        "cai_pass_rate": np.mean(cai_pass),
        "fatigue_pass_rate": np.mean(fatigue_pass),
        "joint_pass_rate": np.mean(cai_pass & fatigue_pass),
    }

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

def correlation_sensitivity(inputs, y, names):
    ranking = []
    for i, name in enumerate(names):
        corr = np.corrcoef(inputs[:, i], y)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        ranking.append((name, abs(corr), corr))
    ranking.sort(key=lambda t: t[1], reverse=True)
    return ranking

def validate_against_literature(cai_mean, void_mean, und_min, und_max):
    lit = {
        "cai_strength": {"mean": 450, "range": (430, 470)},
        "void_fraction": {"mean": 2.7, "range": (1.0, 4.0)},
        "undamaged_strength": {"range": (600, 1200)},
    }
    cai_ae = np.abs(cai_mean - lit["cai_strength"]["mean"])

    validation = {
        "cai_strength": {
            "sim_mean": float(cai_mean),
            "lit_mean": float(lit["cai_strength"]["mean"]),
            "lit_range": [float(lit["cai_strength"]["range"][0]),
                          float(lit["cai_strength"]["range"][1])],
            "ae": float(cai_ae),
            "relative_error": float((cai_ae / lit["cai_strength"]["mean"]) * 100),
            "within_range": bool(
                lit["cai_strength"]["range"][0] <= cai_mean <= lit["cai_strength"]["range"][1]
            ),
        },
        "void_fraction": {
            "sim_mean": float(void_mean),
            "lit_mean": float(lit["void_fraction"]["mean"]),
            "lit_range": [float(lit["void_fraction"]["range"][0]),
                          float(lit["void_fraction"]["range"][1])],
            "within_range": bool(
                lit["void_fraction"]["range"][0] <= void_mean <= lit["void_fraction"]["range"][1]
            ),
        },
        "undamaged_strength": {
            "sim_min": float(und_min),
            "sim_max": float(und_max),
            "lit_range": [float(lit["undamaged_strength"]["range"][0]),
                          float(lit["undamaged_strength"]["range"][1])],
            "within_range": bool(
                lit["undamaged_strength"]["range"][0] <= und_min
                and und_max <= lit["undamaged_strength"]["range"][1]
            ),
        },
    }
    return validation

# ----------------------------------------------------------------------------
# MAIN SIMULATION
# ----------------------------------------------------------------------------
def run_simulation(n_samples=3000, impact_energy_J=30.0, cycles=5e5):
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

# ----------------------------------------------------------------------------
# EXPORT FUNCTIONS
# ----------------------------------------------------------------------------
def export_results_csv(results, filename="simulation_results.csv"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fiber_volume_fraction", "binder_density", "waviness",
            "yarn_orientation_angle_deg", "thickness_mm", "compaction_pressure_MPa",
            "resin_flow_rate", "cure_temp_deviation_C", "void_fraction_%",
            "resin_rich_index", "waviness_amplification", "defect_severity",
            "undamaged_strength_MPa", "damage_index", "CAI_strength_MPa",
            "fatigue_strength_MPa", "fatigue_knockdown", "pass_criteria"
        ])
        n = len(results["arch"])
        for i in range(n):
            pass_crit = "PASS" if results["cai_strength"][i] >= CAI_LIMIT else "FAIL"
            writer.writerow([
                results["arch"][i, 0], results["arch"][i, 1], results["arch"][i, 2],
                results["arch"][i, 3], results["arch"][i, 4],
                results["proc"][i, 0], results["proc"][i, 1], results["proc"][i, 2],
                results["defects"]["void_fraction"][i] * 100,
                results["defects"]["resin_rich_index"][i],
                results["defects"]["waviness_amplification"][i],
                results["defects"]["defect_severity"][i],
                results["undamaged_strength"][i],
                results["damage_index"][i],
                results["cai_strength"][i],
                results["fatigue_strength"][i],
                results["fatigue_knockdown"][i],
                pass_crit
            ])
    print(f"  - Results CSV: {path}")

def export_summary_csv(results, sensitivity, reliability, validation, filename="summary_metrics.csv"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Mean undamaged strength (MPa)", np.mean(results["undamaged_strength"])])
        writer.writerow(["Std undamaged strength (MPa)", np.std(results["undamaged_strength"])])
        writer.writerow(["Mean CAI strength (MPa)", np.mean(results["cai_strength"])])
        writer.writerow(["Std CAI strength (MPa)", np.std(results["cai_strength"])])
        writer.writerow(["Mean fatigue strength (MPa)", np.mean(results["fatigue_strength"])])
        writer.writerow(["Std fatigue strength (MPa)", np.std(results["fatigue_strength"])])
        writer.writerow(["Mean void fraction (%)", np.mean(results["defects"]["void_fraction"]) * 100])
        writer.writerow(["Std void fraction (%)", np.std(results["defects"]["void_fraction"]) * 100])
        writer.writerow(["CAI pass rate (%)", results["certification"]["cai_pass_rate"] * 100])
        writer.writerow(["Fatigue pass rate (%)", results["certification"]["fatigue_pass_rate"] * 100])
        writer.writerow(["Joint pass rate (%)", results["certification"]["joint_pass_rate"] * 100])
        writer.writerow(["CAI failure probability (%)", reliability["prob_fail_cai"] * 100])
        writer.writerow(["Fatigue failure probability (%)", reliability["prob_fail_fatigue"] * 100])
        writer.writerow(["Joint failure probability (%)", reliability["prob_fail_joint"] * 100])
        writer.writerow(["Validation AE (MPa)", validation["cai_strength"]["ae"]])
        writer.writerow(["Validation relative error (%)", validation["cai_strength"]["relative_error"]])
        writer.writerow([])
        writer.writerow(["Sensitivity ranking", "Abs(correlation)", "Signed correlation"])
        for name, abs_corr, corr in sensitivity:
            writer.writerow([name, f"{abs_corr:.4f}", f"{corr:.4f}"])
    print(f"  - Summary CSV: {path}")

def export_validation_json(validation, filename="validation_results.json"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"  - Validation JSON: {path}")

def export_latex_tables(results, sensitivity, reliability, validation, filename="latex_tables.tex"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% LaTeX tables for manuscript\n")
        f.write("\\begin{table}[h]\n\\centering\n\\caption{Parameter ranges (literature-based)}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\nParameter & Lower & Upper \\\\\n\\hline\n")
        for p, (lo, hi) in {**ARCH_BOUNDS, **PROC_BOUNDS}.items():
            f.write(f"{p.replace('_', ' ')} & {lo} & {hi} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        f.write("\\begin{table}[h]\n\\centering\n\\caption{Performance summary}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\nMetric & Mean & Std & Units \\\\\n\\hline\n")
        f.write(f"Undamaged strength & {np.mean(results['undamaged_strength']):.1f} & {np.std(results['undamaged_strength']):.1f} & MPa \\\\\n")
        f.write(f"CAI strength & {np.mean(results['cai_strength']):.1f} & {np.std(results['cai_strength']):.1f} & MPa \\\\\n")
        f.write(f"Fatigue strength & {np.mean(results['fatigue_strength']):.1f} & {np.std(results['fatigue_strength']):.1f} & MPa \\\\\n")
        f.write(f"Void fraction & {np.mean(results['defects']['void_fraction']) * 100:.2f} & {np.std(results['defects']['void_fraction']) * 100:.2f} & \\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        f.write("\\begin{table}[h]\n\\centering\n\\caption{Reliability metrics}\n")
        f.write("\\begin{tabular}{lc}\n\\hline\nMetric & Value \\\\\n\\hline\n")
        f.write(f"CAI pass rate & {reliability['reliability_cai'] * 100:.1f}\\% \\\\\n")
        f.write(f"CAI failure probability & {reliability['prob_fail_cai'] * 100:.1f}\\% \\\\\n")
        f.write(f"Joint pass rate & {reliability['reliability_joint'] * 100:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        f.write("\\begin{table}[h]\n\\centering\n\\caption{Validation against literature}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\nMetric & Simulated & Literature & Units \\\\\n\\hline\n")
        f.write(f"CAI strength & {validation['cai_strength']['sim_mean']:.1f} & {validation['cai_strength']['lit_mean']:.1f} & MPa \\\\\n")
        f.write(f"Void fraction & {validation['void_fraction']['sim_mean']:.2f} & {validation['void_fraction']['lit_mean']:.2f} & \\% \\\\\n")
        f.write(f"AE (calibration-consistency check) & {validation['cai_strength']['ae']:.1f} & -- & MPa \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  - LaTeX tables: {path}")

# ----------------------------------------------------------------------------
# FIGURE GENERATION (Figures 3–7)
# ----------------------------------------------------------------------------
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
    
    # Add annotation clarifying clipping boundaries
    ax.text(0.05, 0.95, "Data clusters at 380 & 560 MPa reflect model clipping boundaries", 
            transform=ax.transAxes, fontsize=8, color="dimgray", style="italic",
            verticalalignment="top")

    ax.set_xlabel("Void Fraction (%)")
    ax.set_ylabel("CAI Strength (MPa)")
    ax.set_title("Void Fraction vs. CAI Strength")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Defect Severity")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_void_vs_cai.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_scatter_defect_vs_fatigue(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    sev = MathSeverity = results["defects"]["defect_severity"]
    kd = results["fatigue_knockdown"]
    void = results["defects"]["void_fraction"] * 100

    sc = ax.scatter(sev, kd, c=void, cmap="plasma", alpha=0.6, s=20)
    z = safe_polyfit(sev, kd, 1)
    p = np.poly1d(z)
    xline = np.linspace(np.min(sev), np.max(sev), 100)
    ax.plot(xline, p(xline), "r--", label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")

    ax.set_xlabel("Defect Severity Index")
    ax.set_ylabel("Fatigue Knockdown Factor")
    ax.set_title("Defect Severity vs. Fatigue Knockdown")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Void Fraction (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_defect_vs_fatigue.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_histogram_cai(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    cai = results["cai_strength"]

    ax.hist(cai, bins=35, alpha=0.7, edgecolor="black")
    ax.axvline(CAI_LIMIT, linestyle="--", label=f"Certification Limit = {CAI_LIMIT} MPa")
    ax.axvspan(430, 470, alpha=0.2, color="green", label="Ricks et al. range")
    
    # Add explanation of the clipping boundaries
    ax.text(0.05, 0.90, "Outer peaks represent physical clipping bounds [380, 560] MPa", 
            transform=ax.transAxes, fontsize=8, color="dimgray", style="italic")

    ax.set_xlabel("CAI Strength (MPa)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of CAI Strength")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_cai_distribution.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_sensitivity_bar(sensitivity):
    # Format and map variable names beautifully, correcting braid references
    formatted_names = []
    for s in sensitivity:
        name = s[0].replace("_", " ").title()
        if "Yarn Orientation" in name:
            name = "Yarn Orientation Angle (°)"
        elif "Mpa" in name:
            name = name.replace("Mpa", "(MPa)")
        elif "Mm" in name:
            name = name.replace("Mm", "(mm)")
        elif "C" in name:
            name = name.replace(" C", " (°C)")
        formatted_names.append(name)
        
    names = formatted_names[::-1]
    values = [s[1] for s in sensitivity][::-1]
    colors = ["green" if s[2] > 0 else "red" for s in sensitivity[::-1]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel("Absolute Correlation with CAI Strength")
    ax.set_ylabel("Input Variable")
    ax.set_title("Sensitivity Ranking")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_sensitivity_ranking.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_reliability_histogram(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cai = results["cai_strength"]

    _, _, patches = ax1.hist(cai, bins=35, alpha=0.7, edgecolor="black")
    for patch in patches:
        if patch.get_x() < CAI_LIMIT:
            patch.set_facecolor("salmon")
        else:
            patch.set_facecolor("lightgreen")

    ax1.axvline(CAI_LIMIT, linestyle="--", label="Certification Limit")
    ax1.set_xlabel("CAI Strength (MPa)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("CAI Strength Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    pass_rate = np.mean(cai >= CAI_LIMIT) * 100
    fail_rate = 100 - pass_rate
    ax2.pie(
        [pass_rate, fail_rate],
        labels=["Pass", "Fail"],
        colors=["lightgreen", "salmon"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.05),
    )
    ax2.set_title(f"Reliability: {pass_rate:.1f}% Pass Rate")

    fig.suptitle("Reliability Distribution of CAI Strength")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_reliability_distribution.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_parameter_distributions(results):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    all_params = {
        "fiber_vol_frac": results["arch"][:, 0],
        "binder_density": results["arch"][:, 1],
        "waviness": results["arch"][:, 2],
        "yarn_orientation_angle": results["arch"][:, 3],
        "thickness": results["arch"][:, 4],
        "compaction_pressure": results["proc"][:, 0],
        "resin_flow_rate": results["proc"][:, 1],
        "cure_temp_dev": results["proc"][:, 2],
    }

    last_i = -1
    for i, (name, vals) in enumerate(all_params.items()):
        axes[i].hist(vals, bins=30, alpha=0.7, edgecolor="black")
        axes[i].set_xlabel(name.replace("_", " "))
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{name}\nμ={np.mean(vals):.3f}, σ={np.std(vals):.3f}")
        axes[i].grid(alpha=0.3)
        last_i = i

    for j in range(last_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Input Parameter Distributions")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_parameter_distributions.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

def save_defect_distributions(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    defects = results["defects"]
    titles = ["Void Fraction (%)", "Resin-Rich Index", "Waviness Amplification", "Defect Severity"]
    data = [
        defects["void_fraction"] * 100,
        defects["resin_rich_index"],
        defects["waviness_amplification"],
        defects["defect_severity"],
    ]

    for i, (title, vals) in enumerate(zip(titles, data)):
        axes[i].hist(vals, bins=30, alpha=0.7, edgecolor="black")
        axes[i].set_xlabel(title)
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{title}\nμ={np.mean(vals):.3f}, σ={np.std(vals):.3f}")
        axes[i].grid(alpha=0.3)

    fig.suptitle("Defect Distributions")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figure_defect_distributions.png")
    fig.savefig(path, dpi=300)
    print(f"  - Figure: {os.path.basename(path)}")
    return fig, path

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("LITERATURE-CALIBRATED PROBABILISTIC SIMULATION FRAMEWORK")
    print("FOR 3D WOVEN COMPOSITES")
    print("=" * 70)
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    n_samples = 3000
    impact_energy = 30.0
    cycles = 5e5

    # Run simulation
    results = run_simulation(n_samples, impact_energy, cycles)

    # Sensitivity analysis
    input_names = list(ARCH_BOUNDS.keys()) + list(PROC_BOUNDS.keys())
    full_inputs = np.column_stack([results["arch"], results["proc"]])
    sensitivity = correlation_sensitivity(full_inputs, results["cai_strength"], input_names)

    # Reliability
    reliability = reliability_analysis(results["cai_strength"], results["fatigue_strength"])

    # Calibration-consistency checks
    validation = validate_against_literature(
        np.mean(results["cai_strength"]),
        np.mean(results["defects"]["void_fraction"]) * 100,
        np.min(results["undamaged_strength"]),
        np.max(results["undamaged_strength"]),
    )

    # Print summary
    print("\n" + "-" * 50)
    print("SIMULATION SUMMARY")
    print("-" * 50)
    print(f"Undamaged strength: {mean_std_text(results['undamaged_strength'])} MPa")
    print(f"CAI strength:       {mean_std_text(results['cai_strength'])} MPa")
    print(f"Fatigue strength:   {mean_std_text(results['fatigue_strength'])} MPa")
    print(f"Void fraction:      {mean_std_text(results['defects']['void_fraction'] * 100)}%")
    print(f"\nCAI pass rate:      {results['certification']['cai_pass_rate'] * 100:.1f}%")
    print(f"CAI failure prob:   {reliability['prob_fail_cai'] * 100:.1f}%")
    print(f"\nValidation AE:      {validation['cai_strength']['ae']:.1f} MPa (calibration-consistency check)")
    print(f"Relative error:     {validation['cai_strength']['relative_error']:.1f}%")

    # Export data
    export_results_csv(results)
    export_summary_csv(results, sensitivity, reliability, validation)
    export_validation_json(validation)
    export_latex_tables(results, sensitivity, reliability, validation)

    # Generate figures (3–7)
    print("\n" + "-" * 50)
    print("GENERATING FIGURES")
    print("-" * 50)

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

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("\nGenerated figures:")
    for p in fig_paths:
        print(f"  - {os.path.basename(p)}")

    print("\n" + "-" * 50)
    print("MANUSCRIPT FIGURE MAPPING")
    print("-" * 50)
    print("figure_void_vs_cai.png          → Figure 3")
    print("figure_defect_vs_fatigue.png    → Figure 4")
    print("figure_cai_distribution.png     → Figure 5")
    print("figure_sensitivity_ranking.png  → Figure 6")
    print("figure_reliability_distribution.png → Figure 7")
    print("figure_parameter_distributions.png → Figure 8 (Supplementary)")
    print("figure_defect_distributions.png → Figure 9 (Supplementary)")
    print("\nNOTE: Figure 1 (Framework Architecture) and Figure 2 (Convergence)")
    print("      must be created manually. Figure 2 can be generated using")
    print("      the provided HTML file or the convergence code snippet.")
    print("=" * 70)

if __name__ == "__main__":
    main()
