from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path("output/pinn_benchmark_runs")
FIG_DIR = Path("output/benchmark_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS = [
    "wave",
    "diffusion",
    "burgers",
    "sod"
]

METHODS = [
    "vanilla",
    "acto",
    "sa",
    "sa_acto"
]

METHOD_LABELS = {
    "vanilla": "Vanilla PINN",
    "acto": "PINN-ACTO",
    "sa": "SA-PINN",
    "sa_acto": "SA-PINN-ACTO"
}

METHOD_COLORS = {
    "vanilla": "#4C72B0",
    "acto": "#55A868",
    "sa": "#C44E52",
    "sa_acto": "#8172B3"
}

METHOD_STYLES = {
    "vanilla": "-",
    "acto": "--",
    "sa": "-.",
    "sa_acto": ":"
}

sns.set_theme(style="whitegrid", context="talk")

# ----------------------------
# HELPERS
# ----------------------------
def load_run_csv(problem, method):
    path = BASE_DIR / problem / method / "history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {"epoch", "wall_time_sec", "train_loss", "l2_error"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.sort_values("epoch").reset_index(drop=True)
    df["problem"] = problem
    df["method"] = method
    return df

def make_metric_plot(df, problem, metric, xcol, ylabel, title, outname, logy=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.plot(
            sub[xcol],
            sub[metric],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=2.5,
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch" if xcol == "epoch" else "Wall time [s]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=11, ncols=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / outname, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# LOAD DATA
# ----------------------------
all_runs = []
missing_runs = []

for problem in PROBLEMS:
    for method in METHODS:
        df = load_run_csv(problem, method)
        if df is None:
            missing_runs.append((problem, method))
        else:
            all_runs.append(df)

if not all_runs:
    raise RuntimeError("No run CSV files found. Check BASE_DIR and filenames.")

data = pd.concat(all_runs, ignore_index=True)

# Save merged table for inspection
data.to_csv(FIG_DIR / "merged_training_logs.csv", index=False)

# ----------------------------
# PER-PROBLEM PLOTS
# ----------------------------
for problem in PROBLEMS:
    d = data[data["problem"] == problem].copy()

    make_metric_plot(
        d, problem, metric="train_loss", xcol="epoch",
        ylabel="Training loss",
        title=f"{problem.capitalize()} — Loss vs Epoch",
        outname=f"{problem}_loss_vs_epoch.png",
        logy=True
    )

    make_metric_plot(
        d, problem, metric="train_loss", xcol="wall_time_sec",
        ylabel="Training loss",
        title=f"{problem.capitalize()} — Loss vs Wall Time",
        outname=f"{problem}_loss_vs_time.png",
        logy=True
    )

    make_metric_plot(
        d, problem, metric="l2_error", xcol="epoch",
        ylabel=r"Relative $L^2$ error",
        title=f"{problem.capitalize()} — L2 Error vs Epoch",
        outname=f"{problem}_l2_vs_epoch.png",
        logy=True
    )

    make_metric_plot(
        d, problem, metric="l2_error", xcol="wall_time_sec",
        ylabel=r"Relative $L^2$ error",
        title=f"{problem.capitalize()} — L2 Error vs Wall Time",
        outname=f"{problem}_l2_vs_time.png",
        logy=True
    )

# ----------------------------
# SUMMARY TABLES
# ----------------------------
summary_rows = []
for problem in PROBLEMS:
    for method in METHODS:
        sub = data[(data["problem"] == problem) & (data["method"] == method)].copy()
        if sub.empty:
            continue

        row = {
            "problem": problem,
            "method": method,
            "final_epoch": int(sub["epoch"].iloc[-1]),
            "final_wall_time_sec": float(sub["wall_time_sec"].iloc[-1]),
            "final_train_loss": float(sub["train_loss"].iloc[-1]),
            "best_train_loss": float(sub["train_loss"].min()),
            "final_l2_error": float(sub["l2_error"].iloc[-1]),
            "best_l2_error": float(sub["l2_error"].min()),
            "time_to_best_l2": float(sub.loc[sub["l2_error"].idxmin(), "wall_time_sec"]),
            "epoch_to_best_l2": int(sub.loc[sub["l2_error"].idxmin(), "epoch"]),
        }
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(FIG_DIR / "benchmark_summary.csv", index=False)

# Pretty display tables
loss_rank = summary.sort_values(["problem", "best_train_loss"])
l2_rank = summary.sort_values(["problem", "best_l2_error"])

loss_rank.to_csv(FIG_DIR / "ranked_by_loss.csv", index=False)
l2_rank.to_csv(FIG_DIR / "ranked_by_l2.csv", index=False)

print("Saved figures and summary tables to:", FIG_DIR)
print("Missing runs:", missing_runs)
