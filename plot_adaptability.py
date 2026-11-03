#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare logged epoch-final A_mean values for 7 distillation baselines.

Input: Ultralytics tqdm text logs with columns:
Epoch GPU_mem box_loss cls_loss dfl_loss log_loss fea_loss rev_loss A_mean Instances Size

Important:
  A_lc = (KT * KS).detach() is a genuine layer-category score.
  A_mean = mean(all_A) is its aggregated scalar.
  The output reflects EPOCH-LEVEL AGGREGATE LOG VALUES, not the distribution
  of individual A_lc scores. Per-layer / per-category data cannot be
  reconstructed from the mean alone.

Usage:
  python plot_adaptability.py --log-dir /home/lenovo/data/liujiaji/powerGit/distill/logs --out-dir a_analysis
  python plot_adaptability.py --log-dir /path/to/logs --smooth 5

Dependencies: numpy, matplotlib (pip install numpy matplotlib)
"""

import argparse
import csv
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = {
    "Mimic+Ours": "11mimic-aks-rev_vis.log",
    "SPKD+Ours": "11sp-aks-rev_vis.log",
    "ICKD+Ours": "11chsim-aks-rev_vis.log",
    "CWD+Ours": "11cwd-aks-rev_vis.log",
    "PKD+Ours": "11pkd-aks-rev_vis.log",
    "MGD+Ours": "11mgd-aks-rev_vis.log",
    "SCD+Ours": "11scd-aks-rev_vis.log",
    "SL2+Ours": "11sl2-aks-rev_vis.log",
}

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+\d+(?:\.\d+)?G\s")


def load_epoch_last_values(path):
    """
    Ultralytics tqdm prints each evolving epoch row twice; the last
    parsable A_mean for a given epoch is retained (no double counting).
    This assumes the last displayed value is the epoch aggregate being logged.
    """
    final = {}
    seen = {}
    expected = set()
    with path.open("r", encoding="utf-8", errors="replace", newline=None) as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            match = EPOCH_RE.match(line)
            if match is None:
                continue
            ep, total = int(match.group(1)), int(match.group(2))
            expected.add(total)
            parts = line.split(":", 1)[0].split()
            # Epoch, GPU_mem, box, cls, dfl, log, fea, rev, A_mean, Instances, Size
            if len(parts) != 11:
                continue
            try:
                a = float(parts[8])
                instances = int(parts[9])
                image_size = int(parts[10])
            except ValueError:
                continue
            if not np.isfinite(a) or instances < 0 or image_size <= 0:
                continue
            final[ep] = a
            seen[ep] = seen.get(ep, 0) + 1

    expected_epochs = max(expected, default=None)
    return dict(sorted(final.items())), expected_epochs, seen


def rolling_mean(x, window):
    """Centered moving average, visualization only, no modification to CSV."""
    if window <= 1:
        return np.asarray(x, dtype=float)
    arr = np.asarray(x, dtype=float)
    out = np.empty(len(arr), dtype=float)
    left, right = (window - 1) // 2, window // 2
    for i in range(len(arr)):
        out[i] = np.mean(arr[max(0, i - left):min(len(arr), i + right + 1)])
    return out


def save_csv(output, datasets):
    all_epochs = sorted({e for series in datasets.values() for e in series})
    with (output / "A_mean_epoch_series.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch"] + list(datasets))
        for ep in all_epochs:
            writer.writerow([ep] + [
                f"{series[ep]:.8g}" if ep in series else ""
                for series in datasets.values()
            ])

    with (output / "A_mean_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "N_epochs", "First_epoch", "First_A_mean",
            "Last_epoch", "Last_A_mean", "Min_epoch_mean", "Max_epoch_mean",
            "Mean_of_epoch_means", "Std_of_epoch_means",
            "P10_of_epoch_means", "Median_of_epoch_means",
            "P90_of_epoch_means", "First_20_epoch_mean",
            "Last_20_epoch_mean", "Relative_change_percent"
        ])
        for method, series in datasets.items():
            x = np.asarray(list(series), dtype=int)
            a = np.asarray(list(series.values()), dtype=float)
            baseline = a[0]
            percentage = 100 * (a[-1] - baseline) / baseline if baseline != 0 else float("nan")
            r = [method, len(series), x[0], a[0], x[-1], a[-1],
                 np.min(a), np.max(a), np.mean(a), np.std(a),
                 np.percentile(a, 10), np.median(a), np.percentile(a, 90),
                 np.mean(a[:20]), np.mean(a[-20:]), percentage]
            writer.writerow([
                v if isinstance(v, (int, np.integer, str)) else f"{v:.8g}"
                for v in r
            ])


def style_plot(ax, title, ylabel, xmax):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=1, right=max(2, xmax))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9, ncol=2)


def save_line_figure(out, datasets, smooth, filename, normalized=False, zoom_end=None):
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    for method, series in datasets.items():
        x = np.asarray(list(series.keys()))
        y = np.asarray(list(series.values()))
        if normalized:
            if abs(y[0]) < 1e-12:
                print(f"[WARN] {method} starts at zero; skip normalized curve")
                continue
            y = y / y[0]
        y = rolling_mean(y, smooth)
        if zoom_end is not None:
            mask = x <= zoom_end
            x, y = x[mask], y[mask]
        if not len(x):
            continue
        ax.plot(x, y, linewidth=1.7, label=method)

    title = (
        "Normalized A_mean Dynamics (Epoch 1 = 1)"
        if normalized else
        ("Early-stage A_mean Dynamics" if zoom_end is not None else
         "Average Adaptability Score Across Training")
    )
    ylabel = "A_mean / initial A_mean" if normalized else "A_mean (logged)"
    xmax = zoom_end if zoom_end is not None else max(max(s) for s in datasets.values())
    style_plot(ax, title, ylabel, xmax)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=300)
    plt.close(fig)


def save_boxplot(out, datasets):
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    labels = list(datasets)
    ax.boxplot([list(s.values()) for s in datasets.values()], showfliers=True)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Epoch-final A_mean")
    ax.set_title("Distribution of Epoch-level Average Scores")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "A_mean_epoch_boxplot.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("a_analysis"))
    parser.add_argument("--smooth", type=int, default=1,
                        help="Centered rolling window for figures (CSV remains unsmoothed)")
    parser.add_argument("--zoom-epochs", type=int, default=40)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for method, filename in FILES.items():
        path = args.log_dir / filename
        if not path.exists():
            print(f"[SKIP] {filename}: not found")
            continue
        series, total, seen = load_epoch_last_values(path)
        if not series:
            print(f"[SKIP] {filename}: no matching epoch A_mean rows")
            continue
        datasets[method] = series
        missing = sorted(set(range(1, total + 1)) - set(series)) if total else []
        print(
            f"[OK] {method}: {len(series)}/{total or '?'} epochs, "
            f"epoch {min(series)}={series[min(series)]:.6f}, "
            f"epoch {max(series)}={series[max(series)]:.6f}"
        )
        if missing:
            print(f"     Missing epochs: {missing[:12]}{' ...' if len(missing)>12 else ''}")

    if not datasets:
        raise SystemExit("No valid log found; check directory and headers.")

    save_csv(args.out_dir, datasets)
    save_line_figure(args.out_dir, datasets, args.smooth,
                     "A_mean_comparison.png")
    save_line_figure(args.out_dir, datasets, args.smooth,
                     "A_mean_early_epochs.png",
                     zoom_end=args.zoom_epochs)
    save_line_figure(args.out_dir, datasets, args.smooth,
                     "A_mean_normalized.png", normalized=True)
    save_boxplot(args.out_dir, datasets)
    print("\nSaved outputs to:", args.out_dir.resolve())
    print("Interpretation: an epoch-level mean curve does not reveal individual")
    print("A_(l,c) distributions or directly prove suppression of noisy knowledge.")


if __name__ == "__main__":
    main()
