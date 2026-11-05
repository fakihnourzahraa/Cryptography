#!/usr/bin/env python3
"""entropy_lab — measure how much randomness the ESP32-CAM sensor really gives.

  # 1) capture: one run with a normal scene, one with the lens covered
  python entropy_lab.py capture --label live    --frames 40
  python entropy_lab.py capture --label covered --frames 40

  # 2) analyse and compare (prints a table, writes plots + NIST input files)
  python entropy_lab.py analyze data/live.npz data/covered.npz

Why raw frames: the 32-byte blocks the firmware sends are SHA-256 outputs, and
a hash output always *looks* random, even if the input had almost no entropy.
So the honest measurement is on the raw pixels, before hashing.

Outputs (in report/):
  summary.csv                 one row per dataset
  <label>_overview.png        example frame, per-pixel noise map, residual histogram, LSB plane
  comparison.png              live vs covered on the key metrics
  nist/<label>_lsb.bin        1 sample per byte, 1 bit/sample   -> ea_non_iid <file> 1
  nist/<label>_diff.bin       frame-to-frame differences mod 256 -> ea_non_iid <file> 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from pqcam.source import open_source

NIST_SAMPLES = 1_000_000          # SP 800-90B wants at least 1e6 samples
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ------------------------------------------------------------------ capture
def cmd_capture(args):
    src = open_source(args.port, mock=args.mock, covered=args.label.startswith("cover"))
    if args.mock:
        print("!! MOCK source: synthetic frames, only for testing the pipeline.", file=sys.stderr)
    print(src.info())
    frames = []
    t0 = time.perf_counter()
    for i in range(args.frames):
        frames.append(src.raw_frame())
        print(f"\rframe {i + 1}/{args.frames}", end="", flush=True)
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    src.close()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.label}.npz")
    np.savez_compressed(path, frames=np.stack(frames), label=args.label,
                        source=getattr(src, "name", "?"), captured=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"saved {len(frames)} frames to {path}")


# ------------------------------------------------------------------ metrics
def shannon(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def min_entropy(values: np.ndarray) -> float:
    """Most-common-value estimate: -log2(p_max). Conservative, what crypto cares about."""
    _, counts = np.unique(values, return_counts=True)
    return float(-np.log2(counts.max() / counts.sum()))


def analyse(frames: np.ndarray) -> dict:
    f = frames.astype(np.int16)
    n, h, w = f.shape
    diff = f[1:] - f[:-1]                       # removes the static scene
    temporal_sigma = float(diff.std() / np.sqrt(2))
    residual = f - np.rint(f.mean(axis=0)).astype(np.int16)
    lsb = (frames & 1).astype(np.uint8)
    p1 = float(lsb.mean())
    flat = lsb.reshape(-1).astype(np.int8)
    lsb_corr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if flat.std() else 1.0
    h_res = min_entropy(residual)
    stuck = float((frames.std(axis=0) == 0).mean())   # pixels that never changed
    return {
        "frames": n, "width": w, "height": h,
        "mean_brightness": float(frames.mean()),
        "temporal_noise_sigma": temporal_sigma,
        "residual_shannon_bits": shannon(residual),
        "residual_min_entropy_bits": h_res,
        "lsb_ones_fraction": p1,
        "lsb_min_entropy_bits": float(-np.log2(max(p1, 1 - p1))),
        "lsb_neighbour_corr": lsb_corr,
        "stuck_pixel_fraction": stuck,
        # Upper-bound style estimate assuming independent pixels; the NIST tool
        # is the real check because neighbouring pixels are correlated.
        "est_bits_per_frame": h_res * h * w * (1 - stuck),
    }


def export_nist(frames: np.ndarray, label: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    lsb = (frames & 1).astype(np.uint8).reshape(-1)[:NIST_SAMPLES]
    diff = ((frames[1:].astype(np.int16) - frames[:-1]) % 256).astype(np.uint8).reshape(-1)[:NIST_SAMPLES]
    lsb.tofile(os.path.join(outdir, f"{label}_lsb.bin"))
    diff.tofile(os.path.join(outdir, f"{label}_diff.bin"))
    return len(lsb), len(diff)


# ------------------------------------------------------------------ plots
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })


def plot_overview(frames, label, m, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    seq = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.suptitle(f"{label}: {m['frames']} frames, {m['width']}x{m['height']}", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")

    a = ax[0, 0]; a.grid(False)
    a.imshow(frames[0], cmap="gray", vmin=0, vmax=255); a.set_title("Example frame"); a.axis("off")

    a = ax[0, 1]; a.grid(False)
    im = a.imshow(frames.astype(np.float32).std(axis=0), cmap=seq)
    a.set_title("Per-pixel noise (std over frames)"); a.axis("off")
    cb = fig.colorbar(im, ax=a, fraction=0.046); cb.outline.set_visible(False)
    cb.set_label("grey levels", color=INK2)

    a = ax[1, 0]
    res = (frames.astype(np.int16) - np.rint(frames.mean(axis=0))).reshape(-1)
    lo, hi = np.percentile(res, [0.1, 99.9]).astype(int)
    bins = np.arange(lo - 0.5, hi + 1.5)
    a.hist(res, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    a.set_title("Noise residual (pixel minus its mean)")
    a.set_xlabel("grey levels"); a.set_ylabel("pixels")
    a.text(0.98, 0.95, f"σ = {m['temporal_noise_sigma']:.2f}\nmin-entropy = {m['residual_min_entropy_bits']:.2f} bits/px",
           transform=a.transAxes, ha="right", va="top", color=INK2)

    a = ax[1, 1]; a.grid(False)
    a.imshow(frames[0] & 1, cmap="gray", interpolation="nearest")
    a.set_title(f"Least significant bit (ones = {m['lsb_ones_fraction']:.3f})"); a.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_comparison(results, path):
    import matplotlib.pyplot as plt

    _style(plt)
    metrics = [("temporal_noise_sigma", "Temporal noise σ (grey levels)"),
               ("residual_min_entropy_bits", "Residual min-entropy (bits/pixel)"),
               ("lsb_min_entropy_bits", "LSB min-entropy (bits/bit)")]
    labels = [r["label"] for r in results]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for a, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in results]
        colors = [SERIES[i % len(SERIES)] for i in range(len(results))]
        bars = a.bar(labels, vals, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
        a.set_title(title, loc="left"); a.grid(axis="x", visible=False)
        for b, v in zip(bars, vals):
            a.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                       textcoords="offset points", ha="center", color=INK2)
        a.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def cmd_analyze(args):
    os.makedirs(args.report, exist_ok=True)
    results = []
    for path in args.datasets:
        d = np.load(path)
        frames, label = d["frames"], str(d["label"])
        if len(frames) < 2:
            sys.exit(f"{path}: need at least 2 frames")
        m = analyse(frames)
        m["label"], m["source"] = label, str(d["source"])
        n_lsb, n_diff = export_nist(frames, label, os.path.join(args.report, "nist"))
        m["nist_lsb_samples"], m["nist_diff_samples"] = n_lsb, n_diff
        plot_overview(frames, label, m, os.path.join(args.report, f"{label}_overview.png"))
        results.append(m)

    keys = ["label", "source", "frames", "mean_brightness", "temporal_noise_sigma",
            "residual_shannon_bits", "residual_min_entropy_bits", "lsb_ones_fraction",
            "lsb_min_entropy_bits", "lsb_neighbour_corr", "stuck_pixel_fraction",
            "est_bits_per_frame", "nist_lsb_samples", "nist_diff_samples"]
    with open(os.path.join(args.report, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})
    if len(results) > 1:
        plot_comparison(results, os.path.join(args.report, "comparison.png"))

    rows = [("dataset", lambda r: r["label"]),
            ("mean brightness", lambda r: f"{r['mean_brightness']:.1f}"),
            ("temporal noise σ", lambda r: f"{r['temporal_noise_sigma']:.2f}"),
            ("residual Shannon (bits/px)", lambda r: f"{r['residual_shannon_bits']:.2f}"),
            ("residual min-entropy (bits/px)", lambda r: f"{r['residual_min_entropy_bits']:.2f}"),
            ("LSB ones fraction", lambda r: f"{r['lsb_ones_fraction']:.4f}"),
            ("LSB neighbour correlation", lambda r: f"{r['lsb_neighbour_corr']:+.4f}"),
            ("stuck pixels", lambda r: f"{100 * r['stuck_pixel_fraction']:.2f}%"),
            ("est. bits/frame (upper bound)", lambda r: f"{r['est_bits_per_frame']:,.0f}")]
    width = max(len(n) for n, _ in rows)
    for name, fn in rows:
        print(name.ljust(width), *[str(fn(r)).rjust(14) for r in results])
    print(f"\nplots and summary.csv in {args.report}/")
    print(f"NIST SP 800-90B: ea_non_iid -v {args.report}/nist/<label>_lsb.bin 1"
          f"   |   ea_non_iid -v {args.report}/nist/<label>_diff.bin 8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--mock", action="store_true", help="synthetic frames (pipeline test only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capture raw frames")
    c.add_argument("--label", required=True, help="e.g. live, covered")
    c.add_argument("--frames", type=int, default=40)
    c.add_argument("--out", default="data")
    a = sub.add_parser("analyze", help="analyse captured datasets")
    a.add_argument("datasets", nargs="+")
    a.add_argument("--report", default="report")
    args = p.parse_args(argv)
    {"capture": cmd_capture, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
