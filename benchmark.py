#!/usr/bin/env python3
"""benchmark — compare the post-quantum algorithms for the report (uses liboqs).

  python benchmark.py                 # all algorithms, 50 runs each
  python benchmark.py --runs 200
  python benchmark.py --camera        # also measure entropy throughput from the ESP32-CAM
  python benchmark.py --camera --mock

KEMs:       ML-KEM-512/768/1024 vs HQC vs BIKE vs NTRU vs Classic McEliece
Signatures: ML-DSA-44/65/87 vs Falcon (FN-DSA) vs SPHINCS+ (SLH-DSA)

Classic McEliece is OFF in default liboqs builds. To include it, rebuild
liboqs with -DOQS_ENABLE_KEM_CLASSIC_MCELIECE=ON (see README).

liboqs is optimised C, so its timings are the fair comparison between
algorithms. (pqcf.py uses pure-Python ML-KEM/ML-DSA so the camera RNG can be
plugged in visibly; those are much slower and aren't what we compare here.)
Only algorithms enabled in your liboqs build are run; others are skipped.

Writes report/benchmark.csv and report/benchmark_kem.png / benchmark_sig.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

KEM_WANTED = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
              "HQC-1", "HQC-3", "HQC-128", "HQC-192",          # HQC was renamed in newer liboqs
              "BIKE-L1", "BIKE-L3", "NTRU-HPS-2048-509", "NTRU-HPS-2048-677",
              "Classic-McEliece-348864", "Classic-McEliece-460896"]
SIG_WANTED = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
              "Falcon-512", "Falcon-1024",
              "SPHINCS+-SHA2-128f-simple", "SPHINCS+-SHA2-128s-simple",
              "SLH_DSA_PURE_SHA2_128F", "SLH_DSA_PURE_SHA2_128S"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
FAMILY = {"ML-KEM": 0, "HQC": 1, "Classic-McEliece": 2, "BIKE": 3, "NTRU": 4,
          "ML-DSA": 0, "Falcon": 1, "SPHINCS+": 2, "SLH_DSA": 2}


def timed(fn, runs):
    times = []
    out = None
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times), out


def family(name):
    for prefix, i in FAMILY.items():
        if name.startswith(prefix):
            return prefix, i
    return name, 4


def bench_kems(oqs, runs):
    enabled = set(oqs.get_enabled_kem_mechanisms())
    rows = []
    for name in KEM_WANTED:
        if name not in enabled:
            print(f"  skip {name} (not enabled in this liboqs build)")
            continue
        n = max(3, runs // 10) if "McEliece" in name else runs   # McEliece keygen is slow
        with oqs.KeyEncapsulation(name) as kem:
            t_kg, pk = timed(kem.generate_keypair, n)
            t_enc, (ct, ss) = timed(lambda: kem.encap_secret(pk), runs)
            t_dec, ss2 = timed(lambda: kem.decap_secret(ct), runs)
            assert ss == ss2
            d = kem.details
            rows.append({"kind": "KEM", "algorithm": name, "nist_level": d["claimed_nist_level"],
                         "keygen_ms": t_kg, "op1_ms": t_enc, "op2_ms": t_dec,
                         "public_key_B": d["length_public_key"], "secret_key_B": d["length_secret_key"],
                         "output_B": d["length_ciphertext"]})
            print(f"  {name:28s} keygen {t_kg:9.3f} ms  encaps {t_enc:7.3f} ms  decaps {t_dec:7.3f} ms  "
                  f"pk {d['length_public_key']:>7} B  ct {d['length_ciphertext']:>6} B")
    return rows


def bench_sigs(oqs, runs):
    enabled = set(oqs.get_enabled_sig_mechanisms())
    rows = []
    msg = os.urandom(1024)
    for name in SIG_WANTED:
        if name not in enabled:
            print(f"  skip {name} (not enabled in this liboqs build)")
            continue
        n = max(3, runs // 10) if ("SPHINCS" in name or "SLH" in name) else runs
        with oqs.Signature(name) as sig:
            t_kg, pk = timed(sig.generate_keypair, n)
            t_sign, s = timed(lambda: sig.sign(msg), n)
            t_ver, ok = timed(lambda: sig.verify(msg, s, pk), runs)
            assert ok
            d = sig.details
            rows.append({"kind": "SIG", "algorithm": name, "nist_level": d["claimed_nist_level"],
                         "keygen_ms": t_kg, "op1_ms": t_sign, "op2_ms": t_ver,
                         "public_key_B": d["length_public_key"], "secret_key_B": d["length_secret_key"],
                         "output_B": d["length_signature"]})
            print(f"  {name:28s} keygen {t_kg:9.3f} ms  sign {t_sign:9.3f} ms  verify {t_ver:7.3f} ms  "
                  f"pk {d['length_public_key']:>6} B  sig {d['length_signature']:>6} B")
    return rows


def bench_camera(args):
    from pqcam.source import open_source
    src = open_source(args.port, mock=args.mock)
    print(f"\nEntropy source: {src.info()}")
    out = []
    for mode, label in (("E", "mixed block"), ("C", "camera-only block"), ("H", "hw RNG block")):
        n = 20
        t = time.perf_counter()
        for _ in range(n):
            src.block(mode)
        dt = (time.perf_counter() - t) / n
        print(f"  {label:18s} {dt * 1e3:8.1f} ms/block  ({32 / dt:8.0f} B/s of seed material)")
        out.append((label, dt))
    t = time.perf_counter()
    src.raw_frame()
    print(f"  raw frame          {(time.perf_counter() - t) * 1e3:8.1f} ms")
    src.close()


def plot(rows, kind, path, op_names):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False})
    rows = [r for r in rows if r["kind"] == kind]
    if not rows:
        return
    names = [r["algorithm"] for r in rows]
    colors = [SERIES[family(n)[1]] for n in names]
    panels = [("keygen_ms", "Key generation (ms, log)"), ("op1_ms", f"{op_names[0]} (ms, log)"),
              ("op2_ms", f"{op_names[1]} (ms, log)"), ("public_key_B", "Public key (bytes, log)"),
              ("output_B", f"{op_names[2]} (bytes, log)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 0.42 * len(rows) + 1.6), sharey=True)
    for a, (key, title) in zip(axes, panels):
        vals = [r[key] for r in rows]
        a.barh(names, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        a.set_xscale("log"); a.xaxis.set_minor_formatter(NullFormatter()); a.set_title(title, loc="left", fontsize=10, fontweight="bold")
        a.grid(axis="y", visible=False); a.invert_yaxis() if a is axes[0] else None
    fams = []
    for n in names:
        f = family(n)
        if f not in fams:
            fams.append(f)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for _, i in fams]
    fig.legend(handles, [f for f, _ in fams], loc="upper right", frameon=False, ncol=len(fams))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--camera", action="store_true", help="also time the entropy source")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--report", default="report")
    args = p.parse_args(argv)
    try:
        import oqs
    except ImportError:
        sys.exit("liboqs-python is not installed; see README (Setup, step 4)")
    print(f"liboqs {oqs.oqs_version()}\nKEMs:")
    rows = bench_kems(oqs, args.runs)
    print("Signatures:")
    rows += bench_sigs(oqs, args.runs)
    os.makedirs(args.report, exist_ok=True)
    with open(os.path.join(args.report, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows, "KEM", os.path.join(args.report, "benchmark_kem.png"), ("Encapsulate", "Decapsulate", "Ciphertext"))
    plot(rows, "SIG", os.path.join(args.report, "benchmark_sig.png"), ("Sign", "Verify", "Signature"))
    print(f"\nresults in {args.report}/benchmark.csv and benchmark_*.png "
          "(op1/op2 = encaps/decaps for KEMs, sign/verify for signatures)")
    if args.camera:
        bench_camera(args)


if __name__ == "__main__":
    main()
