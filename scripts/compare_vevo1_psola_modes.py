"""
Compare Vevo 1 PSOLA modes head-to-head: raw vs uniform-shift vs full-contour.

Question: is the existing per-frame contour transplant doing real work, or
would a uniform mean-shift be enough to unify Vevo 1 and Vevo 2 under a single
PSOLA algorithm?
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent.parent
SR = 24000
FRAME_PERIOD_MS = 32 / 3

SRC = REPO / "examples" / "female" / "source.wav"
TGT = REPO / "examples" / "female" / "target.wav"
VARIANTS = {
    "vevo1 raw":           REPO / "outputs" / "_female_vevo1_raw.wav",
    "vevo1 + uniform":     REPO / "outputs" / "_female_vevo1_uniform.wav",
    "vevo1 + contour":     REPO / "outputs" / "_female_vevo1_contour.wav",
}
COLORS = {
    "source":          "tab:blue",
    "target":          "tab:orange",
    "vevo1 raw":       "tab:gray",
    "vevo1 + uniform": "tab:green",
    "vevo1 + contour": "tab:red",
}


def load_24k(p):
    a, sr = sf.read(str(p))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    return a.astype(np.float32)


def log2_f0_contour(audio):
    _f0, t = pw.harvest(audio.astype(np.float64), fs=SR, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(audio.astype(np.float64), _f0, t, fs=SR)
    voiced = f0 > 0
    lf0 = np.where(voiced, np.log2(np.maximum(f0, 1e-5)), np.nan)
    times = np.arange(len(lf0)) * FRAME_PERIOD_MS / 1000.0
    return times, lf0, f0


def ltas(audio, n_fft=4096):
    win = np.hanning(n_fft)
    hop = n_fft // 2
    mags = []
    for start in range(0, max(1, len(audio) - n_fft), hop):
        frame = audio[start:start + n_fft] * win
        if len(frame) < n_fft:
            break
        mags.append(np.abs(np.fft.rfft(frame)))
    avg_db = 20 * np.log10(np.mean(np.stack(mags), axis=0) + 1e-8)
    k = 81
    kernel = np.hanning(k); kernel /= kernel.sum()
    return np.convolve(avg_db, kernel, mode="same")


def cos_sim(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    src_audio = load_24k(SRC)
    tgt_audio = load_24k(TGT)
    variants = {name: load_24k(p) for name, p in VARIANTS.items()}

    # ---- F0 stats ----
    print("=== PITCH ===")
    src_t, src_lf0, src_f0 = log2_f0_contour(src_audio)
    src_mean = float(src_f0[src_f0 > 0].mean())
    print(f"source mean: {src_mean:.1f} Hz")
    for name, a in variants.items():
        _, _, f0 = log2_f0_contour(a)
        m = float(f0[f0 > 0].mean())
        st = np.log2(m / src_mean) * 12
        # Also report contour RMSE to source on overlapping voiced frames.
        # Need to align lengths — resample short to long.
        _, var_lf0, _ = log2_f0_contour(a)
        n = min(len(src_lf0), len(var_lf0))
        s = src_lf0[:n]
        v = var_lf0[:n]
        both_voiced = ~np.isnan(s) & ~np.isnan(v)
        if both_voiced.any():
            rmse_st = float(np.sqrt(np.nanmean((s[both_voiced] - v[both_voiced]) ** 2)) * 12)
        else:
            rmse_st = float("nan")
        print(f"  {name:18s}  mean {m:6.1f} Hz  ({st:+5.2f} st vs src)   "
              f"contour RMSE vs src = {rmse_st:.2f} st")

    # ---- LTAS (target = right reference for timbre) ----
    print("\n=== TIMBRE (LTAS cosine sim, 200–5000 Hz) ===")
    src_l_full = ltas(src_audio)
    tgt_l_full = ltas(tgt_audio)
    freqs = np.linspace(0, SR / 2, len(src_l_full))
    mask = (freqs >= 200) & (freqs <= 5000)
    src_l = src_l_full[mask]
    tgt_l = tgt_l_full[mask]
    print(f"  src↔tgt: {cos_sim(src_l, tgt_l):.4f}  (reference)")
    for name, a in variants.items():
        l = ltas(a)[mask]
        print(f"  {name:18s}  ↔tgt={cos_sim(l, tgt_l):.4f}   ↔src={cos_sim(l, src_l):.4f}")

    # ---- Plot F0 contours ----
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(src_t, src_lf0, color=COLORS["source"], lw=1.4, label="source", alpha=0.9)
    _, tgt_lf0, _ = log2_f0_contour(tgt_audio)
    tgt_t = np.arange(len(tgt_lf0)) * FRAME_PERIOD_MS / 1000.0
    ax.plot(tgt_t, tgt_lf0, color=COLORS["target"], lw=1.0, label="target", alpha=0.6)
    for name, a in variants.items():
        t, lf0, _ = log2_f0_contour(a)
        ax.plot(t, lf0, color=COLORS[name], lw=1.3, label=name, alpha=0.85)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("log2(Hz)")
    ax.set_title("Vevo 1 PSOLA modes — F0 contours")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    out_f0 = REPO / "outputs" / "_vevo1_psola_modes_f0.png"
    fig.tight_layout()
    fig.savefig(out_f0, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_f0}")


if __name__ == "__main__":
    main()
