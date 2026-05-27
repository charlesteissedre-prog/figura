"""
Compare source / target / Vevo 2 output along pitch and timbre axes.

Produces two PNGs and prints numerical similarity metrics:
  - _vevo2_nopitch_f0.png    : F0 contours overlaid (pitch comparison)
  - _vevo2_nopitch_ltas.png  : long-term-average spectra (timbre comparison;
                               peaks are formants)

If output's F0 stats track source AND its LTAS shape tracks target, then
source-pitch + target-timbre is delivered without PitchFlower.
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

FILES = {
    "source":         REPO / "examples" / "female" / "source.wav",
    "target":         REPO / "examples" / "female" / "target.wav",
    "vevo2-out":      REPO / "outputs" / "_female_vevo2_nopitch.wav",
    "vevo2+psola":    REPO / "outputs" / "_female_vevo2_nopitch_uniform_psola.wav",
}
COLORS = {
    "source": "tab:blue", "target": "tab:orange",
    "vevo2-out": "tab:green", "vevo2+psola": "tab:red",
}


def load_24k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def f0_hz(audio: np.ndarray):
    _f0, t = pw.harvest(audio.astype(np.float64), fs=SR, frame_period=FRAME_PERIOD_MS)
    return pw.stonemask(audio.astype(np.float64), _f0, t, fs=SR)


def log2_f0_contour(audio: np.ndarray):
    f0 = f0_hz(audio)
    voiced = f0 > 0
    lf0 = np.where(voiced, np.log2(np.maximum(f0, 1e-5)), np.nan)
    times = np.arange(len(lf0)) * FRAME_PERIOD_MS / 1000.0
    return times, lf0


def ltas(audio: np.ndarray, n_fft: int = 4096):
    """Long-term-average spectrum (dB), smoothed to suppress harmonics."""
    win = np.hanning(n_fft)
    hop = n_fft // 2
    mags = []
    for start in range(0, max(1, len(audio) - n_fft), hop):
        frame = audio[start:start + n_fft] * win
        if len(frame) < n_fft:
            break
        m = np.abs(np.fft.rfft(frame))
        mags.append(m)
    avg = np.mean(np.stack(mags), axis=0)
    avg_db = 20 * np.log10(avg + 1e-8)
    k = 81
    kernel = np.hanning(k); kernel /= kernel.sum()
    smoothed = np.convolve(avg_db, kernel, mode="same")
    freqs = np.linspace(0, SR / 2, len(smoothed))
    return freqs, smoothed


def main():
    audios = {name: load_24k(p) for name, p in FILES.items()}

    # ---------- F0 stats + plot ----------
    print("=== PITCH (F0 over voiced frames, Hz) ===")
    f0_stats = {}
    for name, audio in audios.items():
        f0 = f0_hz(audio)
        voiced = f0[f0 > 0]
        mean, median, std = float(voiced.mean()), float(np.median(voiced)), float(voiced.std())
        f0_stats[name] = (mean, median, std)
        print(f"  {name:10s}  mean={mean:6.1f} Hz  median={median:6.1f} Hz  std={std:5.1f} Hz  "
              f"({len(voiced)/len(f0)*100:.0f}% voiced)")
    s_mean = f0_stats["source"][0]
    t_mean = f0_stats["target"][0]
    for label in ("vevo2-out", "vevo2+psola"):
        if label not in f0_stats:
            continue
        o_mean = f0_stats[label][0]
        s_st = np.log2(o_mean / s_mean) * 12
        t_st = np.log2(o_mean / t_mean) * 12
        closer = "SOURCE" if abs(o_mean - s_mean) < abs(o_mean - t_mean) else "TARGET"
        print(f"\n  {label}: {s_st:+.2f} st vs source, {t_st:+.2f} st vs target  → closer to {closer}")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for name, audio in audios.items():
        t, lf0 = log2_f0_contour(audio)
        mean_hz = f0_stats[name][0]
        ax.plot(t, lf0, color=COLORS[name], lw=1.3,
                label=f"{name} (mean {mean_hz:.0f} Hz, {len(audio)/SR:.2f}s)")
    ax.set_title("F0 contour — source vs target vs Vevo 2 output (use_pitch_shift=False)")
    ax.set_ylabel("log2(Hz)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    out_f0 = REPO / "outputs" / "_vevo2_psola_f0.png"
    fig.tight_layout()
    fig.savefig(out_f0, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_f0}")

    # ---------- LTAS plot + cosine sim metric ----------
    ltas_data = {name: ltas(audio) for name, audio in audios.items()}

    def cos_sim(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Limit similarity to speech-formant band for fair comparison.
    freqs = ltas_data["source"][0]
    mask = (freqs >= 200) & (freqs <= 5000)
    src_l = ltas_data["source"][1][mask]
    tgt_l = ltas_data["target"][1][mask]
    print("\n=== TIMBRE (LTAS cosine similarity, 200–5000 Hz) ===")
    print(f"  source vs target: {cos_sim(src_l, tgt_l):.3f}  (reference: how different src & tgt are)")
    for label in ("vevo2-out", "vevo2+psola"):
        if label not in ltas_data:
            continue
        out_l = ltas_data[label][1][mask]
        s, t = cos_sim(out_l, src_l), cos_sim(out_l, tgt_l)
        closer = "TARGET" if t > s else "SOURCE"
        print(f"  {label}: vs source={s:.3f}  vs target={t:.3f}  → closer to {closer}")

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, (f, db) in ltas_data.items():
        ax.plot(f, db, color=COLORS[name], lw=1.5, label=name)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_title("LTAS — peaks are formants; this is the timbre fingerprint")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    out_ltas = REPO / "outputs" / "_vevo2_psola_ltas.png"
    fig.tight_layout()
    fig.savefig(out_ltas, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_ltas}")


if __name__ == "__main__":
    main()
