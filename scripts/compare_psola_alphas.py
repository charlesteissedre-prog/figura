"""Quick LTAS-similarity-to-target sweep across the PSOLA alpha variants."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pyworld as pw

REPO = Path(__file__).resolve().parent.parent
SR = 24000
FRAME_PERIOD_MS = 32 / 3

VARIANTS = {
    "raw vevo2-out (no PSOLA)": REPO / "outputs" / "_female_vevo2_nopitch.wav",
    "alpha=1.00":               REPO / "outputs" / "_female_vevo2_nopitch_uniform_psola.wav",
    "alpha=0.90":               REPO / "outputs" / "_female_vevo2_psola_a090.wav",
    "alpha=0.80":               REPO / "outputs" / "_female_vevo2_psola_a080.wav",
    "alpha=0.70":               REPO / "outputs" / "_female_vevo2_psola_a070.wav",
    "alpha=0.52":               REPO / "outputs" / "_female_vevo2_psola_overcorrect.wav",
}
SRC = REPO / "examples" / "female" / "source.wav"
TGT = REPO / "examples" / "female" / "target.wav"


def load_24k(p):
    a, sr = sf.read(str(p))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    return a.astype(np.float32)


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


def mean_f0(audio):
    _f0, t = pw.harvest(audio.astype(np.float64), fs=SR, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(audio.astype(np.float64), _f0, t, fs=SR)
    return float(f0[f0 > 0].mean())


def main():
    src = load_24k(SRC)
    tgt = load_24k(TGT)
    src_mean = mean_f0(src)
    src_ltas = ltas(src)
    tgt_ltas = ltas(tgt)

    freqs = np.linspace(0, SR / 2, len(src_ltas))
    mask = (freqs >= 200) & (freqs <= 5000)
    src_l = src_ltas[mask]
    tgt_l = tgt_ltas[mask]
    print(f"source mean F0: {src_mean:.1f} Hz")
    print(f"src<->tgt LTAS sim (reference): {cos_sim(src_l, tgt_l):.4f}\n")

    print(f"{'variant':30s}  {'mean F0':>9s}  {'st vs src':>10s}  "
          f"{'LTAS↔tgt':>9s}  {'LTAS↔src':>9s}")
    print("-" * 80)
    for name, path in VARIANTS.items():
        a = load_24k(path)
        mf0 = mean_f0(a)
        l = ltas(a)[mask]
        st_vs_src = np.log2(mf0 / src_mean) * 12
        print(f"{name:30s}  {mf0:7.1f} Hz  {st_vs_src:+8.2f} st  "
              f"{cos_sim(l, tgt_l):.4f}    {cos_sim(l, src_l):.4f}")


if __name__ == "__main__":
    main()
