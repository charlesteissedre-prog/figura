"""
Test a uniform PSOLA pitch shift to correct the ~0.8 semitone register drift
in Vevo 2's `--disable pitch` output.

Approach:
  1. Measure source's mean F0 (voiced frames, in log2-Hz)
  2. Measure output's mean F0 (same)
  3. Ratio = 2^(src_mean_log2 - out_mean_log2)
  4. PSOLA multiply-frequencies the output by that ratio
  5. Verify: re-extract F0 from corrected output, report new mean

Unlike the failed full-contour PSOLA transplant earlier in this session, this
applies the SAME multiplicative ratio to every pitch pulse — the cleanest
regime PSOLA operates in.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import parselmouth
import pyworld as pw
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SR = 24000
FRAME_PERIOD_MS = 32 / 3

DEFAULT_SRC = REPO / "examples" / "female" / "source.wav"
DEFAULT_IN = REPO / "outputs" / "_female_vevo2_nopitch.wav"


def voiced_mean_hz(audio_np, sr):
    _f0, t = pw.harvest(audio_np.astype(np.float64), fs=sr, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(audio_np.astype(np.float64), _f0, t, fs=sr)
    voiced = f0[f0 > 0]
    return float(voiced.mean()), float(np.median(voiced)), float(voiced.std())


def load_24k(path):
    a, sr = sf.read(str(path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    return a.astype(np.float32)


def uniform_psola_shift(audio_np, sr, ratio):
    """Apply a uniform multiplicative F0 shift via Praat PSOLA."""
    snd = parselmouth.Sound(audio_np.astype(np.float64), sampling_frequency=sr)
    manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.01, 60, 600)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Multiply frequencies",
                           0.0, snd.duration, ratio)
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
    resynth = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    return np.array(resynth.values[0]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SRC,
                    help="Audio whose F0 mean we want to land on")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN,
                    help="Audio to pitch-correct (e.g. Vevo 2 --disable pitch output)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output WAV path (default: alongside --input with suffix)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Single-pass compression compensation. Ignored if --max-iter>1.")
    ap.add_argument("--max-iter", type=int, default=1,
                    help="Number of PSOLA passes. Each pass uses the residual ratio from the previous.")
    ap.add_argument("--tol-st", type=float, default=0.05,
                    help="Stop iterating once |residual| < this many semitones")
    args = ap.parse_args()

    if args.output is None:
        suffix = f"_psola_iter{args.max_iter}" if args.max_iter > 1 else f"_psola_a{args.alpha:.2f}"
        args.output = args.input.with_name(args.input.stem + suffix + ".wav")

    src_audio = load_24k(args.source)
    in_audio = load_24k(args.input)

    src_mean, _, _ = voiced_mean_hz(src_audio, SR)
    in_mean, _, _ = voiced_mean_hz(in_audio, SR)
    print(f"source mean F0:    {src_mean:6.1f} Hz   (from {args.source.name})")
    print(f"input mean F0:     {in_mean:6.1f} Hz   (from {args.input.name})")
    print(f"target shift:      {np.log2(src_mean/in_mean)*12:+.2f} st\n")

    shifted = in_audio
    current_mean = in_mean
    for it in range(1, args.max_iter + 1):
        residual_st = np.log2(current_mean / src_mean) * 12
        if abs(residual_st) < args.tol_st:
            print(f"iter {it}: residual {residual_st:+.3f} st within tolerance — done.")
            break

        naive_ratio = src_mean / current_mean
        # alpha only used when single-pass (--max-iter=1); iteration uses naive ratio.
        requested_ratio = (float(np.power(naive_ratio, 1.0 / args.alpha))
                           if args.max_iter == 1 else naive_ratio)
        shifted = uniform_psola_shift(shifted, SR, requested_ratio)
        new_mean, _, _ = voiced_mean_hz(shifted, SR)
        applied_st = np.log2(new_mean / current_mean) * 12
        print(f"iter {it}: requested {np.log2(requested_ratio)*12:+.2f} st  → "
              f"applied {applied_st:+.2f} st  → mean {new_mean:.1f} Hz  "
              f"(residual vs src {np.log2(new_mean/src_mean)*12:+.2f} st)")
        current_mean = new_mean
    # Trim/pad to original length (PSOLA can produce slightly different length).
    if len(shifted) != len(in_audio):
        if len(shifted) > len(in_audio):
            shifted = shifted[: len(in_audio)]
        else:
            shifted = np.pad(shifted, (0, len(in_audio) - len(shifted)))

    # Trim/pad to original length (PSOLA can produce slightly different length).
    if len(shifted) != len(in_audio):
        if len(shifted) > len(in_audio):
            shifted = shifted[: len(in_audio)]
        else:
            shifted = np.pad(shifted, (0, len(in_audio) - len(shifted)))

    sf.write(str(args.output), shifted, SR)

    out_mean, _, out_std = voiced_mean_hz(shifted, SR)
    print(f"\nfinal:")
    print(f"  mean F0:       {out_mean:6.1f} Hz  (std {out_std:.1f})")
    print(f"  total shift:   {np.log2(out_mean/in_mean)*12:+.2f} st  "
          f"(target {np.log2(src_mean/in_mean)*12:+.2f} st)")
    print(f"  vs source:     {np.log2(out_mean/src_mean)*12:+.2f} st")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
