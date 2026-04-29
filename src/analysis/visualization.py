"""F0 contour (Praat) + mel-spectrogram PNG helpers for the library view."""
from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import librosa
import parselmouth


class F0Contour(TypedDict):
    times: list[float]
    hz: list[float | None]
    median_hz: float | None
    voiced_ratio: float


def extract_f0_praat(
    filepath: str,
    *,
    outer_floor: float = 60.0,
    outer_ceiling: float = 600.0,
    time_step: float = 0.01,
    octave_cost: float = 0.055,
):
    """Run Praat autocorrelation with the Hirst (2011) two-pass method.

    Pass 1 uses wide bounds (60–600 Hz) to measure the speaker's q25/q75.
    Pass 2 re-runs with floor=0.75×q25 and ceiling=1.5×q75 (Hirst's recipe).

    `octave_cost` is raised from Praat's default 0.01 to 0.055 on both
    passes — this is Praat's anti-halving knob (higher cost on choosing the
    subharmonic candidate), and is the real defence against the artefacts
    that make raw single-pass calls cluster a female voice's frames around
    ~75–150 Hz. The two-pass bounds tightening is the secondary defence.

    Returns (f0, times). f0 is in Hz with NaN for unvoiced frames; times in
    seconds. Shared by the profile-stats extractor and the contour renderer
    so both report identical F0 numbers.
    """
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch_cc(
        time_step=time_step,
        pitch_floor=outer_floor,
        pitch_ceiling=outer_ceiling,
        octave_cost=octave_cost,
    )
    f0 = pitch.selected_array["frequency"].astype(float)
    voiced = f0[f0 > 0]

    if voiced.size >= 20:
        q25 = float(np.percentile(voiced, 25))
        q75 = float(np.percentile(voiced, 75))
        refined_floor = max(outer_floor, q25 * 0.75)
        refined_ceiling = min(outer_ceiling, q75 * 1.5)
        # Guard: don't clip expressive voices if pass-1 was itself halved.
        refined_ceiling = max(refined_ceiling, float(np.median(voiced)) * 2.2)
        if refined_ceiling > refined_floor * 1.5:
            pitch = snd.to_pitch_cc(
                time_step=time_step,
                pitch_floor=refined_floor,
                pitch_ceiling=refined_ceiling,
                octave_cost=octave_cost,
            )
            f0 = pitch.selected_array["frequency"].astype(float)

    f0[f0 == 0] = np.nan
    times = np.asarray(pitch.xs(), dtype=float)
    return f0, times


def compute_f0_contour(
    filepath: str,
    *,
    max_points: int = 400,
) -> F0Contour:
    """Extract an F0 contour using Praat (two-pass, see extract_f0_praat).

    Returns times (s), f0 in Hz (None for unvoiced frames), median Hz, and
    voiced ratio. The contour is decimated to at most `max_points` samples
    so payloads stay small on long files.
    """
    f0, times = extract_f0_praat(filepath)

    n = len(f0)
    if n > max_points:
        step = int(math.ceil(n / max_points))
        f0 = f0[::step]
        times = times[::step]

    voiced_mask = np.isfinite(f0)
    voiced_hz = f0[voiced_mask]
    median_hz = round(float(np.median(voiced_hz)), 1) if voiced_hz.size else None
    voiced_ratio = float(voiced_mask.mean()) if voiced_mask.size else 0.0

    hz_out: list[float | None] = [
        None if not np.isfinite(v) else round(float(v), 1) for v in f0
    ]
    times_out = [round(float(t), 3) for t in times]
    return {
        "times": times_out,
        "hz": hz_out,
        "median_hz": median_hz,
        "voiced_ratio": round(voiced_ratio, 3),
    }


def _cache_path(filepath: str, cache_dir: Path, suffix: str) -> Path:
    src = Path(filepath)
    key = f"{src.stem}_{int(src.stat().st_mtime)}{suffix}"
    return cache_dir / key


def render_mel_png(
    filepath: str,
    cache_dir: Path,
    *,
    n_mels: int = 80,
    fmax: float = 8000.0,
    width_px: int = 900,
    height_px: int = 260,
) -> Path:
    """Render a log-mel spectrogram to a PNG (no axes, fills image edge-to-edge
    so x=0 maps to t=0 and x=width maps to t=duration for click-seeking)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # ".v2" suffix invalidates pre-clickable cached renders with axes.
    out = _cache_path(filepath, cache_dir, ".v2.png")
    if out.exists():
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = librosa.load(filepath, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=fmax, ax=ax, cmap="magma"
    )
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor("#1a1a1a")
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    return out


def audio_duration_s(filepath: str) -> float:
    """Lightweight duration probe (reads only the audio header)."""
    return float(librosa.get_duration(path=filepath))
