"""
Voice profile extractor.
Runs all acoustic analysis tools on a dropped audio file and returns
a VoiceProfile dict ready for the UI and comparison view.
"""
import numpy as np
import parselmouth
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceProfile:
    # Identity
    name: str
    filepath: str
    duration_s: float
    sample_rate: int

    # Acoustic measurements (auto-filled on import)
    f0_mean_hz: Optional[float] = None
    f0_min_hz:  Optional[float] = None
    f0_max_hz:  Optional[float] = None
    hnr_db:     Optional[float] = None
    f1_mean_hz: Optional[float] = None
    f2_mean_hz: Optional[float] = None
    f3_mean_hz: Optional[float] = None
    jitter_pct: Optional[float] = None
    shimmer_pct: Optional[float] = None
    cpp_db:     Optional[float] = None
    speech_rate_syl_per_s: Optional[float] = None

    # Perceptual dimensions (0-100, auto-suggested, human-editable)
    brightness:  Optional[float] = None  # proxy: spectral centroid
    breathiness: Optional[float] = None  # proxy: CPP / HNR inverse
    nasality:    Optional[float] = None  # no reliable proxy — manual only
    roughness:   Optional[float] = None  # proxy: jitter + shimmer composite
    tension:     Optional[float] = None  # proxy: F0 range + HNR composite

    # Perceptual tags (human-editable list of strings)
    tags: list = field(default_factory=list)

    # Free-text notes
    notes: str = ""

    # User-set gender: "female" | "male" | "neutral" | None
    gender: Optional[str] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


def extract(filepath: str, name: str = "") -> VoiceProfile:
    """Main entry point. Pass a path to any audio file supported by Praat.
    Returns a fully populated VoiceProfile (perceptual dims are suggestions)."""
    snd = parselmouth.Sound(filepath)
    profile = VoiceProfile(
        name=name or filepath.split("/")[-1],
        filepath=filepath,
        duration_s=round(snd.duration, 2),
        sample_rate=int(snd.sampling_frequency),
    )
    _extract_f0(snd, profile)
    _extract_formants(snd, profile)
    _extract_voice_quality(snd, profile)
    _suggest_perceptual_dims(snd, profile)
    return profile


def _extract_f0(snd, p: VoiceProfile):
    """Praat autocorrelation via visualization.extract_f0_praat (two-pass, anti-halving)."""
    from src.analysis.visualization import extract_f0_praat
    f0, _times = extract_f0_praat(p.filepath)
    voiced = f0[np.isfinite(f0)]
    if len(voiced):
        p.f0_mean_hz = round(float(np.mean(voiced)), 1)
        p.f0_min_hz  = round(float(np.percentile(voiced, 5)), 1)
        p.f0_max_hz  = round(float(np.percentile(voiced, 95)), 1)


def _extract_formants(snd, p: VoiceProfile):
    formants = snd.to_formant_burg(
        time_step=0.01, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025)
    times = np.linspace(snd.start_time, snd.end_time, 100)
    f1, f2, f3 = [], [], []
    for t in times:
        for bucket, fn_num in [(f1, 1), (f2, 2), (f3, 3)]:
            v = formants.get_value_at_time(fn_num, t)
            if v and not np.isnan(v):
                bucket.append(v)
    if f1: p.f1_mean_hz = round(float(np.median(f1)), 0)
    if f2: p.f2_mean_hz = round(float(np.median(f2)), 0)
    if f3: p.f3_mean_hz = round(float(np.median(f3)), 0)


def _extract_voice_quality(snd, p: VoiceProfile):
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60, 400)
    try:
        p.jitter_pct = round(
            parselmouth.praat.call(
                pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100, 2)
        p.shimmer_pct = round(
            parselmouth.praat.call(
                [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100, 2)
    except Exception:
        pass
    harmonicity = snd.to_harmonicity_cc(
        time_step=0.01, minimum_pitch=60, silence_threshold=0.1)
    hnr_vals = harmonicity.values[harmonicity.values != -200]
    if len(hnr_vals):
        p.hnr_db = round(float(np.mean(hnr_vals)), 1)


def _suggest_perceptual_dims(snd, p: VoiceProfile):
    """Map acoustic measurements to 0-100 perceptual dimension suggestions."""
    spectrum = snd.to_spectrum()
    centroid = parselmouth.praat.call(spectrum, "Get centre of gravity", 2)
    # Brightness: centroid 500-4000 Hz → 0-100
    p.brightness = round(float(np.clip((centroid - 500) / 35, 0, 100)), 0)
    # Breathiness: HNR inverse (lower HNR = breathier)
    if p.hnr_db is not None:
        p.breathiness = round(float(np.clip(100 - (p.hnr_db * 4), 0, 100)), 0)
    # Roughness: jitter + shimmer composite
    if p.jitter_pct is not None and p.shimmer_pct is not None:
        p.roughness = round(float(np.clip(p.jitter_pct * 20 + p.shimmer_pct * 5, 0, 100)), 0)
    # Tension: F0 range proxy
    if p.f0_min_hz and p.f0_max_hz:
        f0_range = p.f0_max_hz - p.f0_min_hz
        p.tension = round(float(np.clip(f0_range / 3, 0, 100)), 0)
    # Nasality: no reliable acoustic proxy — left as None (manual input)
