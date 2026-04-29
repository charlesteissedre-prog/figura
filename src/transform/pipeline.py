"""
Voice transform pipeline.

Two backends:
- "world"      — pyworld spectral-envelope pipeline, pure CPU, fast.
- "vevo"       — Amphion Vevo neural VC, timbre-only path (inference_fm).
                 Needs the checkpoints that HuggingFace downloads on first use
                 (tokenizer/vq8192, acoustic_modeling/Vq8192ToMels, Vocoder).
                 The heavier Vq32 / AR components are NOT loaded — that was the
                 Vevo "style"/"full" path, which we don't support here.
"""
from dataclasses import dataclass, field
import contextlib
import os
import numpy as np


@dataclass
class ParamConfig:
    """Per-parameter transform settings."""
    enabled: bool = True
    strength: float = 1.0


@dataclass
class TransformConfig:
    """Full configuration for a single transform run.
    Maps directly to the parameter panel in the UI."""
    pitch:       ParamConfig = field(default_factory=ParamConfig)
    timbre:      ParamConfig = field(default_factory=ParamConfig)
    energy:      ParamConfig = field(default_factory=ParamConfig)
    rhythm:      ParamConfig = field(default_factory=lambda: ParamConfig(enabled=False))
    breathiness: ParamConfig = field(default_factory=ParamConfig)
    formants:    ParamConfig = field(default_factory=ParamConfig)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformConfig":
        return cls(**{k: ParamConfig(**v) for k, v in d.items()})


def run(
    source_path: str,
    target_path: str,
    config: TransformConfig,
    output_path: str,
    backend: str = "auto",
) -> str:
    """Run the voice transform. backend in {"auto", "vevo", "world"}.

    "auto" tries Vevo timbre first, falls back to WORLD if Vevo isn't available
    or errors out (e.g. OOM on constrained machines).
    """
    if backend == "auto":
        if _vevo_available():
            try:
                return _run_vevo(source_path, target_path, config, output_path)
            except Exception as e:
                print(f"[vevo] failed, falling back to WORLD: {e}")
        return _run_world(source_path, target_path, config, output_path)

    if backend == "vevo":
        return _run_vevo(source_path, target_path, config, output_path)
    return _run_world(source_path, target_path, config, output_path)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(path: str, sr: int = None) -> tuple:
    """Load audio from any format. Returns (audio_float64, sample_rate)."""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(path)
    except Exception:
        import librosa
        audio, file_sr = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, file_sr


def _resample_1d(audio, target_len):
    from scipy.interpolate import interp1d
    if len(audio) == target_len:
        return audio
    x_src = np.linspace(0, 1, len(audio))
    x_tgt = np.linspace(0, 1, target_len)
    return interp1d(x_src, audio, kind='linear', fill_value='extrapolate')(x_tgt)


# ---------------------------------------------------------------------------
# Vevo backend (Amphion — timbre-only, inference_fm path)
# ---------------------------------------------------------------------------

_vevo_pipeline = None


def _vevo_available() -> bool:
    """True if the Amphion tree and VevoInferencePipeline are importable."""
    try:
        import torch  # noqa
        import torchaudio  # noqa
        import sys
        from pathlib import Path
        amphion_root = Path(__file__).resolve().parent.parent.parent / "Amphion"
        if not amphion_root.exists():
            return False
        if str(amphion_root) not in sys.path:
            sys.path.insert(0, str(amphion_root))
        from models.vc.vevo.vevo_utils import VevoInferencePipeline  # noqa
        return True
    except Exception:
        return False


def _amphion_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "Amphion"


@contextlib.contextmanager
def _in_amphion_cwd():
    """Amphion's config JSONs reference stats files via paths like
    `./models/vc/vevo/config/hubert_large_l18_mean_std.npz` — resolved relative
    to the process CWD. This context manager pins CWD to Amphion/ while Vevo
    code runs, then restores it."""
    prev = os.getcwd()
    try:
        os.chdir(_amphion_root())
        yield
    finally:
        os.chdir(prev)


def _get_vevo_pipeline():
    """Lazy-load the timbre-only Vevo pipeline. No AR / Vq32 tokenizer —
    that's the style/full path we intentionally don't support."""
    global _vevo_pipeline
    if _vevo_pipeline is not None:
        return _vevo_pipeline

    import sys
    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download

    amphion_root = _amphion_root()
    if str(amphion_root) not in sys.path:
        sys.path.insert(0, str(amphion_root))
    from models.vc.vevo.vevo_utils import VevoInferencePipeline

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cache_dir = str(Path(__file__).resolve().parent.parent.parent / "ckpts" / "Vevo")

    # Content-style tokenizer (vq8192)
    local_dir = snapshot_download(
        repo_id="amphion/Vevo", repo_type="model",
        cache_dir=cache_dir, allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # Flow Matching Transformer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo", repo_type="model",
        cache_dir=cache_dir, allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = str(amphion_root / "models" / "vc" / "vevo" / "config" / "Vq8192ToMels.json")
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # Vocoder
    local_dir = snapshot_download(
        repo_id="amphion/Vevo", repo_type="model",
        cache_dir=cache_dir, allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = str(amphion_root / "models" / "vc" / "vevo" / "config" / "Vocoder.json")
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    def _vram(tag):
        if device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[vevo] VRAM after {tag}: {used:.2f} GB used / {reserved:.2f} GB reserved")

    _vram("pre-init")
    with _in_amphion_cwd():
        _vevo_pipeline = VevoInferencePipeline(
            content_tokenizer_ckpt_path=None,
            content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            ar_cfg_path=None,
            ar_ckpt_path=None,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=device,
        )
    _vram("pipeline-built")

    # VEVO_FP16=1 halves the tokenizer weights (safe, saves ~1 GB VRAM).
    fp16 = os.environ.get("VEVO_FP16", "").lower()
    if fp16 and device.type == "cuda":
        for name in ("content_style_tokenizer",):
            m = getattr(_vevo_pipeline, name, None)
            if m is not None:
                m.half()
        _vram(f"fp16={fp16}")

    return _vevo_pipeline


def _run_vevo(source_path, target_path, config: TransformConfig, output_path):
    """Vevo timbre transfer via inference_fm. Timbre reference = target; the
    source's prosody/content is preserved."""
    import soundfile as sf
    import torch

    pipe = _get_vevo_pipeline()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"[vevo] VRAM pre-inference: {used:.2f} GB used")

    use_autocast = bool(os.environ.get("VEVO_FP16")) and torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16) if use_autocast
        else contextlib.nullcontext()
    )

    with autocast_ctx, _in_amphion_cwd():
        gen_audio = pipe.inference_fm(
            src_wav_path=source_path,
            timbre_ref_wav_path=target_path,
            flow_matching_steps=32,
        )

    audio_out = gen_audio.cpu().numpy().squeeze().astype(np.float32)
    vevo_sr = 24000

    # --- Pitch correction via Praat PSOLA ---
    # inference_fm transfers both timbre AND pitch. PSOLA shifts pitch
    # pulses in the time domain — formants stay exactly where Vevo put
    # them, no spectral re-synthesis, no double-vocoding.
    pitch_blend = config.pitch.strength if config.pitch.enabled else 0.0
    if pitch_blend < 1.0:
        audio_out = _psola_pitch_transplant(
            audio_out, source_path, vevo_sr, pitch_blend=pitch_blend,
        )

    # Strength blend: mix output toward source by (1 - strength) of whichever
    # timbre-like param is driving the transfer.
    blend = max(
        config.timbre.strength if config.timbre.enabled else 0,
        config.formants.strength if config.formants.enabled else 0,
    )
    if blend < 1.0:
        src_audio, _ = _load_audio(source_path)
        src_resampled = _resample_1d(src_audio, len(audio_out))
        audio_out = blend * audio_out + (1 - blend) * src_resampled.astype(np.float32)

    if config.energy.enabled:
        audio_out = _transfer_energy(audio_out, source_path, target_path,
                                     config.energy.strength, vevo_sr)
    if config.breathiness.enabled:
        audio_out = _transfer_breathiness(audio_out, source_path, target_path,
                                          config.breathiness.strength, vevo_sr)

    sf.write(output_path, audio_out, vevo_sr)
    return output_path


def _psola_pitch_transplant(
    vevo_audio: np.ndarray,
    source_path: str,
    sr: int,
    pitch_blend: float = 0.0,
) -> np.ndarray:
    """Replace the Vevo output's pitch contour with the source's via Praat
    PSOLA (overlap-add resynthesis in the time domain).

    Unlike spectral methods (WORLD, mel-domain shift), PSOLA moves pitch
    pulses without touching the spectral envelope at all — formants stay
    exactly where Vevo placed them. No double-vocoding, no re-synthesis
    artefacts.

    pitch_blend=0 → full source pitch (default, pure timbre transfer).
    pitch_blend=1 → keep Vevo pitch (caller should skip this function).
    """
    import parselmouth
    from src.analysis.visualization import extract_f0_praat

    # Build Praat Sound from the Vevo numpy output
    vevo_snd = parselmouth.Sound(vevo_audio.astype(np.float64), sampling_frequency=sr)

    # Extract source F0 contour via the shared two-pass Hirst method
    src_f0, src_times = extract_f0_praat(source_path)

    # Create a Manipulation object from the Vevo audio
    manipulation = parselmouth.praat.call(
        vevo_snd, "To Manipulation", 0.01, 60, 600,
    )

    # Build a new PitchTier from the source F0 contour
    pitch_tier = parselmouth.praat.call(
        manipulation, "Extract pitch tier",
    )
    # Clear the existing (Vevo/target) pitch points
    parselmouth.praat.call(pitch_tier, "Remove points between", 0.0, vevo_snd.duration)

    # Add source pitch points (skip unvoiced frames)
    for i, t in enumerate(src_times):
        f0 = src_f0[i]
        if not np.isfinite(f0) or f0 <= 0:
            continue
        if t > vevo_snd.duration:
            break
        if pitch_blend > 0:
            # Retrieve the Vevo pitch at this time to blend
            vevo_f0 = parselmouth.praat.call(
                vevo_snd.to_pitch_cc(time_step=0.01, pitch_floor=60, pitch_ceiling=600),
                "Get value at time", t, "Hertz", "Linear",
            )
            if vevo_f0 and vevo_f0 > 0:
                f0 = np.exp(
                    (1 - pitch_blend) * np.log(f0) + pitch_blend * np.log(vevo_f0)
                )
        parselmouth.praat.call(pitch_tier, "Add point", float(t), float(f0))

    # Replace and resynthesize
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
    result_snd = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    return np.array(result_snd.values[0]).astype(np.float32)


def _transfer_energy(audio_out, source_path, target_path, strength, sr):
    """Transfer RMS energy envelope from target, blended by strength."""
    hop = int(0.010 * sr)
    frame = int(0.025 * sr)

    def rms_env(x):
        return np.array([
            np.sqrt(np.mean(x[i:i+frame]**2) + 1e-10)
            for i in range(0, max(1, len(x) - frame), hop)
        ])

    src_audio, _ = _load_audio(source_path)
    tgt_audio, _ = _load_audio(target_path)

    src_rms = rms_env(src_audio)
    tgt_rms = rms_env(tgt_audio)
    out_rms = rms_env(audio_out)

    n = len(out_rms)
    src_rms = _resample_1d(src_rms, n)
    tgt_rms = _resample_1d(tgt_rms, n)

    desired = src_rms + strength * (tgt_rms - src_rms)
    gain = desired / (out_rms + 1e-10)
    gain_samples = np.repeat(gain, hop)[:len(audio_out)]
    if len(gain_samples) < len(audio_out):
        gain_samples = np.pad(gain_samples, (0, len(audio_out) - len(gain_samples)),
                              constant_values=gain_samples[-1])
    return (audio_out * gain_samples).astype(np.float32)


def _transfer_breathiness(audio_out, source_path, target_path, strength, sr):
    """Blend target AP onto the Vevo output via a WORLD re-analysis."""
    import pyworld as pw

    out_f64 = audio_out.astype(np.float64)
    out_f0, out_sp, out_ap = pw.wav2world(out_f64, sr)

    tgt_audio, _ = _load_audio(target_path)
    tgt_resampled = _resample_1d(tgt_audio, int(len(tgt_audio) * sr / 44100)) if sr != 44100 else tgt_audio
    _, _, tgt_ap = pw.wav2world(tgt_resampled.astype(np.float64), sr)

    src_ap_mean = np.mean(out_ap, axis=0, keepdims=True)
    tgt_ap_mean = np.mean(tgt_ap, axis=0, keepdims=True)
    ap_shift = strength * (tgt_ap_mean - src_ap_mean)
    blended_ap = np.clip(out_ap + ap_shift, 0.0, 1.0)

    result = pw.synthesize(out_f0, out_sp, blended_ap, sr)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# WORLD backend
# ---------------------------------------------------------------------------

def _run_world(source_path, target_path, config: TransformConfig, output_path):
    """WORLD vocoder pipeline (pyworld). Spectral-envelope swap for
    formant/timbre + f0 contour shift for pitch + AP blend for breathiness."""
    import pyworld as pw
    import soundfile as sf

    src_audio, src_sr = _load_audio(source_path)
    tgt_audio, tgt_sr = _load_audio(target_path)

    src_f0, src_sp, src_ap = pw.wav2world(src_audio, src_sr)
    tgt_f0, tgt_sp, tgt_ap = pw.wav2world(tgt_audio, tgt_sr)

    out_f0 = src_f0.copy()
    if config.pitch.enabled:
        src_voiced = src_f0[src_f0 > 0]
        tgt_voiced = tgt_f0[tgt_f0 > 0]
        if len(src_voiced) and len(tgt_voiced):
            ratio = np.exp(
                config.pitch.strength *
                (np.log(np.mean(tgt_voiced)) - np.log(np.mean(src_voiced)))
            )
            out_f0 = np.where(src_f0 > 0, src_f0 * ratio, src_f0)

    out_sp = src_sp.copy()
    if config.timbre.enabled or config.formants.enabled:
        strength = max(
            config.timbre.strength if config.timbre.enabled else 0,
            config.formants.strength if config.formants.enabled else 0,
        )
        out_sp = _warp_sp_formants(src_sp, tgt_sp, src_sr, tgt_sr, strength)

    out_ap = src_ap.copy()
    if config.breathiness.enabled:
        src_ap_mean = np.mean(src_ap, axis=0, keepdims=True)
        tgt_ap_mean = np.mean(tgt_ap, axis=0, keepdims=True)
        ap_shift = config.breathiness.strength * (tgt_ap_mean - src_ap_mean)
        out_ap = np.clip(src_ap + ap_shift, 0.0, 1.0)

    out_audio = pw.synthesize(out_f0, out_sp, out_ap, src_sr)
    sf.write(output_path, out_audio, src_sr)
    return output_path


def _warp_sp_formants(src_sp, tgt_sp, src_sr, tgt_sr, strength):
    """Warp source spectral envelope along the frequency axis to match
    target speaker's formant structure via a global frequency warp ratio."""
    from scipy.interpolate import interp1d

    n_freq = src_sp.shape[1]

    src_mean = np.mean(np.log(src_sp + 1e-16), axis=0)
    tgt_mean = np.mean(np.log(tgt_sp + 1e-16), axis=0)

    src_peaks = _find_formant_peaks(src_mean, src_sr, n_freq)
    tgt_peaks = _find_formant_peaks(tgt_mean, tgt_sr, n_freq)

    n_pairs = min(len(src_peaks), len(tgt_peaks))
    if n_pairs == 0:
        return src_sp

    src_anchors = [0] + [src_peaks[i] for i in range(n_pairs)] + [n_freq - 1]
    tgt_anchors = [0] + [tgt_peaks[i] for i in range(n_pairs)] + [n_freq - 1]

    identity = np.arange(n_freq, dtype=np.float64)
    full_warp = np.interp(identity, src_anchors, tgt_anchors)
    warped_bins = identity + strength * (full_warp - identity)
    warped_bins = np.clip(warped_bins, 0, n_freq - 1)

    out_sp = np.zeros_like(src_sp)
    for i in range(src_sp.shape[0]):
        f = interp1d(np.arange(n_freq), src_sp[i], kind='linear',
                     bounds_error=False, fill_value=(src_sp[i, 0], src_sp[i, -1]))
        out_sp[i] = f(warped_bins)

    return out_sp


def _find_formant_peaks(log_envelope, sr, n_freq, n_formants=3):
    """Find formant peak bin indices in a log spectral envelope."""
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    smoothed = uniform_filter1d(log_envelope, size=max(n_freq // 50, 5))

    max_bin = min(n_freq, int(5000 / (sr / 2) * n_freq))
    peaks, properties = find_peaks(
        smoothed[:max_bin],
        distance=max(n_freq // 30, 3),
        prominence=0.3,
    )

    if len(peaks) == 0:
        return []

    prom = properties['prominences']
    top_idx = np.argsort(prom)[::-1][:n_formants]
    return sorted(peaks[top_idx].tolist())
