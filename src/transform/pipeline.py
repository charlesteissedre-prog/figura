"""
Voice transform pipeline.

Three backends:
- "world"      — pyworld spectral-envelope pipeline, pure CPU, fast.
- "vevo"       — Amphion Vevo (v1) neural VC, timbre-only path (inference_fm).
                 Needs the checkpoints that HuggingFace downloads on first use
                 (tokenizer/vq8192, acoustic_modeling/Vq8192ToMels, Vocoder).
                 The heavier Vq32 / AR components are NOT loaded — that was the
                 Vevo "style"/"full" path, which we don't support here.
- "vevo2"      — Amphion Vevo2 neural VC, FM-only path (Vevo2InferencePipeline).
                 Larger checkpoints (~11 GB from RMSnow/Vevo2) but the FM model
                 alone runs in ~7 GB RAM on a 16 GB rig after the pagefile fix.
                 AR/prosody components are not downloaded.
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
    """Run the voice transform. backend in {"auto", "vevo", "vevo2", "world"}.

    "auto" tries Vevo (v1) timbre first, falls back to WORLD if Vevo isn't
    available or errors out (e.g. OOM on constrained machines). Vevo2 is
    opt-in only — it's heavier and not yet validated as strictly better than v1.
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
    if backend == "vevo2":
        return _run_vevo2(source_path, target_path, config, output_path)
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

    correction_strength = (1.0 - config.pitch.strength) if config.pitch.enabled else 1.0
    if correction_strength > 0:
        audio_out = _uniform_pitch_correction(
            audio_out, source_path, vevo_sr, strength=correction_strength,
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


# ---------------------------------------------------------------------------
# Vevo2 backend (Amphion — FM-only path, Vevo2InferencePipeline)
# ---------------------------------------------------------------------------

_vevo2_pipeline = None


def _vevo2_available() -> bool:
    """True if the Amphion tree and Vevo2InferencePipeline are importable."""
    try:
        import torch  # noqa
        import torchaudio  # noqa
        import sys
        amphion_root = _amphion_root()
        if not amphion_root.exists():
            return False
        if not (amphion_root / "models" / "svc" / "vevo2").exists():
            return False
        if str(amphion_root) not in sys.path:
            sys.path.insert(0, str(amphion_root))
        from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline  # noqa
        return True
    except Exception:
        return False


def _get_vevo2_pipeline():
    """Lazy-load the FM-only Vevo2 pipeline. No AR / prosody tokenizer
    — those are needed for TTS / style-conversion modes which we don't expose."""
    global _vevo2_pipeline
    if _vevo2_pipeline is not None:
        return _vevo2_pipeline

    import sys
    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download

    amphion_root = _amphion_root()
    if str(amphion_root) not in sys.path:
        sys.path.insert(0, str(amphion_root))
    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_dir = str(Path(__file__).resolve().parent.parent.parent / "ckpts" / "Vevo2")

    # FM-only needs three pieces from RMSnow/Vevo2; the AR model and prosody
    # tokenizer (~5 GB) are intentionally skipped via allow_patterns.
    local_dir = snapshot_download(
        repo_id="RMSnow/Vevo2", repo_type="model",
        local_dir=ckpt_dir,
        allow_patterns=[
            "tokenizer/contentstyle_fvq16384_12.5hz/*",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa/*",
            "vocoder/*",
        ],
        resume_download=True,
    )

    content_style_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )
    fmt_cfg_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json"
    )
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa")
    vocoder_cfg_path = os.path.join(local_dir, "vocoder/config.json")
    vocoder_ckpt_path = os.path.join(local_dir, "vocoder")

    def _vram(tag):
        if device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[vevo2] VRAM after {tag}: {used:.2f} GB used / {reserved:.2f} GB reserved")

    _vram("pre-init")
    with _in_amphion_cwd():
        _vevo2_pipeline = Vevo2InferencePipeline(
            content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=device,
        )
    _vram("pipeline-built")
    return _vevo2_pipeline


def _run_vevo2(source_path, target_path, config: TransformConfig, output_path):
    """Vevo2 timbre transfer via Vevo2InferencePipeline.inference_fm.

    Minimal post-processing: only linear operations (source blend, energy gain)
    touch the audio. PSOLA and the WORLD breathiness round-trip are skipped
    because Vevo2's output quality is high enough that any vocoder re-synthesis
    on top is audible as artefacts.

    Slider mapping: `pitch.enabled` toggles Vevo2's native `use_pitch_shift`
    (on = shift source into target's pitch range, off = keep source's range).
    `pitch.strength` controls a uniform PSOLA correction toward source's mean
    F0 — same helper as Vevo 1, since Vevo 2's residual drift is also register-
    only when `use_pitch_shift=False`. `breathiness` slider is a no-op (would
    require re-introducing the WORLD round-trip, which is what made the
    output artefact-y in the first place)."""
    import soundfile as sf
    import torch

    pipe = _get_vevo2_pipeline()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"[vevo2] VRAM pre-inference: {used:.2f} GB used")

    with _in_amphion_cwd():
        gen_audio = pipe.inference_fm(
            src_wav_path=source_path,
            timbre_ref_wav_path=target_path,
            use_pitch_shift=config.pitch.enabled,
            flow_matching_steps=32,
        )

    audio_out = gen_audio.cpu().numpy().squeeze().astype(np.float32)
    vevo_sr = 24000

    correction_strength = (1.0 - config.pitch.strength) if config.pitch.enabled else 1.0
    if correction_strength > 0:
        audio_out = _uniform_pitch_correction(
            audio_out, source_path, vevo_sr, strength=correction_strength,
        )

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

    sf.write(output_path, audio_out, vevo_sr)
    return output_path


def _uniform_pitch_correction(
    audio_out: np.ndarray,
    source_path: str,
    sr: int,
    *,
    strength: float = 1.0,
    gate_st: float = 0.3,
) -> np.ndarray:
    """Uniform Praat PSOLA shift toward source's mean F0.

    Single multiplicative ratio applied to every pitch pulse — the regime
    PSOLA is cleanest in. Auto-skips when measured drift is below `gate_st`
    semitones (perceptually invisible; resynthesis would cost LTAS without
    audible gain). `strength` interpolates the correction in log2 space:
    1 = full shift to source mean, 0 = no shift.
    """
    if strength <= 0.0:
        return audio_out

    import parselmouth
    import pyworld as pw

    def mean_voiced_f0(a, fs):
        _f0, t = pw.harvest(a.astype(np.float64), fs=fs, frame_period=32 / 3)
        f0 = pw.stonemask(a.astype(np.float64), _f0, t, fs=fs)
        voiced = f0[f0 > 0]
        return float(voiced.mean()) if len(voiced) else 0.0

    src_audio, src_sr = _load_audio(source_path)
    src_mean = mean_voiced_f0(src_audio, src_sr)
    out_mean = mean_voiced_f0(audio_out, sr)
    if src_mean <= 0 or out_mean <= 0:
        return audio_out

    full_ratio = src_mean / out_mean
    if abs(np.log2(full_ratio) * 12) < gate_st:
        return audio_out

    ratio = float(full_ratio ** strength)
    snd = parselmouth.Sound(audio_out.astype(np.float64), sampling_frequency=sr)
    manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.01, 60, 600)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Multiply frequencies",
                           0.0, snd.duration, ratio)
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
    resynth = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    out = np.array(resynth.values[0]).astype(np.float32)
    if len(out) > len(audio_out):
        out = out[: len(audio_out)]
    elif len(out) < len(audio_out):
        out = np.pad(out, (0, len(audio_out) - len(out)))
    return out


def _transfer_energy(audio_out, source_path, target_path, strength, sr):
    """Transfer RMS energy envelope from target, blended by strength.

    Window/hop are 25 ms / 10 ms in REAL time, so each audio's RMS envelope
    is computed at its native sample rate. The three envelopes are then
    resampled to the output frame count for the gain blend — comparing RMS
    values measured over the same physical duration regardless of source
    audio's sample rate."""
    def rms_env(x, x_sr):
        f = int(0.025 * x_sr)
        h = int(0.010 * x_sr)
        return np.array([
            np.sqrt(np.mean(x[i:i+f]**2) + 1e-10)
            for i in range(0, max(1, len(x) - f), h)
        ])

    src_audio, src_sr = _load_audio(source_path)
    tgt_audio, tgt_sr = _load_audio(target_path)

    src_rms = rms_env(src_audio, src_sr)
    tgt_rms = rms_env(tgt_audio, tgt_sr)
    out_rms = rms_env(audio_out, sr)

    n = len(out_rms)
    src_rms = _resample_1d(src_rms, n)
    tgt_rms = _resample_1d(tgt_rms, n)

    desired = src_rms + strength * (tgt_rms - src_rms)
    gain = desired / (out_rms + 1e-10)
    out_hop = int(0.010 * sr)
    gain_samples = np.repeat(gain, out_hop)[:len(audio_out)]
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
