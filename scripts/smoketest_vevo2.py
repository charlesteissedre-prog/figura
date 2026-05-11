"""
Vevo2 FM-only smoke test.

Standalone: does NOT wire into src/transform/pipeline.py. Confirms that on
this rig, after the pagefile fix, we can:
  1. Download Vevo2 checkpoints from HuggingFace
  2. Construct Vevo2InferencePipeline (no AR model)
  3. Run inference_fm on a source+target pair
  4. Write audio to disk

Prints RAM + VRAM at every stage so we can see where any OOM happens.

Usage:
  python scripts/smoketest_vevo2.py
  python scripts/smoketest_vevo2.py --source path/a.wav --target path/b.wav
  python scripts/smoketest_vevo2.py --no-pitch-shift   # preserve source pitch
"""
import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AMPHION_ROOT = REPO_ROOT / "Amphion"
CKPT_DIR = REPO_ROOT / "ckpts" / "Vevo2"
DEFAULT_SOURCE = REPO_ROOT / "examples" / "female" / "source.wav"
DEFAULT_TARGET = REPO_ROOT / "examples" / "female" / "target.wav"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "_smoketest_vevo2.wav"


def stage(label: str):
    print(f"\n=== [{label}] ===", flush=True)


def report_memory(tag: str):
    parts = [f"[mem:{tag}]"]
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        parts.append(f"RAM rss={rss:.2f} GB  avail={avail:.2f} GB")
    except ImportError:
        parts.append("RAM=(psutil not installed)")

    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            parts.append(f"VRAM used={used:.2f} GB reserved={reserved:.2f} GB")
    except Exception:
        pass

    print("  " + " | ".join(parts), flush=True)


@contextlib.contextmanager
def in_amphion_cwd():
    """Vevo2 config JSONs reference auxiliary files via CWD-relative paths,
    same pattern as Vevo v1. Pin CWD to Amphion/ while the pipeline runs."""
    prev = os.getcwd()
    try:
        os.chdir(AMPHION_ROOT)
        yield
    finally:
        os.chdir(prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--no-pitch-shift", action="store_true",
        help="Pass use_pitch_shift=False (preserve source pitch). "
             "Default matches Amphion's reference script (True).",
    )
    args = ap.parse_args()

    for p in (args.source, args.target):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    if not (AMPHION_ROOT / "models" / "svc" / "vevo2").is_dir():
        sys.exit(f"Amphion clone missing models/svc/vevo2 at {AMPHION_ROOT}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"source: {args.source}")
    print(f"target: {args.target}")
    print(f"output: {args.output}")
    print(f"use_pitch_shift: {not args.no_pitch_shift}")
    report_memory("startup")

    stage("import torch + Amphion")
    if str(AMPHION_ROOT) not in sys.path:
        sys.path.insert(0, str(AMPHION_ROOT))
    import torch
    from huggingface_hub import snapshot_download
    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  device: {device}")
    report_memory("imports done")

    stage("download checkpoints from RMSnow/Vevo2")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id="RMSnow/Vevo2",
        repo_type="model",
        local_dir=str(CKPT_DIR),
        resume_download=True,
    )
    print(f"  checkpoint root: {local_dir}")
    print(f"  download/verify took {time.time() - t0:.1f}s")
    report_memory("download done")

    content_style_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz",
    )
    fmt_cfg_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json",
    )
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa")
    vocoder_cfg_path = os.path.join(local_dir, "vocoder/config.json")
    vocoder_ckpt_path = os.path.join(local_dir, "vocoder")
    for p in (content_style_tokenizer_ckpt_path, fmt_cfg_path, fmt_ckpt_path,
              vocoder_cfg_path, vocoder_ckpt_path):
        if not os.path.exists(p):
            sys.exit(f"expected checkpoint path missing: {p}")

    stage("build Vevo2InferencePipeline (FM-only)")
    t0 = time.time()
    with in_amphion_cwd():
        pipeline = Vevo2InferencePipeline(
            content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=device,
        )
    print(f"  pipeline built in {time.time() - t0:.1f}s")
    report_memory("pipeline built")

    stage("run inference_fm")
    t0 = time.time()
    with in_amphion_cwd():
        gen_audio = pipeline.inference_fm(
            src_wav_path=str(args.source),
            timbre_ref_wav_path=str(args.target),
            use_pitch_shift=(not args.no_pitch_shift),
            flow_matching_steps=32,
        )
    print(f"  inference took {time.time() - t0:.1f}s")
    report_memory("inference done")

    stage("save audio")
    # Replicate Amphion's save_audio (vevo2_utils.py:161) but via soundfile
    # to avoid the torchcodec dep that newer torchaudio.save() pulls in.
    # gen_audio is [1, T] @ 24 kHz; normalize to -25 dB then write.
    import soundfile as sf
    sr = 24000
    rms = torch.sqrt(torch.mean(gen_audio ** 2))
    gain_db = -25.0 - 20 * torch.log10(rms + 1e-9)
    waveform = (gen_audio * (10 ** (gain_db / 20))).cpu().numpy().squeeze()
    sf.write(str(args.output), waveform, sr)
    size_mb = args.output.stat().st_size / 1e6
    print(f"  wrote {args.output}  ({size_mb:.2f} MB)")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
