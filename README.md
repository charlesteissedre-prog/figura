# Figura — Voice Transformation System

Analyzes speaker voice characteristics and selectively transforms one voice to match another's acoustic profile. Each acoustic dimension (pitch, timbre, energy, rhythm, breathiness, formants) can be independently enabled, disabled, or blended by strength, so you can hear exactly which aspect of a target voice you're borrowing.

## Pipeline

**extract → transform → compare**

1. **Extract** a `VoiceProfile` (F0, formants, HNR, jitter, shimmer, CPP, brightness, breathiness, nasality) from audio using Praat's autocorrelation with a Hirst two-pass anti-halving step.
2. **Transform** the source toward the target. Three backends:
   - **WORLD** — pyworld spectral-envelope swap. Pure CPU, fast. Pitch shift on F0, formant/timbre transfer via spectral-envelope warping, breathiness via AP blend.
   - **Vevo (v1, timbre)** — Amphion neural VC via `inference_fm`. Timbre-only (the heavier style/full paths needing the Vq32 + AR model are not supported). Checkpoints auto-download via HuggingFace on first use.
   - **Vevo 2 (FM-only)** — Amphion's flow-matching VC. Cleaner timbre transfer than v1 but heavier (FM model ~363 M params + content-style tokenizer + vocoder ≈ 4 GB RAM peak). `pitch.enabled` toggles its native `use_pitch_shift` (target-pitch range vs source-pitch range).

   Both Vevo backends share a uniform Praat-PSOLA pitch correction applied as a thin post-step (`pitch.strength` interpolates how much of the model's pitch to keep, 1 = full model, 0 = land on source's mean). Auto-skips when measured drift is below ~0.3 semitones to avoid resynthesis cost.
3. **Compare** source, output, and target three-ways. Reports per-metric deltas, 0–100% similarity scores, and flags "leakage" when a disabled parameter still shifts by more than 10%.

## Install

```bash
# 1. Clone this repo
git clone https://github.com/<your-org>/figura.git
cd figura

# 2. Python deps
pip install -r requirements.txt

# 3. Vendored Amphion (required for the Vevo backend; not vendored in this repo)
git clone --depth 1 https://github.com/open-mmlab/Amphion.git

# 4. Optional: copy the env template if you want fp16 Vevo
cp .env.example .env

# 5. Web UI deps
cd web && npm install && cd ..
```

The first time a Vevo backend runs, checkpoints auto-download from HuggingFace: ~2 GB into `ckpts/Vevo/` for v1, additional weights into `ckpts/Vevo2/` for v2 (flow-matching model + content-style tokenizer + vocoder). The WORLD backend has no extra downloads.

## Run the app

Backend (FastAPI, from repo root):

```bash
uvicorn src.ui.api:app --reload --port 8000
```

Frontend (Vite dev server, in another terminal):

```bash
cd web
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173). The frontend talks to the backend on :8000.

## CLI

```bash
# Analyze a voice and save its profile
python cli.py analyze path/to/voice.wav --name "Speaker A" \
    --save profiles/speaker.json

# Transform, disabling rhythm and tuning per-parameter strength
python cli.py transform \
    --source source.wav --target target.wav --output output.wav \
    --disable rhythm --strength timbre=0.8 pitch=1.0

# Three-way comparison
python cli.py compare \
    --source source.wav --output output.wav --target target.wav

# Reproduce a run from a saved config
python cli.py transform \
    --source source.wav --target target.wav --output output.wav \
    --config examples/timbre_only.json
```

See [`examples/`](examples/) for ready-made configs.

## Screenshots

> Coming soon.

## Layout

```
cli.py                         # Click CLI: analyze / transform / compare
requirements.txt
LICENSE                        # MIT
src/
  analysis/extractor.py        # VoiceProfile extraction (Praat two-pass)
  analysis/visualization.py    # F0 contour + mel-spectrogram helpers
  transform/pipeline.py        # WORLD + Vevo (v1) + Vevo 2 backends, ParamConfig/TransformConfig
  comparison/engine.py         # Three-way deltas, similarity, leakage warnings
  ui/                          # FastAPI web backend
docs/                          # Pipeline doc + spec
examples/                      # Reproducible transform configs
samples/                       # Your input audio (gitignored — bring your own)
outputs/                       # Generated transform outputs (gitignored)
uploads/                       # Web UI uploads (gitignored)
web/                           # React + Vite frontend
ckpts/                         # Vevo checkpoints (gitignored, auto-downloaded)
Amphion/                       # Cloned manually — see Install (gitignored)
```

## Design notes

- **Config-driven.** `TransformConfig` maps each parameter to a `ParamConfig(enabled, strength)`. Configs serialize to JSON for reproducibility — see `examples/`.
- **Leakage detection.** If you disable `rhythm` but the output's rhythm still drifts >10% from the source, the comparison engine surfaces it as a warning.
- **Shared F0 core.** Profile stats and the contour plot both come from `extract_f0_praat`, so the numbers in the profile card and the plotted contour are always consistent.

## Status

No test suite. WORLD backend is functional. Vevo (v1) and Vevo 2 are functional on CPU and on CUDA where enough VRAM is available; `backend=auto` falls back to WORLD if Amphion or checkpoints are missing.

## License

MIT — see [LICENSE](LICENSE).

### Third-party licenses

This project depends on libraries with their own licenses. Notable ones:

| Dependency | License | Notes |
| --- | --- | --- |
| [Amphion](https://github.com/open-mmlab/Amphion) (Vevo backend) | Apache-2.0 | Cloned by the user; not vendored. |
| [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) | MIT | WORLD vocoder bindings. |
| [parselmouth / Praat](https://github.com/YannickJadoul/Parselmouth) | **GPL-3.0** | Used for F0 / formant analysis. Source distribution is fine; if you redistribute a binary that statically links parselmouth, GPL-3 propagates to the combined work. |

If you plan to redistribute a packaged binary, audit the parselmouth dependency before doing so.
