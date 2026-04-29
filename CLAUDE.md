# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice transformation system that analyzes speaker voice characteristics and selectively transforms one voice to match another's acoustic profile. Written in Python, using Click for CLI and Praat for audio analysis, React + Vite for the frontend, FastAPI for the backend.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze a voice
python cli.py analyze path/to/voice.wav --name "Speaker A"

# Transform with selective parameter control
python cli.py transform --source source.wav --target target.wav --output output.wav \
    --disable rhythm --strength timbre=0.8 pitch=1.0

# Compare source, output, and target
python cli.py compare --source source.wav --output output.wav --target target.wav

# Save/load configs and profiles as JSON
python cli.py analyze voice.wav --save profiles/speaker.json
python cli.py transform --config configs/run_01.json
```

No test suite exists yet.

## Architecture

The system is a three-stage pipeline: **extract → transform → compare**.

### Modules

- **`src/analysis/extractor.py`** — Extracts a `VoiceProfile` dataclass from audio (F0, formants, HNR, jitter, shimmer, CPP, plus perceptual dimensions like brightness/breathiness/nasality). Delegates F0 to `visualization.extract_f0_praat`.

- **`src/analysis/visualization.py`** — `extract_f0_praat` (Hirst two-pass autocorrelation with `octave_cost=0.055` to suppress halving), `compute_f0_contour` (downsampled contour for the UI), `render_mel_png` (axis-less mel PNG for click-to-seek), `audio_duration_s`.

- **`src/transform/pipeline.py`** — two transform backends controlled by `TransformConfig`/`ParamConfig` dataclasses. Each of the 6 parameters (pitch, timbre, energy, rhythm, breathiness, formants) has independent enable/disable and strength (0.0–1.0) interpolation. Backends:
  - **WORLD** (`_run_world`) — pyworld spectral-envelope swap, pure CPU, fast. Pitch shift on F0, formant transfer via spectral-envelope warping, breathiness via AP blend.
  - **Vevo timbre** (`_run_vevo`) — Amphion neural VC via `inference_fm`, timbre-only. The style/full Vevo modes that need the Vq32 + AR model are deliberately NOT supported — that path OOMed on 16 GB / 8 GB rigs. Post-processes with WORLD-based energy + breathiness blends to honour those param sliders.

- **`src/comparison/engine.py`** — Three-way comparison (source→output→target) producing per-metric deltas, similarity scores (0–100%), and leakage warnings when disabled parameters change >10%.

- **`src/ui/api.py`** — FastAPI backend. Endpoints: `/api/upload`, `/api/extract`, `/api/transform`, `/api/compare`, `/api/library`, `/api/f0-contour/{id}`, `/api/mel-spectrogram/{id}`, `/api/audio-info/{id}`.

- **`cli.py`** — Click-based CLI exposing `analyze`, `transform`, and `compare` commands with config file save/load.

- **`web/`** — React + Vite frontend. Key components: `TransformView` (source/target selection, param panel, run), `LibraryView` (voice library with detail tabs), `ComparisonView` (three-way result view). Tabbed detail panes (`DetailTabs`) show profile card + F0 contour (click-to-seek SVG) + mel spectrogram (click-to-seek PNG) per voice.

### Key Design Patterns

- **Config-driven**: `TransformConfig` maps parameter names to `ParamConfig(enabled, strength)`. Configs are JSON-serializable for reproducibility.
- **Leakage detection**: The comparison engine flags unintended changes on parameters that were disabled during transform.
- **Shared F0 core**: profile stats and the contour plot both come from `extract_f0_praat`, so the numbers in the profile card and the plotted contour are always consistent.
