# Figura — Transform Pipeline

## Overview

Figura is a three-stage voice transformation system: **extract**, **transform**, **compare**. A source voice is selectively modified toward a target voice across six independently controllable dimensions (pitch, timbre, energy, rhythm, breathiness, formants), then the result is scored against the target to quantify how close the transfer landed and whether any disabled dimensions leaked.

```
                          ┌─────────────┐
          source.wav ───▶ │             │
                          │   EXTRACT   │──▶ VoiceProfile (F0, formants, HNR, …)
          target.wav ───▶ │             │
                          └─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────▼─────┐          ┌──────▼──────┐
              │   WORLD   │          │    VEVO     │
              │  vocoder  │          │   timbre    │
              └─────┬─────┘          └──────┬──────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                          ┌─────▼─────┐
                          │  COMPARE  │──▶ scores, deltas, leakage warnings
                          └───────────┘
```

---

## 1. Extract

**Module:** `src/analysis/extractor.py` + `src/analysis/visualization.py`

Every voice that enters the system (source, target, or output) goes through extraction, which populates a `VoiceProfile` dataclass with acoustic measurements and perceptual dimension estimates.

### F0 estimation (Hirst two-pass)

Pitch is the most error-prone measurement. A naive single-pass autocorrelation frequently "halves" the pitch — reporting 150 Hz instead of 300 Hz — because the first subharmonic has a strong autocorrelation peak too.

Figura uses a two-pass approach (after Hirst 2011):

1. **Pass 1** runs Praat's `to_pitch_cc` on wide bounds (60–600 Hz) with `octave_cost=0.055` (5.5× higher than Praat's default of 0.01). The elevated octave cost penalizes the subharmonic candidate at the per-frame level, which is the primary defence against halving.
2. From pass 1, compute the 25th and 75th percentiles of voiced frames.
3. **Pass 2** re-runs with tightened bounds: floor = 0.75 × q25, ceiling = 1.5 × q75 (with a safety guard that the ceiling never drops below 2.2 × median, so expressive voices aren't clipped).

Both the profile-card statistics (mean, 5th/95th percentiles) and the F0 contour plot in the UI are computed from the same `extract_f0_praat` function, so they always agree.

### Other measurements

- **Formants (F1–F3):** Praat Burg method, median over 100 evenly spaced time points.
- **Voice quality:** HNR (harmonics-to-noise), jitter, shimmer — all via Praat.
- **Perceptual dimensions (0–100):** brightness (spectral centroid proxy), breathiness (HNR inverse), roughness (jitter + shimmer composite), tension (F0 range proxy). Nasality has no reliable acoustic proxy and is left for manual input.

---

## 2. Transform

The user picks a source voice and a target voice, configures which dimensions to transfer and at what strength, and chooses a backend. Two backends are available.

### 2a. WORLD backend

**Module:** `src/transform/pipeline.py → _run_world`

The WORLD vocoder (Morise et al.) decomposes audio into three components — F0 contour, spectral envelope, and aperiodicity — then re-synthesizes after selective modification.

```
source ──▶ WORLD decompose ──▶ (src_f0, src_sp, src_ap)
target ──▶ WORLD decompose ──▶ (tgt_f0, tgt_sp, tgt_ap)

  pitch enabled?   ──▶  shift src_f0 toward tgt_f0 (log-ratio scaling)
  timbre enabled?  ──▶  warp src_sp toward tgt_sp  (formant-peak-matched freq warping)
  breathiness?     ──▶  blend src_ap toward tgt_ap  (mean AP per frequency bin)

out_audio ◀── WORLD synthesize(out_f0, out_sp, out_ap)
```

**Strengths:** pure CPU, fast (~2s for a 10s clip), no model download, all six parameters independently adjustable.

**Limitations:** spectral-envelope warping is a coarse approximation of timbre — it shifts formant frequencies but can't change the voice's "identity" the way a neural model can.

### 2b. Vevo timbre backend

**Module:** `src/transform/pipeline.py → _run_vevo`

Amphion's Vevo uses a Flow Matching Transformer (FMT) conditioned on content-style tokens (HuBERT vq8192) and a timbre reference to generate mel spectrograms, which a Vocos neural vocoder then converts to audio. This produces dramatically more natural timbre transfer than WORLD — but `inference_fm` replaces *both* timbre and pitch, which isn't what we want.

The full Vevo timbre pipeline has four steps:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: Vevo inference_fm                                            │
│                                                                      │
│   source ──▶ HuBERT tokenizer ──▶ content-style tokens               │
│   target ──▶ HuBERT tokenizer ──▶ timbre reference tokens            │
│                                                                      │
│   FMT(content tokens + timbre reference) ──▶ predicted mel           │
│   Vocos vocoder(predicted mel) ──────────────────────────────────────│──▶ raw Vevo audio
│                                                                      │    (target timbre
│                                                                      │     + target pitch)
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: Praat PSOLA pitch correction                                 │
│                                                                      │
│   source audio ──▶ Praat F0 (two-pass) ──▶ source pitch contour      │
│                                                                      │
│   Vevo audio ──▶ Praat Manipulation object                           │
│   Replace pitch tier with source F0 contour                          │
│   Overlap-add resynthesis ───────────────────────────────────────────│──▶ pitch-corrected
│                                                                      │    audio (Vevo
│                                                                      │    timbre intact,
│                                                                      │    formants intact)
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Steps 3–4: Post-processing                                           │
│                                                                      │
│   Timbre strength < 100%?  ──▶  blend output toward source audio     │
│   Energy enabled?          ──▶  RMS envelope transfer from target    │
│   Breathiness enabled?     ──▶  WORLD AP blend from target           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Step 1** produces the highest-quality timbre transfer available, but it replaces the source's pitch with the target's. This is an inherent property of the FMT — it generates mels conditioned on the full acoustic profile of the target.

**Step 2** corrects this using Praat's PSOLA (Pitch-Synchronous Overlap-Add) resynthesis, which operates entirely in the time domain. The source's F0 contour is extracted using the shared Hirst two-pass method. A Praat Manipulation object is created from the Vevo audio, its pitch tier is replaced with the source contour, and Praat resynthesises via overlap-add.

PSOLA works by identifying individual pitch pulses (glottal cycles) and repositioning them in time to match the new pitch contour, then blending overlapping segments. Because it never decomposes or re-synthesises the spectral envelope, formants stay exactly where Vevo placed them — no double-vocoding, no spectral smearing, no formant shift. This is the standard pitch modification technique in speech research precisely because of this property.

The tradeoff: PSOLA can introduce minor "buzzy" artefacts on very large pitch shifts (more than ~1 octave), but for same-gender timbre transfer the shifts are typically small and the output is clean.

**Steps 3–4** apply the same post-processing as WORLD: timbre strength blending (mixing the output back toward the source if strength < 100%), RMS energy envelope transfer, and WORLD-based aperiodicity blending for breathiness.

### Pitch slider behaviour

The pitch parameter controls how much target pitch bleeds into the output:

| Pitch slider | Behaviour |
|---|---|
| Disabled (default) | 100% source pitch — pure timbre transfer |
| Enabled at 50% | Log-interpolated F0 halfway between source and target |
| Enabled at 100% | Full target pitch (original inference_fm behaviour) |

The log-interpolation operates in the perceptual domain (a semitone is a multiplicative ratio), so a 50% blend doesn't just average Hz values — it finds the geometric midpoint, which sounds perceptually "halfway."

---

## 3. Compare

**Module:** `src/comparison/engine.py`

After a transform, the comparison engine extracts fresh profiles from the source, the output, and the target, then computes:

- **Per-parameter similarity scores (0–100%):** how close the output landed to the target on each dimension that was enabled.
- **Acoustic deltas:** raw numeric differences (source→output, output↔target) for every measured field.
- **Perceptual deltas:** same structure for the perceptual dimensions.
- **Leakage detection:** if a parameter was *disabled* during the transform but the output still drifted more than 10% from the source on that dimension, a warning is surfaced. This catches unintended side effects — e.g., disabling rhythm but finding the output's speech rate changed because the neural model entangled rhythm with timbre.

---

## Upgrade paths

**PitchVC backend.** A lighter neural model designed specifically for the "change timbre, keep pitch" task. Would skip the pitch-correction step entirely since the model handles the decomposition internally. Planned as a third backend option.

---

## Visualizations

The UI offers three views per voice (in a tabbed panel):

- **Profile card** — acoustic stats + perceptual dimension bars.
- **F0 contour** — SVG plot of the Praat two-pass F0, click-to-seek linked to the audio player. In the comparison view, all three contours (source/output/target) overlay on shared axes with colour coding.
- **Mel spectrogram** — pre-rendered PNG (axis-less, edge-to-edge for accurate click-to-seek), generated by librosa and cached on disk keyed by file mtime.

All three visualizations are lazy-loaded per tab (opening the F0 tab triggers the `/api/f0-contour` fetch; opening Mel triggers the `/api/mel-spectrogram` image load). Once loaded, switching between tabs is instant.

---

## Export

The "Download report (zip)" button in the comparison view bundles:

- `source.<ext>`, `target.<ext>`, `output.wav` — the three audio files.
- `report.html` — a self-contained page (inline CSS, local audio players, score bars, delta tables, provenance) that works offline when extracted alongside the audio files.
- `comparison.json` — the full structured data for programmatic use.
