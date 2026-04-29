# Examples

## Reproducible transform configs

Pass them to the CLI with:

```bash
python cli.py transform \
    --source path/to/source.wav --target path/to/target.wav \
    --output path/to/output.wav \
    --config examples/timbre_only.json
```

| Config | What it does |
| --- | --- |
| `timbre_only.json` | Borrow only the target's vocal-tract shape. Keeps source pitch, energy, rhythm, breathiness intact. |
| `pitch_and_timbre_blend.json` | Full pitch + timbre transfer with partial energy/breathiness/formants blending and rhythm disabled — a balanced "make me sound like them, but keep my cadence" preset. |

## Worked examples (audio + comparison reports)

Two end-to-end runs you can listen to without setting up the pipeline yourself.

| Folder | Contents |
| --- | --- |
| [`female/`](female/) | Female-voice transfer: `source.wav`, `target.wav`, `output.wav`, plus `comparison.json` (per-metric deltas + similarity scores) and a standalone `report.html`. |
| [`male/`](male/) | Male-voice transfer: `source.wav`, `target.mp3`, `output.wav`, plus the same comparison artifacts. |

Open the `report.html` in a browser for the rendered three-way comparison, or inspect `comparison.json` for raw numbers.

To regenerate either run, drop the source/target into the CLI:

```bash
python cli.py transform \
    --source examples/female/source.wav \
    --target examples/female/target.wav \
    --output regenerated.wav
python cli.py compare \
    --source examples/female/source.wav \
    --output regenerated.wav \
    --target examples/female/target.wav
```
