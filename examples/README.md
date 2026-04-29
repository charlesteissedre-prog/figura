# Examples

Reproducible transform configs. Pass them to the CLI with:

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

Bring your own `.wav` files. The repo intentionally does not ship audio samples.
