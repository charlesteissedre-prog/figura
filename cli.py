#!/usr/bin/env python3
"""
voice-transform CLI
Usage examples:

  # Analyse a voice file and print its profile
  python cli.py analyze path/to/voice.wav --name "Speaker A"

  # Run a transform (source → target), save output, print comparison
  python cli.py transform \
      --source path/to/source.wav \
      --target path/to/target.wav \
      --output path/to/output.wav \
      --disable rhythm \
      --strength timbre=0.8 pitch=1.0

  # Compare source, output, and target profiles
  python cli.py compare \
      --source path/to/source.wav \
      --output path/to/output.wav \
      --target path/to/target.wav

  # Save a voice profile to JSON
  python cli.py analyze path/to/voice.wav --save profiles/speaker_a.json

  # Load a saved config and re-run
  python cli.py transform --config configs/run_01.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.extractor import extract, VoiceProfile
from src.transform.pipeline import TransformConfig, ParamConfig, run
from src.comparison.engine import compare


PARAMS = ["pitch", "timbre", "energy", "rhythm", "breathiness", "formants"]


def cmd_analyze(args):
    print(f"Analysing {args.file}...")
    profile = extract(args.file, name=args.name or "")
    data = profile.to_dict()
    _print_profile(data)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nProfile saved to {args.save}")


def cmd_transform(args):
    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        config = TransformConfig.from_dict(cfg_dict["params"])
        source = cfg_dict["source"]
        target = cfg_dict["target"]
        output = cfg_dict["output"]
    else:
        config = _build_config(args)
        source = args.source
        target = args.target
        output = args.output

    print(f"Running transform: {source} → {output}")
    print(f"  Target reference: {target}")
    _print_config(config)

    result_path = run(source, target, config, output, backend=args.backend)
    print(f"\nOutput saved to {result_path}")

    if args.compare:
        _run_compare(source, result_path, target, config, backend=args.backend)

    if args.save_config:
        Path(args.save_config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_config, "w") as f:
            json.dump({
                "source": source, "target": target, "output": output,
                "params": config.to_dict()
            }, f, indent=2)
        print(f"Config saved to {args.save_config}")


def cmd_compare(args):
    _run_compare(args.source, args.output, args.target, config=None,
                 backend=getattr(args, "backend", None))


def _run_compare(source_path, output_path, target_path, config=None, backend=None):
    print("\nExtracting profiles for comparison...")
    src = extract(source_path, "source")
    out = extract(output_path, "output")
    tgt = extract(target_path, "target")

    if config is None:
        config = TransformConfig()

    result = compare(src, out, tgt, config, backend=backend)

    print(f"\n--- Comparison ---")
    print(f"Overall similarity to target: {result.overall_score_pct:.0f}%")

    print("\nPer-parameter scores:")
    for ps in result.param_scores:
        leak = " [LEAKED]" if ps.leaked else ""
        print(f"  {ps.param:12s} {ps.score_pct:5.0f}%{leak}")

    if result.leakage_summary:
        print("\nLeakage warnings:")
        for msg in result.leakage_summary:
            print(f"  ! {msg}")

    print("\nAcoustic deltas (src → out | out ↔ tgt):")
    for d in result.acoustic_deltas:
        if d.src_val is None: continue
        d1 = f"{d.delta_src_out:+.1f}" if d.delta_src_out is not None else "  n/a"
        d2 = f"{d.delta_out_tgt:+.1f}" if d.delta_out_tgt is not None else "  n/a"
        flag = " !" if d.status == "leaked" else ("  ✓" if d.status == "transferred" else "")
        print(f"  {d.field:18s}  src={d.src_val}  out={d.out_val}  tgt={d.tgt_val}  "
              f"Δ1={d1}  Δ2={d2}{flag}")

    print("\nTag diff:")
    if result.tags_kept:    print(f"  Kept:    {', '.join(result.tags_kept)}")
    if result.tags_gained:  print(f"  Gained:  {', '.join(result.tags_gained)}")
    if result.tags_lost:    print(f"  Lost:    {', '.join(result.tags_lost)}")
    if result.tags_tgt_missing: print(f"  Missing from target: {', '.join(result.tags_tgt_missing)}")


def _build_config(args) -> TransformConfig:
    disabled = set(args.disable or [])
    strengths = {}
    for s in (args.strength or []):
        k, v = s.split("=")
        strengths[k] = float(v)
    # On Vevo 2, energy transfer adds ~1.6 pp shimmer for marginal gain. Auto-
    # disable unless the user explicitly set its strength on the CLI.
    if getattr(args, "backend", None) == "vevo2" and "energy" not in strengths:
        disabled.add("energy")
    params = {}
    for p in PARAMS:
        params[p] = ParamConfig(
            enabled=(p not in disabled),
            strength=strengths.get(p, 1.0),
        )
    return TransformConfig(**params)


def _print_profile(d: dict):
    print(f"\n  Name:          {d['name']}")
    print(f"  Duration:      {d['duration_s']}s  ({d['sample_rate']} Hz)")
    print(f"  F0:            mean={d['f0_mean_hz']} Hz  range={d['f0_min_hz']}–{d['f0_max_hz']} Hz")
    print(f"  HNR:           {d['hnr_db']} dB")
    print(f"  Formants:      F1={d['f1_mean_hz']}  F2={d['f2_mean_hz']}  F3={d['f3_mean_hz']} Hz")
    print(f"  Jitter:        {d['jitter_pct']}%   Shimmer: {d['shimmer_pct']}%")
    print(f"  Perceptual:    brightness={d['brightness']}  breathiness={d['breathiness']}  "
          f"roughness={d['roughness']}  tension={d['tension']}")


def _print_config(config: TransformConfig):
    print("  Parameters:")
    for p in PARAMS:
        pc = getattr(config, p)
        status = f"strength={pc.strength:.2f}" if pc.enabled else "DISABLED (frozen)"
        print(f"    {p:12s}  {status}")


def main():
    parser = argparse.ArgumentParser(prog="voice-transform")
    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Extract and print a voice profile")
    p_analyze.add_argument("file")
    p_analyze.add_argument("--name", default="")
    p_analyze.add_argument("--save", help="Save profile as JSON")

    # transform
    p_transform = sub.add_parser("transform", help="Run a voice transform")
    p_transform.add_argument("--source")
    p_transform.add_argument("--target")
    p_transform.add_argument("--output", default="output.wav")
    p_transform.add_argument("--disable", nargs="*", choices=PARAMS)
    p_transform.add_argument("--strength", nargs="*", metavar="PARAM=VALUE")
    p_transform.add_argument("--backend", choices=["auto", "vevo", "vevo2", "world"], default="auto")
    p_transform.add_argument("--compare", action="store_true")
    p_transform.add_argument("--config", help="Load run config from JSON")
    p_transform.add_argument("--save-config", help="Save run config to JSON")

    # compare
    p_compare = sub.add_parser("compare", help="Compare profiles from existing audio files")
    p_compare.add_argument("--source", required=True)
    p_compare.add_argument("--output", required=True)
    p_compare.add_argument("--target", required=True)
    p_compare.add_argument("--backend", choices=["auto", "vevo", "vevo2", "world"], default=None,
                           help="Backend used to produce --output (enables coupling-aware leakage detection)")

    args = parser.parse_args()
    if args.cmd == "analyze":      cmd_analyze(args)
    elif args.cmd == "transform":  cmd_transform(args)
    elif args.cmd == "compare":    cmd_compare(args)
    else:                          parser.print_help()


if __name__ == "__main__":
    main()
