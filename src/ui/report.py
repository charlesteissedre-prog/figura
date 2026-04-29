"""Build a self-contained comparison report (HTML) and a zip bundle that
includes it alongside source/target/output audio."""
from __future__ import annotations

import dataclasses
import html
import io
import json
import zipfile
from dataclasses import is_dataclass
from pathlib import Path


_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #faf9f4;
  color: #2a2a28;
  margin: 0;
  padding: 32px 48px;
  line-height: 1.45;
}
h1 { font-size: 22px; margin: 0 0 6px; }
h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #999;
  margin: 32px 0 12px;
  font-weight: 600;
}
.meta { color: #999; font-size: 12px; margin-bottom: 24px; }
.voices-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}
.voice-card {
  background: #fff;
  border: 1px solid #e5e4e0;
  border-radius: 10px;
  padding: 14px;
}
.voice-card.source { border-left: 4px solid #3b82f6; }
.voice-card.output { border-left: 4px solid #2e9e5e; }
.voice-card.target { border-left: 4px solid #8b5cf6; }
.voice-card-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #999;
  margin-bottom: 4px;
}
.voice-card-name { font-weight: 600; font-size: 14px; margin-bottom: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.voice-card audio { width: 100%; margin-bottom: 10px; }
.stat-grid {
  display: grid;
  grid-template-columns: 1fr auto;
  row-gap: 3px;
  column-gap: 12px;
  font-size: 12px;
}
.stat-key { color: #999; }
.stat-val { font-variant-numeric: tabular-nums; text-align: right; }
.score-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.score-label { flex: 0 0 130px; font-size: 13px; }
.score-bar {
  flex: 1;
  height: 10px;
  background: #eee;
  border-radius: 999px;
  overflow: hidden;
  position: relative;
}
.score-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #e8a020, #2e9e5e);
  border-radius: 999px;
}
.score-pct { flex: 0 0 48px; text-align: right; font-weight: 600; font-size: 13px; font-variant-numeric: tabular-nums; }
.leaked { color: #d94040; }
.overall .score-bar { height: 14px; }
.overall .score-label { font-weight: 600; }
.delta-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.delta-table th, .delta-table td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid #eee;
}
.delta-table th {
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.6px;
  color: #999;
  font-weight: 600;
}
.delta-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.delta-table tr.transferred td:first-child { border-left: 3px solid #2e9e5e; }
.delta-table tr.leaked td:first-child { border-left: 3px solid #d94040; }
.delta-table tr.unchanged td:first-child { border-left: 3px solid #ddd; }
.warnings {
  background: #fff4e6;
  border: 1px solid #f0c181;
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 13px;
}
.warnings ul { margin: 6px 0 0; padding-left: 20px; }
.provenance {
  background: #fafaf7;
  border: 1px solid #e5e4e0;
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prov-row { display: flex; gap: 12px; align-items: center; }
.prov-key { flex: 0 0 80px; text-transform: uppercase; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; color: #999; text-align: right; }
.prov-param { display: inline-block; border: 1px solid #e5e4e0; border-radius: 999px; padding: 2px 10px; margin-right: 4px; font-size: 11px; background: #fff; }
"""


_ACOUSTIC_LABELS = {
    "f0_mean_hz": "F0 mean (Hz)",
    "f0_min_hz": "F0 min (Hz)",
    "f0_max_hz": "F0 max (Hz)",
    "hnr_db": "HNR (dB)",
    "f1_mean_hz": "F1 (Hz)",
    "f2_mean_hz": "F2 (Hz)",
    "f3_mean_hz": "F3 (Hz)",
    "jitter_pct": "Jitter (%)",
    "shimmer_pct": "Shimmer (%)",
    "cpp_db": "CPP (dB)",
    "speech_rate_syl_per_s": "Rate (syl/s)",
}

_PERCEPTUAL_LABELS = {
    "brightness": "Brightness",
    "breathiness": "Breathiness",
    "nasality": "Nasality",
    "roughness": "Roughness",
    "tension": "Tension",
}


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def _voice_card(role: str, profile, audio_rel: str) -> str:
    pairs = [(_ACOUSTIC_LABELS[k], _fmt(getattr(profile, k, None))) for k in _ACOUSTIC_LABELS if hasattr(profile, k)]
    rows = "".join(f'<div class="stat-key">{_esc(k)}</div><div class="stat-val">{_esc(v)}</div>' for k, v in pairs)
    label = {"source": "Source", "output": "Output", "target": "Target"}[role]
    return f"""
<div class="voice-card {role}">
  <div class="voice-card-label">{label}</div>
  <div class="voice-card-name" title="{_esc(profile.name)}">{_esc(profile.name)}</div>
  <audio controls src="{_esc(audio_rel)}" preload="metadata"></audio>
  <div class="stat-grid">{rows}</div>
</div>
"""


def _score_row(label: str, pct: float, leaked: bool = False, overall: bool = False) -> str:
    cls = "leaked" if leaked else ""
    flag = " ⚠ leaked" if leaked else ""
    width = max(0.0, min(100.0, float(pct)))
    wrap_cls = "score-row overall" if overall else "score-row"
    return f"""
<div class="{wrap_cls}">
  <div class="score-label {cls}">{_esc(label)}{flag}</div>
  <div class="score-bar"><div class="score-bar-fill" style="width: {width:.1f}%"></div></div>
  <div class="score-pct {cls}">{width:.0f}%</div>
</div>
"""


def _delta_table(title: str, deltas, label_map: dict) -> str:
    rows = []
    for d in deltas:
        field = d.field if hasattr(d, "field") else d.get("field")
        if field not in label_map:
            continue
        status = (d.status if hasattr(d, "status") else d.get("status")) or ""
        src = _fmt(d.src_val if hasattr(d, "src_val") else d.get("src_val"))
        out = _fmt(d.out_val if hasattr(d, "out_val") else d.get("out_val"))
        tgt = _fmt(d.tgt_val if hasattr(d, "tgt_val") else d.get("tgt_val"))
        d1 = d.delta_src_out if hasattr(d, "delta_src_out") else d.get("delta_src_out")
        d2 = d.delta_out_tgt if hasattr(d, "delta_out_tgt") else d.get("delta_out_tgt")
        d1s = f"{d1:+.1f}" if d1 is not None else "—"
        d2s = f"{d2:+.1f}" if d2 is not None else "—"
        rows.append(
            f"<tr class='{_esc(status)}'>"
            f"<td>{_esc(label_map[field])}</td>"
            f"<td class='num'>{_esc(src)}</td>"
            f"<td class='num'>{_esc(out)}</td>"
            f"<td class='num'>{_esc(tgt)}</td>"
            f"<td class='num'>{_esc(d1s)}</td>"
            f"<td class='num'>{_esc(d2s)}</td>"
            f"<td>{_esc(status)}</td>"
            f"</tr>"
        )
    if not rows:
        return ""
    return f"""
<h2>{_esc(title)}</h2>
<table class="delta-table">
  <thead>
    <tr>
      <th>Field</th><th>Source</th><th>Output</th><th>Target</th>
      <th>Δ src→out</th><th>Δ out↔tgt</th><th>Status</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _provenance_block(backend: str, config, created_at: str) -> str:
    from datetime import datetime, timezone
    params = []
    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    for name, pc in cfg_dict.items():
        if name == "mode":
            continue
        if isinstance(pc, dict) and pc.get("enabled"):
            pct = int(round(float(pc.get("strength", 1.0)) * 100))
            params.append(f'<span class="prov-param">{_esc(name)} {pct}%</span>')
    when = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""
<div class="provenance">
  <div class="prov-row"><div class="prov-key">Backend</div><div>{_esc(backend or "—")}</div></div>
  <div class="prov-row"><div class="prov-key">Params</div><div>{''.join(params) or '—'}</div></div>
  <div class="prov-row"><div class="prov-key">Created</div><div>{_esc(when)}</div></div>
</div>
"""


def build_report_html(
    src_profile, out_profile, tgt_profile,
    comparison,
    config,
    backend: str,
    src_audio_name: str,
    out_audio_name: str,
    tgt_audio_name: str,
    created_at: str = "",
    title: str = "Voice transform report",
) -> str:
    """Render a standalone HTML comparison report. Audio srcs reference
    sibling files in the same directory (relative paths), so the HTML works
    when extracted from the export zip."""
    def _pick(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    overall = _pick(comparison, "overall_score_pct", 0) or 0
    param_scores = _pick(comparison, "param_scores", []) or []
    leakage = _pick(comparison, "leakage_summary", []) or []
    acoustic = _pick(comparison, "acoustic_deltas", []) or []
    dim = _pick(comparison, "dim_deltas", []) or []

    voices = (
        _voice_card("source", src_profile, src_audio_name)
        + _voice_card("output", out_profile, out_audio_name)
        + _voice_card("target", tgt_profile, tgt_audio_name)
    )

    score_rows = [_score_row("Overall", overall, overall=True)]
    for ps in param_scores:
        param = _pick(ps, "param", "")
        pct = _pick(ps, "score_pct", 0) or 0
        leaked = bool(_pick(ps, "leaked", False))
        score_rows.append(_score_row(param, pct, leaked=leaked))

    warnings_html = ""
    if leakage:
        items = "".join(f"<li>{_esc(msg)}</li>" for msg in leakage)
        warnings_html = f'<h2>Leakage warnings</h2><div class="warnings"><ul>{items}</ul></div>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
  <h1>{_esc(title)}</h1>
  <div class="meta">{_esc(created_at) or "Generated report"}</div>

  <h2>Voices</h2>
  <div class="voices-row">{voices}</div>

  <h2>Similarity</h2>
  {''.join(score_rows)}

  {_delta_table("Acoustic deltas", acoustic, _ACOUSTIC_LABELS)}
  {_delta_table("Perceptual deltas", dim, _PERCEPTUAL_LABELS)}

  {warnings_html}

  <h2>Provenance</h2>
  {_provenance_block(backend, config, created_at)}
</body>
</html>"""


def build_export_zip(
    *,
    src_path: str,
    out_path: str,
    tgt_path: str,
    src_profile,
    out_profile,
    tgt_profile,
    comparison,
    config,
    backend: str,
    created_at: str = "",
) -> bytes:
    """Bundle the three audio files + a standalone report.html + comparison.json into a zip."""
    src_name = f"source{Path(src_path).suffix.lower()}"
    tgt_name = f"target{Path(tgt_path).suffix.lower()}"
    out_name = f"output{Path(out_path).suffix.lower()}"

    html_str = build_report_html(
        src_profile, out_profile, tgt_profile,
        comparison, config, backend,
        src_audio_name=src_name, out_audio_name=out_name, tgt_audio_name=tgt_name,
        created_at=created_at,
    )

    def _to_jsonable(obj):
        if is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return obj
        return obj

    json_payload = {
        "source_profile": _to_jsonable(src_profile),
        "output_profile": _to_jsonable(out_profile),
        "target_profile": _to_jsonable(tgt_profile),
        "comparison": _to_jsonable(comparison),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "backend": backend,
        "created_at": created_at,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(src_path, src_name)
        z.write(tgt_path, tgt_name)
        z.write(out_path, out_name)
        z.writestr("report.html", html_str)
        z.writestr("comparison.json", json.dumps(json_payload, indent=2, default=str))
    return buf.getvalue()
