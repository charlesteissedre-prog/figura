"""
Comparison engine.
Given a source, output, and target VoiceProfile, computes:
  - per-metric deltas (source→output and output↔target)
  - per-parameter similarity scores
  - leakage detection (unexpected changes on disabled params)
  - tag diff (gained / lost / kept)

Returns a ComparisonResult ready to feed the three-way comparison UI.
"""
from dataclasses import dataclass, field
from typing import Optional
from src.analysis.extractor import VoiceProfile
from src.transform.pipeline import TransformConfig


ACOUSTIC_FIELDS = [
    "f0_mean_hz", "f0_min_hz", "f0_max_hz",
    "hnr_db", "f1_mean_hz", "f2_mean_hz", "f3_mean_hz",
    "jitter_pct", "shimmer_pct", "cpp_db",
]

DIM_FIELDS = ["brightness", "breathiness", "nasality", "roughness", "tension"]

# Map acoustic fields to the TransformConfig param they belong to
# Used for leakage detection: a change on a field whose param is disabled = leak
FIELD_TO_PARAM = {
    "f0_mean_hz": "pitch", "f0_min_hz": "pitch", "f0_max_hz": "pitch",
    "hnr_db": "timbre",
    "f1_mean_hz": "formants", "f2_mean_hz": "formants", "f3_mean_hz": "formants",
    "jitter_pct": "pitch",   # jitter is in the glottal source, adjacent to pitch
    "shimmer_pct": "energy",
    "cpp_db": "breathiness",
    "brightness": "timbre",
    "breathiness": "breathiness",
    "nasality": "timbre",
    "roughness": "timbre",
    "tension": "pitch",
}

LEAK_THRESHOLD = 0.10  # 10% relative change on a disabled param = flagged as leak

# Per-field minimum absolute change before the relative threshold fires.
# Prevents tiny absolute movements on small-valued metrics (e.g. jitter going
# from 3.18% to 2.19% = a 31% relative change but barely audible) from
# producing noisy "leakage" warnings.
FIELD_MIN_ABS_DELTA = {
    "jitter_pct": 1.0,     # absolute percentage points
    "shimmer_pct": 2.0,
}

# Per-backend param couplings: a param that's "disabled" in the config is
# still effectively enabled if the backend transfers it as a side-effect of
# another enabled param. On Vevo 2 the FM model bundles timbre + formants —
# turning timbre on inevitably moves formants too, so changes there aren't
# leakage, they're an intended consequence of timbre transfer.
BACKEND_COUPLINGS = {
    "vevo2": {"formants": "timbre"},
}


@dataclass
class FieldDelta:
    field: str
    src_val: Optional[float]
    out_val: Optional[float]
    tgt_val: Optional[float]
    delta_src_out: Optional[float]   # out - src
    delta_out_tgt: Optional[float]   # out - tgt
    status: str  # "transferred" | "leaked" | "unchanged" | "gap"


@dataclass
class ParamScore:
    param: str
    score_pct: float   # 0-100, how close output is to target for this param
    leaked: bool = False


@dataclass
class ComparisonResult:
    overall_score_pct: float
    param_scores: list = field(default_factory=list)      # list[ParamScore]
    acoustic_deltas: list = field(default_factory=list)   # list[FieldDelta]
    dim_deltas: list = field(default_factory=list)        # list[FieldDelta]
    tags_kept: list = field(default_factory=list)
    tags_gained: list = field(default_factory=list)
    tags_lost: list = field(default_factory=list)
    tags_tgt_missing: list = field(default_factory=list)  # in target but not in output
    leakage_summary: list = field(default_factory=list)   # human-readable strings


def compare(
    src: VoiceProfile,
    out: VoiceProfile,
    tgt: VoiceProfile,
    config: TransformConfig,
    backend: Optional[str] = None,
) -> ComparisonResult:
    result = ComparisonResult(overall_score_pct=0.0)

    _compute_acoustic_deltas(src, out, tgt, config, backend, result)
    _compute_dim_deltas(src, out, tgt, config, backend, result)
    _compute_tag_diff(src, out, tgt, result)
    _compute_param_scores(result, config)
    _compute_overall_score(result)
    _build_leakage_summary(result)

    return result


def _is_param_effectively_enabled(param_name, config, backend) -> bool:
    """A param is effectively enabled if it's enabled directly OR if it's
    coupled to an enabled param under the active backend."""
    param = getattr(config, param_name, None) if param_name else None
    if param is None:
        return True
    if param.enabled:
        return True
    if backend:
        coupled_to = BACKEND_COUPLINGS.get(backend, {}).get(param_name)
        if coupled_to:
            coupled = getattr(config, coupled_to, None)
            if coupled and coupled.enabled:
                return True
    return False


def _delta_status(field_name, delta_src_out, src_val, tgt_val, config, backend) -> str:
    param_name = FIELD_TO_PARAM.get(field_name)
    enabled = _is_param_effectively_enabled(param_name, config, backend)

    if delta_src_out is None or src_val is None:
        return "unchanged"

    rel_change = abs(delta_src_out) / (abs(src_val) + 1e-6)
    min_abs = FIELD_MIN_ABS_DELTA.get(field_name, 0.0)
    significant = rel_change > LEAK_THRESHOLD and abs(delta_src_out) >= min_abs

    if not enabled:
        return "leaked" if significant else "unchanged"

    # Param was enabled (possibly via coupling): did output move toward target?
    if tgt_val is not None and src_val is not None:
        src_dist = abs(tgt_val - src_val)
        out_dist = abs(tgt_val - (src_val + delta_src_out))
        return "transferred" if out_dist < src_dist else "unchanged"

    return "unchanged"


def _compute_acoustic_deltas(src, out, tgt, config, backend, result):
    for f in ACOUSTIC_FIELDS:
        sv = getattr(src, f, None)
        ov = getattr(out, f, None)
        tv = getattr(tgt, f, None)
        d1 = round(ov - sv, 2) if ov is not None and sv is not None else None
        d2 = round(ov - tv, 2) if ov is not None and tv is not None else None
        status = _delta_status(f, d1, sv, tv, config, backend)
        result.acoustic_deltas.append(FieldDelta(
            field=f, src_val=sv, out_val=ov, tgt_val=tv,
            delta_src_out=d1, delta_out_tgt=d2, status=status))


def _compute_dim_deltas(src, out, tgt, config, backend, result):
    for f in DIM_FIELDS:
        sv = getattr(src, f, None)
        ov = getattr(out, f, None)
        tv = getattr(tgt, f, None)
        d1 = round(ov - sv, 1) if ov is not None and sv is not None else None
        d2 = round(ov - tv, 1) if ov is not None and tv is not None else None
        status = _delta_status(f, d1, sv, tv, config, backend)
        result.dim_deltas.append(FieldDelta(
            field=f, src_val=sv, out_val=ov, tgt_val=tv,
            delta_src_out=d1, delta_out_tgt=d2, status=status))


def _compute_tag_diff(src, out, tgt, result):
    src_set = set(src.tags)
    out_set = set(out.tags)
    tgt_set = set(tgt.tags)
    result.tags_kept        = sorted(src_set & out_set)
    result.tags_gained      = sorted(out_set - src_set)
    result.tags_lost        = sorted(src_set - out_set)
    result.tags_tgt_missing = sorted(tgt_set - out_set)


def _compute_param_scores(result, config):
    """
    For each enabled param, compute a 0-100 score based on how close
    the output landed to the target across that param's acoustic fields.
    """
    param_fields = {}
    for f in ACOUSTIC_FIELDS + DIM_FIELDS:
        p = FIELD_TO_PARAM.get(f)
        if p:
            param_fields.setdefault(p, []).append(f)

    all_deltas = {d.field: d for d in result.acoustic_deltas + result.dim_deltas}

    for param_name in ["pitch", "timbre", "energy", "breathiness", "formants"]:
        fields = param_fields.get(param_name, [])
        if not fields:
            continue
        scores = []
        leaked = False
        for f in fields:
            d = all_deltas.get(f)
            if not d or d.src_val is None or d.tgt_val is None or d.out_val is None:
                continue
            if d.status == "leaked":
                leaked = True
            total_dist = abs(d.tgt_val - d.src_val) + 1e-6
            remaining  = abs(d.out_val - d.tgt_val)
            score = max(0, 100 * (1 - remaining / total_dist))
            scores.append(score)
        if scores:
            result.param_scores.append(ParamScore(
                param=param_name,
                score_pct=round(float(sum(scores) / len(scores)), 0),
                leaked=leaked,
            ))


def _compute_overall_score(result):
    if result.param_scores:
        result.overall_score_pct = round(
            float(sum(p.score_pct for p in result.param_scores) / len(result.param_scores)), 0)


def _build_leakage_summary(result):
    all_deltas = result.acoustic_deltas + result.dim_deltas
    leaks = [d for d in all_deltas if d.status == "leaked"]
    for d in leaks:
        change_pct = round(abs(d.delta_src_out or 0) / (abs(d.src_val or 1) + 1e-6) * 100, 0)
        result.leakage_summary.append(
            f"{d.field} changed by {change_pct:.0f}% on a disabled parameter"
        )
