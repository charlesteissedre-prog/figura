"""Pydantic models mirroring the dataclasses in analysis, transform, and comparison."""
from pydantic import BaseModel
from typing import Optional


class ParamConfigSchema(BaseModel):
    enabled: bool = True
    strength: float = 1.0


class TransformConfigSchema(BaseModel):
    pitch: ParamConfigSchema = ParamConfigSchema()
    timbre: ParamConfigSchema = ParamConfigSchema()
    energy: ParamConfigSchema = ParamConfigSchema()
    rhythm: ParamConfigSchema = ParamConfigSchema(enabled=False)
    breathiness: ParamConfigSchema = ParamConfigSchema()
    formants: ParamConfigSchema = ParamConfigSchema()


class VoiceProfileSchema(BaseModel):
    name: str
    filepath: str
    duration_s: float
    sample_rate: int
    f0_mean_hz: Optional[float] = None
    f0_min_hz: Optional[float] = None
    f0_max_hz: Optional[float] = None
    hnr_db: Optional[float] = None
    f1_mean_hz: Optional[float] = None
    f2_mean_hz: Optional[float] = None
    f3_mean_hz: Optional[float] = None
    jitter_pct: Optional[float] = None
    shimmer_pct: Optional[float] = None
    cpp_db: Optional[float] = None
    speech_rate_syl_per_s: Optional[float] = None
    brightness: Optional[float] = None
    breathiness: Optional[float] = None
    nasality: Optional[float] = None
    roughness: Optional[float] = None
    tension: Optional[float] = None
    gender: Optional[str] = None
    tags: list[str] = []
    notes: str = ""


class FieldDeltaSchema(BaseModel):
    field: str
    src_val: Optional[float] = None
    out_val: Optional[float] = None
    tgt_val: Optional[float] = None
    delta_src_out: Optional[float] = None
    delta_out_tgt: Optional[float] = None
    status: str


class ParamScoreSchema(BaseModel):
    param: str
    score_pct: float
    leaked: bool = False


class ComparisonResultSchema(BaseModel):
    overall_score_pct: float
    param_scores: list[ParamScoreSchema] = []
    acoustic_deltas: list[FieldDeltaSchema] = []
    dim_deltas: list[FieldDeltaSchema] = []
    tags_kept: list[str] = []
    tags_gained: list[str] = []
    tags_lost: list[str] = []
    tags_tgt_missing: list[str] = []
    leakage_summary: list[str] = []


class TransformRequest(BaseModel):
    source_id: str
    target_id: str
    config: TransformConfigSchema = TransformConfigSchema()
    backend: str = "auto"  # "auto" | "vevo" | "vevo2" | "world"


class CompareRequest(BaseModel):
    source_id: str
    output_id: str
    target_id: str
    config: TransformConfigSchema = TransformConfigSchema()
    backend: Optional[str] = None  # for backend-aware leakage detection (e.g. coupling on vevo2)


class ExportRequest(BaseModel):
    source_id: str
    output_id: str
    target_id: str
    config: TransformConfigSchema = TransformConfigSchema()
    backend: str = ""
