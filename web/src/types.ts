export interface ParamConfig {
  enabled: boolean;
  strength: number;
}

export type Backend = "auto" | "vevo" | "vevo2" | "world";

export interface TransformConfig {
  pitch: ParamConfig;
  timbre: ParamConfig;
  energy: ParamConfig;
  rhythm: ParamConfig;
  breathiness: ParamConfig;
  formants: ParamConfig;
}

export interface VoiceProfile {
  name: string;
  filepath: string;
  duration_s: number;
  sample_rate: number;
  f0_mean_hz: number | null;
  f0_min_hz: number | null;
  f0_max_hz: number | null;
  hnr_db: number | null;
  f1_mean_hz: number | null;
  f2_mean_hz: number | null;
  f3_mean_hz: number | null;
  jitter_pct: number | null;
  shimmer_pct: number | null;
  cpp_db: number | null;
  speech_rate_syl_per_s: number | null;
  brightness: number | null;
  breathiness: number | null;
  nasality: number | null;
  roughness: number | null;
  tension: number | null;
  tags: string[];
  notes: string;
  gender?: Gender;
}

export type Gender = "female" | "male" | "neutral" | null;

export interface FieldDelta {
  field: string;
  src_val: number | null;
  out_val: number | null;
  tgt_val: number | null;
  delta_src_out: number | null;
  delta_out_tgt: number | null;
  status: "transferred" | "leaked" | "unchanged" | "gap";
}

export interface ParamScore {
  param: string;
  score_pct: number;
  leaked: boolean;
}

export interface ComparisonResult {
  overall_score_pct: number;
  param_scores: ParamScore[];
  acoustic_deltas: FieldDelta[];
  dim_deltas: FieldDelta[];
  tags_kept: string[];
  tags_gained: string[];
  tags_lost: string[];
  tags_tgt_missing: string[];
  leakage_summary: string[];
}

export const DEFAULT_CONFIG: TransformConfig = {
  pitch: { enabled: true, strength: 1.0 },
  timbre: { enabled: true, strength: 1.0 },
  energy: { enabled: true, strength: 1.0 },
  rhythm: { enabled: false, strength: 1.0 },
  breathiness: { enabled: true, strength: 1.0 },
  formants: { enabled: true, strength: 1.0 },
};
