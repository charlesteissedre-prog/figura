import { useEffect, useState } from "react";
import type { VoiceProfile, Gender } from "../types";
import type { Provenance } from "../api";
import GenderIcon from "./GenderIcon";

interface Props {
  profile: VoiceProfile | null;
  label?: string;
  color?: string;
  editable?: boolean;
  provenance?: Provenance | null;
  /** Map of voice id → display name, used to label the source/target chips. */
  voiceNames?: Record<string, string>;
  onJumpToVoice?: (id: string) => void;
  onSave?: (patch: { display_name: string; gender: Gender; tags: string[]; notes: string }) => Promise<void>;
}

const ACOUSTIC = [
  { key: "f0_mean_hz", label: "F0 mean", unit: "Hz" },
  { key: null, label: "F0 range", unit: "Hz" },
  { key: "hnr_db", label: "HNR", unit: "dB" },
  { key: "f1_mean_hz", label: "F1 mean", unit: "Hz" },
  { key: "f2_mean_hz", label: "F2 mean", unit: "Hz" },
  { key: "jitter_pct", label: "Jitter", unit: "%" },
  { key: "shimmer_pct", label: "Shimmer", unit: "%" },
  { key: "cpp_db", label: "CPP", unit: "dB" },
  { key: "speech_rate_syl_per_s", label: "Speech-rate", unit: "syl/s" },
] as const;

const DIMENSIONS: { key: string; left: string; right: string }[] = [
  { key: "brightness", left: "dark", right: "bright" },
  { key: "roughness", left: "smooth", right: "rough" },
  { key: "nasality", left: "oral", right: "nasal" },
  { key: "breathiness", left: "modal", right: "breathy" },
  { key: "tension", left: "lax", right: "tense" },
];

const GENDER_OPTIONS: { value: Gender; label: string }[] = [
  { value: null, label: "Unknown" },
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
  { value: "neutral", label: "Neutral" },
];

const AVATAR_COLORS = ["#e8a020", "#3b82f6", "#8b5cf6", "#2e9e5e", "#d94040", "#ec4899"];

function getInitials(name: string): string {
  const parts = name.replace(/\.[^.]+$/, "").split(/[\s_-]+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function fmt(v: number | null | undefined): string {
  if (v == null) return "--";
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

export default function ProfileCard({ profile, color, editable = false, provenance, voiceNames, onJumpToVoice, onSave }: Props) {
  const [name, setName] = useState(profile?.name ?? "");
  const [gender, setGender] = useState<Gender>(profile?.gender ?? null);
  const [tags, setTags] = useState<string[]>(profile?.tags ?? []);
  const [notes, setNotes] = useState(profile?.notes ?? "");
  const [tagInput, setTagInput] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState("");

  useEffect(() => {
    if (profile) {
      setName(profile.name);
      setGender(profile.gender ?? null);
      setTags(profile.tags ?? []);
      setNotes(profile.notes ?? "");
      setStatus("idle");
      setErr("");
    }
  }, [profile]);

  const persist = async (patch: { display_name: string; gender: Gender; tags: string[]; notes: string }) => {
    if (!onSave) return;
    setStatus("saving");
    setErr("");
    try {
      await onSave(patch);
      setStatus("saved");
    } catch (e) {
      setStatus("error");
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  if (!profile) {
    return (
      <div className="profile-card empty">
        <div className="profile-card-placeholder">No audio loaded</div>
      </div>
    );
  }

  const avatarColor = color || AVATAR_COLORS[Math.abs(hashCode(name)) % AVATAR_COLORS.length];

  // Immediate save for discrete changes
  const changeGender = (g: Gender) => {
    setGender(g);
    persist({ display_name: name.trim() || profile.name, gender: g, tags, notes });
  };

  const addTag = () => {
    const t = tagInput.trim();
    if (!t) return;
    setTagInput("");
    if (tags.includes(t)) return;
    const next = [...tags, t];
    setTags(next);
    persist({ display_name: name.trim() || profile.name, gender, tags: next, notes });
  };

  const removeTag = (t: string) => {
    const next = tags.filter((x) => x !== t);
    setTags(next);
    persist({ display_name: name.trim() || profile.name, gender, tags: next, notes });
  };

  // Save-on-blur for text fields
  const commitName = () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setErr("Name can't be empty");
      setName(profile.name);
      return;
    }
    if (trimmed === profile.name) return;
    persist({ display_name: trimmed, gender, tags, notes });
  };

  const commitNotes = () => {
    if (notes === (profile.notes ?? "")) return;
    persist({ display_name: name.trim() || profile.name, gender, tags, notes });
  };

  return (
    <div className="profile-card">
      <div className="profile-card-header">
        <div className="profile-avatar" style={{ background: avatarColor }}>
          {getInitials(name)}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          {editable ? (
            <input
              className="profile-card-title-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={commitName}
              onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
              placeholder="Voice name"
            />
          ) : (
            <div className="profile-card-title">
              {gender && <GenderIcon gender={gender} size={18} className="profile-gender-icon" />}
              {name}
            </div>
          )}
          <div className="profile-card-meta">
            {profile.sample_rate ? `${(profile.sample_rate / 1000).toFixed(1)} kHz` : ""}
            {profile.duration_s ? ` \u00b7 ${profile.duration_s.toFixed(1)}s` : ""}
          </div>
        </div>
      </div>

      {editable && (
        <div className="profile-section">
          <div className="profile-section-title">Gender</div>
          <div className="gender-segmented">
            {GENDER_OPTIONS.map((o) => (
              <button
                key={String(o.value)}
                type="button"
                className={`gender-chip ${gender === o.value ? "active" : ""}`}
                onClick={() => changeGender(o.value)}
              >
                {o.value && <GenderIcon gender={o.value} size={14} />} {o.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="profile-section">
        <div className="profile-section-title">Acoustic measurements</div>
        <div className="acoustic-grid">
          {ACOUSTIC.map((a) => (
            <div key={a.label}>
              <div className="acoustic-item-label">{a.label}</div>
              <div className="acoustic-item-value">
                {a.key
                  ? fmt(profile[a.key as keyof VoiceProfile] as number | null)
                  : `${fmt(profile.f0_min_hz)}\u2013${fmt(profile.f0_max_hz)}`}
                <span className="acoustic-item-unit"> {a.unit}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {(editable || tags.length > 0) && (
        <div className="profile-section">
          <div className="profile-section-title">Perceptual tags</div>
          <div className="tag-chips">
            {tags.map((t) => (
              <span key={t} className="tag-chip">
                {t}
                {editable && (
                  <button className="tag-chip-remove" onClick={() => removeTag(t)} title="Remove tag">×</button>
                )}
              </span>
            ))}
          </div>
          {editable && (
            <div className="tag-add-row">
              <input
                type="text"
                placeholder="Add a tag…"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
              />
              <button className="btn btn-secondary" onClick={addTag} disabled={!tagInput.trim()}>Add</button>
            </div>
          )}
        </div>
      )}

      <div className="profile-section">
        <div className="profile-section-title">Rated dimensions</div>
        {DIMENSIONS.map((d) => {
          const v = profile[d.key as keyof VoiceProfile] as number | null;
          return (
            <div key={d.key} className="dim-row">
              <span className="dim-label-left">{d.left}</span>
              <div className="dim-track">
                <div className="dim-fill" style={{ width: v != null ? `${v}%` : "0%" }} />
              </div>
              <span className="dim-label-right">{d.right}</span>
              <span className="dim-value">{v != null ? Math.round(v) : "--"}</span>
            </div>
          );
        })}
      </div>

      {(editable || notes) && (
        <div className="profile-section">
          <div className="profile-section-title">Notes</div>
          {editable ? (
            <textarea
              className="profile-notes-input"
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              onBlur={commitNotes}
              placeholder="Recording context, intended use, quirks…"
            />
          ) : (
            <div className="profile-notes">{notes}</div>
          )}
        </div>
      )}

      {provenance && (
        <div className="profile-section">
          <div className="profile-section-title">How it was made</div>
          <ProvenancePanel p={provenance} voiceNames={voiceNames} onJumpToVoice={onJumpToVoice} />
        </div>
      )}

      {editable && onSave && (
        <div className="profile-card-status">
          {status === "saving" && <span className="save-indicator saving">Saving…</span>}
          {status === "saved" && <span className="save-indicator saved">✓ Saved</span>}
          {status === "error" && <span className="save-indicator error">Failed: {err}</span>}
        </div>
      )}
    </div>
  );
}

function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

function ProvenancePanel({
  p,
  voiceNames,
  onJumpToVoice,
}: {
  p: Provenance;
  voiceNames?: Record<string, string>;
  onJumpToVoice?: (id: string) => void;
}) {
  const nameFor = (id: string) => voiceNames?.[id] ?? `(deleted · ${id.slice(0, 6)})`;
  const missing = (id: string) => !voiceNames || !(id in voiceNames);
  const when = new Date(p.created_at);
  const whenLabel = isNaN(when.getTime()) ? p.created_at : when.toLocaleString();

  // Show only parameters that differ from the "neutral" (disabled, strength 1) state
  const paramEntries = Object.entries(p.config).filter(([k, v]) => {
    if (k === "mode") return false;
    const pc = v as { enabled: boolean; strength: number };
    return pc && typeof pc === "object" && pc.enabled;
  });

  return (
    <div className="provenance-panel">
      <div className="provenance-row">
        <span className="provenance-key">Source</span>
        <button
          type="button"
          className={`provenance-chip source ${missing(p.source_id) ? "dangling" : ""}`}
          disabled={missing(p.source_id) || !onJumpToVoice}
          onClick={() => onJumpToVoice?.(p.source_id)}
        >
          {nameFor(p.source_id)}
        </button>
      </div>
      <div className="provenance-row">
        <span className="provenance-key">Target</span>
        <button
          type="button"
          className={`provenance-chip target ${missing(p.target_id) ? "dangling" : ""}`}
          disabled={missing(p.target_id) || !onJumpToVoice}
          onClick={() => onJumpToVoice?.(p.target_id)}
        >
          {nameFor(p.target_id)}
        </button>
      </div>
      <div className="provenance-row">
        <span className="provenance-key">Backend</span>
        <span className="provenance-backend">{p.backend}</span>
      </div>
      {paramEntries.length > 0 && (
        <div className="provenance-row provenance-row-block">
          <span className="provenance-key">Params</span>
          <div className="provenance-params">
            {paramEntries.map(([name, v]) => {
              const pc = v as { enabled: boolean; strength: number };
              return (
                <span key={name} className="provenance-param">
                  <span className="provenance-param-name">{name}</span>
                  <span className="provenance-param-pct">{Math.round(pc.strength * 100)}%</span>
                </span>
              );
            })}
          </div>
        </div>
      )}
      <div className="provenance-row provenance-meta">
        <span className="provenance-key">Created</span>
        <span className="provenance-value">{whenLabel}</span>
      </div>
    </div>
  );
}
