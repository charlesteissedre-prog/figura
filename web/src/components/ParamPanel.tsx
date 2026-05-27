import type { TransformConfig, ParamConfig, Backend } from "../types";

type ParamKey = keyof TransformConfig;

interface Props {
  config: TransformConfig;
  onChange: (config: TransformConfig) => void;
  backend?: Backend;
}

const PARAMS: { key: ParamKey; label: string; desc: string }[] = [
  { key: "pitch", label: "Pitch (F0)", desc: "Fundamental frequency contour" },
  { key: "timbre", label: "Timbre", desc: "Spectral envelope / vocal tract" },
  { key: "energy", label: "Energy / loudness", desc: "RMS envelope shaping" },
  { key: "rhythm", label: "Rhythm / rate", desc: "Speaking rate & pause patterns" },
  { key: "breathiness", label: "Breathiness", desc: "Glottal aperiodicity component" },
  { key: "formants", label: "Formants (F1-F3)", desc: "Resonance frequencies only" },
];

type ParamMode = "normal" | "binary" | "disabled" | "hidden";

interface ParamRule {
  mode?: ParamMode;
  label?: string;
  desc?: string;
  coupledWith?: ParamKey;
}

const BACKEND_RULES: Partial<Record<Backend, Partial<Record<ParamKey, ParamRule>>>> = {
  vevo2: {
    pitch: { desc: "On = shift toward target's pitch range. Off = preserve source pitch. Strength = how much of model's pitch to keep" },
    timbre: {
      label: "Voice transfer",
      desc: "Blend converted output with raw source (100% = full conversion)",
      coupledWith: "formants",
    },
    energy: { desc: "Off by default on Vevo 2 — the FM model's amplitude is already clean, and energy transfer adds shimmer for marginal gain" },
    formants: { mode: "hidden" },
    rhythm: { mode: "hidden" },
    breathiness: { mode: "hidden" },
  },
};

export default function ParamPanel({ config, onChange, backend = "auto" }: Props) {
  const rules = BACKEND_RULES[backend] ?? {};

  const update = (name: ParamKey, patch: Partial<ParamConfig>) => {
    const next = { ...config, [name]: { ...config[name], ...patch } };
    const couple = rules[name]?.coupledWith;
    if (couple) next[couple] = { ...next[couple], ...patch };
    onChange(next);
  };

  return (
    <div className="param-grid">
      {PARAMS.map(({ key, label, desc }) => {
        const rule = rules[key] ?? {};
        if (rule.mode === "hidden") return null;

        const pc = config[key];
        const isDisabled = rule.mode === "disabled";
        const isBinary = rule.mode === "binary";
        const effectiveLabel = rule.label ?? label;
        const effectiveDesc = rule.desc ?? desc;

        return (
          <div key={key} className={`param-card ${pc.enabled && !isDisabled ? "" : "disabled"}`}>
            <div className="param-card-header">
              <span className="param-card-name">{effectiveLabel}</span>
              <button
                className={`param-card-toggle ${pc.enabled && !isDisabled ? "on" : ""}`}
                disabled={isDisabled}
                onClick={() => !isDisabled && update(key, { enabled: !pc.enabled })}
              />
            </div>
            <div className="param-card-desc">{effectiveDesc}</div>
            <div className="param-card-slider">
              {isBinary ? (
                <span className="param-card-pct" style={{ fontStyle: "italic", opacity: 0.7 }}>
                  {pc.enabled ? "ON" : "OFF"}
                </span>
              ) : (
                <>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={pc.strength}
                    disabled={!pc.enabled || isDisabled}
                    onChange={(e) => update(key, { strength: parseFloat(e.target.value) })}
                  />
                  <span className="param-card-pct">{Math.round(pc.strength * 100)}%</span>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
