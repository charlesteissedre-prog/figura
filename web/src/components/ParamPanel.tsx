import type { TransformConfig, ParamConfig } from "../types";

type ParamKey = keyof TransformConfig;

interface Props {
  config: TransformConfig;
  onChange: (config: TransformConfig) => void;
}

const PARAMS: { key: ParamKey; label: string; desc: string }[] = [
  { key: "pitch", label: "Pitch (F0)", desc: "Fundamental frequency contour" },
  { key: "timbre", label: "Timbre", desc: "Spectral envelope / vocal tract" },
  { key: "energy", label: "Energy / loudness", desc: "RMS envelope shaping" },
  { key: "rhythm", label: "Rhythm / rate", desc: "Speaking rate & pause patterns" },
  { key: "breathiness", label: "Breathiness", desc: "Glottal aperiodicity component" },
  { key: "formants", label: "Formants (F1-F3)", desc: "Resonance frequencies only" },
];

export default function ParamPanel({ config, onChange }: Props) {
  const update = (name: ParamKey, patch: Partial<ParamConfig>) => {
    onChange({ ...config, [name]: { ...config[name], ...patch } });
  };

  return (
    <div className="param-grid">
      {PARAMS.map(({ key, label, desc }) => {
        const pc = config[key];
        return (
          <div key={key} className={`param-card ${pc.enabled ? "" : "disabled"}`}>
            <div className="param-card-header">
              <span className="param-card-name">{label}</span>
              <button
                className={`param-card-toggle ${pc.enabled ? "on" : ""}`}
                onClick={() => update(key, { enabled: !pc.enabled })}
              />
            </div>
            <div className="param-card-desc">{desc}</div>
            <div className="param-card-slider">
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={pc.strength}
                disabled={!pc.enabled}
                onChange={(e) => update(key, { strength: parseFloat(e.target.value) })}
              />
              <span className="param-card-pct">{Math.round(pc.strength * 100)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
