export type LibFilter = "all" | "voices" | "outputs";

interface Props {
  value: LibFilter;
  onChange: (f: LibFilter) => void;
  counts?: { all: number; voices: number; outputs: number };
}

const OPTIONS: { key: LibFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "voices", label: "Voices" },
  { key: "outputs", label: "Outputs" },
];

export default function LibraryFilter({ value, onChange, counts }: Props) {
  return (
    <div className="lib-filter">
      {OPTIONS.map(({ key, label }) => (
        <button
          key={key}
          className={`lib-filter-btn ${value === key ? "active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
          {counts && <span className="lib-filter-count">{counts[key]}</span>}
        </button>
      ))}
    </div>
  );
}
