interface Props {
  label: string;
  percent: number;
  leaked?: boolean;
}

export default function ScoreBar({ label, percent, leaked }: Props) {
  const clamped = Math.max(0, Math.min(100, percent));
  return (
    <div className="score-bar">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className={`score-bar-fill ${leaked ? "leaked" : ""}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      <span className="score-bar-value">{Math.round(percent)}%</span>
      {leaked && <span className="leaked-badge">LEAKED</span>}
    </div>
  );
}
