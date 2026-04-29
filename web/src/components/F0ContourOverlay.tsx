import { useEffect, useRef, useState } from "react";
import { fetchF0Contour, type F0ContourData } from "../api";
import { seekAudio } from "../audioControl";

interface Series {
  id: string;
  label: string;
  color: string;
}

interface Props {
  series: Series[];
}

const W = 900;
const H = 260;
const PAD_L = 44;
const PAD_R = 12;
const PAD_T = 16;
const PAD_B = 28;

export default function F0ContourOverlay({ series }: Props) {
  const [data, setData] = useState<Record<string, F0ContourData | null>>({});
  const [error, setError] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData({});
    setError(null);
    Promise.all(
      series.map((s) =>
        fetchF0Contour(s.id).then((d) => [s.id, d] as const),
      ),
    )
      .then((pairs) => {
        if (!cancelled) setData(Object.fromEntries(pairs));
      })
      .catch((e) => { if (!cancelled) setError(String(e.message || e)); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.map((s) => s.id).join("|")]);

  if (error) return <div className="viz-empty">F0 extraction failed: {error}</div>;
  const ready = series.every((s) => data[s.id]);
  if (!ready) return <div className="viz-empty">Analyzing pitch…</div>;

  // Global time & hz bounds across all series
  let tMax = 0;
  const allHz: number[] = [];
  for (const s of series) {
    const d = data[s.id]!;
    if (d.times.length) tMax = Math.max(tMax, d.times[d.times.length - 1]);
    for (const v of d.hz) if (v != null) allHz.push(v);
  }
  if (!allHz.length) return <div className="viz-empty">No voiced frames detected</div>;
  if (tMax <= 0) tMax = 1;
  const hzMin = Math.max(40, Math.min(...allHz) - 10);
  const hzMax = Math.max(...allHz) + 10;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xOf = (t: number) => PAD_L + (t / tMax) * plotW;
  const yOf = (hz: number) => PAD_T + (1 - (hz - hzMin) / (hzMax - hzMin)) * plotH;

  const ticks = niceTicks(hzMin, hzMax, 4);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const xRatio = (e.clientX - rect.left) / rect.width;
    const xInView = xRatio * W;
    const clamped = Math.max(PAD_L, Math.min(W - PAD_R, xInView));
    const t = ((clamped - PAD_L) / plotW) * tMax;
    for (const s of series) seekAudio(s.id, t, { play: true });
  };

  return (
    <div className="viz-panel">
      <div className="viz-meta">
        {series.map((s) => {
          const d = data[s.id]!;
          return (
            <span key={s.id} className="viz-legend-item">
              <span className="viz-legend-swatch" style={{ background: s.color }} />
              {s.label}: median {d.median_hz ?? "—"} Hz · voiced {Math.round(d.voiced_ratio * 100)}%
            </span>
          );
        })}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="viz-svg viz-svg-clickable"
        preserveAspectRatio="none"
        onClick={handleClick}
      >
        {ticks.map((hz) => (
          <g key={hz}>
            <line
              x1={PAD_L} x2={W - PAD_R}
              y1={yOf(hz)} y2={yOf(hz)}
              stroke="var(--border)" strokeWidth={1} strokeDasharray="2 3"
            />
            <text
              x={PAD_L - 6} y={yOf(hz) + 3}
              textAnchor="end" fontSize={10} fill="var(--text-dim)"
            >{hz}</text>
          </g>
        ))}
        <line x1={PAD_L} x2={W - PAD_R} y1={H - PAD_B} y2={H - PAD_B} stroke="var(--border-strong)" />
        <text x={PAD_L} y={H - 8} fontSize={10} fill="var(--text-dim)">0s</text>
        <text x={W - PAD_R} y={H - 8} fontSize={10} fill="var(--text-dim)" textAnchor="end">
          {tMax.toFixed(2)}s
        </text>
        {series.map((s) => {
          const d = data[s.id]!;
          const segments = buildSegments(d, xOf, yOf);
          return segments.map((path, i) => (
            <path
              key={`${s.id}-${i}`}
              d={path}
              fill="none"
              stroke={s.color}
              strokeWidth={1.5}
              opacity={0.85}
            />
          ));
        })}
      </svg>
    </div>
  );
}

function buildSegments(
  d: F0ContourData,
  xOf: (t: number) => number,
  yOf: (hz: number) => number,
): string[] {
  const segments: string[] = [];
  let current: string[] = [];
  for (let i = 0; i < d.times.length; i++) {
    const hz = d.hz[i];
    if (hz == null) {
      if (current.length) segments.push(current.join(" "));
      current = [];
    } else {
      const cmd = current.length ? "L" : "M";
      current.push(`${cmd}${xOf(d.times[i]).toFixed(1)},${yOf(hz).toFixed(1)}`);
    }
  }
  if (current.length) segments.push(current.join(" "));
  return segments;
}

function niceTicks(min: number, max: number, count: number): number[] {
  const step = niceStep((max - min) / count);
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max; v += step) out.push(Math.round(v));
  return out;
}
function niceStep(raw: number): number {
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / pow;
  if (n < 1.5) return pow;
  if (n < 3) return 2 * pow;
  if (n < 7) return 5 * pow;
  return 10 * pow;
}
