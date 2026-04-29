import { useEffect, useRef, useState } from "react";
import { fetchF0Contour, type F0ContourData } from "../api";
import { seekAudio } from "../audioControl";

interface Props {
  fileId: string;
}

const W = 900;
const H = 220;
const PAD_L = 42;
const PAD_R = 12;
const PAD_T = 14;
const PAD_B = 24;

export default function F0Contour({ fileId }: Props) {
  const [data, setData] = useState<F0ContourData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    fetchF0Contour(fileId)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(String(e.message || e)); });
    return () => { cancelled = true; };
  }, [fileId]);

  // Track the audio element's playback head so we can draw a cursor
  useEffect(() => {
    const el = document.getElementById(`audio-${fileId}`) as HTMLAudioElement | null;
    if (!el) return;
    const tick = () => setPlayhead(el.currentTime);
    el.addEventListener("timeupdate", tick);
    el.addEventListener("seeked", tick);
    tick();
    return () => {
      el.removeEventListener("timeupdate", tick);
      el.removeEventListener("seeked", tick);
    };
  }, [fileId, data]);

  if (error) return <div className="viz-empty">F0 extraction failed: {error}</div>;
  if (!data) return <div className="viz-empty">Analyzing pitch…</div>;
  if (!data.times.length) return <div className="viz-empty">No voiced frames detected</div>;

  const voicedHz = data.hz.filter((v): v is number => v != null);
  if (!voicedHz.length) return <div className="viz-empty">No voiced frames detected</div>;

  const tMin = data.times[0];
  const tMax = data.times[data.times.length - 1] || 1;
  const hzMin = Math.max(40, Math.min(...voicedHz) - 10);
  const hzMax = Math.max(...voicedHz) + 10;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xOf = (t: number) => PAD_L + ((t - tMin) / (tMax - tMin)) * plotW;
  const yOf = (hz: number) => PAD_T + (1 - (hz - hzMin) / (hzMax - hzMin)) * plotH;

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const xRatio = (e.clientX - rect.left) / rect.width;
    const xInView = xRatio * W;
    const clamped = Math.max(PAD_L, Math.min(W - PAD_R, xInView));
    const t = tMin + ((clamped - PAD_L) / plotW) * (tMax - tMin);
    seekAudio(fileId, t, { play: true });
    setPlayhead(t);
  };

  // Build path, breaking on unvoiced frames (null hz)
  const segments: string[] = [];
  let current: string[] = [];
  for (let i = 0; i < data.times.length; i++) {
    const hz = data.hz[i];
    if (hz == null) {
      if (current.length) segments.push(current.join(" "));
      current = [];
    } else {
      const cmd = current.length ? "L" : "M";
      current.push(`${cmd}${xOf(data.times[i]).toFixed(1)},${yOf(hz).toFixed(1)}`);
    }
  }
  if (current.length) segments.push(current.join(" "));

  // Y-axis gridlines
  const ticks = niceTicks(hzMin, hzMax, 4);

  return (
    <div className="viz-panel">
      <div className="viz-meta">
        <span>median {data.median_hz ?? "—"} Hz</span>
        <span>voiced {Math.round(data.voiced_ratio * 100)}%</span>
        <span>range {Math.round(hzMin)}–{Math.round(hzMax)} Hz</span>
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
        <text x={PAD_L} y={H - 6} fontSize={10} fill="var(--text-dim)">0s</text>
        <text x={W - PAD_R} y={H - 6} fontSize={10} fill="var(--text-dim)" textAnchor="end">
          {tMax.toFixed(2)}s
        </text>
        {segments.map((d, i) => (
          <path key={i} d={d} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
        ))}
        {playhead != null && playhead >= tMin && playhead <= tMax && (
          <line
            x1={xOf(playhead)} x2={xOf(playhead)}
            y1={PAD_T} y2={H - PAD_B}
            stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="3 3" opacity={0.7}
          />
        )}
      </svg>
    </div>
  );
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
