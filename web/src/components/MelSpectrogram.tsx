import { useEffect, useRef, useState } from "react";
import { fetchAudioInfo, melSpectrogramUrl } from "../api";
import { seekAudio } from "../audioControl";

interface Props {
  fileId: string;
}

export default function MelSpectrogram({ fileId }: Props) {
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const [duration, setDuration] = useState<number | null>(null);
  const [playheadPct, setPlayheadPct] = useState<number | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDuration(null);
    fetchAudioInfo(fileId)
      .then((d) => { if (!cancelled) setDuration(d.duration_s); })
      .catch(() => { /* fall back to DOM duration on click */ });
    return () => { cancelled = true; };
  }, [fileId]);

  // Track playback so we can draw a cursor over the spectrogram
  useEffect(() => {
    const el = document.getElementById(`audio-${fileId}`) as HTMLAudioElement | null;
    if (!el) return;
    const tick = () => {
      const dur = duration ?? (isFinite(el.duration) ? el.duration : null);
      if (dur && dur > 0) setPlayheadPct((el.currentTime / dur) * 100);
    };
    el.addEventListener("timeupdate", tick);
    el.addEventListener("seeked", tick);
    tick();
    return () => {
      el.removeEventListener("timeupdate", tick);
      el.removeEventListener("seeked", tick);
    };
  }, [fileId, duration]);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const xRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    let dur = duration;
    if (dur == null) {
      const el = document.getElementById(`audio-${fileId}`) as HTMLAudioElement | null;
      if (el && isFinite(el.duration)) dur = el.duration;
    }
    if (!dur) return;
    const t = xRatio * dur;
    seekAudio(fileId, t, { play: true });
    setPlayheadPct(xRatio * 100);
  };

  return (
    <div className="viz-panel">
      {status === "loading" && <div className="viz-empty viz-overlay">Rendering spectrogram…</div>}
      {status === "error" && <div className="viz-empty">Failed to load spectrogram</div>}
      <div
        ref={wrapRef}
        className="mel-click-area"
        onClick={status === "ok" ? handleClick : undefined}
        style={{ cursor: status === "ok" && duration ? "pointer" : "default" }}
      >
        <img
          key={fileId}
          src={melSpectrogramUrl(fileId)}
          alt="Mel-spectrogram"
          className="viz-img"
          style={{ display: status === "error" ? "none" : "block" }}
          onLoad={() => setStatus("ok")}
          onError={() => setStatus("error")}
          draggable={false}
        />
        {playheadPct != null && (
          <div className="mel-playhead" style={{ left: `${playheadPct}%` }} />
        )}
      </div>
      {duration != null && (
        <div className="mel-time-axis">
          <span>0.00s</span>
          <span>{(duration / 2).toFixed(2)}s</span>
          <span>{duration.toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}
