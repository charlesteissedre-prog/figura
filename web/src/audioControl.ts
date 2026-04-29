export function seekAudio(audioId: string, timeSec: number, opts: { play?: boolean } = {}): void {
  const el = document.getElementById(`audio-${audioId}`) as HTMLAudioElement | null;
  if (!el) return;
  const clamped = Math.max(0, isFinite(el.duration) ? Math.min(timeSec, el.duration) : timeSec);
  el.currentTime = clamped;
  if (opts.play) void el.play().catch(() => { /* user gesture required — ignore */ });
}
