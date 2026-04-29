import { audioUrl } from "../api";

interface Props {
  audioId: string | null;
  label?: string;
}

export default function AudioPlayer({ audioId, label }: Props) {
  if (!audioId) return null;
  return (
    <div className="audio-player">
      {label && <span className="audio-player-label">{label}</span>}
      <audio
        id={`audio-${audioId}`}
        controls
        src={audioUrl(audioId)}
        preload="metadata"
      />
    </div>
  );
}
