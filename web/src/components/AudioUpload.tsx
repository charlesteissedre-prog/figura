import { useCallback, useRef, useState } from "react";
import { uploadAudio, extractProfile } from "../api";
import type { VoiceProfile } from "../types";

interface Props {
  label: string;
  onReady: (id: string, profile: VoiceProfile) => void;
}

export default function AudioUpload({ label, onReady }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "extracting" | "done" | "error">("idle");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setError("");
    try {
      setStatus("uploading");
      const { id } = await uploadAudio(file);
      setStatus("extracting");
      const profile = await extractProfile(id, file.name);
      setStatus("done");
      onReady(id, profile);
    } catch (e: any) {
      setStatus("error");
      setError(e.message);
    }
  }, [onReady]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div
      className="audio-upload"
      onDrop={onDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      <div className="upload-label">{label}</div>
      {status === "idle" && <div className="upload-hint">Drop audio file or click to browse</div>}
      {status === "uploading" && <div className="upload-status">Uploading {fileName}...</div>}
      {status === "extracting" && <div className="upload-status">Analyzing {fileName}...</div>}
      {status === "done" && <div className="upload-status done">{fileName}</div>}
      {status === "error" && <div className="upload-status error">{error}</div>}
    </div>
  );
}
