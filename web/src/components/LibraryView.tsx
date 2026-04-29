import { useCallback, useEffect, useState } from "react";
import AudioPlayer from "./AudioPlayer";
import ProfileCard from "./ProfileCard";
import GenderIcon from "./GenderIcon";
import DetailTabs from "./DetailTabs";
import F0Contour from "./F0Contour";
import MelSpectrogram from "./MelSpectrogram";
import LibraryFilter, { type LibFilter } from "./LibraryFilter";
import { listLibrary, deleteFromLibrary, extractProfile, uploadAudio, updateLibraryEntry } from "../api";
import type { Provenance } from "../api";
import type { VoiceProfile, Gender } from "../types";

interface LibEntry {
  id: string;
  filename: string;
  display_name: string;
  extension: string;
  gender: Gender;
  notes: string;
  tags: string[];
  kind: "upload" | "output";
  provenance: Provenance | null;
  profile: VoiceProfile | null;
}

function stripExt(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}
function getExt(name: string): string {
  const m = name.match(/\.([^.]+)$/);
  return m ? m[1].toLowerCase() : "";
}

const AVATAR_COLORS = ["#e8a020", "#3b82f6", "#8b5cf6", "#2e9e5e", "#d94040", "#ec4899"];

function getInitials(name: string): string {
  const parts = name.replace(/\.[^.]+$/, "").split(/[\s_-]+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function hashColor(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
}

interface CompareRequest {
  sourceId: string;
  targetId: string;
  outputId: string;
  config: import("../types").TransformConfig;
}

interface LibraryViewProps {
  onRerun?: (provenance: Provenance) => void;
  onViewComparison?: (req: CompareRequest) => void;
}

export default function LibraryView({ onRerun, onViewComparison }: LibraryViewProps = {}) {
  const [entries, setEntries] = useState<LibEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [libFilter, setLibFilter] = useState<LibFilter>("all");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const files = await listLibrary();
      setEntries(files.map((f) => ({
        id: f.id,
        filename: f.filename,
        display_name: f.display_name,
        extension: f.extension ?? getExt(f.filename),
        gender: f.gender,
        notes: f.notes,
        tags: f.tags ?? [],
        kind: f.kind ?? "upload",
        provenance: f.provenance ?? null,
        profile: null,
      })));
    } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleSelect = async (entry: LibEntry) => {
    setSelected(entry.id);
    if (!entry.profile) {
      try {
        const profile = await extractProfile(entry.id, entry.filename);
        setEntries((prev) => prev.map((e) => (e.id === entry.id ? { ...e, profile } : e)));
      } catch { /* ignore */ }
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteFromLibrary(id);
      setEntries((prev) => prev.filter((e) => e.id !== id));
      if (selected === id) setSelected(null);
    } catch { /* ignore */ }
  };

  const handleFileInput = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { id } = await uploadAudio(file);
      const profile = await extractProfile(id, file.name);
      const entry: LibEntry = { id, filename: file.name, display_name: stripExt(file.name), extension: getExt(file.name), gender: null, notes: "", tags: [], kind: "upload", provenance: null, profile };
      setEntries((prev) => [...prev, entry]);
      setSelected(id);
    } catch { /* ignore */ }
    e.target.value = "";
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    try {
      const { id } = await uploadAudio(file);
      const profile = await extractProfile(id, file.name);
      setEntries((prev) => [...prev, { id, filename: file.name, display_name: stripExt(file.name), extension: getExt(file.name), gender: null, notes: "", tags: [], kind: "upload", provenance: null, profile }]);
      setSelected(id);
    } catch { /* ignore */ }
  }, []);

  const selectedEntry = entries.find((e) => e.id === selected) || null;

  // Merge user-editable metadata onto the extracted profile for the editor
  const profileForEditor: VoiceProfile | null = selectedEntry
    ? selectedEntry.profile
      ? {
          ...selectedEntry.profile,
          name: selectedEntry.display_name || selectedEntry.filename,
          gender: selectedEntry.gender,
          tags: selectedEntry.tags,
          notes: selectedEntry.notes,
        }
      : null
    : null;

  return (
    <div className="library-view-layout">
      <div className="library-view-sidebar">
        <div className="library-sidebar-title">Voice library</div>
        <LibraryFilter
          value={libFilter}
          onChange={setLibFilter}
          counts={{
            all: entries.length,
            voices: entries.filter((e) => e.kind !== "output").length,
            outputs: entries.filter((e) => e.kind === "output").length,
          }}
        />
        <div className="library-list">
          {loading && <div style={{ fontSize: 12, color: "var(--text-dim)" }}>Loading...</div>}
          {entries.filter((entry) =>
            libFilter === "all" ? true
            : libFilter === "voices" ? entry.kind !== "output"
            : entry.kind === "output"
          ).map((entry) => {
            const displayName = entry.display_name || entry.filename;
            return (
              <div
                key={entry.id}
                className={`library-item ${selected === entry.id ? "selected" : ""}`}
                onClick={() => handleSelect(entry)}
              >
                <div className="library-avatar" style={{ background: hashColor(displayName) }}>
                  {getInitials(displayName)}
                </div>
                <div className="library-item-info">
                  <div className="library-item-name">
                    {entry.gender && <GenderIcon gender={entry.gender} size={14} className="library-item-gender" />}
                    {displayName}
                  </div>
                  <div className="library-item-meta">
                    {entry.kind === "output" && <span className="role-badge output">OUT</span>}
                    {entry.extension && <span className="ext-badge">{entry.extension.toUpperCase()}</span>}
                    {entry.profile ? `${entry.profile.duration_s}s` : ""}
                  </div>
                </div>
                <div className="library-item-actions">
                  <button
                    className="library-item-btn delete"
                    title="Delete"
                    onClick={(e) => { e.stopPropagation(); handleDelete(entry.id); }}
                  >×</button>
                </div>
              </div>
            );
          })}
        </div>
        <div
          className="library-drop"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById("lib-standalone-input")?.click()}
        >
          <input id="lib-standalone-input" type="file" accept="audio/*" style={{ display: "none" }} onChange={handleFileInput} />
          <div className="library-drop-text">Drop audio file here</div>
        </div>
      </div>

      <div className="library-detail">
        {selectedEntry ? (
          <>
            <AudioPlayer audioId={selectedEntry.id} label={selectedEntry.display_name || selectedEntry.filename} />
            {selectedEntry.provenance && (onRerun || onViewComparison) && (
              <div className="rerun-bar">
                {onRerun && (
                  <button className="btn btn-secondary" onClick={() => onRerun(selectedEntry.provenance!)}>
                    Re-run in Transform
                  </button>
                )}
                {onViewComparison && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => onViewComparison({
                      sourceId: selectedEntry.provenance!.source_id,
                      targetId: selectedEntry.provenance!.target_id,
                      outputId: selectedEntry.id,
                      config: selectedEntry.provenance!.config,
                    })}
                  >
                    View comparison
                  </button>
                )}
              </div>
            )}
            <DetailTabs
              key={selectedEntry.id}
              children={{
                profile: (
                  <ProfileCard
                    profile={profileForEditor}
                    editable
                    provenance={selectedEntry.provenance}
                    voiceNames={Object.fromEntries(entries.map((e) => [e.id, e.display_name || e.filename]))}
                    onJumpToVoice={(id) => setSelected(id)}
                    onSave={async (patch) => {
                      const updated = await updateLibraryEntry(selectedEntry.id, patch);
                      setEntries((prev) => prev.map((e) =>
                        e.id === selectedEntry.id
                          ? { ...e, display_name: updated.display_name, gender: updated.gender, notes: updated.notes, tags: updated.tags ?? [] }
                          : e
                      ));
                    }}
                  />
                ),
                f0: <F0Contour fileId={selectedEntry.id} />,
                mel: <MelSpectrogram fileId={selectedEntry.id} />,
              }}
            />
          </>
        ) : (
          <div className="library-detail-empty">
            Select a voice from the library or import a new one
          </div>
        )}
      </div>
    </div>
  );
}
