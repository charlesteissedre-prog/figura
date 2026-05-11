import { useCallback, useEffect, useState } from "react";
import AudioPlayer from "./AudioPlayer";
import ParamPanel from "./ParamPanel";
import GenderIcon from "./GenderIcon";
import ProfileCard from "./ProfileCard";
import DetailTabs from "./DetailTabs";
import F0ContourOverlay from "./F0ContourOverlay";
import MelSpectrogram from "./MelSpectrogram";
import LibraryFilter, { type LibFilter } from "./LibraryFilter";
import { listLibrary, uploadAudio, extractProfile, runTransform, deleteFromLibrary } from "../api";
import type { TransformConfig, VoiceProfile, ComparisonResult, Gender, Backend } from "../types";
import { DEFAULT_CONFIG } from "../types";

const SOURCE_COLOR = "#3b82f6";
const TARGET_COLOR = "#8b5cf6";

interface LibEntry {
  id: string;
  filename: string;
  display_name: string;
  extension: string;
  gender: Gender;
  notes: string;
  tags: string[];
  kind: "upload" | "output";
  profile: VoiceProfile | null;
}

function stripExt(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}

function getExt(name: string): string {
  const m = name.match(/\.([^.]+)$/);
  return m ? m[1].toLowerCase() : "";
}

interface Props {
  prefill?: import("../api").Provenance | null;
  onPrefillConsumed?: () => void;
  onTransformDone: (result: {
    sourceId: string;
    targetId: string;
    outputId: string;
    sourceProfile: VoiceProfile;
    targetProfile: VoiceProfile;
    outputProfile: VoiceProfile;
    config: TransformConfig;
    comparison: ComparisonResult;
  }) => void;
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

export default function TransformView({ prefill, onPrefillConsumed, onTransformDone }: Props) {
  const [library, setLibrary] = useState<LibEntry[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [targetId, setTargetId] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<"source" | "target" | null>(null);
  const [outputId, setOutputId] = useState<string | null>(null);
  const [config, setConfig] = useState<TransformConfig>(DEFAULT_CONFIG);
  const [backend, setBackend] = useState<Backend>("auto");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [menu, setMenu] = useState<{ x: number; y: number; entryId: string } | null>(null);
  const [libFilter, setLibFilter] = useState<LibFilter>("voices");

  // Dismiss the library context menu on any click / Escape / scroll
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", close, true);
    };
  }, [menu]);

  // Apply prefill from a Re-run action (fired from the Library view)
  useEffect(() => {
    if (!prefill) return;
    setSourceId(prefill.source_id);
    setTargetId(prefill.target_id);
    setConfig(prefill.config);
    if (prefill.backend) setBackend(prefill.backend);
    onPrefillConsumed?.();
  }, [prefill, onPrefillConsumed]);

  useEffect(() => {
    if (!running) return;
    setElapsed(0);
    const startedAt = performance.now();
    const id = window.setInterval(() => {
      setElapsed((performance.now() - startedAt) / 1000);
    }, 100);
    return () => window.clearInterval(id);
  }, [running]);

  // Load library
  useEffect(() => {
    listLibrary()
      .then((files) => setLibrary(files.map((f) => ({
        id: f.id,
        filename: f.filename,
        display_name: f.display_name,
        extension: f.extension ?? getExt(f.filename),
        gender: f.gender,
        notes: f.notes,
        tags: f.tags ?? [],
        kind: f.kind ?? "upload",
        profile: null,
      }))))
      .catch(() => {});
  }, []);

  const sourceEntry = library.find((e) => e.id === sourceId);
  const targetEntry = library.find((e) => e.id === targetId);

  const lazyLoadProfile = useCallback(async (entry: LibEntry) => {
    if (entry.profile) return;
    try {
      const profile = await extractProfile(entry.id, entry.filename);
      setLibrary((prev) => prev.map((e) => (e.id === entry.id ? { ...e, profile } : e)));
    } catch { /* ignore */ }
  }, []);

  // Ensure profiles are loaded for whichever slots are filled (covers prefill)
  useEffect(() => {
    if (sourceEntry && !sourceEntry.profile) lazyLoadProfile(sourceEntry);
    if (targetEntry && !targetEntry.profile) lazyLoadProfile(targetEntry);
  }, [sourceEntry, targetEntry, lazyLoadProfile]);

  const assignAs = useCallback((role: "source" | "target", entry: LibEntry) => {
    if (role === "source") {
      setSourceId(entry.id);
      if (targetId === entry.id) setTargetId(null);
    } else {
      setTargetId(entry.id);
      if (sourceId === entry.id) setSourceId(null);
    }
    lazyLoadProfile(entry);
  }, [sourceId, targetId, lazyLoadProfile]); // used by drag-drop only

  const swapSourceTarget = () => {
    setSourceId(targetId);
    setTargetId(sourceId);
  };

  // Drag from library item
  const handleLibDragStart = (entry: LibEntry) => (e: React.DragEvent) => {
    e.dataTransfer.setData("application/x-figura-lib-id", entry.id);
    e.dataTransfer.effectAllowed = "copyMove";
  };

  // Drop onto source/target slot
  const handleSlotDrop = (role: "source" | "target") => (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(null);
    const id = e.dataTransfer.getData("application/x-figura-lib-id");
    const entry = library.find((x) => x.id === id);
    if (entry) assignAs(role, entry);
  };

  const handleSlotDragOver = (role: "source" | "target") => (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes("application/x-figura-lib-id")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      if (dragOver !== role) setDragOver(role);
    }
  };

  // Upload
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    try {
      const { id } = await uploadAudio(file);
      const profile = await extractProfile(id, file.name);
      setLibrary((prev) => [...prev, {
        id,
        filename: file.name,
        display_name: stripExt(file.name),
        extension: getExt(file.name),
        gender: null,
        notes: "",
        tags: [],
        kind: "upload",
        profile,
      }]);
    } catch { /* ignore */ }
  }, []);

  const handleFileInput = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { id } = await uploadAudio(file);
      const profile = await extractProfile(id, file.name);
      setLibrary((prev) => [...prev, {
        id,
        filename: file.name,
        display_name: stripExt(file.name),
        extension: getExt(file.name),
        gender: null,
        notes: "",
        tags: [],
        kind: "upload",
        profile,
      }]);
    } catch { /* ignore */ }
    e.target.value = "";
  }, []);

  const handleDelete = async (id: string) => {
    try {
      await deleteFromLibrary(id);
      setLibrary((prev) => prev.filter((e) => e.id !== id));
      if (sourceId === id) setSourceId(null);
      if (targetId === id) setTargetId(null);
    } catch { /* ignore */ }
  };

  const canRun = sourceId && targetId && !running;

  const handleTransform = async () => {
    if (!sourceId || !targetId) return;
    setRunning(true);
    setError("");
    setOutputId(null);
    try {
      const result = await runTransform({ source_id: sourceId, target_id: targetId, config, backend });
      setOutputId(result.output_id);
      onTransformDone({
        sourceId, targetId,
        outputId: result.output_id,
        sourceProfile: result.source_profile,
        targetProfile: result.target_profile,
        outputProfile: result.profile,
        config,
        comparison: result.comparison,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="transform-layout">
      {/* Library sidebar */}
      <div className="library-sidebar">
        <div className="library-sidebar-title">Voice library</div>
        <LibraryFilter
          value={libFilter}
          onChange={setLibFilter}
          counts={{
            all: library.length,
            voices: library.filter((e) => e.kind !== "output").length,
            outputs: library.filter((e) => e.kind === "output").length,
          }}
        />
        <div className="library-list">
          {library.filter((entry) =>
            libFilter === "all" ? true
            : libFilter === "voices" ? entry.kind !== "output"
            : entry.kind === "output"
          ).map((entry) => {
            const isSource = entry.id === sourceId;
            const isTarget = entry.id === targetId;
            const displayName = entry.display_name || entry.filename;
            return (
              <div
                key={entry.id}
                className={`library-item ${isSource ? "is-source" : ""} ${isTarget ? "is-target" : ""}`}
                draggable
                onDragStart={handleLibDragStart(entry)}
                onClick={() => lazyLoadProfile(entry)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  setMenu({ x: e.clientX, y: e.clientY, entryId: entry.id });
                }}
                title="Drag to a slot — or right-click for Source / Target"
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
                    {isSource && <span className="role-badge source">SRC</span>}
                    {isTarget && <span className="role-badge target">TGT</span>}
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
          onClick={() => document.getElementById("lib-file-input")?.click()}
        >
          <input id="lib-file-input" type="file" accept="audio/*" style={{ display: "none" }} onChange={handleFileInput} />
          <div className="library-drop-text">Drop audio file here</div>
        </div>
      </div>

      {/* Main area */}
      <div className="transform-main">
        {/* Selection */}
        <div className="selection-area">
          <div className="selection-slots">
            <div
              className={`selection-slot source ${!sourceEntry ? "empty" : ""} ${dragOver === "source" ? "drag-over" : ""}`}
              onDrop={handleSlotDrop("source")}
              onDragOver={handleSlotDragOver("source")}
              onDragLeave={() => setDragOver(null)}
            >
              {sourceEntry ? (
                <div>
                  <div className="selection-slot-label">Source</div>
                  <div className="selection-slot-name">{sourceEntry.display_name || sourceEntry.filename}</div>
                  <button className="selection-slot-clear" onClick={() => setSourceId(null)} title="Clear">×</button>
                </div>
              ) : (
                <span className="slot-placeholder">Drop a voice here</span>
              )}
            </div>
            <button
              className="selection-swap"
              onClick={swapSourceTarget}
              disabled={!sourceId && !targetId}
              title="Swap source and target"
            >⇄</button>
            <div
              className={`selection-slot target ${!targetEntry ? "empty" : ""} ${dragOver === "target" ? "drag-over" : ""}`}
              onDrop={handleSlotDrop("target")}
              onDragOver={handleSlotDragOver("target")}
              onDragLeave={() => setDragOver(null)}
            >
              {targetEntry ? (
                <div>
                  <div className="selection-slot-label">Target</div>
                  <div className="selection-slot-name">{targetEntry.display_name || targetEntry.filename}</div>
                  <button className="selection-slot-clear" onClick={() => setTargetId(null)} title="Clear">×</button>
                </div>
              ) : (
                <span className="slot-placeholder">Drop a voice here</span>
              )}
            </div>
          </div>

          {(sourceEntry?.id || targetEntry?.id) && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
              {sourceEntry?.id && <AudioPlayer audioId={sourceEntry.id} label="Source" />}
              {targetEntry?.id && <AudioPlayer audioId={targetEntry.id} label="Target" />}
            </div>
          )}
        </div>

        {sourceEntry && targetEntry && (
          <div>
            <div className="section-title">Voice comparison</div>
            <DetailTabs
              key={`${sourceEntry.id}-${targetEntry.id}`}
              children={{
                profile: (
                  <div className="profiles-row">
                    <ProfileCard profile={sourceEntry.profile} label="Source" color={SOURCE_COLOR} />
                    <ProfileCard profile={targetEntry.profile} label="Target" color={TARGET_COLOR} />
                  </div>
                ),
                f0: (
                  <F0ContourOverlay
                    series={[
                      { id: sourceEntry.id, label: "Source", color: SOURCE_COLOR },
                      { id: targetEntry.id, label: "Target", color: TARGET_COLOR },
                    ]}
                  />
                ),
                mel: (
                  <div className="mel-stack">
                    {[
                      { id: sourceEntry.id, label: "Source", color: SOURCE_COLOR },
                      { id: targetEntry.id, label: "Target", color: TARGET_COLOR },
                    ].map((s) => (
                      <div key={s.id} className="mel-stack-item">
                        <div className="mel-stack-label">
                          <span className="col-dot" style={{ background: s.color }} />
                          {s.label}
                        </div>
                        <MelSpectrogram fileId={s.id} />
                      </div>
                    ))}
                  </div>
                ),
              }}
            />
          </div>
        )}

        {/* Parameters */}
        <div>
          <div className="section-title">Parameters</div>
          <ParamPanel config={config} onChange={setConfig} backend={backend} />
        </div>

        {/* Controls */}
        <div className="transform-controls">
          <label className="backend-select">
            Backend:
            <select value={backend} onChange={(e) => setBackend(e.target.value as Backend)}>
              <option value="auto">Auto (Vevo, WORLD fallback)</option>
              <option value="vevo">Vevo (timbre)</option>
              <option value="vevo2">Vevo 2 (timbre)</option>
              <option value="world">WORLD</option>
            </select>
          </label>
          <button className="btn btn-secondary">Export config</button>
          <button className="btn btn-primary" disabled={!canRun} onClick={handleTransform}>
            {running ? "Transforming..." : "Run transform"}
          </button>
        </div>

        {running && (
          <div className="transform-progress">
            <div className="transform-progress-bar">
              <div className="transform-progress-indeterminate" />
            </div>
            <div className="transform-progress-label">
              Transforming… {elapsed.toFixed(1)}s
            </div>
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}

        {outputId && (
          <div className="output-section">
            <div className="output-section-title">Output</div>
            <AudioPlayer audioId={outputId} label="Transformed" />
          </div>
        )}
      </div>

      {menu && (() => {
        const entry = library.find((e) => e.id === menu.entryId);
        if (!entry) return null;
        const displayName = entry.display_name || entry.filename;
        const pick = (role: "source" | "target") => (e: React.MouseEvent) => {
          e.stopPropagation();
          assignAs(role, entry);
          setMenu(null);
        };
        return (
          <div
            className="lib-context-menu"
            style={{ left: menu.x, top: menu.y }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="lib-context-menu-header">{displayName}</div>
            <button className="lib-context-menu-item" onClick={pick("source")}>
              <span className="lib-context-menu-dot" style={{ background: "#3b82f6" }} />
              Use as Source
            </button>
            <button className="lib-context-menu-item" onClick={pick("target")}>
              <span className="lib-context-menu-dot" style={{ background: "#8b5cf6" }} />
              Use as Target
            </button>
          </div>
        );
      })()}
    </div>
  );
}
