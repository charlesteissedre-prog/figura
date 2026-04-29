import { useState } from "react";
import ScoreBar from "./ScoreBar";
import DeltaTable from "./DeltaTable";
import AudioPlayer from "./AudioPlayer";
import ProfileCard from "./ProfileCard";
import DetailTabs from "./DetailTabs";
import F0ContourOverlay from "./F0ContourOverlay";
import MelSpectrogram from "./MelSpectrogram";
import { exportZip } from "../api";
import type { ComparisonResult, TransformConfig, VoiceProfile } from "../types";

const SOURCE_COLOR = "#3b82f6";
const OUTPUT_COLOR = "#2e9e5e";
const TARGET_COLOR = "#8b5cf6";

interface TransformResult {
  sourceId: string;
  targetId: string;
  outputId: string;
  sourceProfile: VoiceProfile;
  targetProfile: VoiceProfile;
  outputProfile: VoiceProfile;
  config: TransformConfig;
  comparison: ComparisonResult;
}

interface Props {
  transformResult: TransformResult | null;
}

export default function ComparisonView({ transformResult }: Props) {
  if (!transformResult) {
    return (
      <div className="comparison-view empty">
        <p>Run a transform first to see the comparison.</p>
      </div>
    );
  }

  const { comparison } = transformResult;
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const handleDownload = async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      const { blob, filename } = await exportZip({
        source_id: transformResult.sourceId,
        output_id: transformResult.outputId,
        target_id: transformResult.targetId,
        config: transformResult.config,
        backend: "",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setDownloadError(e.message || String(e));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="comparison-view">
      <div className="comparison-toolbar">
        <button
          className="btn btn-secondary"
          onClick={handleDownload}
          disabled={downloading}
          title="Download source, target, output audio + a standalone HTML report"
        >
          {downloading ? "Preparing…" : "⬇ Download report (zip)"}
        </button>
        {downloadError && <span className="comparison-toolbar-error">{downloadError}</span>}
      </div>

      {/* Column headers + audio players */}
      <div className="comparison-audio-row">
        <div>
          <div className="comparison-col-header source">
            <span className="col-dot" />
            <span className="col-label">{transformResult.sourceProfile.name}</span>
          </div>
          <AudioPlayer audioId={transformResult.sourceId} />
        </div>
        <div>
          <div className="comparison-col-header output">
            <span className="col-dot" />
            <span className="col-label">Output</span>
          </div>
          <AudioPlayer audioId={transformResult.outputId} />
        </div>
        <div>
          <div className="comparison-col-header target">
            <span className="col-dot" />
            <span className="col-label">{transformResult.targetProfile.name}</span>
          </div>
          <AudioPlayer audioId={transformResult.targetId} />
        </div>
      </div>

      <DetailTabs
        key={`${transformResult.sourceId}-${transformResult.outputId}-${transformResult.targetId}`}
        children={{
          profile: (
            <div className="profiles-row profiles-row-3">
              <ProfileCard profile={transformResult.sourceProfile} label="Source" color={SOURCE_COLOR} />
              <ProfileCard profile={transformResult.outputProfile} label="Output" color={OUTPUT_COLOR} />
              <ProfileCard profile={transformResult.targetProfile} label="Target" color={TARGET_COLOR} />
            </div>
          ),
          f0: (
            <F0ContourOverlay
              series={[
                { id: transformResult.sourceId, label: "Source", color: SOURCE_COLOR },
                { id: transformResult.outputId, label: "Output", color: OUTPUT_COLOR },
                { id: transformResult.targetId, label: "Target", color: TARGET_COLOR },
              ]}
            />
          ),
          mel: (
            <div className="mel-stack">
              {[
                { id: transformResult.sourceId, label: "Source", color: SOURCE_COLOR },
                { id: transformResult.outputId, label: "Output", color: OUTPUT_COLOR },
                { id: transformResult.targetId, label: "Target", color: TARGET_COLOR },
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

      {/* Overall score */}
      <div className="overall-score">
        <div className="section-title">Overall similarity</div>
        <ScoreBar label="Overall" percent={comparison.overall_score_pct} />
      </div>

      {/* Per-param scores */}
      <div className="param-scores">
        <div className="section-title">Per-parameter scores</div>
        {comparison.param_scores.map((ps) => (
          <ScoreBar key={ps.param} label={ps.param} percent={ps.score_pct} leaked={ps.leaked} />
        ))}
      </div>

      <DeltaTable deltas={comparison.acoustic_deltas} title="Acoustic deltas" />
      <DeltaTable deltas={comparison.dim_deltas} title="Perceptual deltas" />

      {comparison.leakage_summary.length > 0 && (
        <div className="leakage-warnings">
          <div className="section-title warning">Leakage warnings</div>
          <ul>
            {comparison.leakage_summary.map((msg, i) => (
              <li key={i}>{msg}</li>
            ))}
          </ul>
        </div>
      )}

      {(comparison.tags_kept.length > 0 || comparison.tags_gained.length > 0 || comparison.tags_lost.length > 0) && (
        <div className="tag-diff">
          <div className="section-title">Tags</div>
          {comparison.tags_kept.length > 0 && <div>Kept: {comparison.tags_kept.join(", ")}</div>}
          {comparison.tags_gained.length > 0 && <div className="tag-gained">Gained: {comparison.tags_gained.join(", ")}</div>}
          {comparison.tags_lost.length > 0 && <div className="tag-lost">Lost: {comparison.tags_lost.join(", ")}</div>}
        </div>
      )}
    </div>
  );
}
