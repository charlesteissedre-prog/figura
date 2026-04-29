import { useState } from "react";
import TransformView from "./components/TransformView";
import ComparisonView from "./components/ComparisonView";
import LibraryView from "./components/LibraryView";
import PipelineView from "./components/PipelineView";
import { extractProfile, runCompare } from "./api";
import type { TransformConfig, VoiceProfile, ComparisonResult } from "./types";
import type { Provenance } from "./api";
import "./App.css";

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

type Tab = "library" | "transform" | "compare" | "pipeline";

function App() {
  const [tab, setTab] = useState<Tab>("transform");
  const [transformResult, setTransformResult] = useState<TransformResult | null>(null);
  const [prefill, setPrefill] = useState<Provenance | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const handleViewComparison = async (req: {
    sourceId: string; targetId: string; outputId: string; config: TransformConfig;
  }) => {
    setLoadingComparison(true);
    setTab("compare");
    try {
      const [sourceProfile, targetProfile, outputProfile, comparison] = await Promise.all([
        extractProfile(req.sourceId, "source"),
        extractProfile(req.targetId, "target"),
        extractProfile(req.outputId, "output"),
        runCompare({
          source_id: req.sourceId,
          output_id: req.outputId,
          target_id: req.targetId,
          config: req.config,
        }),
      ]);
      setTransformResult({
        sourceId: req.sourceId,
        targetId: req.targetId,
        outputId: req.outputId,
        sourceProfile,
        targetProfile,
        outputProfile,
        config: req.config,
        comparison,
      });
    } catch (e) {
      console.error("Comparison load failed:", e);
      setTab("library");
    } finally {
      setLoadingComparison(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Voice Transform</h1>
        <nav className="tab-nav">
          <button
            className={tab === "library" ? "active" : ""}
            onClick={() => setTab("library")}
          >
            Library
          </button>
          <button
            className={tab === "transform" ? "active" : ""}
            onClick={() => setTab("transform")}
          >
            Transform
          </button>
          <button
            className={`${tab === "compare" ? "active" : ""} ${transformResult ? "" : "disabled-tab"}`}
            onClick={() => transformResult && setTab("compare")}
          >
            Compare
          </button>
          <button
            className={tab === "pipeline" ? "active" : ""}
            onClick={() => setTab("pipeline")}
          >
            Pipeline
          </button>
        </nav>
      </header>
      <main>
        {tab === "library" && (
          <LibraryView
            onRerun={(p) => {
              setPrefill(p);
              setTab("transform");
            }}
            onViewComparison={handleViewComparison}
          />
        )}
        {tab === "transform" && (
          <TransformView
            prefill={prefill}
            onPrefillConsumed={() => setPrefill(null)}
            onTransformDone={(result) => {
              setTransformResult(result);
              setTab("compare");
            }}
          />
        )}
        {tab === "compare" && (
          loadingComparison
            ? <div className="comparison-view empty"><p>Loading comparison...</p></div>
            : <ComparisonView transformResult={transformResult} />
        )}
        {tab === "pipeline" && <PipelineView />}
      </main>
    </div>
  );
}

export default App;
