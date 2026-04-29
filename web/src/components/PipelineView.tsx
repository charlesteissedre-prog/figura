export default function PipelineView() {
  return (
    <div className="pipeline-view">
      <div className="pipeline-intro">
        <h2>How Figura works</h2>
        <p>
          A three-stage pipeline: <strong>extract</strong> voice characteristics,
          selectively <strong>transform</strong> one voice toward another,
          then <strong>compare</strong> the result to quantify what transferred
          and what leaked.
        </p>
      </div>

      {/* ── Overview flow ── */}
      <div className="pipeline-flow">
        <div className="pflow-stage">
          <div className="pflow-box extract">Extract</div>
          <div className="pflow-desc">VoiceProfile: F0, formants, HNR, jitter, shimmer, perceptual dimensions</div>
        </div>
        <div className="pflow-arrow" />
        <div className="pflow-stage">
          <div className="pflow-box transform">Transform</div>
          <div className="pflow-desc">WORLD or Vevo timbre — each parameter independently controllable</div>
        </div>
        <div className="pflow-arrow" />
        <div className="pflow-stage">
          <div className="pflow-box compare">Compare</div>
          <div className="pflow-desc">Scores, deltas, leakage warnings</div>
        </div>
      </div>

      {/* ── Extract ── */}
      <section className="pipeline-section">
        <h3>1. Extract</h3>
        <p>
          Every voice goes through Praat-based analysis. Pitch (F0) uses a
          <strong> Hirst two-pass</strong> method: pass 1 on wide bounds with
          elevated <code>octave_cost</code> to suppress halving, then pass 2
          with tightened bounds from the q25/q75 of pass 1.
          The same F0 function feeds both the profile-card stats and the contour plot,
          so they always agree.
        </p>
        <div className="pipeline-detail-grid">
          <div><strong>F0</strong> — Praat autocorrelation, two-pass</div>
          <div><strong>Formants</strong> — Praat Burg (F1–F3)</div>
          <div><strong>Voice quality</strong> — HNR, jitter, shimmer</div>
          <div><strong>Perceptual</strong> — brightness, breathiness, roughness, tension (0–100)</div>
        </div>
      </section>

      {/* ── Transform: WORLD ── */}
      <section className="pipeline-section">
        <h3>2a. WORLD backend</h3>
        <p>
          Decomposes audio into three independent components — F0, spectral envelope,
          aperiodicity — then selectively swaps or blends each toward the target.
        </p>
        <div className="pipeline-steps">
          <Step n="1" title="Decompose" color="var(--blue)">
            WORLD extracts (F0, spectral envelope, aperiodicity) from both source and target.
          </Step>
          <Step n="2" title="Selective swap" color="var(--blue)">
            Pitch: log-ratio shift on F0. Timbre: frequency-axis warping on spectral envelope.
            Breathiness: mean AP blend.
          </Step>
          <Step n="3" title="Synthesise" color="var(--blue)">
            WORLD reconstructs audio from the modified components.
          </Step>
        </div>
        <div className="pipeline-traits">
          <span className="trait pro">Pure CPU</span>
          <span className="trait pro">Fast (~2s / 10s clip)</span>
          <span className="trait pro">No download</span>
          <span className="trait con">Coarser timbre</span>
        </div>
      </section>

      {/* ── Transform: Vevo ── */}
      <section className="pipeline-section">
        <h3>2b. Vevo timbre backend</h3>
        <p>
          Amphion's Vevo uses a Flow Matching Transformer conditioned on HuBERT
          content-style tokens + a timbre reference to generate mel spectrograms,
          vocoded by Vocos. This produces dramatically more natural timbre transfer —
          but <code>inference_fm</code> replaces <em>both</em> timbre and pitch.
        </p>
        <div className="pipeline-steps">
          <Step n="1" title="Vevo inference_fm" color="var(--accent)">
            Source content tokens + target timbre reference feed the FMT.
            Vocos vocoder converts the predicted mel to audio.
            Result: <em>target timbre + target pitch</em>.
          </Step>
          <Step n="2" title="Praat PSOLA pitch correction" color="var(--accent)">
            Source F0 is extracted via the Hirst two-pass method.
            A Praat Manipulation object replaces the pitch tier with
            the source contour, then overlap-add resynthesises.
            Operates in the <strong>time domain</strong> — formants stay exactly
            where Vevo placed them. No spectral re-synthesis, no double-vocoding.
          </Step>
          <Step n="3" title="Post-processing" color="var(--accent)">
            Timbre strength blend, RMS energy envelope transfer,
            WORLD-based aperiodicity blend for breathiness.
          </Step>
        </div>
        <div className="pipeline-traits">
          <span className="trait pro">Neural timbre</span>
          <span className="trait pro">Formant-safe pitch fix</span>
          <span className="trait con">Needs GPU</span>
          <span className="trait con">Model download (~2 GB)</span>
        </div>
      </section>

      {/* ── Pitch slider ── */}
      <section className="pipeline-section">
        <h3>Pitch slider</h3>
        <table className="pipeline-table">
          <thead><tr><th>Setting</th><th>Behaviour</th></tr></thead>
          <tbody>
            <tr><td>Disabled (default)</td><td>100% source pitch — pure timbre transfer</td></tr>
            <tr><td>50%</td><td>Log-interpolated F0 halfway between source and target</td></tr>
            <tr><td>100%</td><td>Full target pitch (raw inference_fm behaviour)</td></tr>
          </tbody>
        </table>
      </section>

      {/* ── Compare ── */}
      <section className="pipeline-section">
        <h3>3. Compare</h3>
        <p>
          Fresh profiles are extracted from source, output, and target. The engine computes:
        </p>
        <div className="pipeline-detail-grid">
          <div><strong>Per-parameter scores</strong> — 0–100% similarity to target on each enabled dimension</div>
          <div><strong>Acoustic deltas</strong> — raw numeric diffs (source→output, output↔target)</div>
          <div><strong>Perceptual deltas</strong> — same structure for brightness, breathiness, etc.</div>
          <div><strong>Leakage detection</strong> — warns if a disabled parameter drifted &gt;10% from source</div>
        </div>
      </section>

      {/* ── Visualisations ── */}
      <section className="pipeline-section">
        <h3>Visualisations</h3>
        <div className="pipeline-detail-grid">
          <div><strong>Profile card</strong> — acoustic stats + perceptual dimension bars</div>
          <div><strong>F0 contour</strong> — click-to-seek SVG; overlay in compare view</div>
          <div><strong>Mel spectrogram</strong> — click-to-seek PNG, edge-to-edge for accurate seeking</div>
        </div>
      </section>

      {/* ── Export ── */}
      <section className="pipeline-section">
        <h3>Export</h3>
        <p>
          The compare view's <strong>Download report (zip)</strong> bundles the three
          audio files, a self-contained <code>report.html</code> with inline styling
          and local audio players, and a <code>comparison.json</code> for programmatic use.
        </p>
      </section>
    </div>
  );
}

function Step({ n, title, color, children }: { n: string; title: string; color: string; children: React.ReactNode }) {
  return (
    <div className="pipeline-step">
      <div className="pipeline-step-badge" style={{ background: color }}>{n}</div>
      <div className="pipeline-step-body">
        <div className="pipeline-step-title">{title}</div>
        <div className="pipeline-step-text">{children}</div>
      </div>
    </div>
  );
}
