const BASE = "";

export async function uploadAudio(file: File): Promise<{ id: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function extractProfile(fileId: string, name = ""): Promise<import("./types").VoiceProfile> {
  const res = await fetch(`${BASE}/api/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, name }),
  });
  if (!res.ok) throw new Error(`Extract failed: ${res.statusText}`);
  return res.json();
}

export async function runTransform(req: {
  source_id: string;
  target_id: string;
  config: import("./types").TransformConfig;
  backend: import("./types").Backend;
}): Promise<{
  output_id: string;
  profile: import("./types").VoiceProfile;
  source_profile: import("./types").VoiceProfile;
  target_profile: import("./types").VoiceProfile;
  comparison: import("./types").ComparisonResult;
}> {
  const res = await fetch(`${BASE}/api/transform`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Transform failed: ${msg}`);
  }
  return res.json();
}

export async function runCompare(req: {
  source_id: string;
  output_id: string;
  target_id: string;
  config: import("./types").TransformConfig;
}): Promise<import("./types").ComparisonResult> {
  const res = await fetch(`${BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Compare failed: ${res.statusText}`);
  return res.json();
}

export function audioUrl(fileId: string): string {
  return `${BASE}/api/audio/${fileId}`;
}

export function melSpectrogramUrl(fileId: string): string {
  return `${BASE}/api/mel-spectrogram/${fileId}`;
}

export interface F0ContourData {
  times: number[];
  hz: (number | null)[];
  median_hz: number | null;
  voiced_ratio: number;
}

export async function fetchF0Contour(fileId: string): Promise<F0ContourData> {
  const res = await fetch(`${BASE}/api/f0-contour/${fileId}`);
  if (!res.ok) throw new Error(`F0 contour fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAudioInfo(fileId: string): Promise<{ duration_s: number }> {
  const res = await fetch(`${BASE}/api/audio-info/${fileId}`);
  if (!res.ok) throw new Error(`Audio info fetch failed: ${res.statusText}`);
  return res.json();
}

export async function exportZip(req: {
  source_id: string;
  output_id: string;
  target_id: string;
  config: import("./types").TransformConfig;
  backend: import("./types").Backend | "";
}): Promise<{ blob: Blob; filename: string }> {
  const res = await fetch(`${BASE}/api/export-zip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  // Parse filename from Content-Disposition; fall back to a sensible default
  const disp = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^"]+)"?/i.exec(disp);
  const filename = match?.[1] ?? "figura-report.zip";
  return { blob: await res.blob(), filename };
}

export interface Provenance {
  source_id: string;
  target_id: string;
  backend: import("./types").Backend;
  config: import("./types").TransformConfig;
  created_at: string;
}

export interface LibraryEntry {
  id: string;
  filename: string;
  display_name: string;
  extension: string;
  gender: import("./types").Gender;
  notes: string;
  tags: string[];
  kind: "upload" | "output";
  provenance: Provenance | null;
  path: string;
}

export async function listLibrary(): Promise<LibraryEntry[]> {
  const res = await fetch(`${BASE}/api/library`);
  if (!res.ok) throw new Error(`Library fetch failed: ${res.statusText}`);
  return res.json();
}

export async function deleteFromLibrary(fileId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/library/${fileId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
}

export async function updateLibraryEntry(
  fileId: string,
  patch: Partial<{ display_name: string; gender: import("./types").Gender; notes: string; tags: string[] }>,
): Promise<LibraryEntry> {
  const res = await fetch(`${BASE}/api/library/${fileId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Update failed: ${res.statusText}`);
  return res.json();
}
