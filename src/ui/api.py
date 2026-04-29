"""FastAPI backend wrapping the voice-transform pipeline."""
import sys
import json
import uuid
import dataclasses
from pathlib import Path

from dotenv import load_dotenv
import os
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
print(f"[figura] VEVO_FP16={os.environ.get('VEVO_FP16', '<unset>')}")
try:
    import torch
    print(f"[figura] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
except Exception as _e:
    print(f"[figura] torch import failed: {_e}")

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure voice-transform/ is on sys.path so "from src..." imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.extractor import extract, VoiceProfile
from src.analysis.visualization import compute_f0_contour, render_mel_png, audio_duration_s
from src.transform.pipeline import TransformConfig, ParamConfig, run as run_transform
from src.comparison.engine import compare as run_compare

from .schemas import (
    TransformConfigSchema,
    TransformRequest,
    CompareRequest,
    ExportRequest,
    VoiceProfileSchema,
    ComparisonResultSchema,
)
from .report import build_export_zip

app = FastAPI(title="Voice Transform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = _PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
_REGISTRY_PATH = UPLOAD_DIR / "registry.json"
_MEL_CACHE_DIR = _PROJECT_ROOT / "outputs" / "mel_cache"

# ---------- persistent file registry ----------

_files: dict[str, dict] = {}


def _load_registry():
    global _files
    if _REGISTRY_PATH.exists():
        try:
            _files = json.loads(_REGISTRY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            _files = {}


def _save_registry():
    _REGISTRY_PATH.write_text(json.dumps(_files, indent=2))


def _strip_ext(name: str) -> str:
    return Path(name).stem or name


def _register_file(
    file_id: str,
    original_name: str,
    path: str,
    *,
    kind: str = "upload",
    provenance: dict | None = None,
):
    _files[file_id] = {
        "original_name": original_name,
        "path": path,
        "display_name": _strip_ext(original_name),
        "gender": None,
        "notes": "",
        "kind": kind,
    }
    if provenance is not None:
        _files[file_id]["provenance"] = provenance
    _save_registry()


def _apply_metadata(vp: VoiceProfile, file_id: str) -> VoiceProfile:
    """Merge user-editable metadata onto a fresh profile."""
    entry = _files.get(file_id) or {}
    if entry.get("display_name"):
        vp.name = entry["display_name"]
    if entry.get("gender") is not None:
        vp.gender = entry["gender"]
    if entry.get("notes"):
        vp.notes = entry["notes"]
    if entry.get("tags"):
        vp.tags = list(entry["tags"])
    return vp


_load_registry()


# ---------- helpers ----------

def _schema_to_transform_config(s: TransformConfigSchema) -> TransformConfig:
    dumped = s.model_dump()
    return TransformConfig(**{
        name: ParamConfig(enabled=pc["enabled"], strength=pc["strength"])
        for name, pc in dumped.items()
    })


def _profile_to_schema(vp: VoiceProfile) -> VoiceProfileSchema:
    return VoiceProfileSchema(**vp.to_dict())


def _get_file_path(file_id: str) -> Path:
    entry = _files.get(file_id)
    if not entry:
        raise HTTPException(404, f"Unknown file id: {file_id}")
    p = Path(entry["path"])
    if not p.exists():
        raise HTTPException(404, f"File missing on disk: {file_id}")
    return p


# ---------- endpoints ----------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename or "audio.wav").suffix
    file_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    _register_file(file_id, file.filename or "unknown", str(dest))
    return {"id": file_id, "filename": file.filename}


@app.get("/api/audio/{file_id}")
def serve_audio(file_id: str):
    p = _get_file_path(file_id)
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    mt = media_types.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(p, media_type=mt)


@app.get("/api/f0-contour/{file_id}")
def f0_contour(file_id: str):
    p = _get_file_path(file_id)
    try:
        return compute_f0_contour(str(p))
    except Exception as e:
        raise HTTPException(500, f"F0 extraction failed: {e}")


@app.get("/api/mel-spectrogram/{file_id}")
def mel_spectrogram(file_id: str):
    p = _get_file_path(file_id)
    try:
        png_path = render_mel_png(str(p), _MEL_CACHE_DIR)
    except Exception as e:
        raise HTTPException(500, f"Mel render failed: {e}")
    return FileResponse(
        png_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/audio-info/{file_id}")
def audio_info(file_id: str):
    p = _get_file_path(file_id)
    try:
        return {"duration_s": audio_duration_s(str(p))}
    except Exception as e:
        raise HTTPException(500, f"Audio probe failed: {e}")


@app.post("/api/extract")
def extract_profile(body: dict):
    file_id = body.get("file_id")
    name = body.get("name", "")
    p = _get_file_path(file_id)
    vp = extract(str(p), name)
    vp = _apply_metadata(vp, file_id)
    return _profile_to_schema(vp).model_dump()


@app.post("/api/transform")
def transform(req: TransformRequest):
    src_path = _get_file_path(req.source_id)
    tgt_path = _get_file_path(req.target_id)

    config = _schema_to_transform_config(req.config)
    out_id = uuid.uuid4().hex[:12]
    out_path = UPLOAD_DIR / f"{out_id}.wav"

    try:
        run_transform(str(src_path), str(tgt_path), config, str(out_path), req.backend)
    except Exception as e:
        raise HTTPException(500, f"Transform failed: {e}")

    from datetime import datetime, timezone
    provenance = {
        "source_id": req.source_id,
        "target_id": req.target_id,
        "backend": req.backend,
        "config": config.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _register_file(
        out_id, f"output_{out_id}.wav", str(out_path),
        kind="output", provenance=provenance,
    )

    # Extract all three profiles and run comparison in one shot
    src_vp = extract(str(src_path), "source")
    tgt_vp = extract(str(tgt_path), "target")
    out_vp = extract(str(out_path), "output")
    comparison = run_compare(src_vp, out_vp, tgt_vp, config)

    return {
        "output_id": out_id,
        "profile": _profile_to_schema(out_vp).model_dump(),
        "source_profile": _profile_to_schema(src_vp).model_dump(),
        "target_profile": _profile_to_schema(tgt_vp).model_dump(),
        "comparison": ComparisonResultSchema(**dataclasses.asdict(comparison)).model_dump(),
    }


@app.post("/api/compare")
def compare(req: CompareRequest):
    src_path = _get_file_path(req.source_id)
    out_path = _get_file_path(req.output_id)
    tgt_path = _get_file_path(req.target_id)

    config = _schema_to_transform_config(req.config)

    src_vp = extract(str(src_path), "source")
    out_vp = extract(str(out_path), "output")
    tgt_vp = extract(str(tgt_path), "target")

    result = run_compare(src_vp, out_vp, tgt_vp, config)

    return ComparisonResultSchema(**dataclasses.asdict(result)).model_dump()


@app.post("/api/export-zip")
def export_zip(req: ExportRequest):
    """Bundle source/target/output audio + a standalone report.html + comparison.json."""
    src_path = _get_file_path(req.source_id)
    out_path = _get_file_path(req.output_id)
    tgt_path = _get_file_path(req.target_id)

    config = _schema_to_transform_config(req.config)
    src_vp = extract(str(src_path), "source")
    out_vp = extract(str(out_path), "output")
    tgt_vp = extract(str(tgt_path), "target")
    result = run_compare(src_vp, out_vp, tgt_vp, config)

    from datetime import datetime, timezone
    created_at = (
        (_files.get(req.output_id) or {}).get("provenance", {}).get("created_at")
        or datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    backend = req.backend or (_files.get(req.output_id) or {}).get("provenance", {}).get("backend", "")

    zip_bytes = build_export_zip(
        src_path=str(src_path),
        out_path=str(out_path),
        tgt_path=str(tgt_path),
        src_profile=src_vp,
        out_profile=out_vp,
        tgt_profile=tgt_vp,
        comparison=result,
        config=config,
        backend=backend,
        created_at=created_at,
    )

    stamp = created_at.replace(":", "").replace("-", "")[:15]  # safe for filenames
    filename = f"figura-report-{req.output_id}-{stamp}.zip"
    return Response(
        zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------- library endpoints ----------

@app.get("/api/library")
def list_library():
    """List all uploaded files with their IDs and names."""
    entries = []
    for fid, info in _files.items():
        p = Path(info["path"])
        if p.exists():
            entries.append({
                "id": fid,
                "filename": info["original_name"],
                "display_name": info.get("display_name") or _strip_ext(info["original_name"]),
                "extension": Path(info["original_name"]).suffix.lstrip(".").lower(),
                "gender": info.get("gender"),
                "notes": info.get("notes", ""),
                "tags": info.get("tags", []),
                "kind": info.get("kind") or ("output" if info["original_name"].startswith("output_") else "upload"),
                "provenance": info.get("provenance"),
                "path": info["path"],
            })
    return entries


_ALLOWED_GENDERS = {"female", "male", "neutral", None}


@app.patch("/api/library/{file_id}")
def update_library_entry(file_id: str, body: dict):
    """Edit user-owned fields on a library voice."""
    entry = _files.get(file_id)
    if not entry:
        raise HTTPException(404, f"Unknown file id: {file_id}")
    if "display_name" in body:
        name = (body["display_name"] or "").strip()
        if not name:
            raise HTTPException(400, "display_name must be non-empty")
        entry["display_name"] = name
    if "gender" in body:
        g = body["gender"]
        if g not in _ALLOWED_GENDERS:
            raise HTTPException(400, f"gender must be one of {sorted(x for x in _ALLOWED_GENDERS if x)} or null")
        entry["gender"] = g
    if "notes" in body:
        entry["notes"] = body["notes"] or ""
    if "tags" in body:
        tags = body["tags"] or []
        if not isinstance(tags, list):
            raise HTTPException(400, "tags must be a list of strings")
        entry["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    _save_registry()
    return {
        "id": file_id,
        "filename": entry["original_name"],
        "display_name": entry.get("display_name") or entry["original_name"],
        "gender": entry.get("gender"),
        "notes": entry.get("notes", ""),
        "tags": entry.get("tags", []),
    }


@app.delete("/api/library/{file_id}")
def delete_from_library(file_id: str):
    entry = _files.pop(file_id, None)
    if not entry:
        raise HTTPException(404, f"Unknown file id: {file_id}")
    p = Path(entry["path"])
    if p.exists():
        p.unlink()
    _save_registry()
    return {"status": "deleted"}


# Serve frontend in production
_DIST = _PROJECT_ROOT / "web" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
