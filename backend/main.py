"""
API FastAPI del sistema generativo interactivo de rostros.

Endpoints:
  POST /generate   recibe una imagen -> alinea, invierte (e4e) y devuelve 5 variaciones.
  POST /feedback   recibe aceptar/rechazar por variacion -> genera el JSON de auditoria.
  GET  /files/...  sirve las imagenes guardadas (variaciones, input, etc).

Corre en local con una sola GPU. Los modelos se cargan UNA vez al arrancar y las
peticiones se serializan (lock), porque 8 GB de VRAM no dan para varias a la vez.

Ejecutar:
    cd backend
    uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
"""
import base64
import io
import json
import sys
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config

# Los modelos de inferencia viven en el proyecto (src/) y dependen del repo e4e.
sys.path.insert(0, str(config.E4E_REPO))      # provee models.psp (clase pSp)
sys.path.insert(0, str(config.PROJECT_ROOT))  # provee src.*
from src.align import FaceAligner
from src.inversion import E4EInverter
from src.variations import VariationGenerator

from schemas import FeedbackRequest

app = FastAPI(title="Variaciones de Rostro — API", version="1.0")

config.STORAGE.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(config.STORAGE)), name="files")

_MODELS = {}
_LOCK = threading.Lock()


@app.on_event("startup")
def _load_models():
    _MODELS["aligner"] = FaceAligner(config.PREDICTOR)
    _MODELS["inverter"] = E4EInverter(str(config.E4E_CKPT), device=config.DEVICE)
    inv = _MODELS["inverter"]
    _MODELS["vargen"] = VariationGenerator(
        inv.decoder, inv.latent_avg,
        directions_dir=str(config.DIRECTIONS_DIR), device=config.DEVICE,
    )


def _png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_MODELS), "device": config.DEVICE}


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "El archivo no es una imagen valida.")

    session_id = uuid.uuid4().hex[:8]
    sdir = config.STORAGE / session_id
    sdir.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        aligned = _MODELS["aligner"].align(img)
        if aligned is None:
            raise HTTPException(422, "No se detecto un rostro. Usa una foto mas frontal.")
        w_plus, rec = _MODELS["inverter"].invert(aligned)
        variations = _MODELS["vargen"].generate(w_plus)

    img.save(sdir / "input.png")
    rec.save(sdir / "reconstruction.png")
    var_out = []
    for v in variations:
        fname = f"var_{v.index}.png"
        v.image.save(sdir / fname)
        var_out.append({
            "id": f"V{v.index}",
            "label": v.label,
            "description": v.description,
            "image_path": f"/files/{session_id}/{fname}",
            "image_b64": _png_b64(v.image),
        })

    session = {
        "session_id": session_id,
        "input_image": f"/files/{session_id}/input.png",
        "variations": [{"id": x["id"], "image_path": x["image_path"]} for x in var_out],
    }
    (sdir / "session.json").write_text(json.dumps(session, indent=2))

    return {
        "session_id": session_id,
        "input_image": session["input_image"],
        "reconstruction_b64": _png_b64(rec),
        "variations": var_out,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    sdir = config.STORAGE / req.session_id
    sfile = sdir / "session.json"
    if not sfile.exists():
        raise HTTPException(404, f"session_id desconocido: {req.session_id}")

    session = json.loads(sfile.read_text())
    decisions = {d.id: d for d in req.decisions}

    variations = []
    for v in session["variations"]:
        d = decisions.get(v["id"])
        if d is None:
            raise HTTPException(400, f"Falta la decision para {v['id']}.")
        variations.append({
            "id": v["id"],
            "image_path": v["image_path"],
            "authenticity_score": round(d.score(), 2),
        })

    audit = {
        "session_id": req.session_id,
        "input_image": session["input_image"],
        "variations": variations,
    }
    (sdir / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit
