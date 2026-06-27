# Backend — API de variaciones de rostro

API FastAPI sencilla que expone los modelos fine-tuneados (e4e + generador).

## Requisitos

Usa el mismo entorno del proyecto (`../.venv`, con torch, dlib, etc.) y añade lo del servidor:

```bash
cd backend
../.venv/bin/pip install -r requirements.txt
```

## Correr en local

```bash
cd backend
../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

> Un solo worker: hay una sola GPU y las peticiones se atienden de a una.
> Documentación interactiva en http://127.0.0.1:8000/docs

## Endpoints

### `POST /generate`
Recibe una imagen (multipart, campo `file`) y devuelve 5 variaciones.

```bash
curl -F "file=@mi_foto.jpg" http://127.0.0.1:8000/generate
```

Respuesta (resumen):
```json
{
  "session_id": "ab12cd34",
  "input_image": "/files/ab12cd34/input.png",
  "reconstruction_b64": "<png base64>",
  "variations": [
    {"id":"V1","label":"Mayor edad","description":"...",
     "image_path":"/files/ab12cd34/var_1.png","image_b64":"<png base64>"},
    ... (V2..V5)
  ]
}
```
Cada variación viene como `image_b64` (para mostrar directo) y como `image_path`
(URL servible: abre `http://127.0.0.1:8000/files/ab12cd34/var_1.png`).
Guarda el `session_id` para el siguiente paso.

### `POST /feedback`
Recibe las decisiones del usuario y genera el JSON de auditoría.

Cada decisión admite **`accepted`** (booleano → score 1.0/0.0) **o**
**`authenticity_score`** (0.0–1.0 directo):

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "ab12cd34",
    "decisions": [
      {"id":"V1","accepted":true},
      {"id":"V2","accepted":false},
      {"id":"V3","authenticity_score":0.74},
      {"id":"V4","accepted":false},
      {"id":"V5","authenticity_score":0.68}
    ]
  }'
```

Respuesta (y se guarda en `storage/<session_id>/audit.json`):
```json
{
  "session_id": "ab12cd34",
  "input_image": "/files/ab12cd34/input.png",
  "variations": [
    {"id":"V1","image_path":"/files/ab12cd34/var_1.png","authenticity_score":1.0},
    {"id":"V2","image_path":"/files/ab12cd34/var_2.png","authenticity_score":0.0},
    {"id":"V3","image_path":"/files/ab12cd34/var_3.png","authenticity_score":0.74},
    {"id":"V4","image_path":"/files/ab12cd34/var_4.png","authenticity_score":0.0},
    {"id":"V5","image_path":"/files/ab12cd34/var_5.png","authenticity_score":0.68}
  ]
}
```

## Estructura

```
backend/
├── main.py            # app FastAPI + endpoints
├── config.py          # rutas a modelos y storage
├── schemas.py         # validación de datos (pydantic)
├── requirements.txt
└── storage/<session>/ # input, variaciones y audit.json (se crea al usar)
```
