"""
04_eval_metrics.py — Metricas para la seccion "Evaluacion y analisis" (3 pts) y
"Modelo generativo" (4 pts) de la rubrica.

Tres ejes que pide el enunciado:
  1. CALIDAD visual de las imagenes generadas      -> FID
  2. DIVERSIDAD de las 5 variaciones                -> LPIPS promedio entre pares
  3. COHERENCIA decisiones-resumen                  -> se valida en explainer (fidelidad
                                                       garantizada por construccion del prompt)

FID se calcula con el toolkit del propio stylegan2-ada-pytorch (fid50k_full) durante
el entrenamiento. Aqui medimos la DIVERSIDAD de las variaciones, que es la metrica
mas relevante para "5 variaciones de UNA imagen".

Librerias:
    pip install torch lpips numpy pillow
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _load_tensor(path, device):
    img = Image.open(path).convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def diversity_lpips(image_paths, device="cuda"):
    """LPIPS promedio entre todos los pares de variaciones.
    Mas alto = mas diversas. Reporta tambien min y max para el analisis."""
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    tensors = [_load_tensor(p, device) for p in image_paths]

    dists = []
    for a, b in itertools.combinations(range(len(tensors)), 2):
        with torch.no_grad():
            d = loss_fn(tensors[a], tensors[b]).item()
        dists.append(((a + 1, b + 1), d))

    vals = [d for _, d in dists]
    return {
        "pares": dists,
        "promedio": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def identity_preservation(input_path, variation_paths, device="cuda"):
    """Que tanto se conserva la identidad respecto a la imagen de entrada.
    Util para discutir el trade-off diversidad vs. fidelidad en el analisis.
    Usa similitud coseno de embeddings ArcFace si esta disponible."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return {"nota": "insightface no instalado; se omite preservacion de identidad."}

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if device == "cuda" else -1)

    def embed(path):
        img = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB->BGR
        faces = app.get(img)
        return faces[0].embedding if faces else None

    e0 = embed(input_path)
    if e0 is None:
        return {"nota": "no se detecto rostro en la imagen de entrada."}

    sims = {}
    for i, p in enumerate(variation_paths, 1):
        ev = embed(p)
        if ev is None:
            sims[f"V{i}"] = None
        else:
            cos = float(np.dot(e0, ev) / (np.linalg.norm(e0) * np.linalg.norm(ev)))
            sims[f"V{i}"] = round(cos, 3)
    return sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="imagen de entrada (para identidad)")
    ap.add_argument("--variations", nargs="+", required=True,
                    help="rutas de las 5 variaciones generadas")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"

    print("== Diversidad (LPIPS entre pares) ==")
    div = diversity_lpips(args.variations, device)
    print(f"  Promedio: {div['promedio']:.4f}  (min {div['min']:.4f}, max {div['max']:.4f})")
    for (a, b), d in div["pares"]:
        print(f"  V{a}-V{b}: {d:.4f}")

    if args.input:
        print("\n== Preservacion de identidad (coseno ArcFace, entrada vs variacion) ==")
        ident = identity_preservation(args.input, args.variations, device)
        for k, v in ident.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
