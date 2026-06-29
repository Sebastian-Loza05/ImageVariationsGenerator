import argparse
import os
from pathlib import Path

import numpy as np
import PIL.Image
from tqdm import tqdm


def get_landmarks(img_path, detector, predictor):
    import dlib
    try:
        img = dlib.load_rgb_image(str(img_path))
    except RuntimeError:
        return None
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    det = max(dets, key=lambda d: d.width() * d.height())
    shape = predictor(img, det)
    return np.array([[p.x, p.y] for p in shape.parts()])


def align_face(img_path, landmarks, output_size=256, transform_size=1024):
    lm = landmarks
    lm_eye_left, lm_eye_right = lm[36:42], lm[42:48]
    lm_mouth_outer = lm[48:60]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Vectores del cuadro de recorte orientado.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(img_path).convert("RGB")

    # Reduce en pasos para evitar aliasing.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(img.size[0] / shrink)),
                 int(np.rint(img.size[1] / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--predictor", required=True,
                    help="shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    try:
        import dlib
    except ImportError:
        raise SystemExit("Falta dlib")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in Path(args.raw_dir).rglob("*") if p.suffix.lower() in exts]
    print(f"Encontradas {len(imgs)} imagenes en {args.raw_dir}")

    kept, skipped = 0, 0
    for p in tqdm(imgs, desc="Alineando rostros"):
        lm = get_landmarks(p, detector, predictor)
        if lm is None:
            skipped += 1
            continue
        try:
            out = align_face(p, lm, output_size=args.size)
            out.save(Path(args.out_dir) / f"{kept:06d}.png")
            kept += 1
        except Exception:
            skipped += 1

    print(f"\nListas: {kept}  |  Descartadas (sin rostro detectable): {skipped}")


if __name__ == "__main__":
    main()
