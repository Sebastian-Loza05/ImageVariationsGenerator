"""
align.py — Alineado FFHQ de un rostro sobre un PIL.Image.

CRITICO para inferencia: el e4e (y el generador) fueron entrenados SOLO con
rostros alineados estilo FFHQ (mismo recorte canonico que produjo
scripts/01_prepare_data.py). Si subes una foto sin alinear, la inversion sale
mal. Aqui replicamos exactamente ese alineado para la imagen del usuario.

Requiere dlib + el predictor de 68 landmarks (pretrained/shape_predictor...dat).
"""
import numpy as np
import PIL.Image


class FaceAligner:
    """Carga detector + predictor una vez y alinea PIL.Images al estilo FFHQ."""

    def __init__(self, predictor_path):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))

    def _landmarks(self, np_rgb):
        dets = self.detector(np_rgb, 1)
        if len(dets) == 0:
            return None
        det = max(dets, key=lambda d: d.width() * d.height())  # rostro mas grande
        shape = self.predictor(np_rgb, det)
        return np.array([[p.x, p.y] for p in shape.parts()])

    def align(self, pil_image, output_size=256, transform_size=1024):
        """PIL -> PIL alineado 256x256, o None si no se detecta rostro."""
        img = pil_image.convert("RGB")
        lm = self._landmarks(np.array(img))
        if lm is None:
            return None

        lm_eye_left, lm_eye_right = lm[36:42], lm[42:48]
        lm_mouth_outer = lm[48:60]
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

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
