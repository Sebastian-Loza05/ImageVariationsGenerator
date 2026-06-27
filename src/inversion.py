"""
inversion.py — Inversion de una imagen real al espacio latente W+ usando e4e.

e4e (encoder4editing, SIGGRAPH 2021) incrusta una imagen real en W+ en una
fraccion de segundo, SIN optimizacion por imagen. Esta diseñado especificamente
para que las ediciones latentes posteriores sean estables (a diferencia de pSp,
que reconstruye mejor pero edita peor).

Por eso usamos e4e: nuestro objetivo NO es solo reconstruir, es EDITAR para
producir variaciones coherentes.

Requiere el repo encoder4editing (omertov/encoder4editing) en el PYTHONPATH y un
checkpoint e4e entrenado contra nuestro generador fine-tuneado (ver 03_train_e4e.py).

Librerias:
    pip install torch torchvision pillow numpy
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class E4EInverter:
    """Envuelve el encoder e4e para uso desde la app."""

    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self._load(checkpoint_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load(self, checkpoint_path):
        # Import diferido: depende del repo e4e en PYTHONPATH.
        from models.psp import pSp  # e4e reutiliza la clase pSp
        from argparse import Namespace

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = checkpoint_path
        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.to(self.device)
        self.net = net
        # Exponemos el decoder (NUESTRO generador) y la media latente para las
        # variaciones: asi se generan con el MISMO generador de la inversion.
        self.decoder = net.decoder
        self.latent_avg = net.latent_avg

    @torch.no_grad()
    def invert(self, pil_image):
        """imagen PIL -> (w_plus [1,18,512], imagen reconstruida PIL).

        w_plus es el codigo latente que luego editamos en variations.py.
        """
        x = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        rec, w_plus = self.net(
            x, return_latents=True, randomize_noise=False, resize=False
        )
        return w_plus, self._to_pil(rec[0])

    @staticmethod
    def _to_pil(tensor):
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        arr = (arr * 0.5 + 0.5) * 255
        return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


if __name__ == "__main__":
    # Prueba rapida de humo (requiere GPU + repo + checkpoint).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    inv = E4EInverter(args.ckpt)
    w, rec = inv.invert(Image.open(args.image))
    print("w+ shape:", tuple(w.shape))
    rec.save("reconstruccion.png")
    print("Guardado: reconstruccion.png")
