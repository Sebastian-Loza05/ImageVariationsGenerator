"""
variations.py — Genera EXACTAMENTE 5 variaciones visuales a partir de un codigo w+.

Este es el corazon del "nivel avanzado". El enunciado pide exactamente 5
variaciones de la imagen del usuario. Las producimos editando el codigo latente
w+ en direcciones SEMANTICAS interpretables, lo que ademas da material directo
para la seccion de analisis y para que el agente LLM explique cada decision.

Tres mecanismos combinados:
  1. Direcciones semanticas (InterFaceGAN / GANSpace): edad, pose, sonrisa, genero,
     iluminacion. Editar w+ a lo largo de estos vectores cambia UN atributo de forma
     controlada. Esto es lo que hace que las variaciones sean *interpretables*.
  2. Style-mixing: mezclar w+ del usuario con un w aleatorio en capas finas (textura,
     color) para una variacion mas "creativa" pero reconocible.
  3. Truncation: acerca el codigo a la media para una version mas "promedio/idealizada".

Las 5 variaciones por defecto (justificadas y diversas entre si):
  V1: +edad        (envejecimiento)
  V2: +sonrisa     (expresion)
  V3: cambio de pose
  V4: style-mix en capas finas (estilo/iluminacion)
  V5: truncation hacia la media

Las direcciones semanticas (.npy de forma [1,512] o [18,512]) se obtienen con
InterFaceGAN o GANSpace sobre nuestro generador. Si no estan disponibles,
caemos en variaciones por ruido/truncation para que el sistema NUNCA falle.

Librerias:
    pip install torch numpy pillow
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class Variation:
    """Una variacion: imagen + descripcion legible (la usa el agente LLM)."""
    index: int
    label: str          # ej. "Mayor edad"
    description: str     # ej. "Se desplazo w+ en la direccion de edad (+3.0)"
    image: Image.Image


class VariationGenerator:
    """Produce exactamente 5 variaciones desde un w+ usando el generador StyleGAN2."""

    DEFAULT_PLAN = [
        ("age",    "Mayor edad",        3.0),
        ("smile",  "Mas sonriente",     2.5),
        ("pose",   "Cambio de pose",    2.0),
        ("stylemix", "Estilo/iluminacion alternativa", None),
        ("truncation", "Version idealizada (hacia la media)", 0.5),
    ]

    def __init__(self, generator, latent_avg, directions_dir=None, device="cuda"):
        """
        generator: decoder rosinality del e4e (net.decoder), ya en .eval().
                   Es NUESTRO generador fine-tuneado -> consistente con la inversion.
        latent_avg: media latente del checkpoint e4e (net.latent_avg), [n,512] o [512].
        directions_dir: carpeta con age.npy, smile.npy... (InterFaceGAN/GANSpace).
        """
        self.G = generator
        self.device = device
        self.directions = self._load_directions(directions_dir)
        la = latent_avg.detach().to(device).float()
        self.w_avg = la if la.dim() == 2 else la.unsqueeze(0)  # [n,512] o [1,512]

    def _load_directions(self, directions_dir):
        dirs = {}
        if directions_dir and Path(directions_dir).exists():
            for name in ("age", "smile", "pose", "gender", "light"):
                f = Path(directions_dir) / f"{name}.npy"
                if f.exists():
                    v = torch.from_numpy(np.load(f)).float().to(self.device)
                    dirs[name] = v
        return dirs

    @torch.no_grad()
    def _synthesize(self, w_plus):
        """w+ [1,n,512] -> imagen PIL (decoder rosinality del e4e)."""
        img, _ = self.G([w_plus], input_is_latent=True, randomize_noise=False)
        img = ((img[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0)
        return Image.fromarray(img.to(torch.uint8).cpu().numpy())

    def _edit_direction(self, w_plus, name, strength):
        """Suma strength * direccion al codigo w+. Fallback a ruido si no existe."""
        w = w_plus.clone()
        if name in self.directions:
            d = self.directions[name].float().to(self.device)
            d = d.view(1, 1, -1) if d.numel() == 512 else d.view(1, -1, 512)
            w = w + strength * d
        else:
            # Fallback determinista: pequeño desplazamiento aleatorio fijo por nombre.
            g = torch.Generator(device=self.device).manual_seed(hash(name) % 2**31)
            noise = torch.randn(w.shape, generator=g, device=self.device)
            w = w + 0.15 * strength * noise
        return w

    def _style_mix(self, w_plus, seed=42):
        """Mezcla las capas finas (8.. = textura/color) con un estilo aleatorio."""
        w = w_plus.clone()
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, 512)
        ).float().to(self.device)
        w_rand = self.G.get_latent(z)                            # [1,512]
        w_rand = w_rand.unsqueeze(1).expand(-1, w.shape[1], -1)  # [1,n,512]
        w[:, 8:, :] = w_rand[:, 8:, :]                           # solo capas finas
        return w

    def _truncate(self, w_plus, psi):
        """Acerca w+ a la media: version mas 'promedio/idealizada'."""
        la = self.w_avg.unsqueeze(0)   # [1,n,512] o [1,1,512] (broadcast)
        return la + psi * (w_plus - la)

    @torch.no_grad()
    def generate(self, w_plus):
        """Devuelve EXACTAMENTE 5 Variation."""
        variations = []
        for i, (kind, label, strength) in enumerate(self.DEFAULT_PLAN, start=1):
            if kind == "stylemix":
                w_edit = self._style_mix(w_plus)
                desc = "Style-mixing en capas finas (textura, color, iluminacion)."
            elif kind == "truncation":
                w_edit = self._truncate(w_plus, strength)
                desc = f"Truncation hacia la media (psi={strength})."
            else:
                w_edit = self._edit_direction(w_plus, kind, strength)
                used = "direccion semantica" if kind in self.directions else "fallback por ruido"
                desc = f"Edicion de '{kind}' (fuerza {strength}, {used})."
            img = self._synthesize(w_edit)
            variations.append(Variation(i, label, desc, img))

        assert len(variations) == 5, "El enunciado exige exactamente 5 variaciones."
        return variations


if __name__ == "__main__":
    print("Modulo de variaciones. Se usa desde app.py con un generador cargado.")
    print("Plan de variaciones por defecto:")
    for i, (k, l, s) in enumerate(VariationGenerator.DEFAULT_PLAN, 1):
        print(f"  V{i}: {l}  ({k}, fuerza={s})")
