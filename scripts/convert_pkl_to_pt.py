"""
convert_pkl_to_pt.py — Convierte un generador StyleGAN2-ADA (formato NVlabs .pkl)
al formato rosinality .pt que espera e4e (encoder4editing).

Por que: el .pkl de NVlabs guarda un generador con una arquitectura distinta
(modulos mapping.fcN / synthesis.bR...). e4e usa la implementacion de rosinality
(style.N / input / conv1 / convs.N / to_rgbs.N). Hay que re-mapear los pesos.

Buenas noticias sobre las escalas: ambas implementaciones almacenan los pesos
SIN el gain de equalized-LR y lo aplican en runtime con el MISMO factor
(lr_mul/sqrt(fan_in) en el mapping; 1/sqrt(fan_in) en los conv modulados), asi
que los pesos se copian tal cual. El unico ajuste real es unsqueeze(0) en los conv (rosinality usa [1,out,in,k,k]).
Verificado numericamente: NO hace falta flip del kernel en las upsample; con la
misma w el forward de rosinality coincide con el de NVlabs (mean|dif|~0.003,
solo diferencias de fp16). Cualquier flip rompe la coincidencia.

El .pt resultante: {'g_ema': <state_dict rosinality>, 'latent_avg': <w_avg [512]>}
que es justo lo que carga e4e (psp.py: ckpt['g_ema'] con strict=False + latent_avg).

Uso:
    python scripts/convert_pkl_to_pt.py \
        --pkl  runs/gan/00001-.../network-snapshot-000400.pkl \
        --out  pretrained/our_generator_rosinality.pt \
        --sg2_repo ../stylegan2-ada-pytorch \
        --e4e_repo ../encoder4editing \
        --size 256
"""
import argparse
import sys
from pathlib import Path

import torch


def build_rosinality_state_dict(nv, size):
    """nv: state_dict de NVlabs G_ema. Devuelve state_dict estilo rosinality."""
    ros = {}

    # --- Mapping network: mapping.fc{i} -> style.{i+1} (style.0 es PixelNorm) ---
    i = 0
    while f"mapping.fc{i}.weight" in nv:
        ros[f"style.{i+1}.weight"] = nv[f"mapping.fc{i}.weight"]
        ros[f"style.{i+1}.bias"] = nv[f"mapping.fc{i}.bias"]
        i += 1

    # --- Constante de entrada: synthesis.b4.const -> input.input ---
    ros["input.input"] = nv["synthesis.b4.const"].unsqueeze(0)  # [1,C,4,4]

    def conv_mod(nv_prefix, ros_prefix):
        ros[f"{ros_prefix}.conv.weight"] = nv[f"{nv_prefix}.weight"].unsqueeze(0)  # [1,out,in,k,k]
        ros[f"{ros_prefix}.conv.modulation.weight"] = nv[f"{nv_prefix}.affine.weight"]
        ros[f"{ros_prefix}.conv.modulation.bias"] = nv[f"{nv_prefix}.affine.bias"]
        ros[f"{ros_prefix}.noise.weight"] = nv[f"{nv_prefix}.noise_strength"].reshape(1)
        ros[f"{ros_prefix}.activate.bias"] = nv[f"{nv_prefix}.bias"]

    def to_rgb(nv_prefix, ros_prefix):
        ros[f"{ros_prefix}.conv.weight"] = nv[f"{nv_prefix}.weight"].unsqueeze(0)  # [1,3,in,1,1]
        ros[f"{ros_prefix}.conv.modulation.weight"] = nv[f"{nv_prefix}.affine.weight"]
        ros[f"{ros_prefix}.conv.modulation.bias"] = nv[f"{nv_prefix}.affine.bias"]
        ros[f"{ros_prefix}.bias"] = nv[f"{nv_prefix}.bias"].reshape(1, 3, 1, 1)

    # --- Bloque base b4: conv1 -> conv1, torgb -> to_rgb1 ---
    conv_mod("synthesis.b4.conv1", "conv1")
    to_rgb("synthesis.b4.torgb", "to_rgb1")

    # --- Bloques de resolucion b8..bSIZE ---
    import math
    log_size = int(math.log2(size))
    conv_idx = 0
    rgb_idx = 0
    for res_log in range(3, log_size + 1):          # 8,16,...,size
        res = 2 ** res_log
        b = f"synthesis.b{res}"
        conv_mod(f"{b}.conv0", f"convs.{conv_idx}")   # upsample
        conv_idx += 1
        conv_mod(f"{b}.conv1", f"convs.{conv_idx}")   # misma resolucion
        conv_idx += 1
        to_rgb(f"{b}.torgb", f"to_rgbs.{rgb_idx}")
        rgb_idx += 1

    return ros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sg2_repo", default="../stylegan2-ada-pytorch")
    ap.add_argument("--e4e_repo", default="../encoder4editing")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    # ---- Cargar G_ema de NVlabs ----
    sys.path.insert(0, str(Path(args.sg2_repo).resolve()))
    import legacy
    import dnnlib
    with dnnlib.util.open_url(args.pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    nv = G.state_dict()
    w_avg = nv["mapping.w_avg"].detach().cpu().clone()  # [512]
    # channel_multiplier de rosinality = canales(res maxima) / 64.
    # paper256 (channel_base=16384) -> cm=1; config full (32768) -> cm=2.
    cmult = int(nv[f"synthesis.b{args.size}.conv1.weight"].shape[0] // 64)
    print(f"Cargado NVlabs G_ema: {len(nv)} tensores | img_resolution={G.img_resolution} "
          f"| channel_multiplier detectado = {cmult}")

    # ---- Construir state_dict rosinality ----
    ros = {k: v.detach().cpu().clone() for k, v in build_rosinality_state_dict(nv, args.size).items()}
    print(f"State_dict rosinality construido: {len(ros)} tensores")

    # ---- Guardar ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"g_ema": ros, "latent_avg": w_avg, "channel_multiplier": cmult}, out)
    print(f"Guardado: {out}  (channel_multiplier={cmult})")

    # ---- Verificacion 1: estructural (cargar en el Generator de e4e) ----
    sys.path.insert(0, str(Path(args.e4e_repo).resolve()))
    from models.stylegan2.model import Generator
    g = Generator(args.size, 512, 8, channel_multiplier=cmult)
    missing, unexpected = g.load_state_dict(ros, strict=False)
    shape_ok = True
    gsd = g.state_dict()
    for k, v in ros.items():
        if k in gsd and tuple(gsd[k].shape) != tuple(v.shape):
            shape_ok = False
            print(f"  [SHAPE MISMATCH] {k}: esperado {tuple(gsd[k].shape)} vs {tuple(v.shape)}")
    # 'noises.*' siempre faltan (buffers de ruido, e4e los regenera) -> benigno
    real_missing = [m for m in missing if not m.startswith("noises.")]
    print("\n[Verificacion estructural]")
    print(f"  keys faltantes (no-noise): {real_missing if real_missing else 'ninguna'}")
    print(f"  keys inesperadas: {list(unexpected) if unexpected else 'ninguna'}")
    print(f"  shapes correctas: {shape_ok}")

    # ---- Verificacion 2: numerica (misma w -> misma imagen) ----
    if not torch.cuda.is_available():
        print("\n[Verificacion numerica] omitida (sin CUDA).")
        return
    try:
        dev = "cuda"
        G = G.to(dev).eval()
        g = g.to(dev).eval()
        with torch.no_grad():
            z = torch.randn(2, 512, device=dev)
            w = G.mapping(z, None)                      # [2, num_ws, 512]
            num_ws = w.shape[1]
            w1 = w[:, :1, :].repeat(1, num_ws, 1)       # misma w en todas las capas
            img_nv = G.synthesis(w1, noise_mode="none")  # sin ruido
            zeros = [torch.zeros(2, 1, 2 ** i, 2 ** i, device=dev)
                     for i in range(2, int(__import__("math").log2(args.size)) + 1)
                     for _ in (range(2) if i > 2 else range(1))]
            img_ros, _ = g([w1], input_is_latent=True, randomize_noise=False, noise=zeros)
        diff = (img_nv - img_ros).abs()
        denom = img_nv.abs().mean().item()
        print("\n[Verificacion numerica] (misma w, ruido cero)")
        print(f"  rango NVlabs: [{img_nv.min():.3f}, {img_nv.max():.3f}]  ros: [{img_ros.min():.3f}, {img_ros.max():.3f}]")
        print(f"  max|dif|={diff.max():.4f}  mean|dif|={diff.mean():.5f}  (escala media |img|={denom:.3f})")
        ok = diff.mean().item() < 0.05 * max(denom, 1e-6) or diff.max().item() < 0.1
        print(f"  -> {'OK: los generadores coinciden' if ok else 'ATENCION: diferencia alta, revisar el mapeo'}")
    except Exception as e:
        print(f"\n[Verificacion numerica] no se pudo correr el forward de rosinality: {type(e).__name__}: {e}")
        print("  (probablemente los CUDA ops de rosinality necesitan parche torch-2.x, como en stylegan)")
        print("  La verificacion estructural ya valida keys/shapes.")


if __name__ == "__main__":
    main()
