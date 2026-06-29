import argparse
import sys
from pathlib import Path

import torch


def build_rosinality_state_dict(nv, size):
    ros = {}

    i = 0
    while f"mapping.fc{i}.weight" in nv:
        ros[f"style.{i+1}.weight"] = nv[f"mapping.fc{i}.weight"]
        ros[f"style.{i+1}.bias"] = nv[f"mapping.fc{i}.bias"]
        i += 1

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

    conv_mod("synthesis.b4.conv1", "conv1")
    to_rgb("synthesis.b4.torgb", "to_rgb1")

    import math
    log_size = int(math.log2(size))
    conv_idx = 0
    rgb_idx = 0
    for res_log in range(3, log_size + 1):
        res = 2 ** res_log
        b = f"synthesis.b{res}"
        conv_mod(f"{b}.conv0", f"convs.{conv_idx}")
        conv_idx += 1
        conv_mod(f"{b}.conv1", f"convs.{conv_idx}")
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

    sys.path.insert(0, str(Path(args.sg2_repo).resolve()))
    import legacy
    import dnnlib
    with dnnlib.util.open_url(args.pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    nv = G.state_dict()
    w_avg = nv["mapping.w_avg"].detach().cpu().clone()  # [512]
    cmult = int(nv[f"synthesis.b{args.size}.conv1.weight"].shape[0] // 64)
    print(f"Cargado NVlabs G_ema: {len(nv)} tensores | img_resolution={G.img_resolution} "
          f"| channel_multiplier detectado = {cmult}")

    ros = {k: v.detach().cpu().clone() for k, v in build_rosinality_state_dict(nv, args.size).items()}
    print(f"State_dict rosinality construido: {len(ros)} tensores")

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
    real_missing = [m for m in missing if not m.startswith("noises.")]
    print("\n[Verificacion estructural]")
    print(f"  keys faltantes (no-noise): {real_missing if real_missing else 'ninguna'}")
    print(f"  keys inesperadas: {list(unexpected) if unexpected else 'ninguna'}")
    print(f"  shapes correctas: {shape_ok}")

    if not torch.cuda.is_available():
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
        print(f"  rango NVlabs: [{img_nv.min():.3f}, {img_nv.max():.3f}]  ros: [{img_ros.min():.3f}, {img_ros.max():.3f}]")
        print(f"  max|dif|={diff.max():.4f}  mean|dif|={diff.mean():.5f}  (escala media |img|={denom:.3f})")
        ok = diff.mean().item() < 0.05 * max(denom, 1e-6) or diff.max().item() < 0.1
    except Exception as e:
        print(f"\n[Verificacion numerica] no se pudo correr el forward de rosinality: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
