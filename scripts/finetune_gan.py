"""
02_finetune_gan.py — Fine-tune de StyleGAN2-ADA sobre el dataset de rostros.

Estrategia: TRANSFER LEARNING desde el FFHQ pre-entrenado (NO desde cero).
Esto es lo que hace viable entrenar en una RTX 4070 8GB en una sola noche:
partiendo de FFHQ, en ~10h de una sola GPU se obtiene un FID mejor que entrenar
días desde cero.

Este script es un WRAPPER que arma el comando train.py del repo oficial
stylegan2-ada-pytorch con los flags correctos para 8GB de VRAM.

Requisitos previos:
  git clone https://github.com/NVlabs/stylegan2-ada-pytorch
  # checkpoint FFHQ 256:
  #   ffhq256 -> usa ffhqu/ffhq de NVIDIA y deja que --resume baje el .pkl

Decisiones para 8GB @ 256x256:
  --gpus 1          una sola GPU
  --batch 4         batch total. En el repo oficial de NVlabs el micro-batch
                    por GPU es batch_gpu = batch // gpus; con 1 GPU, batch=4
                    => 4 imagenes por GPU (lo que entra en 8GB a 256).
                    OJO: el flag --batch-gpu NO existe en NVlabs, solo en el
                    fork de dvschultz (que permite gradient accumulation).
  --cfg paper256    config optimizada para 256x256
  --aug ada         ADA: augmentation adaptativa (clave con datasets chicos)
  --mirror 1        espejo horizontal: duplica datos efectivos
  --snap 10         snapshot cada 10 ticks (para no perder la noche si crashea)
  --resume ffhq256  transfer learning desde FFHQ

Uso:
    python scripts/02_finetune_gan.py \
        --repo   ../stylegan2-ada-pytorch \
        --data   data/faces_256.zip \
        --outdir runs/gan
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ruta a stylegan2-ada-pytorch")
    ap.add_argument("--data", required=True, help="dataset .zip (256x256)")
    ap.add_argument("--outdir", default="runs/gan")
    ap.add_argument("--resume", default="ffhq256",
                    help="'ffhq256' usa el pkl oficial; o ruta a un .pkl propio")
    ap.add_argument("--kimg", type=int, default=600,
                    help="miles de imagenes a procesar. ~600 entra en una noche.")
    # En NVlabs/stylegan2-ada-pytorch el micro-batch por GPU es batch // gpus.
    # Con 1 GPU, batch=4 => 4 img/GPU, que es lo seguro para 8GB a 256.
    # (No hay --batch-gpu en este repo; eso es del fork de dvschultz.)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true",
                    help="solo imprime el comando, no entrena")
    args = ap.parse_args()

    train_py = Path(args.repo) / "train.py"
    if not train_py.exists():
        sys.exit(f"No encuentro {train_py}. Clona el repo NVlabs/stylegan2-ada-pytorch.")

    cmd = [
        sys.executable, str(train_py),
        "--outdir", args.outdir,
        "--data", args.data,
        "--gpus", "1",
        "--cfg", "paper256",
        "--batch", str(args.batch),
        "--aug", "ada",
        "--mirror", "1",
        "--snap", "10",
        "--kimg", str(args.kimg),
        "--resume", args.resume,
        # FID cada snapshot para seguir la calidad durante la noche:
        "--metrics", "fid50k_full",
    ]

    print("Comando de entrenamiento:\n")
    print("  " + " \\\n  ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] No se ejecuto el entrenamiento.")
        return

    print("Iniciando fine-tune. Revisa runs/gan/*/fakes*.png para ver el progreso.")
    print("Si la VRAM se desborda, baja --batch a 2.\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


