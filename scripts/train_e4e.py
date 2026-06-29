import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ruta a encoder4editing")
    ap.add_argument("--stylegan_pt", default="pretrained/our_generator_rosinality.pt",
                    help=".pt del generador fine-tuneado (formato rosinality)")
    ap.add_argument("--outdir", default="runs/e4e",
                    help="exp_dir. En corrida nueva DEBE no existir (e4e se niega a sobreescribir)")
    ap.add_argument("--max_steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=2,
                    help="8GB: 2 es lo seguro. Con 4 la VRAM se desborda al avanzar el progressive.")
    ap.add_argument("--save_interval", type=int, default=2000,
                    help="cada cuantos pasos guardar un checkpoint REANUDABLE")
    ap.add_argument("--resume_from", default=None,
                    help="ruta a iteration_N.pt para CONTINUAR exacto (paso/optimizador/discriminador)")
    ap.add_argument("--init_from", default=None,
                    help="warm-start: carga pesos encoder/decoder de un .pt pero reinicia el paso a 0")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    train_py = repo_root / "scripts" / "train.py"
    if not train_py.exists():
        sys.exit(f"No encuentro {train_py}.")

    py = sys.executable

    if args.resume_from:
        resume = str(Path(args.resume_from).resolve())
        cmd = [py, str(train_py), "--resume_training_from_ckpt", resume]
    else:
        exp_dir = Path(args.outdir).resolve()
        if exp_dir.exists():
            sys.exit(f"'{exp_dir}' ya existe y e4e se niega a sobreescribir.\n"
                     f"Usa otro --outdir (p.ej. runs/e4e2) o borra ese directorio.")
        stylegan_pt = str(Path(args.stylegan_pt).resolve())
        cmd = [
            py, str(train_py),
            "--dataset_type", "my_data_encode",
            "--exp_dir", str(exp_dir),
            "--start_from_latent_avg",
            "--use_w_pool",
            "--w_discriminator_lambda", "0.1",
            "--progressive_start", "20000",
            "--id_lambda", "0.5",          # rostros: identidad con ArcFace
            "--val_interval", "5000",
            "--max_steps", str(args.max_steps),
            "--stylegan_size", "256",
            "--stylegan_weights", stylegan_pt,
            "--workers", "2",
            "--batch_size", str(args.batch),
            "--test_batch_size", "2",
            "--test_workers", "1",
            "--save_training_data",                  # guarda paso/optimizador/discriminador
            "--save_interval", str(args.save_interval),
            "--keep_optimizer",                      # al reanudar, restaura el optimizador
        ]
        if args.init_from:
            cmd += ["--checkpoint_path", str(Path(args.init_from).resolve())]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    if args.dry_run:
        print("[dry-run] No se ejecuto.")
        return
    if args.resume_from:
        print("Reanudando desde checkpoint (continua en el paso guardado + 1).\n")
    else:
        print("Iniciando e4e.\n")
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    main()
