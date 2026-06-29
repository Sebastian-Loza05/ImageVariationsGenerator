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
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true",
                    help="solo imprime el comando, no entrena")
    args = ap.parse_args()

    train_py = Path(args.repo) / "train.py"
    if not train_py.exists():
        sys.exit(f"No encuentro {train_py}.")

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
        "--metrics", "fid50k_full",
    ]

    if args.dry_run:
        print("[dry-run] No se ejecuto el entrenamiento.")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


