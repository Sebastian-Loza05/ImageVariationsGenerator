set -e

PYVER="3.9.18"
ENVDIR=".venv"

echo "=================================================="
echo " Setup — StyleGAN2-ADA + e4e (RTX 4070, pyenv+pip)"
echo "=================================================="

echo ""
echo "== 0. Dependencias del sistema (Arch) =="
echo "Instala si no las tienes:"
echo "  sudo pacman -S --needed base-devel cmake ninja gcc11 cuda cudnn"
echo "  # pyenv y sus dependencias de build de Python:"
echo "  sudo pacman -S --needed pyenv openssl zlib xz tk libffi bzip2 readline"
echo ""

echo "== 1. Verificar pyenv =="
if ! command -v pyenv >/dev/null 2>&1; then
  echo "  [FAIL] pyenv no esta en PATH."
  echo "  Instala:  sudo pacman -S pyenv"
  echo "  Y agrega a tu ~/.zshrc (usas zsh):"
  echo '    export PYENV_ROOT="$HOME/.pyenv"'
  echo '    export PATH="$PYENV_ROOT/bin:$PATH"'
  echo '    eval "$(pyenv init -)"'
  echo "  Luego reabre la terminal y reintenta."
  exit 1
fi
echo "  pyenv encontrado: $(pyenv --version)"

echo ""
echo "== 2. Instalar Python ${PYVER} con pyenv =="
if pyenv versions --bare | grep -qx "${PYVER}"; then
  echo "  Python ${PYVER} ya instalado."
else
  echo "  Compilando Python ${PYVER} (toma un par de minutos)..."
  pyenv install "${PYVER}"
fi

echo ""
echo "== 3. Crear venv con ese Python =="
PYBIN="$(pyenv root)/versions/${PYVER}/bin/python"
"${PYBIN}" -m venv "${ENVDIR}"
# shellcheck disable=SC1090
source "${ENVDIR}/bin/activate"
python --version
# setuptools<66: torch 2.0.1 hace `from pkg_resources import packaging` en
# cpp_extension.py; setuptools modernos (>=71) eliminaron pkg_resources y
# rompen el build de los CUDA ops de StyleGAN2-ADA.
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<66"

echo ""
echo "== 4. gcc 11 para nvcc =="
if command -v gcc-11 >/dev/null 2>&1 || command -v g++-11 >/dev/null 2>&1; then
  export CC=gcc-11 CXX=g++-11
  echo "  Usando CC=gcc-11 CXX=g++-11"
else
  echo "  [AVISO] gcc-11 no encontrado. Necesario al compilar los CUDA ops."
  echo "          sudo pacman -S gcc11   (o desde AUR si ya no esta en repos)"
fi

echo ""
echo "== 5. PyTorch 2.0.1 + cu118 (reconoce sm_89 de la 4070) =="
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "== 6. Dependencias del proyecto (dlib aparte) =="
# dlib compila desde fuente con el cmake del sistema. El resto va normal con pip.
grep -v '^dlib' requirements.txt > /tmp/reqs_sin_dlib.txt
pip install -r /tmp/reqs_sin_dlib.txt
echo "  Instalando dlib (usa el cmake del sistema)..."
pip install dlib

echo ""
echo "== 7. Verificar que PyTorch VE la GPU =="
python - << 'PY'
import torch
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print("GPU:", torch.cuda.get_device_name(0), "| compute capability:", sm)
    archs = torch.cuda.get_arch_list()
    print("Arquitecturas soportadas por torch:", archs)
    print("La 4070 (sm_89) esta soportada:", sm in archs or "sm_90" in archs)
else:
    print("[FAIL] PyTorch no ve la GPU. Revisa driver NVIDIA y que el env sea cu118.")
PY

echo ""
echo "=================================================="
echo " Listo. Siguiente paso:"
echo "   source ${ENVDIR}/bin/activate"
echo "   python scripts/00_check_env.py"
echo "=================================================="
