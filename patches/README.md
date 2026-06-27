# Parches a los repos externos

Este proyecto usa dos repos oficiales **como dependencias** (no se incluyen en este
repositorio): se clonan **un nivel arriba** de `generative_faces/`. Para que el
entrenamiento y la inferencia funcionen con PyTorch 2.x y con nuestro modelo, hubo que
aplicarles unos parches. Aquí están guardados para que el proyecto sea reproducible.

## 1. Clonar los repos (al lado de generative_faces/)

```
Proyecto2/
├── generative_faces/        <- este repo
├── stylegan2-ada-pytorch/   <- git clone https://github.com/NVlabs/stylegan2-ada-pytorch
└── encoder4editing/         <- git clone https://github.com/omertov/encoder4editing
```

## 2. Aplicar los parches

```bash
cd ../stylegan2-ada-pytorch && git apply ../generative_faces/patches/stylegan2-ada-pytorch.patch
cd ../encoder4editing       && git apply ../generative_faces/patches/encoder4editing.patch
```

## 3. Qué cambia cada parche (y por qué)

**`stylegan2-ada-pytorch.patch`** — compatibilidad con PyTorch 2.x:
- `grid_sample_gradfix.py`: habilita el custom op de la 2ª derivada de `grid_sample`
  (la R1 fallaba con "derivative ... not implemented") y usa la API nueva de aten.
- `conv2d_gradfix.py`: silencia el warning repetido (cae al conv2d nativo, correcto en 2.x).

**`encoder4editing.patch`** — adaptarlo a nuestro modelo:
- `models/psp.py`: `channel_multiplier=1` (nuestro generador es `paper256`, de medio ancho).
- `training/coach.py`: usa ArcFace para nuestro dataset de rostros (no MoCo); poda
  checkpoints viejos; tolera `batch_size=1` en el logging.
- `configs/data_configs.py` y `paths_config.py`: registran el dataset `my_data_encode`.

> **Ojo:** `paths_config.py` trae rutas ABSOLUTAS de la máquina original. Tras aplicar el
> parche, ajusta `my_data_train` / `my_data_val` a tu ruta local.

## 4. Modelos auxiliares (no versionados)

El entrenamiento del e4e necesita `model_ir_se50.pth` en
`encoder4editing/pretrained_models/` (ver `scripts/03_train_e4e.py`). La descarga está
documentada en ese script.
