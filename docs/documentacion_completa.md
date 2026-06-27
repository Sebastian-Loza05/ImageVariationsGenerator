# Sistema Generativo Interactivo de Rostros — Documentación completa

> Documento de defensa. Explica **qué construimos**, **por qué tomamos cada decisión**,
> **cómo se entrenó**, **cómo fluye una imagen hasta las 5 variaciones** y **cómo se
> conecta a un endpoint FastAPI**. Pensado para entenderse a fondo sin ser experto.

---

## Tabla de contenidos

1. [Visión general: qué hace el sistema](#1-visión-general)
2. [Por qué este enfoque (justificación)](#2-por-qué-este-enfoque)
3. [El dataset y su preparación](#3-el-dataset-y-su-preparación)
4. [Qué fine-tuneamos exactamente](#4-qué-fine-tuneamos-exactamente)
5. [El pipeline completo, paso a paso](#5-el-pipeline-completo-paso-a-paso)
6. [Decisiones técnicas y por qué](#6-decisiones-técnicas-y-por-qué)
7. [Problemas que resolvimos](#7-problemas-que-resolvimos)
8. [El flujo de inferencia: de una foto a 5 variaciones](#8-el-flujo-de-inferencia)
9. [Integración en un endpoint FastAPI](#9-integración-en-un-endpoint-fastapi)
10. [Resultados y métricas](#10-resultados-y-métricas)
11. [Glosario para la exposición](#11-glosario)
12. [Preguntas frecuentes de defensa](#12-preguntas-frecuentes-de-defensa)

---

## 1. Visión general

El sistema recibe **una foto de un rostro** y produce **exactamente 5 variaciones
visuales** de esa persona (más joven/mayor, más sonriente, otra pose, otro
estilo/iluminación, una versión "idealizada"). El usuario **acepta o rechaza** cada
variación, las decisiones se **registran en un JSON estructurado**, y un **agente de
lenguaje** redacta un resumen coherente con lo aceptado/rechazado.

La parte de "nivel avanzado" del proyecto no es solo *usar* un modelo pre-entrenado:
es **entrenar dos modelos propios** sobre nuestro dataset y conectarlos.

**Analogía sencilla:** imagina un dibujante experto (el **generador**) que sabe pintar
caras realistas a partir de una "receta" numérica. Y un segundo experto (el
**encoder**) que mira una foto y deduce la receta que la reproduce. Si cambiamos un
poco la receta, el dibujante pinta una variación de la misma persona. Nosotros
entrenamos a ambos expertos para que trabajen sobre *nuestras* caras.

---

## 2. Por qué este enfoque

El enunciado permitía varias opciones (DCGAN desde cero, StyleGAN desde cero,
difusión, etc.). Elegimos **StyleGAN2-ADA + encoder e4e con transfer learning** porque
es lo único que **cabía y daba calidad** en el hardware disponible.

**Hardware:** RTX 4070 **Laptop** (8 GB VRAM), Intel i7 13ª gen, 16 GB RAM.

| Opción | Veredicto | Razón |
|---|---|---|
| DCGAN desde cero | ❌ | 64×64 borroso, peor que niveles anteriores |
| StyleGAN2/3 desde cero | ❌ | Necesita ≥16 GB y días en multi-GPU |
| CycleGAN / pix2pix | ❌ | Requieren pares de imágenes / dos dominios |
| Difusión propia desde cero | ❌ | Meses de cómputo |
| **StyleGAN2-ADA + e4e (transfer)** | ✅ | Entra en 8 GB; fine-tune en una noche; inversión por encoder en fracciones de segundo |

**La idea clave — transfer learning:** no partimos de cero. Tomamos un StyleGAN2 ya
entrenado en **FFHQ** (un dataset enorme de 70.000 rostros de alta calidad de NVIDIA)
y solo lo **ajustamos** a nuestras caras. Es como un pintor que ya sabe pintar rostros
y al que solo le enseñamos el "estilo" de nuestro dataset. Esto reduce el entrenamiento
de *días en varias GPUs* a *una noche en una sola GPU*.

---

## 3. El dataset y su preparación

**Dataset:** *Human Faces Dataset* (Kaggle). Trae dos carpetas:
- **Real Images** — 5.000 fotos reales de rostros.
- **AI-Generated Images** — 4.630 caras generadas por IA.

**Decisión 1 — usar solo las reales.** El objetivo es generar caras *realistas*. Meter
caras sintéticas de otros generadores le enseñaría artefactos ajenos al modelo. Por eso
entrenamos solo con las **5.000 reales**.

**Problema encontrado — archivos corruptos.** Al preparar los datos, descubrimos que
**1.755 de las 5.000 imágenes reales estaban vacías (0 bytes)**: la descompresión del
ZIP había quedado incompleta. Verificamos que el ZIP original sí tenía el contenido y
**re-extrajimos**, recuperando las 1.755. También endurecimos el script para que un
archivo ilegible no tumbe todo el proceso (lo salta y sigue).

**Alineado estilo FFHQ — el paso más importante de la preparación.** StyleGAN2 (FFHQ)
espera rostros **centrados y alineados** de una forma muy específica: ojos y boca
siempre en la misma posición, encuadre canónico, 256×256. Usando **dlib** (detector de
68 puntos faciales) recortamos y alineamos cada cara igual que el FFHQ original.

> **Por qué es crítico:** tanto el entrenamiento como la inversión asumen ese encuadre.
> Si una cara entra "torcida" o con otro recorte, el modelo no la entiende bien. Esta
> misma alineación se reutiliza en inferencia para cualquier foto que suba el usuario.

**Resultado:** **4.839 rostros alineados a 256×256** (de las 5.000, algunas no pasaron
la detección de rostro). De ahí hicimos un split **4.639 entrenamiento / 200 validación**.

*Script responsable:* `scripts/01_prepare_data.py` + `src/align.py`.

---

## 4. Qué fine-tuneamos exactamente

Hay **dos entrenamientos propios** (y un paso "puente" entre ellos).

### 4.1 El generador (StyleGAN2-ADA)

- **Qué hace:** convierte un código numérico (la "receta", llamada `w`) en una imagen de
  cara de 256×256.
- **Cómo lo entrenamos:** *transfer learning* desde el FFHQ de NVIDIA, ajustándolo a
  nuestras 4.639 caras. Técnica **ADA** (augmentation adaptativa), que es clave cuando el
  dataset es relativamente pequeño: evita que el modelo se "memorice" los datos.
- **Resultado:** generador propio que produce caras con el "aire" de nuestro dataset.
  Métrica **FID = 9.53** (más bajo = mejor; bajó desde 91 hasta ~10 en las primeras
  iteraciones).

*Script:* `scripts/02_finetune_gan.py` (envuelve `train.py` del repo NVlabs).

### 4.2 El encoder (e4e — *encoder4editing*)

- **Qué hace:** el problema inverso. Mira una **foto real** y deduce la "receta" `w+`
  que el generador necesita para reproducirla. A esto se le llama **inversión** (image →
  latente), y el e4e lo hace **sin optimizar por imagen** (en una fracción de segundo).
- **Por qué re-entrenarlo:** el e4e oficial invierte hacia el StyleGAN de FFHQ. Como
  nosotros **cambiamos el generador** (lo fine-tuneamos), su "espacio de recetas" cambió.
  Entrenamos el encoder **contra NUESTRO generador** para que la inversión sea fiel a
  *nuestras* caras. Esto es entrenamiento propio genuino.
- **Por qué e4e y no otro:** e4e está diseñado para que el código `w+` sea **fácil de
  editar** después (variaciones estables), no solo para reconstruir. Justo lo que
  necesitamos para generar las 5 variaciones.
- **Resultado:** encoder propio, **loss de validación 0.649** (combina reconstrucción +
  identidad).

*Script:* `scripts/03_train_e4e.py` (envuelve `train.py` del repo e4e).

### 4.3 El puente: conversión de formato (`.pkl` → `.pt`)

Los dos repos hablan "idiomas" distintos: el generador de NVlabs se guarda en un formato
(`.pkl`) y el e4e espera otro formato (el de la implementación *rosinality*, `.pt`).
Escribimos un **conversor** que traduce los pesos de uno a otro y lo **verificamos
numéricamente**: dándole la misma receta a ambos generadores, producen la misma imagen
(diferencia media de 0.003, imperceptible). Así garantizamos que no se perdió nada en la
traducción.

*Script:* `scripts/convert_pkl_to_pt.py` → produce `pretrained/our_generator_rosinality.pt`.

> **Dato importante para defender:** el checkpoint final del e4e
> (`runs/e4e2/checkpoints/best_model.pt`) **contiene los dos modelos juntos**: el encoder
> y el generador (decoder). Por eso en inferencia con un solo archivo tenemos todo.

---

## 5. El pipeline completo, paso a paso

```
00_check_env.py     Verifica GPU/CUDA/PyTorch antes de gastar la noche entrenando.
        │
01_prepare_data.py  Alinea las caras estilo FFHQ -> 4.839 imágenes 256×256.
        │
02_finetune_gan.py  Fine-tune del generador desde FFHQ (~una noche). FID 9.53.
        │
convert_pkl_to_pt.py  Traduce el generador al formato que entiende el e4e.
        │
03_train_e4e.py     Entrena el encoder contra NUESTRO generador (80.000 pasos).
        │
04_eval_metrics.py  Métricas (FID, diversidad) para el informe.
        │
05_test_inference.py / app.py   Prueba: foto -> 5 variaciones.
```

Cada script es un **wrapper** que arma la llamada correcta a los repos oficiales con los
parámetros adecuados para 8 GB. Esto mantiene reproducibilidad y deja claro qué
decisiones tomamos.

---

## 6. Decisiones técnicas y por qué

| Decisión | Valor | Por qué |
|---|---|---|
| Transfer desde FFHQ | `--resume ffhq256` | Entrenar desde cero = días/multi-GPU; transfer = una noche |
| Config del generador | `paper256` | Optimizada para 256×256 (resolución que cabe en 8 GB) |
| Ancho del generador | `channel_multiplier=1` | `paper256` usa medio ancho; el e4e por defecto asumía ancho completo → lo ajustamos para que cuadre |
| Batch del generador | 4 | En 8 GB, 8 daba *out of memory* |
| Augmentation | `ADA` | Evita sobreajuste con dataset mediano (~4.600 imágenes) |
| Duración GAN | 600 kimg | Convergió antes (FID en meseta desde ~40k); ~una noche |
| Encoder | e4e (no pSp) | e4e produce códigos *editables*, ideal para variaciones |
| Pérdida de identidad | ArcFace (`id_lambda 0.5`) | Conserva la identidad de la persona en la inversión (caras) |
| Batch del e4e | 2 | El *progressive training* activa más capas con el tiempo → la VRAM crece; con 4 daba OOM a mitad |
| Checkpoints reanudables | `--save_training_data` + `--save_interval` | Poder retomar tras un corte/OOM sin perder horas |
| 5 variaciones | edad, sonrisa, pose, style-mix, truncation | Diversas entre sí e interpretables; el enunciado pide exactamente 5 |

---

## 7. Problemas que resolvimos

Los repos de NVlabs y e4e son de 2021 y chocan con las librerías modernas (PyTorch 2.x).
Documentar esto demuestra trabajo real de ingeniería:

- **NumPy 2 incompatible con PyTorch 2.0.1** → fijamos `numpy<2`.
- **`pkg_resources` eliminado en setuptools nuevos** → fijamos `setuptools<66`.
- **Segunda derivada de `grid_sample` no implementada en PyTorch 2.x** (rompía la
  regularización del GAN) → parcheamos el *workaround* de NVIDIA para que se active en
  PyTorch 2.x.
- **El e4e elegía MoCo en vez de ArcFace** para datasets que no se llamaban "ffhq" →
  lo corregimos para que nuestras caras usen ArcFace.
- **OOM a mitad del entrenamiento del e4e** → batch 2 + ajuste del asignador de memoria.

Todos están registrados; no son "magia", son decisiones trazables.

---

## 8. El flujo de inferencia

Esta es la parte central para la demo. **De una foto a 5 variaciones:**

```
   FOTO del usuario (cualquier tamaño/encuadre)
        │
        ▼  (1) ALINEADO            src/align.py  ·  dlib
   Rostro alineado FFHQ 256×256
        │
        ▼  (2) INVERSIÓN           src/inversion.py  ·  nuestro encoder e4e
   w+  =  receta latente [14 vectores × 512]   +   reconstrucción
        │
        ▼  (3) 5 VARIACIONES       src/variations.py  ·  nuestro generador
   V1 edad · V2 sonrisa · V3 pose · V4 style-mix · V5 truncation
        │
        ▼  (4) FEEDBACK HUMANO     src/decision_log.py
   El usuario acepta/rechaza → JSON estructurado
        │
        ▼  (5) EXPLICACIÓN         src/explainer.py
   Agente LLM redacta un resumen coherente con las decisiones
```

### 8.1 Alineado (paso 1)

Recortamos/centramos la cara subida **exactamente igual** que las del entrenamiento. Sin
esto, la inversión sale mal. Si no se detecta rostro, se avisa al usuario. *(Mismo
algoritmo FFHQ del `01_prepare_data.py`, reutilizado en `src/align.py`.)*

### 8.2 Inversión: imagen → `w+` (paso 2)

El encoder e4e convierte la cara alineada en **`w+`**, un tensor de forma **[1, 14, 512]**:
14 vectores de 512 números, uno por cada "capa de estilo" del generador a 256px.

**Qué es `w+` (intuición):** es el "ADN numérico" de esa cara dentro del mundo del
generador. Las **capas bajas** controlan rasgos gruesos (forma de la cara, pose); las
**capas altas**, detalles finos (textura de piel, color, iluminación). Editar partes
distintas de `w+` cambia cosas distintas.

También obtenemos la **reconstrucción** (lo que el generador pinta con esa receta) para
mostrar al usuario qué tan fiel quedó la inversión.

### 8.3 Las 5 variaciones (paso 3)

Tomamos el `w+` de la persona y lo modificamos de 5 maneras; cada `w+` modificado se
vuelve a pintar con **nuestro generador**:

| # | Variación | Mecanismo | Qué cambia |
|---|---|---|---|
| **V1** | Mayor edad | Sumamos una *dirección semántica* de edad a `w+` | Envejece el rostro |
| **V2** | Más sonriente | Dirección de sonrisa | Expresión |
| **V3** | Cambio de pose | Dirección de pose | Orientación de la cabeza |
| **V4** | Estilo/iluminación | *Style-mixing*: mezclamos las capas finas con un estilo aleatorio | Textura, color, luz |
| **V5** | Versión idealizada | *Truncation*: acercamos `w+` a la cara "promedio" | Más limpia/genérica |

**Sobre las direcciones semánticas:** V1–V3 usan vectores `[512]` que representan "hacia
dónde mover la receta para envejecer / sonreír / girar". Estos vectores se calculan con
métodos como **InterFaceGAN/GANSpace**. Si esos vectores aún no están calculados, el
sistema usa un **desplazamiento controlado de respaldo** para que **nunca falle** y
siempre entregue 5 variaciones distintas (decisión de robustez para la demo).

**Style-mixing (V4):** generamos una receta aleatoria y reemplazamos solo las **capas
finas** del usuario por las de esa receta → cambia el "estilo" (luz, textura) pero
mantiene la identidad.

**Truncation (V5):** mezclamos el `w+` con la cara promedio del modelo → versión más
"de catálogo", suave e idealizada.

> **Punto clave de coherencia:** las 5 variaciones se generan con el **mismo generador**
> que usó la inversión (el decoder que vive dentro del checkpoint del e4e). No cargamos
> un segundo generador: garantiza consistencia y evita conflictos entre los dos repos.

### 8.4 Feedback humano y explicación (pasos 4–5)

El usuario marca aceptar/rechazar cada variación. `decision_log.py` guarda un **JSON
estructurado** (qué se aceptó, qué se rechazó, con etiquetas y descripciones).
`explainer.py` (agente de lenguaje) genera un **resumen textual** coherente con ese log
— es la parte de "explicabilidad" e "interacción humano-IA" de la rúbrica.

---

## 9. Integración en un endpoint FastAPI

La lógica de inferencia es **independiente de la interfaz**: las mismas tres clases
(`FaceAligner`, `E4EInverter`, `VariationGenerator`) se conectan a Gradio **o** a FastAPI
**o** a cualquier servidor. Aquí el ejemplo con FastAPI.

### 9.1 Reglas de oro

1. **Cargar los modelos UNA vez al arrancar** (no por petición): tarda unos segundos y la
   primera pasada compila operaciones de GPU.
2. **Serializar las peticiones** (un *lock*): los modelos no son seguros para uso
   concurrente y la VRAM de 8 GB no da para varias inferencias a la vez.
3. **Devolver las imágenes** como PNG en base64 (o como archivos), no como objetos PIL.
4. En inferencia **solo** se necesita el repo `encoder4editing` en el `PYTHONPATH`, dlib y
   torch. **No** hacen falta el repo de StyleGAN, ni IR-SE50, ni MoCo (eso era para
   entrenar).

### 9.2 Código de ejemplo (`server.py`)

```python
import io, base64, sys, threading
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- rutas del proyecto (ajústalas a tu despliegue) ---
BASE   = Path("/ruta/a/generative_faces")
E4EREPO = Path("/ruta/a/encoder4editing")
sys.path.insert(0, str(E4EREPO))   # provee models.psp (clase pSp del e4e)
sys.path.insert(0, str(BASE))      # provee src.*

from src.align import FaceAligner
from src.inversion import E4EInverter
from src.variations import VariationGenerator

app = FastAPI(title="Generador de Variaciones de Rostro")

# --- carga ÚNICA de modelos al arrancar el servidor ---
ALIGNER = INVERTER = VARGEN = None
LOCK = threading.Lock()            # serializa el acceso a la GPU

@app.on_event("startup")
def load_models():
    global ALIGNER, INVERTER, VARGEN
    ALIGNER  = FaceAligner(BASE / "pretrained/shape_predictor_68_face_landmarks.dat")
    INVERTER = E4EInverter(BASE / "runs/e4e2/checkpoints/best_model.pt", device="cuda")
    VARGEN   = VariationGenerator(INVERTER.decoder, INVERTER.latent_avg,
                                  directions_dir=str(BASE / "directions"), device="cuda")

def _png_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

class VariationOut(BaseModel):
    index: int
    label: str
    description: str
    image_b64: str

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    # 1) leer la foto subida
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    with LOCK:                       # una inferencia a la vez (8 GB)
        # 2) alinear estilo FFHQ
        aligned = ALIGNER.align(img)
        if aligned is None:
            raise HTTPException(400, "No se detectó un rostro. Usa una foto más frontal.")
        # 3) invertir: imagen -> w+  (+ reconstrucción)
        w_plus, rec = INVERTER.invert(aligned)
        # 4) generar EXACTAMENTE 5 variaciones
        variations = VARGEN.generate(w_plus)

    # 5) responder en JSON (imágenes en base64)
    return {
        "aligned_b64": _png_b64(aligned),
        "reconstruction_b64": _png_b64(rec),
        "variations": [
            VariationOut(index=v.index, label=v.label,
                         description=v.description, image_b64=_png_b64(v.image)).dict()
            for v in variations
        ],
    }
```

Lanzarlo: `uvicorn server:app --host 0.0.0.0 --port 8000` **con un solo worker**
(`--workers 1`) porque hay una sola GPU.

### 9.3 Endpoint opcional de feedback

Para cerrar el ciclo humano-IA, un segundo endpoint recibe las decisiones
(aceptar/rechazar por variación), las pasa a `SessionLog` y devuelve el resumen del
agente con `explain(...)`. La estructura es idéntica: recibes JSON con las 5 decisiones,
construyes el log, lo guardas y devuelves el texto explicativo.

### 9.4 Cómo lo consume el cliente

El front (web/móvil) hace `POST /generate` con la foto (multipart) y recibe un JSON con la
reconstrucción y las 5 variaciones en base64, que pinta directamente en `<img>`. Luego
envía las decisiones del usuario al endpoint de feedback.

---

## 10. Resultados y métricas

| Modelo | Métrica | Valor | Interpretación |
|---|---|---|---|
| Generador (GAN) | **FID** | **9.53** | Realismo/parecido al dataset (más bajo = mejor; bajó desde 91) |
| Encoder (e4e) | **Loss val.** | **0.649** | Combina reconstrucción + identidad (más bajo = mejor) |
| Conversión `.pkl→.pt` | dif. media | **0.003** | Verificación de que la traducción no perdió nada |
| Dataset final | imágenes | **4.839** | Caras reales alineadas (split 4.639 / 200) |

- El GAN **convergió rápido**: el salto grande (FID 91 → 15) ocurrió en las primeras
  40k imágenes; luego meseta ~10. Esto valida que el *transfer learning* fue eficiente.
- La inversión es **fiel**: la reconstrucción conserva identidad, pose, expresión y pelo.

---

## 11. Glosario

- **Espacio latente / receta (`w`, `w+`):** representación numérica compacta de una cara.
  `w+` = 14 vectores de 512 (uno por capa de estilo); cada zona controla rasgos distintos.
- **Generador (StyleGAN2):** red que convierte una receta en una imagen.
- **Inversión:** proceso inverso, de imagen a receta. Lo hace el **encoder e4e**.
- **Transfer learning:** partir de un modelo pre-entrenado (FFHQ) y solo ajustarlo.
- **Fine-tuning:** ese ajuste sobre nuestros datos.
- **ADA:** *Adaptive Discriminator Augmentation*; aumenta los datos de forma adaptativa
  para evitar sobreajuste con datasets no enormes.
- **FID:** *Fréchet Inception Distance*; mide qué tan parecidas (realismo + diversidad)
  son las imágenes generadas a las reales. Más bajo = mejor.
- **ArcFace / IR-SE50:** red de reconocimiento facial usada como "pérdida de identidad"
  para que la inversión mantenga a la misma persona.
- **Style-mixing:** mezclar capas de dos recetas para combinar identidad y estilo.
- **Truncation:** acercar una receta a la "cara promedio" para una versión más idealizada.
- **Alineado FFHQ:** recorte/encuadre canónico de rostro que el modelo espera.

---

## 12. Preguntas frecuentes de defensa

**¿Qué entrenaron ustedes, exactamente?**
Dos modelos: (1) el **generador** StyleGAN2-ADA, fine-tuneado desde FFHQ sobre nuestras
4.639 caras; y (2) el **encoder e4e**, entrenado desde cero contra *ese* generador. Más un
conversor verificado que los conecta.

**¿No es solo usar un modelo pre-entrenado?**
No. El transfer learning ajusta los pesos del generador a nuestro dataset (es
entrenamiento real), y el encoder se entrena **específicamente contra nuestro generador**
(no serviría el oficial). Ambos producen pesos propios.

**¿Por qué 256×256 y no 1024?**
Por la VRAM de 8 GB. 256 da buena calidad y entra en memoria; 1024 requeriría mucho más.

**¿Por qué exactamente 5 variaciones y cómo garantizan que siempre sean 5?**
Lo pide el enunciado. El generador tiene un plan fijo de 5 mecanismos; si una dirección
semántica no está disponible, hay un respaldo determinista para que nunca falle.

**¿Cómo aseguran que la cara generada es la misma persona?**
Por la **pérdida de identidad ArcFace** durante el entrenamiento del encoder, y porque las
variaciones son *ediciones pequeñas* de la receta original, no caras nuevas.

**¿Qué pasa si suben una foto sin rostro o muy de lado?**
El alineador detecta el rostro; si no encuentra uno, el sistema responde con un aviso en
vez de generar basura.

**¿Es esto escalable a un servidor?**
Sí: se cargan los modelos una vez y se atienden peticiones serializadas (una GPU). El
ejemplo FastAPI de la sección 9 lo muestra.

---

*Archivos relevantes:* `scripts/01_prepare_data.py`, `scripts/02_finetune_gan.py`,
`scripts/convert_pkl_to_pt.py`, `scripts/03_train_e4e.py`, `scripts/05_test_inference.py`,
`src/align.py`, `src/inversion.py`, `src/variations.py`, `src/decision_log.py`,
`src/explainer.py`, `app.py`.
Modelos: `pretrained/our_generator_rosinality.pt`, `runs/e4e2/checkpoints/best_model.pt`.
