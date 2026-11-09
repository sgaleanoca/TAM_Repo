# Proyecto TAM: Reconocimiento de Números Manuscritos del 0 al 99

Aplicación local en Python y Streamlit que reutiliza el modelo de clasificación de dígitos desarrollado en `Clase_05_11_26_TAM.ipynb` (TensorFlow/Keras sobre MNIST) e incorpora un pipeline de visión por computador para segmentar, adaptar y reconocer números manuscritos completos entre **0 y 99**.

---

## 1. Auditoría Técnica del Cuaderno Original (`Clase_05_11_26_TAM.ipynb`)

### 1.1 Arquitectura Identificada
El modelo original utiliza la **Functional API** de Keras y corresponde a un Perceptrón Multicapa (MLP) totalmente conectado:
- **Capa de entrada (`InputLayer`)**: Vector plano de `784` características ($28 \times 28$ píxeles) — `0` parámetros.
- **Primera capa oculta (`Dense`)**: `64` neuronas con activación `ReLU` — $784 \times 64 + 64 = 50,240$ parámetros.
- **Segunda capa oculta (`Dense`)**: `64` neuronas con activación `ReLU` — $64 \times 64 + 64 = 4,160$ parámetros.
- **Capa de salida (`Dense`)**: `10` neuronas con activación `Softmax` (clases `0` a `9`) — $64 \times 10 + 10 = 650$ parámetros.
- **Total de parámetros entrenables**: **55,050 parámetros**.

### 1.2 Funcionamiento del Entrenamiento
- **Optimizador**: `adam`.
- **Función de pérdida**: `sparse_categorical_crossentropy` (las etiquetas `y_train` son enteros del `0` al `9`).
- **Métrica**: `accuracy`.
- **Hiperparámetros de ajuste**: `epochs=30`, `batch_size=32`.
- **Observación de auditoría**: El cuaderno original no contenía una instrucción `Model.save(...)` ni evaluación sobre `x_test`. Por ello, en el módulo `src/model_manager.py` se ejecutó el entrenamiento fiel de 30 épocas conservando intacta la arquitectura, alcanzando **`97.71%` de exactitud en entrenamiento** y **`95.66%` en prueba (`x_test`)**, persistiendo el modelo en `models/mnist_mlp.keras`.

### 1.3 Preprocesamiento Original (Hallazgo Crítico)
En la celda 4 del cuaderno original únicamente se aplica:
```python
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
```
1. **Sin normalización al rango `[0, 1]`**: Los tensores conservan sus valores enteros originales en el rango **`[0, 255]`** (sin dividir entre `255.0`).
2. **Polaridad tonal de MNIST**: Fondo negro (`0`) y trazos del dígito blancos/claros (`> 0` hasta `255`).
3. **Geometría canónica de MNIST**: Cada dígito individual está inscrito proporcionalmente en una caja de $20 \times 20$ píxeles y centrado mediante su **centro de masa** dentro del lienzo de $28 \times 28$ píxeles (margen de 4 píxeles por lado).

### 1.4 Capacidades y Limitaciones del Modelo
- **Capacidades**: Clasifica con alta velocidad y buena exactitud (`95.66%` en test) dígitos aislados del `0` al `9` que respeten estrictamente la distribución geométrica y radiométrica de MNIST.
- **Limitaciones**:
  1. **Salida restringida a 10 clases (`0–9`)**: No puede clasificar directamente un número de dos cifras (ej. `47`) en una sola pasada sin antes segmentar los dígitos.
  2. **Ausencia de invarianza espacial (MLP vs. CNN)**: Al aplanar la imagen a un vector de `784` elementos en lugar de usar capas convolucionales (`Conv2D`), el peso de cada neurona está atado a una coordenada fija $(x, y)$. Si un dígito está descentrado o tiene una escala distinta, las activaciones fallan.
  3. **Sobreconfianza Softmax en entradas en `[0, 255]`**: Al multiplicar entradas de magnitud `255` por capas `ReLU`, los *logits* previos a la capa `Softmax` alcanzan valores grandes. Esto provoca que la distribución `Softmax` asigne probabilidades cercanas al `99%–100%` incluso a recortes ruidosos o fuera de distribución (*Out-of-Distribution*).

---

## 2. Estrategia Implementada para Reconocer Números del 0 al 99

Para reconocer números de una o dos cifras reutilizando el modelo original sin alterar su arquitectura:

1. **Preprocesamiento Global (`src/preprocessing.py`)**:
   - Validación de dimensiones, composición de canales transparentes (`RGBA` sobre blanco) y conversión a escala de grises.
   - Detección automática de polaridad midiendo la intensidad mediana de los bordes: si el fondo es claro (papel blanco o gris), se invierte automáticamente para obtener fondo negro (`0`) y trazo blanco (`[0, 255]`).
   - Corrección morfológica de iluminación de fondo (*Top-Hat*) y umbralización de Otsu con filtro de apertura para obtener una máscara binaria limpia.
2. **Segmentación Espacial de Dígitos (`src/segmentation.py`)**:
   - Extracción de componentes conexos (`cv2.connectedComponentsWithStats`) y descarte de ruido menor.
   - **Fusión horizontal**: Si un dígito fue escrito en dos trazos desconectados verticalmente (por ejemplo, la barra superior del `5` o un `4`), sus cajas se fusionan cuando el solapamiento horizontal supera el `42%`.
   - **Separación por perfil de proyección vertical**: Si dos dígitos están en contacto ligero formando un único componente ancho ($w/h \ge 1.18$), se analiza el histograma de suma vertical de píxeles buscando el valle mínimo central para dividirlos limpiamente.
   - **Ordenamiento y validación**: Las regiones se ordenan de izquierda a derecha (Decenas $\rightarrow$ Unidades) y se valida que existan **1 o 2 dígitos**.
3. **Adaptación al Estándar MNIST (`format_digit_for_mnist`)**:
   - Regulación morfológica del grosor del trazo (engrosando trazos ultra-finos de alta resolución o erosionando suavemente trazos saturados).
   - Redimensionamiento proporcional a caja de $20 \times 20$ píxeles, centrado por **centro de masa** en lienzo de $28 \times 28$ píxeles, reescalado al rango **`[0, 255]`** y aplanado a `(1, 784)`.
4. **Inferencia y Reconstrucción (`src/inference.py`)**:
   - Predicción independiente de cada dígito con `models/mnist_mlp.keras` y reconstrucción posicional:
     $$\text{Número} = \sum_{i=0}^{n-1} d_i \cdot 10^{n - 1 - i} \in [0, 99]$$

---

## 3. Estructura Modular del Proyecto

```text
/home/zerendor/Documents/TAM/gemini/task1/
├── Clase_05_11_26_TAM.ipynb     # Cuaderno original (intacto, sin modificaciones)
├── app.py                       # Interfaz gráfica web local en Streamlit
├── requirements.txt             # Dependencias del proyecto
├── pytest.ini                   # Configuración de pruebas automatizadas
├── models/
│   ├── mnist_mlp.keras          # Modelo original entrenado (30 épocas) y serializado
│   └── training_metrics.json    # Métricas de entrenamiento y evaluación en MNIST
├── src/
│   ├── __init__.py
│   ├── model_manager.py         # Construcción, entrenamiento fiel y carga del modelo .keras
│   ├── preprocessing.py         # Preprocesamiento radiométrico y adaptación geométrica MNIST
│   ├── segmentation.py          # Detección, fusión, división por proyección y validación (1-2 dígitos)
│   └── inference.py             # Inferencia por dígito y reconstrucción posicional (0-99)
└── tests/
    ├── generate_test_images.py  # Generador de imágenes manuscritas propias (externas a MNIST)
    ├── evaluate_suite.py        # Evaluación sistemática y generación de reporte experimental
    ├── test_pipeline.py         # Suite de pruebas unitarias e integrales (pytest + Streamlit AppTest)
    ├── test_results.json        # Resultados estructurados de las pruebas
    ├── REPORTE_PRUEBAS.md       # Informe detallado de pruebas y análisis de errores
    └── sample_images/           # 26 imágenes manuscritas de prueba en 7 categorías
```

---

## 4. Ejecución en Lubuntu

El proyecto cuenta con un entorno virtual local en `.venv` (configurado con Python 3.12 para compatibilidad oficial con los binarios de TensorFlow en Lubuntu):

### 4.1 Iniciar la Interfaz Gráfica (Streamlit)
```bash
cd /home/zerendor/Documents/TAM/gemini/task1
.venv/bin/streamlit run app.py
```

### 4.2 Ejecutar las Pruebas Automatizadas y Evaluación Experimental
```bash
# Ejecutar suite unitaria e integral (incluye prueba de interfaz Streamlit)
.venv/bin/pytest -v

# Regenerar imágenes manuscritas de prueba y reporte experimental
.venv/bin/python -m tests.evaluate_suite
```
