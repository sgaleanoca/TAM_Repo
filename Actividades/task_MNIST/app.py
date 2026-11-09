"""
Interfaz gráfica web local desarrollada en Streamlit para el reconocimiento
de números manuscritos entre 0 y 99 utilizando el modelo original de TAM.

Ejecución local:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from src.inference import NumberRecognitionResult, predict_number_from_image
from src.model_manager import DEFAULT_MODEL_PATH, get_training_metrics, load_trained_model
from src.preprocessing import ImageProcessingError
from src.segmentation import SegmentationError
from tests.generate_test_images import SAMPLE_IMAGES_DIR, generate_all_test_images


st.set_page_config(
    page_title="Proyecto TAM — Reconocimiento de Números (0 al 99)",
    page_icon="🔢",
    layout="wide",
)


@st.cache_resource
def get_cached_model():
    """Carga el modelo `.keras` una sola vez en memoria caché de Streamlit."""
    return load_trained_model(DEFAULT_MODEL_PATH, auto_train_if_missing=True)


def ensure_samples_available() -> Dict[str, Path]:
    """Garantiza que las imágenes manuscritas de ejemplo existan en disco."""
    if not SAMPLE_IMAGES_DIR.exists() or not list(SAMPLE_IMAGES_DIR.glob("*.png")):
        generate_all_test_images(SAMPLE_IMAGES_DIR)
    files = sorted(SAMPLE_IMAGES_DIR.glob("*.png"))
    return {f.name: f for f in files}


def render_probabilities_chart(probabilities: list[float], predicted_digit: int):
    """Genera una gráfica de barras clara para las 10 probabilidades Softmax (0 al 9)."""
    fig, ax = plt.subplots(figsize=(5.5, 2.6), dpi=110)
    classes = list(range(10))
    colors = ["#1f77b4" if c != predicted_digit else "#2ca02c" for c in classes]
    bars = ax.bar(classes, [p * 100.0 for p in probabilities], color=colors, width=0.65)

    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Probabilidad (%)", fontsize=9)
    ax.set_xlabel("Clase de dígito (0 - 9)", fontsize=9)
    ax.set_title("Distribución de probabilidades Softmax", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, prob in zip(bars, probabilities):
        pct = prob * 100.0
        if pct >= 1.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(103.0, pct + 2.0),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig.tight_layout()
    return fig


def main() -> None:
    st.title("🔢 Reconocimiento de Números Manuscritos (0 al 99)")
    st.markdown(
        "**Proyecto TAM** — Aplicación local basada en el modelo MLP entrenado sobre MNIST "
        "(`Clase_05_11_26_TAM.ipynb`) con segmentación espacial e inferencia independiente por dígito."
    )

    # Barra lateral con información técnica del modelo y selector rápido de pruebas
    with st.sidebar:
        st.header("⚙️ Información del Modelo")
        metrics = get_training_metrics()
        st.markdown(
            f"""
            - **Archivo**: `{DEFAULT_MODEL_PATH.name}`
            - **Arquitectura**: `Input(784) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
            - **Parámetros entrenables**: `{metrics.get('total_params', 55050):,}`
            - **Rango de entrada**: `[0, 255]` *(sin normalizar a 0–1)*
            - **Épocas de entrenamiento**: `{metrics.get('epochs', 30)}`
            - **Exactitud Train / Test MNIST**: `{metrics.get('final_train_accuracy', 0.9771)*100:.2f}%` / `{metrics.get('test_accuracy', 0.9566)*100:.2f}%`
            """
        )

        st.divider()
        st.subheader("🧪 Modo de Entrada")
        input_mode = st.radio(
            "Seleccione el origen de la imagen:",
            options=[
                "Subir archivo propio (Arrastrar y soltar)",
                "Probar con galería de imágenes manuscritas",
            ],
            index=0,
        )

        st.divider()
        st.info(
            "💡 **Nota académica sobre Softmax**:\n\n"
            "El modelo original opera sobre intensidades `[0, 255]` sin normalizar. "
            "Esto genera *logits* elevados que saturan la función Softmax cerca del 100%. "
            "Por tanto, una probabilidad cercana a 1.0 indica alta activación relativa, "
            "no garantía absoluta de exactitud."
        )

    selected_image: Image.Image | None = None
    source_name: str = ""

    if input_mode == "Subir archivo propio (Arrastrar y soltar)":
        uploaded_file = st.file_uploader(
            "Arrastre y suelte una imagen manuscrita aquí o haga clic en 'Browse files'",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Admite imágenes con 1 o 2 dígitos escritos a mano (números del 0 al 99).",
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                source_name = uploaded_file.name
            except Exception as exc:
                st.error(f"❌ No se pudo leer el archivo subido: {exc}")
    else:
        samples = ensure_samples_available()
        chosen_name = st.selectbox(
            "Seleccione una imagen manuscrita de prueba:",
            options=list(samples.keys()),
            index=list(samples.keys()).index("two_digits_47.png")
            if "two_digits_47.png" in samples
            else 0,
        )
        if chosen_name:
            selected_image = Image.open(samples[chosen_name])
            source_name = chosen_name

    if selected_image is None:
        st.markdown(
            "👆 **Cargue una imagen desde su computador** en el recuadro superior "
            "(o elija una imagen de la galería en el panel lateral) para comenzar."
        )
        return

    # Mostrar la imagen cargada y el botón de ejecución
    col_preview, col_action = st.columns([1, 1.4])
    with col_preview:
        st.subheader("🖼️ Imagen Cargada")
        st.image(selected_image, caption=f"Archivo: {source_name}", width=280)

    with col_action:
        st.subheader("🚀 Ejecutar Reconocimiento")
        st.write(
            "Al presionar el botón, el sistema preprocesará la imagen al formato MNIST (`[0, 255]`, "
            "fondo negro, trazo blanco), segmentará los dígitos de izquierda a derecha y reconstruirá "
            "el número entre **0 y 99**."
        )
        run_btn = st.button("🔍 Reconocer Número", type="primary", width="stretch")

    if not run_btn:
        return

    try:
        model = get_cached_model()
        with st.spinner("Procesando imagen, segmentando dígitos y ejecutando inferencia..."):
            result: NumberRecognitionResult = predict_number_from_image(selected_image, model=model)
    except (ImageProcessingError, SegmentationError) as err:
        st.error(f"⚠️ **No se pudo procesar la imagen**: {err}")
        st.caption(
            "Sugerencia: Verifique que la imagen contenga únicamente 1 o 2 dígitos claramente visibles "
            "y separados entre sí."
        )
        return
    except Exception as unexpected:
        st.error(f"❌ **Error inesperado al procesar la imagen**: {unexpected}")
        return

    st.divider()

    # 1. Resultado principal del número identificado (0 al 99)
    st.subheader("✅ Resultado del Reconocimiento")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(
            label="Número Identificado (0 – 99)",
            value=f"{result.recognized_number}",
            delta=f"Lectura posicional: '{result.formatted_string}'",
        )
    with res_col2:
        st.metric(
            label="Dígitos Detectados",
            value=f"{result.num_digits}",
            delta="Un dígito (0–9)" if result.num_digits == 1 else "Dos dígitos (Decena y Unidad)",
        )
    with res_col3:
        st.metric(
            label="Probabilidad Softmax Conjunta",
            value=f"{result.joint_confidence * 100:.2f}%",
            delta="Producto de confianzas individuales",
        )

    for warn in result.warnings:
        st.warning(f"ℹ️ {warn}")

    # 2. Etapas de Preprocesamiento y Segmentación
    with st.expander("🔬 Ver etapas de preprocesamiento y segmentación espacial", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.image(
                result.segmentation.annotated_rgb,
                caption="1. Detección y orden de lectura (Izquierda → Derecha)",
                width=260,
            )
        with p_col2:
            st.image(
                result.preprocessed.binary_mask,
                caption="2. Máscara binaria umbralizada (Otsu + limpieza)",
                width=260,
            )
        with p_col3:
            st.image(
                result.preprocessed.normalized_gray,
                caption=(
                    "3. Intensidades estilo MNIST [0, 255] "
                    f"({'Polaridad invertida' if result.preprocessed.inverted_polarity else 'Polaridad original'})"
                ),
                width=260,
            )

    # 3. Desglose individual por dígito y probabilidades de clasificación
    st.subheader("📊 Predicciones Individuales por Dígito")
    digit_cols = st.columns(result.num_digits)

    for idx, dp in enumerate(result.digit_predictions):
        with digit_cols[idx]:
            st.markdown(f"### Dígito {idx + 1}: **{dp.role_label}**")
            sub_c1, sub_c2 = st.columns([1, 1.5])
            with sub_c1:
                st.image(
                    dp.segmented_digit.mnist_28x28,
                    caption="Entrada 28×28 px [0, 255]",
                    width=130,
                )
            with sub_c2:
                st.markdown(
                    f"""
                    - **Clase predicha**: **`{dp.predicted_digit}`**
                    - **Probabilidad Softmax**: **`{dp.confidence * 100:.2f}%`**
                    - **Caja (x, y, w, h)**: `{dp.segmented_digit.bbox}`
                    - **Separado por proyección**: `{"Sí" if dp.segmented_digit.split_by_projection else "No"}`
                    """
                )

            fig = render_probabilities_chart(dp.probabilities, dp.predicted_digit)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            with st.expander(f"Ver tabla completa de probabilidades (Dígito {idx + 1})"):
                prob_rows = {
                    f"Clase {c}": f"{dp.probabilities[c] * 100:.4f}%"
                    for c in range(10)
                }
                st.table(prob_rows)


if __name__ == "__main__":
    main()
