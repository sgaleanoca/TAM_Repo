"""
Módulo de inferencia por dígito y reconstrucción aritmética del número (0 al 99).

Integra el flujo completo:
1. Preprocesamiento global de la imagen (`preprocess_full_image`).
2. Segmentación espacial de 1 o 2 dígitos (`segment_digits`).
3. Inferencia independiente mediante el modelo Keras entrenado (`models/mnist_mlp.keras`).
4. Reconstrucción posicional de izquierda a derecha (decenas y unidades).
5. Análisis de incertidumbre sobre la distribución Softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import tensorflow as tf
from PIL import Image

from src.model_manager import load_trained_model
from src.preprocessing import PreprocessedImage, preprocess_full_image
from src.segmentation import SegmentedDigit, SegmentationResult, segment_digits


@dataclass
class DigitPrediction:
    """Predicción individual para un dígito segmentado."""

    position_index: int  # 0 para primer dígito (izquierda), 1 para segundo dígito (derecha)
    role_label: str  # "Dígito único", "Decenas" o "Unidades"
    predicted_digit: int
    confidence: float  # Probabilidad máxima Softmax en [0, 1]
    probabilities: List[float]  # Distribución Softmax de las 10 clases (0..9)
    segmented_digit: SegmentedDigit


@dataclass
class NumberRecognitionResult:
    """Resultado integral del reconocimiento de un número manuscrito entre 0 y 99."""

    recognized_number: int
    formatted_string: str
    num_digits: int
    digit_predictions: List[DigitPrediction]
    joint_confidence: float
    preprocessed: PreprocessedImage
    segmentation: SegmentationResult
    warnings: List[str]


def predict_number_from_image(
    image_source: Union[np.ndarray, Image.Image, str],
    model: tf.keras.Model | None = None,
) -> NumberRecognitionResult:
    """
    Ejecuta el pipeline completo de extremo a extremo sobre una imagen y devuelve
    el número reconocido entre 0 y 99 junto con los artefactos intermedios.
    """
    if model is None:
        model = load_trained_model()

    preprocessed = preprocess_full_image(image_source)
    segmentation = segment_digits(preprocessed)

    num_digits = len(segmentation.digits)
    batch_vectors = np.vstack([d.mnist_vector for d in segmentation.digits])  # (N, 784)

    # Inferencia en lote para los dígitos segmentados (N = 1 o 2)
    raw_probs = model.predict(batch_vectors, verbose=0)  # (N, 10)

    digit_predictions: List[DigitPrediction] = []
    warnings: List[str] = list(segmentation.warnings)

    for idx, seg_digit in enumerate(segmentation.digits):
        probs = [float(p) for p in raw_probs[idx]]
        pred_class = int(np.argmax(probs))
        conf = float(probs[pred_class])

        if num_digits == 1:
            role = "Dígito único (Unidades)"
        else:
            role = "Decenas (Izquierda)" if idx == 0 else "Unidades (Derecha)"

        if conf < 0.75:
            warnings.append(
                f"Confianza moderada/baja en {role}: {conf * 100:.1f}% para el dígito '{pred_class}'. "
                "Verifique las probabilidades secundarias."
            )

        digit_predictions.append(
            DigitPrediction(
                position_index=idx,
                role_label=role,
                predicted_digit=pred_class,
                confidence=conf,
                probabilities=probs,
                segmented_digit=seg_digit,
            )
        )

    # Reconstrucción posicional respetando el orden de lectura (izquierda a derecha)
    recognized_number = 0
    digits_str_parts: List[str] = []
    joint_conf = 1.0

    for dp in digit_predictions:
        recognized_number = recognized_number * 10 + dp.predicted_digit
        digits_str_parts.append(str(dp.predicted_digit))
        joint_conf *= dp.confidence

    return NumberRecognitionResult(
        recognized_number=recognized_number,
        formatted_string="".join(digits_str_parts),
        num_digits=num_digits,
        digit_predictions=digit_predictions,
        joint_confidence=float(joint_conf),
        preprocessed=preprocessed,
        segmentation=segmentation,
        warnings=warnings,
    )
