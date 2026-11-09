"""
Pruebas unitarias e integrales automatizadas (pytest) para todos los módulos del proyecto.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import predict_number_from_image
from src.model_manager import DEFAULT_MODEL_PATH, build_original_model, load_trained_model
from src.preprocessing import (
    ImageProcessingError,
    format_digit_for_mnist,
    preprocess_full_image,
)
from src.segmentation import SegmentationError, segment_digits
from tests.generate_test_images import SAMPLE_IMAGES_DIR, generate_all_test_images


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_images():
    generate_all_test_images(SAMPLE_IMAGES_DIR)


@pytest.fixture(scope="session")
def trained_model():
    return load_trained_model(DEFAULT_MODEL_PATH)


def test_model_architecture_matches_original_notebook(trained_model):
    """Verifica que el modelo cargado conserva exactamente la arquitectura original de 55,050 parámetros."""
    assert trained_model.input_shape == (None, 784)
    assert trained_model.output_shape == (None, 10)
    assert trained_model.count_params() == 55050

    dense_layers = [layer for layer in trained_model.layers if hasattr(layer, "units")]
    assert [layer.units for layer in dense_layers] == [64, 64, 10]


def test_format_digit_for_mnist_preserves_unnormalized_range():
    """Comprueba que el formateo a MNIST produce (28, 28) y (1, 784) en [0, 255] sin dividir entre 255."""
    synthetic_crop = np.zeros((60, 40), dtype=np.uint8)
    synthetic_crop[10:50, 15:25] = 200
    canvas_28, vec_784 = format_digit_for_mnist(synthetic_crop)

    assert canvas_28.shape == (28, 28)
    assert vec_784.shape == (1, 784)
    assert float(vec_784.max()) == pytest.approx(255.0, abs=1.0)
    assert float(vec_784.min()) == 0.0


def test_single_and_two_digit_predictions(trained_model):
    """Verifica el reconocimiento de dígitos individuales y de dos cifras (ej. 47)."""
    res_7 = predict_number_from_image(str(SAMPLE_IMAGES_DIR / "single_digit_7.png"), model=trained_model)
    assert res_7.num_digits == 1
    assert res_7.recognized_number == 7

    res_47 = predict_number_from_image(str(SAMPLE_IMAGES_DIR / "two_digits_47.png"), model=trained_model)
    assert res_47.num_digits == 2
    assert [d.predicted_digit for d in res_47.digit_predictions] == [4, 7]
    assert res_47.recognized_number == 47


def test_touching_digits_vertical_projection_split(trained_model):
    """Comprueba que dígitos en contacto ligero ('25') se separan por proyección vertical."""
    res_25 = predict_number_from_image(str(SAMPLE_IMAGES_DIR / "close_touching_25.png"), model=trained_model)
    assert res_25.num_digits == 2
    assert res_25.recognized_number == 25
    assert any(d.segmented_digit.split_by_projection for d in res_25.digit_predictions)


def test_error_handling_on_blank_and_three_digits(trained_model):
    """Verifica que imágenes vacías o con más de 2 dígitos lancen excepciones controladas."""
    with pytest.raises(ImageProcessingError):
        predict_number_from_image(str(SAMPLE_IMAGES_DIR / "invalid_blank.png"), model=trained_model)

    with pytest.raises(SegmentationError):
        predict_number_from_image(
            str(SAMPLE_IMAGES_DIR / "invalid_three_digits_123.png"),
            model=trained_model,
        )


def test_streamlit_app_execution():
    """Verifica mediante AppTest que la interfaz Streamlit carga y reconoce el ejemplo '47' sin errores."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(SAMPLE_IMAGES_DIR.parent.parent / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.radio[0].set_value("Probar con galería de imágenes manuscritas").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "47"

