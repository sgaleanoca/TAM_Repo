"""
Módulo de preprocesamiento de imágenes para compatibilidad con MNIST y el MLP original.

Responsabilidades:
1. Validar la imagen de entrada y convertirla a escala de grises (manejando transparencia RGBA).
2. Detectar e invertir automáticamente el contraste para garantizar fondo negro (0) y trazo blanco (0-255).
3. Eliminar iluminación irregular o ruido de fondo y generar una máscara binaria limpia.
4. Adaptar cada recorte de dígito individual al estándar geométrico y radiométrico de MNIST:
   - Escalado proporcional (conservando relación de aspecto) dentro de una caja de 20x20 píxeles.
   - Centrado mediante centro de masa dentro de un lienzo de 28x28 píxeles.
   - Rango de intensidades en [0, 255] (SIN dividir entre 255, tal como se entrenó el modelo original).
   - Aplanado al vector de entrada (1, 784).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image


class ImageProcessingError(ValueError):
    """Excepción controlada cuando una imagen no es válida o no contiene trazos procesables."""


@dataclass
class PreprocessedImage:
    """Resultado del preprocesamiento global de la imagen de entrada."""

    original_rgb: np.ndarray
    grayscale: np.ndarray
    normalized_gray: np.ndarray
    binary_mask: np.ndarray
    inverted_polarity: bool
    contrast_range: float


def load_and_validate_image(image_source: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
    """
    Carga una imagen desde ruta, PIL.Image o np.ndarray y devuelve un arreglo RGB uint8.
    Si la imagen tiene canal Alfa (RGBA/LA), la compone sobre fondo blanco.
    """
    if isinstance(image_source, (str, bytes)):
        try:
            pil_img = Image.open(image_source)
        except Exception as exc:
            raise ImageProcessingError(f"No se pudo abrir el archivo de imagen: {exc}") from exc
    elif isinstance(image_source, Image.Image):
        pil_img = image_source
    elif isinstance(image_source, np.ndarray):
        arr = image_source
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGBA" if arr.shape[2] == 4 else "RGB"
            pil_img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        else:
            raise ImageProcessingError(f"Dimensiones de imagen no soportadas: {arr.shape}")
    else:
        raise ImageProcessingError("Formato de entrada no válido. Use archivo, PIL.Image o np.ndarray.")

    # Si tiene transparencia (RGBA o LA), componer sobre lienzo blanco
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(white_bg, rgba).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    rgb = np.array(pil_img, dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        raise ImageProcessingError(
            f"La imagen es demasiado pequeña ({w}x{h} px). Se requiere al menos 10x10 píxeles."
        )
    return rgb


def preprocess_full_image(image_source: Union[np.ndarray, Image.Image, str]) -> PreprocessedImage:
    """
    Convierte la imagen a escala de grises, detecta polaridad (fondo claro vs oscuro),
    suaviza ruido y obtiene tanto la imagen de intensidades sobre fondo negro como
    la máscara binaria para segmentación.
    """
    rgb = load_and_validate_image(image_source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Verificar que la imagen tenga suficiente contraste (que no esté en blanco/vacía)
    contrast_range = float(int(gray.max()) - int(gray.min()))
    if contrast_range < 15.0 or float(gray.astype(np.float32).std()) < 1.5:
        raise ImageProcessingError(
            "La imagen carece de contraste suficiente o está vacía. "
            "Asegúrese de que el número sea visible respecto al fondo."
        )

    # Suavizado ligero para reducir ruido de sensor/compresión
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Estimar el color del fondo usando los bordes de la imagen y la mediana global
    border_pixels = np.concatenate(
        [
            blurred[0, :],
            blurred[-1, :],
            blurred[:, 0],
            blurred[:, -1],
        ]
    )
    bg_estimate = float(np.median(border_pixels))
    global_median = float(np.median(blurred))

    # En MNIST el fondo es negro (0) y el dígito es blanco (255).
    # Si el fondo estimado es claro (> 115) o la mediana global es alta, invertimos.
    inverted_polarity = (bg_estimate > 115.0) or (global_median > 127.0)
    if inverted_polarity:
        fg_gray = 255 - blurred
    else:
        fg_gray = blurred.copy()

    # Sustracción morfológica de fondo (Top-Hat) para corregir iluminación irregular o fondos grises
    h, w = fg_gray.shape
    kernel_size = max(15, int(min(h, w) * 0.25) | 1)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    local_bg = cv2.morphologyEx(fg_gray, cv2.MORPH_OPEN, bg_kernel)
    fg_corrected = cv2.subtract(fg_gray, local_bg)

    # Estirar contraste al rango completo [0, 255]
    max_val = int(fg_corrected.max())
    if max_val < 10:
        raise ImageProcessingError(
            "No se detectaron trazos distinguibles del fondo tras normalizar el contraste."
        )
    fg_stretched = np.clip((fg_corrected.astype(np.float32) / max_val) * 255.0, 0, 255).astype(
        np.uint8
    )

    # Umbralización de Otsu combinada con umbral mínimo para evitar amplificar ruido de fondo
    otsu_thresh, _ = cv2.threshold(fg_stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    effective_thresh = max(float(otsu_thresh), 55.0)
    _, binary_mask = cv2.threshold(fg_stretched, effective_thresh, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica de puntos aislados diminutos (solo si la imagen es suficientemente grande)
    if min(h, w) >= 50:
        clean_kernel = np.ones((2, 2), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, clean_kernel)

    # Limpiar el fondo en la imagen de intensidades usando la máscara dilatada suavemente
    mask_dilated = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
    normalized_gray = np.where(mask_dilated > 0, fg_stretched, 0).astype(np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        raise ImageProcessingError("No se encontraron trazos válidos en la imagen.")

    return PreprocessedImage(
        original_rgb=rgb,
        grayscale=gray,
        normalized_gray=normalized_gray,
        binary_mask=binary_mask,
        inverted_polarity=inverted_polarity,
        contrast_range=contrast_range,
    )


def _compute_center_of_mass(img_28x28: np.ndarray) -> Tuple[float, float]:
    """Calcula el centro de masa (cy, cx) ponderado por intensidad en una imagen 2D."""
    arr = img_28x28.astype(np.float64)
    total_mass = float(arr.sum())
    if total_mass <= 1e-6:
        h, w = arr.shape
        return h / 2.0, w / 2.0
    y_indices, x_indices = np.indices(arr.shape, dtype=np.float64)
    cy = float((y_indices * arr).sum() / total_mass)
    cx = float((x_indices * arr).sum() / total_mass)
    return cy, cx


def format_digit_for_mnist(
    digit_gray: np.ndarray,
    digit_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adapta el recorte de un dígito individual al formato exacto esperado por el modelo MNIST:
    1. Recorta la caja delimitadora ajustada del dígito.
    2. Regula el grosor del trazo (engrosa trazos ultra-finos en imágenes de alta resolución
       o adelgaza trazos excesivamente gruesos que cerrarían los ojales de dígitos como 6, 8, 9, 0).
    3. Escala proporcionalmente para que el lado mayor mida 20 píxeles (caja 20x20).
    4. Inserta en un lienzo negro de 28x28 y alinea su centro de masa con el centro (13.5, 13.5).
    5. Normaliza intensidades al rango [0, 255] (sin dividir entre 255) y aplana a (1, 784).

    Retorna:
        - canvas_28x28: np.ndarray de forma (28, 28) en uint8 [0, 255] para visualización.
        - vector_784: np.ndarray de forma (1, 784) en float32 [0.0, 255.0] para inferencia.
    """
    work_gray = digit_gray.copy()
    if digit_mask is not None:
        work_gray = np.where(digit_mask > 0, work_gray, 0).astype(np.uint8)

    # Recortar bordes vacíos si existen
    coords = cv2.findNonZero(work_gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        work_gray = work_gray[y : y + h, x : x + w]
        if digit_mask is not None:
            digit_mask = digit_mask[y : y + h, x : x + w]

    h, w = work_gray.shape[:2]
    max_dim = max(h, w)
    if max_dim == 0:
        empty = np.zeros((28, 28), dtype=np.uint8)
        return empty, empty.reshape(1, 784).astype(np.float32)

    # Regular el grosor del trazo antes de redimensionar a 20x20:
    # En MNIST, el grosor típico del trazo en la caja 20x20 es ~2.2-2.8 px (~12-14% de max_dim).
    active_pixels = np.count_nonzero(work_gray > 30)
    fill_ratio = active_pixels / float(max(1, h * w))

    # Estimar grosor medio mediante transformada de distancia
    bin_crop = (work_gray > 40).astype(np.uint8) * 255
    dist = cv2.distanceTransform(bin_crop, cv2.DIST_L2, 3)
    mean_radius = float(dist[dist > 0].mean()) if np.any(dist > 0) else 1.0
    estimated_stroke_in_20 = (2.0 * mean_radius) * (20.0 / max_dim)

    if estimated_stroke_in_20 < 1.8 and max_dim > 24:
        # El trazo es demasiado fino y perdería energía al submuestrear a 20x20
        target_radius_px = max(1, int(round((1.15 * max_dim / 20.0) - mean_radius)))
        ksize = max(2, min(7, 2 * target_radius_px + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        work_gray = cv2.dilate(work_gray, kernel, iterations=1)
    elif estimated_stroke_in_20 > 3.8 and fill_ratio > 0.50 and max_dim > 24:
        # El trazo es excesivamente grueso y podría colapsar cavidades internas al reducir
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(work_gray, kernel, iterations=1)
        if np.count_nonzero(eroded > 30) > 0.2 * active_pixels:
            work_gray = eroded

    # Escalar conservando relación de aspecto a caja máxima de 20x20
    scale = 20.0 / float(max_dim)
    new_w = max(1, min(20, int(round(w * scale))))
    new_h = max(1, min(20, int(round(h * scale))))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(work_gray, (new_w, new_h), interpolation=interp)

    # Colocar inicialmente centrado geométricamente en el lienzo de 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized.astype(np.float32)

    # Alinear el centro de masa al centro del lienzo (13.5, 13.5) como en MNIST
    cy, cx = _compute_center_of_mass(canvas)
    shift_y = int(round(13.5 - cy))
    shift_x = int(round(13.5 - cx))

    # Limitar el desplazamiento para nunca recortar píxeles activos del dígito
    nz_y, nz_x = np.nonzero(canvas > 15)
    if len(nz_y) > 0:
        min_shift_y = -int(nz_y.min())
        max_shift_y = 27 - int(nz_y.max())
        min_shift_x = -int(nz_x.min())
        max_shift_x = 27 - int(nz_x.max())
        shift_y = int(np.clip(shift_y, min_shift_y, max_shift_y))
        shift_x = int(np.clip(shift_x, min_shift_x, max_shift_x))

    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    canvas = cv2.warpAffine(
        canvas,
        translation_matrix,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Re-escalar intensidades para que el máximo sea 255 y el fondo sea 0 (estándar MNIST [0, 255])
    c_max = float(canvas.max())
    if c_max > 0:
        canvas = (canvas / c_max) * 255.0
    canvas[canvas < 12.0] = 0.0

    canvas_uint8 = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    vector_784 = canvas_uint8.reshape(1, 28 * 28).astype(np.float32)
    return canvas_uint8, vector_784
