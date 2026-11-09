"""
Módulo de segmentación de dígitos para números manuscritos entre 0 y 99.

Implementa un pipeline híbrido de segmentación en dos fases:
1. Detección de componentes conexos y filtrado adaptativo de ruido morfológico.
2. Fusión de trazos discontinuos verticalmente superpuestos (p. ej., la barra superior
   de un '5' o un '4' trazado en dos partes).
3. Separación por perfil de proyección vertical (búsqueda de valle mínimo entre picos)
   cuando dos dígitos próximos se tocan ligeramente formando un único componente ancho.
4. Ordenamiento espacial de izquierda a derecha (decenas -> unidades) y validación
   del rango permitido (1 o 2 dígitos para números del 0 al 99).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.preprocessing import PreprocessedImage, format_digit_for_mnist


class SegmentationError(ValueError):
    """Excepción controlada cuando la segmentación no encuentra entre 1 y 2 dígitos válidos."""


@dataclass
class SegmentedDigit:
    """Representa un dígito individual extraído y adaptado al formato MNIST."""

    index: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) en coordenadas de la imagen original
    crop_gray: np.ndarray
    crop_mask: np.ndarray
    mnist_28x28: np.ndarray  # (28, 28) uint8 en [0, 255]
    mnist_vector: np.ndarray  # (1, 784) float32 en [0.0, 255.0]
    split_by_projection: bool = False


@dataclass
class SegmentationResult:
    """Resultado completo del proceso de segmentación."""

    digits: List[SegmentedDigit]
    annotated_rgb: np.ndarray
    warnings: List[str]


def _horizontal_overlap_ratio(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Calcula la fracción de solapamiento horizontal respecto al ancho menor de dos cajas."""
    x1, _, w1, _ = box1
    x2, _, w2, _ = box2
    overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    min_w = max(1, min(w1, w2))
    return overlap / float(min_w)


def _merge_boxes(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Une dos cajas delimitadoras (x, y, w, h) en una sola caja envolvente."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    nx = min(x1, x2)
    ny = min(y1, y2)
    nw = max(x1 + w1, x2 + w2) - nx
    nh = max(y1 + h1, y2 + h2) - ny
    return (nx, ny, nw, nh)


def _try_vertical_projection_split(
    bbox: Tuple[int, int, int, int],
    binary_mask: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Evalúa si una región ancha corresponde a dos dígitos tocándose y, de ser viable,
    la divide en dos cajas delimitadoras usando el valle del histograma de proyección vertical.
    """
    x, y, w, h = bbox
    aspect_ratio = w / float(max(1, h))
    if aspect_ratio < 1.18 or w < 18:
        return [bbox]

    roi_mask = (binary_mask[y : y + h, x : x + w] > 0).astype(np.float32)
    col_profile = roi_mask.sum(axis=0)

    # Suavizar perfil 1D con ventana de 3 elementos
    kernel_1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    smooth_profile = np.convolve(col_profile, kernel_1d, mode="same")

    # Buscar el valle mínimo en la franja central [28%, 72%] del ancho
    left_bound = max(3, int(round(w * 0.28)))
    right_bound = min(w - 3, int(round(w * 0.72)))
    if right_bound <= left_bound:
        return [bbox]

    center_slice = smooth_profile[left_bound:right_bound]
    rel_min_idx = int(np.argmin(center_slice))
    split_col = left_bound + rel_min_idx
    valley_val = float(smooth_profile[split_col])

    left_peak = float(smooth_profile[:split_col].max()) if split_col > 0 else 0.0
    right_peak = float(smooth_profile[split_col + 1 :].max()) if split_col + 1 < w else 0.0
    min_peak = min(left_peak, right_peak)

    # Condición de corte: ambos lados tienen masa significativa y el valle central es profundo
    if min_peak < 2.0 or valley_val > 0.38 * min_peak:
        return [bbox]

    # Ajustar las cajas resultantes izquierda y derecha excluyendo 1px del puente de contacto
    left_end = max(1, split_col - 1)
    right_start = min(w - 1, split_col + 2)
    left_sub = binary_mask[y : y + h, x : x + left_end]
    right_sub = binary_mask[y : y + h, x + right_start : x + w]

    c_left = cv2.findNonZero(left_sub)
    c_right = cv2.findNonZero(right_sub)
    if c_left is None or c_right is None:
        return [bbox]

    lx, ly, lw, lh = cv2.boundingRect(c_left)
    rx, ry, rw, rh = cv2.boundingRect(c_right)

    box_left = (x + lx, y + ly, lw, lh)
    box_right = (x + right_start + rx, y + ry, rw, rh)
    return [box_left, box_right]


def segment_digits(preprocessed: PreprocessedImage) -> SegmentationResult:
    """
    Detecta, separa y normaliza los dígitos presentes en una imagen preprocesada.

    Valida que la imagen contenga exactamente 1 o 2 dígitos para representar números
    en el rango [0, 99].
    """
    mask = preprocessed.binary_mask
    gray = preprocessed.normalized_gray
    img_h, img_w = mask.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise SegmentationError("No se detectaron regiones de dígitos en la imagen.")

    # Extraer componentes candidatos ignorando el fondo (label 0)
    raw_components: List[Tuple[int, int, int, int, int]] = []
    for lbl in range(1, num_labels):
        bx = int(stats[lbl, cv2.CC_STAT_LEFT])
        by = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        raw_components.append((bx, by, bw, bh, area))

    max_area = max(c[4] for c in raw_components)
    max_h = max(c[3] for c in raw_components)

    # Filtrar ruido pequeño (motas, puntos aislados) conservando trazos legítimos
    filtered_boxes: List[Tuple[int, int, int, int]] = []
    for bx, by, bw, bh, area in raw_components:
        if area < max(6, int(0.03 * max_area)) and bh < int(0.28 * max_h):
            continue
        if bw < 2 and bh < 4:
            continue
        filtered_boxes.append((bx, by, bw, bh))

    if not filtered_boxes:
        raise SegmentationError("Los trazos detectados son demasiado pequeños o ruidosos.")

    # Ordenar de izquierda a derecha y fusionar fragmentos superpuestos horizontalmente
    filtered_boxes.sort(key=lambda b: b[0])
    merged_boxes: List[Tuple[int, int, int, int]] = []
    for box in filtered_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue
        prev = merged_boxes[-1]
        if _horizontal_overlap_ratio(prev, box) >= 0.42:
            merged_boxes[-1] = _merge_boxes(prev, box)
        else:
            merged_boxes.append(box)

    warnings: List[str] = []
    final_boxes: List[Tuple[Tuple[int, int, int, int], bool]] = []

    # Si solo hay 1 componente pero es muy ancho, intentar división por proyección vertical
    if len(merged_boxes) == 1:
        split_candidates = _try_vertical_projection_split(merged_boxes[0], mask)
        if len(split_candidates) == 2:
            warnings.append(
                "Se detectaron dos dígitos muy próximos o en contacto; "
                "se separaron mediante perfil de proyección vertical."
            )
            for sb in split_candidates:
                final_boxes.append((sb, True))
        else:
            w, h = merged_boxes[0][2], merged_boxes[0][3]
            if w / float(max(1, h)) > 1.35:
                warnings.append(
                    "Se detectó una única región ancha que no pudo separarse limpiamente. "
                    "Si escribió dos dígitos unidos, intente dejar una ligera separación entre ellos."
                )
            final_boxes.append((merged_boxes[0], False))
    else:
        for mb in merged_boxes:
            final_boxes.append((mb, False))

    # Si hay más de 2 componentes, verificar si alguno es ruido secundario descartable
    if len(final_boxes) > 2:
        box_areas = [b[0][2] * b[0][3] for b in final_boxes]
        largest_two_min = sorted(box_areas, reverse=True)[1]
        pruned = [
            b for b in final_boxes if (b[0][2] * b[0][3]) >= 0.18 * largest_two_min
        ]
        if len(pruned) <= 2:
            final_boxes = pruned
            warnings.append("Se descartaron artefactos secundarios pequeños fuera de los dígitos principales.")
        else:
            raise SegmentationError(
                f"Se detectaron {len(pruned)} dígitos/regiones independientes, pero el sistema "
                "solo admite números entre 0 y 99 (máximo 2 dígitos)."
            )

    # Ordenar estrictamente de izquierda a derecha (orden de lectura: decena -> unidad)
    final_boxes.sort(key=lambda item: item[0][0])

    # Construir objetos SegmentedDigit y anotar imagen RGB
    annotated_rgb = preprocessed.original_rgb.copy()
    segmented_digits: List[SegmentedDigit] = []

    for idx, (bbox, was_split) in enumerate(final_boxes):
        x, y, w, h = bbox
        crop_gray = gray[y : y + h, x : x + w]
        crop_mask = mask[y : y + h, x : x + w]

        mnist_28x28, mnist_vector = format_digit_for_mnist(crop_gray, crop_mask)
        segmented_digits.append(
            SegmentedDigit(
                index=idx,
                bbox=bbox,
                crop_gray=crop_gray,
                crop_mask=crop_mask,
                mnist_28x28=mnist_28x28,
                mnist_vector=mnist_vector,
                split_by_projection=was_split,
            )
        )

        # Dibujar rectángulo y etiqueta sobre la imagen anotada
        color = (0, 180, 90) if not was_split else (230, 140, 0)
        thickness = max(1, min(3, max(img_h, img_w) // 90))
        cv2.rectangle(annotated_rgb, (x, y), (x + w, y + h), color, thickness)
        label_y = max(12, y - 5)
        cv2.putText(
            annotated_rgb,
            f"D{idx + 1}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, min(0.9, max(img_h, img_w) / 220.0)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    return SegmentationResult(
        digits=segmented_digits,
        annotated_rgb=annotated_rgb,
        warnings=warnings,
    )
