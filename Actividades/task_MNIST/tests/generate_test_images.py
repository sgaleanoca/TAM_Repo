"""
Generador de imágenes manuscritas propias (externas a MNIST) para evaluación experimental.

Importante: Ninguna de las imágenes generadas aquí proviene del dataset MNIST (`x_train` ni `x_test`).
Todas se trazan sintéticamente mediante trayectorias caligráficas vectoriales con interpolación
suave y variaciones controladas de tamaño, grosor de trazo, color/textura de fondo, contraste
y separación inter-dígito.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "tests" / "sample_images"


@dataclass
class SampleTestCase:
    """Metadatos de un caso de prueba manuscrito."""

    filename: str
    category: str
    expected_number: int | None  # None si se espera un error controlado (p. ej. 3 dígitos o vacía)
    description: str
    expect_error: bool = False


# Definición de trazos caligráficos normalizados en coordenadas [0.0, 1.0] x [0.0, 1.0]
# Cada dígito es una lista de trazos (polilíneas) que emulan la escritura manual humana.
HANDWRITTEN_STROKES: Dict[int, List[List[Tuple[float, float]]]] = {
    0: [
        [
            (0.50, 0.08),
            (0.26, 0.16),
            (0.15, 0.38),
            (0.15, 0.65),
            (0.26, 0.86),
            (0.50, 0.92),
            (0.74, 0.85),
            (0.84, 0.63),
            (0.83, 0.36),
            (0.72, 0.15),
            (0.50, 0.08),
        ]
    ],
    1: [
        [
            (0.52, 0.08),
            (0.51, 0.35),
            (0.50, 0.65),
            (0.49, 0.92),
        ]
    ],
    2: [
        [
            (0.20, 0.26),
            (0.32, 0.12),
            (0.55, 0.10),
            (0.76, 0.18),
            (0.80, 0.34),
            (0.68, 0.52),
            (0.42, 0.72),
            (0.18, 0.89),
            (0.48, 0.89),
            (0.84, 0.88),
        ]
    ],
    3: [
        [
            (0.20, 0.16),
            (0.46, 0.09),
            (0.74, 0.15),
            (0.80, 0.30),
            (0.66, 0.45),
            (0.42, 0.48),
            (0.68, 0.52),
            (0.82, 0.68),
            (0.74, 0.85),
            (0.46, 0.91),
            (0.18, 0.83),
        ]
    ],
    4: [
        [
            (0.24, 0.12),
            (0.20, 0.38),
            (0.18, 0.56),
            (0.52, 0.56),
            (0.84, 0.55),
        ],
        [
            (0.64, 0.10),
            (0.63, 0.50),
            (0.62, 0.92),
        ],
    ],
    5: [
        [
            (0.78, 0.12),
            (0.32, 0.13),
            (0.25, 0.45),
            (0.52, 0.42),
            (0.76, 0.50),
            (0.82, 0.68),
            (0.70, 0.85),
            (0.44, 0.91),
            (0.18, 0.84),
        ]
    ],
    6: [
        [
            (0.70, 0.10),
            (0.45, 0.22),
            (0.24, 0.44),
            (0.18, 0.68),
            (0.28, 0.86),
            (0.52, 0.91),
            (0.76, 0.83),
            (0.80, 0.65),
            (0.66, 0.52),
            (0.40, 0.52),
            (0.20, 0.64),
        ]
    ],
    7: [
        [
            (0.16, 0.14),
            (0.50, 0.13),
            (0.84, 0.13),
            (0.68, 0.42),
            (0.52, 0.68),
            (0.40, 0.92),
        ]
    ],
    8: [
        [
            (0.50, 0.48),
            (0.26, 0.36),
            (0.24, 0.20),
            (0.48, 0.10),
            (0.74, 0.18),
            (0.74, 0.35),
            (0.50, 0.48),
            (0.22, 0.62),
            (0.20, 0.80),
            (0.48, 0.91),
            (0.78, 0.82),
            (0.78, 0.62),
            (0.50, 0.48),
        ]
    ],
    9: [
        [
            (0.76, 0.34),
            (0.58, 0.48),
            (0.32, 0.46),
            (0.20, 0.32),
            (0.28, 0.14),
            (0.52, 0.10),
            (0.74, 0.18),
            (0.76, 0.34),
            (0.68, 0.64),
            (0.54, 0.91),
        ]
    ],
}


def _interpolate_smooth_stroke(
    points: List[Tuple[float, float]],
    steps_per_segment: int = 12,
) -> np.ndarray:
    """Interpola suavemente una polilínea mediante interpolación de Chaikin + muestreo denso."""
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return pts

    # Dos iteraciones de suavizado de esquinas de Chaikin conservando extremos
    for _ in range(2):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = np.array(new_pts, dtype=np.float32)

    dense_pts = []
    for i in range(len(pts) - 1):
        for t in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            dense_pts.append((1.0 - t) * pts[i] + t * pts[i + 1])
    dense_pts.append(pts[-1])
    return np.array(dense_pts, dtype=np.float32)


def draw_handwritten_digit(
    canvas: np.ndarray,
    digit: int,
    top_left: Tuple[int, int],
    box_size: Tuple[int, int],
    stroke_color: Tuple[int, int, int],
    thickness: int,
    jitter_seed: int = 0,
) -> None:
    """Dibuja un dígito manuscrito sobre `canvas` dentro de la región especificada."""
    rng = np.random.default_rng(jitter_seed + digit * 37)
    x0, y0 = top_left
    bw, bh = box_size

    for stroke in HANDWRITTEN_STROKES[digit]:
        jittered = []
        for idx, (nx, ny) in enumerate(stroke):
            # Pequeña variación orgánica para simular pulso humano
            jx = nx + float(rng.normal(0.0, 0.008)) if 0 < idx < len(stroke) - 1 else nx
            jy = ny + float(rng.normal(0.0, 0.008)) if 0 < idx < len(stroke) - 1 else ny
            jittered.append((jx, jy))

        smooth = _interpolate_smooth_stroke(jittered)
        pixel_pts = np.zeros((len(smooth), 1, 2), dtype=np.int32)
        pixel_pts[:, 0, 0] = np.round(x0 + smooth[:, 0] * bw).astype(np.int32)
        pixel_pts[:, 0, 1] = np.round(y0 + smooth[:, 1] * bh).astype(np.int32)

        cv2.polylines(
            canvas,
            [pixel_pts],
            isClosed=False,
            color=stroke_color,
            thickness=max(1, thickness),
            lineType=cv2.LINE_AA,
        )


def render_handwritten_number_image(
    digits: List[int],
    canvas_size: Tuple[int, int] = (240, 160),
    digit_box: Tuple[int, int] = (68, 96),
    spacing: int = 24,
    thickness: int = 9,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    stroke_color: Tuple[int, int, int] = (15, 15, 15),
    add_gradient_or_noise: str | None = None,
    bridge_digits: bool = False,
    seed: int = 100,
) -> np.ndarray:
    """
    Crea una imagen RGB con los dígitos indicados y parámetros de estilo configurables.
    """
    cw, ch = canvas_size
    dw, dh = digit_box
    canvas = np.full((ch, cw, 3), bg_color, dtype=np.uint8)

    if add_gradient_or_noise == "paper_texture":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 6.0, size=(ch, cw, 1))
        grad_x = np.linspace(-12, 12, cw).reshape(1, cw, 1)
        textured = np.clip(canvas.astype(np.float32) + noise + grad_x, 0, 255).astype(np.uint8)
        canvas = textured

    n = len(digits)
    if n > 0:
        total_w = n * dw + (n - 1) * spacing
        start_x = (cw - total_w) // 2
        start_y = (ch - dh) // 2

        boxes = []
        for idx, d in enumerate(digits):
            x0 = start_x + idx * (dw + spacing)
            y0 = start_y
            draw_handwritten_digit(
                canvas=canvas,
                digit=d,
                top_left=(x0, y0),
                box_size=(dw, dh),
                stroke_color=stroke_color,
                thickness=thickness,
                jitter_seed=seed + idx * 13,
            )
            boxes.append((x0, y0, dw, dh))

        # Si se solicita conectar ligeramente los dos dígitos (caso de contacto / ligadura fina)
        if bridge_digits and len(boxes) == 2:
            b1, b2 = boxes[0], boxes[1]
            y_left = b1[1] + int(0.87 * dh)
            y_right = b2[1] + int(0.84 * dh)
            x_left = b1[0] + int(0.80 * dw)
            x_right = b2[0] + int(0.20 * dw)
            cv2.line(
                canvas,
                (x_left, y_left),
                (x_right, y_right),
                stroke_color,
                max(3, thickness // 3),
                cv2.LINE_AA,
            )

    return canvas


def generate_all_test_images(output_dir: Path = SAMPLE_IMAGES_DIR) -> List[SampleTestCase]:
    """
    Genera el conjunto completo de imágenes manuscritas de prueba cubriendo las 6 categorías
    solicitadas más casos de validación de errores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: List[SampleTestCase] = []

    # 1. Dígitos individuales (0 al 9)
    for d in [0, 3, 5, 7, 8, 9]:
        fname = f"single_digit_{d}.png"
        img = render_handwritten_number_image(
            digits=[d],
            canvas_size=(180, 160),
            digit_box=(72, 100),
            thickness=10,
            seed=200 + d,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="1. Dígitos individuales",
                expected_number=d,
                description=f"Dígito individual '{d}' en trazo negro sobre fondo blanco.",
            )
        )

    # 2. Números de dos cifras (ej. 47, 12, 25, 68, 90, 34)
    for num in [47, 12, 25, 68, 90, 34]:
        d1, d2 = num // 10, num % 10
        fname = f"two_digits_{num}.png"
        img = render_handwritten_number_image(
            digits=[d1, d2],
            canvas_size=(260, 160),
            digit_box=(68, 98),
            spacing=26,
            thickness=10,
            seed=300 + num,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="2. Números de dos cifras",
                expected_number=num,
                description=f"Número de dos cifras '{num}' con separación estándar.",
            )
        )

    # 3. Diferentes tamaños de escritura (pequeño, mediano, grande)
    size_configs = [
        ("size_small_42.png", [4, 2], 42, (260, 180), (34, 48), 16, 5, "Escritura pequeña (48px alto)"),
        ("size_large_85.png", [8, 5], 85, (320, 220), (110, 160), 34, 16, "Escritura grande (160px alto)"),
        ("size_tiny_single_6.png", [6], 6, (200, 200), (28, 40), 10, 4, "Dígito individual pequeño (40px alto)"),
    ]
    for fname, digs, expected, csize, dbox, sp, th, desc in size_configs:
        img = render_handwritten_number_image(
            digits=digs,
            canvas_size=csize,
            digit_box=dbox,
            spacing=sp,
            thickness=th,
            seed=400 + expected,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="3. Diferentes tamaños de escritura",
                expected_number=expected,
                description=desc,
            )
        )

    # 4. Diferentes grosores de trazos (fino, medio, grueso)
    thickness_configs = [
        ("stroke_thin_73.png", [7, 3], 73, 3, "Trazo fino de bolígrafo (3px)"),
        ("stroke_medium_59.png", [5, 9], 59, 9, "Trazo medio estándar (9px)"),
        ("stroke_thick_24.png", [2, 4], 24, 15, "Trazo grueso de marcador (15px)"),
    ]
    for fname, digs, expected, th, desc in thickness_configs:
        img = render_handwritten_number_image(
            digits=digs,
            canvas_size=(260, 160),
            digit_box=(68, 98),
            spacing=26,
            thickness=th,
            seed=500 + expected,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="4. Diferentes grosores de trazos",
                expected_number=expected,
                description=desc,
            )
        )

    # 5. Diferentes fondos y contrastes (fondo oscuro, papel grisáceo con textura, bajo contraste)
    bg_configs = [
        (
            "bg_dark_white_ink_38.png",
            [3, 8],
            38,
            (18, 20, 25),
            (240, 240, 240),
            None,
            "Fondo oscuro tipo pizarra con trazo blanco",
        ),
        (
            "bg_paper_texture_64.png",
            [6, 4],
            64,
            (225, 222, 210),
            (35, 35, 45),
            "paper_texture",
            "Fondo tipo papel con gradiente de iluminación y ruido",
        ),
        (
            "bg_low_contrast_19.png",
            [1, 9],
            19,
            (210, 210, 210),
            (110, 110, 110),
            None,
            "Contraste reducido (trazo gris medio sobre fondo gris claro)",
        ),
    ]
    for fname, digs, expected, bg_c, fg_c, tex, desc in bg_configs:
        img = render_handwritten_number_image(
            digits=digs,
            canvas_size=(260, 160),
            digit_box=(68, 98),
            spacing=26,
            thickness=10,
            bg_color=bg_c,
            stroke_color=fg_c,
            add_gradient_or_noise=tex,
            seed=600 + expected,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="5. Diferentes fondos y contrastes",
                expected_number=expected,
                description=desc,
            )
        )

    # 6. Números cuyos dígitos están próximos entre sí (casi tocándose o con contacto ligero)
    close_configs = [
        ("close_digits_47.png", [4, 7], 47, 4, False, "Dígitos '4' y '7' muy próximos (separación de 4px)"),
        ("close_touching_25.png", [2, 5], 25, -4, True, "Dígitos '2' y '5' en contacto ligero (separados por proyección vertical)"),
        ("close_overlapping_hard_88.png", [8, 8], 88, -22, True, "Caso límite extremo: dos '8' fuertemente solapados"),
    ]
    for fname, digs, expected, sp, bridge, desc in close_configs:
        img = render_handwritten_number_image(
            digits=digs,
            canvas_size=(260, 160),
            digit_box=(68, 98),
            spacing=sp,
            thickness=10,
            bridge_digits=bridge,
            seed=700 + expected,
        )
        cv2.imwrite(str(output_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cases.append(
            SampleTestCase(
                filename=fname,
                category="6. Dígitos próximos o en contacto",
                expected_number=expected,
                description=desc,
            )
        )

    # 7. Casos de validación de errores (imagen en blanco y número de 3 cifras > 99)
    blank_img = np.full((160, 240, 3), 250, dtype=np.uint8)
    cv2.imwrite(str(output_dir / "invalid_blank.png"), blank_img)
    cases.append(
        SampleTestCase(
            filename="invalid_blank.png",
            category="7. Validación de errores",
            expected_number=None,
            description="Lienzo vacío sin contraste (debe rechazar con error controlado)",
            expect_error=True,
        )
    )

    three_digits_img = render_handwritten_number_image(
        digits=[1, 2, 3],
        canvas_size=(320, 160),
        digit_box=(64, 96),
        spacing=22,
        thickness=10,
        seed=800,
    )
    cv2.imwrite(
        str(output_dir / "invalid_three_digits_123.png"),
        cv2.cvtColor(three_digits_img, cv2.COLOR_RGB2BGR),
    )
    cases.append(
        SampleTestCase(
            filename="invalid_three_digits_123.png",
            category="7. Validación de errores",
            expected_number=None,
            description="Número de 3 cifras '123' fuera del rango 0-99 (debe rechazar con error controlado)",
            expect_error=True,
        )
    )

    return cases


if __name__ == "__main__":
    generated = generate_all_test_images()
    print(f"Se generaron {len(generated)} imágenes de prueba en '{SAMPLE_IMAGES_DIR}'.")
