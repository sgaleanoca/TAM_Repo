"""
Script de evaluación sistemática sobre las imágenes manuscritas propias.

Ejecuta el pipeline completo sobre todas las categorías de prueba, registra los
resultados detallados (incluyendo probabilidades Softmax y advertencias) y genera
tanto `tests/test_results.json` como `tests/REPORTE_PRUEBAS.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.inference import predict_number_from_image
from src.model_manager import load_trained_model
from src.preprocessing import ImageProcessingError
from src.segmentation import SegmentationError
from tests.generate_test_images import SAMPLE_IMAGES_DIR, generate_all_test_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON_PATH = PROJECT_ROOT / "tests" / "test_results.json"
REPORT_MD_PATH = PROJECT_ROOT / "tests" / "REPORTE_PRUEBAS.md"


def run_evaluation_suite() -> Dict[str, Any]:
    cases = generate_all_test_images(SAMPLE_IMAGES_DIR)
    model = load_trained_model()

    records: List[Dict[str, Any]] = []
    valid_cases = 0
    correct_valid_cases = 0
    error_cases_total = 0
    error_cases_caught = 0

    for case in cases:
        img_path = SAMPLE_IMAGES_DIR / case.filename
        record: Dict[str, Any] = {
            "filename": case.filename,
            "category": case.category,
            "description": case.description,
            "expected_number": case.expected_number,
            "expect_error": case.expect_error,
        }

        try:
            res = predict_number_from_image(str(img_path), model=model)
            record["status"] = "success"
            record["predicted_number"] = res.recognized_number
            record["num_digits_detected"] = res.num_digits
            record["joint_confidence"] = round(res.joint_confidence, 4)
            record["digit_details"] = [
                {
                    "role": dp.role_label,
                    "predicted_digit": dp.predicted_digit,
                    "confidence": round(dp.confidence, 4),
                    "split_by_projection": dp.segmented_digit.split_by_projection,
                }
                for dp in res.digit_predictions
            ]
            record["warnings"] = res.warnings

            if case.expect_error:
                error_cases_total += 1
                record["correct"] = False
            else:
                valid_cases += 1
                is_correct = res.recognized_number == case.expected_number
                record["correct"] = is_correct
                if is_correct:
                    correct_valid_cases += 1

        except (ImageProcessingError, SegmentationError) as exc:
            record["status"] = "controlled_error"
            record["error_message"] = str(exc)
            if case.expect_error:
                error_cases_total += 1
                error_cases_caught += 1
                record["correct"] = True
            else:
                valid_cases += 1
                record["correct"] = False

        records.append(record)

    summary = {
        "total_cases": len(cases),
        "valid_handwritten_cases": valid_cases,
        "correct_predictions": correct_valid_cases,
        "accuracy_on_valid_cases": round(correct_valid_cases / max(1, valid_cases), 4),
        "error_validation_cases": error_cases_total,
        "error_validation_caught": error_cases_caught,
        "records": records,
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_markdown_report(summary)
    return summary


def _write_markdown_report(summary: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# Reporte de Pruebas Experimentales — Imágenes Manuscritas Propias",
        "",
        "Este documento registra los resultados de evaluar el pipeline de preprocesamiento,",
        "segmentación e inferencia con el modelo original (`models/mnist_mlp.keras`) sobre imágenes",
        "manuscritas sintéticas externas (no pertenecientes al conjunto de entrenamiento ni prueba de MNIST).",
        "",
        "## Resumen Cuantitativo",
        "",
        f"- **Casos manuscritos evaluados (0–99)**: `{summary['valid_handwritten_cases']}`",
        f"- **Predicciones exactas del número completo**: `{summary['correct_predictions']} / {summary['valid_handwritten_cases']}` (`{summary['accuracy_on_valid_cases'] * 100:.1f}%`)",
        f"- **Casos de validación de errores (lienzo vacío / >2 dígitos)**: `{summary['error_validation_caught']} / {summary['error_validation_cases']}` capturados correctamente.",
        "",
        "## Resultados Detallados por Categoría",
        "",
        "| Archivo | Categoría | Descripción | Esperado | Obtenido | Confianza Softmax Conjunta | Estado / Observaciones |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for r in summary["records"]:
        exp_str = str(r["expected_number"]) if r["expected_number"] is not None else "Error esperado"
        if r["status"] == "controlled_error":
            obt_str = "Rechazado"
            conf_str = "N/A"
            obs = f"Error controlado: _{r['error_message']}_"
        else:
            obt_str = str(r["predicted_number"])
            conf_str = f"{r['joint_confidence'] * 100:.2f}%"
            details_str = ", ".join(
                f"D{i+1}={d['predicted_digit']} ({d['confidence']*100:.1f}%)"
                for i, d in enumerate(r["digit_details"])
            )
            warn_str = f" | Nota: {'; '.join(r['warnings'])}" if r.get("warnings") else ""
            obs = f"{details_str}{warn_str}"

        badge = "✅ Correcto" if r["correct"] else "❌ Fallo / Límite"
        lines.append(
            f"| `{r['filename']}` | {r['category']} | {r['description']} | **{exp_str}** | **{obt_str}** | {conf_str} | {badge} — {obs} |"
        )

    lines.extend(
        [
            "",
            "## Análisis Crítico: Probabilidad Softmax vs. Garantía de Aciertos",
            "",
            "1. **Sobreconfianza del MLP en datos fuera de distribución (Out-of-Distribution)**:",
            "   Como el modelo original fue entrenado con entradas sin normalizar en el rango `[0, 255]` y capas",
            "   `Dense` con activación `ReLU`, las combinaciones lineales producen *logits* de gran magnitud",
            "   ($z_k \\gg 10$). Al aplicar la función $\\text{Softmax}(z)_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$,",
            "   cualquier diferencia moderada entre el mayor logit y el segundo produce probabilidades cercanas al",
            "   `99.9%` o `100.0%`, **incluso cuando la segmentación falla o el dígito es clasificado erróneamente**.",
            "2. **Caso límite de dígitos fuertemente solapados (`close_overlapping_hard_88.png`)**:",
            "   Cuando dos dígitos se superponen físicamente de forma severa sin dejar un valle pronunciado en el",
            "   histograma de proyección vertical, el segmentador no puede separarlos sin cortar trazos estructurales,",
            "   o el corte altera la geometría de cada dígito. En tales casos, la red neuronal aún puede emitir una",
            "   probabilidad Softmax alta para una clase incorrecta.",
            "3. **Sensibilidad geométrica de la arquitectura MLP (Ausencia de capas convolucionales)**:",
            "   Dado que la entrada es un vector plano de `784` píxeles conectado densamente, la red carece de",
            "   invarianza espacial intrínseca. La precisión sobre imágenes propias depende críticamente de que",
            "   el preprocesamiento recorte cada dígito, regule el grosor del trazo, lo escale a una caja de `20x20`",
            "   y alinee su centro de masa en `(13.5, 13.5)` dentro del lienzo `28x28`.",
        ]
    )

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run_evaluation_suite()
    print(
        f"Evaluación completada: {results['correct_predictions']}/{results['valid_handwritten_cases']} "
        f"aciertos en números manuscritos ({results['accuracy_on_valid_cases']*100:.1f}%) y "
        f"{results['error_validation_caught']}/{results['error_validation_cases']} errores controlados."
    )
