"""
Módulo de gestión del modelo (Carga, Entrenamiento y Persistencia).

Replica fielmente la arquitectura y el flujo de entrenamiento definidos en el
cuaderno original `Clase_05_11_26_TAM.ipynb` y persiste el modelo en formato
`.keras` para su reutilización sin necesidad de reentrenar en cada ejecución.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mnist_mlp.keras"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"


def build_original_model() -> tf.keras.Model:
    """
    Construye exactamente la arquitectura definida en `Clase_05_11_26_TAM.ipynb`:
    - Input(shape=(784,))
    - Dense(64, activation='relu')
    - Dense(64, activation='relu')
    - Dense(10, activation='softmax')
    - Compilado con optimizer='adam', loss='sparse_categorical_crossentropy'
    """
    In = tf.keras.layers.Input(shape=(784,))
    Cp1 = tf.keras.layers.Dense(64, activation="relu")(In)
    Cp2 = tf.keras.layers.Dense(64, activation="relu")(Cp1)
    Out = tf.keras.layers.Dense(10, activation="softmax")(Cp2)
    model = tf.keras.Model(inputs=In, outputs=Out)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_original_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carga y preprocesa MNIST exactamente como en el cuaderno original:
    - Aplanado a vectores de 784 elementos (28 * 28).
    - Conservando las intensidades originales en el rango [0, 255] (sin dividir entre 255).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    return (x_train, y_train), (x_test, y_test)


def train_and_save_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta el entrenamiento original de 30 épocas sobre MNIST y guarda el
    modelo en formato `.keras` junto con sus métricas de evaluación.
    """
    tf.keras.utils.set_random_seed(seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_original_mnist_data()
    model = build_original_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(model_path)

    metrics_summary: Dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "total_params": int(model.count_params()),
        "input_intensity_range": [int(x_train.min()), int(x_train.max())],
        "normalized_0_1": False,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    return metrics_summary


def load_trained_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    auto_train_if_missing: bool = False,
) -> tf.keras.Model:
    """
    Carga el modelo entrenado desde `models/mnist_mlp.keras`.
    Si no existe y `auto_train_if_missing=True`, ejecuta el entrenamiento original.
    """
    if not model_path.exists():
        if auto_train_if_missing:
            train_and_save_model(model_path=model_path)
        else:
            raise FileNotFoundError(
                f"No se encontró el modelo guardado en '{model_path}'. "
                "Ejecute `python -m src.model_manager` para entrenarlo y guardarlo."
            )
    return tf.keras.models.load_model(model_path)


def get_training_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> Dict[str, Any]:
    """Devuelve las métricas registradas del entrenamiento si están disponibles."""
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    print("Iniciando entrenamiento fiel al cuaderno original (30 épocas)...")
    summary = train_and_save_model()
    print("\nEntrenamiento finalizado y modelo guardado exitosamente:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
