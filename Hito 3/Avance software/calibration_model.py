
# ===============================================
# LEXIIII - CALIBRACION OCULAR PERSONALIZADA
# Caracteristicas de los ojos -> coordenadas X,Y
# ===============================================

import numpy as np
import json

from pathlib import Path


class CalibrationModel:

    def __init__(self, regularization=1.0):

        self.regularization = regularization

        self.feature_mean = None
        self.feature_std = None
        self.coefficients = None
        self.screen_size = None

        self.trained = False

    # -------------------------------------------
    # VALIDAR CARACTERISTICAS OCULARES
    # -------------------------------------------

    def check_features(self, features):

        values = np.asarray(features, dtype=float)

        if values.shape != (4,):
            raise ValueError(
                "Se necesitan 4 caracteristicas oculares"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Se detectaron mediciones invalidas"
            )

        return values

    # -------------------------------------------
    # ENTRENAR CON PUNTOS CONOCIDOS
    # -------------------------------------------

    def fit(self, samples, screen_size):

        width, height = screen_size

        if width <= 0 or height <= 0:
            raise ValueError("Pantalla invalida")

        if len(samples) < 9:
            raise ValueError(
                "Faltan muestras para calibrar"
            )

        X = np.array([
            self.check_features(s["features"])
            for s in samples
        ])

        Y = np.array([
            [
                s["x"] / width,
                s["y"] / height
            ]
            for s in samples
        ], dtype=float)

        if not np.all(np.isfinite(Y)):
            raise ValueError(
                "Coordenadas invalidas"
            )

        if np.any(Y < 0) or np.any(Y > 1):
            raise ValueError(
                "Puntos fuera de pantalla"
            )

        unique_points = np.unique(
            np.round(Y, decimals=6),
            axis=0
        )

        if len(unique_points) < 5:
            raise ValueError(
                "Se necesitan al menos 5 puntos distintos"
            )

        # Normalizar las mediciones oculares.

        mean = X.mean(axis=0)

        std = np.maximum(
            X.std(axis=0),
            1e-5
        )

        normalized = (X - mean) / std

        # Matriz con termino independiente.

        A = np.column_stack([
            np.ones(len(normalized)),
            normalized
        ])

        # Ajuste regularizado.
        # Penalizamos los coeficientes, no el sesgo.

        penalty = (
            np.eye(A.shape[1])
            * self.regularization
        )

        penalty[0, 0] = 0

        coefficients = np.linalg.solve(
            A.T @ A + penalty,
            A.T @ Y
        )

        self.feature_mean = mean
        self.feature_std = std

        self.coefficients = coefficients

        self.screen_size = (
            int(width),
            int(height)
        )

        self.trained = True

        print(
            "[LEXIIII] Calibracion entrenada"
        )

    # -------------------------------------------
    # PREDECIR POSICION CONTINUA X,Y
    # -------------------------------------------

    def predict(self, features, clamp=False):

        if not self.trained:
            raise RuntimeError(
                "Primero debes calibrar"
            )

        values = self.check_features(
            features
        )

        normalized = (
            values - self.feature_mean
        ) / self.feature_std

        vector = np.concatenate([
            [1.0],
            normalized
        ])

        xy_normalized = (
            vector @ self.coefficients
        )

        width, height = self.screen_size

        x = float(
            xy_normalized[0] * width
        )

        y = float(
            xy_normalized[1] * height
        )

        if clamp:
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)

        return float(x), float(y)

    # -------------------------------------------
    # EVALUAR CON PUNTOS NO ENTRENADOS
    # -------------------------------------------

    def validate(self, samples):

        if not samples:
            raise ValueError(
                "No hay datos de validacion"
            )

        errors = []

        for sample in samples:

            pred_x, pred_y = self.predict(
                sample["features"]
            )

            real_x = sample["x"]
            real_y = sample["y"]

            error = np.hypot(
                pred_x - real_x,
                pred_y - real_y
            )

            errors.append(error)

        return {
            "samples": len(errors),
            "mean_error_px": round(
                float(np.mean(errors)), 2
            ),
            "median_error_px": round(
                float(np.median(errors)), 2
            ),
            "p90_error_px": round(
                float(np.percentile(errors, 90)), 2
            ),
            "max_error_px": round(
                float(np.max(errors)), 2
            )
        }

    # -------------------------------------------
    # GUARDAR MODELO DE LA SESION
    # -------------------------------------------

    def save(self, filename):

        if not self.trained:
            raise RuntimeError(
                "No hay modelo entrenado"
            )

        data = {
            "regularization": self.regularization,
            "screen_size": self.screen_size,
            "mean": self.feature_mean.tolist(),
            "std": self.feature_std.tolist(),
            "coefficients": self.coefficients.tolist()
        }

        path = Path(filename)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )


# ===============================================
# PRUEBA SIN CAMARA - DATOS SIMULADOS
# ===============================================

if __name__ == "__main__":

    rng = np.random.default_rng(17)

    WIDTH = 1024
    HEIGHT = 600

    training = []
    validation = []

    def simulate_eyes(x, y):

        nx = x / WIDTH
        ny = y / HEIGHT

        return [
            0.35 + 0.24 * nx
            + rng.normal(0, 0.001),

            0.38 + 0.22 * nx
            + rng.normal(0, 0.001),

            0.10 - 0.17 * ny
            + rng.normal(0, 0.001),

            0.14 - 0.19 * ny
            + rng.normal(0, 0.001)
        ]

    # Puntos conocidos para entrenamiento.

    for y in (100, 300, 500):
        for x in (120, 512, 904):

            for _ in range(15):

                training.append({
                    "features": simulate_eyes(x, y),
                    "x": x,
                    "y": y
                })

    # Otros puntos para validacion.
    # NO participan en el entrenamiento.

    test_points = [
        (310, 190),
        (720, 200),
        (300, 420),
        (715, 425)
    ]

    for x, y in test_points:
        for _ in range(5):

            validation.append({
                "features": simulate_eyes(x, y),
                "x": x,
                "y": y
            })

    model = CalibrationModel()

    model.fit(
        training,
        (WIDTH, HEIGHT)
    )

    result = model.predict(
        simulate_eyes(512, 300)
    )

    print()
    print("===== LEXIIII =====")
    print("Modelo entrenado:", model.trained)
    print("Coordenadas estimadas:", result)
    print("Validacion:", model.validate(validation))
