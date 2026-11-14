
# ==============================================
# LEXIIII - DETECCION DE FIJACIONES OCULARES
# Entrada: tiempo, coordenadas X/Y en pixeles
# Salida: fijaciones candidatas
# ==============================================

import numpy as np


class FixationDetector:

    def __init__(
        self,
        min_duration_ms=180,
        max_dispersion_px=45,
        max_gap_ms=150,
        min_samples=4
    ):
        self.min_duration = min_duration_ms / 1000
        self.max_dispersion = max_dispersion_px
        self.max_gap = max_gap_ms / 1000
        self.min_samples = min_samples

        self.reset()

    def reset(self):
        self.samples = []
        self.fixations = []

    def _create_fixation(self):

        if len(self.samples) < self.min_samples:
            return None

        times = np.array([
            s[0] for s in self.samples
        ])

        points = np.array([
            [s[1], s[2]]
            for s in self.samples
        ])

        duration = times[-1] - times[0]

        if duration < self.min_duration:
            return None

        center = np.median(
            points,
            axis=0
        )

        distances = np.linalg.norm(
            points - center,
            axis=1
        )

        max_radius = float(
            np.max(distances)
        )

        if max_radius > self.max_dispersion:
            return None

        fixation = {
            "x": round(float(center[0]), 2),
            "y": round(float(center[1]), 2),
            "start_s": round(float(times[0]), 3),
            "end_s": round(float(times[-1]), 3),
            "duration_ms": round(
                float(duration * 1000), 1
            ),
            "samples": len(self.samples),
            "dispersion_px": round(
                max_radius, 2
            )
        }

        self.fixations.append(fixation)

        return fixation

    def flush(self):
        """
        Finaliza la fijacion candidata actual.
        """

        result = self._create_fixation()

        self.samples = []

        return result

    def update(self, timestamp, x=None, y=None):
        """
        Recibe una medicion ocular.

        timestamp: segundos de tiempo monotónico
        x, y: coordenadas continuas en pixeles

        Devuelve una fijacion cuando termina
        un grupo estable de muestras.
        """

        timestamp = float(timestamp)

        if not np.isfinite(timestamp):
            raise ValueError("Tiempo invalido")

        if x is None or y is None:
            return self.flush()

        x = float(x)
        y = float(y)

        if not np.isfinite([x, y]).all():
            return self.flush()

        new_sample = (timestamp, x, y)

        if not self.samples:
            self.samples = [new_sample]
            return None

        last_time = self.samples[-1][0]

        if timestamp <= last_time:
            raise ValueError(
                "Las muestras deben estar "
                "ordenadas en el tiempo"
            )

        # Interrupcion prolongada de la captura.
        if timestamp - last_time > self.max_gap:

            previous = self.flush()

            self.samples = [new_sample]

            return previous

        current_points = np.array([
            [s[1], s[2]]
            for s in self.samples
        ])

        center = np.median(
            current_points,
            axis=0
        )

        distance = np.linalg.norm(
            np.array([x, y]) - center
        )

        # Movimiento demasiado grande:
        # finalizar fijacion anterior.
        if distance > self.max_dispersion:

            previous = self.flush()

            self.samples = [new_sample]

            return previous

        self.samples.append(new_sample)

        return None


# ==============================================
# PRUEBA SIN CAMARA
# ==============================================

if __name__ == "__main__":

    detector = FixationDetector()

    rng = np.random.default_rng(25)

    t = 0.0
    dt = 1 / 30

    detected = []

    # Primera fijacion cerca de (200, 150).
    for _ in range(20):

        x = 200 + rng.normal(0, 3)
        y = 150 + rng.normal(0, 3)

        result = detector.update(t, x, y)

        if result is not None:
            detected.append(result)

        t += dt

    # Segunda fijacion cerca de (480, 150).
    for _ in range(20):

        x = 480 + rng.normal(0, 3)
        y = 150 + rng.normal(0, 3)

        result = detector.update(t, x, y)

        if result is not None:
            detected.append(result)

        t += dt

    # Finalizar la ultima fijacion.
    result = detector.flush()

    if result is not None:
        detected.append(result)

    print("===== FIJACIONES LEXIIII =====")
    print("Fijaciones detectadas:", len(detected))

    for number, fixation in enumerate(
        detected, start=1
    ):
        print(
            f"Fijacion {number}:",
            fixation
        )
