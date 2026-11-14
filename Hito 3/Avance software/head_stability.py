
# =====================================================
# LEXIIII - ESTABILIDAD DE LA CABEZA
#
# Guarda una referencia durante la calibracion.
# Compara las posiciones posteriores con esa referencia.
#
# No predice coordenadas de mirada.
# No sustituye la calibracion personalizada X,Y.
# =====================================================

import numpy as np


class HeadStability:

    def __init__(
        self,
        max_displacement=0.06,
        min_samples=8
    ):
        self.max_displacement = float(max_displacement)
        self.min_samples = int(min_samples)

        self.reference = None
        self.calibration_samples = []
        self.ready = False

        if self.max_displacement <= 0:
            raise ValueError("Umbral invalido")

        if self.min_samples < 3:
            raise ValueError("Muestras insuficientes")

    # ---------------------------------------------
    # COMPROBAR DATOS
    # ---------------------------------------------

    @staticmethod
    def check_head(head):

        if not isinstance(head, dict):
            return None

        try:
            values = np.array(
                [
                    head["head_x"],
                    head["head_y"]
                ],
                dtype=float
            )

        except (KeyError, TypeError, ValueError):
            return None

        if not np.all(np.isfinite(values)):
            return None

        return values

    # ---------------------------------------------
    # REINICIAR EN CADA SESION
    # ---------------------------------------------

    def reset(self):

        self.reference = None
        self.calibration_samples = []
        self.ready = False

    # ---------------------------------------------
    # REGISTRAR CABEZA DURANTE CALIBRACION
    # ---------------------------------------------

    def add_calibration_sample(self, head):

        values = self.check_head(head)

        if values is None:
            return False

        self.calibration_samples.append(values)

        return True

    # ---------------------------------------------
    # CALCULAR POSICION DE REFERENCIA
    # ---------------------------------------------

    def calibrate(self):

        if len(self.calibration_samples) < self.min_samples:
            raise ValueError(
                "Faltan muestras de cabeza"
            )

        values = np.array(
            self.calibration_samples
        )

        self.reference = np.median(
            values,
            axis=0
        )

        self.ready = True

        print(
            "[LEXIIII] Referencia de cabeza:",
            self.reference.tolist()
        )

    # ---------------------------------------------
    # EVALUAR ESTABILIDAD
    # ---------------------------------------------

    def evaluate(self, head):

        if not self.ready:
            return {
                "stable": False,
                "reason": "not_calibrated",
                "displacement": None
            }

        values = self.check_head(head)

        if values is None:
            return {
                "stable": False,
                "reason": "invalid_head",
                "displacement": None
            }

        displacement = float(
            np.linalg.norm(
                values - self.reference
            )
        )

        stable = (
            displacement <= self.max_displacement
        )

        return {
            "stable": stable,
            "reason": (
                "ok"
                if stable
                else "head_movement"
            ),
            "displacement": round(
                displacement,
                5
            )
        }


# =====================================================
# PRUEBA SIMULADA SIN CAMARA
# =====================================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    stability = HeadStability(
        max_displacement=0.06
    )

    # Referencia simulada durante la calibracion.
    for _ in range(30):

        head = {
            "head_x": 0.52 + rng.normal(0, 0.003),
            "head_y": 0.48 + rng.normal(0, 0.003)
        }

        stability.add_calibration_sample(head)

    stability.calibrate()

    # Caso 1: cabeza cerca de la referencia.
    normal = stability.evaluate({
        "head_x": 0.525,
        "head_y": 0.485
    })

    # Caso 2: desplazamiento grande.
    moved = stability.evaluate({
        "head_x": 0.68,
        "head_y": 0.58
    })

    # Caso 3: no se pudo medir la cabeza.
    missing = stability.evaluate(None)

    print("\n===== ESTABILIDAD LEXIIII =====")

    print("Cabeza estable:", normal)
    print("Cabeza desplazada:", moved)
    print("Sin medicion:", missing)

    assert normal["stable"] is True
    assert moved["stable"] is False
    assert missing["stable"] is False

    print("\nPRUEBA DE ESTABILIDAD SUPERADA")
