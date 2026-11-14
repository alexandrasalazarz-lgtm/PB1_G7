
# =====================================================
# LEXIIII - CAPTURA OCULAR Y CONTROL DE CALIDAD
#
# Camera -> FaceTracker -> GazeEstimator
#
# Devuelve:
# - Caracteristicas de ambos iris
# - Apertura ocular
# - Posicion relativa de cabeza
# - Estado de validez de la medicion
#
# La estimacion de X,Y corresponde a
# CalibrationModel, no a este modulo.
# =====================================================

from collections import deque

import numpy as np


class OcularCapture:

    FEATURE_NAMES = (
        "right_x",
        "left_x",
        "right_y_stable",
        "left_y_stable"
    )

    def __init__(
        self,
        camera=None,
        tracker=None,
        estimator=None
    ):

        self.camera = camera
        self.tracker = tracker
        self.estimator = estimator

        self.frames_processed = 0
        self.valid_measurements = 0

        # Referencia de apertura ocular.
        self.eye_history = deque(maxlen=60)
        self.min_eye_samples = 12

        self.blink_reference = None

        try:

            if self.camera is None:

                from camera import Camera

                self.camera = Camera(
                    width=1280,
                    height=720,
                    mirror=True,
                    fps=30
                )

            if self.tracker is None:

                from face_tracking import FaceTracker

                self.tracker = FaceTracker()

            if self.estimator is None:

                from gaze_estimation import GazeEstimator

                self.estimator = GazeEstimator

        except Exception:

            self.close()
            raise

    # ================================================
    # REINICIAR CALIDAD OCULAR
    # ================================================

    def reset_quality(self):

        self.eye_history.clear()

        self.blink_reference = None

        self.frames_processed = 0
        self.valid_measurements = 0

    # ================================================
    # MEDICION INVALIDA
    # ================================================

    @staticmethod
    def invalid(reason):

        return {
            "valid": False,
            "features": None,
            "reason": reason,
            "eye_openness": None,
            "head": None
        }

    # ================================================
    # COMPROBAR APERTURA DE OJOS
    # ================================================

    def check_blink(
        self,
        right_ear,
        left_ear
    ):

        try:

            ears = np.asarray(
                [right_ear, left_ear],
                dtype=float
            )

        except (TypeError, ValueError):
            return "invalid_eye_openness"

        if (
            ears.shape != (2,)
            or not np.all(np.isfinite(ears))
            or np.any(ears <= 0)
        ):

            return "invalid_eye_openness"

        # Al iniciar, recopilar una referencia.
        # No aceptar mediciones oculares durante
        # este periodo de inicializacion.
        if len(self.eye_history) < self.min_eye_samples:

            self.eye_history.append(ears)

            return "eye_reference_warmup"

        history = np.asarray(
            self.eye_history,
            dtype=float
        )

        reference = np.percentile(
            history,
            75,
            axis=0
        )

        self.blink_reference = reference

        # Umbral experimental.
        if np.any(
            ears < 0.65 * reference
        ):

            return "blink_or_closed_eye"

        # Guardar solo medidas consideradas abiertas.
        self.eye_history.append(ears)

        return "ok"

    # ================================================
    # LEER UN FOTOGRAMA
    # ================================================

    def read(self):

        frame = self.camera.read()

        if frame is None:

            return self.invalid(
                "no_frame"
            )

        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):

            return self.invalid(
                "invalid_frame"
            )

        self.frames_processed += 1

        result = (
            self.tracker.detect_face_landmarks(
                frame
            )
        )

        if result is None:

            return self.invalid(
                "no_face"
            )

        faces = getattr(
            result,
            "face_landmarks",
            None
        )

        if not faces:

            return self.invalid(
                "no_face"
            )

        face = faces[0]

        # Se necesitan landmarks de iris.
        if len(face) < 478:

            return self.invalid(
                "iris_not_found"
            )

        height, width = frame.shape[:2]

        # ============================================
        # EXTRAER CARACTERISTICAS OCULARES
        # ============================================

        try:

            measurements = (
                self.estimator.gaze_features(
                    face,
                    image_size=(width, height)
                )
            )

            features = np.asarray(
                [
                    measurements[name]
                    for name in self.FEATURE_NAMES
                ],
                dtype=float
            )

        except (
            AttributeError,
            KeyError,
            ValueError,
            TypeError,
            IndexError,
            ZeroDivisionError
        ):

            return self.invalid(
                "invalid_eye_geometry"
            )

        if (
            features.shape != (4,)
            or not np.all(np.isfinite(features))
        ):

            return self.invalid(
                "invalid_features"
            )

        # ============================================
        # EXTRAER POSICION DE CABEZA
        # ============================================

        try:

            head_data = (
                self.estimator.head_features(
                    face
                )
            )

            head_x = float(
                head_data["head_x"]
            )

            head_y = float(
                head_data["head_y"]
            )

        except (
            AttributeError,
            KeyError,
            ValueError,
            TypeError,
            IndexError,
            ZeroDivisionError
        ):

            return self.invalid(
                "invalid_head"
            )

        if not np.all(
            np.isfinite([head_x, head_y])
        ):

            return self.invalid(
                "invalid_head"
            )

        head = {
            "head_x": head_x,
            "head_y": head_y
        }

        # ============================================
        # COMPROBAR PARPADEOS
        # ============================================

        right_ear = measurements.get(
            "right_ear"
        )

        left_ear = measurements.get(
            "left_ear"
        )

        eye_status = self.check_blink(
            right_ear,
            left_ear
        )

        if eye_status != "ok":

            return self.invalid(
                eye_status
            )

        # ============================================
        # MEDICION ACEPTADA
        # ============================================

        self.valid_measurements += 1

        return {
            "valid": True,
            "features": features.tolist(),
            "reason": None,
            "eye_openness": {
                "right": float(right_ear),
                "left": float(left_ear)
            },
            "head": head
        }

    # ================================================
    # CERRAR RECURSOS
    # ================================================

    def close(self):

        camera = self.camera
        tracker = self.tracker

        self.camera = None
        self.tracker = None

        if camera is not None:

            camera.release()

        if tracker is not None:

            tracker.close()


# =====================================================
# PRUEBA AUTOMATIZADA SIN CAMARA
# =====================================================

if __name__ == "__main__":

    from types import SimpleNamespace

    class FakeCamera:

        def read(self):

            return np.zeros(
                (720, 1280, 3),
                dtype=np.uint8
            )

        def release(self):
            pass

    class FakeTracker:

        def detect_face_landmarks(
            self,
            frame
        ):

            landmarks = [
                SimpleNamespace(
                    x=0.5,
                    y=0.5,
                    z=0.0
                )
                for _ in range(478)
            ]

            return SimpleNamespace(
                face_landmarks=[landmarks]
            )

        def close(self):
            pass

    class FakeEstimator:

        right_ear = 0.28
        left_ear = 0.29
        head_available = True

        @staticmethod
        def gaze_features(
            face_landmarks,
            image_size
        ):

            assert image_size == (1280, 720)

            return {
                "right_x": 0.42,
                "left_x": 0.51,
                "right_y_stable": 0.08,
                "left_y_stable": 0.12,
                "right_ear": FakeEstimator.right_ear,
                "left_ear": FakeEstimator.left_ear
            }

        @staticmethod
        def head_features(face_landmarks):

            if not FakeEstimator.head_available:
                return None

            return {
                "head_x": 0.52,
                "head_y": 0.48
            }

    capture = OcularCapture(
        camera=FakeCamera(),
        tracker=FakeTracker(),
        estimator=FakeEstimator
    )

    print(
        "\n===== PRUEBA OCULAR LEXIIII ====="
    )

    # 1. Preparacion de referencia ocular.
    for _ in range(12):

        result = capture.read()

        assert result["valid"] is False
        assert (
            result["reason"]
            == "eye_reference_warmup"
        )

    print("[1] Referencia inicial: CORRECTO")

    # 2. Ojos abiertos.
    result = capture.read()

    assert result["valid"] is True

    assert len(
        result["features"]
    ) == 4

    print("[2] Medicion ocular: CORRECTO")

    # 3. Parpadeo.
    FakeEstimator.right_ear = 0.06
    FakeEstimator.left_ear = 0.07

    result = capture.read()

    assert result["valid"] is False
    assert (
        result["reason"]
        == "blink_or_closed_eye"
    )

    print("[3] Parpadeo: CORRECTO")

    # 4. Recuperacion.
    FakeEstimator.right_ear = 0.28
    FakeEstimator.left_ear = 0.29

    result = capture.read()

    assert result["valid"] is True

    print("[4] Recuperacion: CORRECTO")

    # 5. Posicion de cabeza ausente.
    FakeEstimator.head_available = False

    result = capture.read()

    assert result["valid"] is False
    assert result["reason"] == "invalid_head"

    print("[5] Cabeza ausente: CORRECTO")

    FakeEstimator.head_available = True

    # 6. Apertura ocular ausente.
    FakeEstimator.right_ear = None

    result = capture.read()

    assert result["valid"] is False
    assert (
        result["reason"]
        == "invalid_eye_openness"
    )

    print("[6] Apertura ausente: CORRECTO")

    capture.close()

    print()
    print("PRUEBA DE CALIDAD OCULAR SUPERADA")
