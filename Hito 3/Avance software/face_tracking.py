
# ==========================================================
# LEXIIII - FACE TRACKING
# MediaPipe Face Landmarker
# Compatible con laptop y Raspberry Pi 5
# ==========================================================

from pathlib import Path
from typing import Any
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------------------------------
# 1. CONFIGURACION DEL MODELO
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "face_landmarker_v2_with_blendshapes.task"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)


class FaceTracker:
    """
    Deteccion de rostro, ojos e iris con MediaPipe.

    Se conserva la interfaz del proyecto original:

        tracker = FaceTracker()
        result = tracker.detect_face_landmarks(frame)

    El frame recibido debe ser BGR uint8.

    La matriz de transformacion facial se solicita
    para futuros experimentos de pose de cabeza.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        enable_pose: bool = True,
    ) -> None:

        self.model_path = Path(model_path)
        self.enable_pose = bool(enable_pose)

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # No reemplazar un modelo existente.
        if not self.model_path.is_file():

            print(
                "[FaceTracker] Descargando modelo MediaPipe..."
            )

            urllib.request.urlretrieve(
                MODEL_URL,
                str(self.model_path),
            )

        if self.model_path.stat().st_size == 0:
            raise RuntimeError(
                "El archivo del modelo MediaPipe esta vacio."
            )

        print(
            "[FaceTracker] Modelo:",
            self.model_path,
        )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=(
                self.enable_pose
            ),
        )

        self.landmarker = (
            vision.FaceLandmarker.create_from_options(
                options
            )
        )

        print("[FaceTracker] Detector inicializado")
        print(
            "[FaceTracker] Matriz facial habilitada:",
            self.enable_pose,
        )

    # ------------------------------------------------------
    # 2. DETECTAR LANDMARKS
    # ------------------------------------------------------

    def detect_face_landmarks(
        self,
        image: np.ndarray,
    ) -> Any:
        """
        Recibe un fotograma BGR uint8 y devuelve
        el resultado completo de MediaPipe.

        Mantiene la estructura usada por main.py:

            result.face_landmarks[0]
        """

        if image is None:
            return None

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "La imagen debe ser un arreglo NumPy."
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError(
                "Se requiere una imagen BGR uint8 "
                "con tres canales."
            )

        # Convertir una sola vez de BGR a RGB.
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect(
            mp_image
        )

        return result

    # ------------------------------------------------------
    # 3. EXTRAER MATRIZ FACIAL
    # ------------------------------------------------------

    @staticmethod
    def get_face_matrix(
        result: Any,
        face_index: int = 0,
    ) -> np.ndarray | None:
        """
        Extrae la matriz 4x4 de transformacion facial.

        Nota:
        La traslacion de esta matriz NO debe tratarse
        directamente como una distancia real en cm.
        """

        if result is None:
            return None

        matrices = getattr(
            result,
            "facial_transformation_matrixes",
            None,
        )

        if matrices is None:
            return None

        if len(matrices) <= face_index:
            return None

        matrix = np.asarray(
            matrices[face_index],
            dtype=np.float64,
        )

        if matrix.shape != (4, 4):
            return None

        if not np.all(np.isfinite(matrix)):
            return None

        return matrix

    # ------------------------------------------------------
    # 4. VERIFICAR SI EXISTE UN ROSTRO
    # ------------------------------------------------------

    @staticmethod
    def has_face(result: Any) -> bool:
        """Comprueba si MediaPipe detecto un rostro."""

        if result is None:
            return False

        faces = getattr(
            result,
            "face_landmarks",
            None,
        )

        return bool(faces)

    # ------------------------------------------------------
    # 5. CERRAR DETECTOR
    # ------------------------------------------------------

    def close(self) -> None:
        """Libera los recursos del modelo."""

        if self.landmarker is not None:

            self.landmarker.close()
            self.landmarker = None

        print("[FaceTracker] Detector liberado")
