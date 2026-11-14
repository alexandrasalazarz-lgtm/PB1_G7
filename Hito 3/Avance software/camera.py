
# ==========================================================
# LEXIIII - MODULO DE CAMARA
# Raspberry Pi 5 + InnoMaker OV5647
# Compatible con OpenCV y MediaPipe
# ==========================================================

import sys
import time
from pathlib import Path

import cv2
import numpy as np


def running_on_raspberry_pi() -> bool:
    """Comprueba si el programa se ejecuta en una Raspberry Pi."""

    if not sys.platform.startswith("linux"):
        return False

    try:
        model = Path("/proc/device-tree/model").read_bytes()
        return b"Raspberry Pi" in model
    except OSError:
        return False


class Camera:
    """
    Modulo de captura de Lexiiii.

    Mantiene la interfaz utilizada por main.py:

        camera = Camera()
        frame = camera.read()
        camera.release()

    Entrega fotogramas BGR de 8 bits compatibles
    con OpenCV y el FaceTracker de MediaPipe.
    """

    def __init__(
        self,
        camera_index: int | None = None,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
        fps: int = 30,
    ) -> None:

        self.width = int(width)
        self.height = int(height)
        self.mirror = bool(mirror)
        self.fps = int(fps)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Resolucion de camara no valida.")

        if self.fps <= 0:
            raise ValueError("Los FPS deben ser mayores que cero.")

        self.cap = None
        self.picam2 = None
        self.backend = None
        self.camera_index = camera_index

        if running_on_raspberry_pi():
            self._init_picamera2()
        else:
            self._init_opencv()

    # ======================================================
    # 1. CAMARA RASPBERRY PI 5
    # ======================================================

    def _init_picamera2(self) -> None:
        """Inicializa la camara CSI utilizando Picamera2."""

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 no esta disponible en Raspberry Pi."
            ) from exc

        index = (
            0 if self.camera_index is None
            else int(self.camera_index)
        )

        try:
            self.picam2 = Picamera2(camera_num=index)

            frame_duration_us = round(
                1_000_000 / self.fps
            )

            config = self.picam2.create_video_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "RGB888",
                },
                controls={
                    "FrameDurationLimits": (
                        frame_duration_us,
                        frame_duration_us,
                    ),
                },
                buffer_count=4,
            )

            self.picam2.configure(config)
            self.picam2.start()

            # Permite estabilizar la exposicion automatica.
            time.sleep(1.5)

            self.backend = "Picamera2"

            print("[Camera] Camara Raspberry iniciada")
            print(
                f"[Camera] Resolucion: "
                f"{self.width} x {self.height}"
            )
            print(
                f"[Camera] FPS solicitados: {self.fps}"
            )
            print(
                f"[Camera] Espejo horizontal: {self.mirror}"
            )

        except Exception:
            self.release()
            raise

    # ======================================================
    # 2. CAMARA DE COMPUTADORA
    # ======================================================

    def _init_opencv(self) -> None:
        """Inicializa una webcam convencional con OpenCV."""

        index = (
            1 if self.camera_index is None
            else int(self.camera_index)
        )

        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(
                f"No se pudo abrir la camara {index}."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.fps
        )

        self.backend = "OpenCV"

        print("[Camera] Webcam iniciada")

    # ======================================================
    # 3. CAPTURA DE FOTOGRAMAS
    # ======================================================

    def read(self) -> np.ndarray | None:
        """
        Devuelve un fotograma en formato BGR.

        Conserva la relacion de aspecto original.
        Aplica espejo horizontal si mirror=True.
        """

        if self.backend == "Picamera2":

            if self.picam2 is None:
                return None

            # Picamera2 entrega el arreglo en orden BGR
            # para esta configuracion RGB888.
            frame = self.picam2.capture_array("main")

        elif self.backend == "OpenCV":

            if self.cap is None:
                return None

            success, frame = self.cap.read()

            if not success:
                return None

        else:
            return None

        if frame is None:
            return None

        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            raise RuntimeError(
                "Formato de imagen incorrecto. "
                "Se esperaba una imagen BGR de 8 bits."
            )

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    # ======================================================
    # 4. INFORMACION DE CAMARA
    # ======================================================

    def info(self) -> dict:
        """Devuelve la configuracion solicitada."""

        return {
            "backend": self.backend,
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
            "mirror": self.mirror,
            "camera_index": self.camera_index,
        }

    # ======================================================
    # 5. LIBERAR CAMARA
    # ======================================================

    def release(self) -> None:
        """Libera los recursos de la camara."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                self.picam2.close()
                self.picam2 = None

        self.backend = None

        print("[Camera] Camara liberada")


# ==========================================================
# 6. PRUEBA INDEPENDIENTE
# ==========================================================

if __name__ == "__main__":

    camera = Camera()
    total_frames = 60
    captured_frames = 0
    last_frame = None

    print("[Test] Iniciando prueba de captura...")

    start_time = time.perf_counter()

    try:
        for _ in range(total_frames):

            frame = camera.read()

            if frame is not None:
                captured_frames += 1
                last_frame = frame

        duration = time.perf_counter() - start_time

        actual_fps = (
            captured_frames / duration
            if duration > 0
            else 0.0
        )

        print("\n========== RESULTADOS ==========")

        if last_frame is not None:
            print(
                "[Test] Imagen:",
                last_frame.shape
            )

        print(
            "[Test] Fotogramas capturados:",
            captured_frames
        )

        print(
            "[Test] FPS de captura:",
            round(actual_fps, 2)
        )

        print("================================")

    finally:
        camera.release()
