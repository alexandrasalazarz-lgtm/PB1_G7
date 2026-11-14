
# ==========================================================
# LEXIIII - ESTIMACION GEOMETRICA DE MIRADA
# Compatible con gaze_runtime.py y main.py existentes
# ==========================================================

from typing import Any

import numpy as np


class GazeEstimator:
    """
    Extrae caracteristicas de ojos y cabeza.

    Los landmarks de MediaPipe estan normalizados.
    Toda geometria euclidiana se calcula primero
    en pixeles de camara.

    image_size: (ancho, alto) del fotograma original.
    """

    DEFAULT_IMAGE_SIZE = (1280, 720)

    # ------------------------------------------------------
    # 1. COORDENADAS
    # ------------------------------------------------------

    @staticmethod
    def _point(
        face_landmarks: Any,
        index: int,
    ) -> np.ndarray:
        """Devuelve coordenadas normalizadas (x, y)."""

        point = face_landmarks[index]

        return np.array(
            [point.x, point.y],
            dtype=np.float64,
        )

    @staticmethod
    def _pixel_point(
        face_landmarks: Any,
        index: int,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Convierte un landmark normalizado a pixeles."""

        width, height = image_size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Dimensiones de imagen no validas."
            )

        return GazeEstimator._point(
            face_landmarks,
            index,
        ) * np.array(
            [width, height],
            dtype=np.float64,
        )

    @staticmethod
    def _iris_to_pixels(
        iris_center: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:

        return np.asarray(
            iris_center,
            dtype=np.float64,
        ) * np.asarray(
            image_size,
            dtype=np.float64,
        )

    # ------------------------------------------------------
    # 2. CENTROS DE IRIS
    # ------------------------------------------------------

    @staticmethod
    def iris_centers(
        face_landmarks: Any,
    ) -> tuple[np.ndarray, np.ndarray]:

        if (
            face_landmarks is None
            or len(face_landmarks) < 478
        ):
            raise ValueError(
                "Se requieren 478 landmarks."
            )

        right_center = GazeEstimator._point(
            face_landmarks, 468
        )

        left_center = GazeEstimator._point(
            face_landmarks, 473
        )

        return right_center, left_center

    # ------------------------------------------------------
    # 3. MARCO LOCAL DE CADA OJO
    # ------------------------------------------------------

    @staticmethod
    def _eye_local(
        face_landmarks: Any,
        iris_center: np.ndarray,
        corner_1: int,
        corner_2: int,
        image_size: tuple[int, int],
    ) -> tuple[float, float, float]:
        """
        Proyecta el iris sobre el eje entre comisuras
        y su perpendicular.

        horizontal:
            0 = extremo izquierdo de la imagen
            1 = extremo derecho de la imagen

        vertical:
            Desplazamiento firmado respecto al eje
            de comisuras, normalizado por su longitud.

        Retorna horizontal, vertical y ancho en px.
        """

        p1 = GazeEstimator._pixel_point(
            face_landmarks, corner_1, image_size
        )

        p2 = GazeEstimator._pixel_point(
            face_landmarks, corner_2, image_size
        )

        iris = GazeEstimator._iris_to_pixels(
            iris_center, image_size
        )

        # Mantener el eje horizontal con el mismo
        # sentido para ambos ojos.
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        axis = p2 - p1

        eye_width = float(np.linalg.norm(axis))

        if eye_width < 1e-6:
            raise ValueError(
                "Ancho ocular no valido."
            )

        unit_x = axis / eye_width

        # Perpendicular positiva hacia abajo
        # en coordenadas de imagen.
        unit_y = np.array(
            [-unit_x[1], unit_x[0]],
            dtype=np.float64,
        )

        relative = iris - p1

        horizontal = float(
            np.dot(relative, unit_x) / eye_width
        )

        vertical = float(
            np.dot(relative, unit_y) / eye_width
        )

        return horizontal, vertical, eye_width

    # ------------------------------------------------------
    # 4. CARACTERISTICA HEREDADA DE PARPADOS
    # ------------------------------------------------------

    @staticmethod
    def iris_relative_position(
        face_landmarks: Any,
        iris_center: np.ndarray,
        corner_1: int,
        corner_2: int,
        upper_lid: int,
        lower_lid: int,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> tuple[float, float]:
        """
        Mantiene la interfaz anterior.

        X: proyeccion sobre el eje local del ojo.
        Y: posicion respecto a la apertura palpebral.

        La caracteristica Y depende del parpadeo;
        se conserva solo para comparacion historica.
        """

        horizontal, _, _ = GazeEstimator._eye_local(
            face_landmarks,
            iris_center,
            corner_1,
            corner_2,
            image_size,
        )

        iris = GazeEstimator._iris_to_pixels(
            iris_center, image_size
        )

        upper = GazeEstimator._pixel_point(
            face_landmarks, upper_lid, image_size
        )

        lower = GazeEstimator._pixel_point(
            face_landmarks, lower_lid, image_size
        )

        top = min(upper[1], lower[1])
        bottom = max(upper[1], lower[1])

        aperture = bottom - top

        if aperture < 1e-6:
            raise ValueError(
                "Apertura palpebral no valida."
            )

        vertical_lid = float(
            (iris[1] - top) / aperture
        )

        return horizontal, vertical_lid

    # ------------------------------------------------------
    # 5. POSICION VERTICAL ESTABLE
    # ------------------------------------------------------

    @staticmethod
    def iris_vertical_stable(
        face_landmarks: Any,
        iris_center: np.ndarray,
        corner_1: int,
        corner_2: int,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> float:
        """
        Posicion vertical respecto al eje local
        de comisuras, sin dividir por apertura
        de parpados.
        """

        _, vertical, _ = GazeEstimator._eye_local(
            face_landmarks,
            iris_center,
            corner_1,
            corner_2,
            image_size,
        )

        return vertical

    # ------------------------------------------------------
    # 6. EYE ASPECT RATIO
    # ------------------------------------------------------

    @staticmethod
    def eye_aspect_ratio(
        face_landmarks: Any,
        pairs: tuple[tuple[int, int], ...],
        corners: tuple[int, int],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> float:
        """
        Calcula la apertura ocular normalizada
        por el ancho del ojo.

        No aplica automaticamente un umbral
        de parpadeo: debe calibrarse por usuario.
        """

        p1 = GazeEstimator._pixel_point(
            face_landmarks, corners[0], image_size
        )

        p2 = GazeEstimator._pixel_point(
            face_landmarks, corners[1], image_size
        )

        width = float(np.linalg.norm(p2 - p1))

        if width < 1e-6:
            raise ValueError(
                "Ancho ocular no valido."
            )

        distances = []

        for upper_id, lower_id in pairs:
            upper = GazeEstimator._pixel_point(
                face_landmarks, upper_id, image_size
            )

            lower = GazeEstimator._pixel_point(
                face_landmarks, lower_id, image_size
            )

            distances.append(
                float(np.linalg.norm(upper - lower))
            )

        return float(np.mean(distances) / width)

    # ------------------------------------------------------
    # 7. CARACTERISTICAS DE MIRADA
    # ------------------------------------------------------

    @staticmethod
    def gaze_features(
        face_landmarks: Any,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> dict[str, Any]:

        right_center, left_center = (
            GazeEstimator.iris_centers(face_landmarks)
        )

        right_x, right_y = (
            GazeEstimator.iris_relative_position(
                face_landmarks,
                right_center,
                33, 133, 159, 145,
                image_size,
            )
        )

        left_x, left_y = (
            GazeEstimator.iris_relative_position(
                face_landmarks,
                left_center,
                362, 263, 386, 374,
                image_size,
            )
        )

        right_y_stable = (
            GazeEstimator.iris_vertical_stable(
                face_landmarks,
                right_center,
                33, 133,
                image_size,
            )
        )

        left_y_stable = (
            GazeEstimator.iris_vertical_stable(
                face_landmarks,
                left_center,
                362, 263,
                image_size,
            )
        )

        right_ear = GazeEstimator.eye_aspect_ratio(
            face_landmarks,
            ((159, 145), (158, 153), (160, 144)),
            (33, 133),
            image_size,
        )

        left_ear = GazeEstimator.eye_aspect_ratio(
            face_landmarks,
            ((386, 374), (385, 380), (387, 373)),
            (362, 263),
            image_size,
        )

        gaze_x = (right_x + left_x) / 2.0
        gaze_y = (right_y + left_y) / 2.0

        gaze_y_stable = (
            right_y_stable + left_y_stable
        ) / 2.0

        return {
            # Campos originales para gaze_runtime.py
            "right_x": float(right_x),
            "right_y": float(right_y),
            "left_x": float(left_x),
            "left_y": float(left_y),
            "gaze_x": float(gaze_x),
            "gaze_y": float(gaze_y),
            "right_y_stable": float(right_y_stable),
            "left_y_stable": float(left_y_stable),
            "gaze_y_stable": float(gaze_y_stable),
            "right_center": right_center,
            "left_center": left_center,

            # Campos nuevos para futuras pruebas
            "eye_disagreement_x": float(
                abs(right_x - left_x)
            ),
            "eye_disagreement_y": float(
                abs(right_y_stable - left_y_stable)
            ),
            "right_ear": float(right_ear),
            "left_ear": float(left_ear),
            "mean_ear": float(
                (right_ear + left_ear) / 2.0
            ),
        }

    # ------------------------------------------------------
    # 8. CARACTERISTICAS DE CABEZA HEREDADAS
    # ------------------------------------------------------

    @staticmethod
    def head_features(
        face_landmarks: Any,
    ) -> dict[str, float]:
        """
        Se mantiene para compatibilidad.

        Estas caracteristicas NO sustituyen
        una estimacion tridimensional de pose.
        """

        nose = GazeEstimator._point(
            face_landmarks, 1
        )

        side_1 = GazeEstimator._point(
            face_landmarks, 234
        )

        side_2 = GazeEstimator._point(
            face_landmarks, 454
        )

        forehead = GazeEstimator._point(
            face_landmarks, 10
        )

        chin = GazeEstimator._point(
            face_landmarks, 152
        )

        x_min = min(side_1[0], side_2[0])
        x_max = max(side_1[0], side_2[0])

        y_min = min(forehead[1], chin[1])
        y_max = max(forehead[1], chin[1])

        face_width = x_max - x_min
        face_height = y_max - y_min

        if (
            face_width < 1e-6
            or face_height < 1e-6
        ):
            raise ValueError(
                "Geometria facial no valida."
            )

        head_x = (nose[0] - x_min) / face_width
        head_y = (nose[1] - y_min) / face_height

        return {
            "head_x": float(head_x),
            "head_y": float(head_y),
            "nose_x": float(nose[0]),
            "nose_y": float(nose[1]),
        }

    # ------------------------------------------------------
    # 9. COMPENSACION HEREDADA
    # ------------------------------------------------------

    @staticmethod
    def compensate_head_y(
        gaze_y_stable: float,
        head_y: float,
        reference_head_y: float = 0.5447,
        compensation_factor: float = 0.2151,
    ) -> float:
        """
        Solo compatibilidad con experimentos antiguos.
        No usar sin validar estos coeficientes.
        """

        return float(
            gaze_y_stable
            + compensation_factor
            * (head_y - reference_head_y)
        )
