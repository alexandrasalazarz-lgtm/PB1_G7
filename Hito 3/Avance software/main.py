
"""
LEXIIII - EYE TRACKING

Calibracion personalizada antes de cada sesion.
Seguimiento ocular con coordenadas X,Y.
Deteccion de fijaciones y posibles regresiones.

Modo simulado: no necesita camara.
Modo real: requiere los modulos de captura ocular.
"""

import json
import time
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont

import numpy as np

from calibration_model import CalibrationModel
from fixation_detection import FixationDetector
from head_stability import HeadStability
from reading_tracker import ReadingTracker


# ==================================================
# CONFIGURACION
# ==================================================

WIDTH = 1024
HEIGHT = 600

TEXT = (
    "El gato come pescado sobre la mesa junto a la ventana. "
    "Después mira un pájaro que canta en el jardín. "
    "El gato vuelve a mirar la mesa y se duerme."
)


# ==================================================
# FILTRO DE CALIBRACION
# ==================================================

def filter_calibration_samples(samples, minimum=3):

    total = len(samples)

    values = np.asarray(
        samples,
        dtype=float
    )

    if (
        values.ndim != 2
        or values.shape[1] != 4
        or total < minimum
    ):
        return None, 0, total

    values = values[
        np.all(
            np.isfinite(values),
            axis=1
        )
    ]

    if len(values) < minimum:
        return None, 0, total

    center = np.median(
        values,
        axis=0
    )

    deviations = np.abs(
        values - center
    )

    mad = np.median(
        deviations,
        axis=0
    )

    scale = np.maximum(
        1.4826 * mad,
        0.002
    )

    normalized_distance = (
        deviations / scale
    )

    accepted_mask = np.all(
        normalized_distance <= 3.5,
        axis=1
    )

    accepted_values = values[
        accepted_mask
    ]

    required = max(
        minimum,
        int(np.ceil(
            0.65 * len(values)
        ))
    )

    if len(accepted_values) < required:
        return (
            None,
            len(accepted_values),
            total
        )

    median = np.median(
        accepted_values,
        axis=0
    ).tolist()

    return (
        median,
        len(accepted_values),
        total
    )


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

class LexiApp:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "LEXIIII - Eye tracking"
        )

        self.root.geometry(
            f"{WIDTH}x{HEIGHT}"
        )

        self.root.resizable(
            False,
            False
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.close()
        )

        self.w = WIDTH
        self.h = HEIGHT

        self.canvas = tk.Canvas(
            self.root,
            bg="white",
            highlightthickness=0,
            width=WIDTH,
            height=HEIGHT
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )

        self.text_font = tkfont.Font(
            family="Arial",
            size=29
        )

        self.buttons = []

        self.after_id = None

        self.capture = None

        self.mode = "home"

        self.source = "simulated"

        self.latest_head = None

        self.rng = np.random.default_rng(34)

        self.show_home()

    # ============================================
    # TEMPORIZADORES
    # ============================================

    def cancel_timer(self):

        if self.after_id is not None:

            try:

                self.root.after_cancel(
                    self.after_id
                )

            except tk.TclError:
                pass

            self.after_id = None

    def schedule(self, delay, function):

        self.cancel_timer()

        self.after_id = self.root.after(
            delay,
            function
        )

    # ============================================
    # INTERFAZ
    # ============================================

    def clear(self):

        self.canvas.delete("all")

        for button in self.buttons:
            button.destroy()

        self.buttons.clear()

    def title(self, message, y=50):

        self.canvas.create_text(
            self.w / 2,
            y,
            text=message,
            font=("Arial", 22, "bold"),
            fill="#172033"
        )

    def note(self, message, y, size=14):

        self.canvas.create_text(
            self.w / 2,
            y,
            text=message,
            font=("Arial", size),
            fill="#475569",
            width=self.w - 90,
            justify="center"
        )

    def button(self, label, x, y, command):

        button = tk.Button(
            self.canvas,
            text=label,
            command=command,
            font=("Arial", 12, "bold"),
            padx=12,
            pady=7
        )

        self.buttons.append(button)

        self.canvas.create_window(
            x,
            y,
            window=button
        )

    # ============================================
    # CERRAR CAPTURA OCULAR
    # ============================================

    def stop_capture(self):

        capture = self.capture

        self.capture = None

        if capture is not None:

            try:
                capture.close()

            except Exception as error:

                print(
                    "[LEXIIII] Error cerrando cámara:",
                    error
                )

    # ============================================
    # PANTALLA INICIAL
    # ============================================

    def show_home(self):

        self.cancel_timer()

        self.stop_capture()

        self.mode = "home"

        self.clear()

        self.title(
            "LEXIIII - SESIÓN DE LECTURA",
            150
        )

        self.note(
            "Calibración personalizada X,Y "
            "→ Validación → Lectura",
            230,
            16
        )

        self.button(
            "PRUEBA SIMULADA",
            512,
            320,
            lambda: self.start_session(
                "simulated"
            )
        )

        self.button(
            "CÁMARA REAL (EXPERIMENTAL)",
            512,
            395,
            lambda: self.start_session(
                "real"
            )
        )

        self.note(
            "La simulación NO utiliza tus ojos "
            "ni la cámara.",
            505,
            12
        )

    # ============================================
    # SIMULACION DE OJOS
    # ============================================

    def simulate_features(self, x, y):

        nx = x / self.w
        ny = y / self.h

        noise = self.rng.normal(
            0,
            0.0005,
            size=4
        )

        return [
            0.35 + 0.24 * nx + noise[0],
            0.38 + 0.22 * nx + noise[1],
            0.10 - 0.17 * ny + noise[2],
            0.14 - 0.19 * ny + noise[3]
        ]

    # ============================================
    # LEER CARACTERISTICAS OCULARES
    # ============================================

    def read_features(self, target=None):

        self.latest_head = None

        if self.source == "simulated":

            if target is None:
                return None

            self.latest_head = {
                "head_x": 0.52,
                "head_y": 0.48
            }

            return self.simulate_features(
                *target
            )

        result = self.capture.read()

        if result.get("valid"):

            self.latest_head = result.get(
                "head"
            )

            return result["features"]

        return None

    # ============================================
    # INICIAR NUEVA SESION
    # ============================================

    def start_session(self, source):

        self.cancel_timer()

        self.stop_capture()

        self.mode = "loading"

        self.source = source

        self.clear()

        if source == "real":

            self.title(
                "Iniciando cámara...",
                270
            )

            self.root.update_idletasks()

            try:

                from ocular_capture import OcularCapture

                self.capture = OcularCapture()

            except Exception as error:

                self.error(
                    "No se pudo iniciar la cámara:\n"
                    + str(error)
                )

                return

        self.session_id = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        # Nuevo modelo para cada sesion.
        self.model = CalibrationModel()

        # Nueva referencia de cabeza.
        self.head_stability = HeadStability(
            max_displacement=0.06,
            min_samples=8
        )

        self.latest_head = None

        self.training = []

        self.validation = []

        self.targets = []

        # Dos recorridos de nueve puntos.
        for repetition in range(2):

            for yf in (0.20, 0.50, 0.80):

                for xf in (0.15, 0.50, 0.85):

                    self.targets.append(
                        (
                            "train",
                            xf * self.w,
                            yf * self.h
                        )
                    )

        # Cuatro puntos independientes
        # para validar la calibracion.
        for xf, yf in (
            (0.32, 0.34),
            (0.68, 0.34),
            (0.32, 0.66),
            (0.68, 0.66)
        ):

            self.targets.append(
                (
                    "validate",
                    xf * self.w,
                    yf * self.h
                )
            )

        self.target_index = 0

        self.mode = "calibration"

        self.show_target()

    # ============================================
    # MOSTRAR PUNTO DE CALIBRACION
    # ============================================

    def show_target(self):

        if self.mode != "calibration":
            return

        # Entrenar luego de 18 puntos.
        if (
            self.target_index == 18
            and not self.model.trained
        ):

            try:

                self.model.fit(
                    self.training,
                    (self.w, self.h)
                )

                self.head_stability.calibrate()

            except Exception as error:

                self.error(
                    "No se pudo entrenar la calibración:\n"
                    + str(error)
                )

                return

        # Finalizar despues de 22 puntos.
        if self.target_index >= len(self.targets):

            self.finish_calibration()

            return

        kind, x, y = self.targets[
            self.target_index
        ]

        self.clear()

        if kind == "train":

            self.title(
                "CALIBRACIÓN"
            )

        else:

            self.title(
                "VALIDACIÓN"
            )

        if self.source == "simulated":

            instruction = "DATOS SIMULADOS"

        else:

            instruction = (
                "MIRA FIJAMENTE EL PUNTO AZUL"
            )

        self.note(
            f"Punto {self.target_index + 1}/"
            f"{len(self.targets)} - {instruction}",
            95
        )

        self.canvas.create_oval(
            x - 13,
            y - 13,
            x + 13,
            y + 13,
            fill="#2563eb",
            outline=""
        )

        self.canvas.create_oval(
            x - 4,
            y - 4,
            x + 4,
            y + 4,
            fill="white",
            outline=""
        )

        self.samples = []

        self.head_samples_for_point = []

        self.target_start = time.monotonic()

        self.schedule(
            35,
            self.collect_target
        )

    # ============================================
    # RECOGER MEDICIONES DEL PUNTO
    # ============================================

    def collect_target(self):

        if self.mode != "calibration":
            return

        kind, x, y = self.targets[
            self.target_index
        ]

        elapsed = (
            time.monotonic()
            - self.target_start
        )

        if self.source == "simulated":

            settle = 0.25
            collect = 0.35

        else:

            settle = 0.8
            collect = 1.8

        # Esperar a que la persona fije la mirada.
        if settle <= elapsed < settle + collect:

            try:

                features = self.read_features(
                    (x, y)
                )

            except Exception as error:

                self.error(
                    "Error al capturar los ojos:\n"
                    + str(error)
                )

                return

            if features is not None:

                self.samples.append(features)

                if (
                    kind == "train"
                    and self.latest_head is not None
                ):

                    self.head_samples_for_point.append(
                        self.latest_head.copy()
                    )

        if elapsed < settle + collect:

            self.schedule(
                35,
                self.collect_target
            )

            return

        minimum = (
            3 if self.source == "simulated"
            else 8
        )

        # Filtrar las mediciones inestables.
        median, accepted, total = (
            filter_calibration_samples(
                self.samples,
                minimum
            )
        )

        if median is None:

            self.error(
                f"Punto {self.target_index + 1}: "
                "muestras insuficientes o "
                f"inestables ({accepted}/{total}). "
                "Recalibra."
            )

            return

        print(
            f"[CALIBRACION] Punto "
            f"{self.target_index + 1}/"
            f"{len(self.targets)}: "
            f"{accepted}/{total} muestras aceptadas",
            flush=True
        )

        # Asociacion:
        # caracteristicas oculares -> punto X,Y.
        record = {
            "features": median,
            "x": x,
            "y": y
        }

        if kind == "train":

            self.training.append(record)

            for head in self.head_samples_for_point:

                self.head_stability.add_calibration_sample(
                    head
                )

        else:

            self.validation.append(record)

        self.target_index += 1

        self.show_target()

    # ============================================
    # FINALIZAR CALIBRACION
    # ============================================

    def finish_calibration(self):

        try:

            self.metrics = self.model.validate(
                self.validation
            )

        except Exception as error:

            self.error(
                "Error al validar la calibración:\n"
                + str(error)
            )

            return

        # Umbrales experimentales.
        # No garantizan precision por palabra.
        self.word_level_enabled = (
            self.metrics["mean_error_px"] <= 35
            and self.metrics["p90_error_px"] <= 50
        )

        self.mode = "validation"

        self.clear()

        self.title(
            "RESULTADO DE CALIBRACIÓN",
            155
        )

        self.note(
            f"Error medio: "
            f"{self.metrics['mean_error_px']} px "
            f"| P90: "
            f"{self.metrics['p90_error_px']} px",
            230,
            18
        )

        if self.source == "simulated":

            message = (
                "Resultados simulados; "
                "NO indican precisión real."
            )

        elif self.word_level_enabled:

            message = (
                "Precisión preliminar; "
                "falta validar por palabra "
                "con datos reales."
            )

        else:

            message = (
                "Precisión insuficiente: "
                "regresiones por palabra "
                "no evaluables."
            )

        self.note(
            message,
            300
        )

        self.button(
            "RECALIBRAR",
            350,
            405,
            lambda: self.start_session(
                self.source
            )
        )

        self.button(
            "LECTURA EXPERIMENTAL",
            675,
            405,
            self.start_reading
        )

    # ============================================
    # ORGANIZAR PALABRAS EN PAGINAS
    # ============================================

    def layout_pages(self):

        words = TEXT.split()

        pages = []

        index = 0

        gap = self.text_font.measure("  ")

        while index < len(words):

            page = []

            # Tres lineas por pagina.
            for line in range(3):

                x = 95

                y = 170 + line * 105

                while index < len(words):

                    width = self.text_font.measure(
                        words[index]
                    )

                    if (
                        x > 95
                        and x + width > self.w - 80
                    ):
                        break

                    page.append({
                        "text": words[index],
                        "global_index": index,
                        "line": line,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": self.text_font.metrics(
                            "linespace"
                        )
                    })

                    x += width + gap

                    index += 1

            pages.append(page)

        return pages

    # ============================================
    # COMENZAR LECTURA
    # ============================================

    def start_reading(self):

        if self.mode != "validation":
            return

        self.cancel_timer()

        self.pages = self.layout_pages()

        self.page = 0

        # Incertidumbre de la calibracion
        # realizada en esta misma sesion.
        self.reader = ReadingTracker(
            tolerance_px=12,
            uncertainty_px=self.metrics[
                "p90_error_px"
            ]
        )

        for page_number, words in enumerate(
            self.pages
        ):

            self.reader.set_page_words(
                page_number,
                words
            )

        self.detector = FixationDetector()

        self.raw_fixations = []

        self.reading_start = time.monotonic()

        self.mode = "reading"

        self.draw_page()

        if self.source == "simulated":

            # Recorrido exclusivo de simulacion.
            # No se utiliza en modo real.
            self.demo_path = [
                1, 2, 3, 1, 4, 5, 6, 7, 8, 9
            ]

            self.demo_index = 0

            self.demo_time = 0.0

            self.schedule(
                100,
                self.demo_step
            )

        else:

            self.schedule(
                35,
                self.real_step
            )

    # ============================================
    # MOSTRAR PAGINA
    # ============================================

    def draw_page(self):

        self.clear()

        self.title(
            f"LECTURA - PÁGINA "
            f"{self.page + 1}/{len(self.pages)}"
        )

        for word in self.pages[self.page]:

            self.canvas.create_text(
                word["x"],
                word["y"],
                text=word["text"],
                anchor="nw",
                font=self.text_font,
                fill="#172033"
            )

        self.button(
            "ANTERIOR",
            160,
            535,
            lambda: self.change_page(-1)
        )

        self.button(
            "FINALIZAR",
            512,
            535,
            self.finish_reading
        )

        self.button(
            "SIGUIENTE",
            855,
            535,
            lambda: self.change_page(1)
        )

    # ============================================
    # REGISTRAR FIJACIONES
    # ============================================

    def register_fixation(self, fixation):

        if fixation is None:
            return

        self.raw_fixations.append({
            "page": self.page,
            **fixation
        })

        if self.word_level_enabled:

            self.reader.add_fixation(
                fixation,
                self.page
            )

    # ============================================
    # CAMBIAR PAGINA
    # ============================================

    def change_page(self, amount):

        if self.mode != "reading":
            return

        new_page = self.page + amount

        if not 0 <= new_page < len(self.pages):
            return

        # Cerrar la fijacion actual.
        self.register_fixation(
            self.detector.flush()
        )

        # No comparar palabras de paginas distintas.
        self.reader.page_changed()

        self.page = new_page

        self.draw_page()

        # La trayectoria de demostracion
        # esta definida para la primera pagina.
        if self.source == "simulated":

            self.cancel_timer()

            self.note(
                "Demostración pausada tras "
                "cambiar de página.",
                488,
                12
            )

    # ============================================
    # INTERRUPCION DE LA MIRADA
    # ============================================

    def interrupt_gaze(self, timestamp):
        """
        Se utiliza cuando:
        - No se detecta el rostro.
        - Hay un parpadeo.
        - La cabeza esta demasiado desplazada.
        - La coordenada estimada sale de pantalla.

        Cierra la fijacion en curso y evita
        comparar fijaciones anteriores con
        las posteriores a la interrupcion.
        """

        fixation = self.detector.update(
            timestamp,
            None,
            None
        )

        self.register_fixation(
            fixation
        )

        # Reiniciar continuidad entre palabras.
        self.reader.page_changed()

        # Ocultar indicador de mirada.
        self.canvas.delete(
            "gaze_cursor"
        )

    # ============================================
    # LECTURA SIMULADA
    # ============================================

    def demo_step(self):

        if self.mode != "reading":
            return

        if self.demo_index >= len(self.demo_path):

            self.finish_reading()
            return

        index = self.demo_path[
            self.demo_index
        ]

        page_words = self.pages[0]

        if (
            self.page != 0
            or index >= len(page_words)
        ):

            self.finish_reading()
            return

        word = page_words[index]

        x = (
            word["x"]
            + word["width"] / 2
        )

        y = (
            word["y"]
            + word["height"] / 2
        )

        # Simular 20 mediciones por fijacion.
        for sample in range(20):

            features = self.simulate_features(
                x,
                y
            )

            pred_x, pred_y = self.model.predict(
                features
            )

            fixation = self.detector.update(
                self.demo_time,
                pred_x,
                pred_y
            )

            self.register_fixation(
                fixation
            )

            self.demo_time += 1 / 30

        self.register_fixation(
            self.detector.flush()
        )

        # Mostrar la mirada simulada.
        self.canvas.delete(
            "gaze_cursor"
        )

        self.canvas.create_oval(
            x - 7,
            y - 7,
            x + 7,
            y + 7,
            fill="#ea580c",
            outline="white",
            tags="gaze_cursor"
        )

        self.demo_index += 1

        self.schedule(
            420,
            self.demo_step
        )

    # ============================================
    # LECTURA CON CAMARA REAL
    # ============================================

    def real_step(self):

        if self.mode != "reading":
            return

        try:

            features = self.read_features()

            timestamp = (
                time.monotonic()
                - self.reading_start
            )

            # Evaluar posicion de la cabeza.
            head_status = (
                self.head_stability.evaluate(
                    self.latest_head
                )
            )

            # Si hay una medicion invalida,
            # interrumpir el seguimiento.
            if (
                features is None
                or not head_status["stable"]
            ):

                self.interrupt_gaze(
                    timestamp
                )

            else:

                x, y = self.model.predict(
                    features
                )

                if not (
                    0 <= x < self.w
                    and 0 <= y < self.h
                ):

                    self.interrupt_gaze(
                        timestamp
                    )

                else:

                    fixation = self.detector.update(
                        timestamp,
                        x,
                        y
                    )

                    self.register_fixation(
                        fixation
                    )

                    # Indicador de mirada estimada.
                    self.canvas.delete(
                        "gaze_cursor"
                    )

                    self.canvas.create_oval(
                        x - 5,
                        y - 5,
                        x + 5,
                        y + 5,
                        outline="#ea580c",
                        width=2,
                        tags="gaze_cursor"
                    )

        except Exception as error:

            self.error(
                "Error durante el seguimiento:\n"
                + str(error)
            )

            return

        self.schedule(
            35,
            self.real_step
        )

    # ============================================
    # FINALIZAR LECTURA
    # ============================================

    def finish_reading(self):

        if self.mode != "reading":
            return

        self.cancel_timer()

        self.register_fixation(
            self.detector.flush()
        )

        summary = self.reader.summary()

        if not self.word_level_enabled:

            summary["events"] = []

            summary["regressions"] = []

            summary["fixations_assigned"] = 0

            summary["possible_regressions"] = 0

        self.mode = "finished"

        # ========================================
        # GUARDAR RESULTADOS
        # ========================================

        directory = (
            Path(__file__).resolve().parent
            / "reading_data"
            / "sessions"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = directory / (
            f"{self.session_id}_session.json"
        )

        data = {
            "mode": self.source,
            "validation": self.metrics,
            "word_level_enabled_preliminary":
                self.word_level_enabled,
            "raw_fixations": self.raw_fixations,
            "reading": summary,
            "note": (
                "Resultados experimentales. "
                "Requieren validacion ocular real."
            )
        }

        try:

            filename.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

        except OSError as error:

            self.error(
                "No se pudieron guardar los "
                "resultados:\n"
                + str(error)
            )

            return

        self.stop_capture()

        # ========================================
        # MOSTRAR RESULTADOS
        # ========================================

        self.clear()

        self.title(
            "RESULTADOS DE LEXIIII",
            125
        )

        self.note(
            f"Fijaciones detectadas: "
            f"{len(self.raw_fixations)}\n"
            f"Fijaciones asignadas: "
            f"{summary['fixations_assigned']}\n"
            f"Fijaciones indeterminadas: "
            f"{summary['fixations_unassigned']}",
            225,
            16
        )

        if self.word_level_enabled:

            regression_text = str(
                summary["possible_regressions"]
            )

        else:

            regression_text = "NO EVALUABLE"

        self.note(
            "Posibles regresiones por palabra: "
            + regression_text,
            335,
            16
        )

        self.note(
            "Guardado: " + filename.name,
            395,
            12
        )

        self.button(
            "NUEVA SESIÓN",
            512,
            475,
            self.show_home
        )

        print(
            f"[LEXIIII] Sesión terminada. "
            f"Fijaciones: {len(self.raw_fixations)}, "
            f"regresiones posibles: {regression_text}",
            flush=True
        )

    # ============================================
    # ERRORES
    # ============================================

    def error(self, message):

        self.cancel_timer()

        self.stop_capture()

        self.mode = "error"

        self.clear()

        self.title(
            "NO SE PUDO COMPLETAR",
            150
        )

        self.note(
            message,
            280,
            15
        )

        self.button(
            "VOLVER",
            512,
            430,
            self.show_home
        )

        print(
            "[LEXIIII]",
            message,
            flush=True
        )

    # ============================================
    # CERRAR PROGRAMA
    # ============================================

    def close(self):

        self.cancel_timer()

        self.stop_capture()

        self.mode = "closed"

        self.root.destroy()

    def run(self):

        self.root.mainloop()


# ==================================================
# EJECUCION
# ==================================================

if __name__ == "__main__":

    LexiApp().run()
