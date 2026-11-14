
# =====================================================
# LEXIIII - SEGUIMIENTO DE LECTURA
#
# Fijaciones X,Y -> palabras -> posibles regresiones
#
# Incluye incertidumbre basada en la calibracion.
# No inventa una palabra cuando la posicion es dudosa.
# =====================================================

import json
import math


class ReadingTracker:

    def __init__(
        self,
        tolerance_px=12,
        uncertainty_px=0
    ):
        self.tolerance_px = float(tolerance_px)
        self.uncertainty_px = float(uncertainty_px)

        if self.tolerance_px < 0:
            raise ValueError("Tolerancia invalida")

        if self.uncertainty_px < 0:
            raise ValueError("Incertidumbre invalida")

        self.pages = {}
        self.events = []
        self.regressions = []
        self.unassigned = []
        self.last_event = None

    # -------------------------------------------------
    # REGISTRAR POSICIONES DE LAS PALABRAS
    # -------------------------------------------------

    def set_page_words(self, page, words):

        page = int(page)
        registered = []

        for word in words:

            item = {
                "page": page,
                "text": str(word["text"]),
                "global_index": int(
                    word["global_index"]
                ),
                "line": int(word["line"]),
                "x": float(word["x"]),
                "y": float(word["y"]),
                "width": float(word["width"]),
                "height": float(word["height"])
            }

            if (
                not math.isfinite(item["x"])
                or not math.isfinite(item["y"])
                or not math.isfinite(item["width"])
                or not math.isfinite(item["height"])
                or item["width"] <= 0
                or item["height"] <= 0
            ):
                raise ValueError(
                    "Geometria de palabra invalida"
                )

            registered.append(item)

        self.pages[page] = registered

    # -------------------------------------------------
    # DISTANCIA AL RECTANGULO DE UNA PALABRA
    # -------------------------------------------------

    @staticmethod
    def distance_to_box(x, y, word):

        x1 = word["x"]
        y1 = word["y"]

        x2 = x1 + word["width"]
        y2 = y1 + word["height"]

        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)

        return math.hypot(dx, dy)

    # -------------------------------------------------
    # IDENTIFICAR PALABRA CON INCERTIDUMBRE
    # -------------------------------------------------

    def find_word(self, x, y, page):

        x = float(x)
        y = float(y)

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        words = self.pages.get(int(page), [])

        candidates = []

        for word in words:

            distance = self.distance_to_box(
                x, y, word
            )

            if distance <= self.tolerance_px:
                candidates.append(
                    (distance, word)
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        best_distance, best_word = candidates[0]

        # Si dos cajas estan igualmente cerca,
        # no decidimos arbitrariamente.
        if len(candidates) > 1:

            second_distance = candidates[1][0]

            if abs(
                second_distance - best_distance
            ) < 1e-6:
                return None

        # Con incertidumbre mayor que cero,
        # exigimos que la posicion tenga margen
        # suficiente dentro del rectangulo.
        #
        # Es una regla conservadora:
        # el error P90 es un indicador empirico,
        # no una garantia de confianza individual.

        if self.uncertainty_px > 0:

            x1 = best_word["x"]
            y1 = best_word["y"]

            x2 = x1 + best_word["width"]
            y2 = y1 + best_word["height"]

            edge_clearance = min(
                x - x1,
                x2 - x,
                y - y1,
                y2 - y
            )

            if edge_clearance < self.uncertainty_px:
                return None

        return best_word

    # -------------------------------------------------
    # REGISTRAR FIJACION
    # -------------------------------------------------

    def add_fixation(self, fixation, page):

        page = int(page)

        x = float(fixation["x"])
        y = float(fixation["y"])

        word = self.find_word(
            x, y, page
        )

        if word is None:

            self.unassigned.append({
                "page": page,
                "x": x,
                "y": y,
                "reason": "uncertain_word",
                "start_s": float(
                    fixation["start_s"]
                )
            })

            # No inferir un retorno atraves
            # de fijaciones que no conocemos.
            self.last_event = None

            return None

        event = {
            "page": page,
            "word": word["text"],
            "word_index": word["global_index"],
            "line": word["line"],
            "x": x,
            "y": y,
            "start_s": float(
                fixation["start_s"]
            ),
            "end_s": float(
                fixation["end_s"]
            ),
            "duration_ms": float(
                fixation["duration_ms"]
            ),
            "possible_regression": False
        }

        previous = self.last_event

        if (
            previous is not None
            and previous["page"] == page
        ):

            # El orden global de las palabras
            # incluye los cambios normales de linea.
            returned_to_earlier_word = (
                event["word_index"]
                < previous["word_index"]
            )

            if returned_to_earlier_word:

                event["possible_regression"] = True

                self.regressions.append({
                    "page": page,
                    "from_word": previous["word"],
                    "from_index": previous[
                        "word_index"
                    ],
                    "to_word": event["word"],
                    "to_index": event["word_index"],
                    "start_s": event["start_s"]
                })

        self.events.append(event)
        self.last_event = event

        return event

    # -------------------------------------------------
    # CAMBIO DE PAGINA
    # -------------------------------------------------

    def page_changed(self):
        self.last_event = None

    # -------------------------------------------------
    # RESUMEN
    # -------------------------------------------------

    def summary(self):

        duration = sum(
            event["duration_ms"]
            for event in self.events
        )

        return {
            "fixations_assigned": len(self.events),
            "fixations_unassigned": len(self.unassigned),
            "possible_regressions": len(
                self.regressions
            ),
            "total_fixation_ms": round(
                duration, 1
            ),
            "uncertainty_px": self.uncertainty_px,
            "events": self.events,
            "regressions": self.regressions,
            "unassigned": self.unassigned
        }


# =====================================================
# PRUEBA SIN CAMARA
# =====================================================

if __name__ == "__main__":

    tracker = ReadingTracker(
        tolerance_px=12,
        uncertainty_px=8
    )

    words = [
        {
            "text": "gato",
            "global_index": 0,
            "line": 0,
            "x": 100,
            "y": 100,
            "width": 90,
            "height": 45
        },
        {
            "text": "come",
            "global_index": 1,
            "line": 0,
            "x": 220,
            "y": 100,
            "width": 90,
            "height": 45
        },
        {
            "text": "pescado",
            "global_index": 2,
            "line": 0,
            "x": 340,
            "y": 100,
            "width": 125,
            "height": 45
        }
    ]

    tracker.set_page_words(
        page=0,
        words=words
    )

    # gato -> come -> pescado -> gato
    # La ultima fijacion vuelve hacia atras.

    positions = [
        (145, 122),
        (265, 122),
        (395, 122),
        (145, 122)
    ]

    for i, (x, y) in enumerate(positions):

        fixation = {
            "x": x,
            "y": y,
            "start_s": i * 0.4,
            "end_s": i * 0.4 + 0.3,
            "duration_ms": 300
        }

        event = tracker.add_fixation(
            fixation, 0
        )

        if event:
            print(
                event["word"],
                "| Regresion:",
                event["possible_regression"]
            )

    # Esta posicion cae en el espacio
    # entre dos palabras.
    unclear = {
        "x": 205,
        "y": 122,
        "start_s": 1.6,
        "end_s": 1.9,
        "duration_ms": 300
    }

    assert tracker.add_fixation(
        unclear, 0
    ) is None

    # Este punto esta demasiado cerca
    # del borde izquierdo de "come".
    edge = {
        "x": 223,
        "y": 122,
        "start_s": 2.0,
        "end_s": 2.3,
        "duration_ms": 300
    }

    assert tracker.add_fixation(
        edge, 0
    ) is None

    summary = tracker.summary()

    print("\n===== RESUMEN LEXIIII =====")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )

    assert summary["fixations_assigned"] == 4
    assert summary["fixations_unassigned"] == 2
    assert summary["possible_regressions"] == 1

    print("\nPRUEBA DE INCERTIDUMBRE SUPERADA")
