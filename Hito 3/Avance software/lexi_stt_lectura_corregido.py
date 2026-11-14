
"""
reading_metrics.py

Logica independiente del motor de STT (Vosk o Whisper).

Toma:
  - el texto esperado
  - la transcripcion con timestamps por palabra

Calcula:
  - WPM
  - duracion de lectura
  - pausas entre palabras
  - pausas largas
  - palabras dificiles
  - errores:
      ok / substitution / omission / insertion / reordered

Las tildes no se consideran diferencias.

Las palabras claramente anteriores o posteriores al texto
esperado no afectan la evaluacion.
"""

from dataclasses import dataclass, field
import re
import statistics
import unicodedata


@dataclass
class WordEvent:
    word: str
    start: float
    end: float


@dataclass
class WordResult:
    expected_word: str
    said_word: str | None
    status: str
    pause_before: float
    duration: float


@dataclass
class SessionMetrics:
    wpm: float
    total_duration_sec: float
    words: list[WordResult] = field(default_factory=list)
    difficult_words: list[str] = field(default_factory=list)
    error_counts: dict = field(default_factory=dict)
    avg_pause_sec: float = 0.0
    long_pauses: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "wpm": round(
                self.wpm,
                1
            ),
            "total_duration_sec": round(
                self.total_duration_sec,
                2
            ),
            "avg_pause_sec": round(
                self.avg_pause_sec,
                2
            ),
            "error_counts": self.error_counts,
            "difficult_words": self.difficult_words,
            "long_pauses": self.long_pauses,
            "words": [
                {
                    "expected": word.expected_word,
                    "said": word.said_word,
                    "status": word.status,
                    "pause_before": round(
                        word.pause_before,
                        2
                    ),
                    "duration": round(
                        word.duration,
                        2
                    ),
                }
                for word in self.words
            ],
        }


PAUSE_THRESHOLD_SEC = 0.8

SUBSTITUTION_COST = 0.8
OMISSION_COST = 1.0
INSERTION_COST = 1.5


def _normalize(word: str) -> str:
    """
    Normaliza una palabra para compararla.

    Las tildes no se consideran diferencia:

        rapido == rápido
        dias == días
    """

    word = word.lower().strip()

    word = unicodedata.normalize(
        "NFKD",
        word
    )

    word = "".join(
        char
        for char in word
        if not unicodedata.combining(char)
    )

    word = re.sub(
        r"[^\wñü]",
        "",
        word
    )

    return word


def _safe_duration(
    start: float,
    end: float
) -> float:

    return max(
        0.0,
        end - start
    )


def _safe_pause(
    start: float,
    previous_end: float
) -> float:

    return max(
        0.0,
        start - previous_end
    )


def _longest_common_subsequence(
    expected_tokens: list[str],
    said_tokens: list[str]
):
    """
    Encuentra la mayor secuencia de palabras que coincide
    manteniendo el mismo orden.

    Devuelve:

        (indice_esperado, indice_reconocido)
    """

    n = len(expected_tokens)
    m = len(said_tokens)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(
        n - 1,
        -1,
        -1
    ):

        for j in range(
            m - 1,
            -1,
            -1
        ):

            if (
                expected_tokens[i]
                == said_tokens[j]
            ):

                dp[i][j] = (
                    1
                    + dp[i + 1][j + 1]
                )

            else:

                dp[i][j] = max(
                    dp[i + 1][j],
                    dp[i][j + 1]
                )

    matches = []

    i = 0
    j = 0

    while (
        i < n
        and j < m
    ):

        if (
            expected_tokens[i]
            == said_tokens[j]
        ):

            matches.append(
                (
                    i,
                    j
                )
            )

            i += 1
            j += 1

        elif dp[i + 1][j] >= dp[i][j + 1]:

            i += 1

        else:

            j += 1

    return matches


def _find_reading_segment(
    expected_tokens: list[str],
    said_tokens: list[str]
):
    """
    Determina que parte de la transcripcion corresponde
    al intento de lectura.

    Si existe una sola palabra desconocida inmediatamente
    antes o despues del texto esperado, se conserva porque
    puede ser una sustitucion.

    Si existen varias palabras desconocidas consecutivas,
    se consideran habla externa y se excluyen.

    Ejemplo:

        para empezar de cero
        el perro corre...
        la policia...

    se queda con:

        el perro corre...

    Pero:

        cebo perro corre...

    conserva "cebo", porque puede ser sustitucion de "el".
    """

    if not expected_tokens:
        return 0, len(said_tokens)

    if not said_tokens:
        return 0, 0

    expected_set = set(
        expected_tokens
    )

    expected_positions = [
        index
        for index, token in enumerate(
            said_tokens
        )
        if token in expected_set
    ]

    if not expected_positions:
        return 0, len(said_tokens)

    first_match = expected_positions[0]
    last_match = expected_positions[-1]

    start = first_match

    preceding_unknown = 0
    index = first_match - 1

    while (
        index >= 0
        and said_tokens[index] not in expected_set
    ):

        preceding_unknown += 1
        index -= 1

    if preceding_unknown == 1:
        start = first_match - 1

    end = last_match + 1

    following_unknown = 0
    index = last_match + 1

    while (
        index < len(said_tokens)
        and said_tokens[index] not in expected_set
    ):

        following_unknown += 1
        index += 1

    if following_unknown == 1:
        end = last_match + 2

    return start, end


def _align_block(
    expected_indexes,
    said_indexes,
    expected_tokens,
    said_tokens
):
    """
    Alinea un bloque entre dos coincidencias exactas.
    """

    operations = []

    if not expected_indexes:

        for said_index in said_indexes:

            operations.append(
                (
                    None,
                    said_index,
                    "insertion"
                )
            )

        return operations

    if not said_indexes:

        for expected_index in expected_indexes:

            operations.append(
                (
                    expected_index,
                    None,
                    "omission"
                )
            )

        return operations

    if (
        len(expected_indexes) == 1
        and len(said_indexes) == 1
    ):

        expected_index = expected_indexes[0]
        said_index = said_indexes[0]

        if (
            expected_tokens[expected_index]
            == said_tokens[said_index]
        ):

            status = "ok"

        else:

            status = "substitution"

        return [
            (
                expected_index,
                said_index,
                status
            )
        ]

    n = len(expected_indexes)
    m = len(said_indexes)

    dp = [
        [0.0] * (m + 1)
        for _ in range(n + 1)
    ]

    choice = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(
        1,
        n + 1
    ):

        dp[i][0] = (
            i * OMISSION_COST
        )

        choice[i][0] = "omission"

    for j in range(
        1,
        m + 1
    ):

        dp[0][j] = (
            j * INSERTION_COST
        )

        choice[0][j] = "insertion"

    for i in range(
        1,
        n + 1
    ):

        for j in range(
            1,
            m + 1
        ):

            expected_index = expected_indexes[
                i - 1
            ]

            said_index = said_indexes[
                j - 1
            ]

            if (
                expected_tokens[expected_index]
                == said_tokens[said_index]
            ):

                diagonal_cost = (
                    dp[i - 1][j - 1]
                )

            else:

                diagonal_cost = (
                    dp[i - 1][j - 1]
                    + SUBSTITUTION_COST
                )

            omission_cost = (
                dp[i - 1][j]
                + OMISSION_COST
            )

            insertion_cost = (
                dp[i][j - 1]
                + INSERTION_COST
            )

            candidates = [
                (
                    diagonal_cost,
                    "diagonal"
                ),
                (
                    omission_cost,
                    "omission"
                ),
                (
                    insertion_cost,
                    "insertion"
                ),
            ]

            best_cost, best_choice = min(
                candidates,
                key=lambda item: item[0]
            )

            dp[i][j] = best_cost
            choice[i][j] = best_choice

    i = n
    j = m

    reversed_operations = []

    while (
        i > 0
        or j > 0
    ):

        current_choice = choice[i][j]

        if current_choice == "diagonal":

            expected_index = expected_indexes[
                i - 1
            ]

            said_index = said_indexes[
                j - 1
            ]

            if (
                expected_tokens[expected_index]
                == said_tokens[said_index]
            ):

                status = "ok"

            else:

                status = "substitution"

            reversed_operations.append(
                (
                    expected_index,
                    said_index,
                    status
                )
            )

            i -= 1
            j -= 1

        elif current_choice == "omission":

            expected_index = expected_indexes[
                i - 1
            ]

            reversed_operations.append(
                (
                    expected_index,
                    None,
                    "omission"
                )
            )

            i -= 1

        elif current_choice == "insertion":

            said_index = said_indexes[
                j - 1
            ]

            reversed_operations.append(
                (
                    None,
                    said_index,
                    "insertion"
                )
            )

            j -= 1

        else:

            break

    reversed_operations.reverse()

    return reversed_operations


def _find_reordered_expected_index(
    token: str,
    said_index: int,
    expected_tokens: list[str]
):
    """
    Busca la palabra esperada correspondiente a una palabra
    reconocida fuera de posicion.
    """

    candidates = [
        index
        for index, expected_token
        in enumerate(expected_tokens)
        if expected_token == token
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda index: abs(
            index - said_index
        )
    )


def _create_alignment(
    expected_tokens: list[str],
    said_tokens: list[str]
):
    """
    Crea la alineacion final.
    """

    matches = _longest_common_subsequence(
        expected_tokens,
        said_tokens
    )

    matched_said = {
        said_index
        for _, said_index
        in matches
    }

    reordered_said = set()

    for said_index, token in enumerate(
        said_tokens
    ):

        if said_index in matched_said:
            continue

        if token in expected_tokens:

            reordered_said.add(
                said_index
            )

    operations = []

    previous_expected = -1
    previous_said = -1

    for expected_index, said_index in matches:

        block_expected = list(
            range(
                previous_expected + 1,
                expected_index
            )
        )

        block_said = [
            index
            for index in range(
                previous_said + 1,
                said_index
            )
            if index not in reordered_said
        ]

        operations.extend(
            _align_block(
                block_expected,
                block_said,
                expected_tokens,
                said_tokens
            )
        )

        operations.append(
            (
                expected_index,
                said_index,
                "ok"
            )
        )

        previous_expected = expected_index
        previous_said = said_index

    block_expected = list(
        range(
            previous_expected + 1,
            len(expected_tokens)
        )
    )

    block_said = [
        index
        for index in range(
            previous_said + 1,
            len(said_tokens)
        )
        if index not in reordered_said
    ]

    operations.extend(
        _align_block(
            block_expected,
            block_said,
            expected_tokens,
            said_tokens
        )
    )

    for said_index in sorted(
        reordered_said
    ):

        token = said_tokens[
            said_index
        ]

        expected_index = (
            _find_reordered_expected_index(
                token,
                said_index,
                expected_tokens
            )
        )

        if expected_index is not None:

            operations.append(
                (
                    expected_index,
                    said_index,
                    "reordered"
                )
            )

    return operations


def align_and_score(
    expected_text: str,
    recognized_words: list[WordEvent]
) -> SessionMetrics:

    expected_original = (
        expected_text.split()
    )

    expected_tokens = [
        _normalize(word)
        for word in expected_original
    ]

    error_counts = {
        "substitution": 0,
        "omission": 0,
        "insertion": 0,
        "reordered": 0,
        "ok": 0,
    }

    if not expected_original:

        return SessionMetrics(
            wpm=0.0,
            total_duration_sec=0.0,
            words=[],
            difficult_words=[],
            error_counts=error_counts,
            avg_pause_sec=0.0,
            long_pauses=[],
        )

    if not recognized_words:

        error_counts["omission"] = len(
            expected_original
        )

        return SessionMetrics(
            wpm=0.0,
            total_duration_sec=0.0,
            words=[
                WordResult(
                    expected_word=word,
                    said_word=None,
                    status="omission",
                    pause_before=0.0,
                    duration=0.0,
                )
                for word in expected_original
            ],
            difficult_words=expected_original.copy(),
            error_counts=error_counts,
            avg_pause_sec=0.0,
            long_pauses=[],
        )

    all_said_tokens = [
        _normalize(event.word)
        for event in recognized_words
    ]

    segment_start, segment_end = (
        _find_reading_segment(
            expected_tokens,
            all_said_tokens
        )
    )

    segment_events = recognized_words[
        segment_start:segment_end
    ]

    segment_tokens = all_said_tokens[
        segment_start:segment_end
    ]

    if not segment_events:

        error_counts["omission"] = len(
            expected_original
        )

        return SessionMetrics(
            wpm=0.0,
            total_duration_sec=0.0,
            words=[
                WordResult(
                    expected_word=word,
                    said_word=None,
                    status="omission",
                    pause_before=0.0,
                    duration=0.0,
                )
                for word in expected_original
            ],
            difficult_words=expected_original.copy(),
            error_counts=error_counts,
            avg_pause_sec=0.0,
            long_pauses=[],
        )

    operations = _create_alignment(
        expected_tokens,
        segment_tokens
    )

    results_by_expected = {}

    for (
        expected_index,
        said_index,
        status
    ) in operations:

        if expected_index is None:
            continue

        if expected_index in results_by_expected:

            current = results_by_expected[
                expected_index
            ]

            if (
                current["status"] == "omission"
                and status == "reordered"
            ):

                results_by_expected[
                    expected_index
                ] = {
                    "said_index": said_index,
                    "status": status,
                }

            continue

        results_by_expected[
            expected_index
        ] = {
            "said_index": said_index,
            "status": status,
        }

    words = []

    pauses = []

    long_pauses = []

    for index, expected_word in enumerate(
        expected_original
    ):

        result = results_by_expected.get(
            index
        )

        if result is None:

            words.append(
                WordResult(
                    expected_word=expected_word,
                    said_word=None,
                    status="omission",
                    pause_before=0.0,
                    duration=0.0,
                )
            )

            continue

        said_index = result[
            "said_index"
        ]

        status = result[
            "status"
        ]

        if said_index is None:

            words.append(
                WordResult(
                    expected_word=expected_word,
                    said_word=None,
                    status=status,
                    pause_before=0.0,
                    duration=0.0,
                )
            )

            continue

        event = segment_events[
            said_index
        ]

        previous_event = None

        for previous_index in range(
            said_index - 1,
            -1,
            -1
        ):

            previous_event = segment_events[
                previous_index
            ]

            break

        if previous_event is None:

            pause_before = 0.0

        else:

            pause_before = _safe_pause(
                event.start,
                previous_event.end
            )

        duration = _safe_duration(
            event.start,
            event.end
        )

        if pause_before > 0:

            pauses.append(
                pause_before
            )

        if pause_before >= PAUSE_THRESHOLD_SEC:

            long_pauses.append(
                {
                    "before": expected_word,
                    "duration": round(
                        pause_before,
                        2
                    ),
                }
            )

        words.append(
            WordResult(
                expected_word=expected_word,
                said_word=event.word,
                status=status,
                pause_before=pause_before,
                duration=duration,
            )
        )

    for word in words:

        error_counts[
            word.status
        ] += 1

    difficult_words = [
        word.expected_word
        for word in words
        if word.status != "ok"
    ]

    if len(segment_events) >= 2:

        chronological_pauses = []

        for index in range(
            1,
            len(segment_events)
        ):

            previous = segment_events[
                index - 1
            ]

            current = segment_events[
                index
            ]

            pause = _safe_pause(
                current.start,
                previous.end
            )

            if pause > 0:

                chronological_pauses.append(
                    pause
                )

        if chronological_pauses:

            avg_pause_sec = statistics.mean(
                chronological_pauses
            )

        else:

            avg_pause_sec = 0.0

    else:

        avg_pause_sec = 0.0

    first_event = segment_events[0]
    last_event = segment_events[-1]

    total_duration_sec = _safe_duration(
        first_event.start,
        last_event.end
    )

    recognized_word_count = len(
        segment_events
    )

    if total_duration_sec > 0:

        wpm = (
            recognized_word_count
            / total_duration_sec
            * 60.0
        )

    else:

        wpm = 0.0

    return SessionMetrics(
        wpm=wpm,
        total_duration_sec=total_duration_sec,
        words=words,
        difficult_words=difficult_words,
        error_counts=error_counts,
        avg_pause_sec=avg_pause_sec,
        long_pauses=long_pauses,
    )


# ==========================================================
# LEXI - STT en tiempo real + evaluación prudente de lectura
# Configuración validada: Lexi I2S Mic, INMP441 (L/R a GND)
# ==========================================================

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_TEXT = "el perro corre por el parque"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "vosk-model-small-es-0.42"
DEVICE = "hw:2,0"
GAIN_DB = 6
RATE = 16000
CHUNK_SIZE = 4000


def make_audio_pipeline(device, gain_db):
    """Captura 48 kHz/32 bits/2 canales; SoX convierte canal 1 a mono 16 kHz."""
    recorder = subprocess.Popen(
        ["arecord", "-q", "-D", device, "-f", "S32_LE", "-r", "48000",
         "-c", "2", "-t", "raw"],
        stdout=subprocess.PIPE,
    )
    try:
        converter = subprocess.Popen(
            ["sox", "-q", "-t", "raw", "-r", "48000",
             "-e", "signed-integer", "-b", "32", "-c", "2", "-",
             "-t", "raw", "-r", "16000", "-e", "signed-integer", "-b", "16",
             "-c", "1", "-", "remix", "1", "highpass", "80",
             "gain", str(gain_db)],
            stdin=recorder.stdout,
            stdout=subprocess.PIPE,
        )
        recorder.stdout.close()
        return recorder, converter
    except Exception:
        recorder.terminate()
        recorder.wait(timeout=5)
        raise


def stop_process(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    if proc.stdout:
        proc.stdout.close()


def add_words(vosk_result, word_events):
    # Solo resultados finales; los parciales pueden cambiar constantemente.
    for item in vosk_result.get("result", []):
        if all(key in item for key in ("word", "start", "end")):
            word_events.append(WordEvent(str(item["word"]),
                                         float(item["start"]),
                                         float(item["end"])))


def make_report(expected_text, word_events, final_parts, device, gain_db):
    """Conserva las diferencias de Vosk sin declararlas errores del lector.

    Si falta el principio/final, el WPM de la lectura completa es indeterminado:
    Vosk solo cronometró el fragmento reconocido.
    """
    metrics = align_and_score(expected_text, word_events)
    data = metrics.to_dict()
    expected_words = expected_text.split()
    item_results = data["words"]
    correct = sum(item["status"] == "ok" for item in item_results)
    recognized = len(word_events)
    missing_prefix = 0
    for item in item_results:
        if item["status"] == "omission":
            missing_prefix += 1
        else:
            break
    missing_suffix = 0
    for item in reversed(item_results):
        if item["status"] == "omission":
            missing_suffix += 1
        else:
            break

    missing_boundary = bool(missing_prefix or missing_suffix)
    coverage = round(correct / len(expected_words), 3) if expected_words else 0.0
    reliable_for_full_duration = bool(recognized and not missing_boundary)
    # Las medidas siguen siendo ESTIMACIONES, nunca diagnósticos.
    fragment_wpm = data["wpm"]
    data["wpm_recognized_fragment"] = fragment_wpm
    data["wpm"] = fragment_wpm if reliable_for_full_duration else None
    data["wpm_interpretation"] = (
        "Estimación basada en primeras y últimas palabras reconocidas; "
        "verificar posibles errores de ASR y tiempos."
        if reliable_for_full_duration else
        "No calculable para la lectura completa: Vosk no reconoció el inicio "
        "o el final del texto."
    )
    data["duration_is_recognized_fragment_only"] = True
    data["expected_text"] = expected_text
    data["recognized_text"] = " ".join(final_parts)
    data["device"] = device
    data["gain_db"] = gain_db
    data["matching_word_fraction"] = coverage
    data["missing_expected_prefix_words"] = missing_prefix
    data["missing_expected_suffix_words"] = missing_suffix
    data["requires_manual_review"] = (
        missing_boundary or any(item["status"] != "ok" for item in item_results)
        or not recognized
    )
    # El estado de alineación es una DIFERENCIA de transcripción, no un diagnóstico.
    data["differences_to_review"] = [
        {"expected": item["expected"], "recognized": item["said"],
         "asr_status": item["status"],
         "possible_capture_loss": bool(item["status"] == "omission" and
                                      (idx < missing_prefix or
                                       idx >= len(item_results) - missing_suffix))}
        for idx, item in enumerate(item_results) if item["status"] != "ok"
    ]
    data["error_counts_interpretation"] = (
        "Conteos de diferencias detectadas por Vosk, no errores confirmados del niño."
    )
    data["difficult_words_interpretation"] = (
        "Palabras con diferencias de transcripción, NO dificultades lectoras confirmadas."
    )
    return data


def main():
    parser = argparse.ArgumentParser(description="Lexi: STT en vivo + métricas prudentes")
    parser.add_argument("--texto", default=DEFAULT_TEXT, help="Texto esperado de lectura")
    parser.add_argument("--dispositivo", default=DEVICE, help="ALSA; ej. hw:2,0")
    parser.add_argument("--gain", type=float, default=GAIN_DB,
                        help="Ganancia fija en dB (por defecto 6)")
    args = parser.parse_args()

    if not args.texto.strip():
        parser.error("--texto no puede estar vacío")
    if not MODEL_DIR.is_dir():
        print("ERROR: no se encontró el modelo:", MODEL_DIR)
        return 1
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except ImportError:
        print("ERROR: activa el entorno: source ~/Lexi/STT/venv/bin/activate")
        return 1

    SetLogLevel(-1)
    print("Cargando modelo Vosk...", flush=True)
    model = Model(str(MODEL_DIR))
    recognizer = KaldiRecognizer(model, RATE)
    recognizer.SetWords(True)
    events = []
    final_parts = []
    recorder = converter = None
    started = False
    stream_error = False
    print("\nTEXTO QUE SE DEBE LEER:\n  " + repr(args.texto))
    input("\nPulsa ENTER para preparar el micrófono. NO hables todavía...")

    try:
        recorder, converter = make_audio_pipeline(args.dispositivo, args.gain)
        # Estabilización: drenar 1 s de audio ANTES de dar la señal de comenzar.
        # No se pierden palabras mientras el usuario espere la indicación.
        warmup = RATE * 2  # 16 kHz x 2 bytes x 1 segundo
        while warmup:
            chunk = converter.stdout.read(min(CHUNK_SIZE, warmup))
            if not chunk:
                raise RuntimeError("No llegó audio al preparar el micrófono.")
            warmup -= len(chunk)
        started = True
        print("\nESCUCHANDO... Empieza a leer AHORA."
              " Al terminar espera 2 s y pulsa Ctrl+C.\n", flush=True)
        last_partial = ""
        while True:
            chunk = converter.stdout.read(CHUNK_SIZE)
            if not chunk:
                stream_error = True
                print("ERROR: se interrumpió el flujo de audio. Revisa ALSA/SoX.")
                break
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                add_words(result, events)
                phrase = result.get("text", "").strip()
                if phrase:
                    final_parts.append(phrase)
                    print("FINAL:", phrase, flush=True)
                    last_partial = ""
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                if partial and partial != last_partial:
                    print("Parcial:", partial, flush=True)
                    last_partial = partial
    except KeyboardInterrupt:
        print("\nFinalizando la sesión...", flush=True)
    except (OSError, RuntimeError) as exc:
        stream_error = True
        print("ERROR:", exc, flush=True)
    finally:
        if started:
            try:
                result = json.loads(recognizer.FinalResult())
                add_words(result, events)
                phrase = result.get("text", "").strip()
                if phrase:
                    final_parts.append(phrase)
                    print("FINAL:", phrase)
            except Exception as exc:
                print("Advertencia al finalizar:", exc)
        stop_process(converter)
        stop_process(recorder)

    if not started:
        print("No hubo captura válida. No se calcularon métricas.")
        return 1

    report = make_report(args.texto, events, final_parts,
                         args.dispositivo, args.gain)
    if stream_error:
        report["requires_manual_review"] = True
        report["stream_interrupted"] = True
        report["wpm"] = None
        report["wpm_interpretation"] = "Interrupción de audio; velocidad no válida."

    print("\nTRANSCRIPCIÓN FINAL:")
    print(report["recognized_text"] or "(sin palabras reconocidas)")
    print("\nMÉTRICAS ESTIMADAS:")
    print("Coincidencias:", sum(w["status"] == "ok" for w in report["words"]),
          "/", len(report["words"]))
    if report["wpm"] is None:
        print("Velocidad de lectura completa: NO DISPONIBLE (transcripción incompleta)")
    else:
        print("Velocidad aproximada del fragmento reconocido:", report["wpm"], "PPM")
    print("Duración del audio reconocido:", report["total_duration_sec"], "s")
    print("Pausa promedio estimada:", report["avg_pause_sec"], "s")
    print("Diferencias de transcripción (NO errores confirmados):")
    if not report["differences_to_review"]:
        print("  Ninguna detectada")
    for item in report["differences_to_review"]:
        print("  -", item["expected"], "->", item["recognized"] or "(no captado)",
              "[posible pérdida de captura]" if item["possible_capture_loss"] else
              "[revisar transcripción]")
    if report["requires_manual_review"]:
        print("ATENCIÓN: revisar diferencias antes de evaluar lectura.")

    outfile = Path(__file__).resolve().parent / (
        "lexi_resultado_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    outfile.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResultado JSON:", outfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
