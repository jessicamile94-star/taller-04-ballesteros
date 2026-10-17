"""C6 — La traza se escribe en todo camino.

Cada evento es una línea JSON en traza.jsonl, escrita y cerrada en el momento: si la
corrida muere a la mitad, todo lo anterior ya está en disco. Nada se guarda en memoria
esperando un «al final».

Tipos de evento (los campos que P2 suma en la 2.b y lee en la 2.c):
  llamada    agente, subtarea, modelo, tokens_entrada, tokens_salida, latencia_s, fin,
             intento, error
  ejecucion  agente, subtarea, intento, returncode, duracion_s, archivos, error
  decision   agente, subtarea, decision, motivo      (lo que decidió el CÓDIGO)
  error      agente, subtarea, error                 (excepciones capturadas)
  inicio / fin de corrida

Antes de escribir, todo valor de texto pasa por enmascarar(): una traza guarda mensajes
enteros y se sube al repositorio.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from collections import defaultdict
from pathlib import Path

_PATRONES = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)['\"]?[^\s'\",]{8,}"),
]
_NOMBRES_SECRETOS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _secretos_del_entorno() -> list[str]:
    return [v for k, v in os.environ.items()
            if any(s in k.upper() for s in _NOMBRES_SECRETOS) and len(v) >= 8]


def enmascarar(texto: str) -> str:
    # Los datos personales sintéticos de las tareas P no deben entrar a trazas locales
    # ni a Langfuse, incluso cuando aparecen en prompts o excepciones.
    try:
        from solver.guardrails import redact_pii
        texto = redact_pii(texto)
    except Exception:  # la sanitización de trazas no debe impedir el registro de un error
        pass
    for valor in _secretos_del_entorno():
        texto = texto.replace(valor, "***")
    for patron in _PATRONES:
        if patron.groups >= 2:
            texto = patron.sub(lambda m: f"{m.group(1)}{m.group(2)}***", texto)
        else:
            texto = patron.sub("***", texto)
    return texto


def _limpiar(valor):
    if isinstance(valor, str):
        return enmascarar(valor)
    if isinstance(valor, dict):
        # Opaque telemetry IDs are hex strings; a 10-digit substring can look like a
        # national ID and must not be rewritten, or local and remote trace IDs diverge.
        return {k: (v if k in {"traza_id", "trace_id"} and isinstance(v, str) else _limpiar(v))
                for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpiar(v) for v in valor]
    if isinstance(valor, Path):
        return str(valor)
    return valor


class Traza:
    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._tokens = defaultdict(lambda: [0, 0])   # agente -> [entrada, salida]
        # El indexador llama al LLM en paralelo (capa 2): numerar, sumar y escribir es atómico.
        self._candado = threading.RLock()
        self._langfuse = None
        self._langfuse_habilitado = False
        try:
            from dotenv import load_dotenv
            from solver.config import RAIZ
            load_dotenv(RAIZ / "Lab-04-LLMOps" / ".env", override=False)
            from langfuse import get_client
            client = get_client()
            # Langfuse is opt-in so local tests don't upload fixture traces. Taller 4 runs
            # explicitly set USE_LANGFUSE=true in the local environment.
            if (os.getenv("USE_LANGFUSE", "false").lower() == "true"
                    and not os.getenv("PYTEST_CURRENT_TEST")
                    and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
                self._langfuse = client
                self._langfuse_habilitado = True
        except Exception:
            self._langfuse = None

    @contextmanager
    def corrida_langfuse(self, nombre: str, entrada: str = ""):
        """Raíz Langfuse de la corrida; si no hay credenciales mantiene el flujo local."""
        if self._langfuse is None:
            yield None
            return
        stack = ExitStack()
        try:
            from langfuse import propagate_attributes
            session = os.getenv("LANGFUSE_SESSION_ID", "taller4-ballesteros-base")
            etiquetas = ["taller-04", "solver-b"]
            stack.enter_context(propagate_attributes(session_id=session, tags=etiquetas))
            root = stack.enter_context(self._langfuse.start_as_current_observation(
                as_type="span", name=nombre, input=entrada[:20000], metadata={"prompt_version": "base-v1"}
            ))
            # Langfuse 4.x preserves session_id on observations, but propagated OTel tags
            # may not populate the trace tag list. Enqueue the trace update explicitly.
            traza_id = self._langfuse.get_current_trace_id()
            if traza_id:
                self._langfuse._create_trace_tags_via_ingestion(trace_id=traza_id, tags=etiquetas)
        except Exception:
            stack.close()
            yield None
            return
        try:
            yield root
        except BaseException as exc:
            try:
                root.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            stack.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            stack.close()

    @contextmanager
    def agente_langfuse(self, nombre: str):
        if self._langfuse is None:
            yield
            return
        try:
            cm = self._langfuse.start_as_current_observation(as_type="agent", name=nombre)
            agent_span = cm.__enter__()
        except Exception:
            yield
            return
        try:
            yield
        except BaseException as exc:
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            cm.__exit__(None, None, None)

    @contextmanager
    def generacion_langfuse(self, agente: str, modelo: str, entrada):
        if self._langfuse is None:
            yield None
            return
        try:
            cm = self._langfuse.start_as_current_observation(
                as_type="generation", name=agente, model=modelo, input=_limpiar(entrada),
                metadata={"prompt_version": "base-v1"})
            gen = cm.__enter__()
        except Exception:
            yield None
            return
        try:
            yield gen
        except BaseException as exc:
            try:
                gen.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            cm.__exit__(None, None, None)

    @contextmanager
    def herramienta_langfuse(self, nombre: str, entrada):
        if self._langfuse is None:
            yield None
            return
        try:
            cm = self._langfuse.start_as_current_observation(
                as_type="tool", name=nombre, input=_limpiar(entrada))
            span = cm.__enter__()
        except Exception:
            yield None
            return
        try:
            yield span
        except BaseException as exc:
            try:
                span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            cm.__exit__(None, None, None)

    def actualizar_input_corrida(self, texto: str) -> None:
        if self._langfuse is None:
            return
        try:
            if self.traza_id:
                self._langfuse.update_current_span(input=texto[:20000])
        except Exception:
            pass

    def actualizar_output_corrida(self, datos: dict) -> None:
        if self._langfuse is None:
            return
        try:
            self._langfuse.update_current_span(output=_limpiar(datos))
        except Exception:
            pass

    @property
    def traza_id(self) -> str | None:
        if self._langfuse is None:
            return None
        try:
            return self._langfuse.get_current_trace_id()
        except Exception:
            return None

    def flush_langfuse(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:
                pass

    def _observar(self, tipo: str, campos: dict) -> None:
        if self._langfuse is None or tipo != "guardrail":
            return
        clase = "guardrail"
        nombre = campos.get("agente") or campos.get("decision") or tipo
        datos = {k: v for k, v in campos.items() if k not in {"agente", "tokens_entrada", "tokens_salida"}}
        datos = _limpiar(datos)
        try:
            kwargs = {"as_type": clase, "name": str(nombre), "input": datos.get("entrada")}
            with self._langfuse.start_as_current_observation(**kwargs) as span:
                span.update(output=datos.get("salida") or datos.get("decision") or datos.get("archivos"),
                            metadata={k: v for k, v in datos.items() if k not in {"entrada", "salida"}})
                if datos.get("error"):
                    span.update(level="ERROR", status_message=str(datos["error"])[:500])
        except Exception:
            pass

    # ------------------------------------------------------------------ escritura
    def evento(self, tipo: str, **campos) -> dict:
        if tipo == "guardrail":
            from solver.guardrails import GUARDRAIL_VERSION
            campos.setdefault("guardrail_version", GUARDRAIL_VERSION)
        campos = _limpiar(campos)
        self._observar(tipo, campos)
        with self._candado:
            self._seq += 1
            fila = {"seq": self._seq, "ts": round(time.time(), 3), "tipo": tipo, **campos}
            with self.ruta.open("a", encoding="utf-8") as f:   # abrir-escribir-cerrar: sobrevive a un crash
                f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        return fila

    def llamada(self, agente: str, *, modelo: str, tokens_entrada: int, tokens_salida: int,
                latencia_s: float, fin: str | None, intento: int = 1, subtarea: str | None = None,
                error: str | None = None, **extra) -> dict:
        with self._candado:
            self._tokens[agente][0] += tokens_entrada
            self._tokens[agente][1] += tokens_salida
            return self.evento("llamada", agente=agente, subtarea=subtarea, modelo=modelo,
                               tokens_entrada=tokens_entrada, tokens_salida=tokens_salida,
                               latencia_s=latencia_s, fin=fin, intento=intento, error=error, **extra)

    def ejecucion(self, agente: str, *, subtarea: str, intento: int, returncode: int | None,
                  duracion_s: float, archivos: list[str], error: str | None = None, **extra) -> dict:
        return self.evento("ejecucion", agente=agente, subtarea=subtarea, intento=intento,
                           returncode=returncode, duracion_s=duracion_s, archivos=archivos,
                           error=error, **extra)

    def decision(self, agente: str, decision: str, motivo: str = "",
                 subtarea: str | None = None, **extra) -> dict:
        return self.evento("decision", agente=agente, subtarea=subtarea, decision=decision,
                           motivo=motivo, **extra)

    def error(self, agente: str, error: BaseException | str, subtarea: str | None = None) -> dict:
        texto = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
        return self.evento("error", agente=agente, subtarea=subtarea, error=texto)

    # ------------------------------------------------------------------ lectura
    @property
    def tokens_por_agente(self) -> dict[str, dict[str, int]]:
        return {a: {"tokens_entrada": e, "tokens_salida": s} for a, (e, s) in self._tokens.items()}

    @property
    def tokens_totales(self) -> dict[str, int]:
        return {"tokens_entrada": sum(e for e, _ in self._tokens.values()),
                "tokens_salida": sum(s for _, s in self._tokens.values())}


def leer(ruta: str | Path) -> list[dict]:
    """Para P2: las filas de una traza, listas para un DataFrame."""
    return [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines() if l.strip()]
