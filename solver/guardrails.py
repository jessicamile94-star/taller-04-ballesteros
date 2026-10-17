"""Guardrails para documentos, resultados publicados y datos de traza (Taller 4, ruta B)."""
from __future__ import annotations

import re
import unicodedata

GUARDRAIL_VERSION = "solver-guardrails-v1"

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CEDULA = re.compile(r"(?<!\d)(?:0[1-9]|1\d|2[0-4]|30)\d{8}(?!\d)")
TELEFONO = re.compile(r"(?<![\d-])(?:\+?593[\s-]?)?(?:09\d{8}|09\d[\s-]\d{3}[\s-]\d{3,4}|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b)(?![\d-])")
SECRETO = re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|(?:ghp|gho|glpat|AKIA)[A-Za-z0-9_-]{16,})\b")

_INTENCIONES = (
    r"ignora\s+(?:todas\s+)?(?:las\s+|tus\s+)?instrucciones",
    r"olvida\s+(?:todas\s+)?(?:las\s+|tus\s+)?instrucciones",
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"desactiva\s+el\s+sandbox",
    r"copia\s+(?:todas\s+)?las\s+variables\s+de\s+entorno",
    r"(?:revela|imprime|muestra)\s+(?:el\s+)?(?:system\s*prompt|prompt\s+del\s+sistema|tus\s+reglas)",
    r"dame\s+la\s+clave\s+de\s+la\s+api",
)


def normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto or "")
    return " ".join("".join(c for c in descompuesto if not unicodedata.combining(c)).lower().split())


def detect_prompt_injection(texto: str) -> bool:
    normal = normalizar(texto)
    # Menciones académicas explicativas no son instrucciones dirigidas al solver.
    if re.search(r"(?:explica|documenta|describe)\s+.{0,100}(?:por que|como)\s+.{0,80}(?:instruccion escondida|sandbox)", normal):
        return False
    return any(re.search(p, normal) for p in _INTENCIONES)


def quitar_instrucciones_inyectadas(texto: str) -> str:
    """Elimina la línea que contiene la instrucción no confiable del texto almacenado."""
    return "\n".join(linea for linea in (texto or "").splitlines()
                     if not detect_prompt_injection(linea))


def redact_pii(texto: str) -> str:
    texto = EMAIL.sub("[EMAIL_REDACTADO]", texto or "")
    texto = TELEFONO.sub("[TELEFONO_REDACTADO]", texto)
    return CEDULA.sub("[CEDULA_REDACTADA]", texto)


def validar_salida(texto: str) -> tuple[bool, str]:
    if SECRETO.search(texto or ""):
        return False, "[SALIDA_BLOQUEADA_POR_POSIBLE_CREDENCIAL]"
    return True, redact_pii(texto or "")
