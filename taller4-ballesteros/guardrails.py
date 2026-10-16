"""
Guardrails robustos para el Solver (Taller 4 - Parte 3).
Resuelve las debilidades identificadas en la Parte 0.b y 0.c:
1. Normalización de texto (minúsculas, sin tildes, espacios colapsados) y detección de intenciones de inyección.
2. Detección y redacción precisa de PII (correos, cédulas de 10 dígitos y teléfonos), sin romper fechas (ISO) ni listas de números.
3. Detección y protección en salida (claves de API por patrón estructurado, hashes vs secretos).
"""

from __future__ import annotations

import re
import unicodedata

# 1. Normalización de texto
def normalizar(texto: str) -> str:
    """Convierte a minúsculas, elimina diacríticos/tildes y normaliza espacios."""
    if not texto:
        return ""
    # Descomposición canónica
    nfkd = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(sin_tildes.lower().split())


# Patrones de inyección sobre texto normalizado (sin tildes)
INJECTION_INTENTS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"olvida\s+(todas\s+)?(las|tus)?\s*instrucciones",
    r"ignora\s+(todas\s+)?(las|tus)?\s*instrucciones",
    r"revela\s+(el\s+)?(system\s*prompt|prompt\s*del\s*sistema|tus\s*reglas)",
    r"traduce\s+.*tus\s*reglas\s*internas",
    r"you\s+are\s+now",
    r"actua\s+como\s+si\s+no\s+tuvieras\s+restricciones",
    r"eres\s+un\s+asistente\s+sin\s+reglas",
    r"desactiva\s+el\s+sandbox",
    r"ejecuta\s+cualquier\s+comando",
    r"copia\s+todas\s+las\s+variables\s+de\s+entorno",
    r"dame\s+la\s+clave\s+de\s+la\s+api",
]

# Exclusiones legítimas (preguntas académicas o discusiones que contienen términos clave pero sin intención de ataque)
LEGITIMATE_CONTEXTS = [
    r"por\s+que\s+una\s+instruccion\s+escondida",
    r"explica\s+.*instruccion\s+escondida",
    r"solicita\s+documentar\s+por\s+que\s+el\s+sandbox",
    r"como\s+actua\s+el\s+descuento",
]

def detect_prompt_injection(texto: str) -> bool:
    norm = normalizar(texto)
    # Si coincide con un contexto académico legítimo, pasa
    if any(re.search(pat, norm) for pat in LEGITIMATE_CONTEXTS):
        return False
    return any(re.search(pat, norm) for pat in INJECTION_INTENTS)


# 2. Detección y redacción de PII
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Cédula ecuatoriana: exactamente 10 dígitos aislados
CEDULA_RE = re.compile(r"(?<!\d)(?:0[1-9]|1[0-9]|2[0-4]|30)\d{8}(?!\d)")

# Teléfono: números con prefijo + o código móvil (09), o con formato telefónico claro (no fechas año-mes-día)
PHONE_RE = re.compile(r"(?<!\d|\-)(?:\+?593[\s\-]?)?(?:09\d{8}|09\d[\s\-]\d{3}[\s\-]\d{3,4}|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b)(?!\d|\-)")

def redact_pii(texto: str) -> str:
    if not texto:
        return ""
    # Redactar emails
    texto = EMAIL_RE.sub("[EMAIL_REDACTADO]", texto)
    # Redactar teléfonos
    texto = PHONE_RE.sub("[TELEFONO_REDACTADO]", texto)
    # Redactar cédulas
    texto = CEDULA_RE.sub("[CEDULA_REDACTADA]", texto)
    return texto


# 3. Secretos y credenciales en salida
API_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,}|(?:ghp|gho|glpat|AKIA)[A-Za-z0-9_\-]{16,})\b")

def detect_secret_leak(texto: str) -> bool:
    # Detecta solo claves con prefijo específico o estructura real de token/clave, no hashes genéricos ni UUIDs
    return API_KEY_RE.search(texto) is not None


# Validadores de canal
def validar_entrada(texto: str) -> tuple[bool, str]:
    if detect_prompt_injection(texto):
        return False, "Solicitud rechazada por posible prompt injection."
    redactado = redact_pii(texto)
    return True, redactado


def validar_salida(texto: str) -> tuple[bool, str]:
    if detect_secret_leak(texto):
        return False, "Salida bloqueada por posible filtracion de credenciales."
    redactado = redact_pii(texto)
    return True, redactado
