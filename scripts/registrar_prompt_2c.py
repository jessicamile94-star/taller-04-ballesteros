"""Registra el prompt del programador de la Parte 2.c en Langfuse una sola vez."""
from __future__ import annotations

from dotenv import load_dotenv

from solver.config import RAIZ
from solver.agentes.programador import SISTEMA_V2

NOMBRE = "taller4-solver-programador"
ETIQUETA = "taller4-2c-v2"
CHANGELOG = (
    "v2: reduce programmer output length after truncations in "
    "S1/T5, S2/T5 and S2/T7"
)


def main() -> None:
    load_dotenv(RAIZ / "Lab-04-LLMOps" / ".env", override=False)
    from langfuse import get_client

    client = get_client()
    existentes = client.api.prompts.list(name=NOMBRE, limit=100).data
    if any(ETIQUETA in (item.labels or []) for item in existentes):
        prompt = client.get_prompt(NOMBRE, label=ETIQUETA, type="text", cache_ttl_seconds=0)
        if prompt.prompt != SISTEMA_V2:
            raise SystemExit(f"La etiqueta {ETIQUETA} ya existe con otro contenido.")
        print(f"Prompt verificado: {prompt.name}, versión {prompt.version}, etiqueta {ETIQUETA}")
        return

    prompt = client.create_prompt(
        name=NOMBRE,
        type="text",
        prompt=SISTEMA_V2,
        labels=[ETIQUETA],
        commit_message=CHANGELOG,
    )
    print(f"Prompt registrado: {prompt.name}, versión {prompt.version}, etiqueta {ETIQUETA}")


if __name__ == "__main__":
    main()
