"""Deja en un JSONL de Langfuse solo los spans del trace_id de una corrida local."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trazas_local", type=Path, nargs="+", help="una o más traza.jsonl locales")
    p.add_argument("spans_exportados", type=Path, help="exportación completa de Langfuse")
    p.add_argument("salida", type=Path, help="JSONL filtrado que lee resumir_trazas.py")
    a = p.parse_args()

    ids = set()
    for path in a.trazas_local:
        local = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        propios = {str(row["traza_id"]) for row in local
                   if row.get("tipo") == "langfuse_trace" and row.get("traza_id")}
        if len(propios) != 1:
            raise SystemExit(f"Se esperaba exactamente un traza_id en {path}; encontrados: {len(propios)}")
        ids.update(propios)
    spans = [json.loads(line) for line in a.spans_exportados.read_text(encoding="utf-8").splitlines() if line]
    seleccion = [span for span in spans if str(span.get("traza")) in ids]
    encontradas = {str(span.get("traza")) for span in seleccion if span.get("span") == "corrida"}
    if encontradas != ids:
        raise SystemExit(f"Faltan trazas raíz: {sorted(ids - encontradas)}; espera unos segundos y vuelve a exportar")
    a.salida.write_text("".join(json.dumps(span, ensure_ascii=False) + "\n" for span in seleccion),
                        encoding="utf-8")
    print(f"{len(seleccion)} spans de {len(ids)} corridas → {a.salida}")


if __name__ == "__main__":
    main()
