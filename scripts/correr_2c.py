"""Corre el golden completo con el prompt v2 de la Parte 2.c."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "taller4-ballesteros" / "golden_set.json"
OUT = ROOT / "corridas" / "taller4-2c-v2"


def main() -> int:
    tareas = json.loads(GOLDEN.read_text(encoding="utf-8"))["tareas"]
    estado = []
    OUT.mkdir(parents=True, exist_ok=True)
    for tarea in tareas:
        pdf = (GOLDEN.parent / tarea["pdf"]).resolve()
        carpeta = OUT / f"tarea-{tarea['id']}"
        print(f"\n=== {tarea['id']}: {pdf.name} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", "scripts/correr.py", str(pdf), "--salida", str(carpeta)],
            cwd=ROOT, check=False,
        )
        estado.append({"tarea": tarea["id"], "returncode": result.returncode,
                       "corrida": str(carpeta.relative_to(ROOT))})
        (OUT / "estado.json").write_text(
            json.dumps(estado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fallidas = [r for r in estado if r["returncode"] != 0]
    print(f"\nCorridas terminadas: {len(estado) - len(fallidas)}/{len(estado)} sin excepción.",
          flush=True)
    return 1 if fallidas else 0


if __name__ == "__main__":
    raise SystemExit(main())
