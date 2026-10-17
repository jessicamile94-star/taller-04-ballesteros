"""Compara los checks del golden entre base y la corrida del prompt v2."""
from __future__ import annotations

import csv
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "taller4-ballesteros"
BASE = DIR / "resultados_solver_2c_base.csv"
V2 = DIR / "resultados_solver_2c_v2.csv"
OUT = DIR / "comparacion_2c.csv"
FIG = DIR / "comparacion_2c.png"


def leer(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {(r["tarea"], r["check"]): r for r in csv.DictReader(f)}


def main() -> None:
    antes, despues = leer(BASE), leer(V2)
    if antes.keys() != despues.keys():
        raise SystemExit("Los CSV no contienen los mismos checks del golden.")
    rows = []
    cambios = []
    validos = []
    for key in sorted(antes):
        a, b = antes[key], despues[key]
        juez = a["tipo"] == "juez" or b["tipo"] == "juez"
        cambio = ("juez separado" if juez else
                  "sin cambio" if a["ok"] == b["ok"] else
                  "mejora" if a["ok"] == "0" and b["ok"] == "1" else "regresión")
        row = {"tarea": key[0], "check": key[1], "tipo": a["tipo"],
               "base": "" if juez else a["ok"], "v2": "" if juez else b["ok"],
               "cambio": cambio, "detalle_base": a["detalle"], "detalle_v2": b["detalle"]}
        rows.append(row)
        if juez:
            continue
        validos.append(row)
        if cambio in {"mejora", "regresión"}:
            cambios.append(row)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total_base = sum(int(r["base"]) for r in validos)
    total_v2 = sum(int(r["v2"]) for r in validos)
    print(f"Determinísticos: base {total_base}/{len(validos)}; v2 {total_v2}/{len(validos)}")
    for task in sorted({r["tarea"] for r in validos}):
        a = [r for r in validos if r["tarea"] == task]
        print(f"{task}: base {sum(int(r['base']) for r in a)}/{len(a)}; "
              f"v2 {sum(int(r['v2']) for r in a)}/{len(a)}")
    print("Checks que cambiaron:")
    for r in cambios:
        print(f"{r['tarea']}/{r['check']}: {r['base']}→{r['v2']} ({r['cambio']}); "
              f"{r['detalle_base']} → {r['detalle_v2']}")
    maximo = len(validos)
    y0, alto = 365, 270
    def y(score: int) -> float:
        return y0 - alto * score / maximo
    grid = "".join(
        f'<line x1="125" y1="{y(v):.1f}" x2="820" y2="{y(v):.1f}" stroke="#d7d7d7" />'
        f'<text x="105" y="{y(v)+5:.1f}" text-anchor="end" fill="#000">{v}</text>'
        for v in range(0, maximo + 1, 8)
    )
    bars = "".join(
        f'<rect x="{x}" y="{y(score):.1f}" width="180" height="{y0-y(score):.1f}" fill="{color}" />'
        f'<text x="{x+90}" y="{y(score)-12:.1f}" text-anchor="middle" fill="#000" '
        f'font-weight="bold">{score}/{maximo}</text>'
        f'<text x="{x+90}" y="{y0+32}" text-anchor="middle" fill="#000">{label}</text>'
        for x, score, color, label in (
            (270, total_base, "#8594a5", "Base"),
            (560, total_v2, "#355c7d", "Prompt v2"),
        )
    )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460" '
           f'viewBox="0 0 900 460"><rect width="100%" height="100%" fill="white"/>'
           f'<text x="450" y="46" text-anchor="middle" fill="#000" font-size="23" '
           f'font-family="Arial">Comparación del golden set</text>{grid}{bars}'
           f'<line x1="125" y1="{y0}" x2="820" y2="{y0}" stroke="#000" />'
           f'<text x="30" y="220" text-anchor="middle" fill="#000" font-size="16" '
           f'font-family="Arial" transform="rotate(-90 30,220)">Checks determinísticos aprobados '
           f'(de {maximo})</text></svg>')
    doc = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(FIG)
    doc.close()
    print(f"Comparación completa: {OUT.relative_to(ROOT)}")
    print(f"Figura: {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
