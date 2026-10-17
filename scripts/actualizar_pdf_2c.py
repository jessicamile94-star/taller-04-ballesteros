"""Actualiza las páginas del informe del Taller 4 y conserva el anexo del Taller 3."""
from __future__ import annotations

import shutil
import argparse
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "taller4-ballesteros/output/pdf/Informe_Taller4_Ballesteros.pdf"
COPY = ROOT / "informe/Informe_Taller4_Ballesteros.pdf"
FIG = ROOT / "taller4-ballesteros/comparacion_2c.png"


def redact(page: pymupdf.Page, rect: pymupdf.Rect) -> None:
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)


def replace_block(page: pymupdf.Page, needle: str, replacement: str,
                  fontsize: float = 8.7, padding: float = 2) -> pymupdf.Rect:
    block = next((b for b in page.get_text("blocks") if needle in b[4]), None)
    if block is None:
        raise ValueError(f"No se encontró el texto para reemplazar: {needle}")
    rect = pymupdf.Rect(block[:4])
    clear = pymupdf.Rect(rect.x0 - padding, rect.y0 - padding,
                         rect.x1 + padding, rect.y1 + padding)
    redact(page, clear)
    left = page.insert_textbox(rect, replacement, fontname="helv", fontsize=fontsize,
                               lineheight=1.12, color=(0, 0, 0), overlay=True)
    if left < 0:
        raise ValueError(f"Texto nuevo no cabe en el espacio: {needle}")
    return rect


def add_table(page: pymupdf.Page, x: float, y: float, widths: list[float],
              headers: list[str], rows: list[list[str]], row_h: float = 20,
              fontsize: float = 8.0) -> float:
    all_rows = [headers, *rows]
    total_w = sum(widths)
    for ri, row in enumerate(all_rows):
        top = y + ri * row_h
        bg = (0.88, 0.92, 0.96) if ri == 0 else ((0.97, 0.98, 0.99) if ri % 2 else (1, 1, 1))
        x0 = x
        for ci, (cell, width) in enumerate(zip(row, widths)):
            r = pymupdf.Rect(x0, top, x0 + width, top + row_h)
            page.draw_rect(r, color=(0.72, 0.78, 0.83), fill=bg, width=0.4, overlay=True)
            page.insert_textbox(pymupdf.Rect(r.x0 + 5, r.y0 + 4, r.x1 - 4, r.y1 - 2),
                                cell, fontname="hebo" if ri == 0 else "helv",
                                fontsize=fontsize, color=(0, 0, 0),
                                lineheight=1.0, overlay=True)
            x0 += width
    return y + len(all_rows) * row_h


def add_comparison_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(pno=8, width=612, height=792)
    page.insert_text((56, 52), "Parte 2.c. Comparación del prompt versionado",
                     fontname="hebo", fontsize=14, color=(0, 0, 0))
    p1 = ("El cambio se enfocó en el código extenso de S1/T5, S2/T5 y S2/T7. Registré "
          "`taller4-solver-programador` como versión 1 en Langfuse, con la etiqueta "
          "`taller4-2c-v2` y un changelog. Las 27 generaciones del programador quedaron enlazadas.")
    p2 = ("Corrí las siete tareas con el mismo golden de 55 checks, el mismo evaluador "
          "automático y `zai-org/GLM-5.3-Flash`. La tasa incluye 48 checks determinísticos; "
          "los siete de juez se revisan aparte.")
    page.insert_textbox(pymupdf.Rect(56, 65, 556, 111), p1, fontname="helv", fontsize=8.6,
                        lineheight=1.12, color=(0, 0, 0))
    page.insert_textbox(pymupdf.Rect(56, 113, 556, 148), p2, fontname="helv", fontsize=8.6,
                        lineheight=1.12, color=(0, 0, 0))
    page.insert_image(pymupdf.Rect(86, 147, 526, 373), filename=str(FIG))
    page.insert_text((56, 390), "Resultado por tarea", fontname="hebo", fontsize=10,
                     color=(0, 0, 0))
    table_bottom = add_table(
        page, 56, 398, [115, 185, 185],
        ["Tarea", "Base", "Prompt v2"],
        [["A", "9/9", "9/9"], ["B", "9/9", "9/9"], ["C", "8/8", "8/8"],
         ["S1", "9/9", "8/9"], ["S2", "11/12", "12/12"], ["E", "1/1", "0/1"],
         ["P", "No aplica", "No aplica"]], row_h=18, fontsize=7.8)
    page.insert_text((56, table_bottom + 18), "Checks que cambiaron", fontname="hebo",
                     fontsize=9.5, color=(0, 0, 0))
    change_bottom = add_table(
        page, 56, table_bottom + 24, [65, 184, 184, 52],
        ["Check", "Base", "Prompt v2", "Efecto"],
        [["S208", "No halló 0.2040", "Halló 0.2040", "Mejora"],
         ["S103", "Secciones en orden", "Falta ‘Tarea 2’", "Regresión"],
         ["E01", "Contenido presente", "Falta ‘media aritmética’", "Regresión"]],
        row_h=20, fontsize=7.2)
    note = (
        "En los casos objetivo, S1/T5 generó el Parquet requerido, aunque necesitó tres intentos; "
        "S2/T5 y S2/T7 produjeron código ejecutable al primer intento y S208 pasó. "
        "La traza de S2 registra T7 con 18,419 tokens, finish_reason=stop y sandbox aprobado. "
        "La corrida v2 sumó 272,323 tokens de entrada y 550,871 de salida, frente a 339,368 y 885,813 en la base. "
        "El total bajó una aprobación por S103 y E01. Hubo una corrida por versión y no se fijó una "
        "semilla, así que no puedo separar el efecto del prompt de la variación del modelo. La tabla "
        "completa está en `comparacion_2c.csv`; los siete checks de juez no se cuentan como fallidos."
    )
    page.insert_textbox(pymupdf.Rect(56, change_bottom + 11, 556, 782), note,
                        fontname="helv", fontsize=7.5, lineheight=1.12,
                        color=(0, 0, 0), overlay=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PDF,
                        help="PDF original del informe (incluye el anexo del Taller 3)")
    source = parser.parse_args().source
    doc = pymupdf.open(source)

    # Resumen actualizado (página 2).
    replace_block(
        doc[1], "En este taller instrumenté",
        "En este taller instrumenté y evalué el solver multiagente que venía trabajando. Revisé "
        "siete tareas, medí sus trazas, probé los guardrails con entradas de ataque y datos "
        "personales sintéticos, y comparé los resultados con el golden set. La corrida base aprobó "
        "47 de los 48 checks determinísticos; los siete criterios conceptuales se revisaron aparte. "
        "Para la Parte 2.c registré una nueva versión del prompt y repetí las siete tareas con el "
        "mismo golden, evaluador y modelo. S1 y S2 completaron sus entregables y S208 pasó, pero el "
        "total bajó a 46/48 por dos regresiones en S103 y E01. Una corrida por versión no basta "
        "para atribuir todos los cambios al prompt.", fontsize=8.25)

    # El párrafo que cerraba la comparación histórica (página 3) remite a la 2.c actualizada.
    replace_block(doc[2], "Al comparar 47 checks determinísticos",
                  "La comparación del prompt v2, medida sobre el golden completo, se presenta en la Parte 2.c.",
                  fontsize=8.5)

    # Reemplaza la sección que afirmaba que S1 y S2 seguían parciales (página 4).
    p4 = doc[3]
    redact(p4, pymupdf.Rect(52, 48, 560, 158))
    p4.insert_text((56, 67), "Corridas base incompletas y nueva medición",
                   fontname="hebo", fontsize=10.5, color=(0, 0, 0))
    left = p4.insert_textbox(
        pymupdf.Rect(56, 75, 556, 154),
        "En la base y los reintentos anteriores, S1/T5 y S2/T7 quedaron pendientes por salidas "
        "truncadas y presupuesto. En la corrida v2, S1 terminó T1-T5; T5 necesitó tres intentos y "
        "T4 llegó al límite de 32,768 tokens antes de completarse con el siguiente límite. S2 terminó "
        "sus seis subtareas de cálculo, incluida T7, en el primer intento. Las trazas nuevas están "
        "en `corridas/taller4-2c-v2/`.",
        fontname="helv", fontsize=8.2, lineheight=1.1, color=(0, 0, 0), overlay=True)
    if left < 0:
        raise ValueError("El resumen de corridas base y v2 no cabe en la página 4")
    redact(p4, pymupdf.Rect(54, 577, 558, 595))
    p4.insert_text((56, 589.3), "Figura 4. Frecuencias observadas frente a las teóricas en la tarea B.",
                   fontname="helv", fontsize=7.4, color=(0, 0, 0))

    # Redacta la continuación antigua de la 2.c y actualiza la respuesta de la Parte 4.a.
    p9 = doc[8]
    redact(p9, pymupdf.Rect(52, 48, 560, 245))
    old4a = next((b for b in p9.get_text("blocks") if "Con los resultados actuales no ajustaría" in b[4]), None)
    if old4a is None:
        raise ValueError("No se encontró la respuesta anterior de la Parte 4.a")
    r4a = pymupdf.Rect(54, old4a[1] - 3, 558, 744)
    redact(p9, r4a)
    part4a = (
        "Con estos resultados no ajustaría los pesos del modelo. La base falló S208 porque no presentó "
        "el p-value exacto por cambios de signo (0.2040), pero ese valor sí apareció en la corrida v2. "
        "S1/T5 y S2/T7 también terminaron, aunque S1/T5 necesitó tres intentos y hubo una respuesta "
        "truncada en T4. Estos casos apuntan más al manejo de longitud, los reintentos y el traspaso de "
        "resultados que a falta de capacidad estadística.\n\n"
        "La comparación aprobó 47/48 checks determinísticos en la base y 46/48 con v2. S103 no incluyó "
        "el título literal «Tarea 2» y E01 omitió la frase «media aritmética»; ambos son fallos de formato "
        "o cobertura del reporte. El cambio se enfocó en tres subtareas de cálculo, no en esos checks. "
        "Como corrí cada versión una sola vez y el modelo puede variar sus respuestas, no puedo atribuir "
        "con certeza los cambios al prompt. Repetiría varias veces el mismo golden, con las versiones "
        "registradas en Langfuse y la misma configuración, antes de afirmar una mejora. Con siete tareas "
        "y este golden pequeño tampoco tengo evidencia suficiente para justificar un ajuste supervisado "
        "o por preferencias."
    )
    left = p9.insert_textbox(r4a, part4a, fontname="helv", fontsize=8.2, lineheight=1.1,
                             color=(0, 0, 0), overlay=True)
    if left < 0:
        raise ValueError("La respuesta actualizada de la Parte 4.a no cabe en la página 9")

    cropped = pymupdf.open()
    cp = cropped.new_page(width=612, height=792)
    cp.show_pdf_page(pymupdf.Rect(0, 55, 612, 602), doc, 8,
                     clip=pymupdf.Rect(0, 245, 612, 792))
    doc.delete_page(8)
    doc.insert_pdf(cropped, from_page=0, to_page=0, start_at=8)
    cropped.close()

    add_comparison_page(doc)

    tmp = PDF.with_name("Informe_Taller4_Ballesteros.tmp.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(PDF)
    COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, COPY)
    print(f"PDF actualizado: {PDF}")
    print(f"Copia actualizada: {COPY}")


if __name__ == "__main__":
    main()
