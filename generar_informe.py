from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "informe" / "informe_sismo_ai_peru.pdf"
IMG_DIR = BASE_DIR / "static" / "img"
LABEL_NAMES = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Alta"}
FEATURES = ["magnitud", "profundidad_km", "distancia_km"]


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(regular)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold)))
        return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with (BASE_DIR / "models" / "resultados.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    bundle = joblib.load(BASE_DIR / "models" / "modelos_por_k.joblib")
    dataset = pd.read_csv(BASE_DIR / "data" / "sismos.csv", encoding="utf-8-sig")
    face, bold_face = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName=bold_face, fontSize=28, leading=34, textColor=colors.HexColor("#0b2947"), alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName=face, fontSize=14, leading=22, textColor=colors.HexColor("#2276a8"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="SectionHeading", parent=styles["Heading1"], fontName=bold_face, fontSize=17, leading=21, textColor=colors.HexColor("#103856"), spaceBefore=8, spaceAfter=9))
    styles.add(ParagraphStyle(name="SubHeading", parent=styles["Heading2"], fontName=bold_face, fontSize=12, leading=16, textColor=colors.HexColor("#1b587d"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodySpanish", parent=styles["BodyText"], fontName=face, fontSize=9.5, leading=14, alignment=TA_LEFT, spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallSpanish", parent=styles["BodyText"], fontName=face, fontSize=8, leading=11, textColor=colors.HexColor("#3d5267"), spaceAfter=5))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName=bold_face, fontSize=10, leading=15, textColor=colors.HexColor("#805b14"), backColor=colors.HexColor("#fff4d8"), borderColor=colors.HexColor("#e8c66c"), borderWidth=.7, borderPadding=8, spaceBefore=7, spaceAfter=10))

    def para(text: str, style: str = "BodySpanish") -> Paragraph:
        return Paragraph(text, styles[style])

    story = [Spacer(1, 25 * mm), para("SENATI", "CoverSub"), Spacer(1, 13 * mm), para("INTELIGENCIA ARTIFICIAL", "CoverSub"), Spacer(1, 8 * mm), para("RETO INDIVIDUAL", "CoverSub"), Spacer(1, 19 * mm), para("SISMO AI PERÚ", "CoverTitle"), para("Sistema Inteligente de Análisis y Estimación Sísmica con KNN", "CoverSub"), Spacer(1, 20 * mm)]
    cover_rows = [
        ["Autora", "Saira Ruby Mego Burgos"],
        ["Método", "K-Nearest Neighbors (KNN)"],
        ["Datos", "MIXTO · IGP 1960–2026, M ≥ 5.0 (solo Perú) + intensidad estimada"],
        ["Fecha", date.today().strftime("%d/%m/%Y")],
    ]
    cover_table = Table(cover_rows, colWidths=[35 * mm, 95 * mm], hAlign="CENTER")
    cover_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), face), ("FONTNAME", (0, 0), (0, -1), bold_face),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#2276a8")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f6fa")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c8d6e2")),
        ("FONTSIZE", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [cover_table, Spacer(1, 18 * mm), para("Proyecto académico. Fuente de eventos: Instituto Geofísico del Perú (IGP), Catálogo de Sismos Instrumentales 1960–2026, solo M ≥ 5.0 (antes de 1990 los datos son menos precisos). Solo sismos de Perú y su zona marina cercana. La intensidad Mercalli es estimada, no observada.", "SmallSpanish"), PageBreak()]

    story += [para("1. Introducción", "SectionHeading"), para("En este reto construí una aplicación web que integra un clasificador KNN para estimar una categoría agrupada de intensidad Mercalli a partir de magnitud, profundidad y distancia epicentral. El trabajo recorre la generación y limpieza de datos, el entrenamiento con separación por evento, la evaluación y una interfaz para explorar escenarios hipotéticos."), para("El objetivo es practicar un flujo de aprendizaje automático. El sistema no predice la fecha, ubicación ni magnitud de futuros terremotos y no es una alerta oficial ni una herramienta de seguridad.", "Callout"), para("2. Dataset y procedencia", "SectionHeading"), para(f"Usé una sola fuente real y solo sismos de Perú y su zona marina cercana: el Catálogo de Sismos Instrumentales del Instituto Geofísico del Perú (IGP), del 13/01/1960 al 23/11/2026 (columnas fecha UTC, hora UTC, latitud, longitud, profundidad y magnitud). Con Pandas convertí fechas y tipos, eliminé nulos, duplicados y rangos inválidos, y conservé solo M ≥ 5.0: antes de 1990 los datos son menos precisos y el catálogo registra de forma incompleta los sismos pequeños, por eso un umbral de 5.0 mantiene un conjunto comparable en todo el periodo. Eliminé los epicentros de Ecuador, Chile, Brasil, Bolivia y Colombia con los polígonos de los departamentos del Perú (tierra) y una franja marina al oeste de la costa entre los límites marítimos con Ecuador y Chile (hasta 250 km). Quedaron {dataset['evento_id'].nunique()} eventos entre {dataset['fecha'].min()[:10]} y {dataset['fecha'].max()[:10]}. Fecha, epicentro, magnitud y profundidad son datos reales."), para(f"Cada evento se relacionó con sus 3 a 5 localidades más cercanas entre 46 ciudades reales de los 24 departamentos y el Callao, con distancia Haversine entre el epicentro y la localidad, para un total de {len(dataset)} registros evento-localidad. El catálogo no incluye la intensidad Mercalli, por lo que la estimé con una relación de atenuación didáctica (crece con la magnitud, disminuye con el logaritmo de la distancia hipocentral y con la profundidad) más ruido normal con semilla fija, y la agrupé en I–IV (Baja), V–VI (Moderada) y VII–XII (Alta). Por eso tipo_dato = MIXTO."), para("3. Preparación y modelo", "SectionHeading"), para("Usé magnitud, profundidad_km y distancia_km. El Pipeline aplica StandardScaler antes de KNeighborsClassifier. Separé el 20% de los datos mediante GroupShuffleSplit agrupado por evento_id y comprobé que no hubiera eventos compartidos entre entrenamiento y prueba. Elegí K comparando 3, 5, 7 y 9 con cinco pliegues GroupKFold únicamente sobre entrenamiento; el criterio principal fue el F1 macro promedio."), para(f"La partición final contiene {metrics['train_rows']} filas de entrenamiento y {metrics['test_rows']} de prueba, agrupadas en {metrics['train_events']} y {metrics['test_events']} eventos, respectivamente. Eventos compartidos: {metrics['shared_events']}. K seleccionado: {metrics['best_k']}.")]

    cv_rows = [["K", "F1 macro CV", "Desviación", "Accuracy CV"]]
    for score in metrics["cv_scores"]:
        cv_rows.append([str(score["k"]), f"{score['f1_macro_cv']:.4f}", f"{score['f1_std_cv']:.4f}", f"{score['accuracy_cv']:.4f}"])
    cv_table = Table(cv_rows, colWidths=[30 * mm, 40 * mm, 40 * mm, 40 * mm], repeatRows=1)
    cv_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), face), ("FONTNAME", (0, 0), (-1, 0), bold_face),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeaf3")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b9cad7")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [para("Comparación de K", "SubHeading"), cv_table, Spacer(1, 7 * mm), Image(str(IMG_DIR / "comparacion_k.png"), width=155 * mm, height=87 * mm), PageBreak()]

    story += [para("4. Evaluación", "SectionHeading"), para("Evalué el mejor pipeline en el conjunto de prueba reservado. Las cifras miden la concordancia con la intensidad estimada por la relación de atenuación, no con intensidades observadas en campo.")]
    metric_rows = [["Métrica", "Resultado"], ["Accuracy", f"{metrics['accuracy']:.4f}"], ["Precisión macro", f"{metrics['precision_macro']:.4f}"], ["Recall macro", f"{metrics['recall_macro']:.4f}"], ["F1 macro", f"{metrics['f1_macro']:.4f}"]]
    metric_table = Table(metric_rows, colWidths=[70 * mm, 45 * mm], hAlign="LEFT")
    metric_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), face), ("FONTNAME", (0, 0), (-1, 0), bold_face), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeaf3")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b9cad7")), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [metric_table, Spacer(1, 4 * mm), Image(str(IMG_DIR / "metricas.png"), width=150 * mm, height=84 * mm), para("Matriz de confusión", "SubHeading"), Image(str(IMG_DIR / "matriz_confusion.png"), width=116 * mm, height=96 * mm), PageBreak()]

    story += [para("5. Evidencias de la aplicación", "SectionHeading"), para("El mapa interactivo se construye con Folium y el dashboard resume eventos, distribución de magnitudes, regiones y métricas. Las siguientes figuras se generaron con Matplotlib a partir del dataset y los resultados de esta ejecución.")]
    story += [Image(str(IMG_DIR / "mapa_evidencia.png"), width=125 * mm, height=94 * mm), Spacer(1, 4 * mm), Image(str(IMG_DIR / "dashboard_evidencia.png"), width=160 * mm, height=96 * mm), PageBreak()]

    scenarios = [
        {"title": "Moderado, superficial y cercano a Huánuco", "magnitud": 6.0, "profundidad_km": 30.0, "distancia_km": 25.0},
        {"title": "Grande, profundo y lejano (Cusco)", "magnitud": 7.0, "profundidad_km": 150.0, "distancia_km": 350.0},
        {"title": "Moderado frente a la costa de Lima", "magnitud": 5.2, "profundidad_km": 40.0, "distancia_km": 60.0},
    ]
    story += [para("6. Escenarios hipotéticos", "SectionHeading"), para(f"Calculé estos tres casos en esta generación del informe con el modelo K={metrics['best_k']}. Son ejemplos didácticos, no eventos ni pronósticos.")]
    for scenario in scenarios:
        x = pd.DataFrame([[scenario["magnitud"], scenario["profundidad_km"], scenario["distancia_km"]]], columns=FEATURES)
        model = bundle["models"][int(metrics["best_k"])]
        predicted = str(model.predict(x)[0])
        probabilities = model.predict_proba(x)[0]
        probability_text = ", ".join(f"{LABEL_NAMES[str(label)]}: {probability:.0%}" for label, probability in zip(model.classes_, probabilities))
        story += [para(scenario["title"], "SubHeading"), para(f"Entrada: magnitud {scenario['magnitud']} Mw, profundidad {scenario['profundidad_km']} km y distancia {scenario['distancia_km']} km. Salida KNN: {LABEL_NAMES[predicted]} ({predicted}). Probabilidades: {probability_text}.")]

    story += [para("7. Limitaciones", "SectionHeading"), para("La principal limitación es que, aunque los eventos son reales, la intensidad es estimada y no observada. Además, solo hay 15 eventos con M ≥ 7.5 entre 1960 y 2026, por lo que el KNN extrapola mal en ese rango: para un M 7.8 lejano puede devolver Alta porque sus vecinos más parecidos son grandes sismos más cercanos. El modelo solo recibe tres variables y no conoce tipo de suelo, calidad de construcciones, población, topografía ni vulnerabilidad local. La intensidad describe efectos en un lugar y puede variar; no es intercambiable con la magnitud. La evaluación mide consistencia con la simulación creada, no desempeño ante terremotos reales."), para("No uso el sistema para anticipar terremotos ni para informar alertas. IGP, CENSIS y las autoridades competentes son las referencias para información oficial."), para("8. Preguntas de sustentación", "SectionHeading"), para("¿Por qué escalar? Para que profundidad y distancia no dominen la distancia euclidiana por sus rangos. ¿Cómo influye K? Un K bajo es sensible a vecinos puntuales; uno alto suaviza la decisión. ¿Por qué separar por evento? Para impedir que observaciones del mismo sismo se filtren entre entrenamiento y prueba. ¿Magnitud e intensidad? La primera describe la fuente y la segunda los efectos en un lugar. ¿Por qué no anticipa un terremoto? El clasificador estima clases a partir de entradas, no el momento futuro de ocurrencia, y la intensidad de sus ejemplos es estimada."), para("9. Conclusiones", "SectionHeading"), para("Con este reto dejé un flujo completo desde el catálogo real del IGP hasta una aplicación Flask conectada al pipeline KNN. La separación por evento me permitió evaluar con eventos reservados, y la interfaz muestra cómo cambian las probabilidades y los vecinos al modificar la entrada. Los resultados son útiles para explicar escalado, validación agrupada y clasificación; como la intensidad es estimada, limito las conclusiones a la práctica educativa."), para("Aviso final: proyecto educativo. No predice terremotos ni es una alerta oficial. Datos: eventos del IGP con intensidad estimada.", "Callout")]

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(face, 8)
        canvas.setFillColor(colors.HexColor("#718195"))
        canvas.drawString(18 * mm, 12 * mm, "SENATI · SISMO AI PERÚ · IGP 1960–2026 · Intensidad estimada · Uso educativo")
        canvas.drawRightString(192 * mm, 12 * mm, f"Página {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(OUT_FILE), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm, title="SISMO AI PERÚ · Informe", author="Saira Ruby Mego Burgos")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe PDF generado: {OUT_FILE}")


if __name__ == "__main__":
    main()
