"""SISMO AI PERÚ · aplicación Flask.

Solo sismos de Perú y su zona marina cercana: IGP 1960–2026 (Catálogo de Sismos
Instrumentales), solo M ≥ 5.0. La intensidad Mercalli es ESTIMADA.
Toda clasificación sale del Pipeline KNN guardado (predict, predict_proba y
kneighbors); no hay reglas if/else que lo sustituyan.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sismos.csv"
FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
LABELS = ["I–IV", "V–VI", "VII–XII"]
LABEL_NAMES = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Alta"}
LABEL_COLORS = {"I–IV": "#2fbf8f", "V–VI": "#f2a33a", "VII–XII": "#ef4d5f"}
RANGES = {"magnitud": (3.0, 9.5), "profundidad_km": (0.0, 700.0), "distancia_km": (0.0, 1500.0)}
FIELD_NAMES = {"magnitud": "La magnitud", "profundidad_km": "La profundidad", "distancia_km": "La distancia al epicentro"}
MAG_CATEGORIES = {"M50": (5.0, 5.5), "M55": (5.5, 6.0), "M60": (6.0, 10.0)}  # colores del mapa por magnitud
AVISO = "Proyecto educativo. No predice terremotos ni es una alerta oficial. Datos: eventos del IGP con intensidad estimada."
FUENTE = ("IGP · Catálogo de Sismos Instrumentales (1960–2026), M ≥ 5.0 · "
          "solo Perú y su zona marina cercana")

app = Flask(__name__)
data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
data["_buscar"] = data.astype(str).agg(" ".join, axis=1).str.lower()
bundle = joblib.load(BASE_DIR / "models" / "modelos_por_k.joblib")
metrics = json.loads((BASE_DIR / "models" / "resultados.json").read_text(encoding="utf-8"))
models = bundle["models"]
models[int(metrics["best_k"])] = joblib.load(BASE_DIR / "models" / "modelo_knn.joblib")  # modelo final guardado
x_train = bundle["x_train"].reset_index(drop=True)
y_train = bundle["y_train"].reset_index(drop=True)
meta_train = bundle["meta_train"].reset_index(drop=True)
# El primer registro de cada evento corresponde a la localidad más cercana al epicentro.
events = data.drop_duplicates("evento_id").reset_index(drop=True)
DEPARTMENTS = sorted(data["departamento"].unique())
YEARS = (int(events["fecha"].str[:4].min()), int(events["fecha"].str[:4].max()))
PERIOD = f"{pd.to_datetime(events['fecha']).min():%d/%m/%Y} – {pd.to_datetime(events['fecha']).max():%d/%m/%Y}"


def _events_payload() -> str:
    ordered = events.sort_values("magnitud")  # los de mayor magnitud se dibujan encima
    zona = np.where(ordered["zona_epicentro"] == "Mar peruano", "Mar peruano · frente a " + ordered["departamento"], ordered["zona_epicentro"])
    rows = [[float(r.latitud), float(r.longitud), float(r.magnitud), float(r.profundidad_km), int(r.fecha[:4]), r.fecha[:16], z, r.fuente, r.escala_magnitud]
            for r, z in zip(ordered.itertuples(index=False), zona)]
    return json.dumps({"campos": ["lat", "lon", "mag", "prof", "anio", "fecha", "zona", "fuente", "escala"], "eventos": rows}, ensure_ascii=False, separators=(",", ":"))


EVENTS_JSON = _events_payload()

NAV = [
    ("dashboard", "Dashboard", "grid"), ("knn_analysis", "Análisis KNN", "network"),
    ("simulator", "Simulador de sismo", "wave"), ("event_map", "Mapa de eventos", "map"),
    ("dataset_page", "Dataset", "table"), ("evaluation", "Evaluación del modelo", "check"),
    ("reports", "Reportes", "file"), ("help_page", "Ayuda", "help"),
]


@app.context_processor
def common_context() -> dict:
    return {"nav": NAV, "aviso": AVISO, "fuente": FUENTE, "labels": LABELS, "label_names": LABEL_NAMES,
            "label_colors": LABEL_COLORS, "metrics": metrics, "departments": DEPARTMENTS, "ranges": RANGES,
            "period": PERIOD, "years": YEARS}


def estimate(magnitude: float, depth: float, distance: float, k: int) -> dict:
    features = pd.DataFrame([[magnitude, depth, distance]], columns=FEATURES)
    estimator = models[k]
    predicted = str(estimator.predict(features)[0])
    probabilities = {str(label): float(p) for label, p in zip(estimator.classes_, estimator.predict_proba(features)[0])}
    # El Pipeline no expone kneighbors: se escala la entrada y se consulta el paso KNN entrenado.
    scaled = estimator.named_steps["escalador"].transform(features)
    distances, indices = estimator.named_steps["knn"].kneighbors(scaled, n_neighbors=k)
    votes = {label: 0 for label in LABELS}
    neighbors = []
    for index, model_distance in zip(indices[0], distances[0]):
        label = str(y_train.iloc[int(index)])
        votes[label] += 1
        row, meta = x_train.iloc[int(index)], meta_train.iloc[int(index)]
        neighbors.append({"evento_id": str(meta["evento_id"]), "fecha": str(meta["fecha"])[:10],
                          "localidad": f"{meta['localidad']} ({meta['departamento']})",
                          "magnitud": float(row["magnitud"]), "profundidad_km": float(row["profundidad_km"]),
                          "distancia_km": float(row["distancia_km"]), "intensidad": label,
                          "intensidad_nombre": LABEL_NAMES[label], "distancia_punto": round(float(model_distance), 4)})
    training_range = {name: (float(x_train[name].min()), float(x_train[name].max())) for name in FEATURES}
    values = {"magnitud": magnitude, "profundidad_km": depth, "distancia_km": distance}
    outside = [FIELD_NAMES[n].split(" ", 1)[1] for n, (lo, hi) in training_range.items() if not lo <= values[n] <= hi]
    return {"predicted": predicted, "label_name": LABEL_NAMES[predicted], "color": LABEL_COLORS[predicted],
            "probabilities": probabilities, "votes": votes, "neighbors": neighbors, "k": k,
            "fuera_de_rango": outside, "rango_entrenamiento": training_range}


def validate_payload(payload) -> tuple[dict | None, str | None]:
    values: dict = {}
    for name, (minimum, maximum) in RANGES.items():
        raw = payload.get(name, "")
        label = FIELD_NAMES[name]
        if raw is None or str(raw).strip() == "":
            return None, f"{label} es obligatoria."
        try:
            number = float(str(raw).replace(",", "."))
        except ValueError:
            return None, f"{label} debe ser un número (por ejemplo 5.8)."
        if not math.isfinite(number):
            return None, f"{label} debe ser un número finito."
        if not minimum <= number <= maximum:
            return None, f"{label} debe estar entre {minimum:g} y {maximum:g}."
        values[name] = number
    try:
        values["k"] = int(payload.get("k", metrics["best_k"]))
    except (TypeError, ValueError):
        return None, "K debe ser 3, 5, 7 o 9."
    if values["k"] not in models:
        return None, "K debe ser 3, 5, 7 o 9."
    department = str(payload.get("departamento", "Huánuco")).strip() or "Huánuco"
    if department not in DEPARTMENTS:
        return None, "Selecciona un departamento de la lista."
    values["departamento"] = department
    return values, None


def map_filters(args) -> dict:
    """Filtros del mapa validados: categoría de magnitud, rango de magnitud y rango de años."""
    def number(name, default, cast=float):
        try:
            value = cast(args.get(name, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    filters = {"cat": args.get("cat", "Todas"), "min": number("min", 5.0), "max": number("max", 9.5),
               "desde": number("desde", YEARS[0], int), "hasta": number("hasta", YEARS[1], int)}
    if filters["cat"] not in MAG_CATEGORIES:
        filters["cat"] = "Todas"
    if filters["min"] > filters["max"]:
        filters["min"], filters["max"] = filters["max"], filters["min"]
    if filters["desde"] > filters["hasta"]:
        filters["desde"], filters["hasta"] = filters["hasta"], filters["desde"]
    return filters


SCENARIOS = [
    {"title": "Moderado, superficial y cercano a Huánuco", "departamento": "Huánuco", "magnitud": 6.0, "profundidad_km": 30, "distancia_km": 25,
     "interpretation": "Epicentro hipotético a 25 km de Huánuco; poca distancia y poca profundidad suelen elevar la intensidad estimada."},
    {"title": "Grande, profundo y lejano", "departamento": "Cusco", "magnitud": 7.0, "profundidad_km": 150, "distancia_km": 350,
     "interpretation": "Aunque la magnitud es alta, la profundidad y la distancia atenúan la intensidad estimada."},
    {"title": "Moderado frente a la costa de Lima", "departamento": "Lima", "magnitud": 5.2, "profundidad_km": 40, "distancia_km": 60,
     "interpretation": "Evento frecuente en la costa central; el modelo compara con eventos de características similares."},
]


def live_scenarios() -> list[dict]:
    return [dict(s, result=estimate(s["magnitud"], s["profundidad_km"], s["distancia_km"], int(metrics["best_k"]))) for s in SCENARIOS]


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    bins = pd.cut(events["magnitud"], bins=[0, 5.5, 6.0, 7.0, 10], right=False, labels=["5.0 – 5.4", "5.5 – 5.9", "6.0 – 6.9", "≥ 7.0"])
    magnitude_bins = bins.value_counts().reindex(bins.cat.categories, fill_value=0)
    top = events["departamento"].value_counts().head(5)
    charts = {"magnitudes": {"labels": list(magnitude_bins.index), "values": [int(v) for v in magnitude_bins.values]},
              "departamentos": {"labels": list(top.index), "values": [int(v) for v in top.values]}}
    sources = events["fuente"].value_counts().to_dict()
    kpis = {"eventos": len(events), "magnitud": round(float(events["magnitud"].mean()), 2),
            "profundidad": round(float(events["profundidad_km"].mean()), 1), "regiones": int(data["departamento"].nunique())}
    return render_template("dashboard.html", active="dashboard", title="Dashboard", kpis=kpis, charts=charts, records=len(data), sources=sources)


@app.route("/api/eventos")
def api_events():
    return Response(EVENTS_JSON, mimetype="application/json", headers={"Cache-Control": "public, max-age=600"})


@app.route("/mapa/embed")
def map_embed():
    return render_template("map_embed.html", filters=map_filters(request.args), compact=request.args.get("compacto") == "1")


@app.route("/knn")
def knn_analysis():
    return render_template("knn.html", active="knn_analysis", title="Análisis KNN",
                           features=FEATURES, train_stats=x_train.describe().round(2).to_dict())


@app.route("/simulador", methods=["GET", "POST"])
def simulator():
    result, error = None, None
    form = {"magnitud": "6.0", "profundidad_km": "30", "distancia_km": "25", "k": str(metrics["best_k"]), "departamento": "Huánuco"}
    if request.method == "POST":
        form.update(request.form.to_dict())
        validated, error = validate_payload(request.form)
        if validated:
            result = estimate(validated["magnitud"], validated["profundidad_km"], validated["distancia_km"], validated["k"])
    return render_template("simulator.html", active="simulator", title="Simulador de sismo", result=result, error=error, form=form), (400 if error else 200)


@app.route("/mapa")
def event_map():
    return render_template("map.html", active="event_map", title="Mapa de eventos", filters=map_filters(request.args), categories=list(MAG_CATEGORIES))


@app.route("/dataset")
def dataset_page():
    query = request.args.get("q", "").strip()
    try:
        page_number = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page_number = 1
    view = data[data["_buscar"].str.contains(query.lower(), regex=False)] if query else data
    page_size = 25
    pages = max(1, math.ceil(len(view) / page_size))
    page_number = min(page_number, pages)
    subset = view.iloc[(page_number - 1) * page_size: page_number * page_size]
    columns = ["evento_id", "fecha", "fuente", "latitud", "longitud", "zona_epicentro", "localidad", "departamento", "magnitud",
               "escala_magnitud", "profundidad_km", "distancia_km", "mercalli_estimada", "intensidad", "tipo_dato"]
    return render_template("dataset.html", active="dataset_page", title="Dataset", rows=subset[columns].to_dict("records"), columns=columns,
                           query=query, page_number=page_number, pages=pages, matches=len(view), total_rows=len(data),
                           total_events=len(events), sources=events["fuente"].value_counts().to_dict(),
                           summary=data[FEATURES].describe().round(2).to_dict())


@app.route("/descargar/csv")
def download_csv():
    return send_file(DATA_FILE, as_attachment=True, download_name="sismos.csv", mimetype="text/csv")


@app.route("/evaluacion")
def evaluation():
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    errors_by_class = matrix.sum(axis=1) - np.diag(matrix)
    return render_template("evaluation.html", active="evaluation", title="Evaluación del modelo", matrix=matrix.tolist(),
                           confused_class=LABELS[int(np.argmax(errors_by_class))], class_report=metrics["class_report"])


@app.route("/reportes")
def reports():
    return render_template("reports.html", active="reports", title="Reportes", scenarios=live_scenarios())


@app.route("/ayuda")
def help_page():
    return render_template("help.html", active="help_page", title="Ayuda")


@app.route("/informe/pdf")
def download_report():
    report = BASE_DIR / "informe" / "informe_sismo_ai_peru.pdf"
    return send_file(report, as_attachment=True, download_name=report.name, mimetype="application/pdf")


@app.route("/api/estimar", methods=["POST"])
def api_estimate():
    payload = request.get_json(silent=True) or request.form
    validated, error = validate_payload(payload)
    if error:
        return jsonify({"error": error}), 400
    answer = estimate(validated["magnitud"], validated["profundidad_km"], validated["distancia_km"], validated["k"])
    answer["departamento"] = validated["departamento"]
    answer["aviso"] = AVISO
    return jsonify(answer)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
