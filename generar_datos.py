"""Construye data/sismos.csv SOLO con sismos de Perú y su zona marina cercana.

Fuente única (datos reales de fecha, epicentro, magnitud y profundidad):
  * IGP · Catálogo de Sismos Instrumentales (igp_sismos_1960_2026.csv), 13/01/1960 – 23/11/2026.
Se usan solo eventos con M ≥ 5.0: antes de 1990 el catálogo es menos preciso y completo para sismos
pequeños, así que el umbral 5.0 mantiene un conjunto homogéneo en todo el periodo.
La intensidad Mercalli NO viene en el catálogo: se ESTIMA con una relación de atenuación didáctica
más ruido con semilla fija. Por eso tipo_dato = MIXTO.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolyPath

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IGP_FILE = RAW_DIR / "igp_sismos_1960_2026.csv"
DEPARTAMENTOS_GEOJSON = BASE_DIR / "static" / "geo" / "peru_departamentos.geojson"
CONTORNO_GEOJSON = BASE_DIR / "static" / "geo" / "peru_contorno.geojson"
SEED = 20261123
MAGNITUD_MINIMA = 5.0
# Límites marítimos aproximados: con Ecuador (paralelo 3°23' S) y con Chile (paralelo 18°21' S).
LAT_MAR_NORTE, LAT_MAR_SUR = -3.39, -18.35
MAR_MAX_KM = 250.0

LOCALIDADES = [
    ("Chachapoyas", "Amazonas", -6.2317, -77.8690), ("Bagua Grande", "Amazonas", -5.7553, -78.4425),
    ("Huaraz", "Áncash", -9.5278, -77.5278), ("Chimbote", "Áncash", -9.0745, -78.5936),
    ("Abancay", "Apurímac", -13.6339, -72.8814), ("Andahuaylas", "Apurímac", -13.6556, -73.3872),
    ("Arequipa", "Arequipa", -16.4090, -71.5375), ("Camaná", "Arequipa", -16.6228, -72.7111),
    ("Ayacucho", "Ayacucho", -13.1631, -74.2236), ("Puquio", "Ayacucho", -14.6939, -74.1244),
    ("Cajamarca", "Cajamarca", -7.1617, -78.5128), ("Jaén", "Cajamarca", -5.7083, -78.8083),
    ("Callao", "Callao", -12.0566, -77.1181),
    ("Cusco", "Cusco", -13.5319, -71.9675), ("Sicuani", "Cusco", -14.2694, -71.2261),
    ("Huancavelica", "Huancavelica", -12.7866, -74.9769),
    ("Huánuco", "Huánuco", -9.9306, -76.2422), ("Tingo María", "Huánuco", -9.2953, -75.9978),
    ("Ica", "Ica", -14.0678, -75.7286), ("Nazca", "Ica", -14.8350, -74.9328), ("Pisco", "Ica", -13.7100, -76.2032),
    ("Huancayo", "Junín", -12.0651, -75.2049), ("La Merced", "Junín", -11.0547, -75.3289),
    ("Trujillo", "La Libertad", -8.1116, -79.0288), ("Huamachuco", "La Libertad", -7.8122, -78.0486),
    ("Chiclayo", "Lambayeque", -6.7714, -79.8409),
    ("Lima", "Lima", -12.0464, -77.0428), ("Huacho", "Lima", -11.1067, -77.6050), ("Cañete", "Lima", -13.0750, -76.3840),
    ("Iquitos", "Loreto", -3.7437, -73.2516), ("Yurimaguas", "Loreto", -5.9000, -76.0833),
    ("Puerto Maldonado", "Madre de Dios", -12.5933, -69.1891),
    ("Moquegua", "Moquegua", -17.1950, -70.9350), ("Ilo", "Moquegua", -17.6394, -71.3375),
    ("Cerro de Pasco", "Pasco", -10.6675, -76.2567), ("Oxapampa", "Pasco", -10.5800, -75.4050),
    ("Piura", "Piura", -5.1945, -80.6328), ("Talara", "Piura", -4.5772, -81.2719),
    ("Puno", "Puno", -15.8402, -70.0219), ("Juliaca", "Puno", -15.5000, -70.1333),
    ("Moyobamba", "San Martín", -6.0342, -76.9723), ("Tarapoto", "San Martín", -6.4850, -76.3650),
    ("Tacna", "Tacna", -18.0066, -70.2463), ("Tumbes", "Tumbes", -3.5669, -80.4515),
    ("Pucallpa", "Ucayali", -8.3791, -74.5539), ("Atalaya", "Ucayali", -10.7300, -73.7550),
]
NOMBRES_DEP = {"ANCASH": "Áncash", "APURIMAC": "Apurímac", "HUANUCO": "Huánuco", "JUNIN": "Junín",
               "SAN MARTIN": "San Martín", "LA LIBERTAD": "La Libertad", "MADRE DE DIOS": "Madre de Dios"}


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------- Geografía de Perú
def cargar_departamentos() -> list[tuple[str, list[np.ndarray]]]:
    geo = json.loads(DEPARTAMENTOS_GEOJSON.read_text(encoding="utf-8"))
    departamentos = []
    for feature in geo["features"]:
        nombre = feature["properties"]["NOMBDEP"].strip().upper()
        nombre = NOMBRES_DEP.get(nombre, nombre.title())
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        departamentos.append((nombre, [np.asarray(poly[0], dtype=float) for poly in polygons]))
    return departamentos


def contorno_peru(departamentos) -> list[list[list[float]]]:
    """Bordes que pertenecen a un solo departamento = límite exterior de Perú."""
    conteo: dict[tuple, int] = {}
    for _, rings in departamentos:
        for ring in rings:
            pts = [tuple(np.round(p, 5)) for p in ring]
            for a, b in zip(pts, pts[1:]):
                key = (a, b) if a <= b else (b, a)
                conteo[key] = conteo.get(key, 0) + 1
    return [[list(a), list(b)] for (a, b), n in conteo.items() if n == 1]


def encadenar_anillos(segmentos: list) -> list[list[list[float]]]:
    """Une los bordes exteriores en anillos cerrados (contorno de Perú) para dibujar y enmascarar."""
    vecinos: dict[tuple, list[tuple]] = {}
    for a, b in segmentos:
        a, b = tuple(a), tuple(b)
        vecinos.setdefault(a, []).append(b)
        vecinos.setdefault(b, []).append(a)
    usados: set = set()
    anillos = []
    for a, b in segmentos:
        a, b = tuple(a), tuple(b)
        if frozenset((a, b)) in usados:
            continue
        usados.add(frozenset((a, b)))
        anillo, actual = [a, b], b
        while actual != anillo[0]:
            siguiente = next((n for n in vecinos[actual] if frozenset((actual, n)) not in usados), None)
            if siguiente is None:
                break
            usados.add(frozenset((actual, siguiente)))
            anillo.append(siguiente)
            actual = siguiente
        if len(anillo) >= 4 and anillo[0] == anillo[-1]:
            anillos.append([list(pt) for pt in anillo])
    return sorted(anillos, key=len, reverse=True)


def clasificar_ubicacion(lat: np.ndarray, lon: np.ndarray, departamentos, borde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (en_peru_o_mar, zona). Tierra: departamento por polígono. Mar: oeste de la costa peruana."""
    puntos = np.column_stack([lon, lat])
    zona = np.full(len(lat), "", dtype=object)
    for nombre, rings in departamentos:
        dentro = np.zeros(len(lat), dtype=bool)
        for ring in rings:
            dentro |= PolyPath(ring).contains_points(puntos)
        zona[dentro & (zona == "")] = nombre
    en_tierra = zona != ""
    # Mar: ninguna tierra peruana al oeste, tierra peruana al este, entre los límites marítimos y cerca de la costa.
    candidatos = (~en_tierra) & (lat <= LAT_MAR_NORTE) & (lat >= LAT_MAR_SUR)
    en_mar = np.zeros(len(lat), dtype=bool)
    for i in np.where(candidatos)[0]:
        cerca_lat = np.abs(borde[:, 1] - lat[i]) < 0.15
        if not cerca_lat.any():
            continue
        lon_borde = borde[cerca_lat, 0]
        if (lon_borde < lon[i]).any():  # hay Perú al oeste → no es el Pacífico peruano
            continue
        if haversine_km(lat[i], lon[i], borde[:, 1], borde[:, 0]).min() <= MAR_MAX_KM:
            en_mar[i] = True
            zona[i] = "Mar peruano"
    return en_tierra | en_mar, zona


# ---------------------------------------------------------------- Catálogos
def cargar_igp() -> pd.DataFrame:
    raw = pd.read_csv(IGP_FILE, dtype=str, encoding="utf-8-sig")
    raw.columns = ["fecha", "hora", "latitud", "longitud", "profundidad", "magnitud"]
    hora = raw["hora"].str.strip().str.split(".").str[0]  # algunas horas traen centésimas
    fecha = pd.to_datetime(raw["fecha"].str.strip() + " " + hora, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = pd.DataFrame({
        "evento_id": "IGP-" + fecha.dt.strftime("%Y%m%d%H%M%S").fillna("NA") + "-" + raw.index.astype(str),
        "fecha": fecha,
        "latitud": pd.to_numeric(raw["latitud"], errors="coerce"),
        "longitud": pd.to_numeric(raw["longitud"], errors="coerce"),
        "profundidad_km": pd.to_numeric(raw["profundidad"], errors="coerce"),
        "magnitud": pd.to_numeric(raw["magnitud"], errors="coerce"),
    })
    df["escala_magnitud"] = "M"  # el catálogo instrumental publica la magnitud como "magnitud (M)"
    df["fuente"] = "IGP"
    print(f"IGP: {len(raw)} filas en el catálogo ({fecha.min():%d/%m/%Y} – {fecha.max():%d/%m/%Y}); "
          f"{int((df['magnitud'] >= MAGNITUD_MINIMA).sum())} con M ≥ {MAGNITUD_MINIMA}.")
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fecha", "latitud", "longitud", "profundidad_km", "magnitud"])
    df = df.drop_duplicates(subset=["evento_id"]).drop_duplicates(subset=["fecha", "latitud", "longitud", "magnitud"])
    return df[df["profundidad_km"].between(0, 700) & df["magnitud"].between(MAGNITUD_MINIMA, 10)
              & df["latitud"].between(-25, 5) & df["longitud"].between(-90, -60)]


def intensidad_estimada(magnitud: float, profundidad: float, distancia: float, ruido: float) -> tuple[int, str]:
    """Relación de atenuación didáctica (no operacional): sube con M, baja con log(distancia) y con la profundidad."""
    distancia_hipocentral = np.sqrt(distancia ** 2 + profundidad ** 2)
    valor = 9.0 + 1.4 * magnitud - 5.0 * np.log10(distancia_hipocentral + 10.0) - 0.002 * profundidad + ruido
    mercalli = int(np.clip(np.rint(valor), 1, 12))
    if mercalli <= 4:
        return mercalli, "I–IV"
    if mercalli <= 6:
        return mercalli, "V–VI"
    return mercalli, "VII–XII"


def generar_dataframe(seed: int = SEED) -> pd.DataFrame:
    departamentos = cargar_departamentos()
    anillos = encadenar_anillos(contorno_peru(departamentos))
    CONTORNO_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"nombre": "Perú"},
                                "geometry": {"type": "MultiPolygon", "coordinates": [[ring] for ring in anillos]}}]}), encoding="utf-8")
    borde = np.unique(np.asarray([p for ring in anillos for p in ring]), axis=0)

    eventos = limpiar(cargar_igp()).reset_index(drop=True)
    dentro, zona = clasificar_ubicacion(eventos["latitud"].to_numpy(), eventos["longitud"].to_numpy(), departamentos, borde)
    eventos["zona_epicentro"] = zona
    print(f"Eventos M ≥ {MAGNITUD_MINIMA}: {len(eventos)}; fuera de Perú y su mar cercano (eliminados): {int((~dentro).sum())}.")
    eventos = eventos[dentro].sort_values("fecha").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    loc = pd.DataFrame(LOCALIDADES, columns=["localidad", "departamento", "lat", "lon"])
    filas = []
    for ev in eventos.itertuples(index=False):
        distancias = haversine_km(ev.latitud, ev.longitud, loc["lat"].to_numpy(), loc["lon"].to_numpy())
        for i in np.argsort(distancias)[: int(rng.integers(3, 6))]:
            mercalli, clase = intensidad_estimada(ev.magnitud, ev.profundidad_km, float(distancias[i]), float(rng.normal(0, 0.5)))
            filas.append({
                "evento_id": ev.evento_id, "fecha": ev.fecha.strftime("%Y-%m-%d %H:%M:%S"),
                "latitud": round(float(ev.latitud), 4), "longitud": round(float(ev.longitud), 4),
                "zona_epicentro": ev.zona_epicentro, "localidad": loc.at[i, "localidad"], "departamento": loc.at[i, "departamento"],
                "magnitud": round(float(ev.magnitud), 2), "profundidad_km": round(float(ev.profundidad_km), 1),
                "distancia_km": round(float(distancias[i]), 2), "mercalli_estimada": mercalli, "intensidad": clase,
                "escala_magnitud": ev.escala_magnitud, "fuente": ev.fuente, "tipo_dato": "MIXTO",
            })
    return pd.DataFrame(filas).drop_duplicates(subset=["evento_id", "localidad"]).reset_index(drop=True)


def main() -> None:
    data = generar_dataframe()
    salida = DATA_DIR / "sismos.csv"
    data.to_csv(salida, index=False, encoding="utf-8-sig")
    ev = data.drop_duplicates("evento_id")
    print(f"CSV generado: {salida}")
    print(f"Registros: {len(data)} | eventos: {len(ev)} | {data['fecha'].min()} → {data['fecha'].max()}")
    print(data["intensidad"].value_counts().reindex(["I–IV", "V–VI", "VII–XII"], fill_value=0).to_string())


if __name__ == "__main__":
    main()
