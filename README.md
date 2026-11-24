# SISMO AI PERÚ

**Sistema Inteligente de Análisis y Estimación Sísmica con KNN** · Proyecto educativo (SENATI · Inteligencia Artificial).

Aplicación web Flask que usa un clasificador KNN para estimar una categoría de intensidad Mercalli (Baja I–IV, Moderada V–VI, Alta VII–XII) a partir de la magnitud, la profundidad y la distancia epicentral. Incluye un dashboard de monitoreo, un simulador, un mapa interactivo, el dataset, la evaluación del modelo, reportes y el informe en PDF.

## Aviso importante · datos MIXTOS

> Proyecto educativo. No predice terremotos ni es una alerta oficial. Datos: eventos del IGP con intensidad estimada.

**Fuente única: IGP. Fecha de los datos: 13/01/1960 – 22/11/2026 (catálogo hasta el 23/11/2026). Solo sismos de Perú y su zona marina cercana, M ≥ 5.0.**

- **Fuente:** Instituto Geofísico del Perú (IGP), *Catálogo de Sismos Instrumentales*. Archivo: [data/raw/igp_sismos_1960_2026.csv](data/raw/igp_sismos_1960_2026.csv) con las columnas `fecha UTC`, `hora UTC`, `latitud (º)`, `longitud (º)`, `profundidad (km)` y `magnitud (M)`.
- **Aclaración:** antes de 1990 los datos son menos precisos (menos estaciones y registro incompleto de sismos pequeños). Por eso se usa solo magnitud 5.0 o más en todo el periodo 1960–2026.
- **Reales:** fecha (UTC), latitud, longitud, magnitud y profundidad (`fuente = IGP`, `escala_magnitud = M`).
- **Calculados:** distancia Haversine a las 3–5 localidades más cercanas (46 ciudades reales de los 24 departamentos y el Callao).
- **Estimados:** la intensidad Mercalli **no** viene en el catálogo; se estima con una relación de atenuación didáctica más ruido con semilla fija. Por eso `tipo_dato = MIXTO`.
- Las conclusiones sirven solo como práctica educativa. Para información oficial consulta al IGP y al INDECI.

## Instalación y ejecución en Windows

Requisitos: Python 3.10 o superior desde python.org (marca *Add python.exe to PATH*). Si `python` no se reconoce, usa `py`.

1. Descomprime `SISMO_AI_PERU.zip` y abre **PowerShell** en la carpeta `SISMO_AI_PERU` (la que contiene `app.py`).
2. Crea el entorno virtual:
   ```powershell
   python -m venv .venv
   ```
3. Actívalo:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   Si PowerShell bloquea el script, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. En CMD usa `.venv\Scripts\activate.bat`. También puedes no activarlo y usar `.venv\Scripts\python.exe` en cada comando.
4. Instala las librerías:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Inicia la aplicación y abre `http://127.0.0.1:5000` (detén el servidor con `Ctrl+C`):
   ```powershell
   python app.py
   ```

El ZIP ya incluye el catálogo del IGP, el CSV procesado, el modelo entrenado, las gráficas y el informe PDF. Para regenerar todo desde cero:

```powershell
python generar_datos.py     # lee data/raw (IGP), filtra M ≥ 5.0 y Perú, y crea data/sismos.csv
python entrenar.py          # entrena KNN, compara K = 3, 5, 7, 9 y guarda models/ y static/img/
python generar_informe.py   # crea informe/informe_sismo_ai_peru.pdf
```

Si falta el archivo del IGP, guárdalo como `data/raw/igp_sismos_1960_2026.csv` (Catálogo de Sismos Instrumentales del IGP).

## Estructura

```text
SISMO_AI_PERU/
├── app.py                  # aplicación Flask
├── generar_datos.py        # IGP → limpieza → registros evento-localidad
├── entrenar.py             # Pipeline StandardScaler + KNN y evaluación
├── generar_informe.py      # informe PDF (ReportLab + Matplotlib)
├── requirements.txt
├── data/
│   ├── raw/                # igp_sismos_1960_2026.csv (catálogo del IGP)
│   ├── sismos.csv          # dataset procesado (MIXTO)
│   └── diccionario_datos.md
├── models/                 # modelo_knn.joblib, modelos_por_k.joblib, resultados.json
├── static/                 # css, js, geo (contornos de Perú) e img (gráficos del PDF)
├── templates/              # base.html, map_embed.html (Leaflet) y una plantilla por sección
└── informe/informe_sismo_ai_peru.pdf
```

## Datos y limpieza

`generar_datos.py`:

1. Lee el catálogo del IGP (25 880 sismos, 1960–2026), une `fecha UTC` + `hora UTC` (algunas horas traen centésimas) y convierte los tipos.
2. Elimina nulos, infinitos, duplicados y rangos inválidos; conserva solo **M ≥ 5.0** (5 200 eventos antes del filtro geográfico).
3. **Solo Perú:** se conserva un epicentro solo si cae dentro de un departamento del Perú (polígonos de [static/geo/peru_departamentos.geojson](static/geo/peru_departamentos.geojson)) o en el mar al oeste de la costa peruana, entre los límites marítimos con Ecuador (3°23' S) y Chile (18°21' S) y a 250 km o menos de la costa. Se eliminaron 551 epicentros de otros países (Ecuador, Chile, Brasil, Bolivia, Colombia).
4. Resultado: **4 646 eventos** y **18 519 registros** evento-localidad. Clases: I–IV 5 860 (31,6 %), V–VI 10 396 (56,1 %), VII–XII 2 263 (12,2 %).

Ver [data/diccionario_datos.md](data/diccionario_datos.md).

## Modelo y resultados

- Variables: `magnitud`, `profundidad_km`, `distancia_km`; Pipeline `StandardScaler` + `KNeighborsClassifier` (pesos por distancia).
- Separación 80/20 con `GroupShuffleSplit` agrupado por `evento_id`. Un `assert` garantiza **0 eventos compartidos** (3 716 eventos de entrenamiento y 930 de prueba).
- K = 3, 5, 7 y 9 comparados con `GroupKFold` (5 pliegues) solo sobre entrenamiento; criterio F1 macro.
- Resultado en prueba: **K = 9**, **accuracy = 0.772**, **precisión macro = 0.757**, **recall macro = 0.739**, **F1 macro = 0.747**. La matriz de confusión está en la app, el PDF y `models/resultados.json`.
- La app usa `predict`, `predict_proba` y `kneighbors` del modelo guardado; no hay reglas if/else que reemplacen al KNN. El departamento es solo contexto de la interfaz.

## Interfaz

Tema oscuro azul marino tipo centro de monitoreo: menú lateral con iconos (colapsable en celular), barra superior con sismógrafo SVG, KPI animados, simulador con barras de probabilidad, gráficos interactivos con Chart.js (CDN), matriz de confusión coloreada y tarjeta de rendimiento.

**Mapa (Leaflet):** encuadre ajustado a Perú con límites de navegación, mapa base Esri World Dark Gray (sin clave ni marca de agua), contorno de Perú en celeste, límites departamentales tenues y el resto del mapa atenuado. Los contornos se guardan en `static/geo/`, así que funcionan sin internet; el mapa base y las librerías CDN sí necesitan conexión. Puntos semitransparentes por magnitud (amarillo 5.0–5.4, naranja 5.5–5.9, rojo ≥ 6.0, los grandes encima), leyenda «Magnitud (Mw)», «OCÉANO PACÍFICO», escala, Lima y Huánuco destacado, tarjeta al pasar el mouse o hacer clic (zona, magnitud, profundidad, fecha y fuente) y filtros por categoría, rango de magnitud y rango de años.

## Limitaciones

- La intensidad es estimada, no observada: las métricas miden la concordancia con esa estimación.
- Solo tres variables: no se incluyen suelo, construcciones, población ni vulnerabilidad.
- Antes de 1990 los epicentros y magnitudes son menos precisos.
- Hay pocos sismos muy grandes (15 con M ≥ 7.5 entre 1960 y 2026): el KNN extrapola mal en ese rango.
- Las coordenadas de las localidades son aproximadas (centro de la ciudad).
- No anticipa terremotos ni reemplaza al IGP, al INDECI ni a las autoridades.

## Sustentación

- **¿Por qué escalar?** Para que profundidad y distancia, con rangos grandes, no dominen la distancia euclidiana.
- **¿Cómo influye K?** Un K pequeño es sensible a vecinos puntuales; uno grande suaviza la decisión.
- **¿Por qué separar por evento?** Las localidades de un mismo sismo comparten magnitud y profundidad; mezclarlas filtraría información a la prueba.
- **¿Magnitud o intensidad?** La magnitud mide la energía en la fuente; la intensidad, los efectos en un lugar.
- **¿Por qué no predice terremotos?** Clasifica escenarios dados, no estima cuándo ni dónde ocurrirá un sismo.
