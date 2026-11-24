# Diccionario de datos · sismos.csv

**Fuente de los eventos (única): IGP · Catálogo de Sismos Instrumentales, 13/01/1960 – 23/11/2026.** Archivo `raw/igp_sismos_1960_2026.csv` (`fecha UTC, hora UTC, latitud (º), longitud (º), profundidad (km), magnitud (M)`). Se usan solo sismos de Perú y su zona marina cercana con M ≥ 5.0; antes de 1990 los datos son menos precisos, por eso el umbral.

Cada fila es un par evento-localidad. `tipo_dato = MIXTO`: los campos del evento son reales; la distancia es calculada y la intensidad es **estimada**.

| Campo | Tipo | Origen | Descripción |
|---|---|---|---|
| `evento_id` | texto | Calculado | `IGP-<fechahora>-<fila>` (el catálogo no trae ID); agrupa las localidades del mismo sismo. |
| `fecha` | fecha y hora | IGP | Fecha y hora UTC (`fecha UTC` + `hora UTC`). |
| `zona_epicentro` | texto | Calculado | Departamento donde cae el epicentro o «Mar peruano». |
| `latitud`, `longitud` | decimal | IGP | Epicentro en grados. |
| `localidad`, `departamento` | texto | Calculado | Una de las 3–5 localidades más cercanas al epicentro (46 ciudades, 24 departamentos y Callao). |
| `magnitud` | decimal | IGP | Magnitud (M) tal como la publica el IGP; siempre ≥ 5.0. |
| `profundidad_km` | decimal | IGP | Profundidad del foco en km. |
| `distancia_km` | decimal | Calculado | Distancia Haversine entre el epicentro y la localidad. |
| `mercalli_estimada` | entero 1–12 | Estimado | Intensidad Mercalli de la relación de atenuación didáctica + ruido (semilla 20261123). |
| `intensidad` | categoría | Estimado | `I–IV` (Baja), `V–VI` (Moderada), `VII–XII` (Alta). |
| `escala_magnitud` | texto | IGP | `M`. |
| `tipo_dato` | texto | — | Siempre `MIXTO`. |
| `fuente` | texto | — | Siempre `IGP`. |

## Limpieza y selección

1. Conversión de tipos y fechas con `errors="coerce"`; infinitos → nulos; se eliminan nulos y duplicados.
2. Magnitud ≥ 5.0 (5 200 eventos); profundidad 0–700 km; todo el periodo 1960–2026.
3. Solo Perú: polígonos departamentales del Perú o franja marina al oeste de la costa (entre 3°23' S y 18°21' S, hasta 250 km). Se eliminaron 551 epicentros de otros países.
4. Resultado: 4 646 eventos y 18 519 registros. Clases: I–IV 5 860, V–VI 10 396, VII–XII 2 263.

## Relación de atenuación (didáctica, no operacional)

`I = 9.0 + 1.4·M − 5.0·log10(R + 10) − 0.002·h + ε`, con `R = √(distancia² + h²)` y `ε ~ N(0, 0.5)`, redondeada y limitada a 1–12.
