from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sismos.csv"
MODEL_DIR = BASE_DIR / "models"
IMG_DIR = BASE_DIR / "static" / "img"
FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
LABELS = ["I–IV", "V–VI", "VII–XII"]
K_VALUES = [3, 5, 7, 9]
PALETTE = {"I–IV": "#39c6a5", "V–VI": "#f2b84b", "VII–XII": "#ff637d"}


def crear_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("escalador", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
    ])


def guardar_graficos(scores: list[dict[str, float]], matrix: np.ndarray, report: dict) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0b1424")
    ax.set_facecolor("#111e32")
    ks = [int(item["k"]) for item in scores]
    means = [float(item["f1_macro_cv"]) for item in scores]
    errors = [float(item["f1_std_cv"]) for item in scores]
    ax.bar([str(k) for k in ks], means, yerr=errors, color="#49bfff", capsize=5)
    ax.set(title="Validación cruzada agrupada · F1 macro", xlabel="Número de vecinos (K)", ylabel="F1 macro")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "comparacion_k.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0b1424")
    ax.set_facecolor("#111e32")
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=15)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set(xlabel="Predicción", ylabel="Clase estimada", title="Matriz de confusión · prueba")
    threshold = max(matrix.max() / 2, 1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else "#182238", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "matriz_confusion.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    values = [float(report[name]) for name in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0b1424")
    ax.set_facecolor("#111e32")
    bars = ax.bar(["Accuracy", "Precisión", "Recall", "F1 macro"], values, color=["#49bfff", "#39c6a5", "#f2b84b", "#ff637d"])
    ax.set_ylim(0, 1)
    ax.set_title("Rendimiento en el conjunto de prueba")
    ax.grid(axis="y", alpha=.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .025, f"{value:.3f}", ha="center", color="white")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "metricas.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def crear_evidencias(data: pd.DataFrame, report: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0b1424")
    ax.set_facecolor("#111e32")
    events = data.drop_duplicates("evento_id")
    colors = events["intensidad"].map(PALETTE)
    ax.scatter(events["longitud"], events["latitud"], c=colors, s=(events["magnitud"] - 4.6) * 8, alpha=.68, edgecolors="white", linewidths=.25)
    ax.set(xlim=(-83, -67), ylim=(-19, 1), xlabel="Longitud", ylabel="Latitud", title="Perú · epicentros IGP 1960–2026, M ≥ 5.0 (color = intensidad estimada)")
    ax.grid(alpha=.12)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "mapa_evidencia.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    departments = data.drop_duplicates("evento_id")["departamento"].value_counts().head(5).sort_values()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor="#0b1424")
    for axis in axes.flat:
        axis.set_facecolor("#111e32")
        axis.tick_params(colors="#dce8f8")
        for spine in axis.spines.values():
            spine.set_color("#30415b")
    axes[0, 0].bar(departments.index, departments.values, color="#49bfff")
    axes[0, 0].set_title("Eventos por departamento", color="white")
    counts = data.drop_duplicates("evento_id")["magnitud"].map(lambda value: "5.0–5.4" if value < 5.5 else "5.5–5.9" if value < 6 else "6.0–6.9" if value < 7 else "7.0+").value_counts()
    axes[0, 1].pie(counts.values, labels=counts.index, colors=["#49bfff", "#39c6a5", "#f2b84b", "#ff637d"], textprops={"color": "white"})
    axes[0, 1].set_title("Distribución de magnitud", color="white")
    axes[1, 0].bar(["Accuracy", "Precisión", "Recall", "F1"], [report["accuracy"], report["precision_macro"], report["recall_macro"], report["f1_macro"]], color="#39c6a5")
    axes[1, 0].set_title("Métricas (ver evaluación)", color="white")
    class_counts = data["intensidad"].value_counts().reindex(LABELS, fill_value=0)
    axes[1, 1].bar(LABELS, class_counts.values, color=[PALETTE[label] for label in LABELS])
    axes[1, 1].set_title("Clases de intensidad estimada", color="white")
    fig.suptitle("SISMO AI PERÚ · tablero de evidencias", color="white", fontsize=15)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "dashboard_evidencia.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    if len(data) < 500 or data["intensidad"].nunique() != 3:
        raise ValueError("El dataset debe tener 500+ filas y las tres clases de intensidad.")

    X = data[FEATURES].astype(float)
    y = data["intensidad"].astype(str)
    groups = data["evento_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    assert not overlap, f"Fuga de eventos entre conjuntos: {sorted(overlap)}"

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    cv = GroupKFold(n_splits=5)
    score_rows: list[dict[str, float]] = []
    models: dict[int, Pipeline] = {}
    for k in K_VALUES:
        pipeline = crear_pipeline(k)
        cv_results = cross_validate(
            pipeline, X_train, y_train, groups=groups_train, cv=cv,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            return_train_score=False,
        )
        score_rows.append({
            "k": k,
            "f1_macro_cv": float(np.mean(cv_results["test_f1_macro"])),
            "f1_std_cv": float(np.std(cv_results["test_f1_macro"])),
            "accuracy_cv": float(np.mean(cv_results["test_accuracy"])),
        })
        models[k] = pipeline.fit(X_train, y_train)

    best_k = max(score_rows, key=lambda item: (item["f1_macro_cv"], item["accuracy_cv"], -item["k"]))["k"]
    best_model = models[best_k]
    predictions = best_model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    class_report = classification_report(y_test, predictions, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "best_k": int(best_k),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)),
        "class_report": {label: {metric: float(class_report[label][metric]) for metric in ["precision", "recall", "f1-score", "support"]} for label in LABELS},
        "labels": LABELS,
        "confusion_matrix": matrix.astype(int).tolist(),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_events": int(len(train_groups)),
        "test_events": int(len(test_groups)),
        "shared_events": int(len(overlap)),
        "dataset_rows": int(len(data)),
        "dataset_events": int(data["evento_id"].nunique()),
        "class_counts": {label: int(count) for label, count in data["intensidad"].value_counts().reindex(LABELS, fill_value=0).items()},
        "cv_scores": score_rows,
    }
    meta_train = data.iloc[train_idx][["evento_id", "fecha", "localidad", "departamento"]].reset_index(drop=True)
    bundle = {"models": models, "x_train": X_train, "y_train": y_train, "meta_train": meta_train, "features": FEATURES, "labels": LABELS}
    import joblib
    joblib.dump(best_model, MODEL_DIR / "modelo_knn.joblib")
    joblib.dump(bundle, MODEL_DIR / "modelos_por_k.joblib")
    (MODEL_DIR / "resultados.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    guardar_graficos(score_rows, matrix, metrics)
    crear_evidencias(data, metrics)
    print(f"Filas: {len(data)} | eventos: {data['evento_id'].nunique()} | train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"Eventos compartidos: {len(overlap)} | K elegido: {best_k}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1 macro: {metrics['f1_macro']:.4f}")
    print(f"Clases: {metrics['class_counts']}")


if __name__ == "__main__":
    main()
