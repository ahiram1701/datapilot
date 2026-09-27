"""Herramientas de análisis y ML que los agentes pueden invocar.

Todas reciben un DataFrame y devuelven dicts pequeños y serializables:
el LLM solo necesita un resumen, no los datos crudos.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .from_scratch import LinearRegressionGD

MODEL_TYPES = ["linear", "tree", "random_forest"]


def _r(x: float) -> float:
    return round(float(x), 4)


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df), "columns": len(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
        "numeric_summary": df.describe().round(3).to_dict(),
        "categorical_uniques": {c: int(df[c].nunique())
                                for c in df.select_dtypes(exclude="number").columns},
        "head": df.head(3).to_dict(orient="records"),
    }


def correlations(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    corr = df.select_dtypes("number").corr()
    if target:
        if target not in corr:
            return {"error": f"'{target}' no es numérica o no existe"}
        s = corr[target].drop(target).sort_values(key=abs, ascending=False)
        return {"target": target, "correlations": {k: _r(v) for k, v in s.items()}}
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
             .stack().sort_values(key=abs, ascending=False).head(10))
    return {"top_pairs": [{"a": a, "b": b, "r": _r(v)} for (a, b), v in pairs.items()]}


def segment_analysis(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Promedio del objetivo por segmento: por categoría en columnas de texto y
    por cuartil en columnas numéricas. Con un objetivo 0/1 es la tasa (p. ej.
    de abandono) de cada grupo; complementa a `correlations`, que ignora las
    variables categóricas y las relaciones no lineales."""
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe. Columnas: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        return {"error": f"'{target}' debe ser numérica (p. ej. 0/1) para promediarla"}
    data = df.dropna(subset=[target])
    segments: dict[str, Any] = {}
    for col in data.columns.drop(target):
        s = data[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            groups = pd.qcut(s, 4, duplicates="drop").astype(str)
        elif s.nunique() <= 20:
            groups = s.astype(str)
        else:
            continue  # texto libre o IDs: no aportan segmentos útiles
        stats = data.groupby(groups, observed=True)[target].agg(["mean", "size"])
        segments[col] = {str(k): {"mean": _r(r["mean"]), "n": int(r["size"])}
                         for k, r in stats.sort_values("mean", ascending=False).iterrows()}
    return {"target": target, "overall_mean": _r(data[target].mean()), "segments": segments}


def is_classification(y: pd.Series) -> bool:
    """Heurística: texto/booleano o pocos valores enteros distintos => clasificación."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    return pd.api.types.is_integer_dtype(y) and y.nunique() <= 10


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe. Columnas: {list(df.columns)}")
    data = df.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])
    # One-hot solo para categóricas de baja cardinalidad (evita explosión de columnas)
    cats = [c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() <= 20]
    X = pd.get_dummies(X[list(X.select_dtypes("number").columns) + cats], columns=cats, dtype=float)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def _build_model(model_type: str, classification: bool):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type debe ser uno de {MODEL_TYPES}")
    if classification:
        return {"linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)}[model_type]
    return {"linear": make_pipeline(StandardScaler(), LinearRegression()),
            "tree": DecisionTreeRegressor(max_depth=5, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=42)}[model_type]


def _importances(model, columns) -> dict[str, Any]:
    """Qué variables usa más el modelo, SIN perder la dirección del efecto cuando
    el modelo la tiene: el LLM necesita saber si una variable sube o baja el objetivo."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        top = sorted(zip(columns, est.feature_importances_), key=lambda kv: -kv[1])[:8]
        return {"feature_importance": {k: _r(v) for k, v in top},
                "note": "La importancia de un árbol indica cuánto se usa la variable, no si "
                        "sube o baja el objetivo; para la dirección usa correlations o segment_analysis."}
    if hasattr(est, "coef_"):
        coef = np.atleast_2d(est.coef_)
        if coef.shape[0] == 1:  # regresión o clasificación binaria: el signo es interpretable
            top = sorted(zip(columns, coef[0]), key=lambda kv: -abs(kv[1]))[:8]
            return {"coefficients": {k: _r(v) for k, v in top},
                    "note": "Coeficientes sobre variables estandarizadas: positivo = al aumentar la "
                            "variable aumenta el objetivo (o la probabilidad de la clase 1); negativo = lo reduce."}
        vals = np.abs(coef).mean(axis=0)  # multiclase: sin un único signo por variable
        top = sorted(zip(columns, vals), key=lambda kv: -kv[1])[:8]
        return {"feature_importance": {k: _r(v) for k, v in top},
                "note": "Magnitud media de coeficientes (multiclase); no indica dirección."}
    return {}


def train_model(df: pd.DataFrame, target: str, model_type: str = "random_forest",
                test_size: float = 0.2) -> dict[str, Any]:
    X, y = prepare_xy(df, target)
    clf = is_classification(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if clf else None)
    model = _build_model(model_type, clf).fit(X_tr, y_tr)
    pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
    if clf:
        metrics = {"accuracy_train": _r(accuracy_score(y_tr, pred_tr)),
                   "accuracy_test": _r(accuracy_score(y_te, pred_te)),
                   "f1_macro_test": _r(f1_score(y_te, pred_te, average="macro"))}
    else:
        metrics = {"r2_train": _r(r2_score(y_tr, pred_tr)),
                   "r2_test": _r(r2_score(y_te, pred_te)),
                   "mae_test": _r(mean_absolute_error(y_te, pred_te)),
                   "rmse_test": _r(np.sqrt(mean_squared_error(y_te, pred_te)))}
    return {"task": "classification" if clf else "regression", "model": model_type,
            "n_train": len(X_tr), "n_test": len(X_te), "metrics": metrics,
            **_importances(model, X.columns)}


def cross_validate(df: pd.DataFrame, target: str, model_type: str = "random_forest",
                   folds: int = 5) -> dict[str, Any]:
    X, y = prepare_xy(df, target)
    clf = is_classification(y)
    scoring = "accuracy" if clf else "r2"
    scores = cross_val_score(_build_model(model_type, clf), X, y, cv=folds, scoring=scoring)
    return {"model": model_type, "scoring": scoring, "folds": folds,
            "scores": [_r(s) for s in scores], "mean": _r(scores.mean()), "std": _r(scores.std())}


def gradient_descent_regression(df: pd.DataFrame, target: str, learning_rate: float = 0.05,
                                epochs: int = 500) -> dict[str, Any]:
    """Regresión lineal implementada a mano (NumPy) comparada con scikit-learn."""
    X, y = prepare_xy(df, target)
    if is_classification(y):
        return {"error": "gradient_descent_regression solo aplica a objetivos numéricos continuos"}
    Xs = StandardScaler().fit_transform(X)
    gd = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs).fit(Xs, y.to_numpy(float))
    sk = LinearRegression().fit(Xs, y)
    return {
        "epochs": epochs, "learning_rate": learning_rate,
        "loss_curve_every_10pct": [_r(l) for l in gd.loss_history[:: max(1, epochs // 10)]],
        "final_mse": _r(gd.loss_history[-1]),
        "r2_gradient_descent": _r(r2_score(y, gd.predict(Xs))),
        "r2_sklearn_closed_form": _r(r2_score(y, sk.predict(Xs))),
        "max_coef_difference_vs_sklearn": _r(np.max(np.abs(gd.w - sk.coef_))),
    }
