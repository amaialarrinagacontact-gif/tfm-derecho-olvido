"""Pipelines de clasificación y explicabilidad SHAP (Entrega 4, secciones 3 y 6)."""
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def preprocesador() -> ColumnTransformer:
    # add_indicator=True crea la columna explícita "antigüedad desconocida" (Entrega 4, sección 4)
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.FEATURES_CAT),
        ("bin", "passthrough", config.FEATURES_BIN),
        ("num", num, config.FEATURES_NUM),
    ])


def candidatos() -> dict:
    """nombre -> (pipeline, rejilla de hiperparámetros)"""
    return {
        "baseline_clase_mayoritaria": (
            Pipeline([("pre", preprocesador()), ("clf", DummyClassifier(strategy="most_frequent"))]), {}),
        "regresion_logistica": (
            Pipeline([("pre", preprocesador()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.1, 0.3, 1.0, 3.0]}),
        "random_forest": (
            Pipeline([("pre", preprocesador()),
                      ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                     random_state=config.SEED, n_jobs=-1))]),
            {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [3, 5, 10]}),
    }


def _variable_original(nombre_columna: str) -> str:
    """'cat__tipo_solicitante_privada' -> 'tipo_solicitante'; indicador de nulos -> antigüedad."""
    limpio = nombre_columna.split("__", 1)[-1]
    for f in sorted(config.FEATURES, key=len, reverse=True):
        if f in limpio:
            return f
    return limpio


class Explicador:
    """Contribuciones SHAP por variable original (suma de sus columnas one-hot)."""

    def __init__(self, pipeline: Pipeline, X_fondo: pd.DataFrame):
        self.pre, self.clf = pipeline["pre"], pipeline["clf"]
        fondo = self.pre.transform(X_fondo)
        if isinstance(self.clf, LogisticRegression):
            self.exp = shap.LinearExplainer(self.clf, fondo)
        elif isinstance(self.clf, RandomForestClassifier):
            self.exp = shap.TreeExplainer(self.clf)
        else:
            self.exp = None
        self.columnas = [_variable_original(c) for c in self.pre.get_feature_names_out()]
        self.clases = list(self.clf.classes_)

    def _valores(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve array (n_casos, n_variables_originales, n_clases)."""
        v = self.exp(self.pre.transform(X)).values
        if v.ndim == 2:
            v = v[:, :, None]
        salida = np.zeros((v.shape[0], len(config.FEATURES), v.shape[2]))
        for j, f in enumerate(config.FEATURES):
            idx = [i for i, c in enumerate(self.columnas) if c == f]
            salida[:, j, :] = v[:, idx, :].sum(axis=1)
        return salida

    def explicar(self, x: pd.DataFrame, clase: str) -> list[dict]:
        """Contribución de cada variable a la probabilidad de `clase` para un único caso."""
        if self.exp is None:
            return []
        v = self._valores(x)[0, :, self.clases.index(clase)]
        filas = [{"variable": f, "valor": x.iloc[0][f], "contribucion": float(c)}
                 for f, c in zip(config.FEATURES, v)]
        return sorted(filas, key=lambda r: abs(r["contribucion"]), reverse=True)

    def importancia_global(self, X: pd.DataFrame) -> dict:
        if self.exp is None:
            return {}
        v = np.abs(self._valores(X)).mean(axis=(0, 2))
        return dict(sorted(zip(config.FEATURES, map(float, v)), key=lambda t: -t[1]))
