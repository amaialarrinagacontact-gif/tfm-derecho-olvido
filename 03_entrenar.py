"""PASO 3. Entrenamiento y evaluación (Entrega 4, secciones 6 y 7).

- Partición TEMPORAL: último 25% de casos por año_resolucion = test.
- CV estratificada k=5 SOLO dentro de train para elegir hiperparámetros y modelo.
- Métrica principal F1-macro; se compara siempre con el baseline de clase mayoritaria.
- Regla de decisión: mejor F1-macro en CV; si la ventaja sobre la regresión logística
  es < MARGEN_EMPATE, se elige la regresión logística (más interpretable).
- El modelo final para la app se reentrena con todo el corpus.
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.modelo import Explicador, candidatos


def particion_temporal(df: pd.DataFrame):
    df = df.sort_values([config.COLUMNA_TEMPORAL, "caso_id"]).reset_index(drop=True)
    corte = int(len(df) * (1 - config.TEST_FRACCION))
    return df.iloc[:corte], df.iloc[corte:]


def main():
    df = pd.read_parquet(config.GOLD_CASOS)
    conteo = df[config.TARGET].value_counts().reindex(config.CLASES, fill_value=0)
    if len(df) < 30 or conteo.min() < 5:
        raise SystemExit(f"\nTodavía hay pocos casos para entrenar: {len(df)} en total, por clase {conteo.to_dict()}.\n"
                         "Se necesitan al menos 30 casos y 5 de cada clase (lo recomendable son 250 y 20 por clase).\n"
                         "Sigue añadiendo y revisando resoluciones; el modelo anterior no se ha tocado.")
    train, test = particion_temporal(df)
    X_tr, y_tr = train[config.FEATURES], train[config.TARGET]
    X_te, y_te = test[config.FEATURES], test[config.TARGET]

    print(f"Corpus: {len(df)} casos | train {len(train)} ({train['año_resolucion'].min()}-{train['año_resolucion'].max()})"
          f" | test {len(test)} ({test['año_resolucion'].min()}-{test['año_resolucion'].max()})")
    print("Clases en test:", y_te.value_counts().to_dict())

    k = int(min(5, y_tr.value_counts().min()))
    if k < 2:
        raise SystemExit("Alguna clase tiene menos de 2 casos en train: no se puede validar. Etiqueta más casos.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.SEED)

    resultados, ajustados = {}, {}
    for nombre, (pipe, rejilla) in candidatos().items():
        gs = GridSearchCV(pipe, rejilla or {"clf__strategy": ["most_frequent"]}, scoring="f1_macro", cv=cv)
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        mejor = gs.best_estimator_
        pred = mejor.predict(X_te)
        r = {
            "cv_f1_macro_media": round(float(gs.cv_results_["mean_test_score"][i]), 3),
            "cv_f1_macro_std": round(float(gs.cv_results_["std_test_score"][i]), 3),
            "test_f1_macro": round(float(f1_score(y_te, pred, average="macro", labels=config.CLASES, zero_division=0)), 3),
            "hiperparametros": {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in gs.best_params_.items()},
        }
        try:
            r["test_auc_ovr"] = round(float(roc_auc_score(y_te, mejor.predict_proba(X_te), multi_class="ovr",
                                                          labels=list(mejor.classes_))), 3)
        except ValueError:
            r["test_auc_ovr"] = None   # falta alguna clase en test
        resultados[nombre], ajustados[nombre] = r, mejor
        print(f"{nombre:28s} CV F1 {r['cv_f1_macro_media']:.3f} ± {r['cv_f1_macro_std']:.3f} | test temporal F1 {r['test_f1_macro']:.3f}")

    # Regla de decisión
    reales = {n: r for n, r in resultados.items() if n != "baseline_clase_mayoritaria"}
    elegido = max(reales, key=lambda n: reales[n]["cv_f1_macro_media"])
    if (elegido != "regresion_logistica" and
            reales[elegido]["cv_f1_macro_media"] - reales["regresion_logistica"]["cv_f1_macro_media"] < config.MARGEN_EMPATE):
        elegido = "regresion_logistica"

    base = resultados["baseline_clase_mayoritaria"]["test_f1_macro"]
    ganancia = resultados[elegido]["test_f1_macro"] - base
    aceptado = ganancia >= config.MARGEN_ACEPTACION

    pred = ajustados[elegido].predict(X_te)
    informe = classification_report(y_te, pred, labels=config.CLASES, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_te, pred, labels=config.CLASES).tolist()

    # Análisis de errores por franja de antigüedad (Entrega 4, sección 7)
    franjas = pd.cut(test["antiguedad_info_años"], [-1, 2, 5, 10, 200], labels=["0-2", "3-5", "6-10", ">10"])
    franjas = franjas.cat.add_categories("desconocida").fillna("desconocida")
    por_franja = {str(f): {"n": int(m.sum()),
                           "f1_macro": round(float(f1_score(y_te[m], pred[m], average="macro", zero_division=0)), 3)}
                  for f in franjas.unique() for m in [(franjas == f).values]}

    # Modelo final: mismo pipeline e hiperparámetros, reentrenado con todo el corpus
    final = clone(ajustados[elegido]).fit(df[config.FEATURES], df[config.TARGET])
    explicador = Explicador(final, df[config.FEATURES].sample(min(100, len(df)), random_state=config.SEED))

    metricas = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "n_casos": len(df), "n_train": len(train), "n_test": len(test),
        "año_corte_test": int(test["año_resolucion"].min()),
        "modelo_elegido": elegido,
        "aceptado": bool(aceptado),
        "ganancia_f1_sobre_baseline": round(float(ganancia), 3),
        "modelos": resultados,
        "test_informe_por_clase": informe,
        "test_matriz_confusion": {"orden": config.CLASES, "valores": matriz},
        "test_f1_por_franja_antiguedad": por_franja,
        "importancia_global_shap": explicador.importancia_global(df[config.FEATURES]),
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nElegido: {elegido} | ganancia F1 sobre baseline en test temporal: {ganancia:+.3f}")
    print("ACEPTADO: la app mostrará la estimación." if aceptado else
          "NO ACEPTADO: la app mostrará solo precedentes (Entrega 4, sección 7).")
    print("Importancia global:", metricas["importancia_global_shap"])


if __name__ == "__main__":
    main()
