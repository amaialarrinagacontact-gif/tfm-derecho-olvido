"""PASO 5 (para la memoria). ¿Extrae mejor el LLM que las reglas?

Compara, variable a variable, lo que propuso cada método con tu etiqueta final
revisada a mano (filas de revision.csv con revisado = si). Es la evidencia
cuantitativa del enfoque del proyecto: usar un LLM como preprocesamiento del
texto jurídico en lugar de reglas rígidas.

Uso:  python 05_evaluar_extraccion.py
Salida: tabla en pantalla + models/evaluacion_extraccion.json + .csv
"""
import json

import pandas as pd

from src import config

VARIABLES = [*config.FEATURES, "resultado"]


def normalizar(serie: pd.Series) -> pd.Series:
    num = pd.to_numeric(serie, errors="coerce")
    if num.notna().sum() >= serie.notna().sum() * 0.9:   # variable numérica / binaria
        return num.round().astype("Int64").astype(object).where(num.notna(), "vacío").astype(str)
    return serie.fillna("vacío").astype(str).str.strip().str.lower().replace({"none": "vacío", "nan": "vacío", "": "vacío"})


def main():
    maquina = pd.read_parquet(config.CASOS_EXTRAIDOS)
    humano = pd.read_csv(config.REVISION_CSV, encoding="utf-8-sig", dtype={"revisado": str})
    humano = humano[humano["revisado"].fillna("").str.strip().str.lower().isin(["si", "sí", "x", "1"])]
    if humano.empty:
        raise SystemExit("No hay filas revisadas todavía (pon 'si' en la columna 'revisado' de revision.csv).")

    datos = humano.merge(maquina, on="caso_id", suffixes=("_humano", ""))
    # Solo casos en los que se extrajo algo y que siguen en el corpus (no excluidos por ti)
    datos = datos[datos["metodo_extraccion"].isin(["reglas", "llm"]) &
                  (datos["resultado_humano"].astype(str).str.strip().str.lower() != "excluir")]
    con_llm = datos[datos["metodo_extraccion"] == "llm"]
    print(f"Casos revisados: {len(datos)} (de ellos extraídos con LLM: {len(con_llm)})\n")

    filas = []
    for v in VARIABLES:
        verdad = normalizar(datos[f"{v}_humano"])
        fila = {"variable": config.ETIQUETAS.get(v, v)}
        if f"reglas_{v}" in datos:
            fila["acierto_reglas"] = round(float((normalizar(datos[f"reglas_{v}"]) == verdad).mean()), 3)
        if len(con_llm):
            fila["acierto_llm"] = round(float((normalizar(con_llm[v]) ==
                                               normalizar(con_llm[f"{v}_humano"])).mean()), 3)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    medias = tabla.drop(columns="variable").mean().round(3).to_dict()
    print(tabla.to_string(index=False))
    print("\nAcierto medio:", medias)
    if "acierto_llm" in medias and "acierto_reglas" in medias:
        dif = medias["acierto_llm"] - medias["acierto_reglas"]
        print(f"El LLM {'mejora' if dif > 0 else 'no mejora'} a las reglas en {abs(dif):.1%} de acierto medio.")

    config.MODELS_DIR.mkdir(exist_ok=True)
    tabla.to_csv(config.MODELS_DIR / "evaluacion_extraccion.csv", index=False, encoding="utf-8-sig")
    (config.MODELS_DIR / "evaluacion_extraccion.json").write_text(
        json.dumps({"n_revisados": len(datos), "n_llm": len(con_llm), "por_variable": filas,
                    "acierto_medio": medias}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
