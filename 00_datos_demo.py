"""Genera un gold_casos SINTÉTICO para probar la app y el entrenamiento mientras
etiquetas los casos reales. NO usar sus métricas en la memoria del TFM.

Uso: python 00_datos_demo.py   (sobrescribe data/gold/gold_casos.parquet)
"""
import numpy as np
import pandas as pd

from src import config

rng = np.random.default_rng(config.SEED)
N = 320


def generar():
    años = rng.choice(np.arange(2014, 2026), N, p=np.r_[np.full(4, 0.04), np.full(8, 0.105)])
    df = pd.DataFrame({
        "caso_id": [f"DEMO-{i:04d}" for i in range(N)],
        "tipo_solicitante": rng.choice(config.TIPO_SOLICITANTE, N, p=[0.8, 0.2]),
        "tipo_responsable": rng.choice(config.TIPO_RESPONSABLE, N, p=[0.45, 0.25, 0.12, 0.12, 0.06]),
        "tipo_contenido": rng.choice(config.TIPO_CONTENIDO, N, p=[0.5, 0.1, 0.2, 0.12, 0.08]),
        "condena_penal_previa": rng.binomial(1, 0.15, N),
        "persona_fallecida": rng.binomial(1, 0.04, N),
        "interes_historico": rng.binomial(1, 0.1, N),
        "alega_libertad_informacion": rng.binomial(1, 0.35, N),
        "antiguedad_info_años": rng.integers(0, 25, N).astype(float),
        "jurisdiccion": "AEPD",
        "año_resolucion": años,
    })
    df.loc[rng.random(N) < 0.2, "antiguedad_info_años"] = np.nan

    # Puntuaciones latentes inspiradas en los criterios de ponderación (Google Spain / Directrices 5/2019)
    ant = df["antiguedad_info_años"].fillna(8)
    s_est = (1.2 * (df.tipo_solicitante == "privada") + 0.08 * ant + 0.8 * (df.tipo_responsable == "motor_busqueda")
             - 1.0 * df.condena_penal_previa * (ant < 10) - 1.0 * df.interes_historico - 0.6 * df.alega_libertad_informacion)
    s_par = 0.9 * (df.tipo_responsable == "hemeroteca") + 0.5 * (df.tipo_contenido == "noticia") + 0.3
    s_des = (1.3 * (df.tipo_solicitante == "publica") + 1.0 * (df.tipo_responsable == "registro_publico")
             + 0.6 * df.condena_penal_previa + 0.5 * df.alega_libertad_informacion + 0.5)
    logits = np.c_[s_est, s_par, s_des] + rng.gumbel(size=(N, 3)) * 0.9
    df["resultado"] = np.array(config.CLASES)[logits.argmax(1)]

    e = config.ETIQUETAS
    df["texto_hechos"] = [
        f"[SINTÉTICO] La parte reclamante, {e[r.tipo_solicitante].lower()}, solicitó la supresión de "
        f"{e[r.tipo_contenido].lower()} publicada "
        f"{'hace ' + str(int(r.antiguedad_info_años)) + ' años' if pd.notna(r.antiguedad_info_años) else 'en fecha no determinada'}"
        f" ante {e[r.tipo_responsable].lower()}. "
        f"{'Los hechos guardan relación con una condena penal firme. ' if r.condena_penal_previa else ''}"
        f"{'El responsable alegó interés histórico. ' if r.interes_historico else ''}"
        for r in df.itertuples()]
    df["fichero"] = "sintetico"
    return df


if __name__ == "__main__":
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = generar()
    df.to_parquet(config.GOLD_CASOS, index=False)
    print(f"{len(df)} casos SINTÉTICOS en {config.GOLD_CASOS}")
    print(df["resultado"].value_counts().to_string())
