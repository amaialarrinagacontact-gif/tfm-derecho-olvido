"""Capa de apoyo: precedentes similares (Entrega 3, secciones 5.2 y 6).

Se invoca DESPUÉS de la predicción. Todo es local: embeddings multilingual-e5 en tu
ordenador y búsqueda por coseno con NumPy (sin API ni base de datos externa). Si no
existe el índice, usa similitud estructurada para que la app funcione igualmente.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config


def caso_a_texto(caso: dict) -> str:
    """Convierte el formulario en una descripción en lenguaje natural para buscar en los hechos."""
    e = config.ETIQUETAS
    partes = [
        f"Reclamación de derecho de supresión presentada por una {e[caso['tipo_solicitante']].lower()}",
        f"frente a {e[caso['tipo_responsable']].lower()}",
        f"sobre {e[caso['tipo_contenido']].lower()}",
    ]
    if pd.notna(caso.get("antiguedad_info_años")):
        partes.append(f"publicada hace {int(caso['antiguedad_info_años'])} años")
    if caso.get("condena_penal_previa"):
        partes.append("relacionada con una condena penal firme")
    if caso.get("persona_fallecida"):
        partes.append("referida a una persona fallecida")
    if caso.get("interes_historico"):
        partes.append("el responsable alega interés histórico")
    if caso.get("alega_libertad_informacion"):
        partes.append("el responsable alega la libertad de información o el interés público")
    return ", ".join(partes) + "."


def cobertura(caso: dict, gold: pd.DataFrame) -> int:
    """Nº de casos del corpus con el mismo solicitante, responsable y contenido."""
    m = ((gold["tipo_solicitante"] == caso["tipo_solicitante"]) &
         (gold["tipo_responsable"] == caso["tipo_responsable"]) &
         (gold["tipo_contenido"] == caso["tipo_contenido"]))
    return int(m.sum())


def _estructurado(caso: dict, gold: pd.DataFrame, k: int) -> pd.DataFrame:
    g = gold.copy()
    score = sum((g[c] == caso[c]).astype(float) for c in config.FEATURES_CAT) * 2
    score += sum((g[c] == int(caso[c])).astype(float) for c in config.FEATURES_BIN)
    if pd.notna(caso.get("antiguedad_info_años")):
        dif = (g["antiguedad_info_años"] - caso["antiguedad_info_años"]).abs()
        score += (1 - (dif / 10).clip(upper=1)).fillna(0)
    g["similitud"] = score / score.max() if score.max() > 0 else 0
    return g.sort_values(["similitud", "año_resolucion"], ascending=False).head(k)


@lru_cache(maxsize=1)
def _modelo_embeddings():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDINGS_LOCAL)      # local: la 1.ª vez se descarga (~470 MB)


@lru_cache(maxsize=1)
def _indice():
    d = np.load(config.INDICE_PRECEDENTES, allow_pickle=False)
    return d["vectores"], d["caso_ids"]


def _rag(caso: dict, gold: pd.DataFrame, k: int) -> pd.DataFrame | None:
    if not config.INDICE_PRECEDENTES.exists():
        return None
    vectores, caso_ids = _indice()
    q = _modelo_embeddings().encode(["query: " + caso_a_texto(caso)], normalize_embeddings=True)[0]
    similitud = vectores @ q                                  # coseno (vectores ya normalizados)
    vistos = {}
    for i in np.argsort(-similitud):
        vistos.setdefault(str(caso_ids[i]), float(similitud[i]))   # mejor chunk de cada caso (1:N -> 1)
        if len(vistos) == k:
            break
    sim = pd.DataFrame({"caso_id": list(vistos), "similitud": list(vistos.values())})
    return sim.merge(gold, on="caso_id", how="inner")


def buscar(caso: dict, gold: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, str]:
    try:
        res = _rag(caso, gold, k)
        if res is not None and len(res):
            return res, "semantica"
    except Exception as e:   # el RAG es prescindible (Entrega 3, sección 11)
        print("RAG no disponible, uso similitud estructurada:", e)
    return _estructurado(caso, gold, k), "estructurada"
