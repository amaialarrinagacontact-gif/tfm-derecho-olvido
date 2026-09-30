"""PASO 4 (opcional). Índice semántico LOCAL de los hechos para buscar precedentes.

- Embeddings: intfloat/multilingual-e5-small (multilingüe, se ejecuta en tu PC, gratis).
  La primera vez se descarga el modelo (~470 MB); después funciona sin internet.
- Índice: matriz NumPy en data/gold/indice_precedentes.npz. Con unos cientos de casos
  la búsqueda exacta por coseno es instantánea, así que no hace falta una base de
  datos vectorial.
- Chunks de ~1000 caracteres con solape de 200; cada chunk guarda su caso_id (relación 1:N).

Sin este paso la app sigue funcionando (precedentes por coincidencia de características).
"""
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.precedentes import _modelo_embeddings


def main():
    gold = pd.read_parquet(config.GOLD_CASOS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos, ids = [], []
    for _, fila in gold.iterrows():
        for trozo in splitter.split_text(fila["texto_hechos"] or ""):
            textos.append("passage: " + trozo)      # prefijo que requiere e5
            ids.append(fila["caso_id"])
    print(f"{len(textos)} chunks de {len(gold)} casos. Calculando embeddings en local...")
    vectores = _modelo_embeddings().encode(textos, normalize_embeddings=True, batch_size=16,
                                           show_progress_bar=True).astype(np.float32)
    np.savez_compressed(config.INDICE_PRECEDENTES, vectores=vectores, caso_ids=np.array(ids, dtype=str))
    print(f"Índice guardado en {config.INDICE_PRECEDENTES} ({vectores.shape[0]} vectores de {vectores.shape[1]} dimensiones).")


if __name__ == "__main__":
    main()
