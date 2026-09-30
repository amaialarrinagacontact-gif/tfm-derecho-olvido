"""PASO 2. revision.csv (corregido a mano) -> data/gold/gold_casos.parquet

Valida cada valor contra las taxonomías de config.py y excluye los casos con
resultado 'excluir' o vacío (criterio de registro válido, Entrega 3 sección 10).

Uso:
    python 02_construir_gold.py                  # usa todos los casos
    python 02_construir_gold.py --solo-revisados  # solo filas con revisado = si
"""
import argparse

import pandas as pd

from src import config

VALIDOS = {
    "tipo_solicitante": config.TIPO_SOLICITANTE,
    "tipo_responsable": config.TIPO_RESPONSABLE,
    "tipo_contenido": config.TIPO_CONTENIDO,
    "resultado": config.CLASES,
}


def main(solo_revisados: bool):
    rev = pd.read_csv(config.REVISION_CSV, encoding="utf-8-sig")
    extraidos = pd.read_parquet(config.CASOS_EXTRAIDOS)[["caso_id", "texto_hechos"]]
    n0 = len(rev)

    if solo_revisados:
        rev = rev[rev["revisado"].astype(str).str.strip().str.lower().isin(["si", "sí", "x", "1"])]

    if "relevante" in rev:
        rev = rev[rev["relevante"].astype(str).str.lower().isin(["true", "1", "si", "sí"])]
    rev["resultado"] = rev["resultado"].astype(str).str.strip().str.lower()
    descartados = rev[~rev["resultado"].isin(config.CLASES)]
    rev = rev[rev["resultado"].isin(config.CLASES)].copy()

    errores = []
    for col, validos in VALIDOS.items():
        rev[col] = rev[col].astype(str).str.strip().str.lower()
        malos = rev[~rev[col].isin(validos)]
        errores += [f"{r.caso_id}: {col}='{r[col]}'" for _, r in malos.iterrows()]
    for col in config.FEATURES_BIN:
        rev[col] = pd.to_numeric(rev[col], errors="coerce").fillna(0).astype(int)  # 1 solo si consta
    rev["antiguedad_info_años"] = pd.to_numeric(rev["antiguedad_info_años"], errors="coerce")  # NaN = no consta
    rev["año_resolucion"] = pd.to_numeric(rev["año_resolucion"], errors="coerce")
    errores += [f"{c}: falta año_resolucion" for c in rev.loc[rev["año_resolucion"].isna(), "caso_id"]]

    if errores:
        print("Corrige estos valores en revision.csv y vuelve a ejecutar:")
        print("\n".join("  - " + e for e in errores[:50]))
        raise SystemExit(1)

    if rev.empty:
        raise SystemExit("\nNo hay ningún caso válido todavía, así que no se ha modificado gold_casos.\n"
                         "Revisa revision.csv: cada fila necesita un 'resultado' válido "
                         "(estimada / desestimada / estimada_parcialmente) y, si usas --solo-revisados, 'si' en 'revisado'.")
    gold = rev.merge(extraidos, on="caso_id", how="left")
    gold["jurisdiccion"] = "AEPD"
    gold["año_resolucion"] = gold["año_resolucion"].astype(int)
    cols = ["caso_id", *config.FEATURES, "jurisdiccion", "año_resolucion", "resultado", "texto_hechos", "fichero"]
    gold[cols].to_parquet(config.GOLD_CASOS, index=False)

    print(f"gold_casos: {len(gold)} casos válidos de {n0} ({len(descartados)} excluidos por fallo no clasificable).")
    print(gold["resultado"].value_counts().to_string())
    print(f"Sin antigüedad conocida: {gold['antiguedad_info_años'].isna().mean():.0%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solo-revisados", action="store_true")
    main(p.parse_args().solo_revisados)
