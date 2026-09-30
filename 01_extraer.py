"""PASO 1. PDFs de la AEPD -> data/processed/casos_extraidos.parquet + revision.csv

Uso:
    python 01_extraer.py --sin-llm  # SIN ninguna API: extracción por reglas (gratis, al instante)
    python 01_extraer.py            # con modelo de lenguaje (clave en .env; ver src/config.py)

Se puede interrumpir y relanzar: continúa donde lo dejó. Las filas de revision.csv
que ya marcaste como revisadas NO se sobrescriben al volver a ejecutarlo.

Siempre se guardan las dos extracciones (reglas_* y la del LLM) para poder
compararlas después con 05_evaluar_extraccion.py.
"""
import argparse
import hashlib
import re

import pandas as pd

from src import config, reglas
from src.secciones import procesar_pdf

# Relevante = pide borrar/desindexar (incluye "cancelación"/"oposición" de la etapa LOPD, 2014-2018)
#             Y trata de contenido en internet (no bases de datos de empresas o administraciones).
RE_SUPRESION = re.compile(r"supresi[oó]n|olvido|desindexa|cancelaci[oó]n|oposici[oó]n|art[íi]culo\s+17", re.IGNORECASE)
RE_INTERNET = re.compile(r"buscador|google|bing|yahoo|internet|url|enlace|p[áa]gina web|bolet[íi]n oficial|"
                         r"diario|peri[óo]dico|hemeroteca|red social|facebook|twitter|instagram|youtube|tiktok",
                         re.IGNORECASE)
VARIABLES = [*config.FEATURES, "resultado"]


def caso_id(expediente, fichero):
    if expediente:
        return expediente.replace("/", "-")
    return "PDF-" + hashlib.md5(fichero.encode()).hexdigest()[:10]


def guardar(filas):
    pd.DataFrame(filas).to_parquet(config.CASOS_EXTRAIDOS, index=False)


def exportar_revision(df: pd.DataFrame):
    """Genera revision.csv conservando las filas que ya revisaste a mano."""
    cols = ["caso_id", "fichero", "año_resolucion", "secciones_ok", "relevante", "metodo_extraccion",
            *VARIABLES, "evidencia"]
    nuevo = df.reindex(columns=cols)
    nuevo["revisado"] = ""
    if config.REVISION_CSV.exists():
        viejo = pd.read_csv(config.REVISION_CSV, encoding="utf-8-sig", dtype={"revisado": str})
        hechos = viejo[viejo["revisado"].fillna("").str.strip() != ""]
        nuevo = pd.concat([hechos, nuevo[~nuevo["caso_id"].isin(hechos["caso_id"])]], ignore_index=True)
        if len(hechos):
            print(f"Se conservan {len(hechos)} filas que ya habías revisado.")
    nuevo.to_csv(config.REVISION_CSV, index=False, encoding="utf-8-sig")


def main(sin_llm: bool):
    usar_llm = not sin_llm
    if usar_llm and not config.LLM_DISPONIBLE:
        print("No hay modelo de lenguaje configurado en .env: se usará solo la extracción por reglas.\n")
        usar_llm = False

    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No hay PDFs en {config.RAW_DIR}")

    previo = pd.read_parquet(config.CASOS_EXTRAIDOS) if config.CASOS_EXTRAIDOS.exists() else pd.DataFrame()
    if len(previo) and usar_llm and "metodo_extraccion" in previo:
        pendiente = previo["secciones_ok"] & previo["relevante"] & (previo["metodo_extraccion"] != "llm")
        previo = previo[~pendiente]            # los hechos solo con reglas se repiten con el LLM
    elif len(previo) and "metodo_extraccion" not in previo:
        previo = pd.DataFrame()                # formato antiguo: se rehace todo
    ya = set(previo["fichero"]) if len(previo) else set()
    filas = previo.to_dict("records") if len(previo) else []

    for n, ruta in enumerate(pdfs, 1):
        if ruta.name in ya:
            continue
        print(f"[{n}/{len(pdfs)}] {ruta.name}")
        r = procesar_pdf(ruta)
        fila = {
            "caso_id": caso_id(r.expediente, r.fichero),
            "fichero": r.fichero,
            "expediente": r.expediente,
            "año_resolucion": r.año,
            "secciones_ok": r.secciones_ok,
            "relevante": bool(RE_SUPRESION.search(r.texto_hechos) and RE_INTERNET.search(r.texto_hechos)),
            "texto_hechos": r.texto_hechos,
            "texto_fundamentos": r.texto_fundamentos,
            "metodo_extraccion": "ninguno",
        }
        if r.secciones_ok and fila["relevante"]:
            # 1) Reglas: siempre (gratis). Se guardan aparte como línea base.
            por_reglas = {**reglas.extraer_features(r.texto_hechos),
                          "resultado": reglas.extraer_resultado(r.texto_fallo, r.texto_fundamentos)}
            fila.update({f"reglas_{k}": por_reglas[k] for k in VARIABLES})
            fila.update(por_reglas)
            fila["metodo_extraccion"] = "reglas"
            # 2) LLM: si está disponible, sustituye a las reglas en las columnas principales.
            if usar_llm:
                from src.llm import CuotaAgotada, extraer_features, extraer_resultado
                try:
                    fila.update(extraer_features(r.texto_hechos).model_dump())       # SOLO hechos
                    fila["resultado"] = extraer_resultado(r.texto_fallo, r.texto_fundamentos).resultado  # fallo + motivo
                    fila["metodo_extraccion"] = "llm"
                except CuotaAgotada:
                    guardar(filas)
                    raise SystemExit(
                        f"\nEl proveedor ({config.LLM_PROVIDER}) rechaza las peticiones por límite de uso.\n"
                        f"Guardados {len(filas)} PDF. Espera unos minutos y vuelve a ejecutar: continuará "
                        "donde lo dejó.\nO sigue sin API:  python 01_extraer.py --sin-llm")
        filas.append(fila)
        guardar(filas)                          # guardado tras cada PDF

    df = pd.DataFrame(filas).drop_duplicates("caso_id", keep="last")
    df.to_parquet(config.CASOS_EXTRAIDOS, index=False)
    exportar_revision(df)

    print(f"\n{len(df)} casos | por LLM: {(df['metodo_extraccion'] == 'llm').sum()} | "
          f"por reglas: {(df['metodo_extraccion'] == 'reglas').sum()}")
    print(f"No tratan de olvido digital: {(~df['relevante']).sum()} | "
          f"Secciones sin separar: {(~df['secciones_ok']).sum()} (revísalos a mano)")
    print(f"Ahora revisa {config.REVISION_CSV}: corrige lo que haga falta y pon 'si' en 'revisado'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sin-llm", action="store_true", help="No usar ninguna API (solo reglas)")
    main(p.parse_args().sin_llm)
