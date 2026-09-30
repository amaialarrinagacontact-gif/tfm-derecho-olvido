"""VERIFICACIÓN ANTES DE ENTREGAR. Revisa todo el proyecto y dice qué está bien y qué falta.

Uso:  python verificar.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
resultados = []


def ok(msg):    resultados.append(("OK", msg))
def aviso(msg): resultados.append(("AVISO", msg))
def falta(msg): resultados.append(("FALTA", msg))


def seccion(titulo):
    resultados.append(("", f"\n--- {titulo} ---"))


# 1. Entorno ---------------------------------------------------------------
seccion("1. Entorno")
if sys.version_info >= (3, 10):
    ok(f"Python {sys.version.split()[0]}")
else:
    falta(f"Python {sys.version.split()[0]}: se necesita 3.10 o superior")
if sys.prefix == sys.base_prefix:
    aviso("No estás dentro del entorno .venv (actívalo con .\\.venv\\Scripts\\Activate.ps1)")
faltan_libs = []
for mod in ["pandas", "sklearn", "shap", "streamlit", "pdfplumber", "joblib", "pydantic"]:
    try:
        __import__(mod)
    except ImportError:
        faltan_libs.append(mod)
if faltan_libs:
    falta(f"Faltan librerías {faltan_libs}: pip install -r requirements.txt")
    sys.exit("\n".join(f"[{e}] {m}" for e, m in resultados))
ok("Librerías principales instaladas")

from src import config  # noqa: E402

# 2. Datos -----------------------------------------------------------------
seccion("2. Datos")
pdfs = list(config.RAW_DIR.glob("*.pdf"))
(ok if len(pdfs) >= 250 else aviso if pdfs else falta)(
    f"{len(pdfs)} PDF en data/raw/aepd_resoluciones (objetivo: 250-400)")

import pandas as pd  # noqa: E402

if config.CASOS_EXTRAIDOS.exists():
    ext = pd.read_parquet(config.CASOS_EXTRAIDOS)
    sin_procesar = len(pdfs) - len(ext)
    ok(f"Paso 1 hecho: {len(ext)} PDF procesados "
       f"(LLM: {(ext.get('metodo_extraccion') == 'llm').sum()}, "
       f"reglas: {(ext.get('metodo_extraccion') == 'reglas').sum()})")
    if sin_procesar > 0:
        aviso(f"{sin_procesar} PDF nuevos sin procesar: vuelve a ejecutar 01_extraer.py")
    if (~ext["secciones_ok"]).any():
        aviso(f"{(~ext['secciones_ok']).sum()} PDF sin secciones separadas: revísalos o retíralos")
else:
    falta("Paso 1 sin hacer: python 01_extraer.py --sin-llm")

if config.REVISION_CSV.exists():
    rev = pd.read_csv(config.REVISION_CSV, encoding="utf-8-sig", dtype={"revisado": str})
    n_rev = rev["revisado"].fillna("").str.strip().str.lower().isin(["si", "sí", "x", "1"]).sum()
    (ok if n_rev == len(rev) else aviso)(f"Revisión humana: {n_rev} de {len(rev)} filas marcadas como revisadas")

# 3. Dataset gold ---------------------------------------------------------
seccion("3. Dataset gold")
demo = False
if config.GOLD_CASOS.exists():
    gold = pd.read_parquet(config.GOLD_CASOS)
    demo = gold["caso_id"].astype(str).str.startswith("DEMO").any()
    if demo:
        falta("gold_casos contiene DATOS SINTÉTICOS de prueba: ejecuta 02_construir_gold.py con tus datos reales")
    else:
        (ok if len(gold) >= 200 else aviso)(f"gold_casos: {len(gold)} casos reales (objetivo: 250-400)")
    faltan_cols = [c for c in config.FEATURES if c not in gold]
    if faltan_cols:
        falta(f"gold_casos no tiene las columnas {faltan_cols}: vuelve a ejecutar 02_construir_gold.py")
    clases = gold["resultado"].value_counts()
    minima = clases.reindex(config.CLASES, fill_value=0).min()
    (ok if minima >= 20 else aviso)(f"Casos por clase: {clases.to_dict()} (conviene al menos 20 por clase)")
    años = gold["año_resolucion"]
    (ok if años.nunique() >= 4 else aviso)(f"Años cubiertos: {años.min()}-{años.max()} ({años.nunique()} distintos)")
else:
    falta("No existe gold_casos: python 02_construir_gold.py")

# 4. Modelo ---------------------------------------------------------------
seccion("4. Modelo")
if config.MODEL_PATH.exists() and config.METRICS_PATH.exists():
    import joblib
    m = config.leer_json(config.METRICS_PATH)
    modelo = joblib.load(config.MODEL_PATH)
    usadas = list(modelo["pre"].feature_names_in_)
    if set(usadas) != set(config.FEATURES):
        falta("El modelo se entrenó con otras variables (versión antigua): python 03_entrenar.py")
    if config.GOLD_CASOS.exists() and m.get("n_casos") != len(gold):
        falta(f"El modelo se entrenó con {m.get('n_casos')} casos y gold tiene {len(gold)}: python 03_entrenar.py")
    elegido = m["modelo_elegido"]
    f1, base = m["modelos"][elegido]["test_f1_macro"], m["modelos"]["baseline_clase_mayoritaria"]["test_f1_macro"]
    (ok if m["aceptado"] else aviso)(
        f"{elegido}: F1-macro test temporal {f1:.2f} vs baseline {base:.2f} -> "
        + ("ACEPTADO" if m["aceptado"] else "NO supera al baseline (la app mostrará solo precedentes)"))
else:
    falta("No hay modelo entrenado: python 03_entrenar.py")

# 5. Extras para la memoria -----------------------------------------------
seccion("5. Extras")
(ok if config.INDICE_PRECEDENTES.exists() else aviso)(
    "Índice semántico de precedentes " + ("creado" if config.INDICE_PRECEDENTES.exists()
                                          else "no creado (opcional): python 04_indexar.py"))
ev = config.MODELS_DIR / "evaluacion_extraccion.json"
(ok if ev.exists() else aviso)("Comparación LLM vs reglas " + ("hecha" if ev.exists()
                                                             else "sin hacer (recomendable): python 05_evaluar_extraccion.py"))

# 6. Seguridad --------------------------------------------------------------
seccion("6. Seguridad")
ejemplo = ROOT / ".env.ejemplo"
if ejemplo.exists() and any(l.split("=", 1)[1].strip() for l in ejemplo.read_text(encoding="utf-8").splitlines()
                            if "_API_KEY=" in l and not l.lstrip().startswith("#")):
    falta(".env.ejemplo contiene una clave: bórrala (ese archivo se sube a GitHub)")
else:
    ok(".env.ejemplo sin claves")
gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
(ok if ".env" in gi.split() else falta)(".env excluido de GitHub" if ".env" in gi.split() else ".env NO está en .gitignore")

# 7. La app arranca ---------------------------------------------------------
seccion("7. App")
if not config.GOLD_CASOS.exists() or pd.read_parquet(config.GOLD_CASOS).empty:
    aviso("App no probada: el dataset está vacío (se probará cuando haya casos)")
elif config.MODEL_PATH.exists():
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        if at.exception:
            falta(f"La app da error al abrirse: {at.exception[0].message[:200]}")
        elif not at.button:
            aviso("La app se abre, pero muestra un aviso en lugar del formulario (revisa dataset y modelo)")
        elif at.button[0].click().run().exception:
            falta(f"La app da error al calcular: {at.exception[0].message[:200]}")
        else:
            ok("La app se abre y calcula una estimación sin errores")
    except Exception as e:
        falta(f"No se pudo probar la app: {str(e)[:200]}")

# Informe -------------------------------------------------------------------
print("\n=========== VERIFICACIÓN DEL PROYECTO ===========")
for estado, msg in resultados:
    print(msg if not estado else f"  [{estado:5s}] {msg}")
n_falta = sum(e == "FALTA" for e, _ in resultados)
n_aviso = sum(e == "AVISO" for e, _ in resultados)
print("\n" + ("TODO LISTO PARA ENTREGAR." if not n_falta and not n_aviso else
              f"Pendiente: {n_falta} cosas imprescindibles (FALTA) y {n_aviso} recomendaciones (AVISO)."))
