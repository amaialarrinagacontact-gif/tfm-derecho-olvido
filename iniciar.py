"""LANZADOR DEL PROYECTO. En VS Code: abre la carpeta y pulsa F5 (o ▶ en este archivo).

La primera vez crea un entorno virtual (.venv) e instala todo solo (tarda unos minutos).
Después muestra un menú con todos los pasos.
"""
import os
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY_VENV = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV_FILE = ROOT / ".env"


# ---------------------------------------------------------------- entorno
def asegurar_entorno():
    if sys.version_info < (3, 10):
        sys.exit(f"Necesitas Python 3.10 o superior (tienes {sys.version.split()[0]}). Descárgalo en python.org")
    dentro_de_venv = sys.prefix != sys.base_prefix
    if not dentro_de_venv and not os.environ.get("TFM_SIN_VENV"):
        if not PY_VENV.exists():
            print("Creando entorno virtual en .venv (solo la primera vez)...")
            venv.create(VENV, with_pip=True)
        sys.exit(subprocess.call([str(PY_VENV), str(Path(__file__).resolve())], cwd=ROOT))

    marca = Path(sys.prefix) / ".tfm_instalado"
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if not marca.exists() or marca.read_text(encoding="utf-8") != req:
        print("Instalando dependencias (solo la primera vez, puede tardar 3-5 minutos)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], cwd=ROOT)
        marca.write_text(req, encoding="utf-8")
        print("Dependencias instaladas.\n")


# ---------------------------------------------------------------- utilidades
def ejecutar(*script) -> bool:
    print(f"\n>>> {' '.join(script)}\n")
    return subprocess.call([sys.executable, *script], cwd=ROOT) == 0


def pedir_api_key() -> bool:
    from src import config
    if config.LLM_DISPONIBLE:
        print(f"Modelo de lenguaje: {config.LLM_PROVIDER} ({config.LLM_MODEL})")
        return True
    print("\nEste paso necesita un modelo de lenguaje. Elige proveedor (todos tienen opción gratuita):")
    print("  1. Mistral  (europeo; clave en https://console.mistral.ai/api-keys)")
    print("  2. Groq     (clave en https://console.groq.com/keys)")
    print("  3. Gemini   (clave en https://aistudio.google.com/apikey)")
    opcion = input("Opción (o Enter para cancelar): ").strip()
    nombre = {"1": "mistral", "2": "groq", "3": "gemini"}.get(opcion)
    if not nombre:
        return False
    variable = config.PROVEEDORES[nombre]["clave"]
    clave = input(f"Pega tu clave de {nombre}: ").strip()
    if not clave:
        return False
    ENV_FILE.write_text(f"LLM_PROVIDER={nombre}\n{variable}={clave}\n", encoding="utf-8")
    print("Guardada en .env. Vuelve a elegir la opción para empezar.")
    sys.exit(subprocess.call([sys.executable, str(Path(__file__).resolve())], cwd=ROOT))


def hay_datos_reales() -> bool:
    from src import config
    if not config.GOLD_CASOS.exists():
        return False
    import pandas as pd
    return not pd.read_parquet(config.GOLD_CASOS, columns=["caso_id"])["caso_id"].astype(str).str.startswith("DEMO").all()


def abrir_app():
    from src import config
    if not (config.GOLD_CASOS.exists() and config.MODEL_PATH.exists()):
        print("\nTodavía no hay dataset o modelo. Usa la opción 1 (prueba) o las opciones 3 y 4 (datos reales).")
        return
    print("\nAbriendo la app en http://localhost:8501  (para cerrarla: Ctrl+C en esta terminal)")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT)
    time.sleep(4)
    webbrowser.open("http://localhost:8501")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nApp cerrada.")


# ---------------------------------------------------------------- opciones
def op_demo():
    if hay_datos_reales() and input("Ya tienes datos reales; se sobrescribirán. ¿Seguir? (s/n): ").lower() != "s":
        return
    if ejecutar("00_datos_demo.py") and ejecutar("03_entrenar.py"):
        abrir_app()


def op_descargar():
    if not (ROOT / "expedientes.txt").exists():
        (ROOT / "expedientes.txt").write_text("# Un expediente por línea, por ejemplo:\n# TD/00247/2021\n", encoding="utf-8")
        print("\nHe creado expedientes.txt. Pon un código de expediente por línea y vuelve a elegir esta opción.")
        return
    email = input("Tu email (se envía como contacto al servidor): ").strip()
    if email:
        ejecutar("descargar_aepd.py", "--email", email)


def op_extraer():
    from src import config
    n = len(list(config.RAW_DIR.glob("*.pdf")))
    if n == 0:
        print(f"\nNo hay PDF. Cópialos en:\n  {config.RAW_DIR}")
        if os.name == "nt":
            os.startfile(config.RAW_DIR)
        return
    print(f"\n{n} PDF encontrados. Primero compruebo que se separan bien las secciones (sin gastar API)...")
    if not ejecutar("01_extraer.py", "--sin-llm"):
        return
    from src import config as c
    print("\nYa tienes las variables extraídas por reglas (sin API).")
    if input("¿Mejorarlas con un modelo de lenguaje gratuito (Mistral/Groq/Gemini)? (s/n): ").lower() == "s" \
            and pedir_api_key():
        ejecutar("01_extraer.py")
        print(f"\nAhora revisa y corrige {c.REVISION_CSV} (Excel), pon 'si' en 'revisado' y usa la opción 4.")
        if os.name == "nt":
            os.startfile(c.REVISION_CSV)


def op_entrenar():
    if ejecutar("02_construir_gold.py") and ejecutar("03_entrenar.py"):
        print("\nMétricas guardadas en models/metricas.json")


def op_indexar():
    print("\nSe instala un modelo de embeddings local (gratis, sin API). La primera vez descarga ~1 GB.")
    subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements-rag.txt", "-q"], cwd=ROOT)
    ejecutar("04_indexar.py")


MENU = [
    ("Ver la app con datos de prueba (empieza por aquí)", op_demo),
    ("Descargar PDF de la AEPD desde expedientes.txt", op_descargar),
    ("Procesar mis PDF (paso 1: reglas y, si quieres, modelo de lenguaje)", op_extraer),
    ("Construir dataset y entrenar el modelo (pasos 2 y 3)", op_entrenar),
    ("Activar búsqueda semántica de precedentes (opcional, local, sin API)", op_indexar),
    ("Comparar extracción LLM vs reglas (para la memoria)", lambda: ejecutar("05_evaluar_extraccion.py")),
    ("Abrir la app", abrir_app),
]


def main():
    asegurar_entorno()
    while True:
        print("\n=== TFM Derecho al olvido ===")
        for i, (texto, _) in enumerate(MENU, 1):
            print(f"  {i}. {texto}")
        print("  0. Salir")
        eleccion = input("Elige una opción: ").strip()
        if eleccion == "0":
            break
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(MENU):
            try:
                MENU[int(eleccion) - 1][1]()
            except KeyboardInterrupt:
                print("\nCancelado.")


if __name__ == "__main__":
    main()
