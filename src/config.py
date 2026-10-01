"""Configuración central del proyecto: rutas, taxonomías y features.

Todas las categorías siguen el criterio de etiquetado de la Entrega 2 (sección 2.5).
Si cambias una categoría, cámbiala SOLO aquí.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw" / "aepd_resoluciones"
PROCESSED_DIR = DATA / "processed"
GOLD_DIR = DATA / "gold"
MODELS_DIR = ROOT / "models"

# Carga .env (GEMINI_API_KEY=...) sin dependencias extra
_env = ROOT / ".env"
if _env.exists():
    for _l in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

CASOS_EXTRAIDOS = PROCESSED_DIR / "casos_extraidos.parquet"
REVISION_CSV = PROCESSED_DIR / "revision.csv"          # se edita a mano (Excel / LibreOffice)
GOLD_CASOS = GOLD_DIR / "gold_casos.parquet"
INDICE_PRECEDENTES = GOLD_DIR / "indice_precedentes.npz"   # embeddings locales de los hechos
EMBEDDINGS_LOCAL = os.getenv("EMBEDDINGS_LOCAL", "intfloat/multilingual-e5-small")  # se ejecuta en tu PC
MODEL_PATH = MODELS_DIR / "modelo.joblib"
METRICS_PATH = MODELS_DIR / "metricas.json"

# --- Modelo de lenguaje: SOLO para el paso 1 (extracción). La app no lo usa. ---
# Pon en .env la clave de UNO de estos proveedores (y, si quieres, LLM_PROVIDER):
#   MISTRAL_API_KEY=...   (europeo, gratis en console.mistral.ai)
#   GROQ_API_KEY=...      (gratis en console.groq.com)
#   GEMINI_API_KEY=...    (Google AI Studio)
#   LLM_PROVIDER=ollama   (local, gratis, sin cuenta; requiere instalar Ollama)
# Sin ninguno: python 01_extraer.py --sin-llm  (extracción por reglas)
PROVEEDORES = {
    "mistral": {"url": "https://api.mistral.ai/v1", "clave": "MISTRAL_API_KEY",
                "modelo": "mistral-small-latest"},
    "groq": {"url": "https://api.groq.com/openai/v1", "clave": "GROQ_API_KEY",
             "modelo": "llama-3.3-70b-versatile"},
    "gemini": {"url": None, "clave": "GEMINI_API_KEY",
               "modelo": "gemini-2.5-flash"},
    "ollama": {"url": "http://localhost:11434/v1", "clave": None,
               "modelo": "qwen2.5:7b"},
}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower() or next(
    (n for n, p in PROVEEDORES.items() if p["clave"] and os.getenv(p["clave"])), "")
_P = PROVEEDORES.get(LLM_PROVIDER, {})
LLM_API_KEY = os.getenv(_P["clave"], "") if _P.get("clave") else ("ollama" if LLM_PROVIDER == "ollama" else "")
LLM_MODEL = os.getenv("LLM_MODEL", _P.get("modelo", ""))
LLM_DISPONIBLE = bool(LLM_PROVIDER and LLM_API_KEY)
PAUSA_ENTRE_LLAMADAS = float(os.getenv("PAUSA_ENTRE_LLAMADAS", "2"))  # segundos

# --- Taxonomías (Entrega 2, sección 2.5) ---
CLASES = ["estimada", "estimada_parcialmente", "desestimada"]
TIPO_SOLICITANTE = ["privada", "publica"]
TIPO_RESPONSABLE = ["motor_busqueda", "hemeroteca", "red_social", "registro_publico", "otro"]
TIPO_CONTENIDO = ["noticia", "imagen", "dato_judicial", "dato_registral", "otro"]

# --- Features del modelo (Entrega 4, sección 4) ---
# NO se incluyen: año_resolucion (leakage), jurisdiccion (constante), textos (van al RAG).
FEATURES_CAT = ["tipo_solicitante", "tipo_responsable", "tipo_contenido"]
FEATURES_BIN = ["condena_penal_previa", "persona_fallecida", "interes_historico", "alega_libertad_informacion"]
FEATURES_NUM = ["antiguedad_info_años"]
FEATURES = FEATURES_CAT + FEATURES_BIN + FEATURES_NUM
TARGET = "resultado"
COLUMNA_TEMPORAL = "año_resolucion"   # solo para la partición train/test, nunca como feature

# --- Evaluación (Entrega 4, secciones 6 y 7) ---
TEST_FRACCION = 0.25          # último 25% del corpus por fecha = test temporal
MARGEN_ACEPTACION = 0.05      # F1-macro modelo - F1-macro baseline mínimo para mostrar predicción
MARGEN_EMPATE = 0.02          # si otro modelo gana por menos que esto, se prefiere la regresión logística
SEED = 42

# --- Textos legibles para la interfaz ---
ETIQUETAS = {
    "estimada": "Estimada",
    "estimada_parcialmente": "Estimada parcialmente",
    "desestimada": "Desestimada",
    "privada": "Persona privada",
    "publica": "Persona con proyección pública",
    "motor_busqueda": "Motor de búsqueda",
    "hemeroteca": "Medio de comunicación / hemeroteca",
    "red_social": "Red social",
    "registro_publico": "Registro o boletín público",
    "otro": "Otro",
    "noticia": "Noticia",
    "imagen": "Imagen o vídeo",
    "dato_judicial": "Dato judicial",
    "dato_registral": "Dato registral",
    "tipo_solicitante": "Tipo de solicitante",
    "tipo_responsable": "Responsable del tratamiento",
    "tipo_contenido": "Tipo de contenido",
    "condena_penal_previa": "Condena penal firme previa",
    "persona_fallecida": "Datos de persona fallecida",
    "interes_historico": "Interés histórico alegado",
    "alega_libertad_informacion": "Libertad de información alegada",
    "antiguedad_info_años": "Antigüedad de la información",
}

# --- Fundamento jurídico de cada variable (se muestra junto a los factores SHAP) ---
MARCO_POR_VARIABLE = {
    "tipo_solicitante": "Relevancia pública de la persona, criterio central de la ponderación con la libertad "
                        "de información (art. 20.1.d CE, art. 11 CDFUE, art. 10 CEDH; STJUE Google Spain, C-131/12).",
    "tipo_responsable": "El buscador responde de forma autónoma respecto de la fuente (C-131/12; art. 93 LOPDGDD). "
                        "Frente a medios y hemerotecas la libertad de información pesa más (STC 58/2018; TEDH, Hurbain c. Bélgica).",
    "antiguedad_info_años": "Con el paso del tiempo los datos pueden dejar de ser adecuados y pertinentes "
                            "(art. 5.1.c y e RGPD; art. 93 LOPDGDD; STC 58/2018).",
    "tipo_contenido": "Los datos penales y de categorías especiales tienen protección reforzada "
                      "(arts. 9 y 10 RGPD; STJUE GC y otros, C-136/17).",
    "condena_penal_previa": "Datos relativos a condenas penales (art. 10 RGPD; C-136/17); se valoran junto al tiempo transcurrido.",
    "persona_fallecida": "El RGPD no se aplica a personas fallecidas (considerando 27); en España, familiares y "
                         "herederos pueden pedir la supresión (art. 3 LOPDGDD).",
    "interes_historico": "Excepción por fines de archivo, investigación histórica o estadísticos (art. 17.3.d RGPD).",
    "alega_libertad_informacion": "Excepción por ejercicio de la libertad de expresión e información "
                                  "(art. 17.3.a RGPD; art. 20 CE; art. 11 CDFUE; art. 10 CEDH).",
}

MARCO_GENERAL = [
    ("Constitución Española", "Art. 18.1 (honor e intimidad), art. 18.4 (protección de datos) y art. 20.1.a y d "
                              "(libertad de expresión e información). STC 58/2018 sobre hemerotecas digitales."),
    ("Unión Europea", "Carta de Derechos Fundamentales, arts. 7, 8 y 11. RGPD, art. 17 y sus excepciones del 17.3. "
                      "TJUE: Google Spain (C-131/12), GC y otros (C-136/17) y TU y RE (C-460/20, sobre información inexacta)."),
    ("Consejo de Europa", "Convenio Europeo de Derechos Humanos, arts. 8 (vida privada) y 10 (libertad de expresión). "
                          "TEDH: M.L. y W.W. c. Alemania y Hurbain c. Bélgica."),
    ("Legislación española", "LOPDGDD (Ley Orgánica 3/2018): art. 93, derecho al olvido en búsquedas de Internet, "
                             "y art. 94, en redes sociales."),
    ("Criterios de las autoridades de control", "Grupo del Artículo 29, Directrices WP225 (2014): 13 criterios comunes para "
                                                "resolver reclamaciones de desindexación. CEPD, Directrices 5/2019 sobre el derecho "
                                                "al olvido en buscadores."),
]


def leer_json(ruta):
    """Lee un JSON en UTF-8; si se escribió con la codificación de Windows (versiones antiguas), también."""
    import json
    datos = ruta.read_bytes()
    try:
        return json.loads(datos.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(datos.decode("cp1252"))
