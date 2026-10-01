"""Llamadas al modelo de lenguaje: extracción de features (solo HECHOS), propuesta de
resultado (solo FALLO). Funciona con Mistral, Groq, Gemini u Ollama
(ver src/config.py). Las salidas se validan contra esquemas cerrados.
"""
import json
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src import config


class FeaturesCaso(BaseModel):
    tipo_solicitante: Literal["privada", "publica"] = Field(
        description="'publica' SOLO si el texto identifica al solicitante como cargo electo, alto cargo "
                    "o figura con proyección pública reconocida. En cualquier otro caso o ante la duda: 'privada'.")
    tipo_responsable: Literal["motor_busqueda", "hemeroteca", "red_social", "registro_publico", "otro"] = Field(
        description="Quién es el responsable del tratamiento reclamado en ESTE expediente (no el origen de la información). "
                    "Medios de comunicación y sus archivos = 'hemeroteca'. BOE/boletines/registros = 'registro_publico'.")
    tipo_contenido: Literal["noticia", "imagen", "dato_judicial", "dato_registral", "otro"]
    antiguedad_info_años: Optional[int] = Field(
        description="Años completos entre la publicación original y la solicitud de supresión. "
                    "null si el texto NO permite reconstruir ambas fechas con precisión de año. No estimes.")
    condena_penal_previa: Literal[0, 1] = Field(
        description="1 SOLO si se menciona una condena penal FIRME relacionada. Archivo, absolución o investigación = 0.")
    persona_fallecida: Literal[0, 1]
    interes_historico: Literal[0, 1] = Field(
        description="1 SOLO si el responsable invoca expresamente interés histórico, científico o estadístico.")
    alega_libertad_informacion: Literal[0, 1] = Field(
        description="1 SOLO si el responsable (p. ej. Google o el medio) rechaza o se opone a la supresión invocando "
                    "expresamente la libertad de información o de expresión, o el interés público de la información.")
    evidencia: str = Field(description="Máximo 40 palabras citando las frases del texto en que te basas, para la revisión humana.")


class ResultadoCaso(BaseModel):
    resultado: Literal["estimada", "desestimada", "estimada_parcialmente", "excluir"] = Field(
        description="'estimada': se ordena supresión/bloqueo/desindexación TOTAL. 'desestimada': se deniega íntegramente. "
                    "'estimada_parcialmente': se concede solo parte, para algunos canales o con condiciones. "
                    "'excluir': inadmisión, archivo, desistimiento, estimación solo por motivos formales (falta de respuesta "
                    "en plazo) sin ordenar la supresión, desestimación porque lo pedido ya se había obtenido (sin objeto), o si el fallo no permite decidirlo sin ambigüedad.")


PROMPT_FEATURES = """Eres un asistente de etiquetado jurídico. A continuación tienes ÚNICAMENTE la sección de
ANTECEDENTES/HECHOS de una resolución de la AEPD sobre derecho de supresión (art. 17 RGPD).
Extrae las variables siguiendo estrictamente las definiciones del esquema. No infieras lo que el texto no dice.

HECHOS:
\"\"\"{hechos}\"\"\""""

PROMPT_RESULTADO = """Lee el final de los fundamentos y la parte dispositiva (fallo) de una resolución de la AEPD
sobre derecho de supresión y clasifica su sentido según lo que se RESUELVE, teniendo en cuenta el motivo.

FINAL DE LOS FUNDAMENTOS:
\"\"\"{contexto}\"\"\"

FALLO:
\"\"\"{fallo}\"\"\""""


def _comprobar():
    if not config.LLM_DISPONIBLE:
        raise RuntimeError("No hay modelo de lenguaje configurado. Pon en el archivo .env la clave de un "
                           "proveedor, por ejemplo MISTRAL_API_KEY=... (ver src/config.py).")


def _cliente_openai():
    from openai import OpenAI
    return OpenAI(base_url=config.PROVEEDORES[config.LLM_PROVIDER]["url"], api_key=config.LLM_API_KEY)


def _normalizar(datos: dict) -> dict:
    """Corrige tipos que algunos modelos devuelven como texto ("1", "null", true...)."""
    for k, v in list(datos.items()):
        if isinstance(v, bool):
            datos[k] = int(v)
        elif isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            datos[k] = None if k != "evidencia" else ""
        elif isinstance(v, str) and v.strip() in ("0", "1") and k != "evidencia":
            datos[k] = int(v.strip())
        elif isinstance(v, float) and v.is_integer():
            datos[k] = int(v)
        elif isinstance(v, str) and k not in ("evidencia",):
            datos[k] = v.strip().lower()
    return datos


def _llamar_gemini(prompt: str, esquema):
    from google import genai
    from google.genai import types
    r = genai.Client(api_key=config.LLM_API_KEY).models.generate_content(
        model=config.LLM_MODEL, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=esquema, temperature=0))
    return r.text


def _llamar_openai(prompt: str, esquema):
    instrucciones = ("Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, que cumpla este "
                     "JSON Schema (usa exactamente los valores permitidos en cada 'enum'):\n"
                     + json.dumps(esquema.model_json_schema(), ensure_ascii=False))
    r = _cliente_openai().chat.completions.create(
        model=config.LLM_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": instrucciones}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content


class CuotaAgotada(RuntimeError):
    """El proveedor sigue rechazando por límite de peticiones tras varios intentos."""


_ultima_llamada = [0.0]


def _estructurado(prompt: str, esquema, reintentos: int = 5):
    _comprobar()
    llamar = _llamar_gemini if config.LLM_PROVIDER == "gemini" else _llamar_openai
    for intento in range(reintentos):
        # Ritmo suave: como máximo una petición cada PAUSA segundos (planes gratuitos)
        espera = config.PAUSA_ENTRE_LLAMADAS - (time.time() - _ultima_llamada[0])
        if espera > 0:
            time.sleep(espera)
        _ultima_llamada[0] = time.time()
        try:
            texto = llamar(prompt, esquema).strip().removeprefix("```json").removesuffix("```")
            return esquema.model_validate(_normalizar(json.loads(texto)))
        except Exception as e:
            limite = "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower()
            if intento == reintentos - 1:
                if limite:
                    raise CuotaAgotada(str(e)[:200]) from e
                raise
            pausa = (30 if limite else 5) * (2 ** intento)       # 30, 60, 120, 240 s si es límite
            motivo = "límite de peticiones" if limite else f"{e.__class__.__name__}: {str(e)[:100]}"
            print(f"  esperando {pausa}s ({motivo})")
            time.sleep(pausa)


def extraer_features(hechos: str) -> FeaturesCaso:
    return _estructurado(PROMPT_FEATURES.format(hechos=hechos[:30000]), FeaturesCaso)


def extraer_resultado(fallo: str, contexto: str = "") -> ResultadoCaso:
    return _estructurado(PROMPT_RESULTADO.format(fallo=fallo, contexto=contexto[-2000:]), ResultadoCaso)
