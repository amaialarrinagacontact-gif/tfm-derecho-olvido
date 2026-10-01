"""Extracción por REGLAS (expresiones regulares), sin ninguna API.

Dos usos:
1. Plan B: si no hay modelo de lenguaje disponible, pre-rellena revision.csv para
   que la revisión humana sea más rápida.
2. Línea base metodológica: 05_evaluar_extraccion.py compara su acierto con el del
   LLM frente a las etiquetas revisadas a mano. Así se cuantifica la tesis del
   proyecto: el LLM extrae mejor que un enfoque basado en reglas.

Mismas reglas que el LLM: solo lee HECHOS para las features y solo el FALLO para el resultado.
"""
import re

FLAGS = re.IGNORECASE

RESPONSABLE = {   # categoría -> términos; gana el que aparece ANTES en el texto (suele ser la reclamada)
    "motor_busqueda": r"google|bing|yahoo|microsoft|buscador|motor de b[úu]squeda",
    "red_social": r"facebook|twitter|\bx corp|instagram|youtube|tiktok|linkedin|red(?:es)? social",
    "hemeroteca": r"diario|peri[óo]dico|hemeroteca|medio de comunicaci[óo]n|editorial|prensa",
    "registro_publico": r"bolet[íi]n oficial|\bBOE\b|\bBOP\b|diario oficial|registro mercantil|registro de la propiedad",
}
CONTENIDO = [     # orden de prioridad
    ("dato_judicial", r"sentencia|condena|juzgado|tribunal|procedimiento penal|detenci[óo]n|delito|imputad"),
    ("dato_registral", r"bolet[íi]n oficial|\bBOE\b|edicto|subasta|indulto|nombramiento|embargo"),
    ("imagen", r"fotograf[íi]a|\bfotos?\b|imagen|im[áa]genes|v[íi]deo"),
    ("noticia", r"noticia|art[íi]culo (?:period[íi]stico|de prensa|publicado)|diario digital|peri[óo]dico|reportaje|\\bblog\\b"),
]
PUBLICA = (r"concejal|alcalde|diputad|senador|ministr|magistrad|\bjuez\b|jueza|fiscal|cargo p[úu]blico|"
           r"pol[íi]tic[oa]|presidente|consejer[oa] de|director general|alto cargo")
LIBERTAD_INFO = r"libertad de (?:informaci[óo]n|expresi[óo]n)|inter[ée]s p[úu]blico|relevancia p[úu]blica|derecho a la informaci[óo]n"
HISTORICO = r"inter[ée]s hist[óo]rico|investigaci[óo]n hist[óo]rica|fines de archivo|archivo hist[óo]rico"
CONDENA = r"condenad[oa]|condena firme|sentencia firme|fue condenad"
NO_CONDENA = r"absuel|archiv|sobrese[íi]|absoluci[óo]n"
AÑO = r"(?:19[89]\d|20[0-4]\d)"


def _primera_posicion(patron: str, texto: str) -> int:
    m = re.search(patron, texto, FLAGS)
    return m.start() if m else 10 ** 9


def _antiguedad(hechos: str):
    """Años entre publicación y solicitud; None si no se pueden leer ambas fechas (no se estima)."""
    solicitud = None
    # "Con fecha 3 de junio de 2019, D. A.A.A. ejerció..." (el año va antes del verbo; ojo con "D.")
    for m in re.finditer(r"(?:con fecha|el d[íi]a|en fecha)\s+[^,;]{0,40}?(" + AÑO + ")", hechos, FLAGS):
        if re.search(r"ejerci|solicit", hechos[m.end():m.end() + 300], FLAGS):
            solicitud = int(m.group(1))
            break
    m = re.search(r"publicad[oa]s?[^.]{0,60}?(" + AÑO + ")", hechos, FLAGS) or \
        re.search(r"(?:noticia|art[íi]culo|publicaci[óo]n|edicto)[^.]{0,40}?(?:de|del a[ñn]o|en)\s+(" + AÑO + ")",
                  hechos, FLAGS)
    publicacion = int(m.group(1)) if m else None
    if solicitud and publicacion and 0 <= solicitud - publicacion <= 60:
        return solicitud - publicacion
    return None


def _responsable(hechos: str) -> str:
    """Busca primero justo después de "frente a" / "contra" (la parte reclamada)."""
    zonas = [hechos[m.end():m.end() + 150] for m in re.finditer(r"frente a|contra\b", hechos[:3000], FLAGS)]
    for zona in [*zonas, hechos[:3000]]:
        pos = {c: _primera_posicion(p, zona) for c, p in RESPONSABLE.items()}
        if min(pos.values()) < 10 ** 9:
            return min(pos, key=pos.get)
    return "otro"


def extraer_features(hechos: str) -> dict:
    responsable = _responsable(hechos)
    contenido = next((c for c, p in CONTENIDO if re.search(p, hechos, FLAGS)), "otro")
    condena = int(any(re.search(CONDENA, f, FLAGS) and not re.search(NO_CONDENA, f, FLAGS)
                      for f in re.split(r"(?<=[.;])\s+", hechos)))
    return {
        "tipo_solicitante": "publica" if re.search(PUBLICA, hechos, FLAGS) else "privada",
        "tipo_responsable": responsable,
        "tipo_contenido": contenido,
        "antiguedad_info_años": _antiguedad(hechos),
        "condena_penal_previa": condena,
        "persona_fallecida": int(bool(re.search(r"fallecid", hechos, FLAGS))),
        "interes_historico": int(bool(re.search(HISTORICO, hechos, FLAGS))),
        "alega_libertad_informacion": int(bool(re.search(LIBERTAD_INFO, hechos, FLAGS))),
        "evidencia": "[reglas] extracción automática sin IA: revisar",
    }


SIN_OBJETO = (r"no existir objeto|carencia sobrevenida|p[ée]rdida sobrevenida|pretensiones .{0,60}satisfechas|"
              r"ha sido atendid[oa] extempor|atendi[óo] extempor")


def extraer_resultado(fallo: str, contexto: str = "") -> str:
    """Sentido del fallo. `contexto` = final de los fundamentos (para el TARGET sí se pueden leer:
    no es una feature, así que no hay fuga de información)."""
    inicio = fallo[:700]
    if re.search(SIN_OBJETO, contexto[-2000:] + inicio, FLAGS):
        return "excluir"                      # desestimada/estimada solo porque ya se había obtenido lo pedido
    if re.search(r"motivos formales", inicio, FLAGS):
        return "excluir"                      # estimación formal (falta de respuesta), no ordena suprimir
    if re.search(r"ESTIMAR\s+PARCIALMENTE", inicio):
        return "estimada_parcialmente"
    if re.search(r"DESESTIMAR", inicio):
        return "desestimada"
    if re.search(r"INADMITIR|ARCHIVAR|ARCHIVO|DECLARAR (?:CONCLUSO|TERMINADO)|DESISTIMIENTO", inicio, FLAGS):
        return "excluir"
    if re.search(r"ESTIMAR", inicio):
        return "estimada"
    return "excluir"
