"""Lectura de PDFs de la AEPD y separación en secciones.

Las resoluciones de la AEPD siguen una estructura fija:
    ANTECEDENTES (o HECHOS) -> FUNDAMENTOS DE DERECHO -> RESUELVE

Se corta con reglas (no con el LLM) para garantizar de forma verificable que la
extracción de features solo ve los HECHOS (corrección de leakage, Entrega 4).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

RE_ANTECEDENTES = re.compile(r"^\s*(ANTECEDENTES(?: DE HECHO)?|HECHOS)\s*:?\s*$", re.MULTILINE)
RE_FUNDAMENTOS = re.compile(r"^\s*FUNDAMENTOS\s+DE\s+DERECHO\s*:?\s*$", re.MULTILINE)
RE_RESUELVE = re.compile(r"\bRESUELVE\b\s*:?")          # mayúsculas, puede ir a mitad de línea
RE_EXP_ANTIGUO = re.compile(r"\b(TD|PD|PS|E|AT)\s*/\s*\d{3,6}\s*/\s*(\d{4})")
RE_EXP_NUEVO = re.compile(r"\bEXP(20\d{2})\d{3,}")


@dataclass
class Resolucion:
    fichero: str
    expediente: str | None
    año: int | None
    texto_hechos: str
    texto_fundamentos: str
    texto_fallo: str
    secciones_ok: bool       # False si no se encontraron los encabezados: revisar a mano


def leer_pdf(ruta: Path) -> str:
    with pdfplumber.open(ruta) as pdf:
        paginas = [p.extract_text() or "" for p in pdf.pages]
    texto = "\n".join(paginas)
    # Quita líneas típicas de cabecera/pie repetidas (URL, "Página x de y")
    texto = re.sub(r"^.*(www\.aepd\.es|sedeagpd\.gob\.es|P[áa]gina \d+ de \d+).*$", "", texto,
                   flags=re.MULTILINE)
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def _expediente(texto: str) -> tuple[str | None, int | None]:
    m = RE_EXP_NUEVO.search(texto)
    if m:
        return m.group(0), int(m.group(1))
    m = RE_EXP_ANTIGUO.search(texto)
    if m:
        return re.sub(r"\s+", "", m.group(0)), int(m.group(2))
    return None, None


def separar(texto: str, fichero: str = "") -> Resolucion:
    expediente, año = _expediente(texto)
    m_fund = RE_FUNDAMENTOS.search(texto)
    m_ant = RE_ANTECEDENTES.search(texto)

    if not m_fund:
        # Sin frontera clara NO se envía nada al LLM como "hechos": podría incluir el fallo.
        return Resolucion(fichero, expediente, año, "", "", "", False)

    inicio_hechos = m_ant.end() if (m_ant and m_ant.start() < m_fund.start()) else 0
    hechos = texto[inicio_hechos:m_fund.start()].strip()

    resto = texto[m_fund.end():]
    resuelves = list(RE_RESUELVE.finditer(resto))
    if resuelves:
        m_res = resuelves[-1]
        fundamentos = resto[:m_res.start()].strip()
        fallo = resto[m_res.end():].strip()
    else:
        fundamentos, fallo = resto.strip(), ""

    ok = bool(hechos) and bool(fallo)
    return Resolucion(fichero, expediente, año, hechos, fundamentos, fallo[:4000], ok)


def procesar_pdf(ruta: Path) -> Resolucion:
    return separar(leer_pdf(ruta), ruta.name)
