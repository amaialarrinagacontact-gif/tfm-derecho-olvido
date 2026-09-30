"""Descarga resoluciones de la AEPD a partir de una lista de expedientes, respetando robots.txt.

1. Crea `expedientes.txt` con una línea por resolución. Vale cualquiera de estos formatos:
       TD/00247/2021
       td-00247-2021
       https://www.aepd.es/es/documento/td-00247-2021.pdf
   (Los códigos salen del buscador de resoluciones de la AEPD: filtra por
    "Tutela de derechos" y busca "supresión" u "olvido".)
2. python descargar_aepd.py --email tu@correo.com

Si robots.txt permite la descarga automática: baja los PDF con pausa entre peticiones.
Si no la permite: NO descarga nada y genera `pendientes.html` con los enlaces para
descargarlos a mano desde el navegador a data/raw/aepd_resoluciones/.
"""
import argparse
import re
import time
import urllib.robotparser
from pathlib import Path

import requests

from src import config

BASE = "https://www.aepd.es/es/documento/{}.pdf"


def a_url(linea: str) -> str | None:
    linea = linea.strip()
    if not linea or linea.startswith("#"):
        return None
    if linea.startswith("http"):
        return linea
    m = re.match(r"(?i)(td|pd|ps|reposicion-td)\s*[/-]\s*(\d{3,6})\s*[/-]\s*(\d{4})", linea)
    if m:
        return BASE.format(f"{m.group(1).lower()}-{int(m.group(2)):05d}-{m.group(3)}")
    return BASE.format(linea.lower())   # p. ej. un nombre de documento ya en formato web


def main(email: str, pausa: float):
    urls = [u for u in map(a_url, Path("expedientes.txt").read_text(encoding="utf-8").splitlines()) if u]
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pendientes = [u for u in urls if not (config.RAW_DIR / u.rsplit("/", 1)[-1]).exists()]
    print(f"{len(urls)} expedientes, {len(pendientes)} por descargar.")

    agente = f"TFM-DerechoOlvido/1.0 (uso academico; {email})"
    rp = urllib.robotparser.RobotFileParser("https://www.aepd.es/robots.txt")
    try:
        rp.read()
        permitido = all(rp.can_fetch(agente, u) for u in pendientes[:5])
    except Exception:
        permitido = False

    if not permitido:
        enlaces = "\n".join(f'<li><a href="{u}" download>{u.rsplit("/", 1)[-1]}</a></li>' for u in pendientes)
        Path("pendientes.html").write_text(
            f"<meta charset='utf-8'><h2>Resoluciones pendientes ({len(pendientes)})</h2>"
            f"<p>Guarda cada PDF en data/raw/aepd_resoluciones/</p><ol>{enlaces}</ol>", encoding="utf-8")
        print("robots.txt no permite la descarga automática (o no se pudo leer).")
        print("Abre pendientes.html en el navegador y descárgalos a mano.")
        return

    sesion = requests.Session()
    sesion.headers["User-Agent"] = agente
    ok = 0
    for i, u in enumerate(pendientes, 1):
        destino = config.RAW_DIR / u.rsplit("/", 1)[-1]
        try:
            r = sesion.get(u, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                destino.write_bytes(r.content)
                ok += 1
            else:
                print(f"  {destino.name}: no disponible (HTTP {r.status_code})")
        except requests.RequestException as e:
            print(f"  {destino.name}: error {e.__class__.__name__}")
        if i % 25 == 0:
            print(f"  {i}/{len(pendientes)}")
        time.sleep(pausa)
    print(f"Descargados {ok} PDF en {config.RAW_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True, help="Contacto que se envía en el User-Agent")
    p.add_argument("--pausa", type=float, default=3.0, help="Segundos entre descargas")
    a = p.parse_args()
    main(a.email, a.pausa)
