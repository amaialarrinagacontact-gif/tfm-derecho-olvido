"""Interfaz del MVP (Entrega 5). Ejecutar con:  streamlit run app.py"""
import html
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.modelo import Explicador
from src.precedentes import buscar, cobertura

E = config.ETIQUETAS
COLOR = {"estimada": "#2E6B4F", "estimada_parcialmente": "#A8741A", "desestimada": "#8E3B3B"}

st.set_page_config(page_title="Derecho al olvido: estimación orientativa", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Public+Sans:wght@400;500;600&display=swap');
.stApp { font-family: 'Public Sans', system-ui, sans-serif; }
h1, h2, h3, .serif { font-family: 'Source Serif 4', Georgia, serif !important; color: #1C2B3A; letter-spacing: -0.01em; }
h1 { font-weight: 600; font-size: 2.1rem !important; line-height: 1.15; }
.block-container { max-width: 1180px; padding-top: 2.2rem; }
.sub { color: #5E6A78; font-size: 1.02rem; max-width: 62ch; margin-top: -0.4rem; }
.aviso { border-left: 3px solid #1C2B3A; background: #EEF1F4; padding: 0.8rem 1rem; margin: 1.1rem 0 1.6rem;
         font-size: 0.93rem; color: #1C2B3A; max-width: 78ch; }
.demo { border-left: 3px solid #A8741A; background: #F7EFE0; padding: 0.6rem 1rem; font-size: 0.9rem; margin-bottom: 1rem; }
.veredicto { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.55rem; color: #1C2B3A; margin: 0.2rem 0 0.9rem; }
.reparto { display: flex; height: 38px; width: 100%; overflow: hidden; border-radius: 2px; }
.reparto div { height: 100%; }
.leyenda { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.55rem; }
.leyenda .pct { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; color: #1C2B3A; }
.leyenda .nom { font-size: 0.86rem; color: #5E6A78; }
.marca { display: inline-block; width: 10px; height: 10px; border-radius: 1px; margin-right: 6px; }
.factor { display: grid; grid-template-columns: 1fr 90px; gap: 0.8rem; padding: 0.5rem 0; border-bottom: 1px solid #D8DDE3; }
.factor .txt { font-size: 0.93rem; color: #1C2B3A; }
.factor .dir { font-size: 0.82rem; color: #5E6A78; }
.factor .ley { font-size: 0.8rem; color: #5E6A78; font-style: italic; margin-top: 0.15rem; }
.marco { font-size: 0.9rem; color: #1C2B3A; line-height: 1.5; }
.marco b { font-weight: 600; }
.factor .barra { align-self: center; height: 6px; background: #1C2B3A; }
.nota { font-size: 0.85rem; color: #5E6A78; margin-top: 0.8rem; }
.prec { padding: 0.8rem 0; border-bottom: 1px solid #D8DDE3; }
.prec .cab { display: flex; gap: 0.8rem; align-items: baseline; flex-wrap: wrap; }
.prec .id { font-weight: 600; color: #1C2B3A; }
.prec .año { color: #5E6A78; font-size: 0.88rem; }
.chip { font-size: 0.8rem; color: #fff; padding: 0.1rem 0.5rem; border-radius: 2px; }
.prec .hechos { color: #3A4654; font-size: 0.9rem; margin-top: 0.3rem; max-width: 90ch; line-height: 1.5; }
.vacio { color: #5E6A78; border: 1px dashed #C4CBD3; padding: 2rem 1.4rem; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar():
    if not config.GOLD_CASOS.exists() or pd.read_parquet(config.GOLD_CASOS).empty:
        return None, None, {}, None
    gold = pd.read_parquet(config.GOLD_CASOS)
    for c in config.FEATURES:                       # datasets de versiones anteriores
        if c not in gold:
            gold[c] = 0
    modelo = joblib.load(config.MODEL_PATH) if config.MODEL_PATH.exists() else None
    metricas = config.leer_json(config.METRICS_PATH) if config.METRICS_PATH.exists() else {}
    if modelo is not None and set(modelo["pre"].feature_names_in_) != set(config.FEATURES):
        modelo, metricas = None, {**metricas, "aceptado": False, "desactualizado": True}
    expl = Explicador(modelo, gold[config.FEATURES].sample(min(100, len(gold)), random_state=config.SEED)) if modelo else None
    return gold, modelo, metricas, expl


def valor_legible(var, v):
    if var == "antiguedad_info_años":
        return "desconocida" if pd.isna(v) else f"{int(v)} años"
    if var in config.FEATURES_BIN:
        return "Sí" if int(v) else "No"
    return E.get(v, v)


def chip(clase):
    return f'<span class="chip" style="background:{COLOR[clase]}">{E[clase]}</span>'


gold, modelo, metricas, explicador = cargar()
if gold is None:
    st.title("¿Prosperaría una reclamación de derecho al olvido ante la AEPD?")
    st.warning("Todavía no hay casos en el dataset. Revisa data/processed/revision.csv y ejecuta "
               "02_construir_gold.py y 03_entrenar.py. Para ver la app con datos de prueba: "
               "python 00_datos_demo.py y después python 03_entrenar.py.")
    st.stop()
if metricas.get("desactualizado"):
    st.info("El modelo guardado es de una versión anterior. Ejecuta python 03_entrenar.py para actualizarlo.")

st.title("¿Prosperaría una reclamación de derecho al olvido ante la AEPD?")
st.markdown('<p class="sub">Describe el caso y verás cómo se resolvieron reclamaciones parecidas '
            'y qué resultado sugiere el histórico de resoluciones.</p>', unsafe_allow_html=True)
st.markdown('<div class="aviso"><strong>Herramienta orientativa, no asesoramiento jurídico.</strong> '
            'La estimación se calcula por semejanza con resoluciones pasadas de la AEPD y no sustituye la '
            'ponderación individual de derechos que exige cada caso. Consulta con un profesional antes de decidir.</div>',
            unsafe_allow_html=True)
if gold["caso_id"].astype(str).str.startswith("DEMO").any():
    st.markdown('<div class="demo">Se están usando datos sintéticos de prueba. Las cifras no reflejan resoluciones reales.</div>',
                unsafe_allow_html=True)

izq, der = st.columns([5, 7], gap="large")

with izq:
    st.subheader("Tu caso")
    with st.form("caso"):
        sol = st.radio("¿Quién pide la supresión?", config.TIPO_SOLICITANTE, format_func=E.get, horizontal=True)
        resp = st.selectbox("¿Dónde está publicada la información?", config.TIPO_RESPONSABLE, format_func=E.get)
        cont = st.selectbox("¿Qué tipo de contenido es?", config.TIPO_CONTENIDO, format_func=E.get)
        no_se = st.checkbox("No sé cuándo se publicó")
        ant = st.number_input("Años desde que se publicó", min_value=0, max_value=60, value=5, disabled=no_se)
        cond = st.radio("¿Está relacionado con una condena penal firme?", [0, 1],
                        format_func=lambda v: "Sí" if v else "No", horizontal=True)
        libinf = st.checkbox("Al pedir la supresión, el responsable la rechazó alegando libertad de información "
                             "o interés público")
        with st.expander("Datos opcionales"):
            fall = st.checkbox("La información se refiere a una persona fallecida")
            hist = st.checkbox("El responsable alega interés histórico o de investigación")
        enviar = st.form_submit_button("Calcular estimación", type="primary", use_container_width=True)

caso = {"tipo_solicitante": sol, "tipo_responsable": resp, "tipo_contenido": cont,
        "condena_penal_previa": int(cond), "persona_fallecida": int(fall), "interes_historico": int(hist),
        "alega_libertad_informacion": int(libinf),
        "antiguedad_info_años": np.nan if no_se else float(ant)}

with der:
    if not enviar:
        st.markdown('<div class="vacio">Completa el formulario y pulsa <strong>Calcular estimación</strong>. '
                    'Puedes cambiar cualquier dato y volver a calcular para ver cómo varía el resultado.</div>',
                    unsafe_allow_html=True)
    else:
        x = pd.DataFrame([caso])[config.FEATURES]
        n_sim = cobertura(caso, gold)

        if modelo is None or not metricas.get("aceptado", False):
            st.subheader("Estimación no disponible")
            st.markdown("Con los casos etiquetados hasta ahora, el modelo no supera de forma clara a una referencia "
                        "trivial, así que no se muestra una predicción. Revisa los precedentes de abajo.")
        else:
            proba = dict(zip(modelo.classes_, modelo.predict_proba(x)[0]))
            ganadora = max(proba, key=proba.get)
            st.subheader("Estimación orientativa")
            st.markdown(f'<div class="veredicto">Resultado más probable: {E[ganadora].lower()}</div>',
                        unsafe_allow_html=True)
            barras = "".join(f'<div style="width:{proba[c]*100:.1f}%;background:{COLOR[c]}" '
                             f'title="{E[c]} {proba[c]:.0%}"></div>' for c in config.CLASES)
            leyenda = "".join(f'<div><div class="pct">{proba[c]:.0%}</div><div class="nom">'
                              f'<span class="marca" style="background:{COLOR[c]}"></span>{E[c]}</div></div>'
                              for c in config.CLASES)
            st.markdown(f'<div class="reparto" role="img" aria-label="Reparto de probabilidades">{barras}</div>'
                        f'<div class="leyenda">{leyenda}</div>', unsafe_allow_html=True)

            factores = [f for f in explicador.explicar(x, ganadora) if abs(f["contribucion"]) > 1e-6][:4]
            if factores:
                st.markdown("#### Qué pesa más en esta estimación")
                tope = max(abs(f["contribucion"]) for f in factores)
                filas = ""
                for f in factores:
                    sentido = "Acerca a" if f["contribucion"] > 0 else "Aleja de"
                    filas += (f'<div class="factor"><div><div class="txt">{E[f["variable"]]}: '
                              f'{html.escape(str(valor_legible(f["variable"], f["valor"])))}</div>'
                              f'<div class="dir">{sentido} «{E[ganadora].lower()}»</div>'
                              f'<div class="ley">{html.escape(config.MARCO_POR_VARIABLE[f["variable"]])}</div></div>'
                              f'<div class="barra" style="width:{abs(f["contribucion"])/tope*100:.0f}%"></div></div>')
                st.markdown(filas, unsafe_allow_html=True)

        cautela = " Tómala con especial cautela." if n_sim < 10 else ""
        st.markdown(f'<p class="nota">El corpus tiene {n_sim} resoluciones con el mismo tipo de solicitante, '
                    f'responsable y contenido.{cautela}</p>', unsafe_allow_html=True)
        if metricas.get("modelos") and not metricas.get("desactualizado"):
            m = metricas["modelos"]
            st.markdown(f'<p class="nota">Modelo entrenado con {metricas["n_casos"]} resoluciones. En las resoluciones '
                        f'más recientes (desde {metricas["año_corte_test"]}) obtuvo un F1-macro de '
                        f'{m[metricas["modelo_elegido"]]["test_f1_macro"]:.2f}, frente a '
                        f'{m["baseline_clase_mayoritaria"]["test_f1_macro"]:.2f} de la referencia trivial.</p>',
                        unsafe_allow_html=True)

if enviar:
    st.divider()
    precedentes, modo = buscar(caso, gold, k=5)
    st.subheader("Resoluciones parecidas")
    st.caption("Por similitud del texto de los hechos." if modo == "semantica"
               else "Por coincidencia en las características del caso.")
    bloque = ""
    for r in precedentes.itertuples():
        hechos = html.escape((r.texto_hechos or "")[:320]) + ("…" if len(r.texto_hechos or "") > 320 else "")
        bloque += (f'<div class="prec"><div class="cab"><span class="id">{html.escape(str(r.caso_id))}</span>'
                   f'<span class="año">{int(r.año_resolucion)}</span>{chip(r.resultado)}</div>'
                   f'<div class="hechos">{hechos}</div></div>')
    st.markdown(bloque, unsafe_allow_html=True)

st.divider()
with st.expander("Marco jurídico de la ponderación"):
    st.markdown("".join(f'<p class="marco"><b>{html.escape(t)}.</b> {html.escape(d)}</p>'
                        for t, d in config.MARCO_GENERAL), unsafe_allow_html=True)
