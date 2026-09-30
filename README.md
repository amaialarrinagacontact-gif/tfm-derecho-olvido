# Derecho al olvido: estimación orientativa ante la AEPD

MVP del TFM (Evolve Academy, 2025-2026). Clasificador supervisado sobre resoluciones de la AEPD,
explicado con SHAP, más recuperación de precedentes similares como capa de apoyo.

## Estado del proyecto (entrega)

Prueba de concepto completa. Qué hay en cada sitio:

| Ruta | Contenido |
|---|---|
| `data/raw/aepd_resoluciones/` | 6 resoluciones reales de la AEPD (documentos públicos) |
| `data/processed/revision.csv` | Etiquetas propuestas y razonadas: **pendientes de validación por la autora** (columna `revisado`) |
| `docs/base_datos_derecho_olvido_AEPD.xlsx` | Catálogo de 37 expedientes con fuente y nivel de verificación |
| `docs/GUIA_ETIQUETADO.md` | Anexo: reglas de etiquetado y relación con los criterios del WP225 |
| `docs/MEMORIA_puntos_clave.md` | Cambios respecto a las entregas, hallazgos, limitaciones y cómo presentarlo |

## Cómo empezar (3 pasos)

1. Instala **Python 3.10 o superior** desde python.org. En Windows, marca "Add Python to PATH".
2. En VS Code: *Archivo → Abrir carpeta* y elige esta carpeta. Si te ofrece instalar la extensión de Python, acepta.
3. Pulsa **F5** (o abre `iniciar.py` y dale al botón ▶).

La primera vez tarda unos minutos porque crea el entorno e instala todo. Después aparece un menú en la terminal:
empieza por la opción 1 para ver la app funcionando con datos de prueba.

El paso 1 necesita un modelo de lenguaje. Vale cualquiera de estos (todos con opción gratuita): **Mistral**, **Groq**, **Gemini** u **Ollama** (local). Pon la clave en `.env` (ver `.env.ejemplo`) o deja que el menú te la pida.

## Coste: 0 €. Qué depende de una API y qué no

| Parte | ¿Usa API? | Alternativa gratuita |
|---|---|---|
| **App web** (predicción, SHAP, precedentes) | **No** | Todo se ejecuta en local |
| Búsqueda semántica de precedentes | **No** | Embeddings `multilingual-e5-small` en tu PC + NumPy |
| Extracción de variables (paso 1, una sola vez) | Opcional | Reglas (`--sin-llm`), o LLM gratuito: Mistral Experiment, Groq u Ollama local |

El LLM solo interviene en la **preparación del dataset**, como herramienta de preprocesamiento
(texto no estructurado -> variables). Nunca en la app.

### Comparación LLM vs reglas (evidencia para la memoria)

El paso 1 guarda siempre las dos extracciones. Tras revisar a mano una muestra en `revision.csv`,
`python 05_evaluar_extraccion.py` calcula el acierto de cada método por variable frente a tu
etiqueta final. Consejo: extrae con el LLM **antes** de revisar, para que ambos métodos se comparen
contra la misma revisión.

## Qué hace cada opción del menú por dentro

| Paso | Comando | Qué hace |
|---|---|---|
| 0 | `python descargar_aepd.py --email tu@correo.com` | Lee `expedientes.txt`; descarga si robots.txt lo permite o genera `pendientes.html` para bajarlos a mano. También puedes copiar tus PDF directamente a `data/raw/aepd_resoluciones/` |
| 1 | `python 01_extraer.py --sin-llm` | Sin API: corta hechos/fallo, descarta lo no relevante y extrae las variables por reglas |
| 1 | `python 01_extraer.py` | El modelo de lenguaje extrae las features **solo de los hechos** y propone el resultado **solo del fallo** |
| ✋ | editar `data/processed/revision.csv` | Revisión humana: corregir y poner `si` en `revisado` |
| 2 | `python 02_construir_gold.py` | Valida categorías y genera `data/gold/gold_casos.parquet` |
| 3 | `python 03_entrenar.py` | Partición temporal, CV en train, baseline, selección, métricas y SHAP |
| 4 | `python 04_indexar.py` (opcional) | Índice semántico local de los hechos (sin API) |
| 5 | `python 05_evaluar_extraccion.py` | Acierto LLM vs reglas frente a tu revisión |
| 5 | `streamlit run app.py` | Interfaz |

Sin el paso 4 la app sigue funcionando: busca precedentes por coincidencia de características.

## Decisiones que implementa el código

- **Sin leakage:** `año_resolucion` solo se usa para ordenar la partición train/test, nunca como feature.
  Las features se extraen del texto anterior a "FUNDAMENTOS DE DERECHO", cortado con reglas.
- **Evaluación:** test = último 25% del corpus por fecha; F1-macro frente a `DummyClassifier`.
- **Regla de decisión:** si otro modelo mejora a la regresión logística en menos de 0,02 de F1-macro,
  se queda la regresión logística.
- **Criterio de aceptación:** si el modelo no supera al baseline en al menos 0,05 de F1-macro en el test
  temporal, la app no muestra estimación y enseña solo precedentes.
- Todas las métricas quedan en `models/metricas.json` para la memoria.

Parámetros ajustables en `src/config.py`.

## Despliegue gratuito

Sube el repo a GitHub con `data/gold/gold_casos.parquet` y `models/modelo.joblib` incluidos
(no los PDFs). En share.streamlit.io elige `app.py` y añade `GEMINI_API_KEY` en *Secrets*
solo si usas el índice semántico.
