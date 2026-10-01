# Puntos clave para la memoria y la defensa

## 1. Cómo presentar el resultado (prueba de concepto)

> El proyecto construye y valida de extremo a extremo un sistema híbrido LLM + Data Science para
> estimar el resultado de reclamaciones de derecho al olvido ante la AEPD. El pipeline de
> extracción se valida con resoluciones reales; el modelo predictivo y la interfaz se demuestran
> con un corpus sintético, declarado como tal, porque el volumen de resoluciones etiquetadas
> alcanzado no permite entrenar un clasificador fiable. Las métricas del modelo sobre datos
> sintéticos acreditan que el sistema funciona, no su capacidad predictiva real.

## 2. Cambios respecto a las entregas

| Entrega | Previsto | Implementado | Motivo |
|---|---|---|---|
| 2 | Scraping con requests + BeautifulSoup | Descarga respetando robots.txt + descarga manual | La AEPD limita el acceso automatizado |
| 2 | AEPD + GDPRhub | AEPD (GDPRhub bloquea bots) + catálogo desde fuentes públicas | Acceso |
| 3 | Gemini + LangChain + ChromaDB | LLM intercambiable (Mistral, Groq, Gemini, Ollama) + reglas sin API; embeddings locales e5 + NumPy | Coste 0 € y sin dependencia de APIs |
| 4 | Features del PDF completo | Solo de los HECHOS, con corte por reglas verificable | Corrección de leakage |
| 4 | — | Variable `alega_libertad_informacion` | Art. 17.3.a RGPD; criterio 11 del WP225 |
| 5 | Frontal en Lovable | Streamlit (sin backend separado) | Un solo lenguaje, despliegue gratuito |

## 3. Hallazgos empíricos (reales)

1. **El corte de secciones funciona con PDF reales:** 5 de 6 separados correctamente. El sexto
   es un archivo de actuaciones con otra estructura.
2. **El filtro de relevancia acierta:** descarta un derecho de acceso y un archivo de actuaciones.
3. **Extracción por reglas frente a revisión humana** (3 casos; indicativo): 93 % de acierto
   medio, pero 33 % en `tipo_contenido`, la variable más interpretativa. Es donde un LLM debería
   aportar valor, lo que apoya el enfoque híbrido propuesto por el profesor.
4. **La mayoría de los "ESTIMAR" posteriores al RGPD son formales:** la AEPD insta a responder o
   denegar motivadamente, sin ponderar. Obliga a excluirlos y reduce el número de estimadas.
5. **Las estimaciones de fondo se concentran en la etapa LOPD (2015-2017)**, y varias fueron
   revisadas judicialmente (TD/02177/2016 → STC 89/2022; TD/00725/2017 → STC 105/2022).
   Consecuencia: con partición temporal, el test reciente sería casi todo desestimadas. Es una
   deriva real del criterio, no un defecto del modelo.
6. **Casos "sin objeto"** (el reclamante ya obtuvo lo pedido) se etiquetaban como desestimados
   leyendo solo el fallo: el sistema lee ahora el motivo para excluirlos.

## 4. Limitaciones

- **Volumen:** 3 casos válidos del corpus real y 11 utilizables en el catálogo, frente a los 250 previstos.
- **Datos sintéticos:** solo demuestran el pipeline; sus métricas no deben interpretarse.
- **Revisión anclada:** revisar partiendo de la propuesta automática puede sesgar la comparación
  LLM vs reglas. Mitigación: extraer con ambos métodos antes de revisar.
- **Criterios del WP225 no modelados** (ver la guía de etiquetado, apartado 4).
- **Sesgo histórico:** el modelo reproduciría la doctrina pasada de la AEPD.

## 5. Trabajo futuro

1. Descargar los PDF de la hoja Corpus y Pendientes del catálogo (xlsx) y ampliar el corpus.
2. Extracción con un LLM gratuito (Mistral Experiment) y tabla LLM vs reglas con más casos.
3. Añadir las variables del WP225 que aparecen en el corpus: exactitud y vida profesional.
4. Ampliar a GDPRhub y a otras autoridades de la UE (art. 17) para ganar volumen.

## 6. Referencias a añadir

Grupo del Artículo 29, WP225 (2014) · CEPD, Directrices 5/2019 · TJUE C-131/12, C-136/17,
C-460/20 · STC 58/2018, 89/2022 y 105/2022 · LOPDGDD, arts. 93 y 94 · Álvarez Rigaudias, C.
(2014), "Sentencia Google Spain y derecho al olvido", *Actualidad Jurídica Uría Menéndez*, 38, 110-118.
