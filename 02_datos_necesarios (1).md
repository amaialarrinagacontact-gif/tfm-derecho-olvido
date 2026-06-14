# Entrega 2 — Selección de idea de proyecto y análisis de datos necesarios

## 1. Idea seleccionada

**Sistema inteligente de análisis predictivo para reclamaciones de derecho al olvido digital en la UE**

---

### Problema que resuelve

El derecho al olvido digital, reconocido en el artículo 17 del Reglamento General de Protección de Datos (RGPD, Reglamento UE 2016/679), permite a cualquier ciudadano europeo solicitar la supresión de información personal publicada en internet cuando concurren determinadas circunstancias: que los datos ya no sean necesarios para el fin con que fueron recogidos, que el interesado retire su consentimiento, que los datos hayan sido tratados ilícitamente, o que exista una obligación legal de supresión. Sin embargo, la aplicación de este derecho no es automática: exige en cada caso una ponderación individualizada entre derechos fundamentales en colisión, principalmente el derecho a la privacidad y protección de datos (artículo 8 de la Carta de Derechos Fundamentales de la UE) y el derecho a la libertad de expresión e información (artículo 11). Esta ponderación depende de factores como la relevancia pública de la persona afectada, la antigüedad de la información, la naturaleza del responsable del tratamiento o la existencia de interés público en mantener el dato accesible, y sus resultados son difícilmente predecibles incluso para profesionales del derecho especializados.

Esta incertidumbre tiene consecuencias prácticas directas y cuantificables. Según el Informe de Transparencia de Google correspondiente al ejercicio 2023, en España se han presentado más de 442.000 solicitudes de desindexación desde 2014, de las cuales solo el 50,7% fue atendida favorablemente. A escala europea, el agregado supera los 5,8 millones de solicitudes, con tasas de estimación que varían significativamente entre jurisdicciones y tipos de contenido. Esta heterogeneidad refleja la complejidad del ejercicio de ponderación y la dificultad de anticipar el resultado de una reclamación sin contar con un análisis comparado de la jurisprudencia previa. Muchos ciudadanos desisten de ejercer un derecho que les asiste por falta de información sobre sus posibilidades de éxito; otros, en sentido contrario, inician procedimientos ante la Agencia Española de Protección de Datos (AEPD) o los tribunales sin una base sólida de viabilidad, incurriendo en costes de tiempo y recursos innecesarios. En ambos casos, el resultado es una asimetría de información estructural entre ciudadanos y grandes plataformas tecnológicas, que cuentan con equipos jurídicos especializados y criterios internos de evaluación que no son públicos.

El sistema propuesto aborda directamente esta laguna: proporcionar a ciudadanos, abogados y operadores jurídicos una herramienta basada en datos que permita realizar una evaluación preliminar objetiva de la viabilidad de una reclamación de derecho al olvido, reduciendo la incertidumbre en la toma de decisiones y contribuyendo a un acceso más equitativo a la tutela de los derechos digitales.

---

### Solución planteada

El proyecto propone desarrollar un sistema de análisis predictivo basado en un corpus de resoluciones y sentencias públicas sobre derecho al olvido, estructurado en tres módulos funcionales complementarios.

El primer módulo es un **clasificador supervisado** que predice el resultado probable de una reclamación (éxito, denegación o éxito parcial) a partir de un conjunto de variables estructuradas extraídas de cada caso: la naturaleza del solicitante, el tipo de responsable del tratamiento, la antigüedad de la información, el tipo de contenido reclamado, la existencia de condena penal previa, la jurisdicción del órgano resolvente y el año de la resolución, entre otras. Se evaluarán y compararán varios algoritmos de clasificación —regresión logística como baseline, Random Forest y XGBoost— seleccionando el que ofrezca mejor equilibrio entre rendimiento predictivo y explicabilidad. La explicabilidad de las predicciones se implementará mediante SHAP (SHapley Additive exPlanations), que permite cuantificar la contribución de cada variable a la predicción final y traducirla al lenguaje de los criterios jurídicos de ponderación que los tribunales aplican en la práctica.

El segundo módulo es un **estimador de tiempo de resolución**, implementado como modelo de regresión sobre las mismas variables estructuradas, con transformación logarítmica de la variable objetivo para corregir la asimetría típica de las distribuciones de tiempos administrativos y judiciales. Este módulo permite ofrecer al usuario una estimación del tiempo previsible entre la interposición de la reclamación y su resolución, información de alto valor práctico para la planificación jurídica.

El tercer módulo es un **sistema de recuperación de jurisprudencia análoga** basado en similitud semántica. Mediante embeddings de texto generados con modelos de lenguaje multilingüe (multilingual-e5-large o similar) y búsqueda eficiente de vecinos próximos con FAISS (Facebook AI Similarity Search), el sistema identifica los casos del corpus más similares al caso de entrada y los presenta con sus metadatos y resultado. Este componente permite contextualizar la predicción con precedentes reales y proporciona al usuario —especialmente si es abogado— material jurisprudencial relevante de forma inmediata.

Los tres módulos se alimentan de un corpus construido a partir de fuentes públicas institucionales (AEPD, TJUE, TEDH), procesado mediante técnicas de NLP para la extracción de variables estructuradas a partir de texto no estructurado, y enriquecido con etiquetado manual supervisado para las variables de resultado y tiempo.

---

### MVP del proyecto final

El producto mínimo viable consistirá en tres componentes entregables y funcionalmente integrados.

El primero es un **notebook de análisis exploratorio y modelado** (EDA + modelling) que documente todo el pipeline técnico: carga y limpieza del dataset, análisis descriptivo de las variables (distribuciones, correlaciones, outliers), construcción de las features, entrenamiento y comparación de modelos de clasificación, evaluación mediante métricas estándar (AUC-ROC, F1-score, precisión, recall) y visualización de la importancia de variables con SHAP. El notebook estará disponible en el repositorio de GitHub y será completamente reproducible.

El segundo es una **interfaz interactiva sencilla** desarrollada con Streamlit que permita al usuario introducir las características de un caso hipotético mediante un formulario y obtener en tiempo real: (a) la probabilidad estimada de éxito de la reclamación, (b) los factores jurídicos que más influyen en esa estimación con su peso relativo, y (c) una lista de los cinco casos más similares del corpus con sus metadatos y resultado. Esta interfaz tiene como objetivo demostrar la utilidad práctica del sistema más allá del ámbito técnico.

El tercer componente es la **documentación del dataset** construido, incluyendo la descripción de las fuentes utilizadas, el proceso de extracción y etiquetado, las limitaciones conocidas y las decisiones metodológicas tomadas. Esta documentación sigue el formato Datasheet for Datasets (Gebru et al., 2021) y garantiza la trazabilidad y reproducibilidad del trabajo.

---

## 2. Datos necesarios

### Variables o campos requeridos

Cada registro del dataset corresponde a una resolución o sentencia individual. A continuación se detallan las variables previstas:

**Variables de entrada — características del caso (features):**

| Variable | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `tipo_solicitante` | Categórica | Persona pública o privada | `privada` |
| `tipo_responsable` | Categórica | Motor de búsqueda, hemeroteca, red social, registro público, otro | `motor_busqueda` |
| `antiguedad_info_años` | Numérica | Años transcurridos desde la publicación hasta la solicitud | `12` |
| `tipo_contenido` | Categórica | Noticia, imagen, dato judicial, dato registral, otro | `noticia` |
| `condena_penal_previa` | Binaria | Existencia de condena penal relacionada con los hechos | `1` (sí) |
| `articulos_invocados` | Texto/lista | Artículos del RGPD o normativa alegados | `Art. 17 RGPD` |
| `jurisdiccion` | Categórica | AEPD, TJUE, TEDH, tribunal nacional | `AEPD` |
| `año_resolucion` | Numérica | Año en que se dictó la resolución | `2021` |
| `pais_solicitante` | Categórica | País de residencia del solicitante | `España` |
| `persona_fallecida` | Binaria | Si la solicitud se refiere a datos de persona fallecida | `0` (no) |
| `interes_historico` | Binaria | Si se alega interés histórico o de investigación por el responsable | `0` (no) |
| `texto_hechos` | Texto libre | Resumen despersonalizado de los hechos del caso | — |
| `texto_fundamentos` | Texto libre | Principales argumentos jurídicos de la resolución | — |

**Variables de salida (targets):**

| Variable | Tipo | Descripción |
|---|---|---|
| `resultado` | Categórica (3 clases) | `estimada` / `desestimada` / `estimada_parcialmente` |
| `dias_resolucion` | Numérica continua | Días entre la solicitud y la resolución (para el módulo de regresión) |

**Variables de apoyo — embeddings semánticos:**
Las columnas `texto_hechos` y `texto_fundamentos` se utilizarán para generar vectores de embeddings con modelos de lenguaje multilingüe. Estos vectores no forman parte del dataset tabular sino del índice vectorial (FAISS) usado por el módulo de recuperación de jurisprudencia análoga.

---

### Granularidad

Una fila por resolución o sentencia individual (nivel de expediente). No se trabaja a nivel de usuario, transacción o producto, sino a nivel de procedimiento jurídico. Cada resolución es una unidad de observación independiente con su propio conjunto de características y resultado.

---

### Profundidad histórica necesaria

El punto de partida natural del corpus es **mayo de 2014**, fecha de la sentencia Google Spain del TJUE (Asunto C-131/12), que estableció por primera vez el derecho al olvido en el contexto europeo de los motores de búsqueda y generó la primera oleada de resoluciones de las autoridades de control. El periodo 2014–2018 es relevante para capturar la jurisprudencia pre-RGPD; el periodo 2018–2025 recoge la aplicación del RGPD en vigor. Disponer de aproximadamente **10 años de histórico** permite analizar la evolución temporal de los criterios de resolución y capturar los cambios jurisprudenciales introducidos por el RGPD.

---

### Volumen aproximado

Se estima un corpus inicial de entre **400 y 800 resoluciones etiquetadas**, distribuidas entre las distintas fuentes. Este volumen es modesto pero coherente con las características del dominio Legal AI, donde los datos etiquetados son escasos por naturaleza y el coste del etiquetado manual es elevado. Con este volumen es posible:

- Entrenar y evaluar modelos de clasificación supervisada con variables estructuradas mediante validación cruzada estratificada
- Construir un índice FAISS funcional para recuperación semántica
- Realizar análisis exploratorio estadísticamente significativo

Si el volumen resultara insuficiente para el módulo de clasificación, se considerará ampliar el corpus a resoluciones generales de protección de datos (no exclusivamente derecho al olvido) para aumentar el tamaño del dataset, manteniendo el derecho al olvido como subconjunto de análisis prioritario.

---

### Datos imprescindibles vs. deseables

**Imprescindibles:**
- Resultado de la reclamación (variable objetivo de clasificación)
- Tipo de solicitante (persona pública o privada)
- Tipo de responsable del tratamiento
- Antigüedad aproximada de la información
- Jurisdicción del órgano resolvente
- Año de la resolución

**Deseables pero no obligatorios:**
- Texto completo o resumen de la resolución (necesario para el módulo semántico, pero el MVP puede funcionar sin él)
- Tiempo exacto de resolución en días (necesario para el módulo de regresión; si no está disponible, este módulo se pospone)
- Desglose de artículos del RGPD invocados
- País de residencia del solicitante
- Identificador del expediente para trazabilidad

---

## 3. Fuentes de datos previstas

### Fuente 1 — AEPD: Resoluciones en materia de derecho de supresión (fuente principal)

- **URL:** [https://www.aepd.es/resoluciones](https://www.aepd.es/resoluciones)
- **Descripción:** La Agencia Española de Protección de Datos publica la totalidad de sus resoluciones de forma pública y gratuita. Las resoluciones sobre derecho de supresión (artículo 17 RGPD) y derecho al olvido son accesibles mediante el buscador por materia.
- **Acceso:** Público y abierto, sin registro ni restricciones de uso para fines académicos.
- **Formato:** Documentos PDF individuales con texto extraíble.
- **Histórico disponible:** Desde aproximadamente 2014, con cobertura completa desde la entrada en vigor del RGPD en 2018.
- **Volumen estimado:** Varias centenas de resoluciones específicas sobre derecho al olvido y supresión.
- **Estabilidad:** Fuente institucional oficial de máxima estabilidad. La AEPD mantiene y amplía regularmente su repositorio.
- **Riesgos:**
  - Los documentos están en formato PDF no estructurado, lo que requiere extracción de texto (pdfplumber, PyMuPDF) y etiquetado manual de variables.
  - No existe API de descarga masiva; la recopilación requiere scraping controlado o descarga manual iterativa.
  - Algunas resoluciones antiguas pueden tener calidad de OCR inferior.
- **Método de obtención previsto:** Scraping con Python (requests + BeautifulSoup) respetando los términos de uso, seguido de extracción de texto con pdfplumber y etiquetado asistido.

---

### Fuente 2 — GDPRhub: Repositorio europeo de resoluciones de protección de datos

- **URL:** [https://gdprhub.eu/index.php?title=Welcome_to_GDPRhub](https://gdprhub.eu/index.php?title=Welcome_to_GDPRhub)
- **Descripción:** GDPRhub es un repositorio colaborativo mantenido por noyb (organización europea de privacidad) que recoge y sistematiza resoluciones de autoridades de protección de datos de todos los Estados miembros de la UE. Incluye metadatos ya estructurados: autoridad, fecha, artículo invocado, resultado y resumen en inglés.
- **Acceso:** Público y abierto. Datos exportables en formato wiki estructurada.
- **Formato:** Wiki con campos etiquetados; exportación posible en JSON o CSV mediante la API de MediaWiki.
- **Histórico disponible:** Desde 2018 (entrada en vigor del RGPD).
- **Estabilidad:** Fuente activamente mantenida por una organización especializada en privacidad europea.
- **Ventaja diferencial:** Algunos campos clave (resultado, autoridad, artículo invocado) ya están estructurados, lo que reduce significativamente el trabajo de etiquetado manual. Cubre resoluciones de múltiples países europeos, no solo España.
- **Riesgos:**
  - Los resúmenes están principalmente en inglés, lo que introduce variabilidad lingüística.
  - La cobertura no es exhaustiva: no todas las resoluciones de todas las autoridades están incluidas.
  - La calidad del etiquetado depende de los colaboradores de la wiki.
- **API:** `https://gdprhub.eu/api.php` (API estándar de MediaWiki)

---

### Fuente 3 — CURIA: Sentencias del Tribunal de Justicia de la UE

- **URL:** [https://curia.europa.eu/juris/recherche.jsf](https://curia.europa.eu/juris/recherche.jsf)
- **Descripción:** Portal oficial del TJUE con todas sus sentencias en las lenguas oficiales de la UE. Permite filtrar por materia, fecha y número de asunto. Las sentencias sobre derecho al olvido más relevantes incluyen: Google Spain (C-131/12, 2014), Google LLC c. CNIL (C-507/17, 2019) y casos relacionados con el artículo 17 RGPD.
- **Acceso:** Público y abierto.
- **Formato:** HTML y PDF con texto extraíble.
- **Histórico disponible:** Completo desde 2014.
- **Volumen estimado:** Reducido en el dominio específico del derecho al olvido (decenas de sentencias), pero de alta relevancia jurídica como casos de referencia.
- **Estabilidad:** Fuente institucional oficial de la UE, máxima estabilidad.
- **Riesgos:**
  - Volumen de sentencias específicas sobre derecho al olvido es limitado. Puede ser necesario ampliar el filtro a casos del artículo 8 CEDH y protección de datos en general.
  - Los textos son multilingües (español, francés, alemán, inglés), lo que requiere gestión de idiomas en el pipeline de NLP.

---

### Fuente 4 — HUDOC: Base de datos del Tribunal Europeo de Derechos Humanos

- **URL:** [https://hudoc.echr.coe.int](https://hudoc.echr.coe.int)
- **Descripción:** Base de datos oficial del TEDH con todas sus sentencias y decisiones. Relevante para el proyecto por la jurisprudencia sobre derecho al olvido en relación con el artículo 8 del CEDH (derecho a la vida privada) y hemerotecas digitales. Casos relevantes: M.L. y W.W. c. Alemania (2018), Hurbain c. Bélgica (2021).
- **Acceso:** Público y abierto, con **API REST JSON documentada**.
- **Formato:** JSON mediante API; también descargable en PDF.
- **API:** `https://hudoc.echr.coe.int/app/query/` con parámetros de filtrado por artículo y palabras clave.
- **Histórico disponible:** Completo desde los años 90; filtrado desde 2014 para este proyecto.
- **Estabilidad:** Fuente institucional oficial del Consejo de Europa.
- **Riesgos:**
  - Los textos están principalmente en inglés y francés, idiomas distintos al español predominante en las resoluciones de la AEPD.
  - El volumen de casos específicamente sobre derecho al olvido digital es reducido.

---

### Fuente 5 — EUR-Lex: Legislación y jurisprudencia europea

- **URL:** [https://eur-lex.europa.eu](https://eur-lex.europa.eu)
- **Descripción:** Portal oficial de legislación de la UE. Relevante para el proyecto como fuente de los textos normativos completos (RGPD, Directiva 95/46/CE, Reglamento de IA) que se indexarán en el módulo RAG para consultas en lenguaje natural sobre legislación aplicable.
- **Acceso:** Público y abierto, con **API SPARQL** y endpoint de descarga en múltiples formatos.
- **Formato:** XML, HTML, PDF, RDF.
- **API:** `https://publications.europa.eu/webapi/rdf/sparql` (SPARQL endpoint público).
- **Uso previsto:** Indexación de los textos normativos completos para el módulo RAG, no como fuente de casos del dataset de entrenamiento.

---

### Fuente 6 — Google Transparency Report

- **URL:** [https://transparencyreport.google.com/eu-privacy/overview](https://transparencyreport.google.com/eu-privacy/overview)
- **Descripción:** Informe público de Google con estadísticas agregadas sobre solicitudes de desindexación recibidas en el marco del derecho al olvido: número de solicitudes por país, porcentaje estimado/desestimado, categorías de contenido.
- **Acceso:** Público, datos descargables en CSV.
- **Formato:** CSV descargable.
- **Uso previsto:** Datos de contexto y estadísticas descriptivas para el análisis exploratorio. No es fuente de casos individuales sino de datos agregados.
- **Riesgos:** No proporciona información sobre casos individuales, solo agregados. Útil para contextualización pero no para entrenamiento del modelo.

---

### Fuente 7 — Tribunal Constitucional Español y CENDOJ

- **URL TC:** [https://hj.tribunalconstitucional.es](https://hj.tribunalconstitucional.es)
- **URL CENDOJ:** [https://www.poderjudicial.es/search/indexAN.jsp](https://www.poderjudicial.es/search/indexAN.jsp)
- **Descripción:** Bases de datos de jurisprudencia constitucional y del Tribunal Supremo español. Relevante para sentencias sobre derecho al olvido en el ámbito nacional (ej. STC 58/2018, STS sobre caso El País).
- **Acceso:** Público y abierto, buscador web con descarga en PDF.
- **Uso previsto:** Complemento al corpus de resoluciones de la AEPD con jurisprudencia judicial española de alto nivel.
- **Riesgos:** El buscador no dispone de API pública documentada; la recopilación requiere búsqueda manual o scraping.

---

## 4. Consideraciones de privacidad y protección de datos

Las resoluciones y sentencias que forman el corpus del proyecto son **documentos públicos** emitidos por autoridades e instituciones judiciales en ejercicio de sus funciones, publicados oficialmente con la finalidad de garantizar la transparencia, el acceso a la jurisprudencia y la seguridad jurídica. En ese sentido, su uso en un proyecto académico es legítimo bajo el principio de limitación de finalidad del RGPD (artículo 5.1.b), que reconoce expresamente la investigación académica como finalidad compatible.

No obstante, se aplican las siguientes cautelas específicas:

**Anonimización de personas afectadas.** Muchas resoluciones de la AEPD ya están anonimizadas en origen (los solicitantes aparecen como "D./Dña. X.X."). En los casos en que figuren nombres completos, el dataset del proyecto no los incluirá: las variables estructuradas no incorporarán datos identificativos de los solicitantes, y los resúmenes de texto libre serán despersonalizados antes de su inclusión en el corpus.

**Ausencia de perfiles individuales.** El sistema no recoge datos directamente de ciudadanos ni genera perfiles sobre personas físicas. El objeto de análisis son los expedientes jurídicos como unidad de observación abstracta. El modelo predice resultados de tipos de caso, no de personas concretas.

**Datos sensibles.** Algunas resoluciones pueden hacer referencia a datos de categorías especiales (artículo 9 RGPD): datos de salud, datos penales, creencias religiosas. Estos datos aparecen contextualizados dentro de los hechos de los expedientes, no como atributos directos de los solicitantes. En cualquier caso, el dataset no incluirá estos datos como variables de entrada del modelo; se limitará a registrar el tipo de dato reclamado de forma categorizada (ej. `dato_judicial`, `dato_salud`) sin reproducir el contenido sensible.

**Uso académico exclusivo.** El dataset construido tiene finalidad exclusivamente académica y de investigación. No será publicado de forma que permita identificar a personas afectadas. El repositorio de GitHub incluirá el código y la documentación del pipeline, pero no necesariamente los documentos fuente completos si ello pudiera implicar riesgos de privacidad.

**Cumplimiento RGPD.** Resulta especialmente relevante señalar que, al tratarse de un proyecto sobre derecho al olvido, se ha prestado atención particular a que el propio sistema respete los principios de minimización de datos (artículo 5.1.c), limitación de la finalidad (artículo 5.1.b) y exactitud (artículo 5.1.d) establecidos en el RGPD. El sistema no está diseñado para identificar ni rastrear a personas concretas, sino para analizar patrones en decisiones jurídicas agregadas.

**Riesgo ético de sesgo algorítmico.** Toda herramienta predictiva basada en jurisprudencia histórica incorpora de forma intrínseca los sesgos presentes en las resoluciones sobre las que ha sido entrenada. El sistema puede reproducir tendencias jurisprudenciales dominantes y subrepresentar criterios emergentes o minoritarios. Este riesgo se documenta explícitamente y se mitiga mediante: (a) transparencia sobre las limitaciones del dataset, (b) explicabilidad de las predicciones con SHAP, y (c) presentación del resultado como estimación probabilística orientativa, no como dictamen jurídico.

---

## 5. Viabilidad inicial del proyecto

**¿Es viable obtener los datos necesarios?**

Sí, con esfuerzo de recopilación y etiquetado controlado. Las fuentes principales (AEPD, CURIA, HUDOC, GDPRhub) son públicas, accesibles sin restricciones relevantes y mantienen histórico desde 2014. La principal dificultad no es el acceso a los documentos sino su estructuración: los textos están en formato PDF y requieren extracción de texto y etiquetado manual de variables clave. GDPRhub actúa como fuente parcialmente pre-estructurada que reduce significativamente el trabajo de etiquetado para las resoluciones que cubre. La combinación de fuentes prevista es suficiente para construir un corpus inicial funcional.

**¿Tiene suficiente calidad, granularidad y profundidad histórica?**

La granularidad es adecuada para el objetivo: una resolución por registro es el nivel natural de análisis en Legal AI. La profundidad histórica desde 2014 (aprox. 10 años) es suficiente para capturar la evolución jurisprudencial en dos etapas regulatorias diferenciadas (pre y post-RGPD). La calidad del dataset dependerá del rigor del proceso de etiquetado, que se documentará explícitamente siguiendo el formato Datasheet for Datasets (Gebru et al., 2021).

**¿Es realista desarrollar el proyecto durante el curso?**

Sí, con un alcance bien delimitado y priorización clara. El MVP no requiere un corpus masivo ni un sistema de producción: con 400–600 resoluciones etiquetadas es posible construir un clasificador funcional evaluable con métricas estándar y un sistema de recuperación semántica operativo. La fase más costosa en tiempo es la recopilación y etiquetado inicial del dataset; el modelado y la interfaz son técnicamente más directos dado el stack de herramientas disponibles (scikit-learn, XGBoost, sentence-transformers, FAISS, Streamlit).

**¿Qué parte del proyecto es más arriesgada?**

La construcción del dataset es el cuello de botella principal. Dos riesgos concretos: (1) que el volumen de resoluciones accesibles con variables completas sea inferior al estimado, lo que limitaría la robustez estadística del clasificador; (2) que el proceso de etiquetado manual consuma más tiempo del previsto, retrasando la fase de modelado. Ambos riesgos están mitigados por la existencia de GDPRhub como fuente alternativa con datos pre-estructurados.

**¿Qué alternativa existe si la fuente principal falla?**

Si la AEPD limitara el acceso a sus resoluciones o el volumen específico de casos de derecho al olvido resultara insuficiente, se activaría el **plan alternativo** en dos niveles:

- *Nivel 1:* Usar GDPRhub como fuente primaria en lugar de complementaria, ampliando el corpus a resoluciones de múltiples autoridades europeas (no solo española) sobre el artículo 17 RGPD.
- *Nivel 2:* Ampliar el scope temático a resoluciones generales de protección de datos (artículos 15–22 RGPD) para aumentar el volumen del dataset, manteniendo el derecho al olvido (artículo 17) como subconjunto de análisis principal y variable de filtrado en los experimentos.

En ningún escenario se contempla el uso de datos que requieran permisos especiales, pagos o accesos restringidos, lo que garantiza la viabilidad del proyecto independientemente del escenario de datos que finalmente se materialice.

---

*Repositorio del proyecto: [pendiente de creación en GitHub]*  
*Autora: Amaia Gil Pérez — Máster en Data Science, ML e IA — Evolve Academy 2025-2026*

---

## 6. Referencias bibliográficas

Bonnici, J. P. M., Steinbauer, G., Koestler, W., & Gottlob, G. (2020). Automated classification of data protection decisions. *Proceedings of the 17th International Conference on Artificial Intelligence and Law (ICAIL 2020)*, 171–175. https://doi.org/10.1145/3322640.3326718

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2019). ECHR-OD: A legal judgment prediction dataset for the European Court of Human Rights. *arXiv preprint*. https://arxiv.org/abs/1904.06083

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The muppets straight out of law school. *Findings of EMNLP 2020*. https://doi.org/10.18653/v1/2020.findings-emnlp.261

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Comité Europeo de Protección de Datos. (2020). *Directrices 5/2019 sobre los criterios del derecho a la supresión («derecho al olvido») en los motores de búsqueda según el RGPD*. https://edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-52019-criteria-right-erasure_es

Consejo de Europa. (1950). *Convenio para la Protección de los Derechos Humanos y de las Libertades Fundamentales* (CEDH). https://www.echr.coe.int/documents/d/echr/Convention_ENG

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint*. https://arxiv.org/abs/1702.08608

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Google LLC. (2024). *Informe de transparencia: Solicitudes de privacidad europeas para la búsqueda de Google*. https://transparencyreport.google.com/eu-privacy/overview

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547. https://doi.org/10.1109/TBDATA.2019.2921572

Katz, D. M., Bommarito, M. J., & Blackman, J. (2017). A general approach for predicting the behavior of the Supreme Court of the United States. *PLOS ONE, 12*(4), e0174698. https://doi.org/10.1371/journal.pone.0174698

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems (NeurIPS 2020), 33*, 9459–9474. https://arxiv.org/abs/2005.11401

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017), 30*. https://arxiv.org/abs/1705.07874

Parlamento Europeo y Consejo de la Unión Europea. (2016). *Reglamento (UE) 2016/679 del Parlamento Europeo y del Consejo, de 27 de abril de 2016, relativo a la protección de las personas físicas en lo que respecta al tratamiento de datos personales y a la libre circulación de estos datos* (Reglamento General de Protección de Datos). *Diario Oficial de la Unión Europea, L 119*, 1–88. https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32016R0679

Parlamento Europeo y Consejo de la Unión Europea. (2024). *Reglamento (UE) 2024/1689 del Parlamento Europeo y del Consejo, de 13 de junio de 2024, por el que se establecen normas armonizadas en materia de inteligencia artificial* (Reglamento de Inteligencia Artificial). *Diario Oficial de la Unión Europea, L 2024/1689*. https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=OJ:L_202401689

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Tribunal de Justicia de la Unión Europea. (2014, 13 de mayo). *Sentencia en el asunto C-131/12: Google Spain SL y Google Inc. contra Agencia Española de Protección de Datos (AEPD) y Mario Costeja González*. ECLI:EU:C:2014:317. https://curia.europa.eu/juris/document/document.jsf?docid=153302

Tribunal de Justicia de la Unión Europea. (2019, 24 de septiembre). *Sentencia en el asunto C-507/17: Google LLC contra Commission nationale de l'informatique et des libertés (CNIL)*. ECLI:EU:C:2019:772. https://curia.europa.eu/juris/document/document.jsf?docid=218105

Tribunal de Justicia de la Unión Europea. (2022, 8 de diciembre). *Sentencia en los asuntos acumulados C-460/20: TU y RE contra Google LLC*. ECLI:EU:C:2022:962. https://curia.europa.eu/juris/document/document.jsf?docid=268781

Tribunal Europeo de Derechos Humanos. (2018, 28 de junio). *Sentencia en el asunto M.L. y W.W. contra Alemania* (nos. 60798/10 y 65599/10). ECLI:CE:ECHR:2018:0628JUD006079810. https://hudoc.echr.coe.int/eng?i=001-184173

Tribunal Europeo de Derechos Humanos. (2021, 22 de junio). *Sentencia de la Gran Sala en el asunto Hurbain contra Bélgica* (no. 57292/16). ECLI:CE:ECHR:2021:0622JUD005729216. https://hudoc.echr.coe.int/eng?i=001-210798

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv preprint*. https://arxiv.org/abs/2212.03533
