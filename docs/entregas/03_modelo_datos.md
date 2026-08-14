# Entrega 3 - Diseño del modelo de datos y capa gold del proyecto

> **Nota sobre esta revisión:** a partir de la retroalimentación recibida, se reincorpora al proyecto un núcleo predictivo de Data Science (clasificación supervisada sobre un corpus estructurado de resoluciones), que faltaba en la versión anterior de esta entrega. El componente RAG se mantiene, pero se reposiciona como capa de apoyo del clasificador —recuperación de precedentes, justificación de la predicción y citas verificables— y no como el producto completo. Esto evita que el resultado final sea equivalente a cargar unos documentos en un asistente conversacional genérico, y devuelve al proyecto un componente medible y explicable propio de un TFM de Data Science.

## 1. Resumen de la idea y datos del proyecto

**Problema que resuelve:** Ante una reclamación de derecho al olvido digital (artículo 17 RGPD), no es fácil anticipar si va a prosperar, ni encontrar de forma rápida precedentes comparables que la sustenten. La documentación oficial es densa y dispersa, pero además el resultado de una reclamación concreta depende de una ponderación de factores (tipo de solicitante, antigüedad del contenido, tipo de responsable, etc.) que rara vez se sistematiza.

**Solución construida — dos componentes con roles distintos:**

1. **Núcleo predictivo (eje del proyecto).** Un clasificador supervisado entrenado sobre un corpus estructurado de resoluciones reales de la AEPD, que predice la categoría de resultado más probable de una reclamación (`estimada` / `desestimada` / `estimada_parcialmente`) a partir de variables del caso, y explica esa predicción con SHAP. Este componente es el que aporta valor de Data Science evaluable: hay una variable objetivo, un criterio de etiquetado cerrado, una tabla de features y métricas de desempeño.
2. **Capa de apoyo (RAG).** Un mecanismo de recuperación semántica que, dado un caso de entrada, busca las resoluciones más similares dentro del mismo corpus y las presenta como precedentes citables junto con la predicción del clasificador. El RAG no responde preguntas abiertas sobre cualquier documento legal: opera sobre el mismo corpus de resoluciones que alimenta al clasificador, y su función es justificar y contextualizar la predicción, no sustituirla.

**Fuente de datos utilizada — una única fuente principal:**
- **AEPD — Resoluciones en materia de derecho de supresión** (`https://www.aepd.es/resoluciones`), como fuente única del corpus estructurado y del corpus documental del RAG. Se abandona, para el alcance de esta entrega, el uso de una segunda fuente en paralelo (GDPRhub), que queda como posible verificación cruzada puntual del criterio de etiquetado, no como fuente de datos del modelo.
- Se retira del corpus principal el uso del texto del RGPD y de la sentencia Google Spain como documentos indexados de forma independiente: siguen siendo referencia normativa del proyecto, pero no forman parte del dataset de entrenamiento ni de la base de precedentes recuperables, precisamente para que el RAG no derive hacia un asistente documental genérico.

**Tipo de información que aporta la fuente:** cada resolución de la AEPD aporta (a) un conjunto de variables estructuradas extraíbles del propio expediente (features + variable objetivo) y (b) un texto no estructurado (hechos y fundamentos) que se trocea e indexa semánticamente para la recuperación de precedentes. Ambas vistas —tabular y semántica— provienen del mismo caso y están enlazadas por un identificador común (ver sección 5).

---

## 2. Variable objetivo y taxonomía de resultados

Este es el punto que la versión anterior de esta entrega no cerraba y que ahora se fija de forma explícita, en línea con el criterio de etiquetado ya definido en la Entrega 2.

| Clase | Definición operativa | Criterio de asignación |
|---|---|---|
| `estimada` | La resolución ordena la supresión, bloqueo o desindexación **total** de lo solicitado | Se lee del fallo o parte dispositiva de la resolución, nunca de los fundamentos jurídicos previos |
| `desestimada` | La resolución **deniega íntegramente** la solicitud | Idem |
| `estimada_parcialmente` | La resolución concede la supresión de **parte** del contenido, para determinados canales, o bajo condiciones | Idem |

- **Tipo de tarea:** clasificación multiclase (3 clases), no binaria — evita colapsar artificialmente los casos parciales, que son frecuentes y jurídicamente relevantes en este dominio.
- **Balance de clases esperado:** se anticipa desbalance (previsiblemente más `desestimada` o `estimada_parcialmente` que `estimada` total), por lo que la métrica principal de evaluación será **F1-macro**, no accuracy, y se reportará la matriz de confusión completa.
- **Variables de entrada (features):** las ya definidas en la Entrega 2 — `tipo_solicitante`, `tipo_responsable`, `antiguedad_info_años`, `tipo_contenido`, `condena_penal_previa`, `jurisdiccion` (aquí siempre `AEPD`, al ser fuente única), `año_resolucion`, `pais_solicitante`, `persona_fallecida`, `interes_historico`. El texto de hechos y fundamentos **no** se usa como feature tabular del clasificador; se reserva para el índice vectorial del RAG.

---

## 3. Tecnología o formato de almacenamiento elegido

El proyecto ya no tiene un único tipo de dato (texto no estructurado), sino dos vistas del mismo corpus que requieren tecnologías distintas:

- **PDF** para las resoluciones originales de la AEPD (capa raw): formato en el que se publican oficialmente, sin conversión previa.
- **Parquet** (o alternativamente SQLite) para la **tabla estructurada de casos** (capa gold tabular): un registro por resolución, con las variables de entrada, la variable objetivo `resultado` y un identificador único `caso_id`. Se prioriza Parquet por simplicidad y por integrarse bien con pandas/scikit-learn sin necesidad de un motor de base de datos adicional; SQLite queda como alternativa si se necesita hacer consultas ad hoc durante el etiquetado.
- **JSON** (opcional) para los chunks de texto tras el splitting (capa processed del RAG), como evidencia intermedia exportable a disco.
- **ChromaDB** (base de datos vectorial local) para los embeddings indexados (capa gold vectorial), usada exclusivamente como capa de apoyo del clasificador.

**Justificación:** a diferencia de la versión anterior, sí existe ahora una necesidad clara de almacenamiento tabular: el clasificador necesita una tabla de features con una fila por caso, con tipos de dato fijos y una variable objetivo bien definida. Esta tabla es la que se entrena, se valida y se evalúa con métricas estándar, y es la que convierte al proyecto en un ejercicio de Data Science evaluable más allá de la recuperación semántica. ChromaDB se mantiene únicamente para la capa de apoyo, por las mismas razones de ligereza e integración señaladas en la entrega anterior, pero deja de ser el único gold dataset del proyecto.

---

## 4. Estructura de capas de datos

```
data/
├── raw/
│   └── aepd_resoluciones/        <- PDFs originales descargados de la AEPD, uno por expediente
│
├── processed/
│   ├── casos_extraidos.parquet   <- tabla intermedia tras extracción de texto + parseo de variables,
│   │                                 antes de la revisión/etiquetado manual del resultado
│   └── chunks_texto.json         <- chunks de hechos/fundamentos tras el splitting (opcional en disco)
│
└── gold/
    ├── gold_casos.parquet        <- tabla final de casos etiquetados (features + variable objetivo),
    │                                 lista para entrenar y evaluar el clasificador
    └── chroma_db/                <- colección vectorial de chunks, indexada para recuperación de
                                      precedentes; cada chunk enlaza a gold_casos vía metadata.caso_id
```

- **raw/**: los PDFs de resoluciones de la AEPD tal como se descargaron, sin modificar.
- **processed/**: dos salidas intermedias — la tabla de variables extraídas automáticamente (antes de la revisión manual del criterio de etiquetado) y los chunks de texto para el RAG.
- **gold/**: ahora son **dos** datasets gold, no uno, con roles distintos (ver sección 5).

---

## 5. Definición de la capa gold

### 5.1 `gold_casos` — dataset tabular para el clasificador (dataset principal)

| Campo | Granularidad | Descripción |
|---|---|---|
| `gold_casos.parquet` | 1 fila por resolución de la AEPD | Features estructuradas + variable objetivo `resultado`, ya revisadas y etiquetadas a mano según el criterio de la sección 2 |

- **Nombre:** `gold_casos.parquet`.
- **Descripción funcional:** tabla de entrenamiento/evaluación del clasificador supervisado. Es el dataset que se divide en train/test, sobre el que se entrenan regresión logística (baseline), Random Forest y XGBoost, y sobre el que se calculan las métricas.
- **Nivel de granularidad:** un registro = una resolución de la AEPD.
- **Número aproximado de registros:** 250–400 casos etiquetados (ver Entrega 2, sección de volumen), consistente con una única fuente principal.
- **Campos principales:** `caso_id` (str, UUID o identificador de expediente), `tipo_solicitante`, `tipo_responsable`, `antiguedad_info_años`, `tipo_contenido`, `condena_penal_previa`, `jurisdiccion`, `año_resolucion`, `pais_solicitante`, `persona_fallecida`, `interes_historico`, `resultado` (target).
- **Identificador principal:** `caso_id`, generado a partir del número de expediente de la AEPD (o UUID si el expediente no es citable de forma limpia).
- **Fase posterior que lo consume:** el pipeline de entrenamiento (scikit-learn / XGBoost) y el módulo de explicabilidad (SHAP); también es la tabla de referencia para calcular, por `caso_id`, qué chunks del índice vectorial pertenecen a cada caso.

### 5.2 `chroma_db` — colección vectorial para la capa de apoyo (RAG)

| Campo | Granularidad | Descripción |
|---|---|---|
| Colección Chroma (`./chroma_db`) | 1 vector por chunk de texto de hechos/fundamentos (~1000 caracteres, overlap 200) | `embedding`, `page_content`, `metadata.caso_id`, `metadata.source`, `metadata.page` |

- **Nombre:** colección Chroma persistida en `./chroma_db`, generada con `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`).
- **Descripción funcional:** índice semántico sobre los textos de hechos y fundamentos de las mismas resoluciones que componen `gold_casos`. Dado un caso nuevo, permite recuperar los `k` precedentes más similares y mostrarlos junto a la predicción del clasificador, con su resultado real y su cita.
- **Nivel de granularidad:** un registro = un chunk de texto (~1000 caracteres).
- **Número aproximado de registros:** del orden de 400–700 chunks, proporcional al número de casos del corpus (250–400) y no a un corpus documental externo más amplio — es sensiblemente menor que la estimación anterior (~1500 chunks) porque ya no se indexan el RGPD ni la sentencia Google Spain como documentos independientes.
- **Campos principales:** `page_content` (str), `embedding` (float[3072]), `metadata.caso_id` (str, clave que enlaza con `gold_casos`), `metadata.source` (str, PDF de origen), `metadata.page` (int).
- **Identificador principal:** UUID interno autogenerado por Chroma para cada vector; `metadata.caso_id` es la clave de enlace funcional con `gold_casos`.
- **Fase posterior que lo consume:** el módulo de recuperación de precedentes, invocado **después** de que el clasificador emita una predicción, para justificarla con casos reales — no como punto de entrada independiente de una interfaz conversacional abierta.

---

## 6. Relaciones entre datos

A diferencia de la versión anterior de esta entrega, el proyecto **sí tiene ahora una relación explícita que modelar**, aunque siga siendo un esquema sencillo:

- **`gold_casos` (1) → `chroma_db` (N):** cada resolución de `gold_casos` puede generar varios chunks en la colección vectorial (uno o más fragmentos de hechos/fundamentos), enlazados mediante `metadata.caso_id`. Es una relación 1:N, no N:M — un chunk pertenece siempre a un único caso.
- **Uso de esta relación:** cuando el clasificador predice el resultado de un caso de entrada, el módulo de recuperación busca los chunks más similares en `chroma_db` y, a través de `metadata.caso_id`, recupera de `gold_casos` el resultado real, el año y la jurisdicción de esos precedentes, para mostrarlos junto a la predicción. Es decir, la relación no es solo para citar la fuente (como en la versión anterior), sino para **contrastar la predicción del modelo con el resultado real de casos similares**, lo cual es justamente el valor añadido que pedía la revisión: el RAG deja de ser un fin en sí mismo y pasa a apoyar al núcleo predictivo.
- No se contemplan relaciones N:M ni joins entre múltiples fuentes tabulares, porque el proyecto mantiene una única fuente principal (AEPD) y un único dataset de casos.

---

## 7. Diccionario de datos inicial

### 7.1 `gold_casos.parquet`

| Campo | Descripción | Tipo | Fuente | Obligatorio | Observaciones |
|---|---|---|---|---|---|
| `caso_id` | Identificador único del caso | str | Nº de expediente AEPD / UUID | Sí | Clave de enlace con `chroma_db.metadata.caso_id` |
| `tipo_solicitante` | Persona pública o privada | categórica | Extracción + revisión manual | Sí | Criterio de etiquetado en Entrega 2 |
| `tipo_responsable` | Motor de búsqueda, hemeroteca, red social, registro público, otro | categórica | Extracción + revisión manual | Sí | — |
| `antiguedad_info_años` | Años entre publicación y solicitud | numérica | Extracción + revisión manual | No (puede ser `NaN`) | No se imputa a mano si no consta la fecha |
| `tipo_contenido` | Noticia, imagen, dato judicial, dato registral, otro | categórica | Extracción + revisión manual | Sí | — |
| `condena_penal_previa` | Condena penal firme relacionada | binaria | Extracción + revisión manual | Sí | Solo `1` si la condena es firme |
| `jurisdiccion` | Órgano resolvente | categórica | Fija: `AEPD` | Sí | Fuente única en esta fase |
| `año_resolucion` | Año de la resolución | numérica | Extracción automática | Sí | — |
| `pais_solicitante` | País de residencia del solicitante | categórica | Extracción + revisión manual | No | — |
| `persona_fallecida` | Solicitud referida a persona fallecida | binaria | Extracción + revisión manual | No | — |
| `interes_historico` | Interés histórico/científico invocado por el responsable | binaria | Extracción + revisión manual | No | Solo si se invoca expresamente |
| `resultado` | Variable objetivo (3 clases) | categórica | Fallo de la resolución | Sí | Ver taxonomía sección 2 |

### 7.2 `chroma_db` (colección vectorial)

| Campo | Descripción | Tipo | Fuente | Obligatorio | Observaciones |
|---|---|---|---|---|---|
| `page_content` | Texto del chunk (hechos/fundamentos) | str | PDFs AEPD | Sí | `RecursiveCharacterTextSplitter` |
| `embedding` | Vector semántico del chunk | float[3072] | Gemini Embeddings | Sí | Generado al indexar |
| `metadata.caso_id` | Enlace al caso en `gold_casos` | str | Pipeline de extracción | Sí | Clave de la relación 1:N |
| `metadata.source` | Fichero PDF de origen | str | `PyPDFDirectoryLoader` | Sí | Para citar la fuente |
| `metadata.page` | Página de origen | int | `PyPDFDirectoryLoader` | Sí | Para citar la fuente |

---

## 8. Métricas de evaluación del núcleo predictivo

Este es otro punto que quedaba abierto y que se cierra aquí, porque es lo que permite demostrar que el sistema aporta algo más que una búsqueda semántica:

- **Métrica principal:** F1-macro (da el mismo peso a las tres clases, apropiado dado el desbalance esperado).
- **Métricas secundarias:** AUC-ROC one-vs-rest por clase, precisión y recall por clase, matriz de confusión completa.
- **Validación:** validación cruzada estratificada (k=5) dado el tamaño moderado del corpus (250–400 casos), para obtener estimaciones de desempeño más estables que con una única partición train/test.
- **Baseline de referencia:** un clasificador que prediga siempre la clase mayoritaria; el modelo entrenado debe superarlo de forma clara en F1-macro para justificar el valor del sistema.
- **Explicabilidad:** SHAP sobre el modelo final seleccionado, reportando la importancia global de variables y, para casos individuales en la interfaz, la contribución de cada variable a la predicción concreta.
- **Evaluación de la capa de apoyo (RAG):** de forma complementaria, y no como sustituto de las métricas anteriores, se revisará manualmente sobre una muestra de casos si los precedentes recuperados por similitud son jurídicamente relevantes (revisión cualitativa, no una métrica de negocio del proyecto).

---

## 9. Problemas de calidad esperados

- **Desbalance de clases** en `resultado`, que puede sesgar el clasificador hacia la clase mayoritaria si no se controla con la métrica adecuada (F1-macro) y, si es necesario, con ponderación de clases (`class_weight='balanced'`).
- **Variables con huecos:** `antiguedad_info_años`, `pais_solicitante` y otras variables "deseables" (Entrega 2) pueden tener una proporción relevante de valores faltantes si la resolución no permite reconstruirlas con precisión; se tratan como `NaN`, no se imputan a mano.
- **Chunks que cortan a mitad de un razonamiento jurídico:** al trocear hechos/fundamentos en fragmentos de tamaño fijo, se puede perder contexto; mitigado parcialmente con `chunk_overlap=200`, y de impacto menor que antes porque el corpus del RAG ahora es más pequeño y homogéneo (solo resoluciones AEPD, no normativa ni jurisprudencia del TJUE con estructura distinta).
- **Ruido de cabeceras y pies de página repetidos** en los PDFs de la AEPD.
- **Consistencia del etiquetado manual de `resultado`:** aunque el criterio está cerrado (sección 2), sigue existiendo el riesgo de error humano en casos límite (resoluciones con fallos poco claros); se mitiga con doble revisión de una muestra aleatoria del corpus.

---

## 10. Decisiones de limpieza y transformación previstas

- **Extracción de features estructuradas:** parseo semiautomático del texto del PDF (patrones + revisión manual) para poblar `casos_extraidos.parquet`, seguido de revisión manual obligatoria del campo `resultado` a partir del fallo, según el criterio de la sección 2.
- **Chunking del texto para el RAG:** `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200`, aplicado únicamente sobre los textos de hechos y fundamentos de las resoluciones que ya forman parte de `gold_casos` (no sobre documentos normativos externos).
- **Sin eliminación agresiva de texto:** se mantiene el criterio de no filtrar cabeceras/pies de forma automática, para no perder contenido legal relevante por error.
- **Codificación de variables categóricas:** one-hot encoding para el baseline de regresión logística; codificación nativa de categóricas para Random Forest / XGBoost.
- **Criterio de registro válido:** un caso se considera válido para `gold_casos` si el fallo permite determinar sin ambigüedad la clase de `resultado`; los expedientes cuyo fallo no sea claro se excluyen del corpus de entrenamiento y se documentan como descartados, en lugar de forzar una etiqueta dudosa.

---

## 11. Riesgos del modelo de datos

- **Parte más clara:** la definición de la tabla `gold_casos` y del criterio de etiquetado de `resultado` es ya un contrato cerrado y reproducible (Entrega 2 + sección 2 de este documento).
- **Parte que genera más incertidumbre:** que el volumen final de casos con features completas sea suficiente para que el clasificador supere de forma robusta al baseline de clase mayoritaria en F1-macro; con corpus pequeños (250–400 casos) el margen de mejora puede ser modesto y debe reportarse con honestidad, incluyendo los intervalos de la validación cruzada.
- **Fuente que puede dar más problemas:** siendo AEPD la única fuente, cualquier limitación de acceso o cambio en el formato de publicación afecta a la totalidad del corpus, no solo a una parte; es el principal punto de fragilidad de haber reducido a una única fuente principal.
- **Qué ocurriría si no se puede construir `gold_casos` con el volumen previsto:** se activaría el plan de contingencia ya descrito en la Entrega 2 (uso de GDPRhub como fuente complementaria puntual, o ampliación del scope temático a artículos 15–22 RGPD), sin renunciar al núcleo predictivo.
- **Qué ocurriría si la capa de apoyo (RAG) da problemas de indexación:** el sistema puede funcionar igualmente mostrando solo la predicción del clasificador y su explicación SHAP, sin precedentes recuperados; el RAG es una capa de apoyo prescindible en última instancia, mientras que el clasificador es el componente que no puede eliminarse sin perder el eje del proyecto.
