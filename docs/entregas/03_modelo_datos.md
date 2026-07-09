# Entrega 3 - Diseño del modelo de datos y capa gold del proyecto

## 1. Resumen de la idea y datos del proyecto

**Problema que resuelve:** Muchas personas desconocen sus derechos sobre supresión de datos personales (derecho al olvido) bajo el RGPD, y la documentación oficial (artículos legales, guías de la AEPD, jurisprudencia) es densa, técnica y está dispersa en varias fuentes.

**Solución construida:** Un agente conversacional basado en RAG (Retrieval-Augmented Generation) que indexa documentación oficial sobre el derecho al olvido digital y responde preguntas de forma precisa, citando la fuente, usando Google Gemini como LLM y ChromaDB como base vectorial. El agente está implementado con LangGraph (a través de `create_agent` de LangChain), incluye memoria conversacional y expone una tool de búsqueda semántica sobre la base documental.

**Fuentes de datos utilizadas:**
- Texto oficial del RGPD (artículo 17 y normativa relacionada).
- Guía de la AEPD sobre el derecho al olvido.
- Sentencia Google Spain vs AEPD (TJUE, 2014).

**Tipo de información que aporta cada fuente:** texto normativo y jurisprudencial no estructurado (no tabular), que se convierte en fragmentos (chunks) indexados semánticamente para su recuperación por similitud.

## 2. Tecnología o formato de almacenamiento elegido

- **PDF** para los documentos originales (capa raw): es el formato en el que se publican oficialmente el RGPD, las guías de la AEPD y las sentencias del TJUE, así que no tiene sentido convertirlos a otro formato antes de procesarlos.
- **JSON** (opcional) para los chunks de texto tras el splitting (capa processed), como evidencia intermedia exportable a disco.
- **ChromaDB** (base de datos vectorial local) para los embeddings indexados (capa gold).

**Justificación:** el proyecto trabaja con texto no estructurado, no con datos tabulares, por lo que una base de datos relacional (SQLite/PostgreSQL) no aporta valor aquí — no hay filas con columnas fijas que relacionar, sino fragmentos semánticos que se comparan por similitud vectorial. ChromaDB es ligero, corre en local, se integra de forma nativa con LangChain/LangGraph y es más que suficiente para el volumen de este proyecto (~1500 chunks); no se necesita una solución cloud como Pinecone o Weaviate para un proyecto académico de este tamaño.

## 3. Estructura de capas de datos

```
data/
├── raw/         <- PDFs originales (RGPD, guía AEPD, sentencia Google Spain)
├── processed/   <- chunks de texto tras el splitting (RecursiveCharacterTextSplitter)
└── gold/        <- referencia lógica a chroma_db/, colección de vectores indexados
```

- **raw/**: los PDFs tal como se descargaron de las fuentes oficiales, sin modificar.
- **processed/**: el resultado de aplicar `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) sobre los documentos cargados con `PyPDFDirectoryLoader`. En la implementación actual estos chunks viven como objetos en memoria de Python (`docs_split`); opcionalmente se pueden exportar a un JSON en `data/processed/` como evidencia física de esta capa intermedia.
- **gold/**: no es una carpeta de ficheros sueltos, sino la colección vectorial persistida en `./chroma_db/`. Es la capa final, lista para ser consumida por el agente.

## 4. Definición de la capa gold

| Dataset gold | Granularidad | Campos clave | Consumidor |
|---|---|---|---|
| Colección Chroma (`./chroma_db`) | 1 vector por chunk (~1000 caracteres, overlap 200) | `embedding` (3072 dim), `page_content`, `metadata.source`, `metadata.page` | Retriever del agente LangGraph → tool `buscar_documentacion` → generación de respuestas con Gemini |

- **Nombre:** colección Chroma persistida en `./chroma_db`, generada con `GoogleGenerativeAIEmbeddings` (modelo `gemini-embedding-001`).
- **Descripción funcional:** conjunto de fragmentos de texto legal indexados semánticamente, que actúan como base de conocimiento consultable por el agente.
- **Nivel de granularidad:** un registro = un chunk de texto (~1000 caracteres).
- **Número aproximado de registros:** ~1500 chunks (resultado de trocear ~277 páginas de PDFs).
- **Campos principales:**
  - `page_content` (str): texto del fragmento.
  - `embedding` (float[3072]): vector semántico generado por Gemini Embeddings.
  - `metadata.source` (str): nombre del PDF de origen.
  - `metadata.page` (int): página de origen dentro del PDF.
- **Identificador principal:** UUID interno autogenerado por Chroma para cada vector.
- **Variables relevantes:** `embedding`, usado para la búsqueda por similitud (`similarity_search`, k=4).
- **Fase posterior que lo consume:** el retriever del agente, a través de la tool `buscar_documentacion`, que alimenta al LLM (Gemini) con el contexto recuperado para generar la respuesta final.

## 5. Relaciones entre datos

El proyecto utiliza un **único dataset**: la colección vectorial de Chroma. No existen múltiples tablas ni fuentes estructuradas que combinar, por lo que no hay relaciones 1:1, 1:N ni N:M que modelar, ni joins o agregaciones entre datasets.

**Justificación de por qué no se necesita un modelo relacional más complejo:** el proyecto no combina datos tabulares de distintas fuentes (como sí ocurriría, por ejemplo, en un modelo con tablas de `clientes`, `productos` y `ventas`), sino que opera sobre un corpus documental único, cuya única relación relevante es implícita: cada chunk se vincula a su documento de origen a través de `metadata.source` y `metadata.page`, únicamente para poder citar la fuente en la respuesta — no para hacer cruces analíticos entre fuentes.

## 6. Diccionario de datos inicial

| Campo | Descripción | Tipo | Fuente | Obligatorio | Observaciones |
|---|---|---|---|---|---|
| `page_content` | Texto del chunk | str | PDFs originales | Sí | Resultado del `RecursiveCharacterTextSplitter` |
| `embedding` | Vector semántico del chunk | float[3072] | Gemini Embeddings (`gemini-embedding-001`) | Sí | Generado automáticamente al indexar |
| `metadata.source` | Fichero PDF de origen | str | `PyPDFDirectoryLoader` | Sí | Se usa para citar la fuente en las respuestas |
| `metadata.page` | Página de origen dentro del PDF | int | `PyPDFDirectoryLoader` | Sí | Se usa junto con `source` para citar |

## 7. Problemas de calidad esperados

- **Chunks que cortan a mitad de un artículo legal:** al trocear el texto en fragmentos de tamaño fijo, un artículo o apartado puede quedar dividido entre dos chunks, perdiendo parte del contexto. Se mitiga parcialmente con `chunk_overlap=200`.
- **Ruido de cabeceras y pies de página repetidos:** los PDFs oficiales suelen repetir encabezados, numeración de página o pies de página en cada hoja, lo que introduce texto irrelevante en varios chunks.
- **Falta de estructura semántica explícita:** tras el chunking por caracteres, se pierde la jerarquía original del documento (artículos, apartados, incisos), lo que puede dificultar que el retriever identifique el nivel exacto de una disposición legal.
- **Desactualización normativa:** si la normativa o la jurisprudencia cambian tras la indexación, la base de conocimiento quedaría desactualizada sin que el sistema lo detecte automáticamente.

## 8. Decisiones de limpieza y transformación previstas

- **Chunking:** `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200`, priorizando mantener el contexto legal lo más completo posible frente a chunks más pequeños y fragmentados.
- **Sin eliminación agresiva de texto:** no se aplican filtros de limpieza adicionales (no se eliminan cabeceras/pies de página de forma automática) para evitar perder contenido legal relevante por error; se acepta cierto ruido residual como compromiso.
- **Variable derivada:** `metadata.source` (y `metadata.page`) se usan para construir la cita de la fuente que el agente incluye en sus respuestas.
- **Criterio de registro válido:** un chunk se considera válido si contiene texto no vacío tras la extracción del PDF; los documentos que el loader no consigue extraer (por ejemplo, PDFs escaneados sin OCR) quedarían fuera y deberían detectarse manualmente antes de indexar.

## 9. Riesgos del modelo de datos

- **Parte más clara:** la indexación y la estructura del retriever ya están implementadas y probadas — la recuperación de fragmentos relevantes funciona correctamente sobre las preguntas de prueba.
- **Parte que genera más incertidumbre:** la calidad de la respuesta ante preguntas ambiguas, preguntas fuera del corpus indexado, o preguntas que requieren combinar información de varias fuentes a la vez.
- **Fuente o tabla que puede dar más problemas:** la sentencia Google Spain vs AEPD, al ser un texto jurisprudencial más largo y con estructura argumentativa compleja, es más propensa a que el chunking corte razonamientos a la mitad.
- **Qué ocurriría si no se puede construir la capa gold tal como está definida:** se podría simplificar a una búsqueda por palabras clave (full-text search) sobre los documentos, renunciando a la búsqueda semántica, como mecanismo de respaldo (fallback).
- **Alternativa para simplificar el modelo si fuera necesario:** reducir la base de conocimiento a un único documento (el texto del RGPD) si las demás fuentes generan demasiado ruido o problemas de indexación, sacrificando cobertura por fiabilidad.
