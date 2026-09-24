# Asistente Experto en Derecho al Olvido Digital

Agente conversacional basado en RAG (Retrieval-Augmented Generation) que responde preguntas sobre el derecho al olvido digital (derecho de supresión de datos personales) utilizando documentación oficial del RGPD, guías de la AEPD y jurisprudencia relevante como base de conocimiento.

El proyecto combina **Google Gemini** (LLM y embeddings), **ChromaDB** (base de datos vectorial) y **LangGraph** (framework de agentes) para construir un asistente con memoria conversacional que cita sus fuentes y admite cuando no tiene información suficiente para responder.

## Descripción del dominio elegido

El derecho al olvido digital, recogido en el **artículo 17 del RGPD**, permite a cualquier persona solicitar la supresión de sus datos personales bajo determinadas circunstancias. Es un dominio jurídico específico, bien documentado por fuentes oficiales, y con preguntas frecuentes claras (qué es, cómo se ejerce, qué excepciones existen, qué jurisprudencia lo sustenta), lo que lo hace idóneo para un sistema de RAG: las respuestas deben ser precisas, verificables y basadas en normativa vigente, no en generación libre del modelo.

### Base de conocimiento

El agente indexa los siguientes documentos:

| Documento | Contenido |
|---|---|
| Artículo 17 del RGPD | Texto oficial del derecho de supresión |
| Reglamento (UE) 2016/679 completo (CELEX) | Texto íntegro del RGPD, incluyendo considerandos relevantes (66, 67) |
| Guía / texto consolidado de protección de datos | Contexto normativo adicional sobre supresión y excepciones |

En total, la base indexada abarca **277 páginas**, divididas en aproximadamente **1.500 fragmentos (chunks)** de texto.

## Stack tecnológico

- **LLM y Embeddings**: Google Gemini (`gemini-2.5-flash` / `gemini-2.5-flash-lite` para generación, `gemini-embedding-001` para embeddings)
- **Base de conocimiento vectorial**: ChromaDB (persistida localmente en `./chroma_db`)
- **Framework de agente**: LangGraph, sobre LangChain
- **Entorno de trabajo**: Jupyter Notebook (VS Code)
- **Interfaz web (bonus)**: Streamlit

## Instrucciones de instalación y ejecución

### 1. Requisitos previos

- Python 3.11 o superior
- Una API key gratuita de Google Gemini, obtenida en [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Clonar el repositorio e instalar dependencias

```bash
git clone <url-de-tu-repositorio>
cd proyecto_asistente_experto

python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configurar la API key

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
GOOGLE_API_KEY=tu_clave_aqui
```

> ⚠️ El archivo `.env` nunca debe subirse al repositorio (ya está incluido en `.gitignore`).

### 4. Ejecutar el notebook

Abre `proyecto.ipynb` en VS Code o Jupyter y ejecuta las celdas en orden. La primera vez, la celda de indexación creará la base vectorial en `./chroma_db`; en ejecuciones posteriores, esa base se carga directamente desde disco sin necesidad de volver a indexar.

### 5. (Opcional) Ejecutar la interfaz Streamlit

```bash
streamlit run app.py
```

## Justificación del system prompt

El agente utiliza el siguiente system prompt:

```
Eres un asistente experto en el derecho al olvido digital (derecho de supresión de
datos personales) en el marco del Reglamento General de Protección de Datos (RGPD)
y la normativa española de protección de datos.

Tu comportamiento debe seguir estas reglas:

1. Responde SIEMPRE basándote en la información recuperada de la base de conocimiento
   (RGPD, guías de la AEPD, jurisprudencia relevante). No inventes artículos, sentencias
   ni datos que no estén respaldados por los documentos indexados.

2. Si la información disponible no es suficiente para responder con seguridad,
   dilo explícitamente ("No dispongo de información suficiente sobre esto en mi
   base de conocimiento") en lugar de especular o inventar una respuesta.

3. Cuando sea posible, menciona de qué documento o artículo proviene la información.

4. Usa un tono claro, profesional y accesible, evitando jerga jurídica innecesaria,
   pero sin perder precisión técnica.

5. Aclara, cuando sea relevante, que tus respuestas son informativas y de carácter
   divulgativo, y que no sustituyen el asesoramiento de un abogado especializado.

6. Mantén coherencia con el contexto de la conversación.
```

**Decisiones de diseño:**

- **Restricción a la base de conocimiento (regla 1 y 2)**: en un dominio legal, una respuesta inventada ("alucinación") es más perjudicial que admitir falta de información. Se prioriza la fiabilidad sobre la exhaustividad.
- **Citación de fuentes (regla 3)**: aporta trazabilidad y permite al usuario verificar la información en la norma original.
- **Tono accesible (regla 4)**: el RGPD y sus excepciones tienen redacción densa; el objetivo del asistente es divulgar, no sustituir el texto legal.
- **Disclaimer legal (regla 5)**: dado que el dominio es normativo, es una buena práctica dejar claro que no se trata de asesoramiento jurídico vinculante.
- **Temperatura baja (0.2)**: se prioriza la precisión y la consistencia frente a la creatividad, adecuado para un caso de uso donde la exactitud normativa importa más que la originalidad.

## Arquitectura del agente

```
Pregunta del usuario
        │
        ▼
   Agente (LangGraph)
        │
        ├── ¿Necesita información específica? → Sí → herramienta `buscar_documentacion`
        │                                              │
        │                                              ▼
        │                                    Retriever (ChromaDB, k=4)
        │                                              │
        │                                              ▼
        │                                    Fragmentos relevantes del RGPD/AEPD
        │
        ▼
   Respuesta generada (Gemini) + memoria de conversación (thread_id)
```

El agente decide de forma autónoma cuándo consultar la base de conocimiento (agentic RAG), en lugar de recuperar contexto en cada turno de forma automática.

## Memoria de conversación

El agente mantiene coherencia entre turnos gracias a un `checkpointer` (`InMemorySaver`) asociado a un `thread_id`. Ejemplo demostrado en el notebook:

> **P1:** ¿Qué es el derecho al olvido según el RGPD?
> **P2:** ¿Y qué excepciones tiene *ese derecho* que acabas de mencionar?

En la segunda pregunta, el usuario no especifica de nuevo a qué derecho se refiere, y el agente responde correctamente basándose en el contexto de la conversación anterior.

## Ejemplos de preguntas documentadas

El notebook incluye, entre otras, las siguientes preguntas de ejemplo con sus respuestas completas:

1. ¿Qué es el derecho al olvido digital?
2. ¿Qué artículo del RGPD regula el derecho de supresión y qué dice exactamente?
3. ¿Y qué excepciones existen a ese derecho que acabas de mencionar? *(demuestra memoria)*
4. ¿Qué relación tiene la sentencia Google Spain contra la AEPD con el derecho al olvido?
5. ¿Cómo se aplica el derecho al olvido digital en Estados Unidos? *(demuestra que el agente admite no tener información suficiente en lugar de inventar una respuesta)*
6. ¿En qué situaciones puede una persona solicitar la eliminación de sus datos personales?

## Limitaciones conocidas

- La base de conocimiento no incluye el texto íntegro de la sentencia *Google Spain vs. AEPD* (C-131/12); el agente responde sobre ella combinando lo indexado con conocimiento general, y lo indica explícitamente.
- El nivel gratuito de la API de Gemini tiene límites diarios de peticiones bastante restrictivos, lo que puede requerir espaciar las consultas o activar facturación (Cloud Billing) para un uso más intensivo, como una demo en vivo.
- Las respuestas dependen de la calidad del chunking (`chunk_size=1000`, `overlap=200`); artículos muy largos podrían fragmentarse de forma subóptima en casos concretos.

## Estructura del repositorio

```
proyecto_asistente_experto/
├── data/
│   └── derecho_olvido/       # Documentos PDF originales (RGPD, guías, jurisprudencia)
├── chroma_db/                 # Base de datos vectorial persistida
├── proyecto.ipynb             # Notebook principal con el desarrollo completo
├── app.py                     # Interfaz Streamlit (bonus)
├── requirements.txt
├── .env                        # API key (no incluido en el repositorio)
├── .gitignore
└── README.md
```

## Requisitos (dependencias)

```
langchain
langchain-community
langchain-google-genai
langchain-chroma
langchain-huggingface
langgraph
chromadb
python-dotenv
jupyter
streamlit
tenacity
pypdf
langchain-text-splitters
```

## Autor

Proyecto desarrollado como entrega final del módulo de IA Generativa.
