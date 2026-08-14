# Ideas de Proyecto Final

> **Nota sobre esta revisión:** el profesor señala que, tal como estaban planteadas, las tres ideas se apoyan sobre todo en datos no estructurados (texto legal en lenguaje natural), lo que las acerca más a un proyecto de IA Generativa/NLP que a un proyecto de datos estructurados con SQL. Su sugerencia es no forzar el dato hacia SQL desde el principio, sino aprovechar que un LLM se desenvuelve mejor en ese tipo de texto: usarlo para **transformar el texto no estructurado en features estructuradas**, y sobre esas features sí construir un modelo de predicción. Esta revisión desarrolla la **idea 1** siguiendo ese enfoque híbrido, que es el hilo que se ha seguido después en las entregas de datos necesarios y modelo de datos. Las ideas 2 y 3 se mantienen como alternativas descartadas, señalando por qué comparten la misma limitación.

---

## 1. Sistema de análisis de probabilidad de éxito en casos de Derecho al Olvido Digital *(idea seleccionada)*

### Problema o necesidad que aborda

El derecho al olvido digital es un derecho relativamente reciente dentro de la normativa europea de protección de datos, y su aplicación depende de múltiples factores como la relevancia pública de la persona afectada, la naturaleza de la información publicada o el tiempo transcurrido desde los hechos.

Además, este derecho se encuentra en una zona limítrofe con otros derechos fundamentales, especialmente con el derecho a la libertad de información, lo que implica que muchas decisiones judiciales se basan en complejos ejercicios de ponderación entre derechos fundamentales.

Para los abogados, analizar estos factores y compararlos con jurisprudencia previa puede ser un proceso largo y complejo.

### Impacto o valor potencial

El proyecto consiste en crear un sistema de análisis basado en datos que estudie casos previos de derecho al olvido a partir de resoluciones judiciales o decisiones de autoridades de protección de datos, e identifique patrones en los casos (por ejemplo, qué características tienen los casos que terminan estimándose o desestimándose) para generar una estimación de probabilidad de éxito orientada a nuevos casos.

**Enfoque técnico revisado — por qué no se fuerza el dato hacia SQL desde el origen:**

El punto de partida real del proyecto no son filas y columnas, sino el texto de cada resolución: hechos, fundamentos jurídicos y fallo, redactados en lenguaje natural jurídico. Intentar estructurar ese texto directamente con SQL desde la extracción supondría, o bien un etiquetado manual muy costoso caso por caso, o bien reglas de extracción rígidas (expresiones regulares, patrones fijos) que funcionan mal ante la variabilidad de redacción entre resoluciones. Siguiendo la observación de que un LLM se desenvuelve mucho mejor que un enfoque basado en reglas sobre este tipo de dato, el proyecto adopta un **pipeline en dos fases**:

1. **Extracción asistida por LLM (fase NLP):** un modelo de lenguaje procesa el texto de cada resolución y extrae, de forma semiestructurada, el valor de cada variable relevante (tipo de solicitante, tipo de responsable del tratamiento, antigüedad de la información, tipo de contenido, existencia de condena penal previa, etc.), siguiendo un criterio de etiquetado cerrado y verificable por revisión humana — el LLM no sustituye el criterio jurídico, lo aplica de forma asistida sobre el texto.
2. **Modelado predictivo sobre features estructuradas (fase Data Science):** una vez extraídas y validadas, estas variables sí se organizan en una tabla estructurada (features + variable objetivo `resultado`), sobre la que se entrena y evalúa un modelo de clasificación supervisada (regresión logística como baseline, con Random Forest/XGBoost como comparación), con métricas estándar y explicabilidad vía SHAP.

Esta secuencia evita dos errores simétricos: no se intenta forzar SQL sobre un dato que en origen es texto libre y heterogéneo, pero tampoco se deja el proyecto únicamente en el terreno de la IA Generativa (un LLM respondiendo preguntas sobre documentos). El LLM se usa como herramienta de **preprocesamiento** para convertir texto no estructurado en dato tabular, y es sobre ese dato tabular donde se construye el componente de Data Science evaluable del proyecto: la predicción y su explicación.

Un componente complementario de recuperación semántica (RAG) puede apoyar este pipeline mostrando, junto a la predicción, precedentes similares recuperados del mismo corpus — pero como capa de apoyo del modelo predictivo, no como el producto final del proyecto.

### Motivación personal

Este tema está directamente relacionado con mi Trabajo de Fin de Grado, en el que analicé la evolución del derecho al olvido digital en Europa y su relación con otros derechos fundamentales.

Además, actualmente estoy formándome en inteligencia artificial y análisis de datos, por lo que me interesa explorar cómo las herramientas de análisis de datos —combinadas con LLMs para el procesamiento del lenguaje natural jurídico— pueden aplicarse al ámbito jurídico para mejorar el estudio de la jurisprudencia.

Cabe señalar que esta idea se plantea en una fase temprana del máster (segundo módulo), por lo que es posible que durante el desarrollo del programa adquiera nuevas herramientas técnicas que permitan ampliar o perfeccionar el alcance del proyecto.

---

## 2. Análisis de patrones en resoluciones de protección de datos en Europa *(descartada)*

Comparte la misma limitación de origen que la idea 1: el punto de partida son resoluciones en texto libre, no una base de datos tabular ya existente. Plantear el análisis directamente con SQL y pandas, como se proponía inicialmente, implicaría el mismo problema de extracción rígida sobre texto heterogéneo, sin que el proyecto llegue a apoyarse en un LLM para resolverlo. Se descarta en favor de la idea 1, que ya incorpora el enfoque híbrido LLM + features estructuradas y tiene además una motivación personal más directa (conexión con el TFG).

## 3. Sistema de clasificación de contenidos potencialmente afectados por el derecho al olvido *(descartada)*

Misma limitación: los criterios de clasificación (antigüedad, relevancia pública, tipo de fuente, naturaleza del contenido) solo son extraíbles de forma fiable a partir de texto no estructurado, por lo que el proyecto necesitaría el mismo paso previo de extracción asistida por LLM antes de poder trabajar con datasets estructurados. Se descarta por solaparse en esencia con la idea 1 sin aportar un problema adicional distinto.
