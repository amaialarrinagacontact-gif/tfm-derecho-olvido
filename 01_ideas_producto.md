# Entrega 1 — Ideas de producto / proyecto de Data Science

*Autora: Amaia Gil Pérez — Máster en Data Science, ML e IA — Evolve Academy 2025-2026*

---

## Contexto personal

Mi formación de base es en Derecho y Comunicación, con especialización en derecho digital. Mi Trabajo de Fin de Grado analizó la evolución del derecho al olvido digital en Europa y su relación con otros derechos fundamentales. Actualmente me estoy formando en inteligencia artificial y análisis de datos, y me interesa explorar cómo las herramientas de Data Science pueden aplicarse al ámbito jurídico para mejorar el estudio de la jurisprudencia y facilitar el acceso a la justicia digital.

Las tres ideas propuestas parten de esta intersección entre derecho digital y análisis de datos, y se plantean en una fase temprana del máster (segundo módulo), por lo que es posible que durante el desarrollo del programa se adquieran nuevas herramientas técnicas que permitan ampliar o perfeccionar su alcance.

---

## Idea 1 — Sistema de análisis de probabilidad de éxito en casos de Derecho al Olvido Digital

### Problema o necesidad que aborda

El derecho al olvido digital es un derecho relativamente reciente dentro de la normativa europea de protección de datos, cuya aplicación depende de múltiples factores: la relevancia pública de la persona afectada, la naturaleza de la información publicada, el tipo de responsable del tratamiento o el tiempo transcurrido desde los hechos. Además, este derecho se encuentra en una zona limítrofe con otros derechos fundamentales, especialmente con el derecho a la libertad de información, lo que implica que muchas decisiones judiciales y administrativas se basan en complejos ejercicios de ponderación entre derechos fundamentales cuyos resultados son difícilmente predecibles.

Para los abogados, ciudadanos y operadores jurídicos, analizar estos factores y compararlos con jurisprudencia previa puede ser un proceso largo, costoso y con un resultado incierto. Esta incertidumbre genera una asimetría de información entre los afectados y las grandes plataformas tecnológicas, que disponen de criterios internos de evaluación que no son públicos.

### Impacto o valor potencial

El proyecto consistiría en crear un sistema de análisis basado en datos que permita estudiar casos previos de derecho al olvido utilizando datasets de resoluciones de autoridades de protección de datos (como la AEPD) y sentencias de tribunales europeos (TJUE, TEDH). Mediante el uso de SQL para organizar la base de datos y Python con pandas para analizar los datos, el sistema podría identificar patrones en los casos (por ejemplo, qué características tienen los casos que terminan estimándose o desestimándose) y generar una estimación de probabilidad de éxito para nuevos casos.

Esta herramienta podría servir como apoyo para abogados o investigadores que trabajan en derecho digital, ayudándoles a realizar un análisis preliminar más rápido de la viabilidad de un caso antes de iniciar un procedimiento formal ante la AEPD o los tribunales.

### Motivación personal

Este tema está directamente relacionado con mi Trabajo de Fin de Grado, en el que analicé la evolución del derecho al olvido digital en Europa y su relación con otros derechos fundamentales. Además, actualmente me estoy formando en inteligencia artificial y análisis de datos, por lo que me interesa explorar cómo las herramientas de análisis de datos pueden aplicarse al ámbito jurídico para mejorar el estudio de la jurisprudencia y contribuir al acceso a la justicia digital.

---

## Idea 2 — Análisis de patrones en resoluciones de protección de datos en Europa

### Problema o necesidad que aborda

Las autoridades de protección de datos europeas (como la AEPD en España o el Comité Europeo de Protección de Datos a nivel europeo) emiten cada año numerosas resoluciones relacionadas con la protección de datos personales. Sin embargo, estas resoluciones suelen estar dispersas en distintos portales institucionales y no siempre es fácil identificar tendencias o patrones en las decisiones: qué tipos de infracciones son más frecuentes, qué sectores están más expuestos, cómo evolucionan las sanciones a lo largo del tiempo o cómo varía la aplicación del RGPD entre distintos Estados miembros.

Esta falta de análisis agregado dificulta tanto la comprensión académica del funcionamiento del RGPD en la práctica como la toma de decisiones estratégicas por parte de empresas y organizaciones que necesitan gestionar su riesgo regulatorio.

### Impacto o valor potencial

Este proyecto propondría construir un dataset con resoluciones de autoridades de protección de datos europeas y analizarlas utilizando SQL y pandas para detectar patrones, como por ejemplo:

- Tipos de infracciones más comunes por categoría y por país
- Sectores económicos más afectados por las sanciones
- Evolución temporal de las cuantías de las sanciones desde la entrada en vigor del RGPD en 2018
- Diferencias entre jurisdicciones nacionales en la aplicación de los mismos artículos del RGPD

El análisis podría ayudar a comprender mejor cómo se está aplicando el RGPD en la práctica y cuáles son las áreas de mayor riesgo legal para empresas y organizaciones, con potencial de uso tanto académico como profesional.

### Motivación personal

Mi formación en derecho digital y comunicación me ha permitido observar cómo la regulación de datos personales tiene cada vez más impacto en empresas, medios de comunicación y plataformas digitales. Me interesa analizar estos datos desde una perspectiva empírica utilizando herramientas de análisis de datos, combinando el conocimiento jurídico del RGPD con las técnicas cuantitativas aprendidas en el máster.

---

## Idea 3 — Sistema de clasificación de contenidos potencialmente afectados por el derecho al olvido

### Problema o necesidad que aborda

Muchas personas desconocen cuándo una información publicada en internet podría vulnerar su derecho a la privacidad o cuándo podría solicitarse su eliminación a través del derecho al olvido. La complejidad del marco normativo (artículo 17 RGPD, jurisprudencia del TJUE, criterios de las autoridades de control) hace difícil para un ciudadano no especializado valorar si un contenido concreto —una noticia antigua, una foto, un dato judicial— es susceptible de ser retirado. Esto dificulta la defensa autónoma de los derechos digitales de las personas.

### Impacto o valor potencial

El proyecto consistiría en crear un sistema que clasifique diferentes tipos de contenidos online (noticias, artículos o publicaciones) según criterios relevantes para el derecho al olvido, como:

- Antigüedad de la información
- Relevancia pública de la persona afectada (persona pública o privada)
- Tipo de fuente (hemeroteca, motor de búsqueda, red social, registro público)
- Naturaleza del contenido (dato judicial, dato de salud, imagen, información económica)

Utilizando datasets estructurados, SQL y análisis con pandas, el sistema podría ayudar a identificar qué tipos de contenido tienen mayor probabilidad de ser considerados susceptibles de eliminación, funcionando como una herramienta de orientación inicial para ciudadanos que deseen ejercer su derecho al olvido.

### Motivación personal

Este proyecto combina mis intereses en derecho digital, comunicación y análisis de datos. Conecta directamente con mi experiencia académica en el estudio del derecho al olvido y con mi interés en comprender cómo la tecnología puede ayudar a proteger los derechos digitales de los ciudadanos, especialmente de aquellos que no tienen acceso a asesoramiento jurídico especializado.

---

## Valoración comparativa de las tres ideas

| Criterio | Idea 1 (Probabilidad de éxito) | Idea 2 (Patrones RGPD) | Idea 3 (Clasificación contenidos) |
|---|---|---|---|
| Originalidad | Alta — no existe sistema equivalente en la literatura revisada | Media — hay trabajos similares de análisis de sanciones RGPD | Media — enfoque ciudadano diferenciador |
| Viabilidad de datos | Media — requiere etiquetado manual de resoluciones | Alta — GDPRhub ofrece datos pre-estructurados | Media — requiere definir criterios de clasificación |
| Impacto potencial | Alto — herramienta de acceso a la justicia | Medio-alto — valor para compliance empresarial | Medio — orientada al ciudadano no especializado |
| Complejidad técnica | Alta — clasificación + regresión + NLP semántico | Media — análisis exploratorio + visualización | Media — clasificación por reglas + ML supervisado |
| Alineación con el TFM | Muy alta — es el núcleo del TFM en desarrollo | Media | Media |

**Idea seleccionada para continuar: Idea 1**, por su mayor originalidad, impacto potencial y alineación con el trabajo académico en curso. Las ideas 2 y 3 pueden integrarse como análisis complementarios dentro del mismo proyecto (el análisis de patrones de la Idea 2 forma parte del EDA del sistema principal; la clasificación de contenidos de la Idea 3 es un módulo derivado natural).

---

*Nota: Esta entrega se plantea en el segundo módulo del máster. Es posible que durante el desarrollo del programa se adquieran nuevas herramientas técnicas —especialmente en el ámbito de los modelos de lenguaje y la IA generativa— que permitan ampliar o perfeccionar el alcance técnico del proyecto.*
