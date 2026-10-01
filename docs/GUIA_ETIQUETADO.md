# Guía de etiquetado (anexo del Datasheet for Datasets)

Versión 1.0. Aplica a cada resolución de la AEPD incluida en el corpus. Objetivo: que cualquier
persona que etiquete el mismo expediente llegue a la misma etiqueta.

## 1. Unidad de observación e inclusión

Una fila = una resolución de tutela o procedimiento de derechos (TD/…, PD/…, EXP…) de la AEPD
en la que se pide **borrar o desindexar información personal publicada en internet**
(supresión, art. 17 RGPD; en la etapa LOPD, cancelación u oposición).

**No se incluyen:** derechos de acceso o rectificación; supresión en bases de datos internas
(clientes, empleados, administraciones); procedimientos sancionadores (PS/…); archivos de
actuaciones (E/…). **Los recursos de reposición** no son filas nuevas: si modifican el resultado,
se anota en la fila de la resolución original.

## 2. Variable objetivo `resultado`

Se lee del **fallo**, teniendo en cuenta el motivo expresado al final de los fundamentos.

| Etiqueta | Regla |
|---|---|
| `estimada` | La AEPD pondera y ordena la supresión o desindexación de todo lo pedido. |
| `estimada_parcialmente` | Ordena solo parte (algunas URL, algunos canales) o con condiciones. |
| `desestimada` | Pondera y deniega: prevalece la libertad de información, el interés público, etc. |
| `excluir` | El fallo no refleja una ponderación. Ver casos abajo. |

**Se excluyen (el fallo no dice nada sobre la ponderación):**
1. Estimación **solo por motivos formales** (el responsable contestó fuera de plazo).
2. "ESTIMAR e **instar a certificar que se ha atendido el derecho o a denegarlo motivadamente**":
   la AEPD no decide sobre el fondo.
3. Desestimación **sin objeto**: lo pedido ya se había obtenido (p. ej., la URL ya no aparece).
4. Desestimación **procedimental**: no se acredita haber ejercido el derecho.
5. Inadmisión, archivo o desistimiento.

## 3. Variables de entrada

**Regla de oro (anti-fuga de información): se extraen SOLO de los HECHOS** (texto anterior a
"FUNDAMENTOS DE DERECHO"). Deben ser datos que un ciudadano conoce antes de reclamar.

| Variable | Regla | Criterio WP225 |
|---|---|---|
| `tipo_solicitante` | `publica` si es cargo electo, alto cargo o figura con proyección pública reconocida en el texto; si no o ante la duda, `privada`. | 2. Papel en la vida pública |
| `tipo_responsable` | Quién es la reclamada en el expediente, no el origen de la información. Si hay varias y una queda inadmitida, la que sigue en el procedimiento. | — |
| `tipo_contenido` | `noticia`, `imagen`, `dato_judicial`, `dato_registral`, `otro`. Blogs de opinión y foros: `otro`. Edictos y anuncios judiciales: `dato_judicial`. | 6, 11, 12 |
| `antiguedad_info_años` | Años entre la publicación y la solicitud. Vacío si alguna fecha no consta o está anonimizada. Nunca se estima. | 7. Actualidad |
| `condena_penal_previa` | 1 solo con condena penal **firme** relacionada. Investigación, archivo o acusación en un blog: 0. | 13. Infracción penal |
| `persona_fallecida` | 1 si los datos son de una persona fallecida. | — |
| `interes_historico` | 1 solo si el responsable invoca expresamente fines de archivo, históricos o estadísticos. | 17.3.d RGPD |
| `alega_libertad_informacion` | 1 si el responsable deniega invocando la libertad de información o expresión o el interés público. | 11. Fines periodísticos |

## 4. Criterios del WP225 no modelados (limitación documentada)

Menor de edad (3), exactitud (4), vida profesional (5a), injurias o calumnias (5b),
opinión frente a hecho (5c), perjuicio (8), riesgo (9) y publicación voluntaria (10).
Con el volumen actual, añadir variables empeoraría el modelo. Los criterios 4, 5a y 5b aparecen
en el corpus real (TD/00071/2020, TD/00006/2020, TD/00277/2020): son candidatos prioritarios
para ampliar el modelo cuando crezca el corpus.

## 5. Ejemplos resueltos (corpus real)

| Expediente | Etiqueta | Por qué |
|---|---|---|
| TD/00006/2020 | desestimada | Noticia; la AEPD aprecia interés público y remite el honor a la vía civil. |
| TD/00071/2020 | desestimada | Información profesional en webs institucionales. |
| TD/00277/2020 | desestimada | Blog de opinión amparado por la libertad de expresión. |
| TD/00173/2021 | excluir (3) | El BOE desindexó durante el procedimiento. |
| TD/00264/2019 | no se incluye | Derecho de acceso. |
| E/01278/2018 | no se incluye | Archivo de actuaciones sobre una cuenta de cliente. |
