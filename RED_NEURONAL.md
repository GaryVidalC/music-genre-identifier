# Red neuronal para clasificación de géneros

Arquitectura inicial para canciones completas de MTG-Jamendo. Cada punto es
un miniobjetivo: definirlo, implementarlo y verificarlo antes de marcarlo.
Los parámetros son un punto de partida, no una configuración óptima garantizada.

## Miniobjetivos

- [ ] **1. Dividir la canción en snippets.** Fragmentos de 15 segundos;
  definir el tratamiento del último fragmento y sus máscaras de padding.
- [ ] **2. Generar log-mel por snippet.** Audio mono y 64 bandas mel;
  fijar sample rate, ventana, salto y normalización para entrenamiento e inferencia.
- [ ] **3. Extraer patrones con una CNN compartida.** Tres bloques de
  32 → 64 → 128 canales, reduciendo resolución temporal y frecuencial.
- [ ] **4. Promediar sobre frecuencias.** Convertir la salida de la CNN en
  una secuencia temporal `[T, 128]` por snippet.
- [ ] **5. Añadir codificación posicional.** Representar la posición temporal
  de los tokens dentro de cada snippet.
- [ ] **6. Aplicar el Transformer Encoder.** Por snippet: 2 capas, 4 heads,
  `d_model=128`, feedforward de 256 y dropout de 0.1; excluir tokens de padding.
- [ ] **7. Obtener el embedding del snippet.** Promedio temporal con máscara
  para producir un vector `[128]`, sin incluir padding.
- [ ] **8. Combinar snippets con attention pooling.** Aprender sus pesos y
  producir un embedding global `[128]`, excluyendo snippets de padding.
- [ ] **9. Construir la cabeza de clasificación.** Dropout y capa lineal con
  una salida por género; fijar el listado y orden de las etiquetas.
- [ ] **10. Obtener probabilidades con sigmoid.** Clasificación multietiqueta:
  una canción puede pertenecer a varios géneros.

El Transformer relaciona patrones dentro de cada snippet. Attention pooling
combina los snippets de la canción. Las etiquetas corresponden a la canción
completa, no a cada fragmento por separado.

Entrenamiento y despliegue se definirán después; este checklist cubre la arquitectura.
