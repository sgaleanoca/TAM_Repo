# Reporte de Pruebas Experimentales — Imágenes Manuscritas Propias

Este documento registra los resultados de evaluar el pipeline de preprocesamiento,
segmentación e inferencia con el modelo original (`models/mnist_mlp.keras`) sobre imágenes
manuscritas sintéticas externas (no pertenecientes al conjunto de entrenamiento ni prueba de MNIST).

## Resumen Cuantitativo

- **Casos manuscritos evaluados (0–99)**: `24`
- **Predicciones exactas del número completo**: `23 / 24` (`95.8%`)
- **Casos de validación de errores (lienzo vacío / >2 dígitos)**: `2 / 2` capturados correctamente.

## Resultados Detallados por Categoría

| Archivo | Categoría | Descripción | Esperado | Obtenido | Confianza Softmax Conjunta | Estado / Observaciones |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `single_digit_0.png` | 1. Dígitos individuales | Dígito individual '0' en trazo negro sobre fondo blanco. | **0** | **0** | 99.96% | ✅ Correcto — D1=0 (100.0%) |
| `single_digit_3.png` | 1. Dígitos individuales | Dígito individual '3' en trazo negro sobre fondo blanco. | **3** | **3** | 100.00% | ✅ Correcto — D1=3 (100.0%) |
| `single_digit_5.png` | 1. Dígitos individuales | Dígito individual '5' en trazo negro sobre fondo blanco. | **5** | **5** | 100.00% | ✅ Correcto — D1=5 (100.0%) |
| `single_digit_7.png` | 1. Dígitos individuales | Dígito individual '7' en trazo negro sobre fondo blanco. | **7** | **7** | 100.00% | ✅ Correcto — D1=7 (100.0%) |
| `single_digit_8.png` | 1. Dígitos individuales | Dígito individual '8' en trazo negro sobre fondo blanco. | **8** | **8** | 99.59% | ✅ Correcto — D1=8 (99.6%) |
| `single_digit_9.png` | 1. Dígitos individuales | Dígito individual '9' en trazo negro sobre fondo blanco. | **9** | **9** | 100.00% | ✅ Correcto — D1=9 (100.0%) |
| `two_digits_47.png` | 2. Números de dos cifras | Número de dos cifras '47' con separación estándar. | **47** | **47** | 74.11% | ✅ Correcto — D1=4 (74.1%), D2=7 (100.0%) | Nota: Confianza moderada/baja en Decenas (Izquierda): 74.1% para el dígito '4'. Verifique las probabilidades secundarias. |
| `two_digits_12.png` | 2. Números de dos cifras | Número de dos cifras '12' con separación estándar. | **12** | **12** | 100.00% | ✅ Correcto — D1=1 (100.0%), D2=2 (100.0%) |
| `two_digits_25.png` | 2. Números de dos cifras | Número de dos cifras '25' con separación estándar. | **25** | **25** | 100.00% | ✅ Correcto — D1=2 (100.0%), D2=5 (100.0%) |
| `two_digits_68.png` | 2. Números de dos cifras | Número de dos cifras '68' con separación estándar. | **68** | **68** | 99.55% | ✅ Correcto — D1=6 (100.0%), D2=8 (99.6%) |
| `two_digits_90.png` | 2. Números de dos cifras | Número de dos cifras '90' con separación estándar. | **90** | **90** | 99.97% | ✅ Correcto — D1=9 (100.0%), D2=0 (100.0%) |
| `two_digits_34.png` | 2. Números de dos cifras | Número de dos cifras '34' con separación estándar. | **34** | **34** | 74.11% | ✅ Correcto — D1=3 (100.0%), D2=4 (74.1%) | Nota: Confianza moderada/baja en Unidades (Derecha): 74.1% para el dígito '4'. Verifique las probabilidades secundarias. |
| `size_small_42.png` | 3. Diferentes tamaños de escritura | Escritura pequeña (48px alto) | **42** | **42** | 99.66% | ✅ Correcto — D1=4 (99.7%), D2=2 (100.0%) |
| `size_large_85.png` | 3. Diferentes tamaños de escritura | Escritura grande (160px alto) | **85** | **85** | 99.78% | ✅ Correcto — D1=8 (99.8%), D2=5 (100.0%) |
| `size_tiny_single_6.png` | 3. Diferentes tamaños de escritura | Dígito individual pequeño (40px alto) | **6** | **6** | 100.00% | ✅ Correcto — D1=6 (100.0%) |
| `stroke_thin_73.png` | 4. Diferentes grosores de trazos | Trazo fino de bolígrafo (3px) | **73** | **73** | 100.00% | ✅ Correcto — D1=7 (100.0%), D2=3 (100.0%) |
| `stroke_medium_59.png` | 4. Diferentes grosores de trazos | Trazo medio estándar (9px) | **59** | **59** | 100.00% | ✅ Correcto — D1=5 (100.0%), D2=9 (100.0%) |
| `stroke_thick_24.png` | 4. Diferentes grosores de trazos | Trazo grueso de marcador (15px) | **24** | **24** | 99.89% | ✅ Correcto — D1=2 (100.0%), D2=4 (99.9%) |
| `bg_dark_white_ink_38.png` | 5. Diferentes fondos y contrastes | Fondo oscuro tipo pizarra con trazo blanco | **38** | **38** | 99.69% | ✅ Correcto — D1=3 (100.0%), D2=8 (99.7%) |
| `bg_paper_texture_64.png` | 5. Diferentes fondos y contrastes | Fondo tipo papel con gradiente de iluminación y ruido | **64** | **64** | 99.94% | ✅ Correcto — D1=6 (100.0%), D2=4 (99.9%) |
| `bg_low_contrast_19.png` | 5. Diferentes fondos y contrastes | Contraste reducido (trazo gris medio sobre fondo gris claro) | **19** | **19** | 100.00% | ✅ Correcto — D1=1 (100.0%), D2=9 (100.0%) |
| `close_digits_47.png` | 6. Dígitos próximos o en contacto | Dígitos '4' y '7' muy próximos (separación de 4px) | **47** | **47** | 87.98% | ✅ Correcto — D1=4 (88.0%), D2=7 (100.0%) |
| `close_touching_25.png` | 6. Dígitos próximos o en contacto | Dígitos '2' y '5' en contacto ligero (separados por proyección vertical) | **25** | **25** | 100.00% | ✅ Correcto — D1=2 (100.0%), D2=5 (100.0%) | Nota: Se detectaron dos dígitos muy próximos o en contacto; se separaron mediante perfil de proyección vertical. |
| `close_overlapping_hard_88.png` | 6. Dígitos próximos o en contacto | Caso límite extremo: dos '8' fuertemente solapados | **88** | **8** | 99.43% | ❌ Fallo / Límite — D1=8 (99.4%) |
| `invalid_blank.png` | 7. Validación de errores | Lienzo vacío sin contraste (debe rechazar con error controlado) | **Error esperado** | **Rechazado** | N/A | ✅ Correcto — Error controlado: _La imagen carece de contraste suficiente o está vacía. Asegúrese de que el número sea visible respecto al fondo._ |
| `invalid_three_digits_123.png` | 7. Validación de errores | Número de 3 cifras '123' fuera del rango 0-99 (debe rechazar con error controlado) | **Error esperado** | **Rechazado** | N/A | ✅ Correcto — Error controlado: _Se detectaron 3 dígitos/regiones independientes, pero el sistema solo admite números entre 0 y 99 (máximo 2 dígitos)._ |

## Análisis Crítico: Probabilidad Softmax vs. Garantía de Aciertos

1. **Sobreconfianza del MLP en datos fuera de distribución (Out-of-Distribution)**:
   Como el modelo original fue entrenado con entradas sin normalizar en el rango `[0, 255]` y capas
   `Dense` con activación `ReLU`, las combinaciones lineales producen *logits* de gran magnitud
   ($z_k \gg 10$). Al aplicar la función $\text{Softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$,
   cualquier diferencia moderada entre el mayor logit y el segundo produce probabilidades cercanas al
   `99.9%` o `100.0%`, **incluso cuando la segmentación falla o el dígito es clasificado erróneamente**.
2. **Caso límite de dígitos fuertemente solapados (`close_overlapping_hard_88.png`)**:
   Cuando dos dígitos se superponen físicamente de forma severa sin dejar un valle pronunciado en el
   histograma de proyección vertical, el segmentador no puede separarlos sin cortar trazos estructurales,
   o el corte altera la geometría de cada dígito. En tales casos, la red neuronal aún puede emitir una
   probabilidad Softmax alta para una clase incorrecta.
3. **Sensibilidad geométrica de la arquitectura MLP (Ausencia de capas convolucionales)**:
   Dado que la entrada es un vector plano de `784` píxeles conectado densamente, la red carece de
   invarianza espacial intrínseca. La precisión sobre imágenes propias depende críticamente de que
   el preprocesamiento recorte cada dígito, regule el grosor del trazo, lo escale a una caja de `20x20`
   y alinee su centro de masa en `(13.5, 13.5)` dentro del lienzo `28x28`.
