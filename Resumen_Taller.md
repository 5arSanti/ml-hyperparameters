# Qué se pedía y qué se hizo

Taller de evaluación de modelos (Universidad Libre). Entrega: `Taller_Evaluacion_Modelos_Johel_Arias.ipynb`.

## Qué se pedía

Seis pasos en Colab, sin subir archivos, más un reto de tiendas (este último no se hizo, a propósito):

1. Cargar datos y partir en entrenamiento y prueba.
2. Medir MAE, RMSE y R² en train vs prueba.
3. Detectar sobreajuste con polinomios de grado 1 a 15.
4. Validación cruzada comparando 3 modelos.
5. `GridSearchCV` y `RandomizedSearchCV`.
6. Clasificación: matriz de confusión, F1 y AUC.

## Qué se hizo

| Paso | Qué se hizo |
|------|-------------|
| 1 | Diabetes de sklearn, split 80/20 (353 / 89), muestra, nulos e histograma. |
| 2 | Regresión lineal + MAE, RMSE, R² y R² ajustado train vs prueba, más residuos. Hay **subajuste**, no sobreajuste. |
| 3 | Grado 15 en diabetes son ~3.3 millones de términos, así que el sobreajuste se muestra en una curva 1D. El mínimo de test cae en grado 5; el 15 ya copia ruido. |
| 4 | CV 5-fold de lineal, árbol y kNN. Gana la lineal (más estable). Extra: k = 3, 5 y 10. |
| 5 | Grid en el árbol (32 combinaciones) y Random en kNN (15 de 120). Sigue ganando la lineal. El test se usa una vez, al final. |
| 6 | Cáncer de mama, split estratificado. Matriz, F1 por clase, AUC y umbrales sobre P(maligno). |
| Reto tiendas | No se aplicó. |

Todo corre con datasets de scikit-learn, `SEED = 42` y el escalado dentro de un `Pipeline` para no filtrar el test.
