# Resumen del taller

**Curso:** Machine Learning – Universidad Libre  
**Autor:** Johel Santiago Arias Becerra  
**Entrega:** [`Taller_Evaluacion_Modelos_Johel_Arias.ipynb`](Taller_Evaluacion_Modelos_Johel_Arias.ipynb)  
**Repo:** [github.com/5arSanti/ml-hyperparameters](https://github.com/5arSanti/ml-hyperparameters)

## Qué se pedía

Tabla *Taller en Colab* (6 pasos). Sin subir archivos. El reto de la base de tiendas **no se aplicó**.

| Paso | Actividad pedida | Tiempo |
|------|------------------|--------|
| 1 | Cargar datos y dividir en entrenamiento y prueba | 10 min |
| 2 | Medir MAE, RMSE y R² en entrenamiento vs. prueba | 15 min |
| 3 | Detectar el sobreajuste: polinomios de grado 1 a 15 | 20 min |
| 4 | Validación cruzada: comparar 3 modelos | 25 min |
| 5 | GridSearchCV y RandomizedSearchCV | 30 min |
| 6 | Clasificación: matriz de confusión, F1 y AUC | 20 min |

## Qué se hizo

| Paso | Qué se hizo | Resultado clave |
|------|-------------|-----------------|
| 1 | Diabetes (sklearn), split 80/20, muestra, nulos e histograma | 353 train / 89 test |
| 2 | Lineal + MAE, RMSE, R², R² ajustado y residuos | R² ≈ 0.53 train / 0.45 test → **subajuste** |
| 3 | Polinomios 1–15 en curva 1D (en diabetes grado 15 son ~3.3 M de términos) | Mejor grado ≈ 5; grado 15 sobreajusta |
| 4 | CV 5-fold: lineal, árbol y kNN (+ k = 3, 5, 10) | Gana la lineal (~55 RMSE) |
| 5 | Grid en árbol (32 combos) y Random en kNN (15/120) | Sigue ganando la lineal; test solo al final |
| 6 | Cáncer de mama, split estratificado, F1, AUC y umbrales sobre P(maligno) | AUC ≈ 0.995; preferible umbral ≤ 0.5 |
| Reto tiendas | No se aplicó | — |

## Datos usados

- **Regresión (pasos 1, 2, 4, 5):** `load_diabetes`
- **Sobreajuste (paso 3):** curva sintética 1D `sin(2πx) + ruido`
- **Clasificación (paso 6):** `load_breast_cancer`

## Cómo se cuidó el flujo

- `SEED = 42` en todo.
- Escalado **dentro** del `Pipeline` (sin filtrar el test).
- CV y búsquedas solo sobre train.
- El test de diabetes se usa para diagnosticar el baseline (paso 2) y reportar al ganador (paso 5), no para elegir el modelo.

## Cómo ejecutarlo

1. Abrir el notebook en Jupyter o Colab.
2. Ejecutar las celdas de arriba abajo.
3. No hace falta subir archivos: todo viene de scikit-learn.
