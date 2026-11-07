# Taller de evaluación de modelos — diseño

**Fecha:** 2026-11-07  
**Curso:** Machine Learning, Universidad Libre  
**Entregable:** un notebook Colab, completo y ejecutado  
**Fuente del enunciado:** tabla *Taller en Colab* (Manos a la obra), 6 pasos

## 1. Objetivo

Reescribir `Taller_Evaluacion_Modelos.ipynb` para que coincida con los 6 pasos oficiales, se pueda ejecutar de arriba abajo en Google Colab (sin archivos externos) y deje constancia del análisis: cada paso incluye objetivo, código, gráfica cuando aporta, e interpretación escrita.

El notebook actual es un borrador útil (ya corre) pero no cumple el enunciado: falta el paso de polinomios 1–15, el orden de clasificación no es el oficial, la CV no compara tres modelos, y el split es 60/20/20 en vez de train/prueba.

## 2. Fuera de alcance

- Reto de la base de tiendas (`VENTAS_NL_procesado.xlsx` y cualquier flujo sobre ventas). No se menciona en el notebook final.
- Hold-out 60/20/20. Se reemplaza por 80/20 + validación cruzada en train.
- Módulos `.py`, tests automatizados, `requirements.txt`, app o informe aparte.
- Modelos fuera de la lista acordada (Ridge/Lasso/RandomForest/SVR no entran en la comparación del paso 4 ni en las búsquedas del paso 5).
- Inicializar git o publicar el notebook. El repo hoy no tiene `.git`.

## 3. Decisiones cerradas

| Tema | Decisión |
|------|----------|
| Forma del entregable | Un solo notebook pedagógico (código + interpretación) |
| Alineación | Tabla oficial 1:1 en el orden de los pasos |
| Extras que se conservan | R² ajustado, análisis de *k* en CV, curva de validación, umbrales 0.3/0.5/0.7 |
| Paso 3 | Curva sintética 1D con ruido, polinomios grado 1 a 15 |
| Paso 4 | `LinearRegression`, `DecisionTreeRegressor`, `KNeighborsRegressor` |
| Reto tiendas | Eliminado |
| Split | 80 % train / 20 % test; `random_state=42` |
| Idioma | Español en markdown, comentarios y etiquetas de gráficas |

## 4. Arquitectura

Un único artefacto: `Taller_Evaluacion_Modelos.ipynb`.

Tres experimentos independientes, cada uno con su propio split. No se mezclan filas, features ni targets entre ellos.

```
diabetes (sklearn)  → pasos 1, 2, 4, 5
sintético 1D        → paso 3
breast cancer       → paso 6
```

Dependencias (todas disponibles en Colab): `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

Convenciones:

- `SEED = 42` en una celda de imports al inicio. Todo `random_state`, `np.random.default_rng` y `KFold(shuffle=True)` usa esa semilla.
- Escalado **dentro** de `Pipeline`. Nunca `fit` de un scaler sobre test ni sobre el dataset completo antes de partir.
- CV y `GridSearchCV` / `RandomizedSearchCV` solo sobre el train del experimento correspondiente.
- El test de diabetes **sí** se usa en el paso 2 (el enunciado pide MAE/RMSE/R² train vs prueba del modelo base) y **otra vez** al cerrar el paso 5 (modelo ya elegido por CV). No se usa para elegir modelo, grado ni hiperparámetros.
- Helpers definidos en el propio notebook (no extraer a `.py`).

### 4.1 Helpers

`metricas_regresion(y_true, y_pred, n_features) -> dict` con `MAE`, `RMSE`, `R2`, `R2_aj`.

```
R2_aj = 1 - (1 - R2) * (n - 1) / (n - k - 1)
```

`n` es `len(y_true)` y `k` es el número de **features de entrada del modelo** (10 en diabetes; el grado del polinomio en el sintético 1D). Si `n <= k + 1` (ocurre en el test sintético de 16 filas con grado ≥ 15), devolver `R2_aj = NaN` y no dividir por cero.

## 5. Flujo de datos

### 5.1 Diabetes — pasos 1, 2, 4 y 5

- `load_diabetes(return_X_y=True, as_frame=True)`.
- `train_test_split(..., test_size=0.20, random_state=SEED)`.
- Tamaños esperados: 442 filas → 353 train / 89 test.
- Features: las 10 del dataset. Target: progresión de la enfermedad (continuo).
- Nota pedagógica: scikit-learn ya deja las features aproximadamente centradas; el `StandardScaler` en los pipelines de LR y kNN se incluye igual, como patrón correcto, no porque el dataset lo exija.

### 5.2 Sintético 1D — paso 3

Generación fija (reproducible):

- `n = 80`
- `x ~ Uniform(0, 1)` con `default_rng(SEED)`
- `y = sin(2πx) + ε`, `ε ~ Normal(0, 0.25)`
- Split 80/20 sobre pares `(x, y)`, misma semilla
- Función verdadera (sin ruido) se usa solo para dibujar la curva de referencia, no para entrenar

Este tamaño pequeño hace visible el sobreajuste en grados altos. No se usa diabetes aquí: un `PolynomialFeatures(degree=15)` sobre 10 variables genera millones de términos y no es ejecutable.

### 5.3 Cáncer de mama — paso 6

- `load_breast_cancer(return_X_y=True, as_frame=True)`
- Split 80/20 **estratificado** (`stratify=y`, `random_state=SEED`)
- Target binario: 0 = maligno / 1 = benigno según las etiquetas de sklearn; las filas y columnas de la matriz de confusión se etiquetan con esos nombres para no dejar `0/1` crudos
- `LogisticRegression(max_iter=1000)` dentro de `Pipeline` con `StandardScaler`

### 5.4 Fugas que este diseño evita

| Riesgo | Regla |
|--------|--------|
| Escalar con test | scaler solo como paso del `Pipeline` |
| Elegir hiperparámetros con test | `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score` solo en train |
| Elegir modelo o hiperparámetros con test | el test de diabetes solo mide el baseline (paso 2) y el ganador ya fijado (paso 5) |
| Polinomios sobre 10 features | experimento 1D aparte |
| CV no estratificada en clasificación | `stratify` en el split; si se usara CV extra, sería `StratifiedKFold` |

## 6. Contenido por paso

Cada paso sigue el mismo patrón de celdas: **título + objetivo (markdown) → código → gráfica o tabla → interpretación (markdown)**. Las interpretaciones van escritas, no como preguntas en blanco.

### Paso 1 — Cargar datos y partir (10 min)

- EDA breve: `shape`, lista de columnas, `describe()` del target, conteo train/test.
- Un histograma del target (entrenamiento vs prueba, o uno solo del dataset).
- Interpretación: qué es hold-out, por qué el test no sirve para elegir modelos, y por qué no hay conjunto de validación aparte (la CV del paso 4 lo sustituye).

### Paso 2 — MAE, RMSE y R² train vs prueba (15 min)

- Modelo: `Pipeline([StandardScaler(), LinearRegression()])` (sin búsqueda), para que el baseline sea comparable al paso 4.
- Tabla `train` vs `test` con MAE, RMSE, R² y R² ajustado. Aquí el test se usa como diagnóstico del baseline, no para seleccionar nada.
- Gráfica: predicho vs real en test (diagonal de referencia).
- Interpretación esperada con este dataset: R² de entrenamiento y prueba cercanos y moderados (~0.5) → **subajuste / sesgo**, no sobreajuste. Contrastar con lo que se verá en el paso 3.

### Paso 3 — Detectar sobreajuste, polinomios 1 a 15 (20 min)

- Para `degree` en `range(1, 16)`: `Pipeline([PolynomialFeatures(degree, include_bias=True), LinearRegression(fit_intercept=False)])` entrenado solo con train.
- Registrar RMSE y R² de train y test por grado (RMSE como métrica principal, alineada con la CV).
- Figura 1: RMSE de train y de test vs grado. Debe verse train bajando y test subiendo tras un mínimo.
- Figura 2: tres paneles (grado 1, grado de menor error de test, grado 15) con nubes de puntos, curva verdadera y curva ajustada.
- Interpretación: sesgo en grado bajo, varianza en grado alto, el grado se elige con test del sintético **solo como demostración visual**; recalcar que en un problema real el grado se elegiría con CV, no con el test.

### Paso 4 — Validación cruzada, comparar 3 modelos (25 min)

Sobre **diabetes train** (nunca test):

| Modelo | Pipeline |
|--------|----------|
| Regresión lineal | `StandardScaler` + `LinearRegression` |
| Árbol | `DecisionTreeRegressor(random_state=SEED)` sin scaler |
| kNN | `StandardScaler` + `KNeighborsRegressor()` |

- `KFold(n_splits=5, shuffle=True, random_state=SEED)`
- Scoring: `neg_root_mean_squared_error` (reportar RMSE positivo: media ± desv.)
- Gráfica: boxplot de RMSE por fold y modelo
- Extra conservado: repetir CV de la regresión lineal con `k ∈ {3, 5, 10}` y comentar que al subir *k* la media se estabiliza y la desviación entre folds suele crecer (folds más pequeños).
- Interpretación: sesgo vs varianza entre los tres modelos; no elegir ganador por error de entrenamiento.

### Paso 5 — GridSearchCV y RandomizedSearchCV (30 min)

Sigue sobre diabetes train.

1. `GridSearchCV` del árbol:
   - `max_depth`: `[1, 2, 3, 4, 5, 6, 8, None]`
   - `min_samples_leaf`: `[1, 2, 5, 10]`
   - `cv=5`, `scoring='neg_root_mean_squared_error'`
2. `RandomizedSearchCV` de kNN (`n_iter=15`, `random_state=SEED`):
   - `n_neighbors`: lista `range(1, 31)` (sin scipy)
   - `weights`: `uniform`, `distance`
   - `p`: 1, 2
   - misma CV y scoring
3. Tabla comparativa: mejor CV RMSE del árbol vs kNN vs la lineal del paso 4 (la lineal no tiene malla).
4. Extra conservado: `validation_curve` de `max_depth` del árbol (`param_range` 1–12) con RMSE de train vs validación. Debe ilustrar sobreajuste al aumentar la profundidad.
5. **Evaluación final de regresión:** entre lineal (paso 4), mejor árbol y mejor kNN, se elige el menor RMSE de CV. Ese estimador se evalúa en el test de diabetes con `metricas_regresion`. El test no entra en la elección.

Interpretación: diferencia práctica grid vs random, por qué el `Pipeline` evita fugas, y que el test no se usó para elegir el modelo.

### Paso 6 — Clasificación: matriz de confusión, F1 y AUC (20 min)

- `Pipeline([StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)])` sobre el train estratificado.
- Matriz de confusión impresa (DataFrame con etiquetas `maligno` / `benigno`) y heatmap.
- `classification_report` (precisión, recall, **F1**) y `roc_auc_score`.
- Curva ROC con `RocCurveDisplay`.
- Extra conservado: umbrales `0.3`, `0.5`, `0.7` → precisión, recall y F1. Relacionarlo con el costo de un falso negativo en un contexto clínico.
- Interpretación: qué dice la matriz, por qué F1 y AUC no son intercambiables, y el efecto del umbral.

### Cierre

Una celda markdown con **seis aprendizajes**, uno por paso. Sin reto, sin lista de “pendientes para el proyecto del curso”.

## 7. Estructura de celdas (orden)

1. Portada: título, curso, tabla de los 6 pasos (actividad + tiempo del enunciado), nota “ejecutar en orden, sin archivos externos”.
2. Imports + `SEED`.
3. Pasos 1–6 según la sección 6, cada uno autocontenido en markdown/código intercalado.
4. Cierre.

El notebook actual se **reescribe in-place**. No se conserva una copia del borrador.

## 8. Manejo de errores y reproducibilidad

- `LogisticRegression(max_iter=1000)` para evitar avisos de no convergencia.
- `DecisionTreeRegressor` y splits con `random_state=SEED`.
- `PolynomialFeatures` + `LinearRegression(fit_intercept=False)` cuando `include_bias=True`, para no duplicar intercepto.
- No silenciar warnings de sklearn de forma global.
- Figuras con ejes y leyendas en español; tamaño de figura razonable para Colab (`figsize` ~ 7×4 por panel).

## 9. Verificación de hecho

Antes de dar el taller por cerrado:

1. Reiniciar el kernel y ejecutar todas las celdas en orden, sin error.
2. Comprobar que existen las tablas de métricas de los pasos 2, 4, 5 y 6.
3. Comprobar que existen las figuras del paso 3 (error vs grado + tres ajustes), el boxplot del paso 4, la curva de validación del paso 5 y la ROC del paso 6.
4. Comprobar que **ninguna** celda carga Excel/CSV ni menciona tiendas.
5. Comprobar que ninguna búsqueda (`GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `validation_curve`) se ajusta con el test de diabetes. El test de diabetes solo aparece en el split (paso 1), en las métricas/gráfico del baseline (paso 2) y en la evaluación final (paso 5).
6. Semilla 42: volver a ejecutar el paso 3 y obtener la misma curva de error (mismos valores hasta 1e-6 en RMSE por grado).

No se piden tests pytest. El criterio de aceptación es el notebook ejecutado con salidas visibles.

## 10. Criterios de aceptación

- Los seis encabezados de paso coinciden con la tabla oficial (texto y orden).
- Paso 3 recorre grados 1 a 15 y muestra sobreajuste.
- Paso 4 compara exactamente tres modelos: lineal, árbol, kNN.
- Paso 5 incluye `GridSearchCV` y `RandomizedSearchCV`.
- Paso 6 reporta matriz de confusión, F1 y AUC.
- Cada paso tiene interpretación escrita (no solo código).
- No existe sección de reto / tiendas.
- El notebook corre en Colab sin subir archivos.
