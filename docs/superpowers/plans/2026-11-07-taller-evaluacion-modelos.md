# Taller Evaluación de Modelos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reescribir `Taller_Evaluacion_Modelos.ipynb` como un taller Colab completo (6 pasos oficiales, interpretaciones escritas, ejecutado de arriba abajo, sin reto de tiendas).

**Architecture:** Un único notebook. Tres experimentos independientes: diabetes (pasos 1, 2, 4, 5), curva sintética 1D (paso 3) y cáncer de mama (paso 6). Escalado y búsquedas van dentro de `Pipeline`/`CV` sobre train. El test de diabetes solo se usa en el diagnóstico del baseline (paso 2) y en la evaluación del ganador ya elegido (paso 5).

**Tech Stack:** Python 3, numpy, pandas, matplotlib, scikit-learn, nbformat, nbclient, ipykernel.

## Global Constraints

- Entregable: un solo notebook pedagógico en `Taller_Evaluacion_Modelos.ipynb` (reescribir in-place).
- Idioma: español en markdown, comentarios y etiquetas de gráficas.
- `SEED = 42` para todo `random_state`, `default_rng` y `KFold(shuffle=True)`.
- Split 80/20. Diabetes: 353 train / 89 test. Sintético: 64 / 16. Cáncer: 455 / 114 estratificado.
- Sin reto de tiendas, sin Excel/CSV, sin módulos `.py`, sin pytest, sin `requirements.txt`.
- No inicializar git ni hacer commits (el repo no tiene `.git`; el spec lo deja fuera).
- Escalado solo dentro de `Pipeline`. `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score` / `validation_curve` nunca reciben el test de diabetes.
- Encabezados oficiales, en este orden: (1) Cargar datos y dividir en entrenamiento y prueba, (2) Medir MAE, RMSE y R² en entrenamiento vs. prueba, (3) Detectar el sobreajuste: polinomios de grado 1 a 15, (4) Validación cruzada: comparar 3 modelos, (5) GridSearchCV y RandomizedSearchCV, (6) Clasificación: matriz de confusión, F1 y AUC.
- `metricas_regresion(y_true, y_pred, n_features) -> dict` con claves `MAE`, `RMSE`, `R2`, `R2_aj`. `R2_aj = NaN` si `n <= k + 1`.
- Figuras `figsize` ≈ 7×4; no silenciar warnings de sklearn de forma global.
- No dejar scripts generadores en el repo: los snippets de este plan se ejecutan y se descartan.
- Dependencias de ejecución local (no se versionan): `pip install numpy pandas matplotlib scikit-learn nbformat nbclient ipykernel`.

## File Structure

- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb` — único artefacto. Se sobreescribe en la Tarea 1 y se le anexan celdas en las tareas 2–7. La Tarea 8 lo ejecuta y deja las salidas dentro del `.ipynb`.
- Read-only: `/home/personal/Projects/machine-learning/ml-hyperparameters/docs/superpowers/specs/2026-11-07-taller-evaluacion-modelos-design.md`
- No crear `tests/`, `scripts/`, ni `requirements.txt`.

Nombres que las tareas posteriores consumen (no renombrar):

| Nombre | Tipo | Origen |
|--------|------|--------|
| `SEED` | `int` | Tarea 1 |
| `metricas_regresion(y_true, y_pred, n_features)` | `dict[str, float]` | Tarea 1 |
| `X, y, X_train, X_test, y_train, y_test` | diabetes | Tarea 2 |
| `X1_train, X1_test, y1_train, y1_test`, `tabla_polinomios`, `grado_optimo` | sintético | Tarea 4 |
| `cv`, `modelos_cv`, `rmse_cv`, `resumen_cv` | CV diabetes | Tarea 5 |
| `grid`, `rs`, `ganador_nombre`, `ganador` | búsqueda | Tarea 6 |
| `Xc_train, Xc_test, yc_train, yc_test` | cáncer | Tarea 7 |

Celdas esperadas al cerrar cada tarea (si se reintenta, recortar `nb.cells` a este largo antes de anexar): T1=2, T2=5, T3=9, T4=15, T5=19, T6=24, T7=30.

---

### Task 1: Esqueleto, imports y `metricas_regresion`

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb` (overwrite)

**Interfaces:**
- Consumes: nothing
- Produces: notebook con 2 celdas; `SEED = 42`; `metricas_regresion(y_true, y_pred, n_features) -> dict` con `MAE`, `RMSE`, `R2`, `R2_aj` (`R2_aj` es `NaN` si `n <= k+1`)

- [ ] **Step 1: Run the helper check against the current notebook (must fail)**

```bash
python3 - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert "def metricas_regresion(" in src, "metricas_regresion aún no está (esperado en este paso)"
assert "PolynomialFeatures" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError: metricas_regresion aún no está` (el borrador actual define `metricas`, no `metricas_regresion`).

- [ ] **Step 2: Overwrite the notebook with portada + imports/helper**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

nb.cells.append(nbf.v4.new_markdown_cell("""# Taller: Evaluación de modelos y ajuste de hiperparámetros
**Machine Learning – Universidad Libre**

Usa datasets incluidos en scikit-learn y un conjunto sintético 1D (no requiere subir archivos). Ejecuta las celdas en orden.

| Paso | Actividad | Tiempo |
|------|-----------|--------|
| 1 | Cargar datos y dividir en entrenamiento y prueba | 10 min |
| 2 | Medir MAE, RMSE y R² en entrenamiento vs. prueba | 15 min |
| 3 | Detectar el sobreajuste: polinomios de grado 1 a 15 | 20 min |
| 4 | Validación cruzada: comparar 3 modelos | 25 min |
| 5 | GridSearchCV y RandomizedSearchCV | 30 min |
| 6 | Clasificación: matriz de confusión, F1 y AUC | 20 min |
"""))

nb.cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

SEED = 42
plt.rcParams["figure.figsize"] = (7, 4)


def metricas_regresion(y_true, y_pred, n_features):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    k = n_features
    r2 = r2_score(y_true, y_pred)
    denom = n - k - 1
    r2_aj = np.nan if denom <= 0 else 1 - (1 - r2) * (n - 1) / denom
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2),
        "R2_aj": r2_aj,
    }
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 2`

- [ ] **Step 3: Verify helper formula and notebook contents**

```bash
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
assert len(nb["cells"]) == 2
src = "".join(nb["cells"][1]["source"])
assert "def metricas_regresion(" in src
assert "SEED = 42" in src
assert "tienda" not in src.lower()
ns = {}
exec(src, ns)
m = ns["metricas_regresion"]([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1)
assert m["MAE"] == 0.0 and m["RMSE"] == 0.0 and abs(m["R2"] - 1.0) < 1e-12
m2 = ns["metricas_regresion"](np.arange(5.0), np.arange(5.0), 4)  # n=5, k=4, denom=0
assert np.isnan(m2["R2_aj"])
print("OK helper")
PY
```

Expected: `OK helper`

- [ ] **Step 4: Skip commit** — no hay git; el spec lo deja fuera.

---

### Task 2: Paso 1 — carga y split 80/20 de diabetes

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `SEED`
- Produces: `X, y, X_train, X_test, y_train, y_test` (diabetes, 353/89)

- [ ] **Step 1: Confirm Paso 1 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert "## Paso 1 – Cargar datos y dividir en entrenamiento y prueba" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 1 cells (truncate to 2 cells first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:2]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 1 – Cargar datos y dividir en entrenamiento y prueba (10 min)

Cargamos el dataset de diabetes de scikit-learn (442 pacientes, 10 variables clínicas, target continuo de progresión) y separamos **80 % entrenamiento / 20 % prueba**. El conjunto de prueba no se usa para elegir modelos ni hiperparámetros; la validación de los pasos 4 y 5 se hace con validación cruzada sobre el entrenamiento.
"""))

nb.cells.append(nbf.v4.new_code_cell("""X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

print("Filas, columnas:", X.shape)
print("Variables:", list(X.columns))
print("Entrenamiento:", len(X_train), "| Prueba:", len(X_test))
print()
print("Target (entrenamiento):")
print(y_train.describe().round(2))

fig, ax = plt.subplots()
ax.hist(y_train, bins=20, alpha=0.65, label="entrenamiento")
ax.hist(y_test, bins=20, alpha=0.65, label="prueba")
ax.set_xlabel("Progresión de diabetes")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del target")
ax.legend()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** El hold-out deja 353 filas para entrenar y **89 para probar**. Esas 89 no deben entrar en comparaciones de modelos ni en `GridSearchCV`: si el test participa en las decisiones, deja de ser una prueba y se convierte en otra validación. No separamos un tercer conjunto de validación porque los pasos 4 y 5 usarán validación cruzada sobre el entrenamiento. Las features de este dataset ya vienen aproximadamente centradas en scikit-learn; aun así, más adelante el `StandardScaler` irá **dentro** del `Pipeline` para no filtrar información del test.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 5`

- [ ] **Step 3: Verify split sizes without touching later steps**

```bash
python3 - <<'PY'
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
assert len(X_train) == 353 and len(X_test) == 89
import json
from pathlib import Path
nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert "test_size=0.20" in src
assert "60/20/20" not in src
assert "tienda" not in src.lower()
print("OK paso 1")
PY
```

Expected: `OK paso 1`

- [ ] **Step 4: Skip commit**

---

### Task 3: Paso 2 — MAE, RMSE, R² train vs prueba (baseline lineal)

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `X_train, X_test, y_train, y_test`, `metricas_regresion`, `SEED`
- Produces: `pipe_lr` (`Pipeline` scaler + `LinearRegression` ya ajustado). El test de diabetes se usa aquí solo como diagnóstico del baseline.

Números esperados con `SEED=42` (redondeo a 3 decimales):

| | train | test |
|--|------:|-----:|
| MAE | 43.484 | 42.794 |
| RMSE | 53.559 | 53.853 |
| R2 | 0.528 | 0.453 |
| R2_aj | 0.514 | 0.382 |

- [ ] **Step 1: Confirm Paso 2 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "## Paso 2 – Medir MAE, RMSE y R² en entrenamiento vs. prueba" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 2 cells (truncate to 5 first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:5]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 2 – Medir MAE, RMSE y R² en entrenamiento vs. prueba (15 min)

Entrenamos una **regresión lineal** (con `StandardScaler` en el pipeline, igual que en el paso 4) y comparamos MAE, RMSE, R² y R² ajustado en entrenamiento vs. prueba. El test se usa aquí como diagnóstico del modelo base, no para elegir entre modelos.
"""))

nb.cells.append(nbf.v4.new_code_cell("""pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression()),
]).fit(X_train, y_train)

k = X_train.shape[1]
tabla_baseline = pd.DataFrame({
    "train": metricas_regresion(y_train, pipe_lr.predict(X_train), k),
    "prueba": metricas_regresion(y_test, pipe_lr.predict(X_test), k),
}).round(3)
tabla_baseline
"""))

nb.cells.append(nbf.v4.new_code_cell("""y_pred_test = pipe_lr.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_test, alpha=0.7)
lo, hi = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
ax.plot([lo, hi], [lo, hi], "--", color="gray", label="predicción perfecta")
ax.set_xlabel("Valor real")
ax.set_ylabel("Predicción")
ax.set_title("Regresión lineal — prueba")
ax.legend()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** R² de entrenamiento 0.528 y de prueba 0.453, con MAE/RMSE casi iguales en ambos conjuntos (≈43.5 / 53.6 vs 42.8 / 53.9). Cuando train y test se parecen y el R² se queda a media tabla, el síntoma no es sobreajuste: el modelo **subajusta** (sesgo). La relación entre las 10 variables y la progresión no es del todo lineal, o faltan interacciones. El R² ajustado cae más en prueba (0.382) porque n=89 es pequeño frente a k=10. En el paso 3 veremos el patrón contrario: un modelo demasiado flexible.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 9`

- [ ] **Step 3: Verify baseline numbers and that GridSearch is not used yet**

```bash
python3 - <<'PY'
import json
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def metricas_regresion(y_true, y_pred, n_features):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n, k = len(y_true), n_features
    r2 = r2_score(y_true, y_pred)
    denom = n - k - 1
    r2_aj = np.nan if denom <= 0 else 1 - (1 - r2) * (n - 1) / denom
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))), "R2": float(r2), "R2_aj": r2_aj}

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("modelo", LinearRegression())]).fit(X_train, y_train)
m = metricas_regresion(y_test, pipe.predict(X_test), 10)
assert abs(m["R2"] - 0.453) < 5e-3
nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert "subajusta" in src
assert "GridSearchCV(" not in src
print("OK paso 2")
PY
```

Expected: `OK paso 2`

- [ ] **Step 4: Skip commit**

---

### Task 4: Paso 3 — polinomios grado 1 a 15 (sintético 1D)

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `SEED`, `metricas_regresion`
- Produces: `X1_train, X1_test, y1_train, y1_test`; `tabla_polinomios` con columnas `grado, RMSE_train, RMSE_test, R2_train, R2_test`; `grado_optimo` (entero, esperado **5**)

Generación fija: `n=80`, `x ~ Uniform(0,1)`, `y = sin(2πx) + ε`, `ε ~ Normal(0, 0.25)`, `default_rng(SEED)`. Pipeline por grado: `PolynomialFeatures(degree, include_bias=True)` + `LinearRegression(fit_intercept=False)`.

RMSE test esperado (aprox.): grado 1 → 0.495; grado 5 (mínimo) → 0.237; grado 15 → 0.265. Train baja de 0.492 a 0.212.

- [ ] **Step 1: Confirm Paso 3 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "## Paso 3 – Detectar el sobreajuste: polinomios de grado 1 a 15" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 3 cells (truncate to 9 first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:9]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 3 – Detectar el sobreajuste: polinomios de grado 1 a 15 (20 min)

Con las 10 variables de diabetes un polinomio de grado 15 genera millones de términos. Para ver el sobreajuste usamos una **curva sintética 1D**: 80 puntos de `y = sin(2πx) + ruido`, partidos 80/20. Ajustamos polinomios de grado 1 a 15 **solo con el train** de este experimento (no usamos diabetes).
"""))

nb.cells.append(nbf.v4.new_code_cell("""rng = np.random.default_rng(SEED)
n_sint = 80
x1 = rng.uniform(0, 1, size=n_sint)
y1 = np.sin(2 * np.pi * x1) + rng.normal(0, 0.25, size=n_sint)
X1 = x1.reshape(-1, 1)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.20, random_state=SEED
)
print("Sintético — train:", len(X1_train), "| test:", len(X1_test))
"""))

nb.cells.append(nbf.v4.new_code_cell("""filas = []
pipes_poly = {}
for grado in range(1, 16):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=grado, include_bias=True)),
        ("lin", LinearRegression(fit_intercept=False)),
    ]).fit(X1_train, y1_train)
    pipes_poly[grado] = pipe
    tr = metricas_regresion(y1_train, pipe.predict(X1_train), n_features=grado)
    te = metricas_regresion(y1_test, pipe.predict(X1_test), n_features=grado)
    filas.append({
        "grado": grado,
        "RMSE_train": tr["RMSE"],
        "RMSE_test": te["RMSE"],
        "R2_train": tr["R2"],
        "R2_test": te["R2"],
    })

tabla_polinomios = pd.DataFrame(filas)
grado_optimo = int(tabla_polinomios.loc[tabla_polinomios["RMSE_test"].idxmin(), "grado"])
print("Grado con menor RMSE de test:", grado_optimo)
tabla_polinomios.round(3)
"""))

nb.cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots()
ax.plot(tabla_polinomios["grado"], tabla_polinomios["RMSE_train"], "o-", label="entrenamiento")
ax.plot(tabla_polinomios["grado"], tabla_polinomios["RMSE_test"], "o-", label="prueba")
ax.axvline(grado_optimo, color="gray", linestyle="--", label=f"mínimo test (grado {grado_optimo})")
ax.set_xlabel("Grado del polinomio")
ax.set_ylabel("RMSE")
ax.set_title("Sesgo vs. varianza según el grado")
ax.legend()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
y_real = np.sin(2 * np.pi * x_grid.ravel())
grados_panel = [1, grado_optimo, 15]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, g in zip(axes, grados_panel):
    ax.scatter(X1_train.ravel(), y1_train, s=18, alpha=0.7, label="train")
    ax.scatter(X1_test.ravel(), y1_test, s=18, alpha=0.7, label="test")
    ax.plot(x_grid, y_real, color="black", linewidth=1.2, label="función verdadera")
    ax.plot(x_grid, pipes_poly[g].predict(x_grid), color="crimson", linewidth=1.8, label="ajuste")
    ax.set_title(f"Grado {g}")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)
fig.suptitle("Subajuste, ajuste razonable y sobreajuste")
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** Grado 1: RMSE de test ≈ 0.50; una recta no puede seguir un seno → **sesgo / subajuste**. Grado 5: mínimo de test (RMSE ≈ 0.237, R² ≈ 0.90) y el panel del medio sigue la onda. Grado 15: el error de train sigue bajando (≈ 0.212) mientras el de test sube (≈ 0.265) y la curva **persigue el ruido** → sobreajuste / varianza. Elegir el grado 5 con este test es solo para la figura. En un problema real el grado se elegiría con validación cruzada sobre el train, igual que en los pasos 4 y 5.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 15`

- [ ] **Step 3: Verify degree-5 minimum and reproducibility to 1e-6**

```bash
python3 - <<'PY'
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

SEED = 42
rng = np.random.default_rng(SEED)
x1 = rng.uniform(0, 1, size=80)
y1 = np.sin(2 * np.pi * x1) + rng.normal(0, 0.25, size=80)
X1_train, X1_test, y1_train, y1_test = train_test_split(x1.reshape(-1, 1), y1, test_size=0.20, random_state=SEED)
rmses = []
for d in range(1, 16):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=True)),
        ("lin", LinearRegression(fit_intercept=False)),
    ]).fit(X1_train, y1_train)
    rmses.append(rmse(y1_test, pipe.predict(X1_test)))
assert int(np.argmin(rmses)) + 1 == 5
assert abs(rmses[4] - 0.2370) < 5e-3
# second draw with same seed must match to 1e-6
rng2 = np.random.default_rng(SEED)
x2 = rng2.uniform(0, 1, size=80)
assert np.max(np.abs(x1 - x2)) < 1e-12
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "range(1, 16)" in src
assert "sobreajuste" in src
print("OK paso 3")
PY
```

Expected: `OK paso 3`

- [ ] **Step 4: Skip commit**

---

### Task 5: Paso 4 — CV de lineal vs árbol vs kNN

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `X_train, y_train`, `SEED` (nunca `X_test`)
- Produces:
  - `cv = KFold(n_splits=5, shuffle=True, random_state=SEED)`
  - `modelos_cv: dict[str, Pipeline]` con claves exactas `"Regresión lineal"`, `"Árbol de decisión"`, `"kNN"`
  - `rmse_cv: dict[str, np.ndarray]` (5 RMSEs por modelo)
  - `resumen_cv: DataFrame` columnas `modelo, RMSE_medio, RMSE_desv`

RMSE CV 5-fold esperado: lineal 55.395 ± 2.361; kNN 62.110 ± 3.580; árbol 80.411 ± 9.862. Extra k=3/5/10 en la lineal: medias 55.791, 55.395, 55.182 y desv. 1.927, 2.361, 4.716.

- [ ] **Step 1: Confirm Paso 4 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "## Paso 4 – Validación cruzada: comparar 3 modelos" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 4 cells (truncate to 15 first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:15]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 4 – Validación cruzada: comparar 3 modelos (25 min)

Comparamos tres capacidades distintas **solo sobre el entrenamiento** de diabetes: regresión lineal (sesgo alto, varianza baja), árbol sin podar (varianza alta) y kNN. `KFold(5, shuffle=True)` con la misma semilla. El scoring de sklearn es `neg_root_mean_squared_error`; reportamos RMSE positivo.
"""))

nb.cells.append(nbf.v4.new_code_cell("""cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

modelos_cv = {
    "Regresión lineal": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression()),
    ]),
    "Árbol de decisión": Pipeline([
        ("modelo", DecisionTreeRegressor(random_state=SEED)),
    ]),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsRegressor()),
    ]),
}

rmse_cv = {}
for nombre, modelo in modelos_cv.items():
    scores = -cross_val_score(
        modelo, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse_cv[nombre] = scores
    print(f"{nombre:20s}  RMSE medio={scores.mean():.3f}  desv={scores.std():.3f}")

resumen_cv = pd.DataFrame({
    "modelo": list(rmse_cv),
    "RMSE_medio": [v.mean() for v in rmse_cv.values()],
    "RMSE_desv": [v.std() for v in rmse_cv.values()],
}).round(3)

fig, ax = plt.subplots()
ax.boxplot(list(rmse_cv.values()))
ax.set_xticklabels(list(rmse_cv.keys()), rotation=15)
ax.set_ylabel("RMSE (folds)")
ax.set_title("Validación cruzada 5-fold — diabetes (train)")
plt.show()
resumen_cv
"""))

nb.cells.append(nbf.v4.new_code_cell("""print("Estabilidad al variar k (solo regresión lineal)")
for k in (3, 5, 10):
    scores = -cross_val_score(
        modelos_cv["Regresión lineal"],
        X_train,
        y_train,
        cv=KFold(n_splits=k, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
    )
    print(f"k={k:2d}  RMSE medio={scores.mean():.3f}  desv={scores.std():.3f}")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** Lineal 55.4 ± 2.4, kNN 62.1 ± 3.6, árbol 80.4 ± 9.9. El árbol por defecto crece sin límite y **sobreajusta** (varianza alta, folds inestables). kNN con 5 vecinos queda en el medio. La lineal, aunque en el paso 2 subajustaba un poco, es la más estable: no hay que elegir ganador por el error de entrenamiento. Al pasar k de 3 a 10 en la lineal, la media se mueve poco (55.8 → 55.2) y la desviación entre folds crece (1.9 → 4.7): folds más chicos, estimación más ruidosa.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 19`

- [ ] **Step 3: Verify three model names and that CV code uses X_train not X_test**

```bash
python3 - <<'PY'
import json, re
from pathlib import Path
cells = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"]
# cells 15-18 are paso 4
p4 = "\n".join("".join(c.get("source", [])) for c in cells[15:19])
assert "Regresión lineal" in p4 and "Árbol de decisión" in p4 and "kNN" in p4
assert "DecisionTreeRegressor" in p4 and "KNeighborsRegressor" in p4
assert "X_train" in p4
assert re.search(r"cross_val_score\([^)]*X_test", p4) is None
assert "Ridge" not in p4 and "Lasso" not in p4
print("OK paso 4")
PY
```

Expected: `OK paso 4`

- [ ] **Step 4: Skip commit**

---

### Task 6: Paso 5 — GridSearchCV, RandomizedSearchCV, curva de validación y test final

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `X_train, y_train, X_test, y_test`, `cv`, `modelos_cv`, `resumen_cv`, `metricas_regresion`, `SEED`
- Produces: `grid` (`GridSearchCV` del árbol), `rs` (`RandomizedSearchCV` de kNN), `ganador_nombre` (esperado `"Regresión lineal"`), `ganador` (pipeline ajustado)

Mallas (exactas):

- Árbol / GridSearchCV: `modelo__max_depth = [1, 2, 3, 4, 5, 6, 8, None]`, `modelo__min_samples_leaf = [1, 2, 5, 10]`, `cv=cv`, `scoring="neg_root_mean_squared_error"`. Esperado: `max_depth=2`, `min_samples_leaf=1`, RMSE CV 63.392.
- kNN / RandomizedSearchCV: `n_iter=15`, `random_state=SEED`, `modelo__n_neighbors=list(range(1, 31))`, `modelo__weights=["uniform", "distance"]`, `modelo__p=[1, 2]`. Esperado: `n_neighbors=12`, `weights=distance`, `p=2`, RMSE CV 58.492.
- `validation_curve` de `modelo__max_depth` con `param_range=list(range(1, 13))` sobre el pipeline del árbol, `X_train, y_train` (nunca test).
- Ganador = menor RMSE de CV entre lineal (paso 4), ` -grid.best_score_` y `-rs.best_score_`. Test del ganador lineal: MAE 42.794, RMSE 53.853, R² 0.453, R²aj 0.382.

- [ ] **Step 1: Confirm Paso 5 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "## Paso 5 – GridSearchCV y RandomizedSearchCV" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 5 cells (truncate to 19 first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:19]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 5 – GridSearchCV y RandomizedSearchCV (30 min)

Ajustamos hiperparámetros **dentro de un Pipeline y solo con el train**. `GridSearchCV` recorre una malla completa del árbol; `RandomizedSearchCV` muestra 15 combinaciones de kNN. La regresión lineal no tiene malla: su RMSE de CV es el del paso 4. El test de diabetes se abre al final, cuando el ganador ya está decidido.
"""))

nb.cells.append(nbf.v4.new_code_cell("""grid = GridSearchCV(
    modelos_cv["Árbol de decisión"],
    param_grid={
        "modelo__max_depth": [1, 2, 3, 4, 5, 6, 8, None],
        "modelo__min_samples_leaf": [1, 2, 5, 10],
    },
    cv=cv,
    scoring="neg_root_mean_squared_error",
)
grid.fit(X_train, y_train)
print("Mejor árbol:", grid.best_params_)
print("RMSE CV árbol:", round(-grid.best_score_, 3))

rs = RandomizedSearchCV(
    modelos_cv["kNN"],
    param_distributions={
        "modelo__n_neighbors": list(range(1, 31)),
        "modelo__weights": ["uniform", "distance"],
        "modelo__p": [1, 2],
    },
    n_iter=15,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    random_state=SEED,
)
rs.fit(X_train, y_train)
print("Mejor kNN:", rs.best_params_)
print("RMSE CV kNN:", round(-rs.best_score_, 3))
"""))

nb.cells.append(nbf.v4.new_code_cell("""comparacion = pd.DataFrame([
    {
        "modelo": "Regresión lineal",
        "búsqueda": "—",
        "RMSE_CV": float(resumen_cv.set_index("modelo").loc["Regresión lineal", "RMSE_medio"]),
        "params": "—",
    },
    {
        "modelo": "Árbol de decisión",
        "búsqueda": "GridSearchCV",
        "RMSE_CV": float(-grid.best_score_),
        "params": grid.best_params_,
    },
    {
        "modelo": "kNN",
        "búsqueda": "RandomizedSearchCV",
        "RMSE_CV": float(-rs.best_score_),
        "params": rs.best_params_,
    },
]).round(3)

ganador_nombre = comparacion.loc[comparacion["RMSE_CV"].idxmin(), "modelo"]
estimadores = {
    "Regresión lineal": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression()),
    ]).fit(X_train, y_train),
    "Árbol de decisión": grid.best_estimator_,
    "kNN": rs.best_estimator_,
}
ganador = estimadores[ganador_nombre]
metricas_final = metricas_regresion(y_test, ganador.predict(X_test), n_features=X_train.shape[1])
print("Ganador según CV:", ganador_nombre)
print("Métricas en prueba:", {k: round(v, 3) for k, v in metricas_final.items()})
comparacion
"""))

nb.cells.append(nbf.v4.new_code_cell("""profundidades = list(range(1, 13))
train_scores, val_scores = validation_curve(
    modelos_cv["Árbol de decisión"],
    X_train,
    y_train,
    param_name="modelo__max_depth",
    param_range=profundidades,
    cv=cv,
    scoring="neg_root_mean_squared_error",
)
fig, ax = plt.subplots()
ax.plot(profundidades, -train_scores.mean(axis=1), "o-", label="entrenamiento")
ax.plot(profundidades, -val_scores.mean(axis=1), "o-", label="validación (CV)")
ax.set_xlabel("max_depth")
ax.set_ylabel("RMSE")
ax.set_title("Curva de validación — árbol de decisión")
ax.legend()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** GridSearch deja el árbol en `max_depth=2` (RMSE CV ≈ 63.4): mejor que el árbol suelto del paso 4 (≈ 80) y peor que la lineal (≈ 55.4). RandomSearch deja kNN en 12 vecinos con distancia (RMSE CV ≈ 58.5): mejora el default y sigue detrás de la lineal. La búsqueda no inventa señal: en diabetes gana el modelo más simple. Grid explora las 32 combinaciones del árbol; Random prueba 15 de kNN y llega a un buen valor sin recorrer toda la malla. El `Pipeline` evita fugas porque el scaler (kNN) se ajusta en cada fold, no sobre el test. El test del ganador (MAE 42.794, RMSE 53.853, R² 0.453) coincide con el diagnóstico del paso 2: no lo usamos para elegir. La curva de `max_depth` debe mostrar el RMSE de train bajando y el de validación estancándose o subiendo al profundizar — el mismo patrón del paso 3, ahora con un hiperparámetro del árbol.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
PY
```

Expected: `cells 24`

- [ ] **Step 3: Verify search fits only X_train and winner selection uses CV**

```bash
python3 - <<'PY'
import json, re
from pathlib import Path
cells = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"]
p5 = "\n".join("".join(c.get("source", [])) for c in cells[19:24])
assert "GridSearchCV" in p5 and "RandomizedSearchCV" in p5 and "validation_curve" in p5
assert "grid.fit(X_train, y_train)" in p5
assert "rs.fit(X_train, y_train)" in p5
assert "validation_curve(" in p5 and "X_train" in p5
assert re.search(r"\.(fit|validation_curve)\([^)]*X_test", p5) is None
assert "ganador.predict(X_test)" in p5
assert "n_iter=15" in p5
print("OK paso 5")
PY
```

Expected: `OK paso 5`

- [ ] **Step 4: Skip commit**

---

### Task 7: Paso 6 — clasificación + cierre

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb`

**Interfaces:**
- Consumes: `SEED`
- Produces: `Xc_train, Xc_test, yc_train, yc_test`, `clf` (pipeline scaler + `LogisticRegression(max_iter=1000, random_state=SEED)`). Experimento independiente: no reutiliza diabetes.

sklearn `target_names`: índice 0 = `malignant` → etiqueta **maligno**; 1 = `benign` → **benigno**. Split estratificado 455/114. Esperado en test: matriz `[[41, 1], [1, 71]]`, accuracy 0.982, AUC 0.995. Umbrales sobre P(benigno): 0.3 → prec 0.973 / rec 1.000 / F1 0.986; 0.5 → 0.986 / 0.986 / 0.986; 0.7 → 0.985 / 0.931 / 0.957.

- [ ] **Step 1: Confirm Paso 6 heading is missing**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = "\n".join("".join(c.get("source", [])) for c in json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())["cells"])
assert "## Paso 6 – Clasificación: matriz de confusión, F1 y AUC" in src
assert "Aprendizajes" in src
print("UNEXPECTED PASS")
PY
```

Expected: `AssertionError`

- [ ] **Step 2: Append Paso 6 + cierre (truncate to 24 first if retrying)**

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat as nbf

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbf.read(NB, as_version=4)
nb.cells = nb.cells[:24]

nb.cells.append(nbf.v4.new_markdown_cell("""## Paso 6 – Clasificación: matriz de confusión, F1 y AUC (20 min)

Cambiamos de problema: `load_breast_cancer` (569 tumores, 30 variables). Clase 0 = **maligno**, clase 1 = **benigno**. Split 80/20 **estratificado**. Modelo: `StandardScaler` + `LogisticRegression` en un `Pipeline`.
"""))

nb.cells.append(nbf.v4.new_code_cell("""Xc, yc = load_breast_cancer(return_X_y=True, as_frame=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.20, stratify=yc, random_state=SEED
)
print("Cáncer — train:", len(Xc_train), "| test:", len(Xc_test))
print("Balance test:", yc_test.value_counts().to_dict(), "(0=maligno, 1=benigno)")

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LogisticRegression(max_iter=1000, random_state=SEED)),
]).fit(Xc_train, yc_train)

pred = clf.predict(Xc_test)
etiquetas = ["maligno", "benigno"]
cm = pd.DataFrame(
    confusion_matrix(yc_test, pred),
    index=["real maligno", "real benigno"],
    columns=["pred maligno", "pred benigno"],
)
print(cm)
print()
print(classification_report(yc_test, pred, target_names=etiquetas, digits=3))
print("AUC =", round(roc_auc_score(yc_test, clf.predict_proba(Xc_test)[:, 1]), 3))

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    yc_test, pred, display_labels=etiquetas, cmap="Blues", ax=ax
)
ax.set_title("Matriz de confusión — prueba")
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(clf, Xc_test, yc_test, ax=ax)
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_title("Curva ROC — prueba")
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""proba_benigno = clf.predict_proba(Xc_test)[:, 1]
print("Umbral sobre P(benigno)")
for t in (0.3, 0.5, 0.7):
    p = (proba_benigno >= t).astype(int)
    print(
        f"t={t:.1f}  precisión={precision_score(yc_test, p):.3f}  "
        f"recall={recall_score(yc_test, p):.3f}  F1={f1_score(yc_test, p):.3f}"
    )
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""**Interpretación.** En 114 pruebas hay 2 errores: 1 maligno declarado benigno (el falso negativo clínico) y 1 benigno declarado maligno. F1 0.976 / 0.986 y AUC 0.995: el modelo ordena muy bien las probabilidades. F1 resume precisión y recall en **un umbral**; el AUC resume el ranking en **todos** los umbrales — no se sustituyen. Bajar el umbral de benigno a 0.3 hace más fácil decir “benigno” (recall de benigno = 1.00) y puede dejar pasar malignos. Subirlo a 0.7 declara menos casos benignos (recall 0.931): más falsas alarmas, menos riesgo de perder un cáncer. En clínica el costo de un maligno perdido suele pesar más que el de una biopsia de más.
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## Cierre — seis aprendizajes

1. **Hold-out:** el test existe para medir, no para decidir. 80/20 y semilla fija bastan si la selección se hace con CV en el train.
2. **Métricas de regresión:** MAE, RMSE y R² train vs prueba del baseline lineal mostraron **subajuste**, no sobreajuste (errores parecidos, R² ≈ 0.5).
3. **Polinomios 1–15:** grado bajo = sesgo, grado alto = varianza. El mínimo de test (grado 5) es una demo visual; en producción el grado se elige con CV.
4. **Validación cruzada:** comparar lineal, árbol y kNN con 5 folds evitó fiarse de un solo corte. Más folds estabilizan la media y ensanchan la desviación entre folds.
5. **Grid vs Random:** el `Pipeline` evita fugas. La búsqueda mejoró árbol y kNN, pero en diabetes siguió ganando la lineal: tunear no crea señal.
6. **Clasificación:** matriz de confusión + F1 + AUC + umbral. El umbral se mueve según el costo de un falso negativo (perder un maligno) frente a un falso positivo.
"""))

nbf.write(nb, NB)
print("cells", len(nb.cells))
src = "\n".join("".join(c.source) for c in nb.cells)
assert "tienda" not in src.lower()
print("sin tiendas")
PY
```

Expected: `cells 30` y `sin tiendas`

- [ ] **Step 3: Verify official headings, F1/AUC, and no stores challenge**

```bash
python3 - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb").read_text())
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for h in [
    "## Paso 1 – Cargar datos y dividir en entrenamiento y prueba",
    "## Paso 2 – Medir MAE, RMSE y R² en entrenamiento vs. prueba",
    "## Paso 3 – Detectar el sobreajuste: polinomios de grado 1 a 15",
    "## Paso 4 – Validación cruzada: comparar 3 modelos",
    "## Paso 5 – GridSearchCV y RandomizedSearchCV",
    "## Paso 6 – Clasificación: matriz de confusión, F1 y AUC",
]:
    assert h in src, h
assert "f1_score" in src and "roc_auc_score" in src and "ConfusionMatrixDisplay" in src
assert "tienda" not in src.lower() and "VENTAS" not in src
assert "read_excel" not in src and "read_csv" not in src
assert len(nb["cells"]) == 30
print("OK paso 6")
PY
```

Expected: `OK paso 6`

- [ ] **Step 4: Skip commit**

---

### Task 8: Ejecutar el notebook y verificar aceptación

**Files:**
- Modify: `/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb` (se reescribe con outputs)

**Interfaces:**
- Consumes: notebook de 30 celdas de las tareas 1–7
- Produces: el mismo archivo, ejecutado, con tablas y figuras en las salidas

- [ ] **Step 1: Ensure a Python 3 kernel exists**

```bash
python3 -m pip install -q numpy pandas matplotlib scikit-learn nbformat nbclient ipykernel
python3 -m ipykernel install --user --name python3 --display-name "Python 3"
python3 -c "import nbformat, nbclient, sklearn, matplotlib; print('deps OK')"
```

Expected: `deps OK`

- [ ] **Step 2: Execute the notebook in place (Agg backend, timeout 300s)**

```bash
python3 - <<'PY'
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import nbformat
from nbclient import NotebookClient

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}})
client.execute()
nbformat.write(nb, NB)
print("executed", len(nb.cells), "cells")
errs = [c for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("errors", len(errs))
PY
```

Expected: `executed 30 cells` y `errors 0`. Si hay error, leer el traceback, corregir la celda en el notebook (sin añadir archivos nuevos) y repetir este paso.

- [ ] **Step 3: Acceptance checklist from the spec**

```bash
python3 - <<'PY'
import json
from pathlib import Path

NB = Path("/home/personal/Projects/machine-learning/ml-hyperparameters/Taller_Evaluacion_Modelos.ipynb")
nb = json.loads(NB.read_text())
assert len(nb["cells"]) == 30
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

for h in [
    "## Paso 1 – Cargar datos y dividir en entrenamiento y prueba",
    "## Paso 2 – Medir MAE, RMSE y R² en entrenamiento vs. prueba",
    "## Paso 3 – Detectar el sobreajuste: polinomios de grado 1 a 15",
    "## Paso 4 – Validación cruzada: comparar 3 modelos",
    "## Paso 5 – GridSearchCV y RandomizedSearchCV",
    "## Paso 6 – Clasificación: matriz de confusión, F1 y AUC",
]:
    assert h in src, f"falta {h}"

assert "range(1, 16)" in src
assert "GridSearchCV" in src and "RandomizedSearchCV" in src
assert "confusion_matrix" in src and "f1_score" in src and "roc_auc_score" in src
assert "tienda" not in src.lower() and "VENTAS" not in src
assert "read_excel" not in src and ".csv" not in src

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert all(c.get("execution_count") for c in code_cells), "quedó alguna celda de código sin ejecutar"

def n_display_or_stream(c):
    return sum(1 for o in c.get("outputs", []) if o.get("output_type") in {"display_data", "execute_result", "stream"})

assert sum(n_display_or_stream(c) for c in code_cells) >= 8, "faltan tablas/figuras"

# diabetes X_test must not appear in search/CV calls
search_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if "GridSearchCV" in "".join(c.get("source", [])) or "RandomizedSearchCV" in "".join(c.get("source", [])) or "cross_val_score" in "".join(c.get("source", [])) or "validation_curve" in "".join(c.get("source", [])))
assert "grid.fit(X_test" not in search_src
assert "rs.fit(X_test" not in search_src
assert "cross_val_score(\n        modelo, X_test" not in search_src
print("ACEPTADO")
PY
```

Expected: `ACEPTADO`

- [ ] **Step 4: Skip commit**

---

## Self-review (plan vs spec)

1. **Spec coverage:** §1–3 entregable y fuera de alcance → T1 overwrite + T7/T8 grep tiendas. §4 helpers y SEED → T1. §5.1 diabetes 353/89 → T2. §5.2 sintético n=80, sin(2πx), ruido 0.25 → T4. §5.3 cáncer estratificado y etiquetas maligno/benigno → T7. §5.4 fugas → T5/T6 asserts `X_train` y T8. §6 pasos 1–6 + cierre → T2–T7. §7 orden de celdas → conteos 2/5/9/15/19/24/30. §8 max_iter, include_bias, figsize → código de T1/T4/T7. §9 ejecución y figuras → T8. §10 criterios → T8 checklist. Enmienda R² ajustado NaN → T1 helper.
2. **Placeholders:** no hay TBD/TODO ni “similar to Task N”; cada append trae el código completo. Commits sustituidos por skip explícito (spec §2).
3. **Nombres:** `metricas_regresion`, `modelos_cv`, `rmse_cv`, `resumen_cv`, `grid`, `rs`, `ganador_nombre`, `grado_optimo` coinciden entre tareas.
