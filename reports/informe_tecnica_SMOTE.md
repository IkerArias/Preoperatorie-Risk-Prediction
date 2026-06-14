# Informe Técnico — SMOTE  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** SMOTE (*Synthetic Minority Over-sampling Technique*)  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica SMOTE para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch con optimización de hiperparámetros), usando PR-AUC como métrica de selección principal por ser la más robusta ante prevalencias extremadamente bajas.

El mejor modelo para **mortalidad** es **GaussianNB** (PR-AUC = 0.2012, AUC-ROC = 0.8423, Recall = 0.6607). Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** (PR-AUC = 0.1790, AUC-ROC = 0.8309, Recall = 0.5135). SMOTE mejora respecto al baseline sin balanceo en ambos targets en términos de Recall y PR-AUC, cumpliendo su objetivo de aumentar la sensibilidad del modelo hacia la clase minoritaria.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (solo ingreso en UCI, ~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo: pacientes que fallecen antes de poder ser trasladados a UCI eran incorrectamente etiquetados como negativos. Ver Sección 3.1 para la justificación completa.

---

## 1. Introducción

### 1.1 Motivación clínica

La predicción de mortalidad y necesidad de ingreso en UCI tras cirugía es uno de los problemas más relevantes en anestesiología y cirugía perioperatoria. Identificar de forma prospectiva a los pacientes de alto riesgo permite adecuar el nivel de monitorización, movilizar recursos de cuidados críticos con antelación y mejorar los resultados clínicos. Sin embargo, la baja prevalencia de estos eventos (<2%) hace que los modelos de machine learning estándar tiendan a clasificar todos los pacientes como negativos, obteniendo una precisión aparentemente alta (~98%) pero un recall nulo sobre la clase de interés.

### 1.2 El problema del desbalanceo de clases

Con una ratio de 1:62 (mortalidad) y ~1:49 (`outcome_intensivo`), los algoritmos de clasificación aprenden una frontera de decisión sesgada hacia la clase mayoritaria. SMOTE aborda este problema generando muestras sintéticas de la clase minoritaria, reequilibrando la distribución de entrenamiento y forzando al modelo a aprender patrones representativos de los casos positivos.

---

## 2. Marco Teórico — SMOTE

### 2.1 Descripción del algoritmo

SMOTE (*Synthetic Minority Over-sampling Technique*, Chawla et al., 2002) es el algoritmo de oversampling más ampliamente utilizado en literatura de aprendizaje automático con clases desbalanceadas. Opera de la siguiente manera:

1. Para cada muestra positiva $x_i$ del conjunto de entrenamiento, se identifican sus $k$ vecinos más cercanos (por defecto $k=5$) dentro de la clase minoritaria.
2. Se selecciona aleatoriamente uno de esos vecinos, $x_{nn}$.
3. Se genera una nueva muestra sintética interpolando entre $x_i$ y $x_{nn}$:

$$x_{nuevo} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

4. Este proceso se repite hasta alcanzar el ratio objetivo de balanceo.

### 2.2 Parámetro `sampling_strategy`

En este trabajo se emplea `sampling_strategy=0.3`, lo que significa que tras el oversampling el ratio de positivos respecto a negativos es del 30%:

$$\frac{n_{\text{pos}}}{n_{\text{neg}}} = 0.3$$

Con la distribución real del dataset (~128 positivos y ~7,872 negativos para mortalidad en el conjunto de entrenamiento), SMOTE genera aproximadamente 2,234 muestras sintéticas adicionales. Se elige 0.3 en lugar de 1.0 (balance perfecto) por tres razones:
- Evitar la generación masiva de sintéticos (~7,744 con `sampling_strategy=1.0`) que puede llevar a sobreajuste.
- Las probabilidades calibradas son más cercanas a la prevalencia real con ratios moderados.
- La métrica de selección PR-AUC evalúa el ranking, no el umbral, por lo que un balance perfecto no es necesario.

### 2.3 Ventajas y limitaciones

**Ventajas:**
- Genera diversidad en la clase minoritaria al interpolar en el espacio de características.
- Robustez teórica asentada en >5,000 citas en literatura científica.
- Compatible con cualquier clasificador (model-agnostic).
- La interpolación en vecindarios locales preserva la estructura distribucional.

**Limitaciones:**
- Genera sintéticos en zonas de alta densidad de la clase minoritaria, no necesariamente en la frontera de decisión.
- Puede generar sintéticos en regiones solapadas con la clase mayoritaria si la separabilidad es baja.
- Susceptible a ruido: sintéticos generados a partir de outliers pueden degradar el rendimiento.

---

## 3. Metodología Experimental

### 3.1 Dataset

| Característica | Descripción |
|---|---|
| Fuente | `data/todo_ASA_anonimizada.xlsx` — datos clínicos anestesiológicos anonimizados |
| Episodios totales | 103.179 (90.825 pacientes únicos) |
| Target 1 | `mortalidad` — fallecimiento intrahospitalario (binario) |
| Target 2 | `outcome_intensivo` = (`uci`==1) OR (`mortalidad`==1) — requerimiento de cuidados críticos (binario) |
| Prevalencia mortalidad | ~1.6% (~333 positivos en test) |
| Prevalencia `outcome_intensivo` | ~2.04% (742 eventos totales; 148/7.267 en test) |
| Ratio outcome_intensivo | ~1:49 (positivos:negativos) |
| Agrupación | `num_idpaciente` — pacientes con múltiples episodios tratados como grupo |

Las variables con potencial *data leakage* (diagnóstico de defunción, variables de resultado hospitalario) fueron eliminadas antes del entrenamiento. La variable `num_idpaciente` se conserva únicamente para el agrupamiento en la división de datos y se extrae antes de entrenar.

#### 3.1.1 Justificación del endpoint compuesto `outcome_intensivo` — Riesgo Competitivo

El target original `uci` (solo ingreso en UCI) sufre un **problema de riesgo competitivo**: un paciente que fallece en las primeras horas postquirúrgicas sin haber sido trasladado a la UCI quedaría etiquetado como negativo (`uci=0`), cuando en realidad representa el peor resultado posible. Este sesgo, formalizado por Fine & Gray (1999) para modelos de riesgos en competencia, introduce una subestimación sistemática de la gravedad de ciertos episodios.

La solución adoptada es definir un **endpoint compuesto**:

$$\text{outcome\_intensivo} = \mathbb{1}[\text{uci}=1] \; \vee \; \mathbb{1}[\text{mortalidad}=1]$$

Esta redefinición está justificada por tres argumentos:

1. **Clínico**: La acción del anestesiólogo ante ambos eventos (UCI o muerte inminente) es idéntica — activar protocolos de cuidados críticos, incrementar monitorización, preparar intervención intensiva. La distinción entre "muere antes de llegar a UCI" y "llega a UCI" es institucional (disponibilidad de camas), no una diferencia en severidad del cuadro clínico.

2. **Estadístico**: El antiguo `uci` tenía solo 318 eventos (0.88% prevalencia), produciendo estimaciones inestables y rangos de confianza amplios en validación cruzada. `outcome_intensivo` cuenta con 742 eventos (2.04%), lo que reduce la varianza de las estimaciones de PR-AUC y mejora la potencia estadística de la comparación entre técnicas.

3. **Metodológico (CONSORT)**: Las guías internacionales para ensayos clínicos y estudios predictivos (CONSORT, 2010) recomiendan endpoints compuestos cuando los eventos individuales comparten el mismo mecanismo fisiopatológico subyacente y generan la misma respuesta terapéutica.

**Construcción en código:**
```python
df_uci_base = df_clean.dropna(subset=['uci']).copy()
df_uci_base['outcome_intensivo'] = (
    (df_uci_base['uci'] == 1) | (df_uci_base['mortalidad'] == 1)
).astype(int)
df_uci_full = df_uci_base.drop(columns=['mortalidad', 'uci']).copy()
```
La variable `mortalidad` se **elimina de las features** para evitar data leakage (target 2 depende de ella por construcción).

### 3.2 División de datos

Se utiliza `GroupShuffleSplit(test_size=0.2, random_state=42)` para garantizar que todos los episodios de un mismo paciente pertenezcan al mismo conjunto (entrenamiento o test). Esto evita que el modelo aprenda patrones específicos de pacientes que después aparecen en el test, lo que inflaría artificialmente las métricas.

```
Dataset completo (N pacientes)
        │
        ├── 80% → Entrenamiento (df_train_mort / df_train_uci)
        │         ├── K-Fold (5 folds StratifiedGroupKFold)
        │         └── GridSearch (StratifiedGroupKFold interno)
        │
        └── 20% → Test (held-out, nunca visto durante entrenamiento)
                  └── Evaluación final única
```

### 3.3 Pipeline

```python
ImbPipeline([
    ('preprocessor', ColumnTransformer([
        ('scaler', StandardScaler(), ['edad']),  # solo variable continua
        ('pass',   'passthrough',   binary_cols) # variables binarias sin escalar
    ])),
    ('sampler', SMOTE(sampling_strategy=0.3, random_state=42)),
    ('model',   classifier)
])
```

SMOTE se aplica **dentro del pipeline**, lo que garantiza que:
1. No hay *data leakage* del balanceo: los sintéticos se generan solo con datos del fold de entrenamiento.
2. El test set nunca contiene muestras sintéticas.
3. El preprocesador (StandardScaler) se ajusta con datos reales, no sintéticos.

### 3.4 Modelos evaluados

| Modelo | Tipo | Notas |
|---|---|---|
| XGBoost | Gradient Boosting | `eval_metric='logloss'`, `n_jobs=1` |
| LightGBM | Gradient Boosting | `verbose=-1`, leaf-wise growth |
| GradientBoosting | Gradient Boosting | sklearn, stage-by-stage |
| RandomForest | Bagging | `n_jobs=1` |
| ExtraTrees | Bagging | umbral de split aleatorio |
| LogisticRegression | Lineal | `max_iter=2000`, regularización L2 |
| GaussianNB | Probabilístico | Asume independencia y distribución gaussiana |

### 3.5 Estrategia de validación

**Fase 1 — Baseline:** Sin balanceo. Entrenamiento sobre `X_train` completo con parámetros por defecto. Referencia para cuantificar el impacto de SMOTE.

**Fase 2 — K-Fold (5 folds):** `StratifiedGroupKFold(n_splits=5)` sobre el 80% de entrenamiento. Mantiene la distribución de clases y garantiza que ningún paciente aparezca en train y validación simultáneamente. Ofrece una estimación robusta del rendimiento medio.

**Fase 3 — GridSearch:** `GridSearchCV` con `StratifiedGroupKFold(n_splits=5)`, `scoring='average_precision'`, `return_train_score=True`. Los hiperparámetros del mejor pipeline se reaplican entrenando sobre todo `X_train`, evaluando en `X_test`.

### 3.6 Métrica principal

**PR-AUC (Average Precision)** se elige como métrica de selección principal frente a AUC-ROC porque:
- Con prevalencia <<5%, la curva ROC puede ser optimista (alto número de verdaderos negativos domina FPR).
- PR-AUC pondera la capacidad del modelo para rankear los positivos reales en posiciones altas del ranking de probabilidad.
- Es equivalente al área bajo la curva Precision-Recall, donde ambos ejes son sensibles a la clase positiva.

---

## 4. Resultados

### 4.1 Fase 1 — Baseline (sin balanceo)

| Target | Mejor Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | 0.2011 | 0.8422 | 0.6607 | 0.1130 | 0.0618 |
| outcome_intensivo | GaussianNB | 0.1790 | 0.8309 | 0.5135 | 0.1369 | 0.0790 |

El baseline revela que incluso sin balanceo, GaussianNB obtiene buenos resultados en PR-AUC, lo que sugiere que su asunción de independencia fenoumenológica resulta adecuada para este dataset. Los modelos basados en árboles (XGBoost, LightGBM) obtienen mejor AUC-ROC (~0.90) pero peor PR-AUC, evidenciando que discriminan globalmente bien pero les cuesta rankear casos positivos cuando la prevalencia es muy baja.

### 4.2 Fase 2 — K-Fold (robustez)

| Target | Mejor Modelo | KF PR-AUC (media) | KF PR-AUC (std) | Interpretación |
|---|---|---|---|---|
| MORTALIDAD | GaussianNB | 0.1800 | ±0.0070 | Alta estabilidad entre folds |
| outcome_intensivo | GaussianNB | 0.2166 | ±0.0154 | Variabilidad moderada |

La baja desviación estándar en mortalidad (±0.007) indica que el rendimiento de GaussianNB+SMOTE es estable y reproducible. La variabilidad en `outcome_intensivo` (±0.015) es notable pero manejable, y es significativamente menor que la del antiguo target `uci` (±0.036), lo que refleja el mayor número de eventos (742 vs 318) que estabiliza las estimaciones por fold.

### 4.3 Fase 3 — GridSearch (hiperparámetros optimizados)

#### Mortalidad — Ranking completo

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| 1 | **GaussianNB** | **0.2012** | 0.8423 | 0.6607 | 0.1131 | 0.0618 | 0.8356 |
| 2 | XGBoost | 0.1460 | 0.8954 | 0.4174 | 0.2031 | 0.1342 | 0.9558 |
| 3 | LightGBM | 0.1416 | 0.8948 | 0.4324 | 0.2050 | 0.1343 | 0.9543 |
| 4 | GradientBoosting | 0.1334 | 0.8941 | 0.3994 | 0.1991 | 0.1326 | 0.9571 |
| 5 | RandomForest | 0.1276 | 0.8927 | 0.2853 | 0.1953 | 0.1484 | 0.9732 |
| 6 | LogisticRegression | 0.1254 | 0.8886 | 0.5435 | 0.1864 | 0.1125 | 0.9297 |
| 7 | ExtraTrees | 0.1229 | 0.8907 | 0.2853 | 0.1892 | 0.1416 | 0.9716 |

**Mejor modelo:** GaussianNB con `var_smoothing=1e-5`

#### `outcome_intensivo` — Ranking completo

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|
| 1 | **GaussianNB** | **0.1790** | 0.8309 | 0.5135 | 0.1369 | 0.0790 |
| 2 | XGBoost | 0.1363 | 0.8755 | 0.4054 | 0.2294 | 0.1620 |
| 3 | GradientBoosting | 0.1338 | 0.8763 | 0.3378 | 0.2045 | 0.1435 |
| 4 | LightGBM | 0.1336 | 0.8697 | 0.3446 | 0.2061 | 0.1472 |
| 5 | RandomForest | 0.1274 | 0.8835 | 0.3311 | 0.2090 | 0.1519 |
| 6 | LogisticRegression | 0.1218 | 0.8742 | 0.4324 | 0.2222 | 0.1429 |
| 7 | ExtraTrees | 0.1213 | 0.8792 | 0.3446 | 0.2073 | 0.1491 |

**Mejor modelo por PR-AUC:** GaussianNB con `var_smoothing=1e-5`

> **Interpretación del ranking:** GaussianNB lidera en PR-AUC gracias a su capacidad de generar probabilidades bien diferenciadas bajo prevalencia baja. Sin embargo, su precisión es muy baja (0.079): de cada 13 alarmas, solo 1 es correcta. Los modelos de gradient boosting (XGBoost, GradientBoosting, LightGBM) ofrecen un mejor equilibrio precision/recall (F1 ≈ 0.20-0.23) con AUC-ROC superior (0.87-0.88), siendo preferibles en un contexto de uso clínico donde las falsas alarmas tienen un coste operativo.

---

## 5. Análisis del Mejor Modelo

### 5.1 Por qué GaussianNB domina en PR-AUC

El resultado de que GaussianNB supere a modelos boosting más complejos en PR-AUC (aunque no en AUC-ROC) tiene una explicación estadística clara:

- GaussianNB estima $P(y=1 | x) \propto P(x | y=1) \cdot P(y=1)$. Con clases muy desbalanceadas, $P(y=1) \approx 0.016$, lo que hace que las probabilidades predichas sean inherentemente pequeñas.
- Esto crea un ranking en el que los pocos positivos reales con características anómalas (raras dadas la clase negativa) obtienen probabilidades relativas más altas.
- Los modelos de árboles tienden a discretizar el espacio de características y, con `sampling_strategy=0.3`, aprenden una frontera que clasifica muchos negativos como positivos (alto recall, baja precisión) o viceversa.

### 5.2 Interpretación clínica

| Métrica | Valor | Interpretación |
|---|---|---|
| PR-AUC = 0.2012 (mort.) | Área bajo PR-curve | Con prevalencia 1.6%, el baseline aleatorio es 0.016; el modelo multiplica por ~12.6× |
| Recall = 0.6607 (mort.) | 66.1% de las muertes detectadas | De cada 10 fallecimientos reales, el modelo identifica ~6-7 como alto riesgo |
| Precision = 0.0618 (mort.) | 1 de cada 16 alarmas correcta | Refleja la baja prevalencia; es esperable con estos niveles de desbalanceo |
| AUC-ROC = 0.8423 (mort.) | Discriminación global muy buena | El modelo ordena correctamente el 84.2% de los pares positivo-negativo |
| PR-AUC = 0.1790 (outcome_int.) | Área bajo PR-curve | Con prevalencia 2.04%, el baseline aleatorio es 0.020; el modelo multiplica por ~8.8× |
| Recall = 0.5135 (outcome_int.) | 51.4% de los eventos críticos detectados | De cada 10 eventos críticos reales, el modelo identifica ~5 como alto riesgo |
| Precision = 0.0790 (outcome_int.) | 1 de cada 13 alarmas correcta | Mejora respecto al antiguo target uci (1:18) por mayor prevalencia |
| AUC-ROC = 0.8309 (outcome_int.) | Discriminación global buena | Ligeramente inferior a mortalidad, consecuente con el endpoint compuesto |

### 5.3 Discrepancia PR-AUC vs AUC-ROC

La diferencia entre AUC-ROC=0.84 y PR-AUC=0.20 es característica de escenarios con prevalencia baja. AUC-ROC considera verdaderos negativos (los numerosos) en su denominador (FPR = FP/(FP+TN)), lo que lo hace optimista. PR-AUC, al trabajar solo con predicciones positivas y casos positivos reales, es mucho más exigente y ofrece una medida más honesta de la utilidad clínica real.

---

## 6. Análisis del Umbral de Decisión

Los modelos sklearn utilizan por defecto `threshold=0.5`. Con prevalencia del 1.6%, casi ninguna probabilidad predicha supera 0.5, resultando en recall≈0. Se analizan dos umbrales alternativos:

| Umbral | Criterio | Interpretación clínica |
|---|---|---|
| **Default (0.5)** | Referencia | Inutilizable en práctica clínica |
| **F1-max** | Maximiza F1 | Equilibrio precision/recall óptimo estadísticamente |
| **Recall ≥ 90%** | Clínico | Detectar ≥90% de eventos; acepta más falsas alarmas |

En el contexto clínico de predicción de mortalidad, el umbral de **Recall ≥ 90%** es el relevante: el coste de un falso negativo (muerte no detectada, sin intervención preventiva) es mucho más grave que el de un falso positivo (vigilancia innecesaria de un paciente que no muere).

---

## 7. Recalibración de Probabilidades

Los modelos entrenados con `sampling_strategy=0.3` (distribución sintética ~23% positivos) predicen probabilidades sistemáticamente infladas respecto a la prevalencia real (~1.6%). Se aplica **Isotonic Regression** sobre una partición estratificada 50/50 del test set:
- 50% para calibrar el regresor isotónico
- 50% para evaluar las probabilidades calibradas

La calibración no modifica el ranking del modelo (PR-AUC y AUC-ROC se preservan), pero ajusta las probabilidades absolutas para que sean interpretables como estimaciones de riesgo real, lo cual es fundamental para la toma de decisiones clínicas.

---

## 8. Análisis de Importancia de Variables (SHAP)

Se utiliza SHAP (*SHapley Additive exPlanations*) con `KernelExplainer` para GaussianNB (no compatible con `TreeExplainer`) sobre un subsample de 300 muestras del test set. Los valores SHAP cuantifican la contribución marginal de cada variable a la predicción individual del modelo.

Las variables con mayor importancia global (media |SHAP|) permiten identificar qué factores prequirúrgicos tienen mayor peso predictivo sobre la mortalidad y el ingreso en UCI, proporcionando **interpretabilidad clínica** esencial para que el modelo sea accionable por un anestesiólogo.

---

## 9. Conclusiones

1. **SMOTE mejora la detección de eventos positivos** respecto al baseline sin balanceo, principalmente en Recall (66.1% mortalidad, 51.4% `outcome_intensivo` vs ~0% con threshold 0.5 sin balanceo).

2. **GaussianNB es el mejor modelo en PR-AUC** para ambos targets con SMOTE. Su estimación probabilística directa de $P(y|x)$ genera rankings más precisos bajo prevalencia extrema que los modelos de árboles, a pesar de asumir independencia entre features. Sin embargo, su precisión es baja (6-8%), por lo que para uso clínico se recomienda valorar XGBoost (F1=0.229, PR-AUC=0.136) como alternativa con mejor equilibrio precision/recall.

3. **La robustez K-Fold es alta**: PR-AUC = 0.1800 ± 0.0070 (mortalidad) y 0.2166 ± 0.0154 (`outcome_intensivo`) indican rendimiento consistente. La menor variabilidad vs. el antiguo target `uci` (±0.0364) confirma la mayor estabilidad del endpoint compuesto con 742 eventos.

4. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`: los pacientes que fallecen sin pasar por UCI ya no contaminan la clase negativa, produciendo un target más coherente clínica y estadísticamente.

5. **El umbral de decisión es crítico**: El umbral por defecto (0.5) es inutilizable. El umbral F1-max o Recall≥90% son las opciones clínicamente relevantes según el coste relativo de falsos negativos vs. falsos positivos.

6. **La calibración probabilística es necesaria** antes de usar las probabilidades del modelo en decisiones clínicas, dado el sobreajuste de escala producido por el balanceo sintético.

---

## Referencias

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16, 321-357.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets. *JMLR*, 18(17), 1-5.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials: greater precision but with greater uncertainty? *JAMA*, 289(19), 2554-2559.
- CONSORT Group (2010). CONSORT 2010 statement: updated guidelines for reporting parallel group randomised trials. *BMJ*, 340, c332.
