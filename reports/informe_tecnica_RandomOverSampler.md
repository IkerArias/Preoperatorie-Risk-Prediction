# Informe Técnico — RandomOverSampler  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** RandomOverSampler (*Random Over-Sampling*)  
**Tipo:** Oversampling por duplicación aleatoria de muestras reales  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **RandomOverSampler** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.2111 (el **segundo mejor** entre todas las técnicas para mortalidad, tras SMOTEENN), AUC-ROC = 0.8369 y Recall = 0.6126. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1755, AUC-ROC = 0.8354 y Recall = 0.4122.

**Hallazgo clave:** RandomOverSampler, la técnica más simple posible (duplica muestras reales sin síntesis), logra el **segundo mejor PR-AUC en mortalidad** (0.2111) superando a todas las técnicas sintéticas excepto SMOTEENN. Para `outcome_intensivo`, XGBoost (PR-AUC=0.1456, F1=0.213) supera clínicamente a GaussianNB en equilibrio precision/recall.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1.

---

## 1. Introducción

### 1.1 Motivación y relevancia

El RandomOverSampler es la técnica de referencia más básica para balanceo de clases: duplica aleatoriamente muestras de la clase minoritaria hasta alcanzar el ratio deseado. Su simplicidad es también su mayor fortaleza: no introduce ningún supuesto sobre la distribución de la clase minoritaria ni genera muestras "artificiales" que no existen en los datos reales.

### 1.2 Hipótesis del ensayo

La hipótesis es que, en datasets clínicos con alta variabilidad interindividual y ruido real, la duplicación de muestras existentes puede ser más robusta que la interpolación sintética: no existe riesgo de generar muestras en regiones del espacio de features clínicamente imposibles.

---

## 2. Marco Teórico — RandomOverSampler

### 2.1 Algoritmo

RandomOverSampler (Lemaitre et al., 2017) implementa la estrategia más directa de balanceo de clases:

1. Identificar la clase minoritaria $C_{min}$ con $n_{min}$ muestras.
2. Calcular el número de copias adicionales necesarias: $n_{add} = \lfloor n_{maj} \cdot \text{sampling\_strategy} \rfloor - n_{min}$.
3. **Muestrear aleatoriamente con reemplazamiento** $n_{add}$ muestras de $C_{min}$.
4. Añadir las copias al conjunto de entrenamiento.

El proceso no genera ninguna muestra nueva: solo repite muestras existentes.

### 2.2 Formulación matemática

$$x_{nuevo} = x_k, \quad k \sim \mathcal{U}\{1, \ldots, n_{min}\} \text{ con reemplazamiento}$$

Con `sampling_strategy = 0.3`, el conjunto resultante tendrá $\lfloor n_{maj} \cdot 0.3 \rfloor$ positivos. Con $n_{maj} \approx 7,432$ negativos, se necesitan $\approx 2,230$ positivos. Con 128 originales, se añaden ~2,102 copias (ratio de duplicación ≈ 17.4×).

### 2.3 Comparativa con técnicas sintéticas

| Característica | RandomOverSampler | SMOTE | ADASYN | BorderlineSMOTE |
|---|---|---|---|---|
| Genera muestras nuevas | **No** (solo copias) | Sí (interpoladas) | Sí (adaptativas) | Sí (frontera) |
| Riesgo de muestras imposibles | **Cero** | Bajo-medio | Medio | Bajo |
| Coste computacional | **Mínimo** | Bajo | Bajo | Bajo |
| Información nueva | **Ninguna** | Poca (interpolada) | Media | Media |
| Riesgo de overfitting | **Alto** (muestras repetidas exactas) | Medio | Medio | Medio |
| Necesita vecindad k-NN | **No** | Sí | Sí | Sí |

### 2.4 Riesgo de overfitting por duplicación exacta

El principal riesgo de RandomOverSampler es el **overfitting a muestras específicas**: si una muestra $x_i$ es duplicada 17 veces, el modelo aprende su representación exacta, no generaliza. Sin embargo, con un buen modelo como GaussianNB (que estima densidades en lugar de memorizar ejemplos), este riesgo se mitiga parcialmente.

### 2.5 Ventajas y limitaciones

**Ventajas:**
- No introduce muestras "artificiales": todos los positivos son casos clínicos reales
- Mínimo coste computacional (solo copia objetos en memoria)
- No asume ninguna estructura en el espacio de features
- Sin riesgo de generar muestras en regiones clínicamente imposibles

**Limitaciones:**
- No aporta información nueva: solo repetición (información entropy sin cambio)
- Alto riesgo de overfitting en modelos memorísticos (k-NN, SVM con kernel fino)
- No refuerza la frontera de decisión de forma dirigida
- Con sampling_strategy=0.3, cada muestra positiva aparece en promedio ~17.4 veces

---

## 3. Metodología Experimental

### 3.1 Dataset

| Característica | Descripción |
|---|---|
| Fuente | `data/todo_ASA_anonimizada.xlsx` — datos clínicos anestesiológicos anonimizados |
| Episodios totales | 103.179 (90.825 pacientes únicos) |
| Target 1 | `mortalidad` — fallecimiento intrahospitalario (~1.6%) |
| Target 2 | `outcome_intensivo` = (`uci`==1) OR (`mortalidad`==1) — requerimiento de cuidados críticos (~2.04%, 742 eventos) |
| Ratio outcome_intensivo | ~1:49 (positivos:negativos) |
| Agrupación | `num_idpaciente` — mismos pacientes en train o test, nunca en ambos |
| Split | GroupShuffleSplit(test_size=0.2, random_state=42) |

**Justificación del endpoint compuesto `outcome_intensivo` — Riesgo Competitivo:**

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): los pacientes que fallecen antes de ser trasladados a UCI quedan etiquetados como negativos, produciendo un sesgo sistemático. El endpoint compuesto `outcome_intensivo = uci OR mortalidad` corrige este sesgo. Justificación: (1) la acción clínica ante ambos eventos es idéntica; (2) 742 eventos producen estimaciones significativamente más estables que 318; (3) las guías CONSORT recomiendan endpoints compuestos cuando comparten el mismo mecanismo fisiopatológico. La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),
    ('sampler',       RandomOverSampler(sampling_strategy=0.3, random_state=42)),
    ('model',         classifier)
])
```

### 3.3 Estrategia de validación en 3 fases

| Fase | Método | Objetivo |
|---|---|---|
| **Fase 1 — Baseline** | Parámetros por defecto | Referencia sin optimización |
| **Fase 2 — K-Fold** | StratifiedGroupKFold, 5 folds | Robustez y generalización |
| **Fase 3 — GridSearch** | GridSearchCV(scoring='average_precision') | Optimización de hiperparámetros |

---

## 4. Resultados

### 4.1 Fase 2 — K-Fold (robustez)

| Target | Modelo | KF PR-AUC (media) | KF PR-AUC (std) | CV | Interpretación |
|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | **0.1907** | ±0.0110 | 5.77% | Alta estabilidad |
| outcome_intensivo | GaussianNB | **0.2162** | ±0.0163 | 7.54% | Buena estabilidad |

**Destacable:** La estabilidad K-Fold de RandomOverSampler (±0.0110) es excelente para mortalidad. Para `outcome_intensivo`, la variabilidad (±0.0163) es notablemente menor que la del antiguo target `uci` (±0.0317), gracias al mayor número de eventos (742 vs 318).

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.2111** | 0.8369 | 0.6126 | 0.1122 | 0.0617 | 0.8427 |
| 2 | XGBoost | 0.1397 | 0.8914 | 0.4505 | 0.1982 | 0.1298 | 0.9529 |
| 3 | LightGBM | 0.1352 | 0.8930 | 0.4084 | 0.2025 | 0.1391 | 0.9585 |
| 4 | GradientBoosting | 0.1312 | 0.8913 | 0.4084 | 0.1985 | 0.1366 | 0.9584 |
| 5 | RandomForest | 0.1255 | 0.8928 | 0.2462 | 0.1793 | 0.1383 | 0.9719 |
| 6 | LogisticRegression | 0.1231 | 0.8876 | 0.5015 | 0.1866 | 0.1130 | 0.9340 |
| 7 | ExtraTrees | 0.1169 | 0.8892 | 0.2884 | 0.1826 | 0.1349 | 0.9686 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-7`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1755** | 0.8354 | 0.4122 | 0.1230 | 0.0723 |
| 2 | XGBoost | 0.1456 | 0.8809 | 0.4730 | 0.2131 | 0.1556 |
| 3 | LightGBM | 0.1309 | 0.8795 | 0.5338 | 0.2087 | 0.1449 |
| 4 | GradientBoosting | 0.1301 | 0.8812 | 0.4797 | 0.2107 | 0.1537 |
| 5 | ExtraTrees | 0.1264 | 0.8827 | 0.4122 | 0.2210 | 0.1660 |
| 6 | RandomForest | 0.1241 | 0.8862 | 0.4662 | 0.2351 | 0.1784 |
| 7 | LogisticRegression | 0.1236 | 0.8723 | 0.4324 | 0.2003 | 0.1314 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Interpretación clínica:** Para `outcome_intensivo`, XGBoost (PR-AUC=0.1456, F1=0.213) y LightGBM (PR-AUC=0.131, Recall=0.534) son preferibles a GaussianNB en despliegue clínico por su mejor equilibrio precision/recall. GaussianNB tiene recall bajo (0.41) y precisión marginal (0.072) para este target.

---

## 5. Análisis del Mejor Modelo — GaussianNB + RandomOverSampler

### 5.1 El resultado sorprendente: segundo mejor en mortalidad

RandomOverSampler obtiene PR-AUC=0.2111 en mortalidad, apenas 0.0001 por debajo de RUS (0.2106) y superando a todas las técnicas sintéticas excepto SMOTEENN (0.2346). Este resultado es notable porque:

1. **GaussianNB estima densidades, no memoriza ejemplos**: la duplicación de muestras actualiza los estimadores $\hat{\mu}_k$ y $\hat{\sigma}_k^2$ sin introducir patrones espurios. Las copias simplemente aumentan el peso estadístico de las muestras positivas reales.

2. **La duplicación exacta preserva la distribución original de la clase positiva**: a diferencia de la interpolación (SMOTE), que puede crear muestras en zonas del espacio no representadas por ningún paciente real, la duplicación mantiene la distribución empírica intacta.

3. **Ranking coherente**: si el modelo aprende a separar bien los positivos reales, las copias refuerzan esa separación sin introducir "confusión sintética".

### 5.2 Métricas clínicas

| Métrica | Mortalidad | UCI | Interpretación clínica |
|---|---|---|---|
| PR-AUC | 0.2111 | 0.1755 | 13.2× y 8.6× sobre el azar aleatorio |
| AUC-ROC | 0.8369 | 0.8354 | Discriminación global robusta en ambos targets |
| Recall | 0.6126 | 0.4122 | 61% de muertes detectadas; 41% de eventos críticos detectados |

**Nota sobre el AUC-ROC para UCI** (0.8623, el mayor entre todas las técnicas): RandomOverSampler produce el mayor AUC-ROC para UCI, pero esto no implica mayor PR-AUC. El AUC-ROC alto con PR-AUC bajo indica que el modelo discrimina bien en términos de TN pero es menos preciso en los positivos más difíciles.

### 5.3 Debilidad en UCI: PR-AUC = 0.1784

Para `outcome_intensivo`, RandomOverSampler obtiene PR-AUC=0.1755, valor por debajo de la media de las 8 técnicas. Con el nuevo target `outcome_intensivo` (2.04% prevalencia), la simple duplicación de muestras no captura la complejidad adicional del endpoint compuesto. XGBoost (PR-AUC=0.1456, F1=0.213) es la alternativa clínicamente recomendada para este target.

---

## 6. Análisis del Umbral de Decisión

| Umbral | Mortalidad Recall | Mortalidad Precision | UCI Recall | UCI Precision |
|---|---|---|---|---|
| 0.5 (defecto) | ~0.00 | N/A | ~0.00 | N/A |
| F1-max | ~0.61 | ~0.06 | ~0.69 | ~0.05 |
| Recall ≥ 90% | ≥0.90 | ~0.03 | ≥0.90 | ~0.03 |

---

## 7. Calibración Probabilística

Con RandomOverSampler la calibración es estrictamente necesaria: el modelo aprende sobre una distribución con ~23% de positivos, mientras que la realidad es ~1.6%. Las probabilidades de salida estarán infladas un orden de magnitud. La **Isotonic Regression** calibra estas probabilidades para uso clínico directo.

---

## 8. Conclusiones

1. **RandomOverSampler + GaussianNB** logra el **segundo mejor PR-AUC en mortalidad** (0.2111) entre las 8 técnicas evaluadas, superando a todas las estrategias sintéticas excepto SMOTEENN. Este resultado desafía la intuición de que técnicas más complejas son siempre superiores.

2. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`: los 742 eventos producen estimaciones K-Fold más estables (±0.016 vs ±0.032 anteriores).

3. La **robustez K-Fold** (PR-AUC mortalidad: 0.1907 ± 0.0110; `outcome_intensivo`: 0.2162 ± 0.0163) muestra alta estabilidad, indicando que la duplicación produce distribuciones consistentes entre folds.

4. Para **`outcome_intensivo`**, XGBoost (PR-AUC=0.1456, F1=0.213) supera clínicamente a GaussianNB en equilibrio precision/recall, siendo la mejor opción de despliegue para este target con RandomOverSampler.

5. **Recomendación de uso**: RandomOverSampler es excelente para mortalidad (segundo mejor PR-AUC) y puede usarse en producción. Para `outcome_intensivo`, SMOTEENN o los modelos de boosting con RandomOverSampler son preferibles.

---

## Referencias

- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *JMLR*, 18(17), 1-5.
- Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
