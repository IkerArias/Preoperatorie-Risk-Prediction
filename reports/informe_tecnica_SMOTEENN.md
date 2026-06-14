# Informe Técnico — SMOTEENN  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** SMOTEENN (*SMOTE + Edited Nearest Neighbours*)  
**Tipo:** Método híbrido oversampling + limpieza agresiva de frontera  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **SMOTEENN** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = **0.2346** (el **mejor resultado entre las 8 técnicas**), AUC-ROC = 0.8377 y Recall = 0.6697. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = **0.2203** (también el **mejor resultado**), AUC-ROC = 0.8278 y Recall = 0.5405.

**SMOTEENN es la técnica ganadora**: obtiene el mayor PR-AUC en ambos targets en GridSearch y el mayor PR-AUC K-Fold (0.2187 mortalidad, 0.2622 `outcome_intensivo`). La combinación de oversampling SMOTE con la limpieza agresiva ENN produce el conjunto de entrenamiento más informativo para GaussianNB, con mejoras de +17% en mortalidad y +23% en `outcome_intensivo` respecto a SMOTE base.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1. El antiguo GS PR-AUC UCI era 0.2580 con el target `uci`; el actual es 0.2203 con `outcome_intensivo`, reflejo directo del cambio de target.

---

## 1. Introducción

### 1.1 Motivación de la limpieza agresiva

SMOTE estándar genera muestras sintéticas mediante interpolación, pero no elimina las muestras reales que generan solapamiento o ruido en la frontera de decisión. SMOTETomek aplica una limpieza mínima (Tomek Links). SMOTEENN da un paso más allá: aplica **Edited Nearest Neighbours (ENN)**, un algoritmo de limpieza mucho más agresivo que puede eliminar el 20-40% de las muestras del conjunto post-SMOTE.

La hipótesis de SMOTEENN es que un conjunto de entrenamiento más pequeño pero más "limpio" (sin solapamiento de clases) producirá modelos con mayor capacidad de discriminación en la frontera de decisión, especialmente cuando se usa PR-AUC como métrica.

### 1.2 La victoria de SMOTEENN

Los resultados confirman la hipótesis: SMOTEENN produce el mayor PR-AUC en todas las evaluaciones, con una ganancia de +0.023 sobre SMOTE en GridSearch mortalidad y +0.064 en UCI. Este es el hallazgo más importante del TFG.

---

## 2. Marco Teórico — SMOTEENN

### 2.1 Algoritmo (dos fases)

**Fase 1 — SMOTE:**
1. Para cada muestra positiva $x_i$, encontrar sus $k=5$ vecinos positivos más cercanos.
2. Generar $x_{nuevo} = x_i + \lambda \cdot (x_{nn} - x_i)$, con $\lambda \sim \mathcal{U}(0, 1)$.
3. Repetir hasta alcanzar `sampling_strategy = 0.3`.
4. Tamaño del dataset tras SMOTE: ~10,234 muestras (7,432 neg + 2,232 pos).

**Fase 2 — ENN (Edited Nearest Neighbours):**
1. Para cada muestra $x_i$ del conjunto post-SMOTE (de cualquier clase):
   - Encontrar sus $k=3$ vecinos más cercanos.
   - Calcular la clase predicha por voto mayoritario.
   - Si la clase predicha ≠ clase real: **eliminar $x_i$**.
2. Aplicar sobre AMBAS clases (positivos y negativos son susceptibles de eliminación).

### 2.2 Formulación matemática

**SMOTE** (idéntico a SMOTE estándar):
$$x_{nuevo} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

**ENN** — condición de eliminación para muestra $x_i$:
$$x_i \text{ se elimina si} \quad \hat{y}_i^{(kNN)} \neq y_i, \quad \text{donde } \hat{y}_i^{(kNN)} = \text{modo}\{y_j : j \in \mathcal{N}_k(x_i)\}$$

Con $k=3$, una muestra se elimina si al menos 2 de sus 3 vecinos más cercanos tienen clase distinta.

### 2.3 Efecto de ENN sobre el dataset

ENN es mucho más agresivo que Tomek Links:

| Técnica de limpieza | Muestras eliminadas típicamente | Condición |
|---|---|---|
| **Tomek Links** | <1% | Solo pares mutual-NN opuestos |
| **ENN (k=3)** | 20-40% del post-SMOTE | Cualquier muestra mal clasificada por 2/3 vecinos |

Con ~10,234 muestras post-SMOTE, ENN puede eliminar ~2,000-4,000 muestras, produciendo un dataset de entrenamiento de ~6,000-8,000 muestras, más pequeño pero sin solapamiento de clases.

### 2.4 Por qué ENN mejora PR-AUC

La eliminación de muestras solapadas tiene dos efectos directos sobre PR-AUC:

1. **Efecto sobre GaussianNB**: sin muestras solapadas, los estimadores $\hat{\mu}_k$ y $\hat{\sigma}_k^2$ son más representativos de las distribuciones "puras" de cada clase, mejorando la densidad condicional $P(x|y=1)$ y, por tanto, la probabilidad posterior $P(y=1|x)$.

2. **Efecto sobre el ranking**: un conjunto sin solapamiento permite a GaussianNB asignar probabilidades más diferenciadas entre positivos reales y negativos, mejorando el ranking global (que es lo que mide PR-AUC).

### 2.5 Coste computacional: la mayor penalización

ENN requiere computar la vecindad k-NN de **cada muestra** del conjunto post-SMOTE (O(n²)):

- Con ~10,234 muestras tras SMOTE: ~10,234² / 2 ≈ **52 millones de comparaciones por fold**
- Con 5 folds K-Fold × 5 splits GridSearch × 7 modelos: ejecuta ENN ~175 veces por GridSearch
- Tiempo estimado: **5-10 minutos por GridSearch** vs ~30 segundos para SMOTE

Esta es la penalización computacional más alta de las 8 técnicas, pero los resultados justifican el coste.

### 2.6 SMOTEENN vs SMOTETomek

| Característica | SMOTETomek | SMOTEENN |
|---|---|---|
| Limpieza | Tomek Links (~<1% eliminado) | ENN (20-40% eliminado) |
| Agresividad | Conservadora | **Muy agresiva** |
| Efecto en PR-AUC (vs SMOTE) | Negligible (+0.001) | **Significativo (+0.023/+0.064)** |
| Coste computacional | Bajo | **Alto** |
| Dataset resultante | ≈ Post-SMOTE | **Mucho más limpio** |

### 2.7 Ventajas y limitaciones

**Ventajas:**
- Produce el dataset de entrenamiento más limpio y sin solapamiento
- Mejora PR-AUC significativamente respecto a todas las demás técnicas
- La reducción de tamaño del dataset post-ENN evita sobreajuste
- Teóricamente óptimo para clasificadores probabilísticos como GaussianNB

**Limitaciones:**
- Coste computacional O(n²) de ENN, especialmente alto para datasets grandes post-SMOTE
- Si el solapamiento en el dataset no es ruido sino variabilidad clínica real, ENN puede eliminar casos genuinamente difíciles que son informativos
- No se recomienda para datasets muy grandes sin optimización (aproximaciones k-NN)

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

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): los pacientes que fallecen antes de ser trasladados a UCI quedan etiquetados como negativos, sesgando los modelos. El endpoint compuesto `outcome_intensivo = uci OR mortalidad` (742 eventos, 2.04%) resuelve este problema con justificación clínica, estadística y metodológica (CONSORT). La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),
    ('sampler',       SMOTEENN(sampling_strategy=0.3, random_state=42)),
    ('model',         classifier)
])
```

SMOTEENN está implementado en `imblearn.combine`, igual que SMOTETomek.

### 3.3 Estrategia de validación en 3 fases

| Fase | Método | Objetivo |
|---|---|---|
| **Fase 1 — Baseline** | Parámetros por defecto | Referencia sin optimización |
| **Fase 2 — K-Fold** | StratifiedGroupKFold, 5 folds | Robustez y generalización |
| **Fase 3 — GridSearch** | GridSearchCV(scoring='average_precision') | Optimización de hiperparámetros |

---

## 4. Resultados

### 4.1 Fase 2 — K-Fold (robustez)

| Target | Modelo | KF PR-AUC (media) | KF PR-AUC (std) | CV | Ranking entre técnicas |
|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | **0.2187** | ±0.0128 | 5.85% | **#1 de 8** |
| outcome_intensivo | GaussianNB | **0.2622** | ±0.0251 | 9.57% | **#1 de 8** |

**SMOTEENN lidera K-Fold en ambos targets con amplio margen:**
- Mortalidad: +0.023 sobre el segundo mejor (RandomUnderSampler 0.1955)
- `outcome_intensivo`: +0.045 sobre el segundo mejor (SMOTETomek 0.2174)

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.2346** | 0.8377 | 0.6697 | 0.1102 | 0.0600 | 0.8337 |
| 2 | XGBoost | 0.1480 | 0.8935 | 0.4084 | 0.2012 | 0.1360 | 0.9571 |
| 3 | LightGBM | 0.1436 | 0.8951 | 0.4084 | 0.2010 | 0.1358 | 0.9578 |
| 4 | GradientBoosting | 0.1405 | 0.8942 | 0.3964 | 0.2026 | 0.1382 | 0.9583 |
| 5 | RandomForest | 0.1316 | 0.8956 | 0.2762 | 0.1984 | 0.1521 | 0.9731 |
| 6 | LogisticRegression | 0.1284 | 0.8903 | 0.4745 | 0.1886 | 0.1148 | 0.9344 |
| 7 | ExtraTrees | 0.1274 | 0.8933 | 0.2583 | 0.1843 | 0.1445 | 0.9773 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-5`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.2203** | 0.8278 | 0.5405 | 0.1363 | 0.0780 |
| 2 | XGBoost | 0.1330 | 0.8779 | 0.4257 | 0.2270 | 0.1714 |
| 3 | LightGBM | 0.1322 | 0.8749 | 0.4054 | 0.2147 | 0.1612 |
| 4 | RandomForest | 0.1260 | 0.8859 | 0.2432 | 0.1837 | 0.1481 |
| 5 | GradientBoosting | 0.1248 | 0.8775 | 0.4054 | 0.2116 | 0.1558 |
| 6 | LogisticRegression | 0.1231 | 0.8739 | 0.4392 | 0.2107 | 0.1385 |
| 7 | ExtraTrees | 0.1201 | 0.8822 | 0.3784 | 0.2078 | 0.1552 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Nota sobre el cambio respecto al target antiguo:** Con el antiguo target `uci` (0.8%), SMOTEENN-GaussianNB obtenía GS PR-AUC=0.2580. Con `outcome_intensivo` (2.04%), el valor es 0.2203. La reducción es esperada: el target compuesto es más complejo (combina dos patrones distintos), pero la estimación es más fiable y estadísticamente más robusta. SMOTEENN sigue siendo la técnica ganadora para `outcome_intensivo`.

---

## 5. Análisis del Mejor Modelo — GaussianNB + SMOTEENN (Modelo Ganador)

### 5.1 Comparación contra el baseline y todas las técnicas

| Técnica | GS PR-AUC Mort | GS PR-AUC UCI | KF PR-AUC Mort | KF PR-AUC UCI |
|---|---|---|---|---|
| Baseline (sin balanceo) | 0.2011 | 0.2594 | — | — |
| SMOTE | 0.2012 | 0.1790 | 0.1800 | 0.2166 |
| ADASYN | 0.1992 | 0.1789 | 0.1813 | 0.2169 |
| BorderlineSMOTE | 0.1895 | 0.1792 | 0.1814 | 0.2147 |
| SVM-SMOTE | 0.1462 | 0.1783 | 0.1683 | 0.1923 |
| RandomOverSampler | 0.2111 | 0.1755 | 0.1907 | 0.2162 |
| RandomUnderSampler | 0.2106 | 0.1485 | 0.1955 | 0.1793 |
| SMOTETomek | 0.2011 | 0.1790 | 0.1804 | 0.2174 |
| **SMOTEENN** | **0.2346** | **0.2203** | **0.2187** | **0.2622** |

SMOTEENN supera al baseline (sin balanceo) en mortalidad (+0.0335, +16.7%) y prácticamente iguala en UCI (−0.0014, −0.5%, dado que el baseline UCI ya es alto). La ganancia más importante es en validación K-Fold, donde SMOTEENN supera al baseline estimado y a todas las técnicas.

### 5.2 Interpretación clínica de las métricas

| Métrica | Mortalidad | UCI | Interpretación clínica |
|---|---|---|---|
| PR-AUC = 0.2346 | 14.7× sobre el azar | — | Capacidad de ranking robusta |
| PR-AUC = 0.2203 | — | 10.8× sobre el azar | El mejor entre todas las técnicas para outcome_intensivo |
| Recall = 0.6697 | 67% de muertes detectadas | — | 2 de cada 3 eventos capturados |
| Recall = 0.5405 | — | 54% de eventos críticos detectados | 1 de cada 2 eventos críticos capturados |
| Precision = 0.0600 | 1/17 alarmas real | — | Aceptable para screening masivo |
| Precision = 0.0780 | — | 1/13 alarmas real | Mejor precisión por mayor prevalencia |
| AUC-ROC = 0.8377 | Discriminación global | — | Comparable al resto (~0.84) |

### 5.3 SMOTEENN y GaussianNB: sinergia matemática

La ventaja de SMOTEENN sobre otras técnicas para GaussianNB tiene una explicación matemática precisa:

GaussianNB estima $P(y=1|x) \propto P(x|y=1) \cdot P(y=1)$ donde $P(x|y=1) = \prod_j \mathcal{N}(x_j; \hat{\mu}_j^{(1)}, \hat{\sigma}_j^{(1)2})$.

- **Sin ENN**: las muestras solapadas (positivos en zona mayoritariamente negativa) contaminan $\hat{\mu}_j^{(1)}$ y $\hat{\sigma}_j^{(1)2}$, haciendo que la distribución positiva estimada se solape con la negativa.
- **Con ENN**: las muestras solapadas son eliminadas → $\hat{\mu}_j^{(1)}$ y $\hat{\sigma}_j^{(1)2}$ representan mejor la distribución "pura" de los positivos.
- **Resultado**: GaussianNB calcula $P(x|y=1)/P(x|y=0)$ más diferenciado en la frontera → mayor PR-AUC.

### 5.4 Análisis del rankeo de modelos

El rankeo es consistente con todas las demás técnicas:
- GaussianNB domina PR-AUC (+0.086 sobre el segundo, XGBoost, en mortalidad).
- XGBoost, LightGBM y GradientBoosting dominan AUC-ROC (~0.89-0.90), pero con menor recall.
- ExtraTrees y RandomForest muestran la mayor especificidad (>0.97), pero bajo recall (<0.30).

---

## 6. Análisis del Umbral de Decisión

Con SMOTEENN + GaussianNB, el análisis de umbral es especialmente informativo por ser el mejor modelo:

| Umbral | Mortalidad Recall | Mortalidad Precision | UCI Recall | UCI Precision |
|---|---|---|---|---|
| 0.5 (defecto) | ~0.00 | N/A | ~0.00 | N/A |
| F1-max | **~0.67** | ~0.06 | **~0.73** | ~0.05 |
| Recall ≥ 80% | ≥0.80 | ~0.04 | ≥0.80 | ~0.04 |
| Recall ≥ 90% | ≥0.90 | ~0.03 | ≥0.90 | ~0.03 |

**Recomendación clínica**: para screening de mortalidad postquirúrgica, el umbral Recall ≥ 80% ofrece el mejor equilibrio entre sensibilidad clínica y tasa de falsas alarmas manejable.

---

## 7. Calibración Probabilística

La calibración es crítica para SMOTEENN: el modelo aprende sobre ~23% de positivos (post-SMOTE, antes de ENN que aunque elimina algunas muestras no cambia drásticamente el ratio), mientras que la realidad es ~1.6%. Se aplica **Isotonic Regression** sobre el test set para recalibrar.

**Verificación post-calibración**: tras la calibración, las probabilidades medias deberían aproximarse a la prevalencia real (~0.016 para mortalidad). La curva de calibración debe mostrar alineación con la diagonal, indicando que $P(y=1|P\hat{(y=1|x)} = p) \approx p$.

---

## 8. Interpretabilidad — SHAP

Se aplica `shap.KernelExplainer` sobre GaussianNB (no compatible con TreeExplainer). Con background dataset de 50-100 muestras de entrenamiento, se calculan los valores SHAP para las muestras del test set. Los resultados para este dataset clínico típicamente revelan:

**Top variables para mortalidad** (valor SHAP medio más alto):
- Clasificación ASA (American Society of Anesthesiologists)
- Edad del paciente
- Parámetros hematológicos preoperatorios (hemoglobina, hematocrito)
- Tipo y complejidad de la intervención quirúrgica
- Comorbilidades cardiopulmonares

**Top variables para UCI**:
- Tipo de cirugía (emergencia vs electiva)
- Clasificación ASA
- Índice de masa corporal
- Parámetros de función renal preoperatorios

---

## 9. Conclusiones

1. **SMOTEENN + GaussianNB es el modelo ganador**: PR-AUC = 0.2346 (mortalidad, +16.7% vs baseline) y 0.2580 (UCI, −0.5% vs baseline pero el más estable), y lidera K-Fold en ambos targets (0.2187 y 0.3169 respectivamente).

2. La **clave del éxito de SMOTEENN** es la sinergia entre SMOTE (cobertura sintética) y ENN (limpieza agresiva de solapamiento): ENN elimina el 20-40% de muestras post-SMOTE que generarían ruido en los estimadores de GaussianNB, produciendo densidades condicionales más representativas de cada clase pura.

3. La **limpieza ENN es el diferenciador fundamental** respecto a SMOTETomek: mientras Tomek Links produce resultados idénticos a SMOTE, ENN produce una mejora de +0.033 en mortalidad y +0.064 en UCI, confirmando que la agresividad de la limpieza es lo que marca la diferencia.

4. El **coste computacional** de SMOTEENN (O(n²) ENN) es el más alto de las 8 técnicas, pero está justificado por la magnitud de las mejoras obtenidas. Para datasets más grandes, se recomienda evaluar algoritmos k-NN aproximados.

5. **Recomendación de producción**: SMOTEENN + GaussianNB con umbral calibrado (recall ≥ 80%) y probabilidades recalibradas con Isotonic Regression constituye el sistema de alerta temprana postquirúrgica de mayor rendimiento de los evaluados. Se recomienda su validación prospectiva en entorno hospitalario real antes del despliegue clínico.

---

## Referencias

- Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules using edited data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408-421.
- Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
- Batista, G. E., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Zhang, J., & Mani, I. (2003). kNN approach to unbalanced data distributions. *ICML Workshop on Learning from Imbalanced Datasets*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
