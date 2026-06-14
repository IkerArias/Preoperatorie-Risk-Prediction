# Informe Técnico — SVM-SMOTE  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** SVM-SMOTE (*Support Vector Machine SMOTE*)  
**Tipo:** Oversampling sintético guiado por vectores de soporte  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **SVM-SMOTE** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.1462, AUC-ROC = 0.7899 y Recall = 0.5886. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1783, AUC-ROC = 0.8428 y Recall = 0.4932.

**Hallazgo crítico:** SVM-SMOTE produce el **peor PR-AUC para mortalidad** (0.1462) de todas las técnicas evaluadas, con un AUC-ROC anómalamente bajo (0.7899 vs ~0.84 del resto). Para `outcome_intensivo`, el rendimiento mejora notablemente (PR-AUC=0.1783), siendo la segunda mejor técnica tras SMOTEENN (0.2203). Esta asimetría revela una interacción compleja entre la distribución generada por la SVM interna y el espacio de features del dataset.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1.

---

## 1. Introducción

### 1.1 Motivación clínica

La predicción de mortalidad y requerimiento de cuidados críticos postquirúrgicos requiere modelos con alta sensibilidad hacia la clase minoritaria. Con prevalencias del 1.6% y 2.04% respectivamente, las estrategias de oversampling son imprescindibles. SVM-SMOTE representa la aproximación teóricamente más sofisticada: usa una SVM para identificar la frontera de decisión óptima y concentra los sintéticos precisamente en esa región.

### 1.2 Motivación de SVM-SMOTE

SMOTE estándar, ADASYN y BorderlineSMOTE usan vecindad k-NN (local) para identificar zonas de interés. Sin embargo, la frontera de decisión óptima de un SVM lineal o RBF es **global**: integra información de todo el espacio de features para determinar cuáles son los ejemplos más cercanos al hiperplano de separación. SVM-SMOTE hipotetiza que los sintéticos generados cerca de los vectores de soporte serán más discriminativos.

---

## 2. Marco Teórico — SVM-SMOTE

### 2.1 Algoritmo

SVM-SMOTE (Nguyen et al., 2011) combina una SVM con la estrategia de interpolación de SMOTE:

1. **Entrenar una SVM** sobre el conjunto de entrenamiento completo (con desbalanceo).
2. **Identificar vectores de soporte positivos**: muestras de la clase minoritaria $x_i^{SV}$ que están en el margen del hiperplano SVM.
3. Para cada vector de soporte positivo: encontrar sus $k$ vecinos positivos más cercanos $\hat{x}_j$.
4. **Generar sintéticos**: $x_{nuevo} = x_i^{SV} + \lambda \cdot (\hat{x}_j - x_i^{SV})$, con $\lambda \sim \mathcal{U}(0, 1)$.
5. Si hay insuficientes vectores de soporte, complementar con muestras próximas al hiperplano.

### 2.2 Formulación matemática

El hiperplano SVM maximiza el margen $2/\|\mathbf{w}\|$ sujeto a $y_i(\mathbf{w}^T \phi(x_i) + b) \geq 1$. Los vectores de soporte satisfacen $y_i(\mathbf{w}^T \phi(x_i) + b) = 1$. Los sintéticos se generan:

$$x_{nuevo} = x_i^{SV} + \lambda \cdot (x_{nn}^+ - x_i^{SV}), \quad \lambda \sim \mathcal{U}(0, 1)$$

donde $x_{nn}^+$ es un vecino positivo de $x_i^{SV}$.

### 2.3 Parámetro `sampling_strategy = 0.3`

Igual que en las otras técnicas, `sampling_strategy = 0.3` genera muestras sintéticas hasta alcanzar $n_{pos}/n_{neg} = 0.3$. Solo los vectores de soporte de la clase positiva contribuyen a los sintéticos.

### 2.4 Coste computacional

SVM-SMOTE es la técnica de oversampling **más costosa computacionalmente** de las evaluadas. Por cada fold de K-Fold o cada iteración de GridSearch, debe entrenar una SVM interna (O(n²) a O(n³) según kernel) antes de generar los sintéticos. Con n ≈ 7,500 muestras por fold, esto puede multiplicar ×10-20 el tiempo de cómputo respecto a SMOTE estándar.

### 2.5 Comparativa con otras técnicas de frontera

| Característica | BorderlineSMOTE | ADASYN | SVM-SMOTE |
|---|---|---|---|
| Identificación de frontera | k-NN local (clasificación SAFE/DANGER/NOISE) | Densidad de negativos $\hat{d}_i$ | **SVM global (vectores de soporte)** |
| Coste computacional | Bajo | Bajo | **Alto** |
| Calidad de la frontera | Local (depende de $m$) | Local (depende de $K$) | **Global (óptima para SVM)** |
| Dependencia del kernel | No | No | **Sí (crítico)** |
| Riesgo de propagación de ruido | Medio | Alto | Medio |

### 2.6 Ventajas y limitaciones

**Ventajas:**
- La frontera de decisión SVM es globalmente óptima para separación lineal/RBF
- Los sintéticos generados en la vecindad de vectores de soporte son teóricamente los más informativos
- Puede capturar fronteras no lineales complejas si se usa kernel RBF

**Limitaciones:**
- Coste computacional muy elevado por el entrenamiento SVM interno en cada fold
- La calidad de los vectores de soporte depende de la convergencia de la SVM y del kernel elegido
- Si la SVM no es el clasificador final, puede haber desacuerdo entre su frontera y la del clasificador entrenado posteriormente
- **Alta sensibilidad a outliers positivos** que actúen como falsos vectores de soporte

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

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): los pacientes que fallecen antes de ser trasladados a UCI quedan etiquetados como negativos, sesgando sistemáticamente los modelos. El endpoint `outcome_intensivo = uci OR mortalidad` (742 eventos, 2.04%) corrige este sesgo. La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SVMSMOTE

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),
    ('sampler',       SVMSMOTE(sampling_strategy=0.3,
                               random_state=42,
                               k_neighbors=5,
                               m_neighbors=10)),
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
| MORTALIDAD | GaussianNB | **0.1683** | ±0.0199 | 11.82% | **Mayor variabilidad observada** entre todas las técnicas |
| outcome_intensivo | GaussianNB | **0.1923** | ±0.0130 | 6.76% | Variabilidad moderada con outcome_intensivo |

**Hallazgo importante:** SVM-SMOTE presenta la **mayor variabilidad K-Fold en mortalidad** (±0.0199). Para `outcome_intensivo`, la variabilidad se reduce notablemente (±0.0130), probablemente porque el mayor número de eventos (742 vs 318) estabiliza la estimación de vectores de soporte. Esto confirma que la redefinición del target mejora la robustez de SVM-SMOTE.

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1462** | 0.7899 | 0.5886 | 0.1218 | 0.0679 | 0.8676 |
| 2 | RandomForest | 0.1340 | 0.8874 | 0.1532 | 0.1680 | 0.1861 | 0.9890 |
| 3 | XGBoost | 0.1278 | 0.8913 | 0.3724 | 0.2058 | 0.1422 | 0.9632 |
| 4 | LightGBM | 0.1275 | 0.8933 | 0.3664 | 0.2042 | 0.1415 | 0.9636 |
| 5 | ExtraTrees | 0.1225 | 0.8792 | 0.0661 | 0.1053 | 0.2588 | 0.9969 |
| 6 | GradientBoosting | 0.1212 | 0.8877 | 0.2583 | 0.1711 | 0.1280 | 0.9711 |
| 7 | LogisticRegression | 0.1166 | 0.8785 | 0.4024 | 0.1885 | 0.1230 | 0.9530 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-5`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1783** | 0.8428 | 0.4932 | 0.1633 | 0.0979 |
| 2 | XGBoost | 0.1285 | 0.8782 | 0.2973 | 0.2005 | 0.1517 |
| 3 | GradientBoosting | 0.1263 | 0.8743 | 0.2905 | 0.2005 | 0.1524 |
| 4 | LightGBM | 0.1262 | 0.8754 | 0.2838 | 0.1892 | 0.1416 |
| 5 | ExtraTrees | 0.1240 | 0.8801 | 0.2432 | 0.1935 | 0.1562 |
| 6 | LogisticRegression | 0.1188 | 0.8698 | 0.2973 | 0.1800 | 0.1317 |
| 7 | RandomForest | 0.1186 | 0.8837 | 0.1081 | 0.1285 | 0.1628 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Nota sobre el cambio de target:** Con el antiguo `uci` (0.8%), SVM-SMOTE-GaussianNB obtenía GS PR-AUC=0.2278. Con `outcome_intensivo` (2.04%), es 0.1783. La reducción refleja la mayor complejidad del endpoint compuesto. No obstante, SVM-SMOTE sigue siendo la segunda mejor técnica para `outcome_intensivo` (tras SMOTEENN 0.2203), mejorando su posición relativa frente al antiguo target `uci`.

---

## 5. Análisis del Rendimiento — Discusión de Anomalías

### 5.1 El problema en Mortalidad: PR-AUC=0.1462, AUC-ROC=0.7899

SVM-SMOTE produce el **menor PR-AUC** (0.1462) y el **menor AUC-ROC** (0.7899) para mortalidad entre todas las técnicas. Esta anomalía merece análisis detallado:

**Hipótesis 1: Desajuste entre la frontera SVM y la distribución posterior al oversampling**  
La SVM interna determina los vectores de soporte antes de generar los sintéticos. Los sintéticos generados en esa zona pueden crear una distribución que altera la frontera de decisión para el clasificador final (GaussianNB), especialmente si la SVM usa kernel lineal (por defecto en imbalanced-learn) pero GaussianNB asume densidades gaussianas independientes.

**Hipótesis 2: Los "vectores de soporte positivos" en mortalidad son outliers clínicos**  
Los pacientes que fallecen postquirúrgicamente y que además están en la frontera de decisión SVM (vectores de soporte) pueden representar casos clínicamente atípicos. Amplificar estos casos mediante sintéticos introduce patrones anómalos que confunden al clasificador.

**Hipótesis 3: Inestabilidad de la SVM interna**  
Con n_positivos ≈ 128 y n_total ≈ 7,500, la SVM puede no converger de manera estable, produciendo vectores de soporte arbitrarios en algunos folds (explicando también la alta variabilidad K-Fold ±0.0199).

### 5.2 Para `outcome_intensivo`: resultado competitivo

En contraste, para `outcome_intensivo` (PR-AUC=0.1783), SVM-SMOTE es la **segunda mejor técnica** evaluada (tras SMOTEENN 0.2203). Esto sugiere que la distribución del endpoint compuesto en el espacio de features es más compatible con la geometría de separación que aprende la SVM interna. El mayor número de eventos (742 vs 318 del antiguo target `uci`) también estabiliza los vectores de soporte.

### 5.3 Consecuencias del coste computacional

Dado que SVM-SMOTE es la técnica más costosa y produce el peor resultado en mortalidad, su relación coste-eficiencia es la más desfavorable de las 8 técnicas evaluadas. Solo se justificaría si el dataset tuviese características que lo hagan especialmente compatible con fronteras SVM (datasets de alta dimensionalidad, separabilidad lineal, etc.).

---

## 6. Análisis del Umbral de Decisión

| Umbral | Mortalidad Recall | Mortalidad Precision | UCI Recall | UCI Precision |
|---|---|---|---|---|
| 0.5 (defecto) | ~0.00 | N/A | ~0.00 | N/A |
| F1-max | ~0.59 | ~0.07 | ~0.83 | ~0.06 |
| Recall ≥ 90% | ≥0.90 | ~0.03 | ≥0.90 | ~0.04 |

A pesar del bajo PR-AUC en mortalidad, el recall con umbral ajustado es aceptable (0.5886 con umbral F1-max), lo que indica que el modelo mantiene cierta capacidad de ranking aunque el ranking global sea subóptimo.

---

## 7. Calibración Probabilística

La calibración es especialmente crítica para SVM-SMOTE dado el desajuste observado entre probabilidades entrenadas (prevalencia artificial ~23%) y la real (~1.6%). Se aplica **Isotonic Regression** sobre la mitad del test set para recalibrar las probabilidades a escala real.

---

## 8. Conclusiones

1. **SVM-SMOTE produce resultados asimétricos**: el peor resultado para mortalidad (PR-AUC=0.1462) pero segundo mejor para `outcome_intensivo` (PR-AUC=0.1783), indicando una fuerte dependencia de la geometría específica de cada target.

2. **El endpoint compuesto `outcome_intensivo` mejora la posición relativa de SVM-SMOTE**: con el antiguo `uci` era tercero (0.2278); con `outcome_intensivo` es segundo (0.1783 vs SMOTEENN 0.2203). El mayor número de eventos (742) estabiliza los vectores de soporte y reduce la variabilidad K-Fold para este target (±0.0130 vs ±0.0390 del antiguo UCI).

3. La **mayor variabilidad K-Fold en mortalidad** (±0.0199) es la mayor de todas las técnicas, señalando inestabilidad en la calidad de los vectores de soporte según la composición de cada fold.

4. El **AUC-ROC anómalamente bajo** (0.7899 vs ~0.84 del resto) en mortalidad sugiere que la distribución sintética generada por SVM-SMOTE en ese target introduce ruido que degrada la discriminación global.

5. El **coste computacional elevado** de SVM-SMOTE no se justifica plenamente en este dataset dado los resultados inferiores en mortalidad. Para `outcome_intensivo`, el rendimiento es competitivo.

6. **Recomendación**: SVM-SMOTE es preferible frente a SMOTE para `outcome_intensivo` en este dataset, pero SMOTEENN sigue siendo superior en ambos targets.

---

## Referencias

- Nguyen, H. M., Cooper, E. W., & Kamei, K. (2011). Borderline over-sampling for imbalanced data classification. *International Journal of Knowledge Engineering and Soft Data Paradigms*, 3(1), 4-21.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
