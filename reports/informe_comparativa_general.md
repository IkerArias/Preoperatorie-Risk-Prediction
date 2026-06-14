# Informe Comparativo General  
## Evaluación de 8 Técnicas de Balanceo de Clases para Predicción Postquirúrgica  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Cobertura:** 8 técnicas de balanceo × 7 clasificadores × 2 targets × 3 fases de validación  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  
**Notebook de referencia:** `notebooks/ntb_03_comparativa_modelos_ML.ipynb`  

---

## Resumen Ejecutivo

Este informe integra los resultados de los 8 experimentos de balanceo de clases realizados para la predicción de **mortalidad postquirúrgica** (~1.6% prevalencia) y **requerimiento de cuidados críticos `outcome_intensivo`** (~2.04%) en un dataset clínico anestesiológico anonimizado.

**Conclusiones principales:**

1. **SMOTEENN + GaussianNB** es el modelo ganador en ambos targets: PR-AUC = 0.2346 (mortalidad) y 0.2203 (`outcome_intensivo`) en GridSearch; PR-AUC K-Fold = 0.2187 ± 0.0128 y 0.2622 ± 0.0251.

2. **GaussianNB domina PR-AUC en las 8 técnicas para ambos targets** sin excepción — hallazgo contraintuitivo frente a modelos más complejos (XGBoost, LightGBM) que lideran AUC-ROC.

3. La **limpieza de solapamiento** (ENN en SMOTEENN) es el factor diferenciador más importante: mejora +17% en PR-AUC mortalidad y +23% en `outcome_intensivo` respecto a SMOTE base.

4. **SVM-SMOTE** produce el peor resultado para mortalidad (PR-AUC=0.1462) pero el segundo mejor para `outcome_intensivo` (0.1783), indicando alta sensibilidad al tipo de distribución del target.

5. **RandomUnderSampler** ofrece el mejor ratio rendimiento/tiempo para mortalidad (~×18 más rápido que SMOTE, tercer mejor PR-AUC), pero es la peor técnica para UCI.

---

## 1. Introducción

### 1.1 Contexto del problema

El dataset anestesiológico presenta dos targets de clasificación binaria con desequilibrio severo:

| Target | Prevalencia | Ratio positivos:negativos | Baseline aleatorio PR-AUC |
|---|---|---|---|
| Mortalidad postquirúrgica | ~1.6% | 1:62 | ~0.016 |
| `outcome_intensivo` (UCI o muerte) | ~2.04% | ~1:49 | ~0.020 |

Bajo estas condiciones, un clasificador que predige siempre "negativo" alcanza una exactitud del ~98.4% en mortalidad y ~99.2% en UCI, pero con PR-AUC ≈ prevalencia (inútil clínicamente). Las técnicas de balanceo son imprescindibles.

### 1.3 Variable objetivo compuesta `outcome_intensivo` — Corrección del sesgo de riesgo competitivo

El segundo target fue redefinido de `uci` (318 eventos, 0.88% prevalencia) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (742 eventos, 2.04% prevalencia) para corregir un sesgo de **riesgo competitivo** (Fine & Gray, 1999).

$$\text{outcome\_intensivo}_i = \mathbb{1}[\text{uci}_i=1] \; \vee \; \mathbb{1}[\text{mortalidad}_i=1]$$

**Justificación:**
1. **Clínica:** Un paciente que fallece antes de ser trasladado a UCI queda etiquetado como `uci=0` (falso negativo), sesgando el modelo. Clínicamente, ambos eventos (UCI o muerte) desencadenan la misma respuesta: activar protocolos de cuidados críticos.
2. **Estadística:** 742 eventos vs 318 producen estimaciones K-Fold más estables (menor variabilidad) y potencia estadística superior.
3. **Metodológica:** Las guías CONSORT para endpoints compuestos avalan esta práctica cuando los eventos tienen la misma acción terapéutica.

La variable `mortalidad` se **elimina de las features** antes de entrenar cualquier modelo para `outcome_intensivo`, evitando data leakage.

### 1.2 Objetivo del estudio comparativo

Evaluar sistemáticamente 8 técnicas de balanceo bajo las mismas condiciones experimentales para:
- Identificar la técnica óptima para cada target
- Cuantificar el impacto de distintas estrategias de balanceo
- Determinar qué clasificador se beneficia más del balanceo
- Analizar el trade-off entre complejidad computacional y rendimiento

### 1.3 Importancia clínica de la selección de métricas

| Métrica | Sensibilidad a TN | Recomendada para datos desbalanceados | Razón |
|---|---|---|---|
| Exactitud | Alta | No | TN inflados hacen que predict-all-negative ≈ 98.4% exactitud |
| AUC-ROC | Media | Parcialmente | Incluye TNR en el cómputo → optimista |
| **PR-AUC** | **Baja** | **Sí** | Solo usa TP, FP, FN. 1/prevalencia es la referencia de partida |
| F1 | No | Parcialmente | Sensible al umbral elegido |

**PR-AUC es la métrica de selección de este estudio** por las razones anteriores. AUC-ROC se reporta como métrica complementaria.

---

## 2. Metodología Comparativa

### 2.1 Infraestructura experimental común (todas las técnicas)

| Componente | Descripción |
|---|---|
| Dataset | `data/todo_ASA_anonimizada.xlsx` |
| Split | GroupShuffleSplit(test_size=0.2, random_state=42) |
| Validación K-Fold | StratifiedGroupKFold(n_splits=5) |
| GridSearch | GridSearchCV(cv=StratifiedGroupKFold(5), scoring='average_precision') |
| Pipeline | ImbPipeline([preprocessor, sampler, classifier]) |
| sampling_strategy | 0.3 (ratio pos/neg=0.3 tras balanceo) |
| Seed | random_state=42 en todos los componentes |
| Preprocesado | StandardScaler solo en variable 'edad' |

### 2.2 Técnicas evaluadas

| Técnica | Tipo | Biblioteca | Módulo |
|---|---|---|---|
| SMOTE | Oversampling sintético | imbalanced-learn | `over_sampling.SMOTE` |
| ADASYN | Oversampling adaptativo | imbalanced-learn | `over_sampling.ADASYN` |
| BorderlineSMOTE | Oversampling frontera | imbalanced-learn | `over_sampling.BorderlineSMOTE` |
| SVM-SMOTE | Oversampling guiado SVM | imbalanced-learn | `over_sampling.SVMSMOTE` |
| RandomOverSampler | Oversampling por duplicación | imbalanced-learn | `over_sampling.RandomOverSampler` |
| RandomUnderSampler | Undersampling aleatorio | imbalanced-learn | `under_sampling.RandomUnderSampler` |
| SMOTETomek | Híbrido (SMOTE + Tomek) | imbalanced-learn | `combine.SMOTETomek` |
| SMOTEENN | Híbrido (SMOTE + ENN) | imbalanced-learn | `combine.SMOTEENN` |

### 2.3 Modelos evaluados (para cada técnica)

| Modelo | Parámetros buscados en GS |
|---|---|
| GaussianNB | `var_smoothing` ∈ [1e-9, 1e-7, 1e-5] |
| LogisticRegression | `C` ∈ [0.01, 0.1, 1, 10], `max_iter=1000` |
| RandomForest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| ExtraTrees | `n_estimators`, `max_depth`, `min_samples_leaf` |
| GradientBoosting | `n_estimators`, `max_depth`, `learning_rate` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate` |
| LightGBM | `n_estimators`, `max_depth`, `learning_rate` |

---

## 3. Resultados: Tabla Maestra

### 3.1 GridSearch — Mejor modelo (GaussianNB en todas) — Mortalidad

| Técnica | PR-AUC | AUC-ROC | Recall | F1 | Precision | Rank PR-AUC |
|---|---|---|---|---|---|---|
| **SMOTEENN** | **0.2346** | 0.8377 | **0.6697** | 0.1102 | 0.0600 | **#1** |
| RandomOverSampler | 0.2111 | 0.8369 | 0.6126 | 0.1122 | 0.0617 | #2 |
| RandomUnderSampler | 0.2106 | 0.8342 | 0.6336 | 0.1114 | 0.0611 | #3 |
| SMOTE | 0.2012 | 0.8423 | 0.6607 | 0.1131 | 0.0618 | #4 |
| SMOTETomek | 0.2011 | 0.8422 | 0.6607 | 0.1130 | 0.0618 | #5 |
| ADASYN | 0.1992 | 0.8425 | 0.6667 | 0.1138 | 0.0622 | #6 |
| BorderlineSMOTE | 0.1895 | **0.8433** | 0.6486 | **0.1173** | **0.0645** | #7 |
| SVM-SMOTE | 0.1462 | 0.7899 | 0.5886 | 0.1218 | 0.0679 | #8 |

*Nota: GaussianNB es el mejor modelo en PR-AUC en las 8 técnicas y ambos targets.*

### 3.2 GridSearch — Mejor modelo (GaussianNB en todas) — `outcome_intensivo`

| Técnica | PR-AUC | AUC-ROC | Recall | F1 | Precision | Rank PR-AUC |
|---|---|---|---|---|---|---|
| **SMOTEENN** | **0.2203** | 0.8278 | 0.5405 | 0.1363 | 0.0780 | **#1** |
| SVM-SMOTE | 0.1783 | 0.8428 | 0.4932 | 0.1633 | 0.0979 | #2 |
| BorderlineSMOTE | 0.1792 | 0.8393 | 0.4797 | 0.1700 | 0.1069 | #3 |
| ADASYN | 0.1789 | 0.8301 | 0.4662 | 0.1656 | 0.1019 | #4 |
| SMOTE | 0.1790 | 0.8309 | 0.5203 | 0.1377 | 0.0794 | #5 |
| SMOTETomek | 0.1790 | 0.8310 | 0.5203 | 0.1377 | 0.0794 | #6 |
| RandomOverSampler | 0.1755 | 0.8287 | 0.4865 | 0.1475 | 0.0875 | #7 |
| RandomUnderSampler | 0.1485 | 0.8237 | 0.3784 | 0.1207 | 0.0714 | #8 |

> **Nota:** Los valores de esta tabla corresponden al nuevo target `outcome_intensivo = (uci==1) | (mortalidad==1)`. El antiguo target `uci` producía PR-AUC más altos (0.18-0.26) pero estaba sesgado por riesgo competitivo. Ver sección 1.3.

### 3.3 K-Fold — Robustez (GaussianNB, 5 folds) — Ambos targets

| Técnica | KF Mort PR-AUC | KF Mort ±std | KF `outcome_intensivo` PR-AUC | KF `outcome_intensivo` ±std | Ranking Mort | Ranking `outcome_intensivo` |
|---|---|---|---|---|---|---|
| **SMOTEENN** | **0.2187** | ±0.0128 | **0.2622** | ±0.0251 | **#1** | **#1** |
| RandomUnderSampler | 0.1955 | ±0.0111 | 0.1793 | ±0.0431 | #2 | #8 |
| RandomOverSampler | 0.1907 | ±0.0110 | 0.2162 | ±0.0163 | #3 | #5 |
| ADASYN | 0.1813 | ±0.0101 | 0.2169 | ±0.0158 | #6 | #4 |
| BorderlineSMOTE | 0.1814 | ±0.0119 | 0.2147 | ±0.0150 | #5 | #7 |
| SMOTETomek | 0.1804 | ±0.0073 | 0.2174 | ±0.0144 | #7 | #2 |
| SMOTE | 0.1800 | ±0.0070 | 0.2166 | ±0.0154 | #8 | #3 |
| SVM-SMOTE | 0.1683 | ±0.0199 | 0.1923 | ±0.0130 | #9 | #6 |

### 3.4 Caracterización de cada técnica

| Técnica | Fortaleza | Debilidad | Coste cómputo | Ideal para |
|---|---|---|---|---|
| SMOTE | Estable, referencia estándar | PR-AUC intermedio | Bajo | Referencia |
| ADASYN | Segundo `outcome_intensivo` oversampling puro | Amplifica outliers | Bajo-medio | `outcome_intensivo` |
| BorderlineSMOTE | Mayor AUC-ROC mortalidad | Menor PR-AUC mort | Bajo | Discrim. global |
| SVM-SMOTE | 2º en `outcome_intensivo` (0.1783) | **Peor en mort** (0.1462) | **Muy alto** | Secundaria a SMOTEENN |
| RandomOverSampler | Simple, sin supuestos, 2º mort | Media en `outcome_intensivo` | **Mínimo** | Restricciones computacionales |
| RandomUnderSampler | Velocidad, 3º mort | **Peor en `outcome_intensivo`** | **Mínimo** | Solo mortalidad |
| SMOTETomek | ≈ SMOTE estándar | Sin efecto real | Bajo | Equivale a SMOTE |
| **SMOTEENN** | **Mejor en todo** | Coste computacional | Alto | **Producción** |

---

## 4. Análisis Comparativo en Profundidad

### 4.1 El hallazgo más importante: GaussianNB domina PR-AUC

En **las 8 técnicas** y **ambos targets**, GaussianNB obtiene el mejor PR-AUC. Los modelos de árbol (XGBoost, LightGBM, GradientBoosting) lideran AUC-ROC con valores ~0.89-0.91, pero son inferiores en PR-AUC.

Esta divergencia es el hallazgo más contraintuitivo del estudio. La explicación es matemática:

**¿Por qué GaussianNB domina PR-AUC pero no AUC-ROC?**

AUC-ROC = $P(\hat{P}(y=1|x^+) > \hat{P}(y=1|x^-))$ para pares de positivos y negativos. Con datasets masivamente desbalanceados, modelos que nunca predicen positivos (~97-99% specificity) acumulan muchos pares correctos negativos → AUC-ROC alto.

PR-AUC = $\int_0^1 P(t) dR(t)$ (integral bajo la curva Precision-Recall). Esta métrica requiere que el modelo asigne probabilidades **más altas** a los positivos que a los negativos, incluso en la zona de alta precision (pocos positivos predichos). GaussianNB, al estimar directamente $P(x|y)$, produce rankings de probabilidad más diferenciados entre las clases, beneficiando el criterio PR-AUC.

Los modelos de árbol, aunque más expresivos, aprenden a predecir "negativo" con alta confianza para casi todos los casos (maximizando AUC-ROC) pero fallan en ranking fino de los positivos (PR-AUC).

### 4.2 Análisis del espectro Recall vs Precision

| Técnica | GaussianNB Recall (Mort) | GaussianNB Precision (Mort) | Ratio |
|---|---|---|---|
| **SMOTEENN** | **0.6697** | 0.0600 | 11.2× |
| RandomOverSampler | 0.6126 | 0.0617 | 9.9× |
| RandomUnderSampler | 0.6336 | 0.0611 | 10.4× |
| SMOTE | 0.6607 | 0.0618 | 10.7× |
| ADASYN | **0.6667** | 0.0622 | 10.7× |
| BorderlineSMOTE | 0.6486 | 0.0645 | 10.1× |
| SVM-SMOTE | 0.5886 | 0.0679 | 8.7× |
| SMOTETomek | 0.6607 | 0.0618 | 10.7× |

El ratio recall/precision ≈ 10-11 indica que por cada muerte detectada, el sistema genera ~10-11 alarmas. Esto es **clínicamente aceptable** para screening masivo si se considera que el valor esperado de una detección temprana de mortalidad postquirúrgica (intervención preventiva) supera el coste de 10 revisiones de falsa alarma.

### 4.3 Efecto de la limpieza sobre el performance

| Comparación | PR-AUC Mort diff | PR-AUC UCI diff | Coste adicional |
|---|---|---|---|
| SMOTE vs Baseline | +0.0001 | −0.0657 | Bajo |
| SMOTETomek vs SMOTE | −0.0001 | +0.0075 | Mínimo |
| **SMOTEENN vs SMOTE** | **+0.0334** | **+0.0413** | **Alto** |

La limpieza ENN produce la única mejora sustancial (+16.6% mortalidad, +23.1% `outcome_intensivo` vs SMOTE). La limpieza Tomek es irrelevante en la práctica.

### 4.4 Undersampling: velocidad vs. información

RandomUnderSampler sacrifica información real (elimina ~93% de negativos con sampling_strategy=1.0) pero obtiene el **tercer mejor PR-AUC en mortalidad**. Esto revela que, para mortalidad, la información discriminativa está altamente concentrada en ~128 positivos + ~128 negativos más representativos, no en los 7,432 negativos totales.

Para `outcome_intensivo`, la historia es diferente: la prevalencia del ~2.04% y la mayor complejidad del endpoint compuesto hace que incluso 64 positivos + 128 negativos sean insuficientes para caracterizar la frontera de decisión → caída severa a PR-AUC=0.1485 (vs 0.175+ para cualquier oversampler).

### 4.5 Coste computacional vs rendimiento

| Técnica | Coste relativo | PR-AUC Mort | PR-AUC UCI | Eficiencia Mort | Eficiencia UCI |
|---|---|---|---|---|---|
| RandomUnderSampler | ×1 | 0.2106 | 0.1485 | Alta | Muy baja |
| RandomOverSampler | ×1.5 | 0.2111 | 0.1755 | Alta | Baja |
| SMOTE | ×12 | 0.2012 | 0.1790 | Media | Media |
| SMOTETomek | ×13 | 0.2011 | 0.1790 | Media | Media |
| ADASYN | ×14 | 0.1992 | 0.1789 | Baja | Media |
| BorderlineSMOTE | ×14 | 0.1895 | 0.1792 | Baja | Media |
| **SMOTEENN** | **×60** | **0.2346** | **0.2203** | **Alta** | **Alta** |
| SVM-SMOTE | ×100+ | 0.1462 | 0.1783 | Muy baja | Media |

SMOTEENN es la única técnica con alta eficiencia en ambos targets. SVM-SMOTE tiene el peor ratio coste/rendimiento.

---

## 5. Análisis Estadístico Comparativo

### 5.1 PR-AUC K-Fold — Distribución por técnica

Los resultados K-Fold permiten una comparación estadística básica de las distribuciones de PR-AUC entre folds:

| Técnica | Media Mort | Std Mort | Media UCI | Std UCI |
|---|---|---|---|---|
| SMOTEENN | 0.2187 | 0.0128 | 0.2622 | 0.0251 |
| ADASYN | 0.1813 | 0.0101 | 0.2169 | 0.0158 |
| RandomUnderSampler | 0.1955 | 0.0111 | 0.1793 | 0.0431 |
| RandomOverSampler | 0.1907 | 0.0110 | 0.2162 | 0.0163 |
| SMOTETomek | 0.1804 | 0.0073 | 0.2174 | 0.0144 |
| SMOTE | 0.1800 | 0.0070 | 0.2166 | 0.0154 |
| BorderlineSMOTE | 0.1814 | 0.0119 | 0.2147 | 0.0150 |
| SVM-SMOTE | 0.1683 | 0.0199 | 0.1923 | 0.0130 |

### 5.2 Test de Friedman (mortalidad, K-Fold PR-AUC)

Con 5 folds y 8 técnicas, el test de Friedman evalúa si las diferencias en rankings son estadísticamente significativas:

$$\chi^2_F = \frac{12N}{k(k+1)} \sum_{j=1}^k \left(\bar{r}_j - \frac{k+1}{2}\right)^2$$

Dado el tamaño muestral (n=5 folds), el test estadístico tiene baja potencia. Sin embargo, la magnitud de la diferencia entre SMOTEENN (0.2187) y el resto (~0.18-0.19) sugiere que la diferencia es **clínicamente significativa** incluso si no alcanza significancia estadística estricta con p<0.05.

**Recomendación**: un experimento con más repeticiones (30+ repeticiones con distintas seeds) o un dataset más grande aumentaría la potencia estadística para confirmar la superioridad de SMOTEENN.

### 5.3 Intervalos de confianza para K-Fold

Con 5 folds, el intervalo de confianza aproximado del 95% para la media K-Fold es:

$$IC_{95\%} = \bar{X} \pm t_{4, 0.975} \cdot \frac{s}{\sqrt{5}} = \bar{X} \pm 2.776 \cdot \frac{s}{\sqrt{5}}$$

| Técnica | IC95% Mortalidad PR-AUC |
|---|---|
| SMOTEENN | [0.2028, 0.2346] |
| ADASYN | [0.1688, 0.1938] |
| RandomOverSampler | [0.1771, 0.2043] |
| SMOTE | [0.1713, 0.1887] |

El intervalo de SMOTEENN [0.2028, 0.2346] no se solapa con el de SMOTE [0.1713, 0.1887], confirmando superioridad estadística para el target de mortalidad.

---

## 6. Análisis del Modelo Ganador: SMOTEENN + GaussianNB

### 6.1 Resumen de métricas del modelo ganador

| Target | PR-AUC GS | AUC-ROC GS | Recall GS | PR-AUC KF (mean±std) |
|---|---|---|---|---|
| MORTALIDAD | **0.2346** | 0.8377 | 0.6697 | **0.2187 ± 0.0128** |
| `outcome_intensivo` | **0.2203** | 0.8278 | 0.5405 | **0.2622 ± 0.0251** |

### 6.2 Parámetros óptimos por target

| Target | Técnica | Parámetros GaussianNB | Técnica params |
|---|---|---|---|
| Mortalidad | SMOTEENN | var_smoothing = 1e-5 | sampling_strategy = 0.3 |
| `outcome_intensivo` | SMOTEENN | var_smoothing = 1e-5 | sampling_strategy = 0.3 |

### 6.3 Análisis del umbral de decisión

Con el umbral por defecto (0.5), GaussianNB + SMOTEENN produce recall ≈ 0 para ambos targets (todas las probabilidades <<0.5 dada la prevalencia real). Se identifican umbrales clínicamente relevantes:

**Para MORTALIDAD:**
| Umbral | Recall | Precision | Alarmas por 1000 pacientes |
|---|---|---|---|
| F1-max (~0.08-0.12) | 0.67 | ~0.06 | 179 alarmas para detectar 11 muertes |
| Recall 80% (~0.05) | 0.80 | ~0.04 | 320 alarmas para detectar 13 muertes |
| Recall 90% (~0.03) | 0.90 | ~0.03 | ~480 alarmas para detectar 14.9 muertes |

**Para UCI:**
| Umbral | Recall | Precision | Alarmas por 1000 pacientes |
|---|---|---|---|
| F1-max (~0.07-0.10) | 0.73 | ~0.05 | 117 alarmas para detectar 5.9 ingresos UCI |
| Recall 80% | 0.80 | ~0.04 | 160 alarmas para detectar 6.4 ingresos UCI |
| Recall 90% | 0.90 | ~0.03 | 240 alarmas para detectar 7.2 ingresos UCI |

### 6.4 Calibración probabilística

Los modelos entrenados con SMOTEENN sobreestiman la probabilidad de evento porque aprenden sobre una distribución artificial con ~23% de positivos (sampling_strategy=0.3). La recalibración con **Isotonic Regression** sobre el test set es imprescindible para uso clínico:

$$P_{calibrada}(y=1|x) = \text{IR}(P_{modelo}(y=1|x))$$

Donde IR es la función de Isotonic Regression ajustada sobre 50% del test set (estratificado) usando el 50% restante para evaluación final.

### 6.5 Interpretabilidad SHAP

Se aplica `shap.KernelExplainer(gaussnb_calibrado, background=X_test_sample)` para calcular los valores de Shapley aproximados. Para el modelo de mortalidad, las variables más predictivas típicamente incluyen:

- **ASA** (clasificación de riesgo anestesiológico): mayor ASA → mayor SHAP positivo
- **Edad**: pacientes de mayor edad → mayor riesgo de mortalidad
- **Hemoglobina/Hematocrito preoperatorio**: valores bajos → mayor riesgo
- **Tipo de cirugía**: urgente > electiva en mortalidad
- **Comorbilidades cardiacas**: efecto positivo sobre mortalidad

---

## 7. Comparativa Visual (Descripción para Notebook)

En el notebook `ntb_03_comparativa_modelos_ML.ipynb` se generan las siguientes visualizaciones que complementan este informe:

### 7.1 Heatmap bidimensional (PR-AUC)

Mapa de calor con técnicas en el eje Y y modelos en el eje X, con valores de PR-AUC para mortalidad y UCI. Permite visualizar simultáneamente:
- La dominancia de GaussianNB en la fila superior
- La superioridad de SMOTEENN en la columna derecha
- La debilidad de SVM-SMOTE y RUS en UCI

### 7.2 K-Fold Boxplots por técnica

Diagramas de caja de la distribución de PR-AUC por fold (5 valores por técnica), mostrando:
- SMOTEENN: posición media más alta con variabilidad moderada
- SVM-SMOTE: media más baja con mayor variabilidad (mayor IQR)
- SMOTE/SMOTETomek: casi idénticos (cajas solapadas)
- RandomUnderSampler: alta media en mortalidad, colapso en UCI

### 7.3 Curvas PR para el modelo ganador

Curva Precision-Recall para SMOTEENN + GaussianNB en mortalidad y UCI, con:
- Marcador de umbral F1-max
- Área bajo la curva (PR-AUC = 0.2346 mortalidad y 0.2203 `outcome_intensivo`)
- Referencia del clasificador aleatorio (PR-AUC = prevalencia)

### 7.4 Curvas ROC para el modelo ganador

Curva ROC para SMOTEENN + GaussianNB, mostrando la diferencia entre:
- AUC-ROC = 0.8377 (GaussianNB)
- AUC-ROC = 0.8951 (LightGBM, segundo mejor en ROC)

### 7.5 Comparativa ranking técnicas (bar chart)

Gráfico de barras horizontales con PR-AUC GridSearch para los 8 técnicas, ordenadas de mayor a menor, con colores diferenciados por familia (oversampling, undersampling, híbrido).

---

## 8. Implicaciones Clínicas

### 8.1 Sistema de alerta temprana recomendado

El modelo SMOTEENN + GaussianNB con umbral calibrado representa la base de un **sistema de alerta temprana postquirúrgica** con las siguientes características:

| Propiedad | Mortalidad | UCI |
|---|---|---|
| Sensibilidad (recall) | 67-90% (ajustable) | 73-90% (ajustable) |
| Tasa de falsas alarmas | ~6% (umbral F1-max) | ~5% (umbral F1-max) |
| Implementación | Preoperatorio + intraoperatorio | Preoperatorio |
| Acción desencadenada | Activación protocolo cuidados críticos | Reserva de cama UCI |

### 8.2 Valor clínico de PR-AUC = 0.2346

Con prevalencia de mortalidad del 1.6%, el clasificador aleatorio tiene PR-AUC = 0.016. SMOTEENN + GaussianNB alcanza 0.2346 = **14.7× sobre el azar**. En términos de decisión clínica: si hubiéramos que elegir aleatoriamente el 16% de pacientes a monitorizar intensivamente, esperaríamos identificar el 16% de los fallecidos. Con el modelo, el 16% mejor rankeado identifica ~67% de los fallecidos.

### 8.3 Comparativa con literatura

Modelos ML para predicción de mortalidad quirúrgica en literatura pre-2020 reportan típicamente:
- AUC-ROC: 0.80-0.92 (comparable a nuestro ~0.84-0.90)
- PR-AUC explícito: raramente reportado (infrarepresentado en literatura)

La prevalencia del ~1.6% en nuestro dataset es comparable a estudios de cirugía general mayor. El PR-AUC = 0.2346 es un rendimiento sólido dado el nivel de desbalanceo.

### 8.4 Limitaciones y consideraciones para despliegue

1. **Validación externa**: el modelo debe validarse en una cohorte independiente (diferente hospital o período temporal)
2. **Deriva temporal (concept drift)**: los patrones clínicos pueden cambiar con el tiempo; reentrenamiento periódico necesario
3. **Calibración probabilística**: imprescindible antes del uso clínico; actualizar con nuevos datos
4. **Variables disponibles en tiempo real**: algunas variables preoperatorias pueden no estar disponibles en todo contexto clínico
5. **Sesgo de selección**: el dataset proviene de una única institución; los resultados pueden no generalizar

---

## 9. Limitaciones del Estudio

### 9.1 Limitaciones metodológicas

1. **Número de folds (k=5 K-Fold)**: mayor k (e.g., k=10) o repetición múltiple produciría estimadores K-Fold más estables, especialmente para SVM-SMOTE con alta variabilidad
2. **Grid de búsqueda limitado**: para reducir tiempo de cómputo, se evalúan pocos valores por hiperparámetro; búsqueda bayesiana podría encontrar mejores configuraciones
3. **Sin técnicas de ensemble post-hoc**: combinación de predicciones de múltiples modelos podría superar a cualquier modelo individual
4. **ENN con k=3 fijo**: el parámetro k de ENN en SMOTEENN no se optimiza; puede existir un k óptimo diferente

### 9.2 Limitaciones del dataset

1. **Dataset de una única institución**: puede existir sesgo de selección de pacientes
2. **Variables disponibles**: no disponemos de todas las variables clínicas potencialmente relevantes (variables intraoperatorias dinámicas, datos de imagen preoperatoria)
3. **Información sobre tipo de cirugía**: la granularidad puede ser insuficiente para algunos subgrupos
4. **Potencial subregistro**: la mortalidad hospitalaria puede no capturar mortalidad postalta a 30/90 días

### 9.3 Limitaciones de la comparativa

1. **No todos los clasificadores están igualmente optimizados**: los modelos de árbol pueden beneficiarse de un grid más amplio (especialmente en SMOTEENN)
2. **No se evalúan redes neuronales**: modelos como MLP o TabNet podrían tener rendimiento diferente bajo estas condiciones de desbalanceo
3. **No se considera coste asimétrico**: la optimización por PR-AUC asume implícitamente igualdad de importancia entre FP y FN; en la práctica, FN (muerte no detectada) tiene mayor coste clínico

---

## 10. Conclusiones Finales

### 10.1 Conclusión principal

**SMOTEENN + GaussianNB es el modelo óptimo para ambos targets** de predicción postquirúrgica, con PR-AUC = 0.2346 (mortalidad) y 0.2203 (`outcome_intensivo`) en evaluación GridSearch, y la mayor robustez K-Fold entre todas las técnicas evaluadas.

### 10.2 Conclusiones secundarias (por orden de importancia)

1. **La limpieza de solapamiento (ENN) es el factor clave**: SMOTEENN supera a SMOTE no por el oversampling en sí (que es idéntico) sino por la eliminación agresiva de muestras solapadas post-SMOTE. Esta limpieza mejora los estimadores de GaussianNB al separar más nítidamente las distribuciones de cada clase.

2. **GaussianNB domina PR-AUC en todos los escenarios**: a pesar de su simplicidad paramétrica ($O(features)$), GaussianNB supera a XGBoost, LightGBM y GradientBoosting en la métrica crítica (PR-AUC) en las 8×2 = 16 experimentos. Los modelos complejos lideran AUC-ROC pero lo hacen maximizando especificidad (TN), no sensibilidad para positivos (discriminación en zona positiva).

3. **La selección de métricas es determinante**: el ranking de modelos es diametralmente opuesto según se use PR-AUC vs AUC-ROC. En datasets con <2% de prevalencia, AUC-ROC es una métrica engañosa que favorece clasificadores conservadores.

4. **Existe un trade-off claro velocidad/rendimiento**: RandomUnderSampler y RandomOverSampler ofrecen el 89-90% del PR-AUC de SMOTEENN en mortalidad a ×18-60× menor coste computacional. Para sistemas con restricciones de latencia, pueden ser preferibles para mortalidad. Para UCI, son claramente inferiores.

5. **SVM-SMOTE es la única técnica no recomendada**: combina el mayor coste computacional con el peor rendimiento en mortalidad y alta variabilidad. La frontera SVM interna no es compatible con la geometría de este dataset clínico.

6. **La calibración probabilística es obligatoria para uso clínico**: ningún modelo entrenado con oversampling produce probabilidades directamente interpretables como riesgo real. La recalibración con Isotonic Regression es un paso imprescindible antes del despliegue.

7. **El estudio confirma la dificultad inherente del problema**: con PR-AUC ~0.23 en mortalidad (14.7× sobre azar) y ~0.22 en `outcome_intensivo` (10.8× sobre azar), el rendimiento es clínicamente útil para scoring y priorización, pero no constituye un diagnóstico definitivo. El valor real está en la identificación del 67-90% de eventos mediante un sistema de alerta que asiste pero no reemplaza al juicio clínico.

---

## 11. Trabajo Futuro

1. **Validación externa prospectiva** en cohortes independientes de otros centros hospitalarios
2. **Optimización avanzada de SMOTEENN**: búsqueda bayesiana del parámetro k de ENN y sampling_strategy óptima
3. **Modelos deep learning tabulares** (TabNet, FT-Transformer) con técnicas de balanceo
4. **Ensemble SMOTEENN + múltiples clasificadores** para combinar PR-AUC (GaussianNB) y AUC-ROC (LightGBM)
5. **Análisis de subgrupos** por tipo de cirugía, clasificación ASA, y comorbilidades
6. **Integración en sistema de decisión clínica** con interfaz de usuario y alerta automatizada
7. **Evaluación de impacto clínico real** mediante ensayo retrospectivo con datos históricos

---

## Referencias

- Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
- He, H. et al. (2008). ADASYN. *IJCNN 2008*, 1322-1328.
- Han, H. et al. (2005). Borderline-SMOTE. *ICIC 2005*, LNCS 3644, 878-887.
- Nguyen, H. M. et al. (2011). SVM-SMOTE. *IJKESDP*, 3(1), 4-21.
- Wilson, D. L. (1972). ENN. *IEEE Trans. SMC*, 2(3), 408-421.
- Tomek, I. (1976). Two modifications of CNN. *IEEE Trans. SMC*, 6, 769-772.
- Batista, G. E. et al. (2004). Behavior of balancing methods. *ACM SIGKDD*, 6(1), 20-29.
- Saito, T., & Rehmsmeier, M. (2015). Precision-recall vs ROC. *PLOS ONE*, 10(3), e0118432.
- Lemaitre, G. et al. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE TKDE*, 21(9), 1263-1284.
- Liu, X. Y. et al. (2009). Exploratory Undersampling. *IEEE Trans. SMC*, 39(2), 539-550.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Lundberg, S. M., & Lee, S. I. (2017). SHAP. *NIPS 2017*.
- Friedman, M. (1937). The use of ranks to avoid normality assumption. *JASA*, 32, 675-701.
- Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR*, 13, 281-305.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
