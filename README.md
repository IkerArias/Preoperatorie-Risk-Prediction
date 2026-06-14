# Perioperative Risk Prediction with Machine Learning

**Final Degree Project (TFG) — Data Science & Artificial Intelligence**  
University of Deusto · 2025–2026

---

## Overview

This repository contains the full machine learning pipeline developed to predict three postoperative surgical outcomes from Electronic Health Records (EHR) provided by [Osakidetza](https://www.osakidetza.euskadi.eus) (Basque Health Service):

| Target | Definition | Positive rate |
|--------|-----------|---------------|
| **30-day mortality** | Death within 30 days of surgery | 1.61 % |
| **Critical care requirement** | Unexpected ICU admission ∨ in-hospital death | 2.04 % |
| **Infectious complications** | Post-surgical infections (SSI, sepsis, UTI, …) | 3.02 % |

The dataset covers **103,179 surgical episodes** from a single tertiary centre and includes 60+ preoperative and intraoperative variables (ASA-PS, Charlson Comorbidity Index, surgical speciality, anaesthetic type, etc.).

---

## Repository Structure

```
TFG/
├── notebooks/               # Jupyter notebooks (run in order)
│   ├── ntb_01_separacion_datos.ipynb          # Temporal split & data pipeline
│   ├── ntb_02_modelo_ML_*.ipynb               # Mortality & ICU: 8 resampling strategies
│   ├── ntb_03_comparativa_modelos_ML.ipynb    # Strategy comparison & final model selection
│   ├── ntb_04_construccion_features_infeccion.ipynb  # Infection feature engineering
│   ├── ntb_04_variable_objetivo_infeccion.ipynb      # Infection target construction
│   ├── ntb_05_modelo_infeccion.ipynb          # Infection prediction pipeline
│   ├── ntb_06_modelos_jerarquicos_supervivencia.ipynb # Survival analysis (proof-of-concept)
│   ├── generate_crisp_dm.py                   # CRISP-DM diagram generator
│   ├── generate_gantt.py                      # Gantt chart generator
│   └── generate_gantt_variants.py
├── src/                     # Reusable Python modules
│   ├── preproccesing/       # Data splitting and preprocessing utilities
│   └── utils/               # Visualisation and evaluation helpers
├── results/                 # Outputs (figures and aggregate tables)
│   ├── *.csv                # Per-strategy and per-fold result tables
│   └── ntb06_*.png          # Survival analysis figures
├── reports/                 # Process documentation (Markdown)
│   └── informe_*.md
└── data/                    # ⚠️  NOT included — see data/README.md
```

---

## Key Results

| Cohort | Best model | Resampling | PR-AUC (95 % CI) | AUC-ROC |
|--------|-----------|------------|-------------------|---------|
| 30-day mortality | GradientBoosting | SMOTE | 0.144 [0.114, 0.177] | 0.895 |
| Critical care req. | XGBoost | SMOTE | 0.148 [0.106, 0.196] | 0.876 |
| Infectious comp. | LightGBM | SMOTEENN | 0.247 [0.158, 0.353] | 0.788 |

SHAP analysis identifies **ASA-PS** and **age** as the dominant predictors for mortality and critical care (>60 % of cumulative importance); **ASA-PS** and **malignant tumour** lead for infectious complications.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the data

The raw data is confidential and not included in this repository (see [`data/README.md`](data/README.md)).  
Place the source files as described there before running any notebook.

### 3. Execute notebooks in order

```
ntb_01  →  ntb_02 (×8 strategies)  →  ntb_03
                                    →  ntb_04 (×2)  →  ntb_05  →  ntb_06
```

Each notebook saves intermediate results to `results/checkpoints/` so that subsequent notebooks can load them without re-training.

---

## Ethical Considerations

- All data was provided under a research agreement with Osakidetza and processed in a local, air-gapped environment.
- The dataset was anonymised at source; no direct identifiers are present.
- Processing complies with **GDPR Art. 89** (scientific research exemption) and the applicable Basque regional health data regulations.
- Models are intended as **clinical decision-support tools** and do not replace professional medical judgement.

---

## License

This project is released for academic and research purposes. The underlying patient data is property of Osakidetza and is not redistributable.

---

## Citation

If you use this work, please cite:

> Arias, I. (2026). *Perioperative Risk Prediction with Machine Learning on EHR Data from Osakidetza*. Final Degree Project, University of Deusto.
