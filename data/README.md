# Data — Not Included

The data used in this project was provided by **Osakidetza** (Basque Health Service) under a research agreement and is **not publicly available**.

## What is not included

| Path | Description |
|------|-------------|
| `data/data_raw/` | 46 source CSV files from the clinical information systems (103,179 surgical episodes) |
| `data/data_divided/` | Temporally split train/test sets |
| `data/data_normalized/` | Preprocessed feature matrices |
| `data/todo_ASA_anonimizada.xlsx` | Fully merged and anonymised dataset (103,179 × 60) |

## How to obtain the data

This dataset is the property of Osakidetza and cannot be redistributed. Researchers interested in replicating this work should contact Osakidetza's Research and Innovation department to apply for access under an equivalent data-sharing agreement.

## How to use your own data

If you have access to a compatible EHR dataset, place the merged file at:

```
data/todo_ASA_anonimizada.xlsx
```

The file must contain at least the following columns (see `ntb_01_separacion_datos.ipynb` for the full schema):

- `id_paciente` — anonymous patient identifier
- `fecha_intervencion` — surgery date (used for temporal splitting)
- `asa` — ASA-PS classification (1–5)
- `mortalidad` — 30-day mortality binary label (0/1)
- `uci` — unexpected ICU admission binary label (0/1)
- Charlson comorbidity flags and remaining preoperative variables
