"""
Diagrama de Gantt del TFG — estilo elegante.
- Barras redondeadas continuas (no cuadricula de celdas)
- Cuadricula muy sutil, solo separadores de mes
- Leyenda debajo del grafico (no superpuesta)
- T21 LaTeX documentation arranca en marzo (semana 5)
- Junio truncado a 2 semanas (entrega 10 junio)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import os

# ─── Fases ───────────────────────────────────────────────────────────────────
PHASES = {
    1: ("Business Understanding", "#C06878"),   # rosa empolvado
    2: ("Data Understanding",     "#5E9E78"),   # verde salvia
    3: ("Data Preparation",       "#C8A030"),   # ambar
    4: ("Modelling",              "#C87040"),   # siena
    5: ("Evaluation",             "#4088B8"),   # azul acero
    6: ("Deployment & Defence",   "#40A8A8"),   # verde-azul
}

# MONTHS: (nombre, num_semanas)
# Junio truncado a 2 semanas (W1=1-7jun, W2=8-14jun, entrega 10jun)
MONTHS = [
    ("February", 4),
    ("March",    4),
    ("April",    4),
    ("May",      4),
    ("June",     4),   # W1-W2 preentrega, W3-W4 entre entrega y defensa
]
NWEEKS = sum(nw for _, nw in MONTHS)  # 20

# ─── Tareas ──────────────────────────────────────────────────────────────────
# (etiqueta, semana_inicio, semana_fin, fase)
# Feb=1-4, Mar=5-8, Abr=9-12, May=13-16, Jun=17-18
TASKS = [
    ("T1.   Surgical risk literature review",         1,  2, 1),
    ("T2.   Three-target outcome definition",          1,  2, 1),
    ("T3.   XAI & performance requirements",           2,  3, 1),
    ("T4.   Dataset ingestion & source fusion",        2,  4, 2),
    ("T5.   EDA & imbalance profiling (3 cohorts)",    3,  5, 2),
    ("T6.   Feature inspection (pre vs periop.)",      4,  5, 2),
    ("T7.   Cleaning, imputation & scaling",           4,  6, 3),
    ("T8.   Leakage detection & corr. filtering",      5,  7, 3),
    ("T9.   Stratified split (patient isolation)",     6,  7, 3),
    ("T10.  Resampling pipeline (8 strategies)",       7,  8, 3),
    ("T11.  Composite critical-care endpoint",         8,  8, 3),
    ("T12.  Classifier grid (7 alg. x 8 strat.)",     8, 11, 4),
    ("T13.  ClassifierChain (mortality -> UCI)",      10, 12, 4),
    ("T14.  Survival models (Cox PH & RSF)",          11, 13, 4),
    ("T15.  Hyperparameter tuning & threshold",       12, 14, 4),
    ("T16.  Probability calibration (IsotonicReg.)",  13, 14, 4),
    ("T17.  Multi-metric evaluation (AUC, Recall)",   13, 15, 5),
    ("T18.  Statistical testing (Friedman-Wilcoxon)", 14, 16, 5),
    ("T19.  SHAP explainability analysis",            15, 17, 5),
    ("T20.  Survival curve generation",               16, 17, 5),
    ("T21.  LaTeX documentation & report writing",     5, 18, 6),
    ("T22.  Repository structuring & code review",    17, 18, 6),
    ("T23.  Defence preparation & submission",        19, 20, 6),
]
N = len(TASKS)  # 23

# ─── Dimensiones (pulgadas) ──────────────────────────────────────────────────
CELL_W = 0.50
CELL_H = 0.31
LBL_W  = 3.9
HDR1_H = 0.40
HDR2_H = 0.30
LEG_H  = 0.80
PAD    = 0.25

FIG_W = LBL_W + NWEEKS * CELL_W + PAD
FIG_H = HDR1_H + HDR2_H + N * CELL_H + LEG_H + PAD

ax_l = LBL_W / FIG_W
ax_b = (LEG_H + PAD * 0.5) / FIG_H
ax_w = (NWEEKS * CELL_W) / FIG_W
ax_h = (HDR1_H + HDR2_H + N * CELL_H) / FIG_H

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
ax  = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
ax.set_xlim(0, NWEEKS)
ax.set_ylim(0, N + 2)
ax.axis('off')

def tb(i): return N - 1 - i
WKHY = N
MNHY = N + 1
MCOL = "#4A5068"   # pizarra

month_x = []
x = 0
for _, nw in MONTHS:
    month_x.append(x)
    x += nw
month_bounds = month_x[1:]

# ── Cabecera de meses ─────────────────────────────────────────────────────────
for (name, nw), x0 in zip(MONTHS, month_x):
    ax.add_patch(mpatches.Rectangle((x0, MNHY), nw, 1,
                 facecolor=MCOL, edgecolor='white', linewidth=1.5, zorder=4))
    ax.text(x0 + nw / 2, MNHY + 0.5, name,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='white', zorder=5)

# ── Cabecera W1-W4 ────────────────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle((0, WKHY), NWEEKS, 1,
             facecolor='#EEEEF2', edgecolor='none', zorder=3))
for w in range(NWEEKS):
    wn = (w % 4) + 1
    ax.text(w + 0.5, WKHY + 0.5, f"W {wn}",
            ha='center', va='center', fontsize=7.2, fontweight='bold',
            color='#555555', zorder=4)
for xb in month_bounds:
    ax.add_line(Line2D([xb, xb], [WKHY, WKHY + 1],
                color='#C0C0CC', linewidth=0.9, zorder=4))
ax.add_patch(mpatches.Rectangle((0, WKHY), NWEEKS, 1,
             facecolor='none', edgecolor='#C0C0CC', linewidth=0.7, zorder=4))

# ── Filas de tareas con barras redondeadas ───────────────────────────────────
for i, (label, s, e, ph) in enumerate(TASKS):
    yb    = tb(i)
    color = PHASES[ph][1]
    alt   = "#F8F8FA" if i % 2 == 0 else "#FFFFFF"
    ax.add_patch(mpatches.Rectangle((0, yb), NWEEKS, 1,
                 facecolor=alt, edgecolor='none', zorder=1))
    x_bar = (s - 1) + 0.08
    w_bar = (e - s + 1) - 0.16
    h_bar = 0.74
    y_bar = yb + 0.13
    ax.add_patch(FancyBboxPatch(
        (x_bar, y_bar), w_bar, h_bar,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor=color, edgecolor='none', alpha=0.80, zorder=2
    ))

# ── Cuadricula minimal ────────────────────────────────────────────────────────
for i in range(1, N):
    ax.add_line(Line2D([0, NWEEKS], [i, i],
                color='#E8E8EE', linewidth=0.30, zorder=3))
for xb in month_bounds:
    ax.add_line(Line2D([xb, xb], [0, N],
                color='#C0C0CC', linewidth=0.80, zorder=3))
ax.add_patch(mpatches.Rectangle((0, 0), NWEEKS, N + 2,
             facecolor='none', edgecolor='#888888', linewidth=1.1, zorder=6))

# ── Columna de etiquetas ─────────────────────────────────────────────────────
lax = fig.add_axes([0, ax_b, ax_l, ax_h])
lax.set_xlim(0, 1)
lax.set_ylim(0, N + 2)
lax.axis('off')

lax.add_patch(mpatches.Rectangle((0, MNHY), 1, 1,
              facecolor=MCOL, edgecolor='#888888', linewidth=1.1, zorder=4))
lax.add_patch(mpatches.Rectangle((0, WKHY), 1, 1,
              facecolor='#EEEEF2', edgecolor='#C0C0CC', linewidth=0.7, zorder=4))
lax.text(0.97, WKHY + 0.5, "Task",
         ha='right', va='center', fontsize=8, fontweight='bold',
         color='#444444', zorder=5)
SBW = 0.025  # anchura de la barra lateral de fase
for i, (label, *_) in enumerate(TASKS):
    yb  = tb(i)
    ph  = TASKS[i][3]
    alt = "#F8F8FA" if i % 2 == 0 else "#FFFFFF"
    lax.add_patch(mpatches.Rectangle((0, yb), 1, 1,
                  facecolor=alt, edgecolor='none', zorder=1))
    lax.add_patch(mpatches.Rectangle((0, yb + 0.06), SBW, 0.88,
                  facecolor=PHASES[ph][1], edgecolor='none', alpha=0.90, zorder=2))
    lax.text(0.97, yb + 0.5, label,
             ha='right', va='center', fontsize=7.2, color='#222222', zorder=2)
for i in range(1, N):
    lax.add_line(Line2D([0, 1], [i, i], color='#E8E8EE', linewidth=0.30, zorder=2))
lax.add_patch(mpatches.Rectangle((0, 0), 1, N + 2,
              facecolor='none', edgecolor='#888888', linewidth=1.1, zorder=6))

# ── Leyenda debajo ────────────────────────────────────────────────────────────
leg_ax = fig.add_axes([0, 0, 1, (LEG_H * 0.88) / FIG_H])
leg_ax.axis('off')
patches = [
    mpatches.Patch(facecolor=PHASES[ph][1], edgecolor='#888888',
                   linewidth=0.6, label=f"Phase {ph} \u2014 {PHASES[ph][0]}")
    for ph in sorted(PHASES)
]
leg_ax.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5),
              fontsize=8.5, ncol=3, frameon=True, framealpha=0.96,
              edgecolor='#CCCCCC', columnspacing=1.5,
              title="CRISP-DM Phases", title_fontsize=9.5)

# ─── Guardar ─────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'reports', 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('png', 'pdf'):
    kw = dict(bbox_inches='tight', facecolor='white')
    if ext == 'png':
        kw['dpi'] = 200
    path = os.path.join(out_dir, f'gantt.{ext}')
    plt.savefig(path, **kw)
    print(f"Saved: {path}")
