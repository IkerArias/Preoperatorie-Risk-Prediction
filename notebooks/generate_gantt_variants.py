"""
generate_gantt_variants.py
Genera 3 variantes visuales del diagrama de Gantt.
  gantt_v1.png  ->  "Jewel"  : tonos joya, cabecera azul marino oscuro
  gantt_v2.png  ->  "Warm"   : tonos calidos, barra de fase en la izquierda
  gantt_v3.png  ->  "Dark"   : fondo oscuro, barras vibrantes, modo dashboard

Junio: 4 semanas (W1-W2 = preentrega, W3-W4 = periodo entre entrega y defensa)
Linea discontinua roja en semana 18 = fecha limite de entrega (10 junio).
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import os

# ── Datos comunes ────────────────────────────────────────────────────────────
MONTHS = [("February",4),("March",4),("April",4),("May",4),("June",4)]
NWEEKS = 20
N_SUB  = 18   # entrega ~ fin W2 junio (10 junio)

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
    ("T10.  Resampling pipeline (9 strategies)",       7,  8, 3),
    ("T11.  Composite critical-care endpoint",         8,  8, 3),
    ("T12.  Classifier grid (7 alg. x 9 strat.)",     8, 11, 4),
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
N = len(TASKS)

MX = []; _x = 0
for _, nw in MONTHS:
    MX.append(_x); _x += nw
MB = MX[1:]

def tb(i): return N - 1 - i

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'reports', 'figures')
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    for ext in ('png', 'pdf'):
        kw = dict(bbox_inches='tight', facecolor=fig.get_facecolor())
        if ext == 'png': kw['dpi'] = 200
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), **kw)
        print(f"  -> {name}.{ext}")
    plt.close(fig)


def draw(phases, s, fname):
    """
    phases : {id: (name, hex_color)}
    s      : style dict
    fname  : output filename base (no extension)
    """
    CW = s.get('cw', 0.48); CH = s.get('ch', 0.31); LW = s.get('lw', 3.9)
    H1 = s.get('h1', 0.40); H2 = s.get('h2', 0.30)
    LH = s.get('lh', 0.80); P  = 0.25
    FW = LW + NWEEKS * CW + P
    FH = H1 + H2 + N * CH + LH + P
    al = LW / FW; ab = (LH + P * 0.5) / FH
    aw = NWEEKS * CW / FW; ah = (H1 + H2 + N * CH) / FH

    fig = plt.figure(figsize=(FW, FH), facecolor=s.get('bg', 'white'))
    ax  = fig.add_axes([al, ab, aw, ah])
    ax.set_xlim(0, NWEEKS); ax.set_ylim(0, N + 2); ax.axis('off')
    WY = N; MY = N + 1

    MHF = s.get('mhf', '#1A2744'); MHT = s.get('mht', 'white')
    MHE = s.get('mhe', 'white')
    WHF = s.get('whf', '#E8E8E8'); WHT = s.get('wht', '#555555')
    EVN = s.get('ev',  '#F5F5F5'); ODD = s.get('od',  '#FDFDFD')
    GRC = s.get('gc',  '#E0E0E0'); GRL = s.get('gl',  0.35)
    MBC = s.get('mc',  '#CCCCCC'); MBL = s.get('ml',  0.75)
    BRC = s.get('bc',  '#7A7A7A'); TXC = s.get('tx',  '#1A1A1A')
    RND = s.get('rn',  0.18);      BAL = s.get('ba',  0.87)
    SBW = s.get('sb',  0.0)       # sidebar width (fraction of label col)
    DC  = s.get('dc',  '#CC3333') # deadline line color

    # ── Month header ──────────────────────────────────────────────────────────
    for (nm, nw), x0 in zip(MONTHS, MX):
        ax.add_patch(mpatches.Rectangle((x0, MY), nw, 1,
                     facecolor=MHF, edgecolor=MHE, lw=1.5, zorder=4))
        ax.text(x0 + nw / 2, MY + 0.5, nm,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color=MHT, zorder=5)

    # ── Week sub-header ───────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle((0, WY), NWEEKS, 1,
                 facecolor=WHF, edgecolor='none', zorder=3))
    for w in range(NWEEKS):
        ax.text(w + 0.5, WY + 0.5, f"W {(w % 4) + 1}",
                ha='center', va='center', fontsize=7.2, fontweight='bold',
                color=WHT, zorder=4)
    for xb in MB:
        ax.add_line(Line2D([xb, xb], [WY, WY + 1], color=MBC, lw=0.9, zorder=4))
    ax.add_patch(mpatches.Rectangle((0, WY), NWEEKS, 1,
                 facecolor='none', edgecolor=MBC, lw=0.7, zorder=4))

    # ── Task rows + bars ──────────────────────────────────────────────────────
    for i, (lbl, bs, be, ph) in enumerate(TASKS):
        yb = tb(i); c = phases[ph][1]; alt = EVN if i % 2 == 0 else ODD
        ax.add_patch(mpatches.Rectangle((0, yb), NWEEKS, 1,
                     facecolor=alt, edgecolor='none', zorder=1))
        xb = (bs - 1) + 0.08; wb = (be - bs + 1) - 0.16
        # optional glow (dark mode)
        if s.get('glow', False):
            ax.add_patch(FancyBboxPatch(
                (xb - 0.05, yb + 0.09), wb + 0.10, 0.82,
                boxstyle=f"round,pad=0,rounding_size={RND + 0.05}",
                facecolor=c, edgecolor='none', alpha=BAL * 0.30, zorder=1))
        ax.add_patch(FancyBboxPatch(
            (xb, yb + 0.13), wb, 0.74,
            boxstyle=f"round,pad=0,rounding_size={RND}",
            facecolor=c, edgecolor='none', alpha=BAL, zorder=2))

    # ── Grid ─────────────────────────────────────────────────────────────────
    for i in range(1, N):
        ax.add_line(Line2D([0, NWEEKS], [i, i], color=GRC, lw=GRL, zorder=3))
    for xb in MB:
        ax.add_line(Line2D([xb, xb], [0, N], color=MBC, lw=MBL, zorder=3))

    # ── Submission deadline dashed line ───────────────────────────────────────
    ax.add_line(Line2D([N_SUB, N_SUB], [0, N + 2],
                color=DC, lw=1.4, ls='--', zorder=5, alpha=0.80))
    ax.text(N_SUB + 0.10, N + 0.18, "Submission\ndeadline",
            va='bottom', ha='left', fontsize=6.5, color=DC,
            style='italic', zorder=6)

    # ── Outer border ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle((0, 0), NWEEKS, N + 2,
                 facecolor='none', edgecolor=BRC, lw=1.1, zorder=6))

    # ── Label column ─────────────────────────────────────────────────────────
    lax = fig.add_axes([0, ab, al, ah])
    lax.set_xlim(0, 1); lax.set_ylim(0, N + 2); lax.axis('off')

    lax.add_patch(mpatches.Rectangle((0, MY), 1, 1,
                  facecolor=MHF, edgecolor=BRC, lw=1.1, zorder=4))
    lax.add_patch(mpatches.Rectangle((0, WY), 1, 1,
                  facecolor=WHF, edgecolor=MBC, lw=0.7, zorder=4))
    lax.text(0.97, WY + 0.5, "Task",
             ha='right', va='center', fontsize=8, fontweight='bold',
             color=WHT if s.get('bg', 'white') != 'white' else '#444444', zorder=5)

    for i, (lbl, *_, ph) in enumerate(TASKS):
        yb = tb(i); alt = EVN if i % 2 == 0 else ODD
        lax.add_patch(mpatches.Rectangle((0, yb), 1, 1,
                      facecolor=alt, edgecolor='none', zorder=1))
        if SBW > 0:
            lax.add_patch(mpatches.Rectangle((0, yb + 0.06), SBW, 0.88,
                          facecolor=phases[ph][1], edgecolor='none',
                          alpha=0.90, zorder=2))
        lax.text(0.97, yb + 0.5, lbl,
                 ha='right', va='center', fontsize=7.2, color=TXC, zorder=2)

    for i in range(1, N):
        lax.add_line(Line2D([0, 1], [i, i], color=GRC, lw=GRL, zorder=2))
    lax.add_patch(mpatches.Rectangle((0, 0), 1, N + 2,
                  facecolor='none', edgecolor=BRC, lw=1.1, zorder=6))

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = fig.add_axes([0, 0, 1, LH * 0.88 / FH])
    leg.axis('off')
    leg.set_facecolor(s.get('bg', 'white'))
    ps = [mpatches.Patch(fc=phases[ph][1], alpha=BAL, ec='#888888', lw=0.6,
                         label=f"Phase {ph} \u2014 {phases[ph][0]}")
          for ph in sorted(phases)]
    lo = leg.legend(handles=ps, loc='center', bbox_to_anchor=(0.5, 0.5),
                    fontsize=8.5, ncol=3, frameon=True, framealpha=0.96,
                    edgecolor=s.get('lec', '#CCCCCC'), columnspacing=1.5,
                    title="CRISP-DM Phases", title_fontsize=9.5)
    lo.get_frame().set_facecolor(s.get('lbg', 'white'))
    lo.get_frame().set_edgecolor(s.get('lec', '#CCCCCC'))
    for t in lo.get_texts():    t.set_color(s.get('ltx', '#222222'))
    lo.get_title().set_color(s.get('ltx', '#222222'))

    save(fig, fname)


# ════════════════════════════════════════════════════════════════════════════
# V1  "Jewel" — tonos joya sobre azul marino oscuro
# ════════════════════════════════════════════════════════════════════════════
print("Generating V1: Jewel...")
draw(
    phases={
        1: ("Business Understanding", "#7B3F7B"),   # purpura oscuro
        2: ("Data Understanding",     "#2D6E5A"),   # esmeralda
        3: ("Data Preparation",       "#C48A10"),   # oro viejo
        4: ("Modelling",              "#C44020"),   # rubi
        5: ("Evaluation",             "#2050A8"),   # zafiro
        6: ("Deployment & Defence",   "#108080"),   # teal oscuro
    },
    s={
        'mhf':'#1A2744', 'mht':'white',  'mhe':'white',
        'whf':'#E8E8EC', 'wht':'#555555',
        'ev':'#F6F6F8',  'od':'#FEFEFF',
        'gc':'#E0E0E0',  'gl':0.30,
        'mc':'#CACACA',  'ml':0.70,
        'bc':'#606060',  'tx':'#111111',
        'rn':0.18, 'ba':0.82, 'dc':'#BB2222',
    },
    fname='gantt_v1',
)

# ════════════════════════════════════════════════════════════════════════════
# V2  "Warm" — tonos calidos, cabecera pizarra, barra de fase en etiquetas
# ════════════════════════════════════════════════════════════════════════════
print("Generating V2: Warm...")
draw(
    phases={
        1: ("Business Understanding", "#C06878"),   # rosa empolvado
        2: ("Data Understanding",     "#5E9E78"),   # verde salvia
        3: ("Data Preparation",       "#C8A030"),   # ambar
        4: ("Modelling",              "#C87040"),   # siena
        5: ("Evaluation",             "#4088B8"),   # azul acero
        6: ("Deployment & Defence",   "#40A8A8"),   # verde-azul
    },
    s={
        'mhf':'#4A5068', 'mht':'white',    'mhe':'white',
        'whf':'#EEEEF2', 'wht':'#555555',
        'ev':'#F8F8FA',  'od':'#FFFFFF',
        'gc':'#E8E8EE',  'gl':0.30,
        'mc':'#C0C0CC',  'ml':0.80,
        'bc':'#888888',  'tx':'#222222',
        'rn':0.22, 'ba':0.80, 'sb':0.025, 'dc':'#AA2222',
    },
    fname='gantt_v2',
)

# ════════════════════════════════════════════════════════════════════════════
# V3  "Dark" — fondo oscuro, barras brillantes, estilo dashboard
# ════════════════════════════════════════════════════════════════════════════
print("Generating V3: Dark...")
draw(
    phases={
        1: ("Business Understanding", "#C080C8"),   # purpura claro
        2: ("Data Understanding",     "#60D888"),   # verde lima
        3: ("Data Preparation",       "#F0D058"),   # amarillo
        4: ("Modelling",              "#F09050"),   # naranja
        5: ("Evaluation",             "#58AAF5"),   # azul cielo
        6: ("Deployment & Defence",   "#50E0E0"),   # cian
    },
    s={
        'bg':'#1A1F2E',
        'mhf':'#252D42', 'mht':'#D0DCFF', 'mhe':'#353D55',
        'whf':'#252D42', 'wht':'#8A98CC',
        'ev':'#1E2438',  'od':'#1A1F2E',
        'gc':'#2A3050',  'gl':0.40,
        'mc':'#3A4460',  'ml':0.80,
        'bc':'#3A4460',  'tx':'#C0CCEC',
        'rn':0.18, 'ba':0.90, 'glow':True, 'dc':'#FF7070',
        'lbg':'#252D42', 'lec':'#3A4460', 'ltx':'#C0CCEC',
    },
    fname='gantt_v3',
)

print("Done. All 3 variants saved in reports/figures/")
