from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def analizar_segmentacion(divided_folder):
    """
    Analiza la distribución de filas y pacientes únicos por segmento para cada archivo.
    
    Parámetros:
        divided_folder: carpeta raíz con subcarpetas pre/intra/post/sin_referencia/static
    """
    divided_folder = Path(divided_folder)
    segmentos = ["pre", "intra", "post", "sin_referencia"]
    colores   = {"pre": "#4C72B0", "intra": "#DD8452", "post": "#55A868", "sin_referencia": "#C44E52"}

    # ── 1. Recopilar métricas por archivo y segmento ──────────────────────────
    archivos_todos = set()
    for seg in segmentos:
        for f in (divided_folder / seg).glob("*.csv"):
            archivos_todos.add(f.name)

    registros = []
    for archivo in sorted(archivos_todos):
        for seg in segmentos:
            path = divided_folder / seg / archivo
            if path.exists():
                df = pd.read_csv(path, engine="python", on_bad_lines="skip")
                n_filas = len(df)
                n_pacs  = df["id_paciente"].nunique() if "id_paciente" in df.columns else 0
            else:
                n_filas = 0
                n_pacs  = 0
            registros.append({
                "archivo":   archivo,
                "segmento":  seg,
                "filas":     n_filas,
                "pacientes": n_pacs,
            })

    df_stats = pd.DataFrame(registros)

    # ── 2. Calcular totales y porcentajes ─────────────────────────────────────
    totales = df_stats.groupby("archivo")["filas"].sum().rename("total_filas")
    df_stats = df_stats.merge(totales, on="archivo")
    df_stats["pct"] = (df_stats["filas"] / df_stats["total_filas"].replace(0, pd.NA) * 100).round(1)

    # ── 3. Tabla resumen ──────────────────────────────────────────────────────
    tabla = df_stats.pivot_table(
        index="archivo",
        columns="segmento",
        values=["filas", "pct", "pacientes"],
        aggfunc="first"
    ).round(1)

    # Aplanar multiindex de columnas
    tabla.columns = [f"{val}_{seg}" for val, seg in tabla.columns]
    tabla["total_filas"] = df_stats.groupby("archivo")["filas"].sum()

    # Archivos vacíos en algún segmento
    vacios = df_stats[df_stats["filas"] == 0].groupby("archivo")["segmento"].apply(list)

    print("=" * 80)
    print("RESUMEN DE SEGMENTACIÓN")
    print("=" * 80)

    # Tabla de filas
    print("\n📊 FILAS POR SEGMENTO:")
    filas_tabla = df_stats.pivot_table(
        index="archivo", columns="segmento", values="filas", aggfunc="first"
    ).fillna(0).astype(int)
    filas_tabla["TOTAL"] = filas_tabla.sum(axis=1)
    print(filas_tabla.to_string())

    # Tabla de porcentajes
    print("\n📊 % POR SEGMENTO:")
    pct_tabla = df_stats.pivot_table(
        index="archivo", columns="segmento", values="pct", aggfunc="first"
    ).fillna(0).round(1)
    print(pct_tabla.to_string())

    # Tabla de pacientes únicos
    print("\n👤 PACIENTES ÚNICOS POR SEGMENTO:")
    pacs_tabla = df_stats.pivot_table(
        index="archivo", columns="segmento", values="pacientes", aggfunc="first"
    ).fillna(0).astype(int)
    print(pacs_tabla.to_string())

    # Archivos con segmentos vacíos
    print("\n⚠️  ARCHIVOS CON SEGMENTOS VACÍOS:")
    if vacios.empty:
        print("   Ninguno")
    else:
        for archivo, segs in vacios.items():
            print(f"   {archivo}: vacío en {segs}")

    # ── 4. Gráficos ───────────────────────────────────────────────────────────
    archivos_lista = sorted(archivos_todos)
    n = len(archivos_lista)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 0.5), 14))
    fig.suptitle("Análisis de Segmentación", fontsize=14, fontweight="bold", y=1.01)

    # — Gráfico 1: filas absolutas apiladas —
    ax1 = axes[0]
    bottom = pd.Series([0.0] * n, index=archivos_lista)
    for seg in segmentos:
        vals = []
        for arch in archivos_lista:
            row = df_stats[(df_stats["archivo"] == arch) & (df_stats["segmento"] == seg)]
            vals.append(row["filas"].values[0] if len(row) else 0)
        vals = pd.Series(vals, index=archivos_lista)
        ax1.bar(archivos_lista, vals, bottom=bottom, label=seg,
                color=colores[seg], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax1.set_title("Filas por segmento (absoluto)")
    ax1.set_ylabel("Nº filas")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(archivos_lista, rotation=45, ha="right", fontsize=7)
    ax1.legend(title="Segmento", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # — Gráfico 2: porcentajes apilados —
    ax2 = axes[1]
    bottom_pct = pd.Series([0.0] * n, index=archivos_lista)
    for seg in segmentos:
        vals_pct = []
        for arch in archivos_lista:
            row = df_stats[(df_stats["archivo"] == arch) & (df_stats["segmento"] == seg)]
            vals_pct.append(row["pct"].values[0] if len(row) else 0)
        vals_pct = pd.Series(vals_pct, index=archivos_lista)
        ax2.bar(archivos_lista, vals_pct, bottom=bottom_pct, label=seg,
                color=colores[seg], edgecolor="white", linewidth=0.5)
        bottom_pct += vals_pct

    ax2.set_title("Distribución porcentual por segmento")
    ax2.set_ylabel("% del total")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(archivos_lista, rotation=45, ha="right", fontsize=7)
    ax2.legend(title="Segmento", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    plt.savefig(divided_folder / "analisis_segmentacion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Gráfico guardado en {divided_folder / 'analisis_segmentacion.png'}")

    return df_stats


