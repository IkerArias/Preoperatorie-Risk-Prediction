from pathlib import Path
import pandas as pd


# ── Archivos sin dimensión temporal ───────────────────────────────────────────
ARCHIVOS_ESTATICOS = {
    "000_POBLACION_DIANA.csv",
    "001_INTERVENCIONES_QUIRURGICAS.csv",          # sin fecha_evento
    "004_DIAGNOSTICO_PRINCIPAL_CIE10.csv",          # sin fecha_evento
    "005_DIAGNOSTICO_PRINCIPAL_CIE9.csv",           # sin fecha_evento
    "006_COMORBILIDADES_CHARLSON.csv",
    "006_COMORBILIDADES_CHARLSON_NUEVO.csv",
    "021_DIAGNOSTICOS_POR_PACIENTE.csv",
    "028_INTERVENCIONES_CHECKLIST_CIRUGIA_MAYOR.csv",
    "029_INTERVENCIONES_CHECKLIST_CIRUGIA_MENOR.csv",
}


def determinar_tramo(df):
    mask_pre   = df["fecha_evento"] < df["Hora (inicio)"]
    mask_intra = (df["fecha_evento"] >= df["Hora (inicio)"]) & \
                 (df["fecha_evento"] <= df["Hora Intervención (fin)"])
    mask_post  = df["fecha_evento"] > df["Hora Intervención (fin)"]
    return df[mask_pre].copy(), df[mask_intra].copy(), df[mask_post].copy()


def guardar_segmento(df, folder, filename):
    if df.shape[0] > 0:
        cols_aux = ["Hora (inicio)", "Hora Intervención (fin)"]
        df = df.drop(columns=[c for c in cols_aux if c in df.columns])
        df.to_csv(folder / filename, index=False)
        print(f"    ✔ {folder.name}: {df.shape[0]} filas")


def normalizar_ref(df_primero):
    df_ref = df_primero.copy()
    df_ref.columns = df_ref.columns.str.strip()

    for col in ["Id Paciente", "Id_Paciente", "Identificador de Paciente"]:
        if col in df_ref.columns and "id_paciente" not in df_ref.columns:
            df_ref = df_ref.rename(columns={col: "id_paciente"})
            break

    df_ref["id_paciente"] = df_ref["id_paciente"].astype(str).str.strip()
    df_ref["Hora (inicio)"] = pd.to_datetime(df_ref["Hora (inicio)"], errors="coerce")
    df_ref["Hora Intervención (fin)"] = pd.to_datetime(
        df_ref["Hora Intervención (fin)"], errors="coerce"
    )
    df_ref = df_ref.drop_duplicates(subset="id_paciente")

    # ── Corregir horas invertidas (inicio > fin) — error en datos del hospital ─
    mask_inv = df_ref["Hora (inicio)"] > df_ref["Hora Intervención (fin)"]
    n_inv = mask_inv.sum()
    if n_inv > 0:
        print(f"  ⚠ {n_inv} cirugía(s) con Hora inicio > Hora fin → intercambiando automáticamente")
        df_ref.loc[mask_inv, ["Hora (inicio)", "Hora Intervención (fin)"]] = \
            df_ref.loc[mask_inv, ["Hora Intervención (fin)", "Hora (inicio)"]].values

    return df_ref[["id_paciente", "Hora (inicio)", "Hora Intervención (fin)"]]


def segmentar_y_guardar(input_folder, df_primero, output_folder):
    input_folder  = Path(input_folder)
    output_folder = Path(output_folder)

    # Carpetas de salida
    pre_folder      = output_folder / "pre"
    intra_folder    = output_folder / "intra"
    post_folder     = output_folder / "post"
    static_folder   = output_folder / "static"
    sinref_folder   = output_folder / "sin_referencia"

    for f in [pre_folder, intra_folder, post_folder, static_folder, sinref_folder]:
        f.mkdir(parents=True, exist_ok=True)

    # Preparar referencia
    ref = normalizar_ref(df_primero)

    archivos = sorted(input_folder.glob("*.csv"))
    print(f"Archivos encontrados: {len(archivos)}\n")

    for archivo in archivos:
        print(f"Procesando {archivo.name}…")

        # ── Estáticos → /static ───────────────────────────────────────────────
        if archivo.name in ARCHIVOS_ESTATICOS:
            df_est = pd.read_csv(archivo, engine="python", on_bad_lines="skip")
            df_est.to_csv(static_folder / archivo.name, index=False)
            print(f"    → estático guardado en /static")
            continue

        try:
            df = pd.read_csv(archivo, engine="python", on_bad_lines="skip")

            if "id_paciente" not in df.columns or "fecha_evento" not in df.columns:
                print(f"    → Saltando: sin id_paciente o fecha_evento")
                continue

            df["id_paciente"]  = df["id_paciente"].astype(str).str.strip()
            df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce")

            n_antes = len(df)
            df = df.dropna(subset=["fecha_evento"])
            if len(df) < n_antes:
                print(f"    → {n_antes - len(df)} filas con fecha_evento nula descartadas")

            # ── Eliminar columnas que colisionarían con las de referencia ─────
            # (ej. 009 tiene 'Hora Intervención (fin)' que pandas convertiría en _x/_y)
            cols_ref = ["Hora (inicio)", "Hora Intervención (fin)"]
            cols_colision = [c for c in cols_ref if c in df.columns]
            if cols_colision:
                df = df.drop(columns=cols_colision)

            # Merge con referencia
            df = df.merge(ref, on="id_paciente", how="left")

            # ── Sin referencia → /sin_referencia ─────────────────────────────
            df_sinref = df[df["Hora (inicio)"].isna()].copy()
            if len(df_sinref) > 0:
                df_sinref = df_sinref.drop(
                    columns=[c for c in ["Hora (inicio)", "Hora Intervención (fin)"]
                             if c in df_sinref.columns]
                )
                df_sinref.to_csv(sinref_folder / archivo.name, index=False)
                print(f"    → sin_referencia: {len(df_sinref)} filas guardadas en /sin_referencia")

            # Segmentar solo los que tienen referencia
            df = df.dropna(subset=["Hora (inicio)", "Hora Intervención (fin)"])

            if df.empty:
                print(f"    → Sin datos con referencia, omitido de pre/intra/post")
                continue

            df_pre, df_intra, df_post = determinar_tramo(df)
            guardar_segmento(df_pre,   pre_folder,   archivo.name)
            guardar_segmento(df_intra, intra_folder, archivo.name)
            guardar_segmento(df_post,  post_folder,  archivo.name)

        except Exception as e:
            print(f"    → ERROR: {type(e).__name__}: {e}")

    print("\nSegmentación completa ✅")
    print(f"\nEstructura de salida:")
    print(f"  {output_folder}/")
    print(f"  ├── pre/             → eventos anteriores a la cirugía")
    print(f"  ├── intra/           → eventos durante la cirugía")
    print(f"  ├── post/            → eventos posteriores a la cirugía")
    print(f"  ├── static/          → archivos sin dimensión temporal")
    print(f"  └── sin_referencia/  → pacientes sin cirugía de referencia")