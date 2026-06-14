import pandas as pd
from pathlib import Path

# ── Patrones ordenados por prioridad ─────────────────────────────────────────

PATRONES_ID_PACIENTE = [
    "id_paciente",
    "Id_Paciente",
    "Id Paciente",
    "idpaci",
    "Identificador de Paciente",   # strip() lo limpia antes de comparar
]

PATRONES_ID_EPISODIO = [
    "id_episodio",
    "Episodio (único)",
    "Episodio (unico)",
    "Episodio (único) INTER",
    "Episodio (unico) INTER",
    "Episodio (único) SOLIC",
    "Episodio (unico) SOLIC",
]

PATRONES_FECHA = [
    # — canónico, nunca se toca si ya existe —
    "fecha_evento",
    # — variantes encontradas en tus archivos —
    "Fecha (administración)",                       # 024_FARMACIA_*
    "Fecha Dispensación",                           # 027_PRESBIDE_*  (fallback)
    "Hora (inicio)",                                # 009_INTERVENCIONES_INTERVENCION
    "Fecha (solicitud)",
    "Fecha Preoperatorio (contacto anestesista)",
    "Fecha validación prueba",
    "Fecha DBP",
    "Fecha (registro)",
    "Fecha (ingreso)",
    "Hora (ingreso)",
    "Hora (realización proc)",
    "Hora (form cte)",
    "Fecha Hora (inicio)",
    "Fecha resultado prueba",
    "Fecha (alta)",
    "Fecha",
    "Dia (Fecha)",
    "Día (Fecha)",
    "Día (Fecha).1",                               # último recurso
]

# Archivos que NO tienen dimensión temporal y deben copiarse tal cual
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _primer_match(columnas_stripped: list[str], patrones: list[str]) -> str | None:
    col_set = set(columnas_stripped)
    for patron in patrones:
        if patron in col_set:
            return patron
    return None


# ── Función por archivo ───────────────────────────────────────────────────────

def normalizar_columnas_csv_seguro(path_csv, output_folder, verbose=True):
    path_csv = Path(path_csv)

    df = pd.read_csv(
        path_csv,
        sep=",",
        engine="python",
        quotechar='"',
        on_bad_lines="skip",
        encoding="utf-8",
    )

    # Strip ANTES de cualquier comparación
    df.columns = df.columns.str.strip()
    columnas = list(df.columns)

    renombres      = {}
    no_encontradas = []

    # ── Archivo estático: copiar sin tocar ───────────────────────────────────
    if path_csv.name in ARCHIVOS_ESTATICOS:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_folder / path_csv.name, index=False)
        if verbose:
            print(f"    → estático, copiado sin cambios")
        return {"renombres": {}, "no_encontradas": [], "estatico": True}

    # ── Caso especial: 008 no tiene id_paciente, se recupera via cod Solicitud ──
    # Join con 009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv (misma carpeta)
    if path_csv.name == "008_INTERVENCIONES_QUIRURGICAS_CONTACTO_ANESTESIA.csv":
        ref_path = path_csv.parent / "009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv"
        if ref_path.exists():
            df_ref_008 = pd.read_csv(ref_path, engine="python", on_bad_lines="skip",
                                     usecols=["cod Solicitud", "Id Paciente"])
            df_ref_008 = df_ref_008.rename(columns={"Id Paciente": "id_paciente"})
            df_ref_008 = df_ref_008.drop_duplicates(subset="cod Solicitud")
            df = df.merge(df_ref_008, on="cod Solicitud", how="left")
            columnas = list(df.columns)
            if verbose:
                matched = df["id_paciente"].notna().sum()
                print(f"    ✔ id_paciente recuperado via cod Solicitud ({matched}/{len(df)} filas)")
        else:
            if verbose:
                print(f"    ⚠ No se encontró 009 para recuperar id_paciente de 008")

    # ── id_paciente ──────────────────────────────────────────────────────────
    if "id_paciente" not in columnas:
        match = _primer_match(columnas, PATRONES_ID_PACIENTE)
        if match:
            df = df.rename(columns={match: "id_paciente"})
            renombres[match] = "id_paciente"
            columnas = list(df.columns)           # actualizar tras renombrar
        else:
            no_encontradas.append("id_paciente")

    # ── id_episodio (opcional) ───────────────────────────────────────────────
    if "id_episodio" not in columnas:
        match = _primer_match(columnas, PATRONES_ID_EPISODIO)
        if match:
            df = df.rename(columns={match: "id_episodio"})
            renombres[match] = "id_episodio"
            columnas = list(df.columns)

    # ── fecha_evento ─────────────────────────────────────────────────────────
    if "fecha_evento" not in columnas:
        match = _primer_match(columnas, PATRONES_FECHA)
        if match:
            df = df.rename(columns={match: "fecha_evento"})
            renombres[match] = "fecha_evento"
        else:
            no_encontradas.append("fecha_evento")

    # ── Logging ──────────────────────────────────────────────────────────────
    if verbose:
        for viejo, nuevo in renombres.items():
            print(f"    ✔ '{viejo}'  →  '{nuevo}'")
        if no_encontradas:
            print(f"    ⚠  No encontradas: {no_encontradas}")

    # ── Guardar ──────────────────────────────────────────────────────────────
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_folder / path_csv.name, index=False)

    return {"renombres": renombres, "no_encontradas": no_encontradas, "estatico": False}


# ── Función de lote ───────────────────────────────────────────────────────────

def normalizar_CSVs(input_folder, output_folder, verbose=True):
    input_folder  = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    archivos = sorted(input_folder.glob("*.csv"))
    print(f"Archivos encontrados: {len(archivos)}\n")

    resumen_problemas = {}
    resumen_estaticos = []

    for archivo in archivos:
        print(f"Procesando {archivo.name}…")
        try:
            resultado = normalizar_columnas_csv_seguro(archivo, output_folder, verbose=verbose)
            if resultado.get("estatico"):
                resumen_estaticos.append(archivo.name)
            elif resultado["no_encontradas"]:
                resumen_problemas[archivo.name] = resultado["no_encontradas"]
        except Exception as e:
            print(f"    ✖ Error: {type(e).__name__}: {e}")
            resumen_problemas[archivo.name] = [f"ERROR: {e}"]

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    if resumen_estaticos:
        print(f"📋 Estáticos copiados sin cambios ({len(resumen_estaticos)}):")
        for n in resumen_estaticos:
            print(f"   {n}")

    if resumen_problemas:
        print(f"\n⚠  Archivos con columnas no resueltas:")
        for nombre, cols in resumen_problemas.items():
            print(f"   {nombre}: {cols}")
    else:
        print("\n✅ Todos los archivos normalizados sin incidencias")
    print("─" * 60)