import pandas as pd
from pathlib import Path


def cargar_csv(path):
    """
    Carga un CSV en un DataFrame de pandas.
    """
    return pd.read_csv(path)


def detectar_columnas_fecha(df, umbral=0.8):
    """
    Detecta columnas que probablemente sean fechas.
    Intenta convertirlas a datetime y mide porcentaje válido.
    """
    columnas_fecha = []

    for col in df.columns:
        serie = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        porcentaje_validos = serie.notna().mean()

        if porcentaje_validos > umbral:
            columnas_fecha.append(col)

    return columnas_fecha


def convertir_a_datetime(df, columnas):
    """
    Convierte columnas a tipo datetime.
    """
    for col in columnas:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def segmentar_por_ingreso_alta(df, hora_ingreso, hora_alta, columna_fecha):
    """
    Segmenta el DataFrame en:
    - antes_ingreso
    - entre_ingreso_alta
    - despues_alta
    """
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce", dayfirst=True)

    antes = df[df[columna_fecha] < hora_ingreso]
    entre = df[(df[columna_fecha] >= hora_ingreso) & (df[columna_fecha] <= hora_alta)]
    despues = df[df[columna_fecha] > hora_alta]

    return {
        "antes_ingreso": antes,
        "entre_ingreso_alta": entre,
        "despues_alta": despues
    }


def guardar_segmentos(segmentos, base_path, nombre_base):
    """
    Guarda los segmentos como CSV.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for nombre, df_seg in segmentos.items():
        output_path = base_path / f"{nombre_base}_{nombre}.csv"
        df_seg.to_csv(output_path, index=False)

def primer_episodio_principal(df):
    """
    Devuelve un DataFrame con el primer episodio principal de cada paciente.
    Prioriza CIRUGIA MAYOR. Si no tiene mayor, toma el primer principal que tenga.
    """

    df_principal = df[df["Principal o secundaria"] == "PRINCIPAL"].copy()


    df_mayor = df_principal[df_principal["Tipo de procedimiento quirúrgico"] == "CIRUGIA MAYOR"].copy()
    df_menor = df_principal[df_principal["Tipo de procedimiento quirúrgico"] != "CIRUGIA MAYOR"].copy()


    df_mayor = df_mayor.sort_values(by=["Id Paciente", "Hora (inicio)"], ascending=[True, True])
    df_menor = df_menor.sort_values(by=["Id Paciente", "Hora (inicio)"], ascending=[True, True])


    mayor_first = df_mayor.groupby("Id Paciente").first()

    pacientes_con_mayor = mayor_first.index
    menor_first = df_menor[~df_menor["Id Paciente"].isin(pacientes_con_mayor)]
    menor_first = menor_first.groupby("Id Paciente").first()


    resultado = pd.concat([mayor_first, menor_first]).reset_index()

    return resultado
