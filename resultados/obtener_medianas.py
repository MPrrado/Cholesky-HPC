import os
import glob
import pandas as pd



def procesar_csv(archivo_entrada, archivo_salida):
    print(f"Procesando {os.path.basename(archivo_entrada)}...")

    # 1. Cargar los datos crudos (una fila por repetición)
    df = pd.read_csv(archivo_entrada)
    df = df.dropna(subset=["tiempo"])                              # descartamos corridas que fallaron (sin tiempo)

    # 2. Agrupar por (dimensión, núcleos) y calcular la mediana del tiempo
    grupos = df.groupby(["dimension_matriz", "numero_de_nucleos"])
    df_mediana = grupos["tiempo"].median().reset_index()

    # 3. Cuántas repeticiones hubo en cada grupo (para controlar que no falte ninguna)
    df_mediana["repeticiones"] = grupos.size().values

    # 4. Ordenar y redondear
    df_mediana = df_mediana.sort_values(by=["dimension_matriz", "numero_de_nucleos"])
    df_mediana["tiempo"] = df_mediana["tiempo"].round(6)

    # 5. Exportar
    df_mediana.to_csv(archivo_salida, index=False)
    print(f"¡Listo! Resultados guardados en {archivo_salida}\n")


# Medianas de los tres
for nombre in ["resultados/resultados_cholesky_txt", "resultados/resultados_cholesky_bin","resultados/resultados_cholesky_scalapack", "resultados/resultados_cholesky_serial"]:
    if os.path.exists(f"{nombre}.csv"):
        procesar_csv(f"{nombre}.csv", f"{nombre}_mediana.csv")
    else:
        print(f"No se encontró {nombre}.csv, se omite.\n")