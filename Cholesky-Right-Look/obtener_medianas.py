import os
import pandas as pd


def convertir_python(archivo_python, archivo_salida):
    """cholesky.py guarda UNA fila por (N, procesos) con todos los tiempos juntos en la
    columna 'tiempos_s', separados por ';'  (ej: "0.52;0.51;0.53").
    Esta función la pasa al mismo formato que los CSV de C: una fila por repetición
        dimension_matriz,numero_de_nucleos,tiempo"""
    print(f"Convirtiendo {archivo_python}...")
    df = pd.read_csv(archivo_python)
    df = df[df["version"] == "paralelizado"]                       # Solo las corridas paralelas

    filas = []
    for _, fila in df.iterrows():
        for t in str(fila["tiempos_s"]).split(";"):                # Separamos los tiempos de cada repetición
            filas.append({"dimension_matriz": int(fila["N"]),
                          "numero_de_nucleos": int(fila["p"]),
                          "tiempo": float(t)})

    pd.DataFrame(filas).to_csv(archivo_salida, index=False)
    print(f"¡Listo! Guardado en {archivo_salida}\n")


def procesar_csv(archivo_entrada, archivo_salida):
    print(f"Procesando {archivo_entrada}...")

    # 1. Cargar los datos crudos (una fila por repetición)
    df = pd.read_csv(archivo_entrada)

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


# Python: primero lo pasamos al formato de C
if os.path.exists("resultados_paralelo.csv"):
    convertir_python("resultados_paralelo.csv", "resultados_cholesky_python.csv")

# Medianas de los tres
for nombre in ["resultados_cholesky_txt", "resultados_cholesky_bin", "resultados_cholesky_python"]:
    if os.path.exists(f"{nombre}.csv"):
        procesar_csv(f"{nombre}.csv", f"{nombre}_mediana.csv")
    else:
        print(f"No se encontró {nombre}.csv, se omite.\n")