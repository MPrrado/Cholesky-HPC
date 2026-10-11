import os
import glob
import pandas as pd

# Carpeta donde está este script (resultados/): así funciona sin importar desde dónde se ejecute
CARPETA = os.path.dirname(os.path.abspath(__file__))


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
    print(f"¡Listo! Resultados guardados en {os.path.basename(archivo_salida)}\n")


# Procesa TODOS los CSV crudos de esta carpeta (serial, txt, bin, scalapack, de PC y de cluster)
for archivo in sorted(glob.glob(os.path.join(CARPETA, "resultados_cholesky_*.csv"))):
    if archivo.endswith("_mediana.csv"):
        continue
    procesar_csv(archivo, archivo.replace(".csv", "_mediana.csv"))