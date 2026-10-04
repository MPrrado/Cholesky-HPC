import pandas as pd

def procesar_csv(archivo_entrada, archivo_salida):
    print(f"Procesando {archivo_entrada}...")
    
    # 1. Cargar los datos crudos
    # Lee el CSV original y lo guarda en una tabla bidimensional llamada DataFrame (df)
    df = pd.read_csv(archivo_entrada)
    
    # 2. Agrupar y calcular la mediana
    # groupby: junta todas las filas que tengan la misma dimensión de matriz Y la misma cantidad de núcleos.
    # as_index=False: mantiene "dimension_matriz" y "numero_de_nucleos" como columnas normales en lugar de convertirlas en índices ocultos.
    # ['tiempo'].median(): de esos grupos, toma específicamente la columna tiempo y calcula el valor central (mediana).
    df_mediana = df.groupby(['dimension_matriz', 'numero_de_nucleos'], as_index=False)['tiempo'].median()
    
    # 3. Ordenar los datos
    # Acomoda la tabla final primero por tamaño de matriz (de menor a mayor) y luego por cantidad de núcleos.
    df_mediana = df_mediana.sort_values(by=['dimension_matriz', 'numero_de_nucleos'])
    
    # 4. Limpiar los decimales
    # Redondea la columna tiempo a 6 decimales para que el CSV final quede prolijo y fácil de leer.
    df_mediana['tiempo'] = df_mediana['tiempo'].round(6)
    
    # 5. Exportar el resultado
    # index=False evita que pandas agregue una columna extra al principio con los números de fila (0, 1, 2, 3...)
    df_mediana.to_csv(archivo_salida, index=False)
    
    print(f"¡Listo! Resultados guardados en {archivo_salida}\n")

# Ejecutamos la función para tus dos archivos
try:
    procesar_csv("resultados_cholesky_bin.csv", "resultados_cholesky_bin_mediana.csv")
    procesar_csv("resultados_cholesky_txt.csv", "resultados_cholesky_txt_mediana.csv")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo. Asegúrate de que los nombres sean correctos. Detalle: {e}")