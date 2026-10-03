#!/bin/bash

# Nombre del archivo ejecutable y del CSV de salida
EJECUTABLE="./cholesky_paralelo_mpi_file"
ARCHIVO_CSV="resultados_cholesky_bin.csv"

# Arreglo con las dimensiones de las matrices a evaluar
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000 10000)
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000)
DIMENSIONES=(100 200 300 400 500 1000)

# Arreglo con la cantidad de núcleos (Escalado estándar en HPC)
# Si deseas probar literalmente todos los números del 1 al 96, cambia esta línea por: NUCLEOS=$(seq 1 96)
# NUCLEOS=(1 2 4 8 16 32 64 96)
NUCLEOS=(1 2 4 8)

# Cantidad de repeticiones para calcular la mediana posteriormente
REPETICIONES=10

# Inicializar el archivo CSV con la cabecera
echo "dimension_matriz,numero_de_nucleos,tiempo" > $ARCHIVO_CSV

# Bucle principal
for dim in "${DIMENSIONES[@]}"; do
    matriz_bin="matriz_${dim}.bin"
    
    # Verificar que el archivo txt exista antes de intentar ejecutar
    if [ ! -f "$matriz_bin" ]; then
        echo "Error: No se encontró $matriz_bin. Saltando..." >&2
        continue
    fi
    
    echo "Iniciando pruebas para matriz de ${dim}x${dim}..." >&2

    for p in "${NUCLEOS[@]}"; do
        echo "  -> Ejecutando con $p núcleo(s)..." >&2
        
        for ((i=1; i<=REPETICIONES; i++)); do
            # Ejecutamos MPI y concatenamos la salida (stdout) al CSV.
            # Los mensajes de error (stderr) seguirán mostrándose en consola gracias al >&2
            mpirun -n $p $EJECUTABLE $dim $matriz_bin >> $ARCHIVO_CSV
            
            # Pequeño indicador de progreso en consola
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done

echo "Todas las pruebas finalizaron. Resultados guardados en $ARCHIVO_CSV" >&2