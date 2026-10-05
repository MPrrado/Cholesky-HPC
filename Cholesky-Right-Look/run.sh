#!/bin/bash

# Nombre del archivo ejecutable y del CSV de salida
EJECUTABLE_TXT="./cholesky_paralelo"
EJECUTABLE_BIN="./cholesky_paralelo_mpi_file"
SCRIPT_PYTHON="../Cholesky-Bloques/cholesky.py"
ARCHIVO_CSV_TXT="resultados_cholesky_txt.csv"
ARCHIVO_CSV_BIN="resultados_cholesky_bin.csv"

# Arreglo con las dimensiones de las matrices a evaluar
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000 10000)
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000)
DIMENSIONES=(100)

# Arreglo con la cantidad de núcleos (Escalado estándar en HPC)
# Si deseas probar literalmente todos los números del 1 al 96, cambia esta línea por: NUCLEOS=$(seq 1 96)
# NUCLEOS=(1 2 4 8 16 32 64 96)
NUCLEOS=(8)

# Cantidad de repeticiones para calcular la mediana posteriormente
REPETICIONES=10

# Inicializar el archivo CSV con la cabecera (solo si no existe, para no borrar resultados anteriores)
if [ ! -f $ARCHIVO_CSV_TXT ]; then
    echo "dimension_matriz,numero_de_nucleos,tiempo" > $ARCHIVO_CSV_TXT
fi

# Bucle principal
for dim in "${DIMENSIONES[@]}"; do
    matriz_txt="matriz_${dim}.txt"
    
    # Verificar que el archivo txt exista antes de intentar ejecutar
    if [ ! -f "$matriz_txt" ]; then
        echo "Error: No se encontró $matriz_txt. Saltando..." >&2
        continue
    fi
    
    echo "Iniciando pruebas para matriz de ${dim}x${dim}..." >&2

    for p in "${NUCLEOS[@]}"; do
        echo "  -> Ejecutando con $p núcleo(s)..." >&2
        
        for ((i=1; i<=REPETICIONES; i++)); do
            # Ejecutamos MPI y concatenamos la salida (stdout) al CSV.
            # Los mensajes de error (stderr) seguirán mostrándose en consola gracias al >&2
            mpirun -n $p $EJECUTABLE_TXT $matriz_txt >> $ARCHIVO_CSV_TXT
            
            # Pequeño indicador de progreso en consola
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done

# Inicializar el archivo CSV con la cabecera (solo si no existe, para no borrar resultados anteriores)
if [ ! -f $ARCHIVO_CSV_BIN ]; then
    echo "dimension_matriz,numero_de_nucleos,tiempo" > $ARCHIVO_CSV_BIN
fi
echo "--------------------------------------------------------------------------------------------------"
echo $'\n\nCHOLESKY CON LECTURA PARALELA\n\n'

for dim in "${DIMENSIONES[@]}"; do
    matriz_bin="matriz_${dim}.bin"
    
    # Verificar que el archivo bin exista antes de intentar ejecutar
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
            mpirun -n $p $EJECUTABLE_BIN $dim $matriz_bin >> $ARCHIVO_CSV_BIN
            
            # Pequeño indicador de progreso en consola
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done

echo "--------------------------------------------------------------------------------------------------"
echo $'\n\nCHOLESKY EN PYTHON\n\n'
 
for dim in "${DIMENSIONES[@]}"; do
    echo "Iniciando pruebas para matriz de ${dim}x${dim}..." >&2
 
    for p in "${NUCLEOS[@]}"; do
        echo "  -> Ejecutando con $p núcleo(s)..." >&2
 
        # El script de Python genera la matriz en memoria y hace las repeticiones por dentro.
        # Guarda los tiempos en resultados_paralelo.csv (obtener_medianas.py lo pasa al mismo formato que los de C)
        mpirun -n $p python3 $SCRIPT_PYTHON --paralelo --N $dim --reps $REPETICIONES > /dev/null
        echo " Listo" >&2
    done
done
 
python3 obtener_medianas.py
echo "Todas las pruebas finalizaron. Resultados guardados en los archivos ..._mediana.csv" >&2