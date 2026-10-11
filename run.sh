#!/bin/bash

echo "COMPILANDO LOS ARCHIVOS..."
mpicc Cholesky-Right-Look/cholesky_paralelo.c -lm -o Cholesky-Right-Look/cholesky_paralelo.out
mpicc Cholesky-Right-Look/cholesky_paralelo_mpi_file.c -lm -o Cholesky-Right-Look/cholesky_paralelo_mpi_file.out
gcc Cholesky_serial/cholesky_serial.c -lm -o Cholesky_serial/cholesky_serial.out
mpicxx ScaLAPACK/cholesky_optimizado.cpp -o ScaLAPACK/scalapack_cholesky.out -lscalapack-openmpi -llapack -lblas -lgfortran

# Nombre del archivo ejecutable y del CSV de salida
EJECUTABLE_TXT="Cholesky-Right-Look/cholesky_paralelo.out"
EJECUTABLE_BIN="Cholesky-Right-Look/cholesky_paralelo_mpi_file.out"
EJECUTABLE_SCALAPACK="ScaLAPACK/scalapack_cholesky.out"

ARCHIVO_CSV_SCALAPACK="resultados/resultados_cholesky_scalapack.csv"
ARCHIVO_CSV_TXT="resultados/resultados_cholesky_txt.csv"
ARCHIVO_CSV_SERIAL="resultados/resultados_cholesky_serial.csv"
ARCHIVO_CSV_BIN="resultados/resultados_cholesky_bin.csv"

# Arreglo con las dimensiones de las matrices a evaluar
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000 10000)
# DIMENSIONES=(100 200 300 400 500 1000 2000 5000)
DIMENSIONES=(100)

# Arreglo con la cantidad de núcleos (Escalado estándar en HPC) 
# Si deseas probar literalmente todos los números del 1 al 96, cambia esta línea por: NUCLEOS=$(seq 1 96)
# NUCLEOS=(1 2 4 8 16 32 64 96)
NUCLEOS=(1 2 4 8)
# NUCLEOS=(8)

# Cantidad de repeticiones para calcular la mediana posteriormente
REPETICIONES=2

# Inicializar el archivo CSV con la cabecera (solo si no existe, para no borrar resultados anteriores)
if [ ! -f $ARCHIVO_CSV_TXT ]; then
    mkdir -p resultados
    echo "dimension_matriz,numero_de_nucleos,tiempo" >> $ARCHIVO_CSV_TXT
    echo "dimension_matriz,numero_de_nucleos,tiempo" >> $ARCHIVO_CSV_SERIAL
fi

# Bucle principal
for dim in "${DIMENSIONES[@]}"; do
    matriz_txt="matrices/matriz_${dim}.txt"
    
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
            if ((p == 1)); then
                Cholesky_serial/cholesky_serial $matriz_txt | tee -a $ARCHIVO_CSV_TXT $ARCHIVO_CSV_SERIAL > /dev/null #el comando tee permite redirigir la salida a archivos y ademas a la salida estandar 
            else
                mpirun -n $p $EJECUTABLE_TXT $matriz_txt >> $ARCHIVO_CSV_TXT
            fi
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
    matriz_bin="matrices/matriz_${dim}.bin"
    
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
             if ((p == 1)); then
                Cholesky_serial/cholesky_serial $matriz_txt | tee -a $ARCHIVO_CSV_BIN $ARCHIVO_CSV_SERIAL > /dev/null
            else
                mpirun -n $p $EJECUTABLE_BIN $dim $matriz_bin >> $ARCHIVO_CSV_BIN
            fi
            
            # Pequeño indicador de progreso en consola
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done


# seccion de la version optimizada de ScaLapack

echo "--------------------------------------------------------------------------------------------------"
echo $'\n\nCHOLESKY CON SCALAPACK (ejemplo original)\n\n'


#Aqui se selecciona la dristribucion en grilla segun la cantidad de procesos, es importante mantener que sea los mas "cuadrado" posible, asi se evitan cuellos de botellas y comunicaciones infinitas, o en su defecto que procesos
#terminen su trabajo y queden desocupados mucho tiempo antes que el resto
grilla() { 
    case $1 in
        1)  echo "1 1" ;;
        2)  echo "1 2" ;;
        4)  echo "2 2" ;;
        8)  echo "2 4" ;;
        16) echo "4 4" ;;
        32) echo "4 8" ;;
        64) echo "8 8" ;;
        96) echo "8 12" ;;
        # *)  echo "1 $1" ;;
    esac
}

TAM_BLOQUE=64

if [ ! -f $ARCHIVO_CSV_SCALAPACK ]; then
    echo "dimension_matriz,numero_de_nucleos,tiempo" > $ARCHIVO_CSV_SCALAPACK
fi


for dim in "${DIMENSIONES[@]}"; do
    echo "Iniciando pruebas para matriz de ${dim}x${dim}..." >&2
    for p in "${NUCLEOS[@]}"; do
        echo "  -> Ejecutando con $p núcleo(s), grilla $(grilla $p)..." >&2
        for ((i=1; i<=REPETICIONES; i++)); do
            mpirun -n $p $EJECUTABLE_SCALAPACK $dim $TAM_BLOQUE $(grilla $p) >> $ARCHIVO_CSV_SCALAPACK
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done
 
python3 resultados/obtener_medianas.py
python3 graficar/graficar.py
echo "Todas las pruebas finalizaron. Resultados guardados en los archivos ..._mediana.csv" >&2