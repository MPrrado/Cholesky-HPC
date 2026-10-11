#!/bin/bash
EJECUTABLE="scalapack_cholesky.out"
ARCHIVO_CSV="../resultados/resultados_cholesky_scalapack.csv"
DIMENSIONES=(100 200 300 400 500 1000 2000 5000 10000)
NUCLEOS=(1 2 4 8 16)
REPETICIONES=1
TAM_BLOQUE=64
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

grilla() {
    case $1 in
        1)  echo "1 1" ;;
        2)  echo "1 2" ;;
        4)  echo "2 2" ;;
        8)  echo "2 4" ;;
        # 16) echo "4 4" ;;
        # 32) echo "4 8" ;;
        # 64) echo "8 8" ;;
        # 96) echo "8 12" ;;
        # *)  echo "1 $1" ;;
    esac
}

if [ ! -f $ARCHIVO_CSV ]; then
    echo "dimension_matriz,numero_de_nucleos,tiempo" > $ARCHIVO_CSV
fi

echo $'\n\nCHOLESKY CON SCALAPACK (ejemplo original)\n\n'

for dim in "${DIMENSIONES[@]}"; do
    echo "Iniciando pruebas para matriz de ${dim}x${dim}..." >&2
    for p in "${NUCLEOS[@]}"; do
        echo "  -> Ejecutando con $p núcleo(s), grilla $(grilla $p)..." >&2
        for ((i=1; i<=REPETICIONES; i++)); do
            mpirun -n $p $EJECUTABLE $dim $TAM_BLOQUE $(grilla $p) >> $ARCHIVO_CSV
            echo -n "." >&2
        done
        echo " Listo" >&2
    done
done

echo "Pruebas finalizadas. Resultados en $ARCHIVO_CSV" >&2
