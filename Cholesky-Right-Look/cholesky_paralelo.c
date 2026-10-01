#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void cholesky_paralelo(double *A, int n, int id_proceso, int cantidad_procesos) {

    /*
        En el metodo paralelizado, sobreescribiremos la matriz A de entrada a fines de ahorrar memoria. 
    */
    
    double *buffer = (double * )calloc(n, sizeof(double)); // COLUMNA DE BUFFER QUE MANDAREMOS

    for(int k = 0; k < n; k++)
    {
        int id_duenio = (k % cantidad_procesos);
        if (id_proceso == id_duenio)
        {
            buffer[0] = (A[k*n+k] = sqrt(A[k*n + k])); //asignacion en cadena a = (b = c) primero b = c y luego lo que tiene b se asigna a a
            for (int i=1; i < n; i++)
            {
                buffer[i] = A[i*n + k] = (double) A[i*n + k] / buffer[k];
            }        
        }

        int columna_comienzo = id_proceso; //esta variable nos permite saber en que columna comenzar a realizar el calculo de los demas elementos de la matriz L
        while(columna_comienzo <= k)
        {
            columna_comienzo += cantidad_procesos; // vamos saltando las columnas 
        }

       
        for (int j = columna_comienzo; j < n; j+= cantidad_procesos)
        {
            for (int i = j ; i <n ; i ++)
            {   
                A[i*n + j] = A[i*n + j] - (buffer[i] * buffer[j]);
            }
        }
    }
    free(buffer);

}

void mostrar_matriz(double *A, int n)
{
    double cero = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            if(j > i)
            {
                printf("%10.5f ", cero);//el numero a imprimir ocupa 10 caracteres minimos y con precision de 5 decimales
            }else
            {
                printf("%10.5f ", A[i * n + j]);//el numero a imprimir ocupa 10 caracteres minimos y con precision de 5 decimales
            }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    
    int id_proceso, cantidad_procesos, root = 0;
    int n;//dimension de las matrices


    double m1[] = {25, 15, -5,
                   15, 18,  0,
                   -5,  0, 11}; // tomar la matriz como un arreglo y hacer aritmetica de indices para poder usarla, ULTRA RECOMENDABLE PARA HPC, por asignar REALMENTE un espacio contiguo ROW-MAJOR 
    // double m2 = []
    

    cholesky_paralelo(A, n,id_proceso, cantidad_procesos);
   
    // free(c1); //liberamos lo que se reservo dentro de la funcion

    // n = 4;
    // double m2[] = {18, 22,  54,  42,
    //                22, 70,  86,  62,
    //                54, 86, 174, 134,
    //                42, 62, 134, 106};
    // cholesky_paralelo(m2, n,id_proceso, cantidad_procesos);
    // if (id_proceso == root)
    // {
    //     mostrar_matriz(m2, n);
    //     printf("\n");
    // }
    return 0;
}