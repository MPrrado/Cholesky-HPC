#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // para poder medir los tiempos

void *cholesky_paralelo(double *L, int n) {
    if (L == NULL) //caso de ingresar un NULL y no haya un segmentation fault
        exit(EXIT_FAILURE);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (i+1); j++)
        {
            double s = 0;
            for (int k = 0; k < j; k++)
            {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (i == j) ? sqrt(L[i * n + i] - s) : (1.0 / L[j * n + j] * (L[i * n + j] - s));  //L[i*n+j] en el indice basicamente hacemos saltos de filas con i y decimos cuanto avanzamos en columnas, esto es solo si guardamos por filas Row-Major

            /*
            Se usan los indices a favor para evitar recorrer mas alla de la diagonal principal, gracias a la condicion j < (i+1)
            */
            /*
            podemos usar la misma sumatoria ya que la formula general cuando i = j 
            tenemos que  l_ji = (a_ji - sum_k=1_i-1(l_jk * l_ik))/l_ii, pero como dijimos que i = j
            entonces dentro de la sumatoria nos quedaria l_ii = (a_ii - sum_k=1_i-1(l_ik * l_ik))/l_ii
            y pasando multiplicando el factor que esta diviendo a la derecha tenemos
            l_ii^2 = (a_ii - sum_k=1_i-1(l_ik * l_ik))
            por lo que l_ii = sqrt{(a_ii -sum_k=1_i-1(l_ik^2))}
            la sumatoria es la misma, se acomoda solo por los indicees
            */
        }
    }

}

void mostrar_matriz(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

int main(int argc, char **argv) {

    int n ;//dimension de las matrices
    double *A;
    FILE *archivo;
    char *nombre_archivo = argv[1];

    struct timespec inicio, fin; // estructuras de datos que nos permiten medir el tiempo

    clock_gettime(CLOCK_MONOTONIC, &inicio);

    archivo = fopen(nombre_archivo, "r");
    fscanf(archivo, "%d", &n);
    A = (double *)calloc(n*n, sizeof(double));  
    for (int i=0; i < (n*n); i++)
    {
        fscanf(archivo, "%lf", &A[i]);
    }
    fclose(archivo);

    cholesky_paralelo(A, n);
    
    clock_gettime(CLOCK_MONOTONIC, &fin);

    /*
    se obtiene la primera parte de segundos enteros y luego se suma la fraccion de nanosegundos 
    1e-9 [s]
    */
    double tiempo_total = (fin.tv_sec - inicio.tv_sec) + (fin.tv_nsec - inicio.tv_nsec) / 1e9;
    printf("%d,1,%f\n", n, tiempo_total);



    return 0;
}