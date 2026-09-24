#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double *cholesky_paralelo(double *A, int n) {
    double *L = (double*)calloc(n * n, sizeof(double)); // reserva memoria e inicializa con cero
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
            L[i * n + j] = (i == j) ? sqrt(A[i * n + i] - s) : (1.0 / L[j * n + j] * (A[i * n + j] - s));  //L[i*n+j] en el indice basicamente hacemos saltos de filas con i y decimos cuanto avanzamos en columnas, esto es solo si guardamos por filas Row-Major

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

    return L;
}

void mostrar_matriz(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

int main() {
    int n = 3;//dimension de las matrices
    double m1[] = {25, 15, -5,
                   15, 18,  0,
                   -5,  0, 11}; // tomar la matriz como un arreglo y hacer aritmetica de indices para poder usarla, ULTRA RECOMENDABLE PARA HPC, por asignar REALMENTE un espacio contiguo
    // double m2 = []
    double *c1 = cholesky_paralelo(m1, n);
    mostrar_matriz(c1, n);
    printf("\n");
    free(c1); //liberamos lo que se reservo dentro de la funcion

    n = 4;
    double m2[] = {18, 22,  54,  42,
                   22, 70,  86,  62,
                   54, 86, 174, 134,
                   42, 62, 134, 106};
    double *c2 = cholesky_paralelo(m2, n);
    mostrar_matriz(c2, n);
    free(c2);

    return 0;
}