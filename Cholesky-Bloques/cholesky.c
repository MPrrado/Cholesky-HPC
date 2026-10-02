/* =============================================================================
 * Factorización de Cholesky (A = L·Lᵀ) — Proyecto 7, Métodos Numéricos II (HPC)
 * Versión en C con MPI.
 *
 *   1) cholesky_literal      : fórmulas de la Unidad 3 tal cual, sin optimizar.
 *   2) cholesky_optimizado   : misma fórmula, pero avanzando de a BLOQUES de filas,
 *                              in-place, con punteros y un producto escalar rápido.
 *   3) cholesky_paralelizado : el optimizado, con los bloques de filas repartidos
 *                              entre procesos (reparto cíclico) + MPI_Bcast.
 *
 * Compilar:  mpicc -O3 -march=native cholesky.c -o cholesky -lm
 * Validar:   mpirun -n 4 ./cholesky test
 * Medir:     mpirun -n 4 ./cholesky 1000 2000 4000 8000
 *            (repetir con -n 1, 2, 4, 8; los resultados se agregan a resultados_c.csv)
 *
 * La matriz se guarda como un arreglo 1D "por filas" (row-major):
 *   el elemento (i, j) de una matriz n×n está en  M[i*n + j]
 * ===========================================================================*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TAM_BLOQUE 64                  /* Filas por bloque (probar 32, 64, 128) */

int yo;                                /* Mi número de proceso (rango): 0, 1, 2... */
int cant_procesos;                     /* Cuántos procesos hay en total */


/* =============================================================================
 * GENERACIÓN DE DATOS
 * A[i][j] = 1 / (1 + |i - j|)  fuera de la diagonal,   A[i][i] = n
 * |i-j| = |j-i| -> simétrica.  Diagonal n > suma del resto de la fila -> definida positiva.
 * ===========================================================================*/
double valor_A(int i, int j, int n)
{
    if (i == j) return n;                              /* Diagonal = n */
    return 1.0 / (1.0 + abs(i - j));                   /* Más lejos de la diagonal -> más chico */
}

double *generar_matriz(int n)
{
    double *A = malloc((size_t)n * n * sizeof(double));    /* size_t: con n grande, n*n no entra en un int */
    if (A == NULL) { printf("Sin memoria para n=%d\n", n); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[(size_t)i * n + j] = valor_A(i, j, n);       /* Elemento (i, j) en la posición i*n + j */
    return A;
}


/* =============================================================================
 * 1) CHOLESKY LITERAL — fórmulas de la Unidad 3, sin optimizar
 *    l_ii = √( a_ii − Σ_{k<i} l_ik² )
 *    l_ji = ( a_ji − Σ_{k<i} l_jk·l_ik ) / l_ii        para j = i+1 … n
 * ===========================================================================*/
double *cholesky_literal(const double *A, int n)
{
    double *L = calloc((size_t)n * n, sizeof(double));     /* L llena de ceros */
    if (L == NULL) { printf("Sin memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < n; i++) {                          /* Recorremos cada columna i de izquierda a derecha */
        double suma = 0.0;
        for (int k = 0; k < i; k++)                        /* Σ l_ik² : cuadrados a la izquierda de la diagonal */
            suma = suma + pow(L[(size_t)i * n + k], 2);
        L[(size_t)i * n + i] = sqrt(A[(size_t)i * n + i] - suma);   /* Elemento diagonal */
        for (int j = i + 1; j < n; j++) {                  /* Cada fila j debajo de la diagonal */
            suma = 0.0;
            for (int k = 0; k < i; k++)                    /* Σ l_jk·l_ik : fila j por fila i */
                suma = suma + L[(size_t)j * n + k] * L[(size_t)i * n + k];
            L[(size_t)j * n + i] = (A[(size_t)j * n + i] - suma) / L[(size_t)i * n + i];  /* Divide cada vez: tosco a propósito */
        }
    }
    return L;
}


/* =============================================================================
 * HERRAMIENTAS DEL OPTIMIZADO Y DEL PARALELIZADO
 * ===========================================================================*/

/* Producto escalar x·y de largo 'largo'.
 * Usa 4 sumas parciales en vez de 1: así cada suma no espera a la anterior y la CPU
 * puede hacer varias multiplicaciones a la vez (vectorización SIMD con -O3 -march=native).
 * 'restrict' le promete al compilador que x e y no se pisan en memoria. */
double producto_escalar(const double *restrict x, const double *restrict y, int largo)
{
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;             /* 4 acumuladores independientes */
    int k = 0;
    for (; k + 3 < largo; k += 4) {                    /* De a 4 elementos por vuelta */
        s0 += x[k]     * y[k];
        s1 += x[k + 1] * y[k + 1];
        s2 += x[k + 2] * y[k + 2];
        s3 += x[k + 3] * y[k + 3];
    }
    for (; k < largo; k++)                             /* Los que sobran (si largo no es múltiplo de 4) */
        s0 += x[k] * y[k];
    return (s0 + s1) + (s2 + s3);                      /* Juntamos las 4 sumas */
}

/* Completa las columnas [col_ini, col_fin) de UNA fila i de L con la fórmula de siempre:
 *     l_ij = ( a_ij − fila_i · fila_j ) / l_jj
 * fila_i        : puntero a la fila i (en esas columnas todavía tiene A; a la izquierda ya tiene L)
 * filas_bloque  : las filas col_ini, col_ini+1, ... de L ya terminadas, una detrás de otra
 * salto         : cuántos doubles hay entre el comienzo de una fila de filas_bloque y la siguiente */
void completar_fila(double *fila_i, const double *filas_bloque, int salto, int col_ini, int col_fin)
{
    for (int j = col_ini; j < col_fin; j++) {                          /* Cada columna j del bloque */
        const double *fila_j = filas_bloque + (size_t)(j - col_ini) * salto;  /* Puntero a la fila j de L */
        double suma = producto_escalar(fila_i, fila_j, j);             /* Σ_{k<j} l_ik·l_jk */
        fila_i[j] = (fila_i[j] - suma) / fila_j[j];                    /* l_ij */
    }
}

/* Termina la fila i del bloque de la diagonal: primero sus columnas del bloque a la
 * izquierda de la diagonal, después la diagonal con la raíz cuadrada. */
void completar_fila_diagonal(double *fila_i, int i, const double *filas_bloque, int salto, int col_ini)
{
    completar_fila(fila_i, filas_bloque, salto, col_ini, i);           /* Columnas col_ini .. i-1 */
    double d = fila_i[i] - producto_escalar(fila_i, fila_i, i);        /* a_ii − Σ l_ik² */
    if (d <= 0) {                                                      /* La raíz de algo ≤ 0 no sirve */
        printf("Error: A no es definida positiva (fila %d)\n", i);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fila_i[i] = sqrt(d);                                               /* l_ii */
}


/* =============================================================================
 * 2) CHOLESKY OPTIMIZADO — secuencial, por bloques, in-place
 *
 *  Misma fórmula que el literal. Cambia el ORDEN: avanzamos de a TAM_BLOQUE filas.
 *  Por cada bloque k:
 *    Paso 1) Terminar las filas del bloque k (hasta la diagonal).
 *    Paso 2) Con esas filas, completar las columnas del bloque k en TODAS las filas de abajo.
 *  Ventaja: las pocas filas del bloque k quedan en la memoria caché y se reusan para
 *  todas las filas de abajo, en vez de ir a buscarlas a la RAM una y otra vez.
 *  In-place: L se escribe encima del triángulo inferior de A (no hace falta otra matriz).
 * ===========================================================================*/
void cholesky_optimizado(double *A, int n)
{
    for (int col_ini = 0; col_ini < n; col_ini += TAM_BLOQUE) {        /* Bloques de izquierda a derecha */
        int col_fin = col_ini + TAM_BLOQUE;                            /* Dónde termina el bloque... */
        if (col_fin > n) col_fin = n;                                  /* ...el último puede ser más chico */
        double *filas_k = A + (size_t)col_ini * n;                     /* Puntero a la primera fila del bloque k */

        for (int i = col_ini; i < col_fin; i++)                        /* Paso 1: filas del bloque k */
            completar_fila_diagonal(A + (size_t)i * n, i, filas_k, n, col_ini);

        for (int i = col_fin; i < n; i++)                              /* Paso 2: todas las filas de abajo */
            completar_fila(A + (size_t)i * n, filas_k, n, col_ini, col_fin);
    }
}


/* =============================================================================
 * 3) CHOLESKY PARALELIZADO — el optimizado, repartiendo bloques de filas con MPI
 *
 *  Reparto CÍCLICO (como repartir cartas): el bloque de filas g es del proceso g % cant_procesos.
 *  Cada proceso guarda SOLO sus filas, una pegada a la otra, en 'mis_filas'.
 *  ¿Por qué cíclico? Porque en cada paso solo se trabaja con las filas de ABAJO; si cada
 *  proceso tuviera un tramo seguido, los de arriba se quedarían sin trabajo enseguida.
 *
 *  Por cada bloque k:
 *    Paso 1) SOLO EL DUEÑO termina las filas del bloque k        (parte secuencial)
 *    Paso 2) El dueño las manda a todos con MPI_Bcast            (comunicación)
 *    Paso 3) TODOS completan sus filas de abajo, al mismo tiempo (parte paralela)
 * ===========================================================================*/

/* Dónde está la fila global i dentro de mis_filas (de quien sea dueño de esa fila). */
int fila_local(int i)
{
    int bloque = i / TAM_BLOQUE;                                       /* En qué bloque global está la fila */
    int bloque_local = bloque / cant_procesos;                         /* Es el 0º, 1º, 2º... bloque de su dueño */
    return bloque_local * TAM_BLOQUE + i % TAM_BLOQUE;                 /* Bloques anteriores + posición dentro del bloque */
}

/* Cuántas filas me tocan a mí. */
int contar_mis_filas(int n)
{
    int total = 0;
    for (int i = 0; i < n; i++)                                        /* Recorro todas las filas... */
        if ((i / TAM_BLOQUE) % cant_procesos == yo) total++;           /* ...y cuento las de mis bloques */
    return total;
}

/* Cada proceso genera SOLO sus filas de A (no hace falta armar la matriz completa). */
double *generar_mis_filas(int n)
{
    double *mis_filas = malloc((size_t)contar_mis_filas(n) * n * sizeof(double) + 1);
    if (mis_filas == NULL) { printf("Sin memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < n; i++)
        if ((i / TAM_BLOQUE) % cant_procesos == yo)                    /* Solo las filas de mis bloques */
            for (int j = 0; j < n; j++)
                mis_filas[(size_t)fila_local(i) * n + j] = valor_A(i, j, n);
    return mis_filas;
}

void cholesky_paralelizado(double *mis_filas, int n)
{
    int total_bloques = (n + TAM_BLOQUE - 1) / TAM_BLOQUE;            /* División redondeando para arriba */
    double *sobre = malloc((size_t)TAM_BLOQUE * n * sizeof(double));   /* "Sobre" para mandar/recibir las filas del bloque k */
    if (sobre == NULL) { printf("Sin memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int k = 0; k < total_bloques; k++) {                          /* Un turno por bloque, de izq. a derecha */
        int col_ini = k * TAM_BLOQUE;                                  /* Primera fila/columna del bloque k */
        int col_fin = col_ini + TAM_BLOQUE;
        if (col_fin > n) col_fin = n;                                  /* El último bloque puede ser más chico */
        int alto_k = col_fin - col_ini;                                /* Cuántas filas tiene el bloque k */
        int dueno = k % cant_procesos;                                 /* A quién le tocó el bloque k */

        /* ---- PASO 1: SOLO EL DUEÑO termina las filas del bloque k (los demás esperan en el Bcast) */
        if (yo == dueno) {
            double *filas_k = mis_filas + (size_t)fila_local(col_ini) * n;     /* Mis filas del bloque k */
            for (int f = 0; f < alto_k; f++) {
                int i = col_ini + f;                                           /* Número global de la fila */
                double *fila_i = filas_k + (size_t)f * n;
                completar_fila_diagonal(fila_i, i, filas_k, n, col_ini);       /* Igual que en el optimizado */
                memcpy(sobre + (size_t)f * col_fin, fila_i, col_fin * sizeof(double));  /* La meto en el sobre (solo hasta col_fin) */
            }
        }

        /* ---- PASO 2: EL DUEÑO MANDA EL SOBRE A TODOS */
        MPI_Bcast(sobre, alto_k * col_fin, MPI_DOUBLE, dueno, MPI_COMM_WORLD); /* Después de esto, todos tienen el sobre */

        /* ---- PASO 3: CADA UNO completa SUS filas de abajo, TODOS AL MISMO TIEMPO */
        for (int g = k + 1; g < total_bloques; g++) {                  /* Bloques debajo del bloque k... */
            if (g % cant_procesos != yo) continue;                     /* ...pero solo los míos */
            int primera = g * TAM_BLOQUE;                              /* Primera fila del bloque g */
            int ultima = primera + TAM_BLOQUE;
            if (ultima > n) ultima = n;
            for (int i = primera; i < ultima; i++)                     /* Cada fila de ese bloque */
                completar_fila(mis_filas + (size_t)fila_local(i) * n,  /* Mi fila i... */
                               sobre, col_fin,                         /* ...usando las filas del sobre (salto = col_fin) */
                               col_ini, col_fin);                      /* ...en las columnas del bloque k */
        }
    }
    free(sobre);
}

/* Junta en el proceso 0 todas las filas de L (solo para validar con n chico).
 * Usa MPI_Send / MPI_Recv fila por fila, en orden. */
double *juntar_L(double *mis_filas, int n)
{
    double *L = NULL;
    if (yo == 0) L = calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; i++) {
        int dueno = (i / TAM_BLOQUE) % cant_procesos;                  /* Quién tiene la fila i */
        double *fila = mis_filas + (size_t)fila_local(i) * n;
        if (yo == 0 && dueno == 0)
            memcpy(L + (size_t)i * n, fila, n * sizeof(double));      /* Es mía: la copio */
        else if (yo == 0)
            MPI_Recv(L + (size_t)i * n, n, MPI_DOUBLE, dueno, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  /* La recibo */
        else if (yo == dueno)
            MPI_Send(fila, n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);       /* Soy el dueño: la mando al 0 */
    }
    if (yo == 0)                                                       /* Ceros arriba de la diagonal */
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) L[(size_t)i * n + j] = 0.0;
    return L;
}


/* =============================================================================
 * VALIDACIÓN
 * ===========================================================================*/

/* Pone ceros arriba de la diagonal (el optimizado deja ahí restos de A). */
void dejar_triangular(double *M, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) M[(size_t)i * n + j] = 0.0;
}

/* residuo = ||A − L·Lᵀ|| / ||A||   (≈ 1e-16 si L es correcta: error de redondeo de double) */
double residuo(const double *A, const double *L, int n)
{
    double error = 0.0, norma = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {                                 /* Simétrica: alcanza con j ≤ i */
            double LLt = producto_escalar(L + (size_t)i * n, L + (size_t)j * n, j + 1);  /* (L·Lᵀ)_ij */
            double dif = A[(size_t)i * n + j] - LLt;
            double peso = (i == j) ? 1.0 : 2.0;                        /* Los de fuera de la diagonal cuentan doble */
            error += peso * dif * dif;
            norma += peso * A[(size_t)i * n + j] * A[(size_t)i * n + j];
        }
    return sqrt(error / norma);
}

/* Diferencia máxima elemento a elemento entre dos matrices. */
double diferencia_maxima(const double *X, const double *Y, int n)
{
    double maxima = 0.0;
    for (size_t t = 0; t < (size_t)n * n; t++)
        if (fabs(X[t] - Y[t]) > maxima) maxima = fabs(X[t] - Y[t]);
    return maxima;
}

void test(void)
{
    if (yo == 0) {
        printf("== Validacion (%d procesos, bloque=%d) ==\n", cant_procesos, TAM_BLOQUE);
        double A3[9] = {25, 15, -5,  15, 18, 0,  -5, 0, 11};           /* Ejemplo con solución conocida */
        double esperado[9] = {5, 0, 0,  3, 3, 0,  -1, 1, 3};
        double *L3 = cholesky_literal(A3, 3);
        printf("  3x3 literal:    %s\n", diferencia_maxima(L3, esperado, 3) < 1e-12 ? "OK" : "FALLA");
        cholesky_optimizado(A3, 3);  dejar_triangular(A3, 3);         /* In-place: A3 pasa a ser L */
        printf("  3x3 optimizado: %s\n", diferencia_maxima(A3, esperado, 3) < 1e-12 ? "OK" : "FALLA");
        free(L3);
    }
    int tamanos[3] = {300, 1000, 1001};                                /* 1001: el último bloque queda más chico */
    for (int t = 0; t < 3; t++) {
        int n = tamanos[t];
        double *mis_filas = generar_mis_filas(n);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        cholesky_paralelizado(mis_filas, n);
        MPI_Barrier(MPI_COMM_WORLD);
        double t_par = MPI_Wtime() - t0;
        double *L_par = juntar_L(mis_filas, n);
        if (yo == 0) {
            double *A = generar_matriz(n);
            t0 = MPI_Wtime();
            double *L_lit = cholesky_literal(A, n);
            double t_lit = MPI_Wtime() - t0;
            double *L_opt = generar_matriz(n);
            t0 = MPI_Wtime();
            cholesky_optimizado(L_opt, n);
            double t_opt = MPI_Wtime() - t0;
            dejar_triangular(L_opt, n);
            printf("  N=%5d literal      tiempo=%8.4f s  residuo=%.2e\n", n, t_lit, residuo(A, L_lit, n));
            printf("  N=%5d optimizado   tiempo=%8.4f s  residuo=%.2e  max|L-Llit|=%.2e\n",
                   n, t_opt, residuo(A, L_opt, n), diferencia_maxima(L_opt, L_lit, n));
            printf("  N=%5d paralelizado tiempo=%8.4f s  residuo=%.2e  max|L-Llit|=%.2e\n",
                   n, t_par, residuo(A, L_par, n), diferencia_maxima(L_par, L_lit, n));
            free(A); free(L_lit); free(L_opt); free(L_par);
        }
        free(mis_filas);
    }
}


/* =============================================================================
 * BENCHMARK: para cada N mide literal, optimizado y paralelizado (mediana de varias corridas)
 * ===========================================================================*/
int comparar(const void *a, const void *b)                             /* Para ordenar con qsort */
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double mediana(double *t, int r)
{
    qsort(t, r, sizeof(double), comparar);                             /* Ordeno los tiempos... */
    return (r % 2) ? t[r / 2] : (t[r / 2 - 1] + t[r / 2]) / 2;         /* ...y tomo el del medio */
}

void benchmark(int n, FILE *csv)
{
    int reps = (n <= 4000) ? 5 : 3;                                    /* Menos repeticiones con N grande */
    double tiempos[5];
    double t_lit = -1, t_opt = 0, t_par = 0;

    /* ---- Secuenciales: solo el proceso 0 (los demás esperan en la barrera) */
    if (yo == 0) {
        double *A = generar_matriz(n);
        if (n <= 2000) {                                               /* El literal es lento: solo N chico, 1 corrida */
            double t0 = MPI_Wtime();
            double *L = cholesky_literal(A, n);
            t_lit = MPI_Wtime() - t0;
            free(L);
        }
        double *copia = malloc((size_t)n * n * sizeof(double));
        for (int r = 0; r < reps; r++) {
            memcpy(copia, A, (size_t)n * n * sizeof(double));         /* Copia nueva: el optimizado sobrescribe */
            double t0 = MPI_Wtime();
            cholesky_optimizado(copia, n);
            tiempos[r] = MPI_Wtime() - t0;
        }
        t_opt = mediana(tiempos, reps);
        free(A); free(copia);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Paralelizado: todos los procesos */
    double *original = generar_mis_filas(n);
    size_t bytes = (size_t)contar_mis_filas(n) * n * sizeof(double);
    double *mis_filas = malloc(bytes + 1);
    for (int r = 0; r < reps; r++) {
        memcpy(mis_filas, original, bytes);
        MPI_Barrier(MPI_COMM_WORLD);                                   /* Todos arrancan juntos */
        double t0 = MPI_Wtime();
        cholesky_paralelizado(mis_filas, n);
        MPI_Barrier(MPI_COMM_WORLD);                                   /* Esperamos al más lento */
        tiempos[r] = MPI_Wtime() - t0;
    }
    t_par = mediana(tiempos, reps);
    free(original); free(mis_filas);

    if (yo == 0) {
        double flops = (double)n * n * n / 3.0;                        /* Cholesky hace ~n³/3 operaciones */
        double speedup = t_opt / t_par;                                /* Contra el MEJOR secuencial (optimizado) */
        double eficiencia = speedup / cant_procesos;
        char lit[32] = "      -";
        if (t_lit >= 0) snprintf(lit, sizeof lit, "%9.3f", t_lit);
        printf("%6d %3d %10s %10.3f %10.3f %8.2f %9.0f%% %8.2f %8.2f\n",
               n, cant_procesos, lit, t_opt, t_par, speedup, 100 * eficiencia,
               flops / t_opt / 1e9, flops / t_par / 1e9);
        fprintf(csv, "%d,%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                n, cant_procesos, TAM_BLOQUE, t_lit, t_opt, t_par, speedup, eficiencia);
        fflush(stdout); fflush(csv);
    }
}


/* =============================================================================
 * MAIN
 * ===========================================================================*/
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);                                            /* Arranca MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &yo);                                /* Quién soy */
    MPI_Comm_size(MPI_COMM_WORLD, &cant_procesos);                     /* Cuántos somos */

    if (argc < 2 || strcmp(argv[1], "test") == 0) {
        test();
    } else {
        FILE *csv = NULL;
        if (yo == 0) {
            FILE *existe = fopen("resultados_c.csv", "r");             /* ¿Ya existe el CSV? */
            csv = fopen("resultados_c.csv", "a");                      /* Lo abro para agregar al final */
            if (existe == NULL)                                        /* Si es nuevo, escribo el encabezado */
                fprintf(csv, "N,p,bloque,t_literal,t_optimizado,t_paralelizado,speedup,eficiencia\n");
            else fclose(existe);
            printf("     N   p  T_literal  T_optimiz  T_paralel  Speedup  Eficiencia  GF/s_opt GF/s_par\n");
        }
        for (int a = 1; a < argc; a++)                                 /* Un benchmark por cada N pedido */
            benchmark(atoi(argv[a]), csv);
        if (yo == 0) fclose(csv);
    }

    MPI_Finalize();                                                    /* Cierra MPI */
    return 0;
}