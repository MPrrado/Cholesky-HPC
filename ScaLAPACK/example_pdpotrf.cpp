#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "mpi.h"

extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern "C" void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
extern "C" void blacs_gridexit_(int*);
extern "C" int numroc_(int*, int*, int*, int*, int*);

int main(int argc, char **argv) {
    int izero=0;
    int ione=1;
    int myrank_mpi, nprocs_mpi;
    
    MPI_Init( &argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    int n = 1000;       
    int nprow = 2;   
    int npcol = 2;   
    int nb = 256;      
    char uplo='L';   
    char layout='R'; 

    if(argc > 1) n = atoi(argv[1]);
    if(argc > 2) nb = atoi(argv[2]);
    if(argc > 3) nprow = atoi(argv[3]);
    if(argc > 4) npcol = atoi(argv[4]);

    assert(nprow * npcol == nprocs_mpi);

    int iam, nprocs;
    int zero = 0;
    int ictxt, myrow, mycol;
    blacs_pinfo_(&iam, &nprocs) ; 
    blacs_get_(&zero, &zero, &ictxt ); 
    blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); 
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol ); 

    int mpA = numroc_( &n, &nb, &myrow, &izero, &nprow ); 
    int nqA = numroc_( &n, &nb, &mycol, &izero, &npcol ); 

    double *A = (double *)calloc(mpA*nqA,sizeof(double));
    if (A==NULL){ 
        // Si hay error de memoria, lo mandamos por la salida de errores (stderr) para no romper el CSV
        fprintf(stderr, "Error of memory allocation A on proc %dx%d\n", myrow, mycol); 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int k = 0;
    for (int j = 0; j < nqA; j++) { 
        int l_j = j / nb; 
        int x_j = j % nb; 
        int J   = (l_j * npcol + mycol) * nb + x_j; 
        for (int i = 0; i < mpA; i++) { 
            int l_i = i / nb; 
            int x_i = i % nb; 
            int I   = (l_i * nprow + myrow) * nb + x_i; 
            if(I == J) {
                A[k] = n*n;
            } else {
                A[k] = I+J;
            }
            k++;
        }
    }

    int descA[9];
    int info;
    int lddA = mpA > 1 ? mpA : 1;
    descinit_( descA,  &n, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info);

    // --- INICIO DE LA MEDICIÓN ---
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizamos todos los procesos antes de iniciar
    double MPIt1 = MPI_Wtime();
    
    pdpotrf_(&uplo, &n, A, &ione, &ione, descA, &info);
    
    MPI_Barrier(MPI_COMM_WORLD); // Esperamos a que el último termine
    double MPIt2 = MPI_Wtime();
    // --- FIN DE LA MEDICIÓN ---

    // Imprimir el formato CSV exacto solo desde el proceso líder
    if (myrank_mpi == 0) {
        printf("%d,%d,%f\n", n, nprocs_mpi, MPIt2 - MPIt1);
    }

    free(A);
    blacs_gridexit_(&ictxt);
    MPI_Finalize();
    return 0;
}