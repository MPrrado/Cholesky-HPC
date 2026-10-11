#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "mpi.h"

/**
 * Compile with something like
 * mpicxx test_dpotrf.cpp \
 *     -L/.../scalapack/2.1.0/lib \
 *     -lscalapack
 */

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

    int n = 1000;       // (Global) Matrix size
    int nprow = 2;   // Number of row procs
    int npcol = 2;   // Number of column procs
    int nb = 256;      // (Global) Block size
    char uplo='L';   // Matrix is lower triangular
    char layout='R'; // Block cyclic, Row major processor mapping

    printf("Usage: ./test matrix_size block_size nprocs_row nprocs_col\n");

    if(argc > 1) {
        n = atoi(argv[1]);
    }
    if(argc > 2) {
        nb = atoi(argv[2]);
    }
    if(argc > 3) {
        nprow = atoi(argv[3]);
    }
    if(argc > 4) {
        npcol = atoi(argv[4]);
    }

    assert(nprow * npcol == nprocs_mpi);

    // Initialize BLACS
    int iam, nprocs;
    int zero = 0;
    int ictxt, myrow, mycol;
    blacs_pinfo_(&iam, &nprocs) ; // BLACS rank and world size
    blacs_get_(&zero, &zero, &ictxt ); // -> Create context
    blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); // Context -> Initialize the grid
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol ); // Context -> Context grid info (# procs row/col, current procs row/col)

    // Compute the size of the local matrices
    int mpA    = numroc_( &n, &nb, &myrow, &izero, &nprow ); // My proc -> row of local A
    int nqA    = numroc_( &n, &nb, &mycol, &izero, &npcol ); // My proc -> col of local A

    printf("Hi. Proc %d/%d for MPI, proc %d/%d for BLACS in position (%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size %d\n",myrank_mpi,nprocs_mpi,iam,nprocs,myrow,mycol,nprow,npcol,mpA,nqA,n,nb);

    // Allocate and fill the matrices A and B
    // A[I,J] = (I == J ? 5*n : I+J)
    double *A;
    A = (double *)calloc(mpA*nqA,sizeof(double)) ;
    if (A==NULL){ printf("Error of memory allocation A on proc %dx%d\n",myrow,mycol); exit(0); }
    int k = 0;
    for (int j = 0; j < nqA; j++) { // local col
        int l_j = j / nb; // which block
        int x_j = j % nb; // where within that block
        int J   = (l_j * npcol + mycol) * nb + x_j; // global col
        for (int i = 0; i < mpA; i++) { // local row
            int l_i = i / nb; // which block
            int x_i = i % nb; // where within that block
            int I   = (l_i * nprow + myrow) * nb + x_i; // global row
            assert(I < n);
            assert(J < n);
            if(I == J) {
                A[k] = n*n;
            } else {
                A[k] = I+J;
            }
            //printf("%d %d -> %d %d -> %f\n", i, j, I, J, A[k]);
            k++;
        }
    }

    // Create descriptor
    int descA[9];
    int info;
    int lddA = mpA > 1 ? mpA : 1;
    descinit_( descA,  &n, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info);
    if(info != 0) {
        printf("Error in descinit, info = %d\n", info);
    }

    // Run dpotrf and time
    double MPIt1 = MPI_Wtime();
    printf("[%dx%d] Starting potrf\n", myrow, mycol);
    pdpotrf_(&uplo, &n, A, &ione, &ione, descA, &info);
    if (info != 0) {
        printf("Error in potrf, info = %d\n", info);
    }
    double MPIt2 = MPI_Wtime();
    printf("[%dx%d] Done, time %e s.\n", myrow, mycol, MPIt2 - MPIt1);
    free(A);

    // Exit and finalize
    blacs_gridexit_(&ictxt);
    MPI_Finalize();
    return 0;
}