#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void cholesky_paralelo(double *A, int n, int id_proceso, int cantidad_procesos);
void mostrar_matriz(double *A, int n); 


int main(int argc, char **argv) {
    
    int id_proceso, cantidad_procesos, root = 0;
   

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proceso);
    MPI_Comm_size(MPI_COMM_WORLD, &cantidad_procesos);

    if (argc < 3) { // condicional para informar como es el uso correcto para la ejecucion del programa
        if (id_proceso == root)
        {
            printf("Uso: mpirun -n <P> %s <N> <archivo.bin>\n", argv[0]);
        } 
        MPI_Finalize();
        return 1;
    }

    //tratamos los parametros de entrada casteando la dimension para que sea un entero
    int n = atoi(argv[1]);
    char *nombre_archivo = argv[2];
    int total_elementos = n * n;
    
    
    double *A = (double *)calloc(total_elementos, sizeof(double));

    int elementos_base = total_elementos / cantidad_procesos;
    int elementos_sobrantes = total_elementos % cantidad_procesos; //elementos que sobran de dividir equitativamente los elementos de la matriz

    int mi_conteo = elementos_base + (id_proceso < elementos_sobrantes ? 1 : 0); //calculo cuantos son los elementos que le corresponde a cada proceso
    long largo_offset = id_proceso * elementos_base + (id_proceso < elementos_sobrantes ? id_proceso : elementos_sobrantes); //se calcula el desplazamiento que le corresponde a cada proceso, esto es donde comienza a leer la matriz dicho proceso
    MPI_Offset mi_offset = largo_offset * sizeof(double); //se traduce el offset en un tipo MPI_Offset para poder usarlo posteriormente con MPI_File_read_at_all
    

    
    MPI_File archivo_binario; //se declara una variable tipo archivo mpi

    //se abre dicho archivo de nombre "nombre_archivo" variable que se pasa por parametro cuando se ejecuta. "archivo_binario" es el archivo binario el cual sera mapeado a la matriz A
    MPI_File_open(MPI_COMM_WORLD, nombre_archivo, MPI_MODE_RDONLY, MPI_INFO_NULL, &archivo_binario);
    
    // se realiza la lectura de todo el documento pero cada proceso lee solamente en su offset correspondiente previamente calculado
    MPI_File_read_at_all(archivo_binario, mi_offset, &A[largo_offset], mi_conteo, MPI_DOUBLE, MPI_STATUS_IGNORE);


    MPI_File_close(&archivo_binario);

    
    int *conteos_recepcion = (int *)malloc(cantidad_procesos * sizeof(int));
    int *desplazamientos = (int *)malloc(cantidad_procesos * sizeof(int));
    
    /*
    Aqui se hace una reconstruccion en un arreglo (ptr)
    para "conteos_recepcion" lo que se hace es iterar  (todos los procesos lo hacen y tienen que tener el mismo valor para usar la posterior funcion mpi) sobre la cantidad de procesos e ir guardando cuantos elementos tienen cada uno. 
    Ppara "desplazamientos" lo que hace es cargar cuantos saltos tienen que dar la funcion mpi para guardar en la posicion correcta de la matriz A los elementos que tiene el proceso correspondiente al indice i
    */
    for (int i = 0; i < cantidad_procesos; i++) {
        conteos_recepcion[i] = elementos_base + (i < elementos_sobrantes ? 1 : 0);
        desplazamientos[i] = i * elementos_base + (i < elementos_sobrantes ? i : elementos_sobrantes);
    }

    /*
    esta funcion lo que hace es reunir en toda la informacion 
    de la memoria de cada proceso y colocarla en el recive buffer. Entonces
    lo que tenemos es que cada proceso comparte la porcion de la matriz que tiene 
    leida con sus correspondientes desplazamiento y cantidad de elementos que va enviar
    Synopsis
        int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                        void *recvbuf, const int *recvcounts, const int *displs,
                        MPI_Datatype recvtype, MPI_Comm comm)
        Input Parameters
            sendbuf: starting address of send buffer (choice)
            sendcount: number of elements in send buffer (integer)
            sendtype: data type of send buffer elements (handle)
            recvcounts: integer array (of length group size) containing the number of elements that are to be received from each process
            displs: integer array (of length group size). Entry i specifies the displacement (relative to recvbuf ) at which to place the incoming data from process i
            recvtype: data type of receive buffer elements (handle)
            comm: communicator (handle)

    Entonces esta funcion les dice a todos los procesos que difundan por la red 
    la cantidad de elementos y el desplazamiento relativo al buffer de recibimiento
    los cuales son cargados en el mismo buffer de recibimiento que en este caso es la matriz A
    */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A, conteos_recepcion, desplazamientos, MPI_DOUBLE, MPI_COMM_WORLD);

    free(conteos_recepcion);
    free(desplazamientos);

    MPI_Barrier(MPI_COMM_WORLD);
    double tiempo_inicio = MPI_Wtime(); 

    cholesky_paralelo(A, n, id_proceso, cantidad_procesos);

    MPI_Barrier(MPI_COMM_WORLD);
    double tiempo_fin = MPI_Wtime(); 

    if(id_proceso == root) {
        printf("%d,%d,%f\n", n, cantidad_procesos, tiempo_fin - tiempo_inicio);
    }
   
    free(A);
    MPI_Finalize();
    return 0;
}


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
            //elemento de la diagonal de la columna k
            buffer[k] = (A[k*n+k] = sqrt(A[k*n + k])); //asignacion en cadena a = (b = c) primero b = c y luego lo que tiene b se asigna a a

            //resto de elementos debajo de la diagonal 
            for (int i=k+1; i < n; i++)
            {
                buffer[i] = A[i*n + k] = (double) A[i*n + k] / buffer[k];
            }        
        }

        /*
            En el caso de que la columna que ese esta analizando no le corresponda al proceso con id id_proceso
            se esperara que toque su ejecucion y termine sus calculos correspondiente para que le llegue el mensaje de la columna y asi poder avanzar
        */

        MPI_Bcast(buffer, n, MPI_DOUBLE, id_duenio, MPI_COMM_WORLD);

        /*
            Hago la copia de la columna ya final dentro de la memoria de cada proceso, asi logramos mantener la matriz L actualizada en todos los procesos
            me ahorro de usar MPI_Gather().
            Esto lo hacemos por que sino cada proceso solo tendra las columnas que les corresponde con las restas de las sumatorias aplicadas y cada proceso tendria una version
            erronea de la matriz L.
        */
        for (int i=k; i < n; i++)
        {
            A[i*n + k] = buffer[i];
        }

        int columna_comienzo = id_proceso; //esta variable nos permite saber en que columna comenzar a realizar el calculo de los demas elementos de la matriz L
        while(columna_comienzo <= k)
        {
            columna_comienzo += cantidad_procesos; // vamos saltando las columnas 
        }

        /*
            En esta parte del metodo se realizan las restas de las sumas correspondiente a los elementos de la izquierda de la columna actual que se esta calculando
            es decir, aqui se hace una resta de un termino de la sumatoria de la columna que se calculo en el buffer, de esta forma
            obtenemos una actualizacion constante en cada iteracion k y en cada proceso mantenemos la matriz lo mas actualizada posible
            dejando asi solamente, cuando corresponda, el calculo del elemento de la diagonal principal sacando su raiz cuadrada (porque la resta de la sumatoria
            ya se realiza en este for de abajo) o diviendo el resto de los elementos de la columna por el de la diagonal principal correspondiente.
        */

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