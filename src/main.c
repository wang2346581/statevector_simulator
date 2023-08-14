#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "init.h"
#include "common.h"
#include "gate.h"
#include <mpi.h>

int main(int argc, char *argv[]) {
    char *ini, *cir;
    int ret = read_args(argc, argv, &ini, &cir);
    if(ret == 2)
        printf("[ini]: %s, [cir]: %s\n", ini, cir);
    else{
        printf("Error \n");
        return 0;
    }
    int provided;

    if(MPI_Init_thread(NULL, NULL,MPI_THREAD_MULTIPLE,&provided)!=MPI_SUCCESS)
    {
        exit(-1);
    }
    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if(MPI_Comm_rank(MPI_COMM_WORLD, &rank)!=MPI_SUCCESS)
    {
        exit(-1);
    }
    set_all(ini, cir);
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);
    omp_set_num_threads(num_thread);
    MEASURET_START;
    run_simulator();
    MEASURET_END;
    if(MPI_Finalize()!=MPI_SUCCESS)
    {
        exit(-1);
    }
    return 0;
}
