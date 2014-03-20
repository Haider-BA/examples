#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
	int   rank,
	      numprocs,
	      len;
	char  hostname[MPI_MAX_PROCESSOR_NAME];
	
	/* Initialise MPI */
	MPI_Init(&argc, &argv);
	
	/* Number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	/* The id of the current process */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* get the name of the current process */
	MPI_Get_processor_name(hostname, &len);
	
	printf("Hello world from process %d out of %d on %s!\n", rank, numprocs, hostname);
	
	/* finish using MPI */
	MPI_Finalize();
	
	return 0;
}
