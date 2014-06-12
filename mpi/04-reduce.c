#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int rank, numprocs, i;
	int array[3], result[3];

	/* Initialise MPI */
	MPI_Init(&argc,&argv);

	/* Number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	/* Current process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Initialise data on all process */
	array[0] = 3*rank;
	array[1] = 3*rank+1;
	array[2] = 3*rank+2;

	/* Print arrays on all processes */
	if (rank==0)
	{
		printf("\nBefore reduce:\n");
	}
	MPI_Barrier(MPI_COMM_WORLD); /* Block till all processes reach this point */
	printf("Process %d: ", rank);
	for (i=0; i<3; i++)
	{
		printf("%d, ", array[i]);
	}
	printf("\n");
	
	MPI_Barrier(MPI_COMM_WORLD);

	/* Reduce operation */
	MPI_Reduce(array, result, 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	/* Print arrays on all processes */
	if (rank==0) printf("\nAfter reduce:\n");
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Process %d: ", rank);
	for (i=0; i<3; i++)
	{
		printf("%d, ", result[i]);
	}
	printf("\n");

	MPI_Finalize();

	return 0;
}
