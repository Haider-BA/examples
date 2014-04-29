#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int rank, numprocs, i;
	int *array;

	/* Initialise MPI */
	MPI_Init(&argc,&argv);

	/* Number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	/* Current process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Allocate memory and set array values on all processes to zero */
	array = (int*)malloc(10*sizeof(int));
	for (i=0; i<10; i++)
	{
		array[i] = 0;
	}

	/* Initialise data on Process 0 */
	if (rank==0)
	{
		for (i=0; i<10; i++)
		{
			array[i] = i;
		}
	}

	/* Print arrays on all processes */
	if (rank==0)
	{
		printf("\nBefore broadcast:\n");
	}
	MPI_Barrier(MPI_COMM_WORLD); /* Block till all processes reach this point */
	printf("Process %d: ", rank);
	for (i=0; i<10; i++)
	{
		printf("%d, ", array[i]);
	}
	printf("\n");
	
	MPI_Barrier(MPI_COMM_WORLD);

	/* Broadcast data to all processes */
	MPI_Bcast(array, 10, MPI_INT, 0, MPI_COMM_WORLD);

	/* Print arrays on all processes */
	if (rank==0) printf("\nAfter broadcast:\n");
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Process %d: ", rank);
	for (i=0; i<10; i++)
	{
		printf("%d, ", array[i]);
	}
	printf("\n");

	free(array);

	MPI_Finalize();

	return 0;
}
