#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
	int         numtasks, rank, dest, source, count;
	char        inmsg[20], outmsg[20];
	MPI_Status  Stat;

	/* Initialise MPI */
	MPI_Init(&argc,&argv);

	/* Number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	/* Current process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) /* Processor 0 */
	{
		strcpy(outmsg, "We're number one.");

		/* Instruction with tag 0 - Send string to process 1 */
		dest = 1;
		MPI_Send(&outmsg, 18, MPI_CHAR, dest, 0, MPI_COMM_WORLD);

		/* Instruction with tag 1 - Receive string from process 1 */
		source = 1;
		MPI_Recv(&inmsg, 15, MPI_CHAR, source, 1, MPI_COMM_WORLD, &Stat);
	}
	else if(rank == 1) /* Processor 1 */
	{
		strcpy(outmsg, "We try harder.");

		/* Instruction with tag 0 - Receive string from process 0 */
		source = 0;
		MPI_Recv(&inmsg, 18, MPI_CHAR, source, 0, MPI_COMM_WORLD, &Stat);

		/* Instruction with tag 1 - Send string to process 0 */
		dest = 0;
		MPI_Send(&outmsg, 15, MPI_CHAR, dest, 1, MPI_COMM_WORLD);
	}

	MPI_Get_count(&Stat, MPI_CHAR, &count);
	printf("Process %d received: \"%s\"  from Process %d with Tag %d \n", rank, inmsg, Stat.MPI_SOURCE, Stat.MPI_TAG);

	MPI_Finalize();

	return 0;
}
