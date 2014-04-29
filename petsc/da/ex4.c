#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
}

int main(int argc,char **argv)
{
	PetscMPIInt      rank;
	PetscInt         nx = 5, ny = 5, i, j;
	PetscErrorCode   ierr;
	DM               uda;
	Vec              rx, uGlobal, uLocal, ul;
	PetscScalar      value;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);

	// Create distributed array and get vectors
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(uda, &uGlobal); CHKERRQ(ierr);
	ierr = VecDuplicate(uGlobal, &rx);
	ierr = DMCreateLocalVector(uda, &uLocal); CHKERRQ(ierr);
	ierr = VecDuplicate(uLocal, &ul);
	
	// get the nodes and sizes of the distributed arrays
	PetscScalar  **u;
	PetscInt     mstart, nstart, m, n;
	DMDAVecGetArray(uda, uLocal, &u);
	DMDAGetGhostCorners(uda, &mstart, &nstart, NULL, &m, &n, NULL);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d, %d\t%d, %d\n", mstart, nstart, m, n); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	
	// initialise the non-ghost nodes of the distributed array
	for(j=nstart+1; j<nstart+n-1; j++)
	{
		for(i=mstart+1; i<mstart+m-1; i++)
		{
			u[j][i] = (j*nx+i)/10.0;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] %d\t%d\t%f\n", rank, i, j, u[j][i]); CHKERRQ(ierr);
		}
	}
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	DMDAVecRestoreArray(uda, uLocal, &u);
	
	// view the local vector
	ierr = VecView(uLocal, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// copy the local vector to the global vector
	DMLocalToGlobalBegin(uda, uLocal, INSERT_VALUES, uGlobal);
	DMLocalToGlobalEnd(uda, uLocal, INSERT_VALUES, uGlobal);
	ierr = VecView(uGlobal, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// perform a LocalToLocal copy so that the ghost cells are updated
	// copy to the same vector
	ierr = DMDALocalToLocalBegin(uda, uLocal, INSERT_VALUES, uLocal); CHKERRQ(ierr);
	ierr = DMDALocalToLocalEnd(uda, uLocal, INSERT_VALUES, uLocal); CHKERRQ(ierr);
	ierr = VecView(uLocal, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = VecDestroy(&rx);CHKERRQ(ierr);
	ierr = VecDestroy(&uGlobal);CHKERRQ(ierr);
	ierr = VecDestroy(&uLocal);CHKERRQ(ierr);
	ierr = VecDestroy(&ul);CHKERRQ(ierr);
	ierr = DMDestroy(&uda);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
