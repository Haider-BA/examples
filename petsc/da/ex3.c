#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
}

int main(int argc,char **argv)
{
	PetscInt         M = 5, N = 5, i, j;
	PetscErrorCode   ierr;
	PetscBool        flg = PETSC_FALSE;
	DM               da;
	Vec              local,global;
	PetscScalar      value;
	DMDAStencilType  stype = DMDA_STENCIL_BOX;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-M", &M, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-N", &N, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetBool(NULL,"-star_stencil", &flg, NULL);CHKERRQ(ierr);
	if (flg) stype = DMDA_STENCIL_STAR;

	/* Create distributed array and get vectors */
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC, stype, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	
	ierr = DMCreateGlobalVector(da, &global); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(da, &local); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated global and local arrays [DMCreateLocalVector, DMCreateLocalVector]"); CHKERRQ(ierr);
	hline();
	
	PetscInt low, high, I;

	ierr = VecGetOwnershipRange(global, &low, &high);
	for(I = low; I<high; I++)
	{
		value = I/10.0;
		VecSetValues(global, 1, &I, &value, INSERT_VALUES);
	}
	ierr = VecAssemblyBegin(global); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(global); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nInitialised the global vector [VecGetOwnershipRange, VecSetValues, VecAssemblyBegin, VecAssemblyEnd]"); CHKERRQ(ierr);
	hline();
	
	// copy from global to local
	ierr  = DMGlobalToLocalBegin(da, global, INSERT_VALUES, local); CHKERRQ(ierr);
	ierr  = DMGlobalToLocalEnd(da, global, INSERT_VALUES, local); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCopied from global to local vector [DMGlobalToLocalBegin, DMGlobalToLocalEnd]"); CHKERRQ(ierr);
	hline();
	
	PetscScalar  **u;
	PetscInt     mstart, nstart, m, n;
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nLocations and sizes of the global vector's node blocks on each processor [DMDAVecGetArray, DMDAGetCorners]"); CHKERRQ(ierr);
	hline();
	DMDAVecGetArray(da, global, &u);
	DMDAGetCorners(da, &mstart, &nstart, NULL, &m, &n, NULL);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d, %d\t\t%d, %d\n", mstart, nstart, m, n);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nGlobal vector elements on Process 0 [DMDARestoreArray]"); CHKERRQ(ierr);
	hline();
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=nstart; i<mstart+m; i++)
		{
			ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%d\t%f\n", i, j, u[j][i]); CHKERRQ(ierr);
		}
	}
	DMDAVecRestoreArray(da, global, &u);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nTrying to access ghost nodes on the global vector leads to a segmentation fault"); CHKERRQ(ierr);
	hline();
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nLocal vector elements with ghost nodes on Process 0 [DMDAGetGhostCorners]"); CHKERRQ(ierr);
	hline();
	DMDAVecGetArray(da, local, &u);
	DMDAGetGhostCorners(da, &mstart, &nstart, NULL, &m, &n, NULL);
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%d\t%f\n", i, j, u[j][i]); CHKERRQ(ierr);
		}
	}
	//ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	DMDAVecRestoreArray(da, local, &u);
	
	//ierr = PetscPrintf(PETSC_COMM_WORLD, "\nLocal vector after copying [DMGlobalToLocalBegin, DMGlobalToLocalEnd, VecView]"); CHKERRQ(ierr);
	//hline();
	//ierr = VecView(local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	// Free memory
	ierr = VecDestroy(&local);CHKERRQ(ierr);
	ierr = VecDestroy(&global);CHKERRQ(ierr);
	ierr = DMDestroy(&da);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
