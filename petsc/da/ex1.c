#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
}

int main(int argc,char **argv)
{
	PetscMPIInt      rank;
	PetscInt         M = 5, N = 5;
	PetscErrorCode   ierr;
	PetscBool        flg = PETSC_FALSE;
	DM               da;
	Vec              local,global;
	PetscScalar      value;
	DMDAStencilType  stype = DMDA_STENCIL_BOX;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-M", &M, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-N", &N, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetBool(NULL,"-star_stencil", &flg, NULL);CHKERRQ(ierr);
	if (flg) stype = DMDA_STENCIL_STAR;

	// Create distributed array and get vectors
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, stype, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da, &global); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated global 2D Distributed Array [DMCreateGlobalVector]"); CHKERRQ(ierr);
	hline();
	ierr = DMCreateLocalVector(da, &local); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated local version of the global array [DMCreateLocalVector]"); CHKERRQ(ierr);
	hline();
	
	PetscInt nLocal, *idx;
	ierr = DMDAGetGlobalIndices(da, &nLocal, &idx); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodes on each process (includes ghost nodes) [DMDAGetGlobalIndices]"); CHKERRQ(ierr);
	hline();
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] numLocal = %d;\n", rank, nLocal); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nDistributed Array properties [DMView]"); CHKERRQ(ierr);
	hline();
	ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// set the values of the global matrix with the same value
	value = -3.0;
	ierr  = VecSet(global,value);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSet all elements of the global vector with the same value [VecSet, VecView]"); CHKERRQ(ierr);
	hline();
	ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	PetscInt gLow, gHigh, lLow, lHigh;
	ierr = VecGetOwnershipRange(global, &gLow, &gHigh);
	ierr = VecGetOwnershipRange(global, &lLow, &lHigh);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nRanges of the local and global vectors on Process 0 [VecGetOwnershipRange]"); CHKERRQ(ierr);
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "globalLow : %d\nglobalHigh: %d\nlocalLow : %d\nlocalHigh: %d\n", gLow, gHigh, lLow, lHigh); CHKERRQ(ierr);
	
	PetscInt I;
	for(I = gLow; I<gHigh; I++)
	{
		value = I/10.0;
		VecSetValues(global, 1, &I, &value, INSERT_VALUES);
	}	
	VecAssemblyBegin(global);
	VecAssemblyEnd(global);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSet elements of the global vector with different values [VecSetValues, VecAssemblyBegin and VecAssemblyEnd, VecView]"); CHKERRQ(ierr);
	hline();
	ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	// Free memory
	ierr = VecDestroy(&local);CHKERRQ(ierr);
	ierr = VecDestroy(&global);CHKERRQ(ierr);
	ierr = DMDestroy(&da);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
