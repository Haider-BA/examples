#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
}

int main(int argc,char **argv)
{
	//PetscMPIInt      rank;
	PetscInt         M = 5, N = 5;
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

	// Create distributed array and get vectors
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, stype, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	
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
	VecAssemblyBegin(global);
	VecAssemblyEnd(global);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nInitialise and view the global vector [VecGetOwnershipRange, VecSetValues, VecAssemblyBegin, VecAssemblyEnd, VecView]"); CHKERRQ(ierr);
	hline();
	ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nThe local vector is unaffected and consists of zeros."); CHKERRQ(ierr);
	hline();
 
	// copy from global to local
	ierr  = DMGlobalToLocalBegin(da, global, INSERT_VALUES, local); CHKERRQ(ierr);
	ierr  = DMGlobalToLocalEnd(da, global, INSERT_VALUES, local); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nLocal vector after copying [DMGlobalToLocalBegin, DMGlobalToLocalEnd, VecView]"); CHKERRQ(ierr);
	hline();
	ierr = VecView(local, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	// Free memory
	ierr = VecDestroy(&local);CHKERRQ(ierr);
	ierr = VecDestroy(&global);CHKERRQ(ierr);
	ierr = DMDestroy(&da);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
