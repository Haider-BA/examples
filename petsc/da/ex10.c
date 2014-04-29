#include <petscdmda.h>
#include <petscdmcomposite.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscInt         rank;
	PetscInt         nx = 5, ny = 5;
	PetscReal        dx = 1.0/nx;
	PetscInt         nb = ceil(2*PETSC_PI*0.25/dx);
	PetscErrorCode   ierr;
	DM               pda, bda, pack;
	Vec              pPacked;
	IS               *isGlobal, *isLocal;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing local and global index sets. Running only one 1 process"); CHKERRQ(ierr);    
	hline();
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nnp: %d\nnb: %d\n", nx*ny, nb); CHKERRQ(ierr);

	// Create distributed array uv and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate P and f vectors [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRQ(ierr);    hline();
	
	ierr = DMCompositeCreate(PETSC_COMM_WORLD, &pack); CHKERRQ(ierr);
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &pda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, pda); CHKERRQ(ierr);
	ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, nb, 1, 1, NULL, &bda);
	ierr = DMCompositeAddDM(pack, bda); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(pack, &pPacked); CHKERRQ(ierr);
	
	ierr = DMCompositeGetGlobalISs(pack, &isGlobal); CHKERRQ(ierr);
	ierr = DMCompositeGetLocalISs(pack, &isLocal); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nisGlobal P"); CHKERRQ(ierr);    hline();
	ierr = ISView(isGlobal[0], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nisGlobal f"); CHKERRQ(ierr);    hline();
	ierr = ISView(isGlobal[1], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nisLocal P"); CHKERRQ(ierr);    hline();
	ierr = ISView(isLocal[0], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nisLocal f"); CHKERRQ(ierr);    hline();
	ierr = ISView(isLocal[1], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// Destroy structures
	ierr = ISDestroy(&isGlobal[0]); CHKERRQ(ierr);
	ierr = ISDestroy(&isGlobal[1]); CHKERRQ(ierr);
	ierr = PetscFree(isGlobal); CHKERRQ(ierr);
	ierr = ISDestroy(&isLocal[0]); CHKERRQ(ierr);
	ierr = ISDestroy(&isLocal[1]); CHKERRQ(ierr);
	ierr = PetscFree(isLocal); CHKERRQ(ierr);
	ierr = VecDestroy(&pPacked); CHKERRQ(ierr);
	ierr = DMDestroy(&pda); CHKERRQ(ierr);
	ierr = DMDestroy(&bda); CHKERRQ(ierr);
	ierr = DMDestroy(&pack); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
