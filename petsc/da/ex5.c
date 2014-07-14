#include <petscdmda.h>

PetscErrorCode hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
	return 0;
}

int main(int argc,char **argv)
{
	PetscMPIInt      rank, M, N, m, n, i;
	PetscInt         nx = 21, ny = 11;
	const PetscInt   *lxu, *lyu;
	PetscInt         *lxv, *lyv;
	PetscInt         *lxp, *lyp;
	PetscErrorCode   ierr;
	DM               uda, vda, pda;
	Vec              u;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);

	// Create distributed array and get vectors
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx-1, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(uda, &u); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated local distributed array U [DMDACreate2d, DMCreateLocalVector]"); CHKERRQ(ierr); hline();
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrinting DA Info for U [DMDAGetInfo]"); CHKERRQ(ierr);	hline();
	ierr = DMDAGetInfo(uda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Total size of array : M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of processors: m, n: %d, %d\n", m, n); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nU-Node distribution along each direction [DMDAGetOwnershipRanges]"); CHKERRQ(ierr);	hline();
	ierr = DMDAGetOwnershipRanges(uda, &lxu, &lyu, NULL); CHKERRQ(ierr);
	for(i=0; i<m; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lxu[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	for(i=0; i<n; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lyu[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);

	ierr = PetscMalloc(m*sizeof(*lxv), &lxv); CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(*lyv), &lyv); CHKERRQ(ierr);
	ierr = PetscMemcpy(lxv, lxu, m*sizeof(*lxv)); CHKERRQ(ierr);
	ierr = PetscMemcpy(lyv, lyu, n*sizeof(*lyv)); CHKERRQ(ierr);
	lxv[m-1]++;
	lyv[n-1]--;
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nV-Node distributed along each direction [Modified from U's info]"); CHKERRQ(ierr); hline();
	for(i=0; i<m; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lxv[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	for(i=0; i<n; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lyv[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated and printing DA Info for V [DMDACreate2d, DMDAGetInfo]"); CHKERRQ(ierr); hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny-1, m, n, 1, 1, lxv, lyv, &vda); CHKERRQ(ierr);
	ierr = DMDAGetInfo(vda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "m, n: %d, %d\n", m, n); CHKERRQ(ierr);

	ierr = PetscMalloc(m*sizeof(*lxp), &lxp); CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(*lyp), &lyp); CHKERRQ(ierr);
	ierr = PetscMemcpy(lxp, lxu, m*sizeof(*lxp)); CHKERRQ(ierr);
	ierr = PetscMemcpy(lyp, lyu, n*sizeof(*lyp)); CHKERRQ(ierr);
	lxp[m-1]++;

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated and printing DA Info for P [DMDACreate2d, DMDAGetInfo]"); CHKERRQ(ierr); hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, m, n, 1, 1, lxp, lyp, &pda); CHKERRQ(ierr);
	ierr = DMDAGetInfo(pda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "m, n: %d, %d\n", m, n); CHKERRQ(ierr);

	ierr = PetscFree(lxv); CHKERRQ(ierr);
	ierr = PetscFree(lyv); CHKERRQ(ierr);	
	ierr = PetscFree(lxp); CHKERRQ(ierr);
	ierr = PetscFree(lyp); CHKERRQ(ierr);

	ierr = VecDestroy(&u);CHKERRQ(ierr);
	ierr = DMDestroy(&uda);CHKERRQ(ierr);
	ierr = DMDestroy(&vda);CHKERRQ(ierr);
	ierr = DMDestroy(&pda);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
