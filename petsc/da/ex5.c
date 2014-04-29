#include <petscdmda.h>

int hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRQ(ierr);
	return 0;
}

int main(int argc,char **argv)
{
	PetscMPIInt      rank, M, N, m, n, i;
	PetscInt         nx = 21, ny = 11, *lx, *ly;
	PetscErrorCode   ierr;
	DM               uda, vda, pda;
	Vec              u;
	PetscScalar      value;

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
	ierr = DMDAGetOwnershipRanges(uda, &lx, &ly, NULL); CHKERRQ(ierr);
	for(i=0; i<m; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lx[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	for(i=0; i<n; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", ly[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	
	lx[m-1]++;
	ly[n-1]--;
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nV-Node distributed along each direction [Modified from U's info]"); CHKERRQ(ierr); hline();
	for(i=0; i<m; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", lx[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	for(i=0; i<n; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t", ly[i]); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated and printing DA Info for V [DMDACreate2d, DMDAGetInfo]"); CHKERRQ(ierr); hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny-1, m, n, 1, 1, lx, ly, &vda); CHKERRQ(ierr);
	ierr = DMDAGetInfo(vda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "m, n: %d, %d\n", m, n); CHKERRQ(ierr);
	
	ly[n-1]++;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreated and printing DA Info for P [DMDACreate2d, DMDAGetInfo]"); CHKERRQ(ierr); hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, m, n, 1, 1, lx, ly, &pda); CHKERRQ(ierr);
	ierr = DMDAGetInfo(vda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "m, n: %d, %d\n", m, n); CHKERRQ(ierr);

	ierr = VecDestroy(&u);CHKERRQ(ierr);
	ierr = DMDestroy(&uda);CHKERRQ(ierr);
	ierr = DMDestroy(&vda);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
