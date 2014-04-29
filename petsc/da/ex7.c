#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscInt         M, N, i, j, I;
	PetscInt         rank, m, n, mstart, nstart, *lx, *ly;
	PetscInt         nx = 5, ny = 5;
	PetscErrorCode   ierr;
	DM               uda, pda;
	Vec              pGlobal, pLocal, uLocal;
	AO               pao;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);

	// Create distributed array and get vectors
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx-1, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(uda, &uLocal); CHKERRQ(ierr);
	ierr = DMDAGetInfo(uda, NULL, NULL, NULL, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = DMDAGetOwnershipRanges(uda, &lx, &ly, NULL); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nU-Node distribution along each direction [DMDAGetOwnershipRanges]"); CHKERRQ(ierr); hline();
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
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, m, n, 1, 1, lx, ly, &pda); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(pda, &pLocal); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(pda, &pGlobal); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrinting DA Info for P [DMDAGetInfo]"); CHKERRQ(ierr);  hline();
	ierr = DMDAGetInfo(pda, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Total size of array : M, N: %d, %d\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of processors: m, n: %d, %d\n", m, n); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nP-Node distribution along each direction [DMDAGetOwnershipRanges]"); CHKERRQ(ierr); hline();
	ierr = DMDAGetOwnershipRanges(pda, &lx, &ly, NULL); CHKERRQ(ierr);
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
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrinting global indices that correspond to local numbering for P [DMDAGetAO, VecGetOwnershipRange, AOPetscToApplication]"); CHKERRQ(ierr);    hline();
	PetscInt         start, end;
	ierr = DMDAGetAO(pda, &pao); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(pLocal, &start, &end); CHKERRQ(ierr);
	for(i=start; i<end; i++)
	{
		j = i;
		ierr = AOPetscToApplication(pao, 1 , &j); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d %d\n", i, j); CHKERRQ(ierr);
	}
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSame as above [DMGetCorners]"); CHKERRQ(ierr);    hline();
	ierr = DMDAGetCorners(pda, &mstart, &nstart, NULL, &m, &n, NULL);
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			I = j*nx+i;
			ierr = PetscPrintf(PETSC_COMM_WORLD, "%d ", I); CHKERRQ(ierr);
			ierr = AOPetscToApplication(pao, 1 , &I); CHKERRQ(ierr);
			ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\n", I); CHKERRQ(ierr);
		}
	}
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrinting global indices [DMDAGetGlobalIndices]"); CHKERRQ(ierr);    hline();
	PetscInt         numLocal, *indices;
	ierr = DMDAGetGlobalIndices(pda, &numLocal, &indices);
	for(i=0; i<numLocal; i++)
	{
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%d %d\n", i, indices[i]); CHKERRQ(ierr);
	}

	ierr = VecDestroy(&uLocal); CHKERRQ(ierr);
	ierr = VecDestroy(&pLocal); CHKERRQ(ierr);
	ierr = VecDestroy(&pGlobal); CHKERRQ(ierr);
	ierr = DMDestroy(&uda); CHKERRQ(ierr);
	ierr = DMDestroy(&pda); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
