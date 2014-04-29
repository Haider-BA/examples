#include <petscdmda.h>
#include <petscdmcomposite.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscInt         i, j;
	PetscInt         rank, m, n, mstart, nstart;
	PetscInt         *lx, *ly;
	const PetscInt   *lxu, *lyu;
	PetscInt         start, end, uLocalSize, pLocalSize;
	PetscInt         row, cols[2];
	PetscInt         nx = 5, ny = 5, localIdx;
	PetscErrorCode   ierr;
	DM               uda, pda, vda, pack;
	Vec              pGlobal, uPacked, bc2;
	AO               pao;
	Mat              G, C;
	PetscScalar      values[2];
	IS               *is;
	const PetscInt   *xindices, *yindices;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Create matrix G that maps from distributed array P to DMComposite of distributed arrays U and V"); CHKERRQ(ierr);    
	hline();

	// Create distributed array and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate staggered U, V and P vectors [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRQ(ierr);    hline();
	ierr = DMCompositeCreate(PETSC_COMM_WORLD, &pack); CHKERRQ(ierr);
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx-1, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, uda); CHKERRQ(ierr);
	ierr = DMDAGetInfo(uda, NULL, NULL, NULL, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = DMDAGetOwnershipRanges(uda, &lxu, &lyu, NULL); CHKERRQ(ierr);
	ierr = PetscMalloc(m*sizeof(*lx), &lx); CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(*ly), &ly); CHKERRQ(ierr);
	ierr = PetscMemcpy(lx ,lxu, m*sizeof(*lx)); CHKERRQ(ierr);
	ierr = PetscMemcpy(ly ,lyu, n*sizeof(*ly)); CHKERRQ(ierr);
	lx[m-1]++;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, m, n, 1, 1, lx, ly, &pda); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(pda, &pGlobal); CHKERRQ(ierr);
	ierr = VecDuplicate(pGlobal, &bc2); CHKERRQ(ierr);
	ly[n-1]--;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny-1, m, n, 1, 1, lx, ly, &vda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, vda); CHKERRQ(ierr);
	PetscFree(lx);
	PetscFree(ly);
	ierr = DMCreateGlobalVector(pack, &uPacked); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPopulate P [DMDAGetAO, DMDAGetCorners, AOApplicationToPetsc, VecSetValues, VecAssemblyBegin, VecAssemblyEnd]"); CHKERRQ(ierr);    hline();
	ierr = DMDAGetAO(pda, &pao); CHKERRQ(ierr);
	ierr = DMDAGetCorners(pda, &mstart, &nstart, NULL, &m, &n, NULL);
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			row = j*nx+i;
			values[0] = row/10.0;
			ierr = AOApplicationToPetsc(pao, 1, &row); CHKERRQ(ierr);
			ierr = VecSetValue(pGlobal, row, values[0], INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(pGlobal); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pGlobal); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix G that maps from P to U [VecGetOwnershipRange, MatCreateAIJ]"); CHKERRQ(ierr);    hline();
	ierr = VecGetOwnershipRange(uPacked, &start, &end); CHKERRQ(ierr);
	uLocalSize = end-start;
	ierr = VecGetOwnershipRange(pGlobal, &start, &end); CHKERRQ(ierr);
	pLocalSize = end-start;
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, pLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 1, NULL, &G); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPopulate G [DMCompositeGetGlobalISs, ISGetIndices, DMDAGetCorners, AOApplicationToPetsc, MatSetValues, MatAssemblyBegin, MatAssemblyEnd]"); CHKERRQ(ierr);    hline();
	ierr = DMCompositeGetGlobalISs(pack, &is); CHKERRQ(ierr);
	ierr = ISGetIndices(is[0], &xindices); CHKERRQ(ierr);
	ierr = DMDAGetCorners(uda, &mstart, &nstart, NULL, &m, &n, NULL);
	localIdx = 0;
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			row = xindices[localIdx]; // uses the indices obtained using DMCompositeGetGlobalISs and ISGetIndices

			cols[0] = j*nx+i+1; // i and j are obtained from DMDAGetCorners
			cols[1] = j*nx+i;
			ierr = AOApplicationToPetsc(pao, 2, cols); CHKERRQ(ierr); // maps the natural ordering to petsc ordering using the AO obtained from DMDAGetAO
			
			values[0] = 1;
			values[1] = -1;
			
			ierr = MatSetValues(G, 1, &row, 2, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			localIdx++;
		}
	}
	ierr = ISRestoreIndices(is[0], &xindices); CHKERRQ(ierr);
	ierr = ISGetIndices(is[1], &yindices); CHKERRQ(ierr);
	ierr = DMDAGetCorners(vda, &mstart, &nstart, NULL, &m, &n, NULL);
	localIdx = 0;
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			row = yindices[localIdx];
			
			cols[0] = (j+1)*nx+i;
			cols[1] = j*nx+i;
			ierr = AOApplicationToPetsc(pao, 2, cols); CHKERRQ(ierr);
			
			values[0] = 1;
			values[1] = -1;
			
			ierr = MatSetValues(G, 1, &row, 2, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			localIdx++;
		}
	}
	ierr = ISRestoreIndices(is[1], &yindices); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatView(G, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nU = G*P [MatMult]"); CHKERRQ(ierr); hline();
	ierr = MatMult(G, pGlobal, uPacked); CHKERRQ(ierr);
	ierr = VecView(uPacked, PETSC_VIEWER_STDOUT_WORLD);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nC = GT*G [MatTransposeMatMult]"); CHKERRQ(ierr); hline();
	ierr = MatTransposeMatMult(G, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C); CHKERRQ(ierr);
	ierr = MatView(C, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nbc2 = C*P [MatMult]"); CHKERRQ(ierr);    hline();
	ierr = MatMult(C, pGlobal, bc2); CHKERRQ(ierr);
	ierr = VecView(bc2, PETSC_VIEWER_STDOUT_WORLD);
	
	ierr = MatDestroy(&C); CHKERRQ(ierr);	
	ierr = MatDestroy(&G); CHKERRQ(ierr);
	ierr = ISDestroy(&is[0]); CHKERRQ(ierr);
	ierr = ISDestroy(&is[1]); CHKERRQ(ierr);
	ierr = PetscFree(is); CHKERRQ(ierr);
	ierr = VecDestroy(&bc2); CHKERRQ(ierr);
	ierr = VecDestroy(&uPacked); CHKERRQ(ierr);
	ierr = VecDestroy(&pGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&bc2); CHKERRQ(ierr);
	ierr = DMDestroy(&vda); CHKERRQ(ierr);
	ierr = DMDestroy(&uda); CHKERRQ(ierr);
	ierr = DMDestroy(&pda); CHKERRQ(ierr);
	ierr = DMDestroy(&pack); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
