#include <petscdmda.h>

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
	PetscInt         row, col[2];
	PetscInt         nx = 5, ny = 5, localIdx;
	PetscInt         *d_nnz, *o_nnz;
	PetscErrorCode   ierr;
	DM               uda, pda;
	Vec              pGlobal, uGlobal, bc2;
	AO               uao, pao;
	Mat              G, C;
	PetscScalar      values[2];

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Create matrix G that maps from distributed array P to distributed array U"); CHKERRQ(ierr);    
	hline();

	// Create distributed array and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate staggered U and P vectors [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRQ(ierr);    hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx-1, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(uda, &uGlobal); CHKERRQ(ierr);
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
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix G that maps from P to U [VecGetOwnershipRange, MatCreateAIJ]"); CHKERRQ(ierr);    hline();
	ierr = VecGetOwnershipRange(uGlobal, &start, &end); CHKERRQ(ierr);
	uLocalSize = end-start;
	ierr = VecGetOwnershipRange(pGlobal, &start, &end); CHKERRQ(ierr);
	pLocalSize = end-start;
	
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &d_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &o_nnz); CHKERRQ(ierr);
	
	ierr = DMDAGetAO(uda, &uao); CHKERRQ(ierr);
	ierr = DMDAGetAO(pda, &pao); CHKERRQ(ierr);	
	ierr = DMDAGetCorners(uda, &mstart, &nstart, NULL, &m, &n, NULL);
	localIdx=0;
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			d_nnz[localIdx]=0;
			o_nnz[localIdx]=0;
			
			// row
			row = j*(nx-1)+i;
			ierr = AOApplicationToPetsc(uao, 1, &row); CHKERRQ(ierr);
			
			// columns
			col[0] = j*nx+i+1;
			col[1] = j*nx+i;
			ierr = AOApplicationToPetsc(pao, 2, col); CHKERRQ(ierr);

			(col[0]>=start && col[0]<end)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			(col[1]>=start && col[1]<end)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			
			localIdx++;
		}
	}
	
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, pLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 0, d_nnz, 0, o_nnz, &G); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPopulate G [DMDAGetAO, DMDAGetCorners, AOApplicationToPetsc, MatSetValues, MatAssemblyBegin, MatAssemblyEnd]"); CHKERRQ(ierr);    hline();
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			// row
			row = j*(nx-1)+i;
			ierr = AOApplicationToPetsc(uao, 1, &row); CHKERRQ(ierr);
			
			// columns
			col[0] = j*nx+i+1;
			col[1] = j*nx+i;
			ierr = AOApplicationToPetsc(pao, 2, col); CHKERRQ(ierr);
			
			// values
			values[0] = 1;
			values[1] = -1;
			
			ierr = MatSetValues(G, 1, &row, 2, col, values, INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatView(G, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPopulate P [DMDAGetAO, DMDAGetCorners, AOApplicationToPetsc, VecSetValues, VecAssemblyBegin, VecAssemblyEnd]"); CHKERRQ(ierr);    hline();
	ierr = DMDAGetCorners(pda, &mstart, &nstart, NULL, &m, &n, NULL);
	for(j=nstart; j<nstart+n; j++)
	{
		for(i=mstart; i<mstart+m; i++)
		{
			// row
			row = j*nx+i;
			values[0] = row/10.0;
			ierr = AOApplicationToPetsc(pao, 1, &row); CHKERRQ(ierr);
			ierr = VecSetValue(pGlobal, row, values[0], INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(pGlobal); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pGlobal); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nU = G*P [MatMult]"); CHKERRQ(ierr);    hline();
	ierr = MatMult(G, pGlobal, uGlobal); CHKERRQ(ierr);
	ierr = VecView(uGlobal, PETSC_VIEWER_STDOUT_WORLD);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nC = GT*G [MatTransposeMatMult]"); CHKERRQ(ierr);    hline();
	ierr = MatTransposeMatMult(G, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C); CHKERRQ(ierr);
	ierr = MatView(C, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nU = C*P [MatMult]"); CHKERRQ(ierr);    hline();
	ierr = MatMult(C, pGlobal, bc2); CHKERRQ(ierr);
	ierr = VecView(bc2, PETSC_VIEWER_STDOUT_WORLD);

	ierr = PetscFree(lx); CHKERRQ(ierr);
	ierr = PetscFree(ly); CHKERRQ(ierr);
	ierr = PetscFree(d_nnz); CHKERRQ(ierr);
	ierr = PetscFree(o_nnz); CHKERRQ(ierr);
	ierr = MatDestroy(&C); CHKERRQ(ierr);
	ierr = MatDestroy(&G); CHKERRQ(ierr);
	ierr = VecDestroy(&uGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&pGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&bc2); CHKERRQ(ierr);
	ierr = DMDestroy(&uda); CHKERRQ(ierr);
	ierr = DMDestroy(&pda); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
