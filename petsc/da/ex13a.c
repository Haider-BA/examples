#include <petscdmda.h>

int main(int argc,char **argv)
{
	PetscInt         i, j;
	PetscInt         pStart, pEnd, uLocalSize, pLocalSize, localIdx, uStart, uEnd;
	PetscInt         row, col[2];
	PetscInt         nx = 5, ny = 5;
	PetscErrorCode   ierr;
	Vec              pGlobal, uGlobal;
	Mat              G;
	PetscReal        value[2];
	PetscInt         *d_nnz, *o_nnz;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, (nx-1)*ny, &uGlobal); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, nx*ny, &pGlobal); CHKERRQ(ierr);
	
	ierr = VecGetOwnershipRange(uGlobal, &uStart, &uEnd); CHKERRQ(ierr);
	uLocalSize = uEnd-uStart;
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &d_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &o_nnz); CHKERRQ(ierr);
	
	ierr = VecGetOwnershipRange(pGlobal, &pStart, &pEnd); CHKERRQ(ierr);
	pLocalSize = pEnd-pStart;
	
	localIdx=0;
	for(row=uStart; row<uEnd; row++)
	{
		j = row/(nx-1);
		i = row%(nx-1);
		
		col[0] = j*nx+i+1;
		col[1] = j*nx+i;
		
		(col[0]>=pStart && col[0]<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
		(col[1]>=pStart && col[1]<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
		
		ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "row: %d cols: %d %d [%d] %d %d\n", row, col[0], col[1], row, d_nnz[localIdx], o_nnz[localIdx]); CHKERRQ(ierr);
		localIdx++;
	}
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "-\n"); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, pLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 0, d_nnz, 0, o_nnz, &G); CHKERRQ(ierr);
	//ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, pLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 2, NULL, &G); CHKERRQ(ierr);

	for(row=uStart; row<uEnd; row++)
	{
		j = row/(nx-1);
		i = row%(nx-1);
		
		col[0] = j*nx+i+1;
		col[1] = j*nx+i;
		
		value[0] = 1;
		value[1] = -1;
		
		ierr = MatSetValues(G, 1, &row, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatView(G, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = MatDestroy(&G); CHKERRQ(ierr);
	ierr = VecDestroy(&uGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&pGlobal); CHKERRQ(ierr);
	ierr = PetscFree(d_nnz); CHKERRQ(ierr);
	ierr = PetscFree(o_nnz); CHKERRQ(ierr);
	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}
