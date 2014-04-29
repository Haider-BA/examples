#include <petscdmda.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscMPIInt      rank, M, N, m, n;
	PetscInt         nx = 4, ny = 3;
	PetscErrorCode   ierr;
	DM               da;
	Mat              A;
	Vec              p;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);

	// Create distributed array and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate local distributed array [DMDACreate2d, DMCreateLocalVector]"); CHKERRQ(ierr); hline();
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(da, &p); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrint DA Info [DMDACreate2d, DMCreateLocalVector]"); CHKERRQ(ierr); hline();
	ierr = DMDAGetInfo(da, NULL, &M, &N, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "M, N: %d, %d (Total number of nodes in each direction)\n", M, N); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "m, n: %d, %d (Number of processors in each direction)\n", m, n); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix and view [DMCreateMatrix, MatView]"); CHKERRQ(ierr); hline();
	ierr = DMCreateMatrix(da, MATMPIAIJ, &A); CHKERRQ(ierr);
	ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nPrint local indices of matrix rows [MatGetOwnershipRange]"); CHKERRQ(ierr); hline();
	PetscInt start, end;
	ierr = MatGetOwnershipRange(A, &start, &end); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%d\n", start, end); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate a Poisson matrix with purely Neumann BCs [DMDAGetCorners, MatStencil, MatSetValuesStencil]"); CHKERRQ(ierr); hline();
	PetscMPIInt i, j, mstart, nstart, cur;
	MatStencil  row, col[5];
	PetscScalar values[5];
	ierr = DMDAGetCorners(da, &mstart, &nstart, NULL, &m, &n, NULL); CHKERRQ(ierr);
	for(j=nstart; j<nstart+n; j++)
	{
		row.j = j;
		for(i=mstart; i<mstart+m; i++)
		{
			cur = 0;
			row.i = i;
			col[0].i = i;
			col[0].j = j;
			values[0] = 4;
			cur++;

			if(j>0)
			{
				col[cur].i = i;
				col[cur].j = j-1;
				values[cur] = -1;
				cur++;
			}
			else
				values[0]-=1;
	
			if(i>0)
			{
				col[cur].i = i-1;
				col[cur].j = j;
				values[cur] = -1;
				cur++;
			}
			else
				values[0]-=1;
	
			if(i<nx-1)
			{
				col[cur].i = i+1;
				col[cur].j = j;
				values[cur] = -1;
				cur++;
			}
			else
				values[0]-=1;
	
			if(j<ny-1)
			{
				col[cur].i = i;
				col[cur].j = j+1;
				values[cur] = -1;
				cur++;
			}
			else
				values[0]-=1;
			
			if(row.i==0 && row.j==0)
				values[0]+=1;
			
			ierr = MatSetValuesStencil(A, 1, &row, cur, col, values, INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nAssemble the matrix and display [MatAssemblyBegin, MatAssemblyEnd, MatView]"); CHKERRQ(ierr); hline();
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = MatDestroy(&A); CHKERRQ(ierr);
	ierr = VecDestroy(&p); CHKERRQ(ierr);
	ierr = DMDestroy(&da); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
