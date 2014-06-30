#include <petscdmda.h>
#include <petscdmcomposite.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscErrorCode   ierr;
	PetscInt         rank, size;
	PetscInt         *lx, N;
	DM               da;
	Vec              vec;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
	
	ierr = PetscMalloc(size*sizeof(*lx), &lx); CHKERRQ(ierr);
	
	N = 0;
	for(int i=0; i<size; i++)
	{
		if(i%2==0)
			lx[i] = 0;
		else
			lx[i] = 2
			*i;
		N += lx[i];
	}
	
	ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, N, 1, 0, lx, &da); CHKERRQ(ierr);
	
	ierr = DMCreateGlobalVector(da, &vec); CHKERRQ(ierr);
	
	ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// Destroy structures
	ierr = DMDestroy(&da); CHKERRQ(ierr);
	ierr = VecDestroy(&vec); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
