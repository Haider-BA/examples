#include <petscdmda.h>
#include <petscdmcomposite.h>

int main(int argc, char **argv)
{
	PetscErrorCode         ierr;
	PetscInt               nx = 4, ny = 4;
	Vec                    q;
	DM                     uda, vda, pack;
	const PetscInt         *lxu, *lyu;
	PetscInt               *lxv, *lyv;
	PetscInt               numX, numY;
	DMDABoundaryType       bx, by;
	PetscInt               mstart, nstart, m, n, i, j;
	PetscInt               qStart, qEnd, qLocalSize;
	PetscInt               cols[5], localCols[5];
	PetscInt               localIdx;
	ISLocalToGlobalMapping *ltogs;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	// set boundary types for velocity variables
	bx = DMDA_BOUNDARY_GHOSTED;
	by = DMDA_BOUNDARY_GHOSTED;
	
	// Create distributed array data structures
	// x-velocity
	numX = nx-1;
	numY = ny;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, bx, by, DMDA_STENCIL_BOX, numX, numY, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCompositeCreate(PETSC_COMM_WORLD, &(pack)); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, uda); CHKERRQ(ierr);
	ierr = DMDAGetOwnershipRanges(uda, &lxu, &lyu, NULL); CHKERRQ(ierr);
	ierr = DMDAGetInfo(uda, NULL, NULL, NULL, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	
	// y-velocity
	ierr = PetscMalloc(m*sizeof(*lxv), &lxv); CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(*lyv), &lyv); CHKERRQ(ierr);
	ierr = PetscMemcpy(lxv, lxu, m*sizeof(*lxv)); CHKERRQ(ierr);
	ierr = PetscMemcpy(lyv, lyu, n*sizeof(*lyv)); CHKERRQ(ierr);
	lyv[n-1]--;
	lxv[m-1]++;
	numX = nx;
	numY = ny-1;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, bx, by, DMDA_STENCIL_BOX, numX, numY, m, n, 1, 1, lxv, lyv, &(vda)); CHKERRQ(ierr);
	PetscFree(lxv);
	PetscFree(lyv);
	ierr = DMCompositeAddDM(pack, vda); CHKERRQ(ierr);

	// create vector
	ierr = DMCreateGlobalVector(pack, &q); CHKERRQ(ierr);

	// map local sub-DM (including ghost) indices to packed global indices
	ierr = DMCompositeGetISLocalToGlobalMappings(pack, &ltogs); CHKERRQ(ierr);

	// ownership range of vector
	ierr = VecGetOwnershipRange(q, &qStart, &qEnd); CHKERRQ(ierr);
	qLocalSize = qEnd-qStart;

	// print global indices of the vector elements
	ierr = DMDAGetCorners(uda, &mstart, &nstart, NULL, &m, &n, NULL); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Number of local U: %d\n", m*n); CHKERRQ(ierr);
	for(j=0; j<n; j++)
	{
		for(i=0; i<m; i++)
		{
			localIdx = (j+1)*(m+2) + (i+1);

			localCols[0] = localIdx-(m+2);
			localCols[1] = localIdx-1;
			localCols[2] = localIdx;
			localCols[3] = localIdx+1;
			localCols[4] = localIdx+(m+2);

			ierr = ISLocalToGlobalMappingApply(ltogs[0], 5, localCols, cols); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d |%3d:%3d |%3d:%3d |%3d:%3d |%3d:%3d |%3d:%3d\n", cols[2], localCols[0], cols[0], localCols[1], cols[1], localCols[2], cols[2], localCols[3], cols[3], localCols[4], cols[4]);
		}
	}

	ierr = DMDAGetCorners(vda, &mstart, &nstart, NULL, &m, &n, NULL); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Number of local V: %d\n", m*n); CHKERRQ(ierr);
	for(j=0; j<n; j++)
	{
		for(i=0; i<m; i++)
		{
			localIdx = (j+1)*(m+2) + (i+1);

			localCols[0] = localIdx-(m+2);
			localCols[1] = localIdx-1;
			localCols[2] = localIdx;
			localCols[3] = localIdx+1;
			localCols[4] = localIdx+(m+2);

			ierr = ISLocalToGlobalMappingApply(ltogs[1], 5, localCols, cols); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d |%3d:%3d | %3d:%3d |%3d:%3d |%3d:%3d |%3d:%3d\n", cols[2], localCols[0], cols[0], localCols[1], cols[1], localCols[2], cols[2], localCols[3], cols[3], localCols[4], cols[4]);
		}
	}
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Total U and V    : %d\n\n", qLocalSize); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);

	ierr = ISLocalToGlobalMappingDestroy(&ltogs[0]); CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingDestroy(&ltogs[1]); CHKERRQ(ierr);
	ierr = PetscFree(ltogs); CHKERRQ(ierr);

	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}