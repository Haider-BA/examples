#include <petscdmda.h>

int main(int argc,char **argv)
{
	PetscErrorCode   ierr;
	PetscInt         mstart, nstart, pstart, m, n, p, i, j, k;
	PetscInt         nx = 3, ny = 3, nz=3;
	Vec              uGlobal;
	PetscReal        ***u;
	PetscViewer      uViewer;
	DM               da;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = DMDACreate3d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da, &uGlobal); CHKERRQ(ierr);
	
	ierr = DMDAVecGetArray(da, uGlobal, &u); CHKERRQ(ierr);
	ierr = DMDAGetCorners(da, &mstart, &nstart, &pstart, &m, &n, &p); CHKERRQ(ierr);
	for(k=pstart; k<pstart+p; k++)
	{
		for(j=nstart; j<nstart+n; j++)
		{
			for(i=mstart; i<mstart+m; i++)
			{
				u[k][j][i]  = i+j+k;
			}
		}
	}
	ierr = DMDAVecRestoreArray(da, uGlobal, &u); CHKERRQ(ierr);
	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "U.dat", FILE_MODE_WRITE, &uViewer); CHKERRQ(ierr);
	ierr = VecView(uGlobal, uViewer); CHKERRQ(ierr);
	
	ierr = PetscViewerDestroy(&uViewer); CHKERRQ(ierr);
	ierr = VecDestroy(&uGlobal); CHKERRQ(ierr);
	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}
