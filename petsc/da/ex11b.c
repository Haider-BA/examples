#include <petscdmda.h>
#include <petscdmcomposite.h>

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

PetscReal dhRoma(PetscReal x, PetscReal h)
{
	PetscReal r = fabs(x)/h;
	if(r>1.5)
		return 0.0;
	else if(r>0.5 && r<=1.5)
		return 1.0/(6*h)*( 5.0 - 3.0*r - sqrt(-3.0*(1-r)*(1-r) + 1.0) );
	else
		return 1.0/(3*h)*( 1.0 + sqrt(-3.0*r*r + 1.0) );
}

PetscReal delta(PetscReal x, PetscReal y, PetscReal h)
{
	return dhRoma(x, h) * dhRoma(y, h);
}

int main(int argc,char **argv)
{
	PetscInt         i, j, k;
	PetscInt         rank, um, un, vm, vn, m, n, umstart, unstart, vmstart, vnstart;
	PetscInt         *lx, *ly, *I, *J;
	const PetscInt   *lxu, *lyu;
	PetscInt         start, end, uLocalSize, fLocalSize;
	PetscInt         row, col;
	PetscInt         nx = 20, ny = 20, localIdx;
	PetscReal        dx = 1.0/nx, dy = 1.0/ny;
	PetscInt         nb = ceil(2*PETSC_PI*0.25/dx);
	PetscErrorCode   ierr;
	DM               uda, vda, pack;
	Vec              uPacked, f;
	Mat              ET, C;
	PetscReal        value, *bx, *by, x, y;
	IS               *is;
	const PetscInt   *uindices, *vindices;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Create matrix ET that maps from a vector f to a DMComposite of distributed arrays U and V"); CHKERRQ(ierr);    
	hline();

	// Create distributed array uv and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate staggered & packed U, V vector [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRQ(ierr);    hline();
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
	ly[n-1]--;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny-1, m, n, 1, 1, lx, ly, &vda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, vda); CHKERRQ(ierr);
	PetscFree(lx);
	PetscFree(ly);
	ierr = DMCreateGlobalVector(pack, &uPacked); CHKERRQ(ierr);
	ierr = VecSet(uPacked, dy); CHKERRQ(ierr);
	
	// Create f vector
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate an arbitrary vector f of size 2*nb [VecCreateMPI]"); CHKERRQ(ierr);    hline();
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 2*nb, &f); CHKERRQ(ierr);
	
	// Create ET
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix ET that maps from f to U [VecGetOwnershipRange, MatCreateAIJ]"); CHKERRQ(ierr);    hline();
	ierr = VecGetOwnershipRange(uPacked, &start, &end); CHKERRQ(ierr);
	uLocalSize = end-start;
	ierr = VecGetOwnershipRange(f, &start, &end); CHKERRQ(ierr);
	fLocalSize = end-start;
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, fLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 5, NULL, 5, NULL, &ET); CHKERRQ(ierr);
	
	// setting coordinates of the Lagrangian boundary
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreating and printing body points and global cell indices on each process [PetscMalloc]"); CHKERRQ(ierr);  hline();
	ierr = PetscMalloc(nb*sizeof(PetscReal), &bx); CHKERRQ(ierr);
	ierr = PetscMalloc(nb*sizeof(PetscReal), &by); CHKERRQ(ierr);
	ierr = PetscMalloc(nb*sizeof(PetscInt), &I); CHKERRQ(ierr);
	ierr = PetscMalloc(nb*sizeof(PetscInt), &J); CHKERRQ(ierr);
	for(k=0; k<nb; k++)
	{
		bx[k] = 0.5 + 0.25*cos(2*PETSC_PI/nb * k);
		by[k] = 0.5 + 0.25*sin(2*PETSC_PI/nb * k);
		I[k]  = floor(bx[k]/dx);
		J[k]  = floor(by[k]/dy);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%f %f %d %d\n", bx[k], by[k], I[k], J[k]); CHKERRQ(ierr);
	}
	
	// set up the matrix
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nAssemble matrix ET [DMCompositeGetGlobalISs, ISGetIndices, DMDAGetCorners, MatSetValues, ISRestoreIndices]"); CHKERRQ(ierr);  hline();
	ierr = DMCompositeGetGlobalISs(pack, &is); CHKERRQ(ierr);
	// x-component
	ierr = ISGetIndices(is[0], &uindices); CHKERRQ(ierr);
	ierr = ISGetIndices(is[1], &vindices); CHKERRQ(ierr);
	ierr = DMDAGetCorners(uda, &umstart, &unstart, NULL, &um, &un, NULL);
	ierr = DMDAGetCorners(vda, &vmstart, &vnstart, NULL, &vm, &vn, NULL);
	for(k=0; k<nb; k++)
	{
		// x-component
		localIdx=0;
		col = k;
		for(j=unstart; j<unstart+un; j++)
		{
			for(i=umstart; i<umstart+um; i++)
			{
				row = uindices[localIdx];
				if(j>=J[k]-1 && j<=J[k]+1 && i>=I[k]-2 && i<=I[k]+1)
				{
					x = i*dx + dx;
					y = j*dy + dy/2;
					value = dx*delta(x-bx[k], y-by[k], dx);
					ierr = MatSetValues(ET, 1, &row, 1, &col, &value, INSERT_VALUES); CHKERRQ(ierr);
				}
				localIdx++;
			}
		}	
		// y-component
		localIdx=0;
		col = k + nb;
		for(j=vnstart; j<vnstart+vn; j++)
		{
			for(i=vmstart; i<vmstart+vm; i++)
			{
				row = vindices[localIdx];
				if(j>=J[k]-2 && j<=J[k]+1 && i>=I[k]-1 && i<=I[k]+1)
				{
					x = i*dx + dx/2;
					y = j*dy + dy;
					value = dy*delta(x-bx[k], y-by[k], dx);
					ierr = MatSetValues(ET, 1, &row, 1, &col, &value, INSERT_VALUES); CHKERRQ(ierr);
				}
				localIdx++;
			}
		}
	}
	ierr = ISRestoreIndices(is[0], &uindices); CHKERRQ(ierr);
	ierr = ISRestoreIndices(is[1], &vindices); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(ET, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(ET, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nf = (ET)T*U [MatMultTranspose]"); CHKERRQ(ierr);    hline();
	ierr = MatMultTranspose(ET, uPacked, f); CHKERRQ(ierr);
	ierr = VecView(f, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nC = E*ET [MatTransposeMatMult]"); CHKERRQ(ierr); hline();
	ierr = MatTransposeMatMult(ET, ET, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C); CHKERRQ(ierr);
	ierr = MatView(C, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	// Destroy structures
	ierr = MatDestroy(&C); CHKERRQ(ierr);
	ierr = MatDestroy(&ET); CHKERRQ(ierr);
	ierr = PetscFree(bx); CHKERRQ(ierr);
	ierr = PetscFree(by); CHKERRQ(ierr);
	ierr = PetscFree(I); CHKERRQ(ierr);
	ierr = PetscFree(J); CHKERRQ(ierr);
	ierr = ISDestroy(&is[0]); CHKERRQ(ierr);
	ierr = ISDestroy(&is[1]); CHKERRQ(ierr);
	ierr = PetscFree(is); CHKERRQ(ierr);
	ierr = PetscFree(bx); CHKERRQ(ierr);
	ierr = PetscFree(by); CHKERRQ(ierr);
	ierr = VecDestroy(&f); CHKERRQ(ierr);
	ierr = VecDestroy(&uPacked); CHKERRQ(ierr);
	ierr = DMDestroy(&vda); CHKERRQ(ierr);
	ierr = DMDestroy(&uda); CHKERRQ(ierr);
	ierr = DMDestroy(&pack); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
