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
	PetscInt         rank, m, n;
	PetscInt         umstart, unstart, um, un;
	PetscInt         vmstart, vnstart, vm, vn;
	PetscInt         *lx, *ly, *I, *J, *d_nnz, *o_nnz, uRows;
	const PetscInt   *lxu, *lyu;
	PetscInt         uStart, uEnd, uLocalSize, phiLocalSize, phiStart, phiEnd;
	PetscInt         row, cols[2], col;
	PetscInt         nx = 12, ny = 12, localIdx;
	PetscReal        dx = 1.0/nx, dy = 1.0/ny;
	PetscInt         nb = ceil(2*PETSC_PI*0.25/dx);
	PetscErrorCode   ierr;
	DM               uda, vda, pack;
	Vec              phi, uPacked;
	Mat              Q, C;
	PetscReal        value, values[2], *bx, *by, x, y;
	IS               *is;
	const PetscInt   *uIndices, *vIndices;

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Create matrix QT that maps from a vector P+f to a DMComposite of distributed arrays U and V"); CHKERRQ(ierr);    
	hline();

	// Create distributed array uv and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate P and f vectors [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRQ(ierr);    hline();
	// uPacked
	// create composite DM
	ierr = DMCompositeCreate(PETSC_COMM_WORLD, &pack); CHKERRQ(ierr);
	// create DA for U
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx-1, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &uda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, uda); CHKERRQ(ierr);
	// determine distribution of U across processes
	ierr = DMDAGetInfo(uda, NULL, NULL, NULL, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
	ierr = DMDAGetOwnershipRanges(uda, &lxu, &lyu, NULL); CHKERRQ(ierr);
	// obtain distribution for V across processes
	ierr = PetscMalloc(m*sizeof(*lx), &lx); CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(*ly), &ly); CHKERRQ(ierr);
	ierr = PetscMemcpy(lx ,lxu, m*sizeof(*lx)); CHKERRQ(ierr);
	ierr = PetscMemcpy(ly ,lyu, n*sizeof(*ly)); CHKERRQ(ierr);
	lx[m-1]++;
	ly[n-1]--;
	// create DA for V
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny-1, m, n, 1, 1, lx, ly, &vda); CHKERRQ(ierr);
	ierr = DMCompositeAddDM(pack, vda); CHKERRQ(ierr);
	PetscFree(lx);
	PetscFree(ly);
	// create global vector UV
 	ierr = DMCreateGlobalVector(pack, &uPacked); CHKERRQ(ierr);
 	// Create PHI
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, nx*ny+2*nb, &phi); CHKERRQ(ierr);
	
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
	}
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix Q that maps from P to U [VecGetOwnershipRange, MatCreateAIJ]"); CHKERRQ(ierr);    hline();
	// ownership range of UV
	ierr = VecGetOwnershipRange(uPacked, &uStart, &uEnd); CHKERRQ(ierr);
	uLocalSize = uEnd-uStart;
	// create arrays to store nnz values
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &d_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &o_nnz); CHKERRQ(ierr);
	// ownership range of PHI
	ierr = VecGetOwnershipRange(phi, &phiStart, &phiEnd); CHKERRQ(ierr);
	phiLocalSize = phiEnd-phiStart;
	
	// count the number of non-zeros in each row
	ierr = DMDAGetCorners(uda, &umstart, &unstart, NULL, &um, &un, NULL); CHKERRQ(ierr);
	ierr = DMDAGetCorners(vda, &vmstart, &vnstart, NULL, &vm, &vn, NULL); CHKERRQ(ierr);
	localIdx = 0;
	for(j=unstart; j<unstart+un; j++)
	{
		for(i=umstart; i<umstart+um; i++)
		{
			d_nnz[localIdx] = 0;
			o_nnz[localIdx] = 0;
			// G portion
			col = j*nx+i;
			(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			col = j*nx+i+1;
			(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			// ET portion
			for(k=0; k<nb; k++)
			{
				col = nx*ny+k;
				if(j>=J[k]-1 && j<=J[k]+1 && i>=I[k]-2 && i<=I[k]+1)
				{
					(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
				}
			}
			localIdx++;
		}
	}
	for(j=vnstart; j<vnstart+vn; j++)
	{
		for(i=vmstart; i<vmstart+vm; i++)
		{
			d_nnz[localIdx] = 0;
			o_nnz[localIdx] = 0;
			// G portion
			col = j*nx+i;
			(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			col = (j+1)*nx+i;
			(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			// ET portion
			for(k=0; k<nb; k++)
			{
				col = nx*ny+nb+k;
				if(j>=J[k]-2 && j<=J[k]+1 && i>=I[k]-1 && i<=I[k]+1)
				{
					(col>=phiStart && col<phiEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
				}
			}
			localIdx++;
		}
	}
	
	// allocate memory for the matrix
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, uLocalSize, phiLocalSize, PETSC_DETERMINE, PETSC_DETERMINE, 0, d_nnz, 0, o_nnz, &Q); CHKERRQ(ierr);
	
	// assemble the matrix
	ierr = DMCompositeGetGlobalISs(pack, &is); CHKERRQ(ierr);
	ierr = ISGetIndices(is[0], &uIndices); CHKERRQ(ierr);
	ierr = ISGetIndices(is[1], &vIndices); CHKERRQ(ierr);
	localIdx = 0;
	for(j=unstart; j<unstart+un; j++)
	{
		for(i=umstart; i<umstart+um; i++)
		{
			// G portion
			row = uIndices[localIdx];
			cols[0] = j*nx+i;
			cols[1] = j*nx+i+1;
			values[0] = -1;
			values[1] = 1;
			ierr = MatSetValues(Q, 1, &row, 2, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			// ET portion
			for(k=0; k<nb; k++)
			{
				col = nx*ny+k;
				if(j>=J[k]-1 && j<=J[k]+1 && i>=I[k]-2 && i<=I[k]+1)
				{
					x = i*dx + dx;
					y = j*dy + dy/2;
					value = dx*delta(x-bx[k], y-by[k], dx);
					ierr = MatSetValues(Q, 1, &row, 1, &col, &value, INSERT_VALUES); CHKERRQ(ierr);
				}
			}
			localIdx++;
		}
	}
	uRows = localIdx;
	for(j=vnstart; j<vnstart+vn; j++)
	{
		for(i=vmstart; i<vmstart+vm; i++)
		{
			// G portion
			row = vIndices[localIdx-uRows];
			cols[0] = j*nx+i;
			cols[1] = (j+1)*nx+i;
			values[0] = -1;
			values[1] = 1;
			ierr = MatSetValues(Q, 1, &row, 2, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			// ET portion
			for(k=0; k<nb; k++)
			{
				col = nx*ny+nb+k;
				if(j>=J[k]-2 && j<=J[k]+1 && i>=I[k]-1 && i<=I[k]+1)
				{
					x = i*dx + dx/2;
					y = j*dy + dy;
					value = dy*delta(x-bx[k], y-by[k], dx);
					ierr = MatSetValues(Q, 1, &row, 1, &col, &value, INSERT_VALUES); CHKERRQ(ierr);
				}
			}
			localIdx++;
		}
	}
	ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = ISRestoreIndices(is[0], &uIndices); CHKERRQ(ierr);
	ierr = ISRestoreIndices(is[1], &vIndices); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	//ierr = MatView(Q, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nC = QT*Q [MatTransposeMatMult]"); CHKERRQ(ierr); hline();
	ierr = MatTransposeMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C); CHKERRQ(ierr);
	ierr = MatView(C, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Destroy structures
	ierr = MatDestroy(&Q); CHKERRQ(ierr);
	ierr = MatDestroy(&C); CHKERRQ(ierr);
	ierr = PetscFree(d_nnz); CHKERRQ(ierr);
	ierr = PetscFree(o_nnz); CHKERRQ(ierr);
	ierr = ISDestroy(&is[0]); CHKERRQ(ierr);
	ierr = ISDestroy(&is[1]); CHKERRQ(ierr);
	ierr = PetscFree(is); CHKERRQ(ierr);
	ierr = PetscFree(bx); CHKERRQ(ierr);
	ierr = PetscFree(by); CHKERRQ(ierr);
	ierr = PetscFree(I); CHKERRQ(ierr);
	ierr = PetscFree(J); CHKERRQ(ierr);
	ierr = VecDestroy(&phi); CHKERRQ(ierr);
	ierr = VecDestroy(&uPacked); CHKERRQ(ierr);
	ierr = DMDestroy(&vda); CHKERRQ(ierr);
	ierr = DMDestroy(&uda); CHKERRQ(ierr);
	ierr = DMDestroy(&pack); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return 0;
}
