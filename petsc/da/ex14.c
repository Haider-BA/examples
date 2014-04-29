#include <petscdmda.h>
#include <petscdmcomposite.h>

typedef struct
{
	PetscInt   nx, ny;
	DM         uda, pda, vda, pack;
	Mat        G, C;
	Vec        uPacked, pGlobal, bc2;
}simInfo;

void hline()
{
	PetscErrorCode ierr;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n------------------------------------------------------------------------------------------------------------------------\n"); CHKERRV(ierr);
}

void createArrays(simInfo *data)
{
	PetscErrorCode   ierr;
	PetscInt         *lx, *ly, m, n;
	const PetscInt   *lxu, *lyu;
	
	// Create distributed array and get vectors
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate staggered U, V and P vectors [DMDACreate2d, DMDAGetInfo, DMDAGetOwnershipRanges, DMCreateGlobalVector]"); CHKERRV(ierr);    hline();
	ierr = DMCompositeCreate(PETSC_COMM_WORLD, &(data->pack)); CHKERRV(ierr);
	
	// create u DA
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, data->nx-1, data->ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &(data->uda)); CHKERRV(ierr);
	ierr = DMCompositeAddDM(data->pack, data->uda); CHKERRV(ierr);
	
	// determine process distribution of v
	ierr = DMDAGetInfo(data->uda, NULL, NULL, NULL, NULL, &m, &n, NULL, NULL, NULL, NULL, NULL, NULL, NULL); CHKERRV(ierr);
	ierr = DMDAGetOwnershipRanges(data->uda, &lxu, &lyu, NULL); CHKERRV(ierr);
	ierr = PetscMalloc(m*sizeof(*lx), &lx); CHKERRV(ierr);
	ierr = PetscMalloc(n*sizeof(*ly), &ly); CHKERRV(ierr);
	ierr = PetscMemcpy(lx ,lxu, m*sizeof(*lx)); CHKERRV(ierr);
	ierr = PetscMemcpy(ly ,lyu, n*sizeof(*ly)); CHKERRV(ierr);
	lx[m-1]++;
	
	// create v DA
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, data->nx, data->ny, m, n, 1, 1, lx, ly, &(data->pda)); CHKERRV(ierr);
	
	// create vectors p and bc2
	ierr = DMCreateGlobalVector(data->pda, &(data->pGlobal)); CHKERRV(ierr);
	ierr = VecDuplicate(data->pGlobal, &(data->bc2)); CHKERRV(ierr);
	ly[n-1]--;
	
	// create p DA
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, data->nx, data->ny-1, m, n, 1, 1, lx, ly, &(data->vda)); CHKERRV(ierr);
	ierr = DMCompositeAddDM(data->pack, data->vda); CHKERRV(ierr);
	
	PetscFree(lx);
	PetscFree(ly);
	
	// create vector uPacked
	ierr = DMCreateGlobalVector(data->pack, &(data->uPacked)); CHKERRV(ierr);
}

void destroyArrays(simInfo *data)
{
	PetscErrorCode ierr;
	//ierr = MatDestroy(&C); CHKERRV(ierr);
	//ierr = MatDestroy(&G); CHKERRV(ierr);
	//ierr = ISDestroy(&is[0]); CHKERRV(ierr);
	//ierr = ISDestroy(&is[1]); CHKERRV(ierr);
	//ierr = PetscFree(is); CHKERRV(ierr);
	ierr = VecDestroy(&(data->bc2)); CHKERRV(ierr);
	ierr = VecDestroy(&(data->uPacked)); CHKERRV(ierr);
	ierr = VecDestroy(&(data->pGlobal)); CHKERRV(ierr);
	ierr = VecDestroy(&(data->bc2)); CHKERRV(ierr);
	ierr = DMDestroy(&(data->vda)); CHKERRV(ierr);
	ierr = DMDestroy(&(data->uda)); CHKERRV(ierr);
	ierr = DMDestroy(&(data->pda)); CHKERRV(ierr);
	ierr = DMDestroy(&(data->pack)); CHKERRV(ierr);
}

int main(int argc,char **argv)
{
	PetscErrorCode   ierr;
	PetscInt         rank;
	PetscInt         i, j;
	PetscInt         um, un, vm, vn, pm, pn;
	PetscInt         umstart, unstart, vmstart, vnstart;
	PetscInt         *d_nnz, *o_nnz;
	PetscInt         uStart, uEnd, pStart, pEnd, uLocalSize, pLocalSize;
	PetscInt         col;
	//PetscInt         row, cols[2];
	PetscInt         localIdx;
	AO               pao;
	
	//PetscScalar      values[2];
	//IS               *is;
	//const PetscInt   *xindices, *yindices;
	simInfo          data;             

	ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
	
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	
	// defaults
	data.nx = 5;
	data.ny = 5;

	// Read options
	ierr = PetscOptionsGetInt(NULL, "-nx", &(data.nx), NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, "-ny", &(data.ny), NULL); CHKERRQ(ierr);
	
	hline();
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Program that reminds you why you need to use AOApplicationToPetsc if you want to map from P to UV"); CHKERRQ(ierr);    
	hline();
	createArrays(&data);	
		
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCreate matrix G that maps from P to U [VecGetOwnershipRange, MatCreateAIJ]"); CHKERRQ(ierr);    hline();
	// ownership range of u
	ierr = VecGetOwnershipRange(data.uPacked, &uStart, &uEnd); CHKERRQ(ierr);
	uLocalSize = uEnd-uStart;
	// create arrays to store nnz values
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &d_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(uLocalSize*sizeof(PetscInt), &o_nnz); CHKERRQ(ierr);
	// ownership range of phi
	ierr = VecGetOwnershipRange(data.pGlobal, &pStart, &pEnd); CHKERRQ(ierr);
	pLocalSize = pEnd-pStart;
	
	ierr = DMDAGetCorners(data.pda, NULL, NULL, NULL, &pm, &pn, NULL); CHKERRQ(ierr);
	
	// count the number of non-zeros in each row
	ierr = DMDAGetAO(data.pda, &pao); CHKERRQ(ierr);
	ierr = DMDAGetCorners(data.uda, &umstart, &unstart, NULL, &um, &un, NULL); CHKERRQ(ierr);
	localIdx = 0;
	for(j=unstart; j<unstart+un; j++)
	{
		for(i=umstart; i<umstart+um; i++)
		{
			d_nnz[localIdx] = 0;
			o_nnz[localIdx] = 0;
			// G portion
			col  = j*data.nx+i;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			ierr = AOApplicationToPetsc(pao, 1, &col); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			col  = pStart + (j-unstart)*pm + i-umstart;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d\t", col); CHKERRQ(ierr);
			(col>=pStart && col<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			col  = j*data.nx+i+1;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			ierr = AOApplicationToPetsc(pao, 1, &col); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			col  = pStart + (j-unstart)*pm + i+1-umstart;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d\n", col); CHKERRQ(ierr);
			(col>=pStart && col<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			localIdx++;
		}
	}
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "-\n"); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);

	ierr = DMDAGetCorners(data.vda, &vmstart, &vnstart, NULL, &vm, &vn, NULL); CHKERRQ(ierr);
	for(j=vnstart; j<vnstart+vn; j++)
	{
		for(i=vmstart; i<vmstart+vm; i++)
		{
			d_nnz[localIdx] = 0;
			o_nnz[localIdx] = 0;
			// G portion
			col = j*data.nx+i;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			ierr = AOApplicationToPetsc(pao, 1, &col); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			col  = pStart + (j-unstart)*pm + i-umstart;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d\t", col); CHKERRQ(ierr);
			(col>=pStart && col<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			col = (j+1)*data.nx+i;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			ierr = AOApplicationToPetsc(pao, 1, &col); CHKERRQ(ierr);
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d,", col); CHKERRQ(ierr);
			col  = pStart + (j+1-unstart)*pm + i-umstart;
			ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%3d\n", col); CHKERRQ(ierr);
			(col>=pStart && col<pEnd)? d_nnz[localIdx]++ : o_nnz[localIdx]++;
			localIdx++;
		}
	}
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "-\n"); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);

	//ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d + %d = %d\n", m*n, p*q, uLocalSize); CHKERRQ(ierr);
	//ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
	
/*	
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
*/
	destroyArrays(&data);
	ierr = PetscFinalize();
	return 0;
}
