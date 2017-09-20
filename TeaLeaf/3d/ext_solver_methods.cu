#include <cstdio>
#include "ext_cuda_chunk.hpp"
#include "kernels/solver_methods.cuknl"

/*
 *		SHARED SOLVER METHODS
 */

// Entry point to copy U.
extern "C"
void ext_solver_copy_u_(
		const int* chunk)
{
	Chunks[*chunk-1]->CopyU();
}

// Entry point for calculating residual.
extern "C"
void ext_calculate_residual_(
		const int* chunk)
{
	Chunks[*chunk-1]->CalculateResidual();
}

// Entry point for calculating 2norm.
extern "C"
void ext_calculate_2norm_(
		const int* chunk,
		const int* normArray,
	   	double* normOut)
{
	Chunks[*chunk-1]->Calculate2Norm(*normArray, normOut);
}

// Entry point for finalising solution.
extern "C"
void ext_solver_finalise_(
		const int* chunk)
{
	Chunks[*chunk-1]->Finalise();
}

// Determines the rx, ry and rz values.
void TeaLeafCudaChunk::CalcRxRyRz(
		const double dt,
		double* rxOut,
		double* ryOut,
		double* rzOut)
{
	double dx, dy, dz;

	cudaMemcpy(&dx, dCellDx, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dy, dCellDy, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dz, dCellDz, sizeof(double), cudaMemcpyDeviceToHost);
	TeaLeafCudaChunk::CheckErrors(__LINE__,__FILE__);

	*rxOut = dt/(dx*dx);
	*ryOut = dt/(dy*dy);
	*rzOut = dt/(dz*dz);
}

// Copies the current value of u
void TeaLeafCudaChunk::CopyU()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCopyU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, dU, dU0);

	POST_KERNEL("Copy U");
}

// Calculates the current residual value.
void TeaLeafCudaChunk::CalculateResidual()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCalculateResidual<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			dU, dU0, dKx, dKy, dKz, dR);

	POST_KERNEL("Calculate Residual");
}

// Calculates the 2norm of a particular space.
void TeaLeafCudaChunk::Calculate2Norm(
		const bool normArray,
		double* normOut)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCalculate2Norm<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			normArray ? dR : dU0, dReduceBuffer1);

	POST_KERNEL("Calculate 2Norm");
	SumReduce(dReduceBuffer1, normOut, numBlocks, "2norm reduction");
}

// Reduces residual values of a buffer
void TeaLeafCudaChunk::SumReduce(
		double* buffer,
		double* result,
		int len,
		std::string kName)
{
	while(len > 1)
	{
		int numBlocks = ceil(len/(float)BLOCK_SIZE);
		START_PROFILING();
		CuKnlSumReduce<<<numBlocks,BLOCK_SIZE>>>(len, buffer);
		POST_KERNEL(kName);
		len = numBlocks;
	}

	cudaMemcpy(result, buffer, sizeof(double), cudaMemcpyDeviceToHost);
	CheckErrors(__LINE__,__FILE__);
}

// Finalises the solution.
void TeaLeafCudaChunk::Finalise()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlFinalise<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, dDensity, dU, dEnergy1);

	POST_KERNEL("Finalise Solver");
}

// Loads alphas and betas onto the device
void TeaLeafCudaChunk::LoadAlphaBeta(
		const double* alphas,
		const double* betas,
		const int numCoefs)
{
	size_t length = numCoefs*sizeof(double);
	cudaMalloc((void**) &dAlphas, length);
	cudaMalloc((void**) &dBetas, length);
	cudaMemcpy(dAlphas, alphas, length, cudaMemcpyHostToDevice);
	cudaMemcpy(dBetas, betas, length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	CheckErrors(__LINE__,__FILE__);
}
