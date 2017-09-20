#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/jacobi_solve.cuknl"

/*
 *		JACOBI SOLVER KERNEL
 */

using std::ceil;

// Entry point for Jacobi initialisation.
extern "C"
void ext_jacobi_kernel_init_(
		const int* chunk,
		const int* coefficient,
		const double* dt,
		double* rx,
		double* ry,
		double* rz)
{
	Chunks[*chunk-1]->JacobiInit(*dt, rx, ry, rz, *coefficient);
}

// Entry point for Jacobi solver main method.
extern "C"
void ext_jacobi_kernel_solve_(
		const int* chunk,
		double* error)
{
	Chunks[*chunk-1]->JacobiSolve(error);
}

// Jacobi solver initialisation method.
void TeaLeafCudaChunk::JacobiInit(
		const double dt,
		double* rx,
		double* ry,
		double* rz,
		const int coefficient)
{
	if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
	{
		Abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
	}

	CalcRxRyRz(dt, rx, ry, rz);

	PRE_KERNEL(HALO_PAD);

	CuKnlJacobiInit<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			dDensity, dEnergy1, *rx, *ry, *rz, dKx, dKy, 
			dKz, dU0, dU, coefficient);

	POST_KERNEL("Jacobi Initialise");
}

void TeaLeafCudaChunk::JacobiCopyU()
{
	PRE_KERNEL(0);

	CuKnlJacobiCopyU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, dU, dR);

	POST_KERNEL("Jacobi Copy U");
}

// Main Jacobi solver method.
void TeaLeafCudaChunk::JacobiSolve(
		double* error)
{
	JacobiCopyU();

	PRE_KERNEL(2*HALO_PAD);

	CuKnlJacobiSolve<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			dKx, dKy, dKz, dU0, dR, dU, dReduceBuffer1);

	POST_KERNEL("Jacobi Solve");

	SumReduce(dReduceBuffer1, error, numBlocks, "Jacobi Reduction");
}

