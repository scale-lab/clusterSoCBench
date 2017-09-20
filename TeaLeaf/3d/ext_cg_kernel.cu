#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/cg_solve.cuknl"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Entry point for CG initialisation.
extern "C"
void ext_cg_solver_init_(
		const int* chunk,
		const int* coefficient,
		const int* preconditioner,
		double* dt,
		double* rx,
		double* ry,
		double* rz,
		double* rro)
{
	Chunks[*chunk-1]->CGInit(
			*coefficient, *preconditioner, *dt, rx, ry, rz, rro);
}

// Entry point for calculating w
extern "C"
void ext_cg_calc_w_(
		const int* chunk,
		double* pw)
{
	Chunks[*chunk-1]->CGCalcW(pw);
}

// Entry point for calculating u and r
extern "C"
void ext_cg_calc_ur_(
		const int* chunk,
		const double* alpha,
		double* rrn)
{
	Chunks[*chunk-1]->CGCalcUr(*alpha, rrn);
}

// Entry point for calculating p
extern "C"
void ext_cg_calc_p_(
		const int* chunk,
		const double* beta)
{
	Chunks[*chunk-1]->CGCalcP(*beta);
}

// Initialises the CG solver
void TeaLeafCudaChunk::CGInit(
		const int coefficient,
		const bool enablePreconditioner,
		const double dt,
		double* rx,
		double* ry,
		double* rz,
		double* rro)
{
	if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
	{
		Abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
	}

	preconditioner = enablePreconditioner;

	CalcRxRyRz(dt, rx, ry, rz);
	CGInitU(coefficient);
	CGInitDirections(*rx, *ry, *rz);
	CGInitOthers(rro);
}

// Initialises u
void TeaLeafCudaChunk::CGInitU(
		const int coefficient)
{
	PRE_KERNEL(0);

	CuKnlCGInitU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, coefficient, dDensity,
			dEnergy1, dU, dP, dR, dW);

	POST_KERNEL("CG Init U");
}

// Initialises the directions kx, ky and kz
void TeaLeafCudaChunk::CGInitDirections(
		double rx, 
		double ry, 
		double rz)
{
	PRE_KERNEL(3);

	CuKnlCGInitDirections<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			dW, dKx, dKy, dKz, rx, ry, rz);

	POST_KERNEL("CG Init Directions");
}

// Initialises the other CG variables
void TeaLeafCudaChunk::CGInitOthers(
		double* rro)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCGInitOthers<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			dU, dKx, dKy, dKz, preconditioner, 
			dReduceBuffer1, dP, dR, dW, dMi, dZ);

	POST_KERNEL("CG Init Others");

	SumReduce(dReduceBuffer1, rro, numBlocks, "CG RRO Reduction");
}

// Calculates a new value for w
void TeaLeafCudaChunk::CGCalcW(
		double* pw)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCGCalcW<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells,
			dKx, dKy, dKz, dP, dReduceBuffer2, dW);

	POST_KERNEL("CG Calculate W");

	SumReduce(dReduceBuffer2, pw, numBlocks, "CG PW Reduction");
}

// Calculates a new value for u and r
void TeaLeafCudaChunk::CGCalcUr(
		const double alpha,
		double* rrn)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCGCalcUr<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			preconditioner, alpha, dMi, dP, dW, dU, dZ, dR, dReduceBuffer3);

	POST_KERNEL("CG Calculate UR");

	SumReduce(dReduceBuffer3, rrn, numBlocks, "CG RRN Reduction");
}

// Calculates a new value for p
void TeaLeafCudaChunk::CGCalcP(
		const double beta)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCGCalcP<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, innerZ, xCells, xCells*yCells, 
			preconditioner, beta, dR, dZ, dP);

	POST_KERNEL("CG Calculate P");
}
