#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/ppcg_solve.cuknl"

/*
 *		PPCG SOLVER KERNEL
 */

using std::ceil;

// Entry point for PPCG initialisation
extern "C"
void ext_ppcg_init_(
		const int* chunk,
		const int* preconditionerOn,
		const double* alphas,
		const double* betas,
		int* numSteps)
{
	Chunks[*chunk-1]->PPCGInit(
			*preconditionerOn, alphas, betas, *numSteps);
}

// Entry point for initialising sd
extern "C"
void ext_ppcg_init_sd_(
		const int* chunk,
		const double* theta)
{
	Chunks[*chunk-1]->PPCGInitSd(*theta);
}

// Entry point for the PPCG inner step
extern "C"
void ext_ppcg_inner_(
		const int* chunk,
		const int* currentStep)
{
	Chunks[*chunk-1]->PPCGInner(*currentStep);
}

// Initialises the PPCG solver
void TeaLeafCudaChunk::PPCGInit(
		const bool preconditionerOn,
		const double* alphas,
		const double* betas,
		const int numSteps)
{
	preconditioner = preconditionerOn;
	LoadAlphaBeta(alphas, betas, numSteps);
}

// Initialises sd
void TeaLeafCudaChunk::PPCGInitSd(
		const double theta)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlPPCGInitSd<<<numBlocks, BLOCK_SIZE>>>(
		innerX, innerY, innerZ, xCells, xCells*yCells,
		theta, preconditioner, dR, dMi, dSd);

	POST_KERNEL("PPCG Init Sd");
}

// The PPCG inner step
void TeaLeafCudaChunk::PPCGInner(
		const int currentStep)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlPPCGUpdateR<<<numBlocks, BLOCK_SIZE>>>(
		innerX, innerY, innerZ, xCells, xCells*yCells, dKx,
		dKy, dKz, dSd, dU, dR);

	POST_KERNEL("PPCG Update R");

	START_PROFILING();

	CuKnlPPCGCalcSd<<<numBlocks, BLOCK_SIZE>>>(
		innerX, innerY, innerZ, xCells, xCells*yCells, currentStep-1,
		preconditioner, dR, dMi, dAlphas, dBetas, dSd);

	POST_KERNEL("PPCG Calc Sd");
}

