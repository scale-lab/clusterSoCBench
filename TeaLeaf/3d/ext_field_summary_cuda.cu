#include <stdio.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/field_summary.cuknl"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// Entry point for field summary method.
extern "C"
void ext_field_summary_kernel_(
		const int* chunk,
		double* volOut,
		double* massOut,
		double* ieOut,
		double* tempOut)
{
	Chunks[*chunk-1]->FieldSummary(volOut, massOut, ieOut, tempOut);
}

// Calculates key values from the current field.
void TeaLeafCudaChunk::FieldSummary(
		double* volOut,
		double* massOut,
		double* ieOut,
		double* tempOut)
{
	PRE_KERNEL(HALO_PAD*2);

	CuKnlFieldSummary<<<numBlocks,BLOCK_SIZE>>>(
			innerX, innerY, innerZ, dVolume, dDensity, dEnergy0,
			dU, dReduceBuffer1, dReduceBuffer2, dReduceBuffer3, dReduceBuffer4);

	POST_KERNEL("Field Summary");

	*volOut = thrust::reduce(reducePtr1, reducePtr1+numBlocks, 0.0);
	*massOut = thrust::reduce(reducePtr2, reducePtr2+numBlocks, 0.0);
	*ieOut = thrust::reduce(reducePtr3, reducePtr3+numBlocks, 0.0);
	*tempOut = thrust::reduce(reducePtr4, reducePtr4+numBlocks, 0.0);
}
