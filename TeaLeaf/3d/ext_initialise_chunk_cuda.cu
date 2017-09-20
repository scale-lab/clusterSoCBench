#include <cstdio>
#include <algorithm>
#include "ext_cuda_chunk.hpp"
#include "kernels/initialise_chunk.cuknl"

/*
 * 		INITIALISE CHUNK KERNEL
 */

// Extended CUDA kernel for the chunk initialisation
extern "C"
void ext_initialise_chunk_cuda_( 
		const int* chunk,
		const double* xMin,
		const double* yMin,
		const double* zMin,
		const double* dx,
		const double* dy,
		const double* dz)
{
	Chunks[*chunk-1]->InitialiseChunk(*xMin, *yMin, *zMin, *dx, *dy, *dz);
}

// Initialises the chunk's primary data fields.
void TeaLeafCudaChunk::InitialiseChunk( 
		const double xMin,
		const double yMin,
		const double zMin,
		const double dx,
		const double dy,
		const double dz)
{
	int numCells = 1+std::max(xCells, std::max(yCells, zCells));
	int numBlocks = std::ceil((float)numCells/(float)BLOCK_SIZE);

	START_PROFILING();

	CuKnlInitialiseChunkVertices<<<numBlocks, BLOCK_SIZE>>>(
			xCells, yCells, zCells, xMin, yMin, zMin, dx, dy, dz,
			dVertexX, dVertexY, dVertexZ, dVertexDx, dVertexDy, dVertexDz);

	POST_KERNEL("Initialise Chunk Vertices");

	numCells = (xCells+1)*(yCells+1)*(zCells+1);
	numBlocks = std::ceil((float)numCells/(float)BLOCK_SIZE);

	START_PROFILING();

	CuKnlInitialiseChunk<<<numBlocks, BLOCK_SIZE>>>(
			xCells, yCells, zCells, dx, dy, dz,
			dVertexX, dVertexY, dVertexZ, 
			dCellX, dCellY, dCellZ, 
			dCellDx, dCellDy, dCellDz, 
			dVolume, dXArea, dYArea, dZArea);

	POST_KERNEL("Initialise Chunk Final");
}
