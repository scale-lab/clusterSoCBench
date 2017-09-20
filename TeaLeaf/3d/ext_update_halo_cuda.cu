#include <cstdio>
#include <iostream>
#include "ext_cuda_chunk.hpp"
#include "kernels/update_halo.cuknl"

/*
 * 		UPDATE HALO KERNEL
 */	

using std::ceil;

// Entry point for update the halo
extern "C"
void ext_update_halo_kernel_(
		const int* chunk,
		const int* chunkNeighbours,
		const int* fields,
		const int* depth)
{
	Chunks[*chunk-1]->UpdateHalo(chunkNeighbours, fields, *depth);
}

// Updates all necessary fields for a particular halo
void TeaLeafCudaChunk::UpdateHalo(
		const int* chunkNeighbours,
		const int* fields,
		const int depth)
{
#define LAUNCH_UPDATE(index, buffer, depth)\
	if(fields[index-1])\
	{\
		UpdateFace(chunkNeighbours, depth, buffer);\
	}

	LAUNCH_UPDATE(FIELD_P, dP, depth);
	LAUNCH_UPDATE(FIELD_DENSITY, dDensity, depth);
	LAUNCH_UPDATE(FIELD_ENERGY0, dEnergy0, depth);
	LAUNCH_UPDATE(FIELD_ENERGY1, dEnergy1, depth);
	LAUNCH_UPDATE(FIELD_U, dU, depth);
	LAUNCH_UPDATE(FIELD_SD, dSd, depth);
}

// Updates a field for each required face of a halo
void TeaLeafCudaChunk::UpdateFace(
		const int* chunkNeighbours,
		const int depth,
		double* buffer)
{
	int innerX = xCells-HALO_PAD*2; 
	int innerY = yCells-HALO_PAD*2;
	int innerZ = zCells-HALO_PAD*2;

#define UPDATE_FACE(face, kernelName, updateKernel) \
	if(chunkNeighbours[face-1] == EXTERNAL_FACE)\
	{\
		START_PROFILING();\
		updateKernel<<<numBlocks, BLOCK_SIZE>>>(\
				xCells, yCells, zCells, innerX, \
				innerY, innerZ, depth, buffer);\
		POST_KERNEL(kernelName);\
	}

	int numBlocks = ceil((innerX*innerZ*depth)/(float)BLOCK_SIZE);
	UPDATE_FACE(CHUNK_TOP, "Halo Top", CuKnlUpdateTop);
	UPDATE_FACE(CHUNK_BOTTOM, "Halo Bottom", CuKnlUpdateBottom);

	numBlocks = ceil((innerX*innerY*depth)/(float)BLOCK_SIZE);
	UPDATE_FACE(CHUNK_BACK, "Halo Back", CuKnlUpdateBack);
	UPDATE_FACE(CHUNK_FRONT, "Halo Front", CuKnlUpdateFront);

	numBlocks = ceil((innerY*innerZ*depth)/(float)BLOCK_SIZE);
	UPDATE_FACE(CHUNK_RIGHT, "Halo Right", CuKnlUpdateRight);
	UPDATE_FACE(CHUNK_LEFT, "Halo Left", CuKnlUpdateLeft);
}
