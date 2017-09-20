#include <cstdio>
#include <numeric>
#include <iostream>
#include "ext_cuda_chunk.hpp"
#include "kernels/pack_kernel.cuknl"

using std::ceil;
using std::accumulate;

#define CELL_DATA 1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4
#define Z_FACE_DATA 5
#define WARP_SIZE 32.0

#define PREPARE_INC(fieldType) \
	int xInc = 0; \
	int yInc = 0; \
	int zInc = 0; \
	switch(fieldType) \
	{ \
		case CELL_DATA:\
			break; \
		case VERTEX_DATA: \
			xInc = yInc = zInc = 1; \
			break; \
		case X_FACE_DATA: \
			xInc = 1; \
			break; \
		case Y_FACE_DATA: \
			yInc = 1; \
			break; \
		case Z_FACE_DATA: \
			zInc = 1; \
			break; \
	} \

// Entry point for packing messages
extern "C"
void ext_pack_message_(
		const int* chunk,
		const int* fields,
		const int* offsets,
		const int* depth,
		const int* face,
		const int* fieldType,
		double* buffer)
{
	Chunks[*chunk-1]->PackUnpackKernel(fields, offsets, *depth, *face, *fieldType, buffer, true);
}

// Entry point for unpacking messages
extern "C"
void ext_unpack_message_(
		const int* chunk,
		const int* fields,
		const int* offsets,
		const int* depth,
		const int* face,
		const int* fieldType,
		double* buffer)
{
	Chunks[*chunk-1]->PackUnpackKernel(fields, offsets, *depth, *face, *fieldType, buffer, false);
}

// Performs buffer packing and unpacking
void TeaLeafCudaChunk::PackUnpackKernel(
		const int* fields,
		const int* offsets,
		const int depth,
		const int face,
		const int fieldType,
		double* buffer,
		const bool pack)
{
	const int exchanges = accumulate(fields, fields+NUM_FIELDS, 0);

	if(exchanges < 1) return;

	PREPARE_INC(fieldType);

	std::string kernelName;
	double* deviceBuffer = NULL;
	CuKnlPackType packKernel = NULL;

	int bufferLength = 0;
	int innerX = xCells-2*HALO_PAD;
	int innerY = yCells-2*HALO_PAD;
	int innerZ = zCells-2*HALO_PAD;
	
	switch(face)
	{
		case CHUNK_LEFT:
			kernelName = (pack) ? "Pack Left" : "Unpack Left";
			packKernel = (pack) ? CuKnlPackLeft : CuKnlUnpackLeft;
			deviceBuffer = dLeftBuffer;
			bufferLength = innerY*innerZ*depth;
			break;
		case CHUNK_RIGHT:
			kernelName = (pack) ? "Pack Right" : "Unpack Right";
			packKernel = (pack) ? CuKnlPackRight : CuKnlUnpackRight;
			deviceBuffer = dRightBuffer;
			bufferLength = innerY*innerZ*depth;
			break;
		case CHUNK_TOP:
			kernelName = (pack) ? "Pack Top" : "Unpack Top";
			packKernel = (pack) ? CuKnlPackTop : CuKnlUnpackTop;
			deviceBuffer = dTopBuffer;
			bufferLength = innerX*innerZ*depth;
			break;
		case CHUNK_BOTTOM:
			kernelName = (pack) ? "Pack Bottom" : "Unpack Bottom";
			packKernel = (pack) ? CuKnlPackBottom : CuKnlUnpackBottom;
			deviceBuffer = dBottomBuffer;
			bufferLength = innerX*innerZ*depth;
			break;
		case CHUNK_FRONT:
			kernelName = (pack) ? "Pack Front" : "Unpack Front";
			packKernel = (pack) ? CuKnlPackFront : CuKnlUnpackFront;
			deviceBuffer = dFrontBuffer;
			bufferLength = innerX*innerY*depth;
			break;
		case CHUNK_BACK:
			kernelName = (pack) ? "Pack Back" : "Unpack Back";
			packKernel = (pack) ? CuKnlPackBack : CuKnlUnpackBack;
			deviceBuffer = dBackBuffer;
			bufferLength = innerX*innerY*depth;
			break;
		default:
			TeaLeafCudaChunk::Abort(__LINE__, __FILE__, 
					"Incorrect face provided: %d.\n", face);
	}

	if(!pack)
	{
		cudaMemcpy(deviceBuffer, buffer, exchanges*bufferLength*sizeof(double), 
				cudaMemcpyHostToDevice);
		TeaLeafCudaChunk::CheckErrors(__LINE__,__FILE__);
	}

	int offset = 0;
	int numBlocks = ceil(bufferLength/(float)BLOCK_SIZE);

	for(int ii = 0; ii != NUM_FIELDS; ++ii)
	{
		if(fields[ii])
		{
			double* deviceField = NULL;
			switch(ii+1)
			{
				case FIELD_DENSITY:
					deviceField = dDensity;
					break;
				case FIELD_ENERGY0:
					deviceField = dEnergy0;
					break;
				case FIELD_ENERGY1:
					deviceField = dEnergy1;
					break;
				case FIELD_U:
					deviceField = dU;
					break;
				case FIELD_P:
					deviceField = dP;
					break;
				case FIELD_SD:
					deviceField = dSd;
					break;
				default:
					TeaLeafCudaChunk::Abort(__LINE__,__FILE__,
							"Incorrect field provided: %d.\n", ii+1);
			}

			START_PROFILING();

			int bufferOffset = bufferLength*offset++;
			packKernel<<<numBlocks, BLOCK_SIZE>>>(
					xCells, yCells, zCells, innerX, innerY, innerZ, 
					deviceField, deviceBuffer+bufferOffset, depth);

			POST_KERNEL(kernelName.c_str());
		}
	}

	if(pack)
	{
		cudaMemcpy(buffer, deviceBuffer, exchanges*bufferLength*sizeof(double),
				cudaMemcpyDeviceToHost);
		TeaLeafCudaChunk::CheckErrors(__LINE__,__FILE__);
	}
}
