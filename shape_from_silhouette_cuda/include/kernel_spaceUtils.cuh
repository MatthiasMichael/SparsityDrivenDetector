#pragma once

#include <vector_functions.h>
#include "cuda_math_utils.h"


namespace sfs
{
	namespace cuda
	{

		__device__ inline bool isValidVoxelIdx(uint x, uint y, uint z, uint offset, uint3 numVoxel)
		{
			return x < numVoxel.x && y < numVoxel.y && z < numVoxel.z && offset < numVoxel.x * numVoxel.y * numVoxel.z;
		}


		__device__ inline bool toValidImageIdx(float x, float y, uint * retX, uint * retY, uint2 sizeImg)
		{
			if (x < 0 || x >= sizeImg.x - 1 || y < 0 && y >= sizeImg.y - 1)
			{
				*retX = 0;
				*retY = 0;
				return false;
			}

			*retX = (uint)(x + 0.5f);
			*retY = (uint)(y + 0.5f);

			return true;
		}


		__device__ inline bool isValidImageIdx(float x, float y, uint2 sizeImg)
		{
			if (x < 0 || x >= sizeImg.x || y < 0 && y >= sizeImg.y)
			{
				return false;
			}

			return true;
		}
	}
}