#pragma once

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector_types.h>

#include "cuda_math_utils.h"

#include "deviceStructs.h"
#include "Face.cuh"
#include "Voxel.h"

namespace sfs
{
	namespace cuda
	{

		__host__ __device__ bool intersectsInPlane(const Roi3DF & roi, const DeviceViewRay & ray);

		void call_removePixelMapEntriesToNonVisiblePoints(int2 * p_dev_pixelMap, Voxel * p_dev_voxel, const std::vector<Face> & faces,
			std::vector<float3> cameraCenters, uint3 numVoxel);


		void call_checkForSingularViewRays(const std::vector<DeviceVoxelCluster> & voxelCluster, DeviceViewRay * p_dev_viewRays,
			unsigned char * p_dev_imagesSegmentation, unsigned char * p_dev_viewRayMaps, uint numImages, uint2 imageSize);
	}
}