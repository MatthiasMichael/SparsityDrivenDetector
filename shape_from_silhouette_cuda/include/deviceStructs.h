#pragma once

#include "cuda_math_utils.h"

#include "Roi3DF.h"


// This file contains structs that are useful for SfS processing in CUDA
// But also stuff like the EigenVector3_MemoryDouble which is just here to be able to read
// serialized Space files.
// TODO: Find a better name or split in smaller more concise files.

namespace sfs
{
	namespace cuda
	{

		// The original ViewRay needs the Eigen library. We do not have access to that
		struct DeviceViewRay
		{
			float3 ray;
			float3 origin;
		};


		// Currently I am not sure if this is needed since we could also pass just the bounding boxes. 
		// However, additional Information might be added later.
		struct DeviceVoxelCluster
		{
			Roi3DF boundingBox;
			bool isGhost;
		};


		// We need this thing since we can't include any eigen header in any files that are touched 
		// by the nvcc compiler but still need to be able to read saved configurations
		struct EigenVector3_MemoryDouble
		{
			double x, y, z;

			float3 toFloat3() const
			{
				return make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
			}
		};


		// Same as above. Memory double of the ViewRay in [algorithms/RadialCameraModel/include/ViewRay.h]
		struct ViewRay_MemoryDouble
		{
			EigenVector3_MemoryDouble ray;
			EigenVector3_MemoryDouble origin;
			int index;

			DeviceViewRay toDeviceViewRay() const
			{
				DeviceViewRay dvr;
				dvr.ray = ray.toFloat3();
				dvr.origin = origin.toFloat3();
				return dvr;
			}
		};
	}
}