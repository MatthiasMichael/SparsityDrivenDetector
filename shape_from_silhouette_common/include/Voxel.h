#pragma once

#include "cuda_vector_functions_interop.h"
#include "cuda_math_utils.h"

#include "Roi3DF.h"


namespace sfs
{
	struct Voxel
	{
		__host__ __device__ Voxel() : Voxel(make_float3(0), make_float3(0))
		{
			// empty
		}


		__host__ __device__ Voxel(float3 center, float3 size) :
			center(center),
			size(size),
			isActive(false)
		{
			//empty
		}


		__host__ __device__ double distTo(const Voxel & other) const
		{
			float3 distVec = center - other.center;
			distVec = distVec / size;
			distVec.z = 0; // Voxels above or underneath each other can always be clustered together

			return length(distVec);
		}


		__host__ Roi3DF getBoundingBox() const
		{
			return Roi3DF(center.x - size.x, center.y - size.y, center.z - size.z,
			              center.x + size.x, center.y + size.y, center.z + size.z);
		}


		__host__ __device__ Corners<float3>::type getCorners() const
		{
			Corners<float3>::type vc;

			vc.push_back(make_float3(center.x - size.x / 2, center.y - size.y / 2, center.z - size.z / 2));
			vc.push_back(make_float3(center.x - size.x / 2, center.y - size.y / 2, center.z + size.z / 2));
			vc.push_back(make_float3(center.x - size.x / 2, center.y + size.y / 2, center.z - size.z / 2));
			vc.push_back(make_float3(center.x - size.x / 2, center.y + size.y / 2, center.z + size.z / 2));
			vc.push_back(make_float3(center.x + size.x / 2, center.y - size.y / 2, center.z - size.z / 2));
			vc.push_back(make_float3(center.x + size.x / 2, center.y - size.y / 2, center.z + size.z / 2));
			vc.push_back(make_float3(center.x + size.x / 2, center.y + size.y / 2, center.z - size.z / 2));
			vc.push_back(make_float3(center.x + size.x / 2, center.y + size.y / 2, center.z + size.z / 2));

			return vc;
		}


		float3 center;
		float3 size;

		bool isActive;
	};


	inline __host__ __device__ bool operator==(const Voxel & a, const Voxel & b)
	{
		return a.center == b.center &&
			a.size == b.size;
	}


	inline __host__ __device__ bool operator!=(const Voxel & a, const Voxel & b)
	{
		return !(a == b);
	}
}
