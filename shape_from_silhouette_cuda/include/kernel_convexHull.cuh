#pragma once

#include <vector_functions.h>

#include "cuda_math_utils.h"

#include "FixedSizeVector.h"

#include "Roi3DF.h"


namespace sfs
{
	namespace cuda
	{
		typedef FixedSizeVector<float2, 2 * Corners<float2>::NumPoints> ConvexVoxelProjection;


		struct BoundingBox
		{
			__host__ __device__ BoundingBox();
			__host__ __device__ BoundingBox(const ConvexVoxelProjection & polygon);

			__host__ __device__ uint numRows() const;
			__host__ __device__ uint numCols() const;

			__host__ __device__ bool isInside(uint2 p);

			uint x1, x2, y1, y2;
		};


		__host__ __device__ ConvexVoxelProjection calculateConvexHull(const Corners<float2>::type & corners);
		__host__ __device__ uint calculatePointsOnHullEdge(const ConvexVoxelProjection & hull, uint2 * pPoints);


		template <typename Vec2D>
		__host__ __device__ bool isInside(const Vec2D & point, ConvexVoxelProjection polygon)
		{
			const float x = static_cast<float>(point.x);
			const float y = static_cast<float>(point.y);

			bool c = false;

			for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
			{
				if (((polygon[i].y > y) != (polygon[j].y > y)) && 
					(x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / 
					(polygon[j].y - polygon[i].y) + polygon[i].x))
				{
					c = !c;
				}
			}
			return c;
		}
	}
}
