#include "kernel_convexHull.cuh"

#include <cfloat> // float limits without numeric_limits (we're on GPU)

#include <thrust/copy.h>
#include <thrust/sort.h>

#include "cuda_analyticGeometry.h"

namespace sfs
{
	namespace cuda
	{

		__host__ __device__ BoundingBox::BoundingBox() : x1(0), x2(0), y1(0), y2(0)
		{
			// empty
		}


		__host__ __device__ BoundingBox::BoundingBox(const ConvexVoxelProjection & polygon) : x1(0), x2(0), y1(0), y2(0)
		{
			if (polygon.empty())
			{
				return;
			}

			float x1f = FLT_MAX, y1f = FLT_MAX;
			float x2f = FLT_MIN, y2f = FLT_MIN;

			for (const auto & p : polygon)
			{
				x1f = p.x < x1f ? p.x : x1f;
				x2f = p.x > x2f ? p.x : x2f;

				y1f = p.y < y1f ? p.y : y1f;
				y2f = p.y > y2f ? p.y : y2f;
			}

			x1 = static_cast<uint>(ceil(x1f));
			y1 = static_cast<uint>(ceil(y1f));

			x2 = static_cast<uint>(floor(x2f));
			y2 = static_cast<uint>(floor(y2f));
		}


		__host__ __device__ uint BoundingBox::numRows() const
		{
			if (y1 == 0 && y2 == 0)
			{
				return 0;
			}
			return y2 - y1 + 1;
		}


		__host__ __device__ uint BoundingBox::numCols() const
		{
			if (x1 == 0 && x2 == 0)
			{
				return 0;
			}
			return x2 - x1 + 1;
		}


		__host__ __device__ bool BoundingBox::isInside(uint2 p)
		{
			return p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2;
		}


		__host__ __device__ bool smaller(const float2 & lhs, const float2 & rhs)
		{
			return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
		}


		__host__ __device__ bool greater(const float2 & lhs, const float2 & rhs)
		{
			return lhs.x > rhs.x || (lhs.x == rhs.x && lhs.y > rhs.y);
		}


		__host__ __device__ float cross(const float2 &O, const float2 &A, const float2 &B)
		{
			return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
		}


		__host__ __device__ ConvexVoxelProjection calculateConvexHull(const Corners<float2>::type & corners)
		{
			float2 pointsSorted[Corners<float2>::NumPoints];

			thrust::copy(thrust::seq, corners.begin(), corners.end(), &pointsSorted[0]);
			thrust::sort(thrust::seq, &pointsSorted[0], &pointsSorted[Corners<float2>::NumPoints], &smaller);

			int k = 0;
			ConvexVoxelProjection H;
			H.resize(2 * Corners<float2>::NumPoints);

			// Build lower hull
			for (int i = 0; i < Corners<float2>::NumPoints; ++i)
			{
				while (k >= 2 && cross(H[k - 2], H[k - 1], pointsSorted[i]) <= 0)
				{
					k--;
				}
				H[k++] = pointsSorted[i];
			}

			// Build upper hull
			for (int i = Corners<float2>::NumPoints - 2, t = k + 1; i >= 0; i--)
			{
				while (k >= t && cross(H[k - 2], H[k - 1], pointsSorted[i]) <= 0)
				{
					k--;
				}
				H[k++] = pointsSorted[i];
			}

			H.resize(k - 1);

			return H;
		}


		__host__ __device__ uint vectorOffsetToLine(const BoundingBox & bb, uint y, uint2 * pPoints)
		{
			uint offset = (y - bb.y1) * 2;
			if (pPoints[offset].x != UINT_MAX)
			{
				++offset;
			}

			return offset;
		}


		__host__ __device__ uint calculatePointsOnHullEdge(const ConvexVoxelProjection & hull, uint2 * pPoints)
		{
			BoundingBox bb(hull);
			uint numPointsInHull = 0;

			thrust::fill(thrust::seq, pPoints, pPoints + 2 * bb.numRows(), make_uint2(UINT_MAX, UINT_MAX));

			for (uint i = 0, j = hull.size() - 1; i < hull.size(); j = i++)
			{
				const uint y_int = static_cast<unsigned int>(hull[i].y);
				uint offset = vectorOffsetToLine(bb, y_int, pPoints);

				pPoints[offset] = make_uint2(static_cast<uint>(hull[i].x), static_cast<uint>(hull[i].y));

				if (offset % 2 == 1)
				{
					if (pPoints[offset].x < pPoints[offset - 1].x)
					{
						const uint2 helper = pPoints[offset];
						pPoints[offset] = pPoints[offset - 1];
						pPoints[offset - 1] = helper;
					}

					numPointsInHull += pPoints[offset].x - pPoints[offset - 1].x + 1;
				}

				const float2 p1 = hull[j].y <= hull[i].y ? hull[j] : hull[i];
				const float2 p2 = hull[j].y > hull[i].y ? hull[j] : hull[i];

				const float2 dir1 = p2 - p1;
				const float2 dir2 = make_float2(1.f, 0.f);

				for (float y = p1.y + 1; y < p2.y; y += 1)
				{
					const float2 pointOnLine = make_float2(p1.x, y);
					float2 intersectionPoint;

					intersectLineAndLine(&intersectionPoint, p1, dir1, pointOnLine, dir2);
					assert(std::abs(intersectionPoint.y - y) < 0.5f);

					const uint2 pointOnEdge = make_uint2(static_cast<uint>(intersectionPoint.x + 0.5f), static_cast<uint>(y));

					const uint y_int2 = static_cast<unsigned int>(y);
					offset = vectorOffsetToLine(bb, y_int2, pPoints);

					pPoints[offset] = pointOnEdge;

					if (offset % 2 == 1)
					{
						if (pPoints[offset].x < pPoints[offset - 1].x)
						{
							const uint2 helper = pPoints[offset];
							pPoints[offset] = pPoints[offset - 1];
							pPoints[offset - 1] = helper;
						}

						numPointsInHull += pPoints[offset].x - pPoints[offset - 1].x + 1;
					}
				}
			}

			if (pPoints[1].x == UINT_MAX)
			{
				pPoints[1] = pPoints[0];
				++numPointsInHull;
			}

			const uint lastIndex = 2 * bb.numRows() - 1;

			if (pPoints[lastIndex].x == UINT_MAX)
			{
				pPoints[lastIndex] = pPoints[lastIndex - 1];
				++numPointsInHull;
			}

			return numPointsInHull;
		}

	}
}