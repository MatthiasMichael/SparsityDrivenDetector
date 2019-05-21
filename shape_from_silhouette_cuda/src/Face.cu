#include "Face.cuh"

#include "cuda_analyticGeometry.h"

namespace sfs
{
	namespace cuda
	{

		bool __host__ __device__ Face::isOccluded(const float3 & point, const float3 & camera) const
		{
			const float3 direction = point - camera;

			if (abs(dot(direction, top3(m_normal))) < 1e-5) // ViewRay has the same direction as the face;
			{
				return false;
			}

			float3 intersectionPoint;
			const double dist = intersectLineAndPlane(&intersectionPoint, camera, direction, m_normal);

			if (dist < 0 || dist > 1)
				return false;

			//assert( std::abs( distPointToPlane(intersectionPoint, m_normal.topRows(3), m_normal(3)) ) < 1e-5 );	

			const float2 barycentricCoords = getBarycentricCoords(intersectionPoint);
			if (isInside(barycentricCoords))
			{
				return true;
			}

			return false;
		}


		float2 __host__ __device__ Face::getBarycentricCoords(const float3 & point) const
		{
			const float3 v0 = m_vertices[1] - m_vertices[0], v1 = m_vertices[2] - m_vertices[0], v2 = point - m_vertices[0];

			const float d00 = dot(v0, v0);
			const float d01 = dot(v0, v1);
			const float d02 = dot(v0, v2);
			const float d11 = dot(v1, v1);
			const float d12 = dot(v1, v2);
	
			const float denominator = d00 * d11 - d01 * d01;

			return make_float2
			(
				(d11 * d02 - d01 * d12) / denominator,
				(d00 * d12 - d01 * d02) / denominator
			);
		}


		bool __host__ __device__ Face::isInside(const float2 & barycentricPoint) const
		{
			return barycentricPoint.x >= 0 && barycentricPoint.y >= 0 &&
				barycentricPoint.x + barycentricPoint.y <= 1;

		}
	}

}