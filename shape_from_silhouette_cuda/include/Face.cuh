#pragma once

#include <vector_functions.h>

#include "cuda_analyticGeometry.h"
#include "cuda_math_utils.h"


namespace sfs
{
	namespace cuda
	{

		class Face
		{
		public:

			// ReSharper disable CppPossiblyUninitializedMember
			Face() {} // Cuda does not allow dynamic initialization for types used in __constant__ variables.
			// ReSharper restore CppPossiblyUninitializedMember

			bool __host__ __device__ isOccluded(const float3 & point, const float3 & camera) const;
			bool __host__ __device__ isValid() const { return m_valid; }

			template<typename It>
			static Face __host__ create(const It verticesBegin, const It verticesEnd);

		private:
			float2 __host__ __device__ getBarycentricCoords(const float3 & point) const;
			bool __host__ __device__ isInside(const float2 & barycentricPoint) const;

			float3 m_vertices[3];
			float4 m_normal;

			bool m_valid;
		};


		template<typename It>
		Face __host__ Face::create(const It verticesBegin, const It verticesEnd)
		{
			Face f;

			size_t i = 0;
			for (It iter = verticesBegin; iter != verticesEnd; ++iter)
			{
				f.m_vertices[i++] = make_float3(x(*iter), y(*iter), z(*iter));
			}

			const float3 ab = f.m_vertices[1] - f.m_vertices[0];
			const float3 ac = f.m_vertices[2] - f.m_vertices[0];

			const float3 normal = normalize(cross(ab, ac));

			convertPointNormal2HesseParams(&f.m_normal, f.m_vertices[0], normal);

			f.m_valid = true;

			return f;
		}
	}
}