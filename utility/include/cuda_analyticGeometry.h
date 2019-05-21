#pragma once

// This header can be used with and without CUDA SDK present.
// Please include <vector_functions.h> or your custom interoperable
// Header before including this and do not add these headers here.

#include "cuda_math_utils.h"

// Helper functions ripped out of rtcvAnalyticGeometry and converted for float3/4

inline __host__ __device__ void convertPointNormal2HesseParams(float4 * hesse, const float3 & p, const float3 & n)
{
	float3 n_norm = normalize(n);
	float lastEntry = dot(n_norm, p);

	hesse->x = n_norm.x;
	hesse->y = n_norm.y;
	hesse->z = n_norm.z;
	hesse->w = lastEntry;
}


inline __host__ __device__ float intersectLineAndLine(float3 * closestPoint, const float3 & p1, const float3 & dir1, 
											          const float3 & p2, const float3 & dir2,
											          float * outl1 = 0, float * outl2 = 0 )
{
	const float3 v1 = normalize(dir1);
	const float3 v2 = normalize(dir2);

	const float a = dot(v1, v2);
	const float b = dot((p2 - p1), v1);
	const float c = dot( (p2 - p1 - b * v1 ), v2);

	const float l2 = 1 / ( a * a - 1 )	* c;

	const float	l1 = dot((p2 - p1 + l2 * v2), v1);

	if ( outl1 ) 
		*outl1 = l1;
	if ( outl2 ) 
		*outl2 = l2;

	*closestPoint = 0.5 * (p1 + l1 * v1 + p2 + l2 * v2);

	return norm(p1 + l1 * v1 - (p2 + l2 * v2));
}


inline __host__ __device__ float intersectLineAndLine(float2 * closestPoint, const float2 & p1, const float2 & dir1, 
													  const float2 & p2, const float2 & dir2,
													  float * outl1 = 0, float * outl2 = 0 )
{
	const float2 v1 = normalize(dir1);
	const float2 v2 = normalize(dir2);

	const float a = dot(v1, v2);
	const float b = dot((p2 - p1), v1);
	const float c = dot( (p2 - p1 - b * v1 ), v2);

	const float l2 = 1 / ( a * a - 1 )	* c;

	const float	l1 = dot((p2 - p1 + l2 * v2), v1);

	if ( outl1 ) 
		*outl1 = l1;
	if ( outl2 ) 
		*outl2 = l2;

	*closestPoint = 0.5 * (p1 + l1 * v1 + p2 + l2 * v2);

	return norm(p1 + l1 * v1 - (p2 + l2 * v2));
}


inline __host__ __device__ float intersectLineAndPlane(float3 * res, const float3 & point, const float3 & dir, const float3 & normal, float dist)
{
	const float a = dot(normal, point);
	const float b = dot(normal, dir);

	const float scal = (dist - a) / b;

	(*res) = point + scal * dir;

	return scal;
}


inline __host__ __device__ float intersectLineAndPlane(float3 * res, const float3 & point, const float3 & dir, const float4 & hesseNormal)
{
	return intersectLineAndPlane(res, point, dir, top3(hesseNormal), hesseNormal.w);
}