#pragma once

#include <vector>

#include "FixedSizeVector.h"

#include "Roi3DF.h"

#include "cuda_vector_functions_interop.h"

typedef unsigned int uint;


typedef FixedSizeVector< float2, 2 * Corners<float2>::NumPoints > ConvexVoxelProjection;

// Can maybe be replaced by the Geometry BoundingBox
struct BoundingBox2D
{
	BoundingBox2D();
	BoundingBox2D(const std::vector<float2> & polygon);

	int numRows() const;
	int numCols() const;

	bool isInside(int2 p);

	int x1, x2, y1, y2;
};


std::vector<float2> calculateConvexHull(const std::vector<float2> & corners);
std::vector<int2> calculatePointsOnHullEdge(const std::vector<float2> & hull);


template<typename Vec2D>
bool isInside(const Vec2D & point, ConvexVoxelProjection polygon)
{
	const float x = static_cast<float>(point.x);
	const float y = static_cast<float>(point.y);

	bool c = false;

	for (size_t i = 0, j = polygon.size()-1; i < polygon.size(); j = i++) 
	{
		if ( ((polygon[i].y > y) != (polygon[j].y > y)) && (x < (polygon[j].x-polygon[i].x) * (y - polygon[i].y) / (polygon[j].y-polygon[i].y) + polygon[i].x) )
		{
			c = !c;
		}
	}
	return c;
}