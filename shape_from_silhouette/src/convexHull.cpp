#include "convexHull.h"

#include <iterator>

#include "rtcvAnalyticGeometry.h"

BoundingBox2D::BoundingBox2D() : x1(0), x2(0), y1(0), y2(0)
{
	// empty
}


BoundingBox::BoundingBox(const std::vector<rtcvPointF> & polygon) : x1(0), x2(0), y1(0), y2(0)
{
	if(polygon.empty())
	{
		return;
	}

	float x1f = FLT_MAX, y1f = FLT_MAX;
	float x2f = FLT_MIN, y2f = FLT_MIN;

	for(const auto & p : polygon)
	{
		x1f = p.x < x1f ? p.x : x1f;
		x2f = p.x > x2f ? p.x : x2f;

		y1f = p.y < y1f ? p.y : y1f;
		y2f = p.y > y2f ? p.y : y2f;
	}

	x1 = static_cast<int>(ceil(x1f));
	y1 = static_cast<int>(ceil(y1f));

	x2 = static_cast<int>(floor(x2f));
	y2 = static_cast<int>(floor(y2f));
}


int BoundingBox::numRows() const
{
	if(y1 == 0 && y2 == 0)
	{
		return 0;
	}
	return y2 - y1 + 1;
}


int BoundingBox::numCols() const
{
	if(x1 == 0 && x2 == 0)
	{
		return 0;
	}
	return x2 - x1 + 1;
}


bool BoundingBox::isInside(rtcvPoint p)
{
	return p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2;
}


bool smaller(const rtcvPointF & lhs, const rtcvPointF & rhs)
{
	return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
}


bool greater(const rtcvPointF & lhs, const rtcvPointF & rhs)
{
	return lhs.x > rhs.x || (lhs.x == rhs.x && lhs.y > rhs.y);
}


float cross(const rtcvPointF &O, const rtcvPointF &A, const rtcvPointF &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}


std::vector<rtcvPointF> calculateConvexHull(const std::vector<rtcvPointF> & corners)
{
	std::vector<rtcvPointF> pointsSorted;

	std::copy(corners.begin(), corners.end(), std::back_inserter(pointsSorted));
	std::sort(pointsSorted.begin(), pointsSorted.end(), &smaller);

	int k = 0;
	std::vector<rtcvPointF> H;
	H.resize(2 * Corners<rtcvPointF>::NumPoints);

	// Build lower hull
	for(int i = 0; i < Corners<rtcvPointF>::NumPoints; ++i) 
	{
		while (k >= 2 && cross(H[k-2], H[k-1], pointsSorted[i]) <= 0) 
		{
			k--;
		}
		H[k++] = pointsSorted[i];
	}

	// Build upper hull
	for(int i = Corners<rtcvPointF>::NumPoints - 2, t = k + 1; i >= 0; i--) 
	{
		while(k >= t && cross(H[k-2], H[k-1], pointsSorted[i]) <= 0) 
		{
			k--;
		}
		H[k++] = pointsSorted[i];
	}

	H.resize(k-1);

	return H;
}


uint vectorOffsetToLine(const BoundingBox & bb, const uint y, std::vector<rtcvPoint> & points)
{
	uint offset = (y - bb.y1) * 2;
	if(points[offset].x != INT_MAX)
	{
		++offset;
	}

	return offset;
}

float dot(const rtcvPointF & a, const rtcvPointF & b)
{
	return a.x * b.x + a.y * b.y;
}

rtcvPointF operator*(float a, const rtcvPointF & b)
{
	return rtcvPointF(a * b.x, a * b.y);
}

float intersectLineAndLine(rtcvPointF * closestPoint, const rtcvPointF & p1, const rtcvPointF & dir1, 
													  const rtcvPointF & p2, const rtcvPointF & dir2,
													  float * outl1 = 0, float * outl2 = 0 )
{
	const rtcvPointF v1 = dir1 / dir1.length();
	const rtcvPointF v2 = dir2 / dir2.length();

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

	return (p1 + l1 * v1 - (p2 + l2 * v2)).length();
}


std::vector<rtcvPoint> calculatePointsOnHullEdge(const std::vector<rtcvPointF> & hull)
{
	BoundingBox bb(hull);
	uint numPointsInHull = 0;

	std::vector<rtcvPoint> points(2 * bb.numRows(), rtcvPoint(INT_MAX, INT_MAX));

	for (size_t i = 0, j = hull.size() - 1; i < hull.size(); j = i++) 
	{
		uint offset = vectorOffsetToLine(bb, static_cast<uint>(hull[i].y), points);

		points[offset] = rtcvPoint(static_cast<uint>(hull[i].x), static_cast<uint>(hull[i].y));

		if(offset % 2 == 1)
		{
			if(points[offset].x < points[offset-1].x)
			{
				const rtcvPoint helper = points[offset];
				points[offset] = points[offset - 1];
				points[offset - 1] = helper;
			}

			numPointsInHull += points[offset].x - points[offset - 1].x + 1;
		}

		const rtcvPointF p1 = hull[j].y <= hull[i].y ? hull[j] : hull[i];
		const rtcvPointF p2 = hull[j].y >  hull[i].y ? hull[j] : hull[i];

		const rtcvPointF dir1 = p2 - p1;
		const rtcvPointF dir2 = rtcvPointF(1.f, 0.f);

		for(float y = p1.y + 1; y < p2.y; y += 1)
		{
			const rtcvPointF pointOnLine = rtcvPointF(p1.x, y); 
			rtcvPointF intersectionPoint;

			intersectLineAndLine(&intersectionPoint, p1, dir1, pointOnLine, dir2);
			assert(std::abs(intersectionPoint.y - y) < 0.5f);

			const rtcvPoint pointOnEdge = rtcvPoint( static_cast<uint>(intersectionPoint.x + 0.5f), static_cast<uint>(y) );

			offset = vectorOffsetToLine(bb, static_cast<uint>(y), points);

			points[offset] = pointOnEdge;

			if( offset % 2 == 1)
			{
				if( points[offset].x < points[offset-1].x)
				{
					const rtcvPoint helper = points[offset];
					points[offset] = points[offset - 1];
					points[offset - 1] = helper;
				}

				numPointsInHull += points[offset].x - points[offset - 1].x + 1;
			}
		}
	}

	if(points[1].x == INT_MAX)
	{
		points[1] = points[0];
		++numPointsInHull;
	}

	const uint lastIndex = 2 * bb.numRows() - 1;

	if(points[lastIndex].x == INT_MAX)
	{
		points[lastIndex] = points[lastIndex - 1];
		++numPointsInHull;
	}

	return points;
}