#include "Face.h"


std::optional<Vector3> Face::intersectLine(const Vector3 & point1, const Vector3 & point2) const
{
	ScalarType dist = 0;
	Vector3 intersectionPoint;

	return intersectLine_impl(point1, point2, &dist, &intersectionPoint) ? 
		std::optional<Vector3>{ intersectionPoint } : std::nullopt;
}

std::optional<Vector3> Face::intersectInfiniteLine(const Vector3 & point, const Vector3 & direction) const
{
	ScalarType dist = 0;
	Vector3 intersectionPoint;

	return intersectInfiniteLine_impl(point, direction, &dist, &intersectionPoint) ?
		std::optional<Vector3>{ intersectionPoint } : std::nullopt;
}


bool Face::doesIntersectLine(const Vector3 & point1, const Vector3 & point2) const
{
	ScalarType dist = 0;
	Vector3 intersectionPoint;

	return intersectLine_impl(point1, point2, &dist, &intersectionPoint);
}


bool Face::doesIntersectInfiniteLine(const Vector3 & point, const Vector3 & dir) const
{
	ScalarType dist = 0;
	Vector3 intersectionPoint;

	return intersectInfiniteLine_impl(point, dir, &dist, &intersectionPoint);
}


const Vector3 & Face::getVertex(size_t i) const
{
	assert(i >= 0 && i < 3);
	return m_vertices[i];
}


Vector3 & Face::getVertex(size_t i)
{
	assert(i >= 0 && i < 3);
	return m_vertices[i];
}


Vector4 Face::makeHesseNormal(const Polygon & p)
{
	const Vector3 ab = p[1] - p[0];
	const Vector3 ac = p[2] - p[0];

	Vector3 normal = ab.cross(ac);
	normal.normalize();

	Vector4 hesseNormal;
	hesseNormal.topRows(3) = normal;
	hesseNormal.row(3) = normal.transpose() * p[0];

	return hesseNormal;
}


bool Face::intersectInfiniteLine_impl(const Vector3 & point, const Vector3 & dir, ScalarType * dist, Vector3 * intersectionPoint) const
{
	assert(dist);
	assert(intersectionPoint);

	constexpr ScalarType epsilon = 1e-5f;

	if (std::abs(dir.dot(m_normal.topRows(3))) < epsilon) // ViewRay has the same direction as the face;
	{
		return false;
	}

	const auto norm = m_normal.topRows(3);
	
	*dist = (m_normal(3) - norm.dot(point)) / norm.dot(dir);
	*intersectionPoint = point + *dist * dir;

	const Vector2 barycentricCoords = getBarycentricCoords(*intersectionPoint);

	return isInside(barycentricCoords);
}


bool Face::intersectLine_impl(const Vector3 & point1, const Vector3 & point2, ScalarType * dist, Vector3 * intersectionPoint) const
{
	const Vector3 dir = point2 - point1;

	return intersectInfiniteLine_impl(point1, dir, dist, intersectionPoint) &&
		*dist >= 0 && *dist <= 1;
}


Vector2 Face::getBarycentricCoords(const Vector3 & point) const
{
	const Vector3 v0 = m_vertices[1] - m_vertices[0], v1 = m_vertices[2] - m_vertices[0], v2 = point - m_vertices[0];

	const ScalarType d00 = v0.dot(v0);
	const ScalarType d01 = v0.dot(v1);
	const ScalarType d02 = v0.dot(v2);
	const ScalarType d11 = v1.dot(v1);
	const ScalarType d12 = v1.dot(v2);

	const ScalarType denominator = d00 * d11 - d01 * d01;

	Vector2 ret;
	ret[0] = (d11 * d02 - d01 * d12) / denominator;
	ret[1] = (d00 * d12 - d01 * d02) / denominator;

	return ret;
}


bool Face::isInside(const Vector2 & barycentricPoint)
{
	return
		barycentricPoint[0] >= 0 &&
		barycentricPoint[1] >= 0 &&
		barycentricPoint[0] + barycentricPoint[1] <= 1;
}



