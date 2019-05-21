#pragma once

#include <optional>
#include <iterator>
#include <array>

#include "Roi3DF_eigen.h"
#include "vec_template.h"

#include "CommonGeometricTypes.h"
#include "CoordinateTransform.h"

// I tried using WorldVector instead of Vector3 to continue the theme of 
// strong typing. This however was really detrimental to performance for
// intersection calculation.

class Face
{
public:
	using Polygon = std::array<Vector3, 3>;

	Face() = default;

	std::optional<Vector3> intersectLine(const Vector3 & point1, const Vector3 & point2) const;
	std::optional<Vector3> intersectInfiniteLine(const Vector3 & point, const Vector3 & dir) const;

	bool doesIntersectLine(const Vector3 & point1, const Vector3 & point2) const;
	bool doesIntersectInfiniteLine(const Vector3 & point, const Vector3 & dir) const;

	const Vector3 & getVertex(size_t i) const;
	Vector3 & getVertex(size_t i);

	const Polygon & getVertices() const { return m_vertices; }
	const Vector4 & getNormal() const { return m_normal; }

	auto begin() { return m_vertices.begin(); }
	auto begin() const { return m_vertices.begin(); }

	auto end() { return m_vertices.end(); }
	auto end() const { return m_vertices.end(); }

	auto cbegin() const { return m_vertices.cbegin(); }
	auto cend() const { return m_vertices.cend(); }

	template<typename It>
	static Face createFromGenericPolygon(It verticesBegin, It verticesEnd);

	template<typename It, typename SourceCoordinateSystem>
	static Face createFromTypedPolygon(It verticesBegin, It verticesEnd);

private:
	static Vector4 makeHesseNormal(const Polygon & p);
	bool intersectInfiniteLine_impl(const Vector3 & point, const Vector3 & dir, ScalarType * dist, Vector3 * ret) const;
	bool intersectLine_impl(const Vector3 & point1, const Vector3 & point2, ScalarType * dist, Vector3 * ret) const;
	

private:
	Vector2 getBarycentricCoords(const Vector3 & point) const;
	static bool isInside(const Vector2 & barycentricPoint);

	Polygon m_vertices;
	Vector4 m_normal;
};


template<typename It>
Face Face::createFromGenericPolygon(const It verticesBegin, const It verticesEnd)
{
	assert(std::distance(verticesBegin, verticesEnd) == 3); // A Face consists of 3 Vertices

	Face f;

	size_t idx = 0;
	for (It iter = verticesBegin; iter != verticesEnd; ++iter)
	{
		const auto vertex = Vector3(x(*iter), y(*iter), z(*iter));
		f.m_vertices[idx++] = vertex;
	}

	f.m_normal = makeHesseNormal(f.m_vertices);

	return f;
}


template <typename It, typename SourceCoordinateSystem>
Face Face::createFromTypedPolygon(It verticesBegin, It verticesEnd)
{
	assert(std::distance(verticesBegin, verticesEnd) == 3); // A Face consists of 3 Vertices

	Face f;

	size_t idx = 0;
	for (It iter = verticesBegin; iter != verticesEnd; ++iter)
	{
		const auto sourceVertex = make_named<SourceCoordinateSystem::NamedVector>(x(*iter), y(*iter), z(*iter));
		const auto vertex = convertTo<WorldCoordinateSystem>(sourceVertex);
		f.m_vertices[idx++] = vertex.get();
	}

	f.m_normal = makeHesseNormal(f.m_vertices);

	return f;
}

