#pragma once

#include "Face.h"

class Mesh
{
public:
	Mesh() = default;

	std::vector<Face> & getFaces() { return m_faces;  }
	const std::vector<Face> & getFaces() const { return m_faces; }

	const Roi3DF & getBoundingBox() const { return m_boundingBox; }

	auto size() const { return m_faces.size(); }
	auto empty() const { return m_faces.empty(); }

	auto begin() { return m_faces.begin(); }
	auto begin() const { return m_faces.begin(); }

	auto end() { return m_faces.end(); }
	auto end() const { return m_faces.end(); }

	auto cbegin() const { return m_faces.cbegin(); }
	auto cend() const { return m_faces.cend(); }

	template<typename It>
	static Mesh fromGenericPolygonVector(It begin, It end);

	template<typename It, typename SourceCoordinateSystem>
	static Mesh fromTypedPolygonVector(It begin, It end);

private:
	static Roi3DF calcBounds(const std::vector<Face> & faces);

private:
	std::vector<Face> m_faces;

	Roi3DF m_boundingBox;
};


template <typename It>
Mesh Mesh::fromGenericPolygonVector(It begin, It end)
{
	Mesh ret;

	for(It it_polygon = begin; it_polygon != end ; ++it_polygon)
	{
		const auto & polygon = *it_polygon;

		if(polygon.size() != 3)
		{
			throw std::runtime_error("Polygon for Face can only have 3 vertices.");
		}

		ret.m_faces.push_back(Face::createFromGenericPolygon(polygon.begin(), polygon.end()));
	}

	ret.m_boundingBox = calcBounds(ret.m_faces);

	return ret;
}


template <typename It, typename SourceCoordinateSystem>
Mesh Mesh::fromTypedPolygonVector(It begin, It end)
{
	Mesh ret;

	for (It it_polygon = begin; it_polygon != end; ++it_polygon)
	{
		const auto & polygon = *it_polygon;

		if (polygon.size() != 3)
		{
			throw std::runtime_error("Polygon for Face can only have 3 vertices.");
		}

		using ItInner = decltype(polygon.begin());

		ret.m_faces.push_back(Face::createFromTypedPolygon<ItInner, SourceCoordinateSystem>(polygon.begin(), polygon.end()));
	}

	ret.m_boundingBox = calcBounds(ret.m_faces);

	return ret;
}


