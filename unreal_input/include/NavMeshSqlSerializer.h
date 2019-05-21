#pragma once

#include <string>

#include "vec_template.h"

#include "SqlSerializer.h"

// Simple struct to hold the vertex coordinates for writing and reading so that we do not
// have to rely on the presence of a particular algebra library when reading or writing the m_map
struct NavMeshVertex { float x; float y; float z; };
using NavMeshPolygon = std::vector<NavMeshVertex>;

template<>
inline float x(const NavMeshVertex & v)
{
	return v.x;
}

template<>
inline float y(const NavMeshVertex & v)
{
	return v.y;
}

template<>
inline float z(const NavMeshVertex & v)
{
	return v.z;
}

template<>
inline NavMeshVertex make_vec3D(float x, float y, float z)
{
	return { x, y, z };
}

template<>
struct is_vector_3D<NavMeshVertex>
{
	using type = std::true_type;
	static constexpr bool value = true;
};


class NavMeshSqlSerializer : private SqlSerializer
{
public:
	NavMeshSqlSerializer(const std::string & dbName);

	void resetTables();
	void writeVertices(const std::vector<NavMeshPolygon> & navMeshPolygons);
	std::vector<NavMeshPolygon> readAll();

	friend int callback_navMesh(void *, int argc, char ** argv, char ** columnNames);
	friend int callback_navMeshId(void *, int argc, char ** argv, char ** columnNames);

private:
	void addNavMeshVertex(const NavMeshVertex & info);
	void addNavMeshPolygonId(int id);

	std::vector<NavMeshPolygon> m_currentDbRead;
	NavMeshPolygon m_currentPolygon;

	std::vector<int> m_polygonIds;
};
