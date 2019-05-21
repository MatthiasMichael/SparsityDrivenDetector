#pragma once

#include <string>
#include <vector>

#include "vec_template.h"

#include "SqlSerializer.h"

// Simple struct to hold the vertex coordinates for writing and reading so that we do not
// have to rely on the presence of a particular algebra library when reading or writing the m_map
struct MapVertex { float x; float y; float z; };
using MapPolygon = std::vector<MapVertex>;

template<>
inline float x(const MapVertex & v)
{
	return v.x;
}

template<>
inline float y(const MapVertex & v)
{
	return v.y;
}

template<>
inline float z(const MapVertex & v)
{
	return v.z;
}

template<>
inline MapVertex make_vec3D(float x, float y, float z)
{
	return { x, y, z };
}

template<>
struct is_vector_3D<MapVertex>
{
	using type = std::true_type;
	static constexpr bool value = true;
};




class MapSqlSerializer : private SqlSerializer
{
public:
	MapSqlSerializer(const std::string & dbName);

	void resetTables();
	void write(const std::vector<MapPolygon> & vertVec);
	std::vector<MapPolygon> readAll();

	friend int callback_map(void *, int argc, char ** argv, char ** columnNames);

private:
	void addMapPolygon(const MapPolygon & info);

	std::vector<MapPolygon> m_currentDbRead;
};