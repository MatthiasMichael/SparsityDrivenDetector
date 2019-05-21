#include "MapSqlSerializer.h"

#include <sstream>


int callback_map(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	constexpr const int NumEntries = 9;

	if (argc != NumEntries)
	{
		throw std::runtime_error("Invalid number of results for Actor Position!");
	}

	auto * instance = static_cast<MapSqlSerializer *>(instance_void);

	MapPolygon p(3, MapVertex{ 0, 0, 0 });

	for (int i = 0; i < NumEntries; ++i)
	{
		std::stringstream ss;
		ss << argv[i];

		switch (i)
		{
		case 0: ss >> p[0].x;
			break;
		case 1: ss >> p[0].y;
			break;
		case 2: ss >> p[0].z;
			break;
		case 3: ss >> p[1].x;
			break;
		case 4: ss >> p[1].y;
			break;
		case 5: ss >> p[1].z;
			break;
		case 6: ss >> p[2].x;
			break;
		case 7: ss >> p[2].y;
			break;
		case 8: ss >> p[2].z;
			break;
		default: break;
		}
	}

	instance->addMapPolygon(p);

	return 0;
}


MapSqlSerializer::MapSqlSerializer(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void MapSqlSerializer::resetTables()
{
	const std::string queryFace("CREATE TABLE IF NOT EXISTS 'l2_face' ("
		"'ID' bigint DEFAULT(NULL) NOT NULL UNIQUE PRIMARY KEY, "
		"'L2_Point_C' bigint DEFAULT(NULL) NOT NULL, "
		"'L2_Point_B' bigint DEFAULT(NULL) NOT NULL, "
		"'L2_Point_A' bigint DEFAULT(NULL) NOT NULL, "
		"'Type' bigint DEFAULT(NULL) NULL, "
		"'Subtype' bigint DEFAULT(NULL) NULL, "
		"'Accuracy' bigint DEFAULT(NULL) NULL );");

	executeQuery(queryFace);

	const std::string queryPoint("CREATE TABLE IF NOT EXISTS 'l2_point' ("
		"'ID' bigint NOT NULL, "
		"'Pos_X' double precision NOT NULL, "
		"'Pos_Y' double precision NOT NULL, "
		"'Pos_Z' double precision NOT NULL );");

	executeQuery(queryPoint);

	executeQuery("DELETE FROM l2_face;");

	executeQuery("DELETE FROM l2_point;");
}


void MapSqlSerializer::write(const std::vector<MapPolygon> & polygons)
{
	int pointIdx = 1;
	int faceIdx = 1;

	// Builder Pattern to minimize the number of insert queries that have to be performed
	class PointWriter
	{
		std::string header = "INSERT INTO l2_point (ID, Pos_X, Pos_Y, Pos_Z) ";
		bool selectInit = false;
		std::stringstream ss;
		int selectCounter = 0;
		const int maxSelectCounter = 200;

	public:
		std::string finalizeQuery()
		{
			ss << ";";
			std::string s = ss.str();
			ss.str("");
			selectCounter = 0;
			selectInit = false;
			return s;
		}


		bool hasQuery() const { return !ss.str().empty(); }


		bool addPoint(const MapVertex & v, int pointIdx)
		{
			if (!hasQuery())
			{
				ss << header;
				selectCounter = 0;
			}

			if (!selectInit)
			{
				ss << "SELECT " << pointIdx << " AS ID, ";
				ss << v.x << " AS Pos_X, ";
				ss << v.y << " AS Pos_Y, ";
				ss << v.z << " AS Pos_Z ";
				selectInit = true;
			}
			else
			{
				ss << "UNION ALL SELECT ";
				ss << pointIdx << ", ";
				ss << v.x << ", ";
				ss << v.y << ", ";
				ss << v.z << " ";
			}
			return ++selectCounter >= maxSelectCounter;
		}
	} pointWriter;

	for (auto & f : polygons)
	{
		for (auto & v : f)
		{
			if (pointWriter.addPoint(v, pointIdx++))
			{
				executeQuery(pointWriter.finalizeQuery());
			}
		}
	}

	if (pointWriter.hasQuery())
	{
		executeQuery(pointWriter.finalizeQuery());
	}

	class VertexWriter
	{
		std::string header = "INSERT INTO l2_face (ID, L2_Point_C, L2_Point_B, L2_Point_A, Type, Subtype, Accuracy) ";
		bool selectInit = false;
		std::stringstream ss;
		int selectCounter = 0;
		const int maxSelectCounter = 200;

	public:
		std::string finalizeQuery()
		{
			ss << ";";
			std::string s = ss.str();
			ss.str("");
			selectCounter = 0;
			selectInit = false;
			return s;
		}


		bool hasQuery() const { return !ss.str().empty(); }


		bool addVertex(int faceIdx, int pointIdx)
		{
			if (!hasQuery())
			{
				ss << header;
				selectCounter = 0;
			}

			if (!selectInit)
			{
				ss << "SELECT " << faceIdx << " AS ID, ";
				ss << pointIdx + 2 << " AS L2_Point_C, ";
				ss << pointIdx + 1 << " AS L2_Point_B, ";
				ss << pointIdx + 0 << " AS L2_Point_A, ";

				ss << "3 AS Type, 2 AS Subtype, 1 AS Accuracy ";

				selectInit = true;
			}
			else
			{
				ss << "UNION ALL SELECT ";
				ss << faceIdx << ", ";
				ss << pointIdx + 2 << ", ";
				ss << pointIdx + 1 << ", ";
				ss << pointIdx + 0 << ", ";

				ss << "3, 2, 1 ";
			}

			return ++selectCounter >= maxSelectCounter;
		}
	} vertexWriter;

	pointIdx = 1;
	for (auto & f : polygons)
	{
		if (vertexWriter.addVertex(faceIdx, pointIdx))
		{
			executeQuery(vertexWriter.finalizeQuery());
		}
		pointIdx += 3;
		faceIdx += 1;
	}

	if (vertexWriter.hasQuery())
	{
		executeQuery(vertexWriter.finalizeQuery());
	}
}


std::vector<MapPolygon> MapSqlSerializer::readAll()
{
	m_currentDbRead.clear();

	executeQuery(
		"SELECT \
			PointA.Pos_X, \
			PointA.Pos_Y, \
			PointA.Pos_Z, \
			PointB.Pos_X, \
			PointB.Pos_Y, \
			PointB.Pos_Z, \
			PointC.Pos_X, \
			PointC.Pos_Y, \
			PointC.Pos_Z \
		FROM \
			l2_face, \
			(SELECT * FROM l2_point) AS PointA, \
			(SELECT * FROM l2_point) AS PointB, \
			(SELECT * FROM l2_point) AS PointC \
			WHERE \
				L2_Point_A = PointA.ID AND \
				L2_Point_B = PointB.ID AND \
				L2_Point_C = PointC.ID;",
		&callback_map, static_cast<void*>(this));

	return m_currentDbRead;
}


void MapSqlSerializer::addMapPolygon(const MapPolygon & polygon)
{
	m_currentDbRead.push_back(polygon);
}
