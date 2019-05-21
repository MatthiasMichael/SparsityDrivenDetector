#include "NavMeshSqlSerializer.h"

#include <sstream>


int callback_navMeshId(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	constexpr int NumEntries = 1;

	if (argc != NumEntries)
	{
		throw std::runtime_error("Invalid number of results for NavMesh ID");
	}

	auto * instance = static_cast<NavMeshSqlSerializer*>(instance_void);

	std::stringstream ss;
	ss << argv[0];

	int id;
	ss >> id;

	instance->addNavMeshPolygonId(id);

	return 0;
}


int callback_navMesh(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	constexpr int NumEntries = 3;

	if (argc != NumEntries)
	{
		throw std::runtime_error("Invalid number of results for NavMesh Vertex");
	}

	auto * instance = static_cast<NavMeshSqlSerializer*>(instance_void);

	NavMeshVertex p;

	for (int i = 0; i < NumEntries; ++i)
	{
		std::stringstream ss;
		ss << argv[i];

		switch (i)
		{
		case 0: ss >> p.x;
			break;
		case 1: ss >> p.y;
			break;
		case 2: ss >> p.z;
			break;
		default: break;
		}
	}

	instance->addNavMeshVertex(p);

	return 0;
}


NavMeshSqlSerializer::NavMeshSqlSerializer(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void NavMeshSqlSerializer::resetTables()
{
	const std::string query("CREATE TABLE IF NOT EXISTS 'NavMeshPolygons' ("
		"'ID' bigint DEFAULT(NULL) NOT NULL UNIQUE PRIMARY KEY, "
		"'ID_Polygon' bigint DEFAULT(NULL) NOT NULL, "
		"'Pos_X' double precision NOT NULL, "
		"'Pos_Y' double precision NOT NULL, "
		"'Pos_Z' double precision NOT NULL );");

	executeQuery(query);
	executeQuery("DELETE FROM NavMeshPolygons;");
}


void NavMeshSqlSerializer::writeVertices(const std::vector<NavMeshPolygon> & meshPolygons)
{
	int globalIndex = 1;
	int polygonIndex = 1;

	bool selectInit = false;

	class PointWriter
	{
		std::string header = "INSERT INTO NavMeshPolygons (ID, ID_Polygon, Pos_X, Pos_Y, Pos_Z) ";
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


		bool addPoint(const NavMeshVertex & v, int globalIndex, int polygonIndex)
		{
			if (!hasQuery())
			{
				ss << header;
				selectCounter = 0;
			}

			if (!selectInit)
			{
				ss << "SELECT " << globalIndex << " AS ID, ";
				ss << polygonIndex << " AS ID_Polygon, ";
				ss << v.x << " AS Pos_X, ";
				ss << v.y << " AS Pos_Y, ";
				ss << v.z << " AS Pos_Z ";
				selectInit = true;
			}
			else
			{
				ss << "UNION ALL SELECT ";
				ss << globalIndex << ", ";
				ss << polygonIndex << ", ";
				ss << v.x << ", ";
				ss << v.y << ", ";
				ss << v.z << " ";
			}
			return ++selectCounter >= maxSelectCounter;
		}
	} pointWriter;

	for (auto & polygon : meshPolygons)
	{
		for (auto & v : polygon)
		{
			if (pointWriter.addPoint(v, globalIndex++, polygonIndex))
			{
				executeQuery(pointWriter.finalizeQuery());
			}
		}

		polygonIndex++;
	}

	if (pointWriter.hasQuery())
	{
		executeQuery(pointWriter.finalizeQuery());
	}
}


std::vector<NavMeshPolygon> NavMeshSqlSerializer::readAll()
{
	m_currentDbRead.clear();
	m_polygonIds.clear();

	executeQuery("SELECT DISTINCT ID_Polygon FROM NavMeshPolygons;", &callback_navMeshId, static_cast<void*>(this));

	for (int id : m_polygonIds)
	{
		m_currentPolygon.clear();
		std::stringstream ss;
		ss << "SELECT Pos_X, Pos_Y, Pos_Z FROM NavMeshPolygons WHERE ID_Polygon == " << id << ";";

		executeQuery(ss.str(), &callback_navMesh, static_cast<void*>(this));

		if(m_currentPolygon.size() != 3)
		{
			throw std::runtime_error("Nav Mesh polygons can only have exactly 3 vertices.");
		}

		m_currentDbRead.push_back(m_currentPolygon);
	}

	return m_currentDbRead;
}


void NavMeshSqlSerializer::addNavMeshPolygonId(int id)
{
	m_polygonIds.push_back(id);
}


void NavMeshSqlSerializer::addNavMeshVertex(const NavMeshVertex & info)
{
	m_currentPolygon.push_back(info);
}
