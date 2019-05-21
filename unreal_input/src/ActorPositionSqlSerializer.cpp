#include "ActorPositionSqlSerializer.h"

#include <sstream>


int callback_actorPosition(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	if (argc != ActorPositionInfo::NumElements)
	{
		throw std::runtime_error("Invalid number of results for Actor Position!");
	}

	auto * instance = static_cast<ActorPositionSqlSerializer *>(instance_void);

	ActorPositionInfo info = { };

	for (int i = 0; i < ActorPositionInfo::NumElements; ++i)
	{
		std::stringstream ss;
		ss << argv[i];

		switch (i)
		{
		case 0: ss >> info.actor_id;
			break;
		case 1: ss >> info.framenumber;
			break;
		case 2: info.timestamp = ss.str(); // Stream Operator cuts off timestamp after date
			break;
		case 3: ss >> info.pos_x;
			break;
		case 4: ss >> info.pos_y;
			break;
		case 5: ss >> info.pos_z;
			break;
		case 6: ss >> info.yaw;
			break;
		case 7: ss >> info.pitch;
			break;
		case 8: ss >> info.roll;
			break;
		default: break;
		}
	}

	instance->addActorPositionInfo(info);

	return 0;
}


ActorPositionSqlSerializer::ActorPositionSqlSerializer(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void ActorPositionSqlSerializer::resetTables()
{
	const std::string queryCreateTable(
		"CREATE TABLE IF NOT EXISTS 'ActorPositions' ("
		"'actor_id' INT NOT NULL, "
		"'framenumber' INT NOT NULL, "
		"'timestamp' TEXT NOT NULL, "
		"'pos_x' REAL NOT NULL, "
		"'pos_y' REAL NOT NULL, "
		"'pos_z' REAL NOT NULL, "
		"'yaw' REAL NOT NULL, "
		"'pitch' REAL NOT NULL, "
		"'roll' REAL NOT NULL, "
		"PRIMARY KEY('actor_id', 'framenumber'), "
		"FOREIGN KEY('actor_id') REFERENCES ActorTypes(ID) );"
	);

	executeQuery(queryCreateTable);

	executeQuery("DELETE FROM ActorPositions;");
}


void ActorPositionSqlSerializer::write(const ActorPositionInfo & actorPositionInfo)
{
	std::stringstream ss;
	ss << "INSERT INTO ActorPositions(actor_id, framenumber, timestamp, pos_x, pos_y, pos_z, yaw, pitch, roll) VALUES (";
	ss << actorPositionInfo.actor_id << ", ";

	ss << actorPositionInfo.framenumber << ", ";
	ss << "\"" << actorPositionInfo.timestamp << "\", ";

	ss << actorPositionInfo.pos_x << ", ";
	ss << actorPositionInfo.pos_y << ", ";
	ss << actorPositionInfo.pos_z << ", ";

	ss << actorPositionInfo.yaw << ", ";
	ss << actorPositionInfo.pitch << ", ";
	ss << actorPositionInfo.roll << ");";

	executeQuery(ss.str());
}


std::vector<ActorPositionInfo> ActorPositionSqlSerializer::readAll()
{
	m_currentDbRead.clear();

	executeQuery("SELECT * FROM ActorPositions;", &callback_actorPosition, static_cast<void*>(this));

	return m_currentDbRead;
}


void ActorPositionSqlSerializer::addActorPositionInfo(const ActorPositionInfo & info)
{
	m_currentDbRead.push_back(info);
}
