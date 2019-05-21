#include "ActorTypeSqlSerializer.h"

#include <sstream>
#include <iostream>


int callback_actorType(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	if (argc != ActorTypeInfo::NumElements)
	{
		throw std::runtime_error("Invalid number of results for Actor Type!");
	}

	auto * instance = static_cast<ActorTypeSqlSerializer *>(instance_void);

	ActorTypeInfo info = { };

	for (int i = 0; i < ActorTypeInfo::NumElements; ++i)
	{
		std::stringstream ss;
		ss << argv[i];

		switch (i)
		{
		case 0: ss >> info.id;
			break;
		case 1: ss >> info.name;
			break;
		case 2: ss >> info.size_x;
			break;
		case 3: ss >> info.size_y;
			break;
		case 4: ss >> info.size_z;
			break;
		default: break;
		}
	}

	instance->addActorTypeInfo(info);

	return 0;
}


ActorTypeSqlSerializer::ActorTypeSqlSerializer(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void ActorTypeSqlSerializer::resetTables()
{
	const std::string queryCreateTable(
		"CREATE TABLE IF NOT EXISTS 'ActorTypes' ("
		"'ID' INT NOT NULL UNIQUE PRIMARY KEY, "
		"'name' TEXT NOT NULL, "
		"'size_x' REAL NOT NULL, "
		"'size_y' REAL NOT NULL, "
		"'size_z' REAL NOT NULL);"
	);

	executeQuery(queryCreateTable);

	executeQuery("DELETE FROM ActorTypes;");
}


void ActorTypeSqlSerializer::write(const ActorTypeInfo & actorTypeInfo)
{
	std::stringstream ss;
	ss << "INSERT INTO ActorTypes (ID, name, size_x, size_y, size_z) VALUES (";

	ss << actorTypeInfo.id << ", ";

	ss << "\"" << actorTypeInfo.name << "\", ";

	ss << actorTypeInfo.size_x << ", ";
	ss << actorTypeInfo.size_y << ", ";
	ss << actorTypeInfo.size_z << ");";

	executeQuery(ss.str());
}


std::vector<ActorTypeInfo> ActorTypeSqlSerializer::readAll()
{
	m_currentDbRead.clear();

	executeQuery("SELECT * FROM ActorTypes;", &callback_actorType, static_cast<void*>(this));

	return m_currentDbRead;
}


void ActorTypeSqlSerializer::addActorTypeInfo(const ActorTypeInfo & info)
{
	m_currentDbRead.push_back(info);
}
