#pragma once

#include <string>
#include <vector>

#include "SqlSerializer.h"

struct ActorTypeInfo
{
	int id;
	std::string name;
	float size_x, size_y, size_z;

	constexpr static const int NumElements = 5;
};

class ActorTypeSqlSerializer : private SqlSerializer
{
public:
	ActorTypeSqlSerializer(const std::string & dbName);

	void resetTables();
	void write(const ActorTypeInfo & actorTypeInfo);
	std::vector<ActorTypeInfo> readAll();

	friend int callback_actorType(void *, int argc, char ** argv, char ** columnNames);

private:
	void addActorTypeInfo(const ActorTypeInfo & info);

	std::vector<ActorTypeInfo> m_currentDbRead;
};