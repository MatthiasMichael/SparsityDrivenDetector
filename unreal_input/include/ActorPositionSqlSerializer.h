#pragma once

#include <string>
#include <vector>

#include "SqlSerializer.h"


struct ActorPositionInfo
{
	int actor_id;
	int framenumber;
	std::string timestamp;
	float pos_x, pos_y, pos_z;
	float yaw, pitch, roll;

	constexpr static const int NumElements = 9;
};

class ActorPositionSqlSerializer : private SqlSerializer
{
public:
	ActorPositionSqlSerializer(const std::string & dbName);

	void resetTables();
	void write(const ActorPositionInfo & actorPositionInfo);
	std::vector<ActorPositionInfo> readAll();

	friend int callback_actorPosition(void *, int argc, char ** argv, char ** columnNames);

private:
	void addActorPositionInfo(const ActorPositionInfo & info);

private:
	std::vector<ActorPositionInfo> m_currentDbRead;
};