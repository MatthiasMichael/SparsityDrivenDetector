#pragma once

#include <string>
#include <map>
#include <mutex>

#include "sqlite3.h"


class SqlSerializer
{
public:
	using Callback = int(*)(void *, int, char **, char **);

	SqlSerializer(const std::string & dbName);
	virtual ~SqlSerializer();

protected:
	void executeQuery(const std::string & query, Callback callback = nullptr, void * callback_first_arg = nullptr);

	sqlite3 * m_db;
	char * m_errorMessage;

	std::string m_dbName;

	static std::map<std::string, std::unique_ptr<std::mutex>> s_mutexToDbName;
	static std::mutex s_mapMutex;
};
