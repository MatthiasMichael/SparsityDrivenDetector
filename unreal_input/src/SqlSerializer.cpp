#include "SqlSerializer.h"

#include <sstream>

std::map<std::string, std::unique_ptr<std::mutex>> SqlSerializer::s_mutexToDbName;
std::mutex SqlSerializer::s_mapMutex;


SqlSerializer::SqlSerializer(const std::string & dbName) : m_db(nullptr), m_errorMessage(nullptr), m_dbName(dbName)
{
	if (sqlite3_open(dbName.c_str(), &m_db))
	{
		std::stringstream ss;
		ss << "Can't open database: " << sqlite3_errmsg(m_db);
		sqlite3_close(m_db);
		m_db = nullptr;
		throw std::runtime_error(ss.str());
	}

	std::lock_guard<std::mutex> mapLock(s_mapMutex);

	if(s_mutexToDbName.find(dbName) == s_mutexToDbName.end())
	{
		s_mutexToDbName.emplace(m_dbName, std::make_unique<std::mutex>());
	}
}


SqlSerializer::~SqlSerializer()
{
	if(m_db)
	{
		std::lock_guard<std::mutex> lock(*s_mutexToDbName[m_dbName]);

		sqlite3_close(m_db);
	}
}


void SqlSerializer::executeQuery(const std::string & query, Callback callback /*= nullptr*/, void * callback_first_arg /*= nullptr*/)
{
	std::lock_guard<std::mutex> lock(*s_mutexToDbName[m_dbName]);

	if(sqlite3_exec(m_db, query.c_str(), callback, callback_first_arg, &m_errorMessage))
	{
		std::stringstream ss;
		ss << "Error during query execution.\nQuery: " << query << "\nError: " << m_errorMessage;
		sqlite3_free(m_errorMessage);
		throw std::runtime_error(ss.str());
	}
}