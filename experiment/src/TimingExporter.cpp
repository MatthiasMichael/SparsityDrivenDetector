#include "TimingExporter.h"
#include "ApplicationTimer.h"

#include <sstream>


TimingExporter::TimingExporter(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void TimingExporter::writeCurrentSession()
{
	for(const auto stream : AT_SESSION().getStreams())
	{
		writeTimingStream(*stream);
	}
}


void TimingExporter::resetTable(const QString tableName)
{
	const QString queryCreateTable(
		"CREATE TABLE IF NOT EXISTS '%1' ("
		"'ID_FRAME' INT NOT NULL, "
		"'ID_TIMING' INT NOT NULL, "
		"'start' REAL NOT NULL, "
		"'stop' REAL NOT NULL, "
		"'duration' REAL NOT NULL, "
		"PRIMARY KEY('ID_FRAME', 'ID_TIMING'));"
	);

	executeQuery(queryCreateTable.arg(tableName).toStdString());

	const QString queryDelete("DELETE FROM '%1';");

	executeQuery(queryDelete.arg(tableName).toStdString());
}

void TimingExporter::writeTimingStream(const TimingStream & s)
{
	const auto tableName = s_tablePrefix + s.getName().c_str();
	
	resetTable(tableName);

	int pointIdx = 1;
	int faceIdx = 1;

	// Builder Pattern to minimize the number of insert queries that have to be performed
	class TimingWriter
	{
		std::string header;
		std::stringstream ss;
		int insertCounter = 0;
		bool needsComma = false;
		const int maxInsertCounter = 200;

	public:
		TimingWriter(const QString tableName) : 
			header(QString("INSERT INTO '%1' (ID_FRAME, ID_TIMING, start, stop, duration) VALUES").arg(tableName).toStdString())
		{
			// empty
		}

		std::string finalizeQuery()
		{
			ss << ";";
			std::string s = ss.str();
			ss.str("");
			insertCounter = 0;
			needsComma = false;
			return s;
		}

		bool hasQuery() const { return !ss.str().empty(); }

		bool addTiming(const TimingEntry & t, int id)
		{
			if (!hasQuery())
			{
				ss << header;
				insertCounter = 0;
				needsComma = false;
			}

			if(needsComma)
			{
				ss << ", ";
			}

			ss << "(" << t.getFramenumber() << ", ";
			ss << id << ", ";
			ss << std::chrono::duration<double, std::milli>(t.getStart() - AT_GET_SESSION_START()).count() << ", ";
			ss << std::chrono::duration<double, std::milli>(t.getStop() - AT_GET_SESSION_START()).count() << ", ";
			ss << t.duration() << ")";

			needsComma = true;
			
			return ++insertCounter >= maxInsertCounter;
		}
	} timingWriter(tableName);

	int previousFramenumber = -1;
	int timingId = 0;

	for(const auto & timing : s.getTimings())
	{
		timingId += (timing.getFramenumber() == previousFramenumber) ? 1 : -timingId;
		previousFramenumber = timing.getFramenumber();

		if(timingWriter.addTiming(timing, timingId))
		{
			executeQuery(timingWriter.finalizeQuery());
		}
	}

	if(timingWriter.hasQuery())
	{
		executeQuery(timingWriter.finalizeQuery());
	}
}
