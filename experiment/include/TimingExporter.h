#pragma once

#include <QString>

#include "SqlSerializer.h"

#include "TimingStream.h"


class TimingExporter : public SqlSerializer
{
public:
	TimingExporter(const std::string & dbName);

	void writeCurrentSession();

private:
	void resetTable(const QString tableName);
	void writeTimingStream(const TimingStream & s);

private:
	inline static const QString s_tablePrefix = "Timing_";
};



