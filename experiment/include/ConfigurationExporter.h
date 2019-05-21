#pragma once

#include "SqlSerializer.h"

#include "Configuration.h"

class ConfigurationExporter : public SqlSerializer
{
public:
	struct StringConstants;

	ConfigurationExporter(const std::string & dbName);

	void resetTables();

	void exportConfiguration(const Configuration & c);
};