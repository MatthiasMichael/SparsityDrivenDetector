#pragma once

#include "Configuration.h"
#include "Controller.h"

#include "ExperimentExporter.h"
#include "TimingExporter.h"
#include "ConfigurationExporter.h"


class Experiment
{
public:
	using ControllerFactory = std::function<Controller *(Context &&)>;

	struct StringConstants;

	Experiment(const Configuration & config, const std::string & outFolder, ControllerFactory c);

	void execute();

private:
	const std::string m_name;
	const std::string m_dbName;

	Controller * mep_controller;

	ExperimentExporter m_experimentExporter;
	TimingExporter m_timingExporter;
	ConfigurationExporter m_configurationExporter;
};
