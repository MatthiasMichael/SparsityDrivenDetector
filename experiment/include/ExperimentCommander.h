#pragma once

#include <QObject>

#include "Experiment.h"


class ExperimentCommander : public QObject
{
	Q_OBJECT
public:
	using ControllerFactory = Experiment::ControllerFactory;

	ExperimentCommander(
		const std::string & experimentConfigFolder, 
		const std::string & outFolder,
		ControllerFactory factory);

public slots:
	void execute();

signals:
	void finished();

private:
	std::string m_experimentFolder;
	std::string m_outFolder;

	ControllerFactory m_factory;
};