#pragma once

#include <QObject>

#include "SqlSerializer.h"

#include "Fusion.h"
#include "SparsityDrivenDetectorPostProcessing.h"

class ExperimentExporter : public QObject, SqlSerializer
{
	Q_OBJECT

public:
	struct StringConstants;

	ExperimentExporter(const std::string & dbName);

	void resetTables();

public slots:
	void begin();
	void commit();

	void newGroundTruth(const Frame & f);
	void newSolution(const Solution & s);
	void newMergedSolution(const MergedSolution & s);
	void newFusedSolution(const FusedSolution & s);
	void newFusedVolume(const FusedSolution & s);
};
