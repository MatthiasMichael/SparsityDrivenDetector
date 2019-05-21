#include "Experiment.h"

#include "boost/filesystem.hpp"

#include "ApplicationTimer.h"

/**
 * Builds a filename in the form:
 * Experiment__scene__optim__init__alg__<templates>__imgX_imgY__gridX_gridY__voxelSize
 */
std::string nameFromConfig(const Configuration & c)
{
	using namespace boost::filesystem;

	// Assume: sfsParam do not change
	// Voxels have all identical side length

	std::stringstream ss;
	ss << "Experiment_";
	ss << "_" << path(c.getSceneFile()).stem().string() << "_";
	ss << "_(" << c.getGridParameters().distancePoints(0) << "_" << c.getGridParameters().distancePoints(1) << "_" << c.getGridParameters().groundPlaneHeight << ")_";
	ss << "_(" << c.getTargetImageSize()(0) << "_" << c.getTargetImageSize()(1) << ")_";
	ss << "_(";
	for (const auto & s : c.getTemplateFiles())
	{
		ss << "(" << path(s).stem().string() << ")";
	}
	ss << ")_";
	ss << "_(" << c.getOptimizationType() << "_" << c.getMaxMergingDistance() << ")_";
	ss << "_(" << c.getAdvancedInitialization() << "_" << c.getRootAlgorithm() << ")_";
	ss << "_(" << c.getComputeLocation() << "_ " << c.getVoxelSize()(0) << ")_";
	ss << "_(" << c.getSfsParameters().minSegmentation << "_" << c.getSfsParameters().maxClusterDistance << ")";

	return ss.str();
}


std::string buildDbFileName(const std::string & name, const std::string & outFolder)
{
	using namespace boost::filesystem;
	return (path(outFolder) / (name + ".db")).string();
}


struct Experiment::StringConstants
{
	inline static const auto TP_INIT = "Experiment::Initialization";
	inline static const auto TP_EXEC = "Experiment::Execute";
};


Experiment::Experiment(const Configuration & config, const std::string & outFolder, ControllerFactory factory) :
	m_name(nameFromConfig(config)),
	m_dbName(buildDbFileName(m_name, outFolder)),
	mep_controller(nullptr),
	m_experimentExporter(m_dbName),
	m_timingExporter(m_dbName),
	m_configurationExporter(m_dbName)
{
	std::cout << "Running Experiment:\n" << m_name << std::endl;

	m_configurationExporter.exportConfiguration(config);

	AT_START_SESSION(m_name);

	AT_SET_FRAME(0);

	AT_START(StringConstants::TP_INIT);
	mep_controller = factory(Context(config));
	AT_STOP(StringConstants::TP_INIT);

	QObject::connect(mep_controller, &Controller::beginProcessFrame, &m_experimentExporter, &ExperimentExporter::begin);
	QObject::connect(mep_controller, &Controller::endProcessFrame, &m_experimentExporter, &ExperimentExporter::commit);
	
	QObject::connect(mep_controller, &Controller::newGroundTruth, &m_experimentExporter, &ExperimentExporter::newGroundTruth);
	QObject::connect(mep_controller, &Controller::newSolution, &m_experimentExporter, &ExperimentExporter::newSolution);
	QObject::connect(mep_controller, &Controller::newMergedSolution, &m_experimentExporter, &ExperimentExporter::newMergedSolution);
	QObject::connect(mep_controller, &Controller::newFusedSolution, &m_experimentExporter, &ExperimentExporter::newFusedSolution);
	QObject::connect(mep_controller, &Controller::newFusedSolution, &m_experimentExporter, &ExperimentExporter::newFusedVolume);
}


void Experiment::execute()
{
	AT_SET_FRAME(0);
	AT_START(StringConstants::TP_EXEC);

	const auto maxFramenumber = mep_controller->getMaximumFramenumber().get();

	while(mep_controller->hasFrame())
	{
		mep_controller->nextFrame();

		const auto framenumber = mep_controller->getCurrentFramenumber().get();

		AT_SET_FRAME(framenumber);

		mep_controller->processFrame();

		std::cout << "\rFrame " << framenumber << " | " << maxFramenumber;
	}

	AT_SET_FRAME(0); // Otherwise timing will fail!
	const auto elapsedTime = AT_STOP_GET_TIME(StringConstants::TP_EXEC);

	m_timingExporter.writeCurrentSession();

	std::cout << "\r" << maxFramenumber << " Frames done, Finished experiment in " << std::setprecision(2) << std::fixed << elapsedTime / 1000.f << "s.\n" << std::endl;
}
