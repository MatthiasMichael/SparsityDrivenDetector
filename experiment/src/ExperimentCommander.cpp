#include "ExperimentCommander.h"

#include "boost/filesystem.hpp"

#include "Experiment.h"


ExperimentCommander::ExperimentCommander(
	const std::string & experimentConfigFolder, const std::string & outFolder, ControllerFactory factory) :
	m_experimentFolder(experimentConfigFolder),
	m_outFolder(outFolder),
	m_factory(factory)
{
	using namespace boost::filesystem;

	const path out(m_outFolder);

	if (!is_directory(out))
	{
		create_directories(out);
	}
}


void ExperimentCommander::execute()
{
	using namespace boost::filesystem;

	for (directory_entry & e : directory_iterator(m_experimentFolder))
	{
		if (!is_regular_file(e))
		{
			continue;
		}

		Experiment exp(Configuration(e.path().string()), m_outFolder, m_factory);
		exp.execute();
	}

	emit finished();
}
