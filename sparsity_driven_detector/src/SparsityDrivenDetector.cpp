#include "SparsityDrivenDetector.h"

#include "serialization_helper.h"
#include "conversion_helper.h"


SparsityDrivenDetector::SparsityDrivenDetector(
	const std::string & filenameDictionary, 
	const std::string & filenameFactory,
	const std::string & filenameOptimizationProblem) :
	m_dictionary(Dictionary::load(filenameDictionary)),
	m_optimizationFactory(OptimizationProblemFactory::load(filenameFactory)),
	m_optimizationProblem(OptimizationProblem::load(filenameOptimizationProblem))
{
	// empty
}


SparsityDrivenDetector::SparsityDrivenDetector(
	const Dictionary & dictionary, 
	std::unique_ptr<OptimizationProblemFactory> && factory) :
	m_dictionary(dictionary),
	m_optimizationFactory(std::move(factory)),
	m_optimizationProblem(m_optimizationFactory->createProblem(m_dictionary, m_optimizationFactory->getDefaultParameters()))
{
	// empty
	// No constructor delegation since we cannot assume that
	// SparsityDrivenDetector(dictionary, std::move(factory), factory->getDefaultParameters())
	// would be evaluated in the correct oder, so that getDefaultParameters() isn't called on an empty pointer
}


SparsityDrivenDetector::SparsityDrivenDetector(
	const Dictionary & dictionary, 
	std::unique_ptr<OptimizationProblemFactory> && factory, 
	const Parameters & parameters
) :
	m_dictionary(dictionary),
	m_optimizationFactory(std::move(factory)),
	m_optimizationProblem(m_optimizationFactory->createProblem(m_dictionary, parameters))
{
	// empty
}


void SparsityDrivenDetector::resetOptimizationProblem(const Parameters & parameters)
{
	m_optimizationProblem = m_optimizationFactory->createProblem(m_dictionary, parameters);
}


void SparsityDrivenDetector::resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory)
{
	const auto p = factory->getDefaultParameters();
	resetOptimizationProblem(std::move(factory), p);
}


void SparsityDrivenDetector::resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory,
	const Parameters & parameters)
{
	m_optimizationFactory = std::move(factory);
	resetOptimizationProblem(parameters);
}


std::vector<SolutionActor> SparsityDrivenDetector::processFrame(const std::vector<cv::Mat> & images)
{
	m_currentSolution = m_optimizationProblem->solve(images);

	return getSolution();	
}


std::vector<SolutionActor> SparsityDrivenDetector::getSolution() const
{
	auto actors = std::vector<SolutionActor>();

	for(auto i : m_currentSolution)
	{
		actors.push_back(m_dictionary.getSolution(i));
	}

	return actors;
}


std::vector<cv::Mat> SparsityDrivenDetector::getReconstructedFrames() const
{
	return m_dictionary.getReconstructedFrames(m_currentSolution);
}


void SparsityDrivenDetector::save(const std::string & filename) const
{
	using namespace boost::filesystem;

	const auto temp = makeTempDir(path(filename).parent_path());
	
	m_dictionary.save((temp / "dictionary.zip").string());

	m_optimizationFactory->save((temp / "factory.txt").string());
	m_optimizationProblem->save((temp / "optimizationProblem.zip").string());

	zipDir(temp, filename);

	remove_all(temp);
}


const OptimizationProblem & SparsityDrivenDetector::tryGetOptimizationProblem() const
{
	return toOptimizationProblem(*m_optimizationProblem);
}


SparsityDrivenDetector SparsityDrivenDetector::load(const std::string & filename)
{
	using namespace boost::filesystem;

	const auto temp = makeTempDir(path(filename).parent_path());

	unzipDir(filename, temp);

	return SparsityDrivenDetector(
		(temp / "dictionary.zip").string(),
		(temp / "factory.txt").string(),
		(temp / "optimizationProblem.zip").string()
	);
}


bool operator==(const SparsityDrivenDetector & lhs, const SparsityDrivenDetector & rhs)
{
	return
		lhs.m_dictionary == rhs.m_dictionary &&
		*(lhs.m_optimizationFactory) == *(rhs.m_optimizationFactory) &&
		*(lhs.m_optimizationProblem) == *(rhs.m_optimizationProblem);
}
