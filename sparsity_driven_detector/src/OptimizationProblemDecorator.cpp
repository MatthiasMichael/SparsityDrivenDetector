#include "OptimizationProblemDecorator.h"

#include "OptimizationProblemFactory.h"


OptimizationProblemDecorator::OptimizationProblemDecorator(std::unique_ptr<IOptimizationProblem> && problem) :
	m_problem(dynamic_cast<OptimizationProblem *>(problem.release()))
{
	if (!m_problem)
	{
		throw std::runtime_error("Decorator only applicable to class OptimizationProblem.");
	}
}


std::vector<int> OptimizationProblemDecorator::solve(const std::vector<cv::Mat> & targets)
{
	return m_problem->solve(targets);
}


void OptimizationProblemDecorator::save(const std::string & filename) const
{
	m_problem->save(filename);
}


bool OptimizationProblemDecorator::operator==(const IOptimizationProblem & rhs) const
{
	return m_problem->operator==(rhs);
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemDecorator::getFactory() const
{
	return m_problem->getFactory();
}


void OptimizationProblemDecorator::setActiveTargetPixelLastFrame(std::list<size_t> && activeTargetPixel)
{
	m_problem->m_activeTargetPixelLastFrame = std::move(activeTargetPixel);
}


std::list<size_t> OptimizationProblemDecorator::makeSparseTarget(const std::vector<cv::Mat> & target) const
{
	return m_problem->makeSparseTarget(target);
}


IOptimizationProblem::ChangedEntries OptimizationProblemDecorator::getChangesToPreviousFrame(
	const std::list<size_t> & activeTargetPixelCurrentFrame) const
{
	return m_problem->getChangesToPreviousFrame(activeTargetPixelCurrentFrame);
}


void OptimizationProblemDecorator::applyChanges(const ChangedEntries & entries)
{
	return m_problem->applyChanges(entries);
}


void OptimizationProblemDecorator::solveCplex()
{
	return m_problem->solveCplex();
}


std::vector<int> OptimizationProblemDecorator::extractSolution()
{
	return m_problem->extractSolution();
}
