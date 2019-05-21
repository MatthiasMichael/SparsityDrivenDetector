#include "OptimizationProblemTimer.h"


struct OptimizationProblemTimer::StringConstants
{
	inline static const auto TP_SOLVE = "OptimizationProblem::Solve";

	inline static const auto TP_MAKE_SPARSE = "OptimizationProblem::MakeSparseTarget";
	inline static const auto TP_GET_CHANGES = "OptimizationProblem::GetChangesToPreviousFrame";
	inline static const auto TP_APPLY_CHANGES = "OptimizationProblem::ApplyChanges";
	inline static const auto TP_SOLVE_CPLEX = "OptimizationProblem::SolveCplex";
	inline static const auto TP_EXTRACT_SOLUTION = "OptimizationProblem::ExtractSolution";

};


OptimizationProblemTimer::OptimizationProblemTimer(std::unique_ptr<IOptimizationProblem> && problem) :
	OptimizationProblemDecorator(std::move(problem))
{
	// empty
}


std::vector<int> OptimizationProblemTimer::solve(const std::vector<cv::Mat> & targets)
{
	// The decorator pattern does not work for methods invoked from within OptimizationProblem
	// Therefore solve needs to be reimplemented so that all relevant methods are called on our 
	// decorator
	// Be careful when solve changes somehow

	AT_START(StringConstants::TP_SOLVE);

	auto activeTargetPixelCurrentFrame = makeSparseTarget(targets);
	const auto & changes = getChangesToPreviousFrame(activeTargetPixelCurrentFrame);

	setActiveTargetPixelLastFrame(std::move(activeTargetPixelCurrentFrame));

	applyChanges(changes);

	solveCplex();

	auto ret = extractSolution();

	AT_STOP(StringConstants::TP_SOLVE);

	return ret;
}


std::list<size_t> OptimizationProblemTimer::makeSparseTarget(const std::vector<cv::Mat> & target) const
{
	AT_START(StringConstants::TP_MAKE_SPARSE);

	const auto & ret = OptimizationProblemDecorator::makeSparseTarget(target);

	AT_STOP(StringConstants::TP_MAKE_SPARSE);

	return ret;
}


IOptimizationProblem::ChangedEntries OptimizationProblemTimer::getChangesToPreviousFrame(
	const std::list<size_t> & activeTargetPixelCurrentFrame) const
{
	AT_START(StringConstants::TP_GET_CHANGES);

	const auto & ret = OptimizationProblemDecorator::getChangesToPreviousFrame(activeTargetPixelCurrentFrame);

	AT_STOP(StringConstants::TP_GET_CHANGES);

	return ret;
}


void OptimizationProblemTimer::applyChanges(const ChangedEntries & entries)
{
	AT_START(StringConstants::TP_APPLY_CHANGES);

	OptimizationProblemDecorator::applyChanges(entries);

	AT_STOP(StringConstants::TP_APPLY_CHANGES);
}


void OptimizationProblemTimer::solveCplex()
{
	AT_START(StringConstants::TP_SOLVE_CPLEX);

	OptimizationProblemDecorator::solveCplex();

	AT_STOP(StringConstants::TP_SOLVE_CPLEX);
}


std::vector<int> OptimizationProblemTimer::extractSolution()
{
	AT_START(StringConstants::TP_EXTRACT_SOLUTION);

	const auto & ret = OptimizationProblemDecorator::extractSolution();

	AT_STOP(StringConstants::TP_EXTRACT_SOLUTION);

	return ret;
}
