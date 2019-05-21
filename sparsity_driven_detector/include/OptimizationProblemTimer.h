#pragma once

#include "ApplicationTimer.h"

#include "OptimizationProblemDecorator.h"

class OptimizationProblemTimer : public OptimizationProblemDecorator
{
public:
	struct StringConstants;

	OptimizationProblemTimer(std::unique_ptr<IOptimizationProblem> && problem);

	std::vector<int> solve(const std::vector<cv::Mat> & targets) override;

protected:
	std::list<size_t> makeSparseTarget(const std::vector<cv::Mat> & target) const override;

	ChangedEntries getChangesToPreviousFrame(const std::list<size_t> & activeTargetPixelCurrentFrame) const override;
	
	void applyChanges(const ChangedEntries & entries) override;
	
	void solveCplex() override;
	
	std::vector<int> extractSolution() override;
};