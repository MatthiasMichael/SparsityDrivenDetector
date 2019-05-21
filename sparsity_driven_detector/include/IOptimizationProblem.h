#pragma once

#include <list>
#include <vector>

#include "opencv2/imgproc.hpp"

// We want a superclass our decorators can inherit from that does not know
// anything about CPLEX and that does not hold any additional data members
// Also it might come in handy once we are using different Optimizers

// TODO: For simplicity this interface currently also describes the protected and private
// TODO: interface for CPLEX optimization problems to make them e.g. timeable with the 
// TODO: decorator pattern. This could be extracted into an extra class.
class IOptimizationProblem
{
public:
	IOptimizationProblem() = default;
	virtual ~IOptimizationProblem() = default;

	IOptimizationProblem(const IOptimizationProblem &) = delete;
	IOptimizationProblem(IOptimizationProblem &&) noexcept = default;

	IOptimizationProblem & operator=(const IOptimizationProblem &) = delete;
	IOptimizationProblem & operator=(IOptimizationProblem &&) noexcept = default;

	virtual std::vector<int> solve(const std::vector<cv::Mat> & targets) = 0;

	virtual void save(const std::string & filename) const = 0;

	virtual bool operator==(const IOptimizationProblem & rhs) const = 0;

protected:
	struct ChangedEntries { std::list<size_t> blackToWhite, whiteToBlack; };

	virtual std::list<size_t> makeSparseTarget(const std::vector<cv::Mat> & target) const = 0;

	virtual ChangedEntries getChangesToPreviousFrame(const std::list<size_t> & activeTargetPixelCurrentFrame) const = 0;

	virtual void applyChanges(const ChangedEntries & entries) = 0;

	virtual void solveCplex() = 0;

	virtual std::vector<int> extractSolution() = 0;
};