#pragma once

#include "OptimizationProblemDecorator.h"


class OptimizationProblemWriter : public OptimizationProblemDecorator
{
public:
	OptimizationProblemWriter(std::unique_ptr<IOptimizationProblem> && problem, const std::string & folder);

	std::vector<int> solve(const std::vector<cv::Mat> & targets) override;
	//std::vector<int> solve_dense(const std::vector<cv::Mat> & targets) override;

private:
	virtual void writeProblem(const std::vector<cv::Mat> & targets);

protected:
	int m_solveCounter;
	std::string m_saveFolder;
};
