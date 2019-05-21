#pragma once

#include "OptimizationProblemWriter.h"

#include "serialization_helper.h"


class OptimizationProblemMatrixWriter : public OptimizationProblemWriter
{
public:
	OptimizationProblemMatrixWriter(std::unique_ptr<IOptimizationProblem> && problem, const std::string & folder);

private:
	using OutStream = NoNewlineOutStream;

	void writeProblem(const std::vector<cv::Mat> & targets) override;
};
