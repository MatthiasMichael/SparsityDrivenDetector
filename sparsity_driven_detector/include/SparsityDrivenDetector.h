#pragma once

#include <memory>

#include "Environment.h"
#include "GridPoints.h"
#include "Dictionary.h"
#include "OptimizationProblemFactory.h"


class SparsityDrivenDetector
{
public:
	using ImageSize = Dictionary::ImageSize;
	using Parameters = OptimizationProblem::Parameters;

	SparsityDrivenDetector(const Dictionary & dictionary, std::unique_ptr<OptimizationProblemFactory> && factory);
	SparsityDrivenDetector(const Dictionary & dictionary, std::unique_ptr<OptimizationProblemFactory> && factory, const Parameters & parameters);

	void resetOptimizationProblem(const Parameters & parameters);
	void resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory);
	void resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory, const Parameters & parameters);

	std::vector<SolutionActor> processFrame(const std::vector<cv::Mat> & images);
	std::vector<SolutionActor> getSolution() const;

	std::vector<cv::Mat> getReconstructedFrames() const;

	void save(const std::string & filename) const;

	const Dictionary & getDictionary() const { return m_dictionary; }
	const GridPoints & getGrid() const { return m_dictionary.getGrid(); }
	const IOptimizationProblem & getIOptimizationProblem() const { return *m_optimizationProblem; }
	const OptimizationProblem & tryGetOptimizationProblem() const;

	static SparsityDrivenDetector load(const std::string & filename);

	friend bool operator==(const SparsityDrivenDetector & lhs, const SparsityDrivenDetector & rhs);

private:
	// Only to be used by load
	SparsityDrivenDetector(
		const std::string & filenameDictionary, 
		const std::string & filenameFactory, 
		const std::string & filenameOptimizationProblem); 

private:
	Dictionary m_dictionary;

	std::unique_ptr<OptimizationProblemFactory> m_optimizationFactory;
	std::unique_ptr<IOptimizationProblem> m_optimizationProblem;

	std::vector<int> m_currentSolution;
};
