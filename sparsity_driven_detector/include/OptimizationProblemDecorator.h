#pragma once

#include "OptimizationProblem.h"
#include "OptimizationProblemFactoryCollection.h"


class OptimizationProblemDecorator : public IOptimizationProblem
{
public:
	OptimizationProblemDecorator(std::unique_ptr<IOptimizationProblem> && problem);

	const OptimizationProblem & getOptimizationProblem() const { return *m_problem; }

	std::vector<int> solve(const std::vector<cv::Mat> & targets) override;

	void save(const std::string & filename) const override;
	
	bool operator==(const IOptimizationProblem & rhs) const override;

protected: // Overriding Members
	std::list<size_t> makeSparseTarget(const std::vector<cv::Mat> & target) const override;
	
	ChangedEntries getChangesToPreviousFrame(const std::list<size_t> & activeTargetPixelCurrentFrame) const override;
	
	void applyChanges(const ChangedEntries & entries) override;
	
	void solveCplex() override;
	
	std::vector<int> extractSolution() override;

protected: // Exposing OptimizationProblem interna to Decorater subclasses
	auto & getEnv() const { return m_problem->env; }

	auto & getParameters() const { return m_problem->m_parameters; }

	auto & getN() const { return m_problem->N; }
	auto & getM() const { return m_problem->M; }

	auto & getF() const { return m_problem->F; }
	auto & getLb() const { return m_problem->Lb; }
	auto & getUb() const { return m_problem->Ub; }

	auto & getModel() const { return m_problem->m_persistentModel; }
	auto & getCplex() const { return m_problem->m_persistentCplex; }

	auto & getLastSolution() const { return m_problem->m_lastSolution; }

	auto getExpressions() const { return m_problem->getExpressions(); }
	auto getMatrixRows() const { return m_problem->getMatrixRows(); }

	std::unique_ptr<OptimizationProblemFactory> getFactory() const;

	void setActiveTargetPixelLastFrame(std::list<size_t> && activeTargetPixel);

	static std::string cplexVarNameFromId(IloInt id) { return OptimizationProblem::cplexVarNameFromId(id); }
	static int idFromCplexVarName(const std::string & varName) { return OptimizationProblem::idFromCplexVarName(varName); }

protected: // The problem
	std::unique_ptr<OptimizationProblem> m_problem;
};


template<typename TDecorator>
std::unique_ptr<OptimizationProblemFactory> makeFactory_withDecorator(const std::string & identifier)
{
	auto f = makeFactory(identifier);
	f->setDecoratorWrapper
	(
		[](std::unique_ptr<IOptimizationProblem> && p) -> std::unique_ptr<IOptimizationProblem>
		{
			return std::make_unique<TDecorator>(std::move(p));
		}
	);
	return f;
}