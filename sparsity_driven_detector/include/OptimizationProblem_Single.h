#pragma once

#include "OptimizationProblem.h"
#include "Dictionary.h"


class OptimizationProblem_Single : public OptimizationProblem
{
public:
	friend class OptimizationProblemFactory_Single; // Can call default ctor

	OptimizationProblem_Single(const Dictionary & dict, const Parameters & parameters);

	OptimizationProblem_Single(OptimizationProblem_Single &&) noexcept(noexcept(IloEnv{}));

	OptimizationProblem_Single & operator=(OptimizationProblem_Single &&) noexcept(noexcept(IloEnv{}));

	std::vector<int> solve_dense(const std::vector<cv::Mat> & targets);
	std::vector<int> solve_withChecks(const std::vector<cv::Mat> & targets);

private:
	void fillMatrixRows(const std::vector<Dictionary::Word> & d, ScalarType factorDict, ScalarType factorI, std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange);

	OptimizationProblem_Single(); //< Only for loading

	void applyChanges(const ChangedEntries & entries) override;

	std::vector<std::vector<IloExpr> *> getExpressions() override;
	std::vector<std::vector<IloRange> *> getMatrixRows() override;

	std::vector<const std::vector<IloExpr> *> getExpressions() const override;
	std::vector<const std::vector<IloRange> *> getMatrixRows() const override;

	std::unique_ptr<OptimizationProblemFactory> getFactory() const override;

	IloInt getNumExpressions() const override { return 2 * M; }
	IloInt getNumVars() const override { return N + M; }
	IloInt getNumTemplates() const override { return 1; }
	
protected:
	// Begin serializable info

	std::vector<IloExpr> m_expr_neg;
	std::vector<IloExpr> m_expr_pos;

	// End serializable info

	std::vector<IloRange> m_matrixRowNeg; //< m_expr_neg <= x
	std::vector<IloRange> m_matrixRowPos; //< m_expr_pos <= x

private:
	std::vector<float> m_lastTargetReconstructed; // Only used for checking in solve_withChecks
};