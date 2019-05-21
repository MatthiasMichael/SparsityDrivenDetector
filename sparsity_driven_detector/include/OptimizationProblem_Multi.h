#pragma once

#include "OptimizationProblem.h"
#include "Dictionary.h"


class OptimizationProblem_Multi : public OptimizationProblem
{
public:
	friend class OptimizationProblemFactory_Multi; // Can call default ctor

	OptimizationProblem_Multi(const Dictionary & dict, const Parameters & parameters);

	OptimizationProblem_Multi(OptimizationProblem_Multi &&) noexcept(noexcept(IloEnv{}));

	OptimizationProblem_Multi & operator=(OptimizationProblem_Multi &&) noexcept(noexcept(IloEnv{}));

private:
	void fillMatrixRows(const Dictionary & dict, ScalarType factorDict, ScalarType factorI, std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange);
	void fillUniqueRows(const Dictionary & dict, ScalarType factor, std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange);

	OptimizationProblem_Multi(); //< Only for loading

	void applyChanges(const ChangedEntries & entries) override;

	std::vector<std::vector<IloExpr> *> getExpressions() override;
	std::vector<std::vector<IloRange> *> getMatrixRows() override;

	std::vector<const std::vector<IloExpr> *> getExpressions() const override;
	std::vector<const std::vector<IloRange> *> getMatrixRows() const override;

	std::unique_ptr<OptimizationProblemFactory> getFactory() const override;

	IloInt getNumExpressions() const override { return 2 * M + N; }
	IloInt getNumVars() const override { return getNumTemplates() * N + M; }
	IloInt getNumTemplates() const override;

protected:
	// Begin serializable info

	std::vector<IloExpr> m_expr_neg;
	std::vector<IloExpr> m_expr_pos;

	std::vector<IloExpr> m_expr_unique;

	// End serializable info

	std::vector<IloRange> m_matrixRowNeg; //< m_expr_neg <= x
	std::vector<IloRange> m_matrixRowPos; //< m_expr_pos <= x

	std::vector<IloRange> m_matrixRowUnique; //< m_expr_unique <= x
};