#pragma once

#include "OptimizationProblem.h"
#include "Dictionary.h"


class OptimizationProblem_MultiLayered : public OptimizationProblem
{
public:
	friend class OptimizationProblemFactory_MultiLayered;

	OptimizationProblem_MultiLayered(const Dictionary & dict, const Parameters & parameters);

	OptimizationProblem_MultiLayered(OptimizationProblem_MultiLayered &&) noexcept(noexcept(IloEnv{}));

	OptimizationProblem_MultiLayered & operator=(OptimizationProblem_MultiLayered &&) noexcept(noexcept(IloEnv{}));


private:
	void fillMatrixRows(const Dictionary & dict,
		ScalarType factorDict, ScalarType factorBeta, ScalarType factorI,
		std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange);

	void fillUniqueRows(const Dictionary & dict, ScalarType factor,
		std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange);

	OptimizationProblem_MultiLayered(); //< Should be only used by load

	void applyChanges(const ChangedEntries & entries) override;

	std::vector<std::vector<IloExpr> *> getExpressions() override;
	std::vector<std::vector<IloRange> *> getMatrixRows() override;

	std::vector<const std::vector<IloExpr> *> getExpressions() const override;
	std::vector<const std::vector<IloRange> *> getMatrixRows() const override;

	std::unique_ptr<OptimizationProblemFactory> getFactory() const override;

	IloInt getNumExpressions() const override { return 4 * M; }
	IloInt getNumVars() const override { return N + 2 * M; }
	IloInt getNumTemplates() const override;

protected:
	// Begin Serializable Info

	std::vector<IloExpr> m_expr_neg_dict_pos_b_pos_one_1;
	std::vector<IloExpr> m_expr_neg_dict_neg_b_pos_one_2;
	std::vector<IloExpr> m_expr_pos_dict_pos_b_neg_one_3;
	std::vector<IloExpr> m_expr_pos_dict_neg_b_neg_one_4;

	std::vector<IloExpr> m_expr_unique;

	// End Serializable Info

	std::vector<IloRange> m_matrixRow_neg_dict_pos_b_pos_one_1;
	std::vector<IloRange> m_matrixRow_neg_dict_neg_b_pos_one_2;
	std::vector<IloRange> m_matrixRow_pos_dict_pos_b_neg_one_3;
	std::vector<IloRange> m_matrixRow_pos_dict_neg_b_neg_one_4;

	std::vector<IloRange> m_matrixRow_unique;

};

