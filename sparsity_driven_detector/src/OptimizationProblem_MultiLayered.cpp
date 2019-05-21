#include "OptimizationProblem_MultiLayered.h"

#include "OptimizationProblemFactory_MultiLayered.h"


OptimizationProblem_MultiLayered::OptimizationProblem_MultiLayered() : OptimizationProblem()
{
	// empty
}


OptimizationProblem_MultiLayered::OptimizationProblem_MultiLayered(
	const Dictionary & dict,
	const Parameters & parameters) :
	OptimizationProblem()
{
	assert(parameters.beta != 0); // Otherwise the solution should be empty anyways

	m_parameters = parameters;

	N = dict.getNumEntries();
	M = dict.getNumPixel();

	const int k = dict.getNumTemplates();

	F = IloNumArray(env, k * N + 2 * M);
	Lb = IloNumArray(env, k * N + 2 * M);
	Ub = IloNumArray(env, k * N + 2 * M);

	m_lastSolution = IloNumArray(env, k * N + 2 * M);

	for (IloInt i = 0; i < k * N; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 1.f;
	}

	for (IloInt i = k * N; i < k * N + M; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 0.f;
	}

	for (IloInt i = k * N + M; i < k * N + 2 * M; ++i)
	{
		F[i] = 1.f;

		Lb[i] = 0.f;
		Ub[i] = 999.f; // IloInfinity; // Technically it should be infinity. However that crashes in operator>>
	}

	makeOptimizationTarget();

	const auto b = parameters.beta;

	fillMatrixRows(dict, -1, (b + 1), -1, &m_expr_neg_dict_pos_b_pos_one_1, &m_matrixRow_neg_dict_pos_b_pos_one_1);
	fillMatrixRows(dict, -1, (-b + 1), -1, &m_expr_neg_dict_neg_b_pos_one_2, &m_matrixRow_neg_dict_neg_b_pos_one_2);
	fillMatrixRows(dict, 1, (b - 1), -1, &m_expr_pos_dict_pos_b_neg_one_3, &m_matrixRow_pos_dict_pos_b_neg_one_3);
	fillMatrixRows(dict, 1, (-b - 1), -1, &m_expr_pos_dict_neg_b_neg_one_4, &m_matrixRow_pos_dict_neg_b_neg_one_4);

	fillUniqueRows(dict, 1, &m_expr_unique, &m_matrixRow_unique);
}


OptimizationProblem_MultiLayered::OptimizationProblem_MultiLayered(
	OptimizationProblem_MultiLayered && other) noexcept(noexcept(IloEnv{ })) :
	OptimizationProblem(std::move(other)),
	m_expr_neg_dict_pos_b_pos_one_1(other.m_expr_neg_dict_pos_b_pos_one_1),
	m_expr_neg_dict_neg_b_pos_one_2(other.m_expr_neg_dict_neg_b_pos_one_2),
	m_expr_pos_dict_pos_b_neg_one_3(other.m_expr_pos_dict_pos_b_neg_one_3),
	m_expr_pos_dict_neg_b_neg_one_4(other.m_expr_pos_dict_neg_b_neg_one_4),
	m_expr_unique(other.m_expr_unique),
	m_matrixRow_neg_dict_pos_b_pos_one_1(other.m_matrixRow_neg_dict_pos_b_pos_one_1),
	m_matrixRow_neg_dict_neg_b_pos_one_2(other.m_matrixRow_neg_dict_neg_b_pos_one_2),
	m_matrixRow_pos_dict_pos_b_neg_one_3(other.m_matrixRow_pos_dict_pos_b_neg_one_3),
	m_matrixRow_pos_dict_neg_b_neg_one_4(other.m_matrixRow_pos_dict_neg_b_neg_one_4),
	m_matrixRow_unique(other.m_matrixRow_unique)
{
	// empty
}


OptimizationProblem_MultiLayered & OptimizationProblem_MultiLayered::operator=(
	OptimizationProblem_MultiLayered && other) noexcept(noexcept(IloEnv{ }))
{
	static_cast<OptimizationProblem&>(*this) = std::move(static_cast<OptimizationProblem&>(other));

	m_expr_neg_dict_pos_b_pos_one_1 = other.m_expr_neg_dict_pos_b_pos_one_1;
	m_expr_neg_dict_neg_b_pos_one_2 = other.m_expr_neg_dict_neg_b_pos_one_2;
	m_expr_pos_dict_pos_b_neg_one_3 = other.m_expr_pos_dict_pos_b_neg_one_3;
	m_expr_pos_dict_neg_b_neg_one_4 = other.m_expr_pos_dict_neg_b_neg_one_4;

	m_expr_unique = other.m_expr_unique;

	m_matrixRow_neg_dict_pos_b_pos_one_1 = other.m_matrixRow_neg_dict_pos_b_pos_one_1;
	m_matrixRow_neg_dict_neg_b_pos_one_2 = other.m_matrixRow_neg_dict_neg_b_pos_one_2;
	m_matrixRow_pos_dict_pos_b_neg_one_3 = other.m_matrixRow_pos_dict_pos_b_neg_one_3;
	m_matrixRow_pos_dict_neg_b_neg_one_4 = other.m_matrixRow_pos_dict_neg_b_neg_one_4;

	m_matrixRow_unique = other.m_matrixRow_unique;

	return *this;
}


void OptimizationProblem_MultiLayered::applyChanges(const ChangedEntries & entries)
{
	const auto k = getNumTemplates();

	for (auto idx : entries.blackToWhite)
	{
		m_matrixRow_neg_dict_pos_b_pos_one_1[idx].setBounds(-IloInfinity, m_parameters.beta);
		m_matrixRow_neg_dict_neg_b_pos_one_2[idx].setBounds(-IloInfinity, -m_parameters.beta);
		m_matrixRow_pos_dict_pos_b_neg_one_3[idx].setBounds(-IloInfinity, m_parameters.beta);
		m_matrixRow_pos_dict_neg_b_neg_one_4[idx].setBounds(-IloInfinity, -m_parameters.beta);

		Lb[k * N + idx] = 1.f;
		Ub[k * N + idx] = 999.f;
	}

	for (auto idx : entries.whiteToBlack)
	{
		m_matrixRow_neg_dict_pos_b_pos_one_1[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_neg_dict_neg_b_pos_one_2[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_pos_dict_pos_b_neg_one_3[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_pos_dict_neg_b_neg_one_4[idx].setBounds(-IloInfinity, 0.f);

		Lb[k * N + idx] = 0.f;
		Ub[k * N + idx] = 0.f;
	}

	X.setBounds(Lb, Ub);
}


void OptimizationProblem_MultiLayered::fillMatrixRows(const Dictionary & dict, ScalarType factorDict,
                                                      ScalarType factorBeta, ScalarType factorI,
                                                      std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange)
{
	const int k_max = dict.getNumTemplates();

	for (int r = 0; r < M; ++r)
	{
		IloExpr expr(env);

		for (int k = 0; k < k_max; ++k)
		{
			const auto & d = dict.getEntries(k);

			for (int c = 0; c < N; ++c)
			{
				if (d[c][r] == 0)
				{
					continue;
				}
				expr += factorDict * X[k * N + c];
			}
		}

		expr += factorBeta * X[k_max * N + r];
		expr += factorI * X[k_max * N + M + r];

		pExpr->push_back(expr);

		pRange->push_back(-IloInfinity <= pExpr->back() <= 0);

		m_persistentModel.add(pRange->back());
	}
}


void OptimizationProblem_MultiLayered::fillUniqueRows(const Dictionary & dict, ScalarType factor,
                                                      std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange)
{
	for (int r = 0; r < N; ++r)
	{
		IloExpr expr(env);

		for (int k = 0; k < dict.getNumTemplates(); ++k)
		{
			expr += factor * X[k * N + r];
		}

		pExpr->push_back(expr);

		pRange->push_back(-IloInfinity <= pExpr->back() <= 1);

		m_persistentModel.add(pRange->back());
	}
}


std::vector<std::vector<IloExpr> *> OptimizationProblem_MultiLayered::getExpressions()
{
	return
	{
		&m_expr_neg_dict_pos_b_pos_one_1,
		&m_expr_neg_dict_neg_b_pos_one_2,
		&m_expr_pos_dict_pos_b_neg_one_3,
		&m_expr_pos_dict_neg_b_neg_one_4,
		&m_expr_unique
	};
}


std::vector<std::vector<IloRange> *> OptimizationProblem_MultiLayered::getMatrixRows()
{
	return
	{
		&m_matrixRow_neg_dict_pos_b_pos_one_1,
		&m_matrixRow_neg_dict_neg_b_pos_one_2,
		&m_matrixRow_pos_dict_pos_b_neg_one_3,
		&m_matrixRow_pos_dict_neg_b_neg_one_4,
		&m_matrixRow_unique
	};
}


std::vector<const std::vector<IloExpr> *> OptimizationProblem_MultiLayered::getExpressions() const
{
	return
	{
		&m_expr_neg_dict_pos_b_pos_one_1,
		&m_expr_neg_dict_neg_b_pos_one_2,
		&m_expr_pos_dict_pos_b_neg_one_3,
		&m_expr_pos_dict_neg_b_neg_one_4,
		&m_expr_unique
	};
}


std::vector<const std::vector<IloRange> *> OptimizationProblem_MultiLayered::getMatrixRows() const
{
	return
	{
		&m_matrixRow_neg_dict_pos_b_pos_one_1,
		&m_matrixRow_neg_dict_neg_b_pos_one_2,
		&m_matrixRow_pos_dict_pos_b_neg_one_3,
		&m_matrixRow_pos_dict_neg_b_neg_one_4,
		&m_matrixRow_unique
	};
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblem_MultiLayered::getFactory() const
{
	return std::make_unique<OptimizationProblemFactory_MultiLayered>();
}


IloInt OptimizationProblem_MultiLayered::getNumTemplates() const
{
	return (F.getSize() - 2 * M) / N;
}
