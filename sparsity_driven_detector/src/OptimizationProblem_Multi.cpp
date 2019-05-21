#include "OptimizationProblem_Multi.h"

#include "OptimizationProblemFactory_Multi.h"


OptimizationProblem_Multi::OptimizationProblem_Multi() :
	OptimizationProblem(),
	m_expr_neg(),
	m_expr_pos(),
	m_expr_unique(),
	m_matrixRowNeg(),
	m_matrixRowPos(),
	m_matrixRowUnique()
{
	// empty
}


OptimizationProblem_Multi::OptimizationProblem_Multi(const Dictionary & dict, const Parameters & parameters) :
	OptimizationProblem()
{
	m_parameters = parameters;

	N = dict.getNumEntries();
	M = dict.getNumPixel();

	const int k = dict.getNumTemplates();

	F = IloNumArray(env, k * N + M);
	Lb = IloNumArray(env, k * N + M);
	Ub = IloNumArray(env, k * N + M);

	m_lastSolution = IloNumArray(env, k * N + M);

	for (IloInt i = 0; i < k * N; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 1.f;

		m_lastSolution[i] = 0.f;
	}

	for (IloInt i = k * N; i < k * N + M; ++i)
	{
		F[i] = 1.f;

		Lb[i] = 0.f;
		Ub[i] = 999.f; // IloInfinity; // Technically it should be infinity. However that crashes in operator>>

		m_lastSolution[i] = 0.f;
	}

	makeOptimizationTarget();

	fillMatrixRows(dict, -1, -1, &m_expr_neg, &m_matrixRowNeg);
	fillMatrixRows(dict, 1, -1, &m_expr_pos, &m_matrixRowPos);

	fillUniqueRows(dict, 1, &m_expr_unique, &m_matrixRowUnique);
}


OptimizationProblem_Multi::OptimizationProblem_Multi(
	OptimizationProblem_Multi && other) noexcept(noexcept(IloEnv{ })) :
	OptimizationProblem(std::move(other)),
	m_expr_neg(other.m_expr_neg),
	m_expr_pos(other.m_expr_pos),
	m_expr_unique(other.m_expr_unique),
	m_matrixRowNeg(other.m_matrixRowNeg),
	m_matrixRowPos(other.m_matrixRowPos),
	m_matrixRowUnique(other.m_matrixRowUnique)
{
	// empty
}


OptimizationProblem_Multi & OptimizationProblem_Multi::operator=(
	OptimizationProblem_Multi && other) noexcept(noexcept(IloEnv{ }))
{
	static_cast<OptimizationProblem&>(*this) = std::move(static_cast<OptimizationProblem&>(other));

	m_expr_neg = other.m_expr_neg;
	m_expr_pos = other.m_expr_pos;

	m_expr_unique = other.m_expr_unique;

	m_matrixRowNeg = other.m_matrixRowNeg;
	m_matrixRowPos = other.m_matrixRowPos;

	m_matrixRowUnique = other.m_matrixRowUnique;

	return *this;
}


void OptimizationProblem_Multi::applyChanges(const ChangedEntries & entries)
{
	for (auto idx : entries.blackToWhite)
	{
		m_matrixRowNeg[idx].setBounds(-IloInfinity, -1.f);
		m_matrixRowPos[idx].setBounds(-IloInfinity, 1.f);
	}

	for (auto idx : entries.whiteToBlack)
	{
		m_matrixRowNeg[idx].setBounds(-IloInfinity, 0);
		m_matrixRowPos[idx].setBounds(-IloInfinity, 0);
	}
}


void OptimizationProblem_Multi::fillMatrixRows(
	const Dictionary & dict, ScalarType factorDict, ScalarType factorI,
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
		expr += factorI * X[k_max * N + r];

		pExpr->push_back(expr);

		pRange->push_back(-IloInfinity <= pExpr->back() <= 0);

		m_persistentModel.add(pRange->back());
	}
}


void OptimizationProblem_Multi::fillUniqueRows(const Dictionary & dict, ScalarType factor,
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


std::vector<std::vector<IloExpr> *> OptimizationProblem_Multi::getExpressions()
{
	return { &m_expr_neg, &m_expr_pos, &m_expr_unique };
}


std::vector<std::vector<IloRange> *> OptimizationProblem_Multi::getMatrixRows()
{
	return { &m_matrixRowNeg, &m_matrixRowPos, &m_matrixRowUnique };
}


std::vector<const std::vector<IloExpr> *> OptimizationProblem_Multi::getExpressions() const
{
	return { &m_expr_neg, &m_expr_pos, &m_expr_unique };
}


std::vector<const std::vector<IloRange> *> OptimizationProblem_Multi::getMatrixRows() const
{
	return { &m_matrixRowNeg, &m_matrixRowPos, &m_matrixRowUnique };
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblem_Multi::getFactory() const
{
	return std::make_unique<OptimizationProblemFactory_Multi>();
}


IloInt OptimizationProblem_Multi::getNumTemplates() const
{
	return (F.getSize() - M) / N;
}
