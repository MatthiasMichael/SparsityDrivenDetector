#include "OptimizationProblem_SingleLayered.h"

#include "TemporaryJoinedVector.h"

#include "OptimizationProblemFactory_SingleLayered.h"


OptimizationProblem_SingleLayered::OptimizationProblem_SingleLayered() : OptimizationProblem()
{
	// empty
}


OptimizationProblem_SingleLayered::OptimizationProblem_SingleLayered(const Dictionary & dict,
                                                                     const Parameters & parameters) :
	OptimizationProblem()
{
	assert(parameters.beta != 0); // Otherwise the solution should be empty anyways

	m_parameters = parameters;

	N = dict.getNumEntries();
	M = dict.getNumPixel();

	F = IloNumArray(env, N + 2 * M);
	Lb = IloNumArray(env, N + 2 * M);
	Ub = IloNumArray(env, N + 2 * M);

	m_lastSolution = IloNumArray(env, N + 2 * M);

	for (IloInt i = 0; i < N; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 1.f;
	}

	for (IloInt i = N; i < N + M; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 0.f;
	}

	for (IloInt i = N + M; i < N + 2 * M; ++i)
	{
		F[i] = 1.f;

		Lb[i] = 0.f;
		Ub[i] = 999.f; // IloInfinity; // Technically it should be infinity. However that crashes in operator>>
	}

	makeOptimizationTarget();

	const auto b = parameters.beta;

	const auto & d = dict.getEntries_single();

	fillMatrixRows(d, -1, (b + 1), -1, &m_expr_neg_dict_pos_b_pos_one_1, &m_matrixRow_neg_dict_pos_b_pos_one_1);
	fillMatrixRows(d, -1, (-b + 1), -1, &m_expr_neg_dict_neg_b_pos_one_2, &m_matrixRow_neg_dict_neg_b_pos_one_2);
	fillMatrixRows(d, 1, (b - 1), -1, &m_expr_pos_dict_pos_b_neg_one_3, &m_matrixRow_pos_dict_pos_b_neg_one_3);
	fillMatrixRows(d, 1, (-b - 1), -1, &m_expr_pos_dict_neg_b_neg_one_4, &m_matrixRow_pos_dict_neg_b_neg_one_4);
}


OptimizationProblem_SingleLayered::OptimizationProblem_SingleLayered(
	OptimizationProblem_SingleLayered && other) noexcept(noexcept(IloEnv{ })) :
	OptimizationProblem(std::move(other)),
	m_expr_neg_dict_pos_b_pos_one_1(other.m_expr_neg_dict_pos_b_pos_one_1),
	m_expr_neg_dict_neg_b_pos_one_2(other.m_expr_neg_dict_neg_b_pos_one_2),
	m_expr_pos_dict_pos_b_neg_one_3(other.m_expr_pos_dict_pos_b_neg_one_3),
	m_expr_pos_dict_neg_b_neg_one_4(other.m_expr_pos_dict_neg_b_neg_one_4),
	m_matrixRow_neg_dict_pos_b_pos_one_1(other.m_matrixRow_neg_dict_pos_b_pos_one_1),
	m_matrixRow_neg_dict_neg_b_pos_one_2(other.m_matrixRow_neg_dict_neg_b_pos_one_2),
	m_matrixRow_pos_dict_pos_b_neg_one_3(other.m_matrixRow_pos_dict_pos_b_neg_one_3),
	m_matrixRow_pos_dict_neg_b_neg_one_4(other.m_matrixRow_pos_dict_neg_b_neg_one_4)
{
	// empty
}


OptimizationProblem_SingleLayered & OptimizationProblem_SingleLayered::operator=(
	OptimizationProblem_SingleLayered && other) noexcept(noexcept(IloEnv{ }))
{
	static_cast<OptimizationProblem&>(*this) = std::move(static_cast<OptimizationProblem&>(other));

	m_expr_neg_dict_pos_b_pos_one_1 = other.m_expr_neg_dict_pos_b_pos_one_1;
	m_expr_neg_dict_neg_b_pos_one_2 = other.m_expr_neg_dict_neg_b_pos_one_2;
	m_expr_pos_dict_pos_b_neg_one_3 = other.m_expr_pos_dict_pos_b_neg_one_3;
	m_expr_pos_dict_neg_b_neg_one_4 = other.m_expr_pos_dict_neg_b_neg_one_4;

	m_matrixRow_neg_dict_pos_b_pos_one_1 = other.m_matrixRow_neg_dict_pos_b_pos_one_1;
	m_matrixRow_neg_dict_neg_b_pos_one_2 = other.m_matrixRow_neg_dict_neg_b_pos_one_2;
	m_matrixRow_pos_dict_pos_b_neg_one_3 = other.m_matrixRow_pos_dict_pos_b_neg_one_3;
	m_matrixRow_pos_dict_neg_b_neg_one_4 = other.m_matrixRow_pos_dict_neg_b_neg_one_4;

	return *this;
}


void OptimizationProblem_SingleLayered::applyChanges(const ChangedEntries & entries)
{
	for (auto idx : entries.blackToWhite)
	{
		m_matrixRow_neg_dict_pos_b_pos_one_1[idx].setBounds(-IloInfinity, m_parameters.beta);
		m_matrixRow_neg_dict_neg_b_pos_one_2[idx].setBounds(-IloInfinity, -m_parameters.beta);
		m_matrixRow_pos_dict_pos_b_neg_one_3[idx].setBounds(-IloInfinity, m_parameters.beta);
		m_matrixRow_pos_dict_neg_b_neg_one_4[idx].setBounds(-IloInfinity, -m_parameters.beta);

		Lb[N + idx] = 1.f;
		Ub[N + idx] = 999.f;
	}

	for (auto idx : entries.whiteToBlack)
	{
		m_matrixRow_neg_dict_pos_b_pos_one_1[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_neg_dict_neg_b_pos_one_2[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_pos_dict_pos_b_neg_one_3[idx].setBounds(-IloInfinity, 0.f);
		m_matrixRow_pos_dict_neg_b_neg_one_4[idx].setBounds(-IloInfinity, 0.f);

		Lb[N + idx] = 0.f;
		Ub[N + idx] = 0.f;
	}

	X.setBounds(Lb, Ub);
}


std::vector<int> OptimizationProblem_SingleLayered::solve_dense(const std::vector<cv::Mat> & targets)
{
	std::vector<TemporaryJoinedVector<float*>::IteratorPair> ranges;

	for (const auto & mat : targets)
	{
		const auto begin = reinterpret_cast<float*>(mat.data);
		const auto end = reinterpret_cast<float*>(mat.data) + mat.rows * mat.cols;

		ranges.push_back({ begin, end });
	}

	TemporaryJoinedVector<float*> tempVector(ranges);

	const auto beta = m_parameters.beta;
	assert(beta != 0);

	for (size_t i = 0; i < tempVector.size(); ++i)
	{
		m_matrixRow_neg_dict_pos_b_pos_one_1[i].setBounds(-IloInfinity, beta * tempVector[i]);
		m_matrixRow_neg_dict_neg_b_pos_one_2[i].setBounds(-IloInfinity, beta * -tempVector[i]);
		m_matrixRow_pos_dict_pos_b_neg_one_3[i].setBounds(-IloInfinity, beta * tempVector[i]);
		m_matrixRow_pos_dict_neg_b_neg_one_4[i].setBounds(-IloInfinity, beta * -tempVector[i]);

		Lb[N + i] = tempVector[i];
		Ub[N + i] = 999 * tempVector[i];
	}

	X.setBounds(Lb, Ub);

	solveCplex();
	return extractSolution();
}


void OptimizationProblem_SingleLayered::fillMatrixRows(const std::vector<Dictionary::Word> & d, ScalarType factorDict,
                                                       ScalarType factorBeta, ScalarType factorI,
                                                       std::vector<IloExpr> * pExpr, std::vector<IloRange> * pRange)
{
	for (int r = 0; r < M; ++r)
	{
		IloExpr expr(env);
		for (int c = 0; c < N; ++c)
		{
			if (d[c][r] == 0)
			{
				continue;
			}
			expr += factorDict * X[c];
		}
		expr += factorBeta * X[N + r];
		expr += factorI * X[N + M + r];

		pExpr->push_back(expr);

		pRange->push_back(-IloInfinity <= pExpr->back() <= 0);

		m_persistentModel.add(pRange->back());
	}
}


std::vector<std::vector<IloExpr> *> OptimizationProblem_SingleLayered::getExpressions()
{
	return
	{
		&m_expr_neg_dict_pos_b_pos_one_1,
		&m_expr_neg_dict_neg_b_pos_one_2,
		&m_expr_pos_dict_pos_b_neg_one_3,
		&m_expr_pos_dict_neg_b_neg_one_4
	};
}


std::vector<std::vector<IloRange> *> OptimizationProblem_SingleLayered::getMatrixRows()
{
	return
	{
		&m_matrixRow_neg_dict_pos_b_pos_one_1,
		&m_matrixRow_neg_dict_neg_b_pos_one_2,
		&m_matrixRow_pos_dict_pos_b_neg_one_3,
		&m_matrixRow_pos_dict_neg_b_neg_one_4
	};
}


std::vector<const std::vector<IloExpr> *> OptimizationProblem_SingleLayered::getExpressions() const
{
	return
	{
		&m_expr_neg_dict_pos_b_pos_one_1,
		&m_expr_neg_dict_neg_b_pos_one_2,
		&m_expr_pos_dict_pos_b_neg_one_3,
		&m_expr_pos_dict_neg_b_neg_one_4
	};
}


std::vector<const std::vector<IloRange> *> OptimizationProblem_SingleLayered::getMatrixRows() const
{
	return
	{
		&m_matrixRow_neg_dict_pos_b_pos_one_1,
		&m_matrixRow_neg_dict_neg_b_pos_one_2,
		&m_matrixRow_pos_dict_pos_b_neg_one_3,
		&m_matrixRow_pos_dict_neg_b_neg_one_4
	};
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblem_SingleLayered::getFactory() const
{
	return std::make_unique<OptimizationProblemFactory_SingleLayered>();
}
