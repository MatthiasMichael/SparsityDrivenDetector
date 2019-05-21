#include "OptimizationProblem_Single.h"

#include "TemporaryJoinedVector.h"

#include "OptimizationProblemFactory_Single.h"



OptimizationProblem_Single::OptimizationProblem_Single() :
	OptimizationProblem(),
	m_expr_neg(),
	m_expr_pos(),
	m_matrixRowNeg(),
	m_matrixRowPos(),
	m_lastTargetReconstructed()
{
	// empty
}


OptimizationProblem_Single::OptimizationProblem_Single(const Dictionary & dict, const Parameters & parameters) :
	OptimizationProblem(),
	m_lastTargetReconstructed(dict.getNumEntries(), 0.f)
{
	N = dict.getNumEntries();
	M = dict.getNumPixel();

	F = IloNumArray(env, N + M);
	Lb = IloNumArray(env, N + M);
	Ub = IloNumArray(env, N + M);

	m_lastSolution = IloNumArray(env, N + M);

	for (IloInt i = 0; i < N; ++i)
	{
		F[i] = 0.f;

		Lb[i] = 0.f;
		Ub[i] = 1.f;

		m_lastSolution[i] = 0.f;
	}

	for (IloInt i = N; i < N + M; ++i)
	{
		F[i] = 1.f;

		Lb[i] = 0.f;
		Ub[i] = 999.f; // IloInfinity; // Technically it should be infinity. However that crashes in operator>>

		m_lastSolution[i] = 0.f;
	}

	makeOptimizationTarget();

	const auto & d = dict.getEntries_single();

	fillMatrixRows(d, -1, -1, &m_expr_neg, &m_matrixRowNeg);
	fillMatrixRows(d,  1, -1, &m_expr_pos, &m_matrixRowPos);
}


void OptimizationProblem_Single::fillMatrixRows(
	const std::vector<Dictionary::Word> & d, 
	ScalarType factorDict, ScalarType factorI,
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
		expr += factorI * X[N + r];

		pExpr->push_back(expr);

		pRange->push_back(-IloInfinity <= pExpr->back() <= 0);

		m_persistentModel.add(pRange->back());
	}
}


OptimizationProblem_Single::OptimizationProblem_Single(
	OptimizationProblem_Single && other) noexcept(noexcept(IloEnv{ })) :
	OptimizationProblem(std::move(other)),
	m_expr_neg(other.m_expr_neg),
	m_expr_pos(other.m_expr_pos),
	m_matrixRowNeg(other.m_matrixRowNeg),
	m_matrixRowPos(other.m_matrixRowPos),
	m_lastTargetReconstructed(std::move(other.m_lastTargetReconstructed))
{
	// empty
}


OptimizationProblem_Single & OptimizationProblem_Single::operator=(
	OptimizationProblem_Single && other) noexcept(noexcept(IloEnv{ }))
{
	static_cast<OptimizationProblem&>(*this) = std::move(static_cast<OptimizationProblem&>(other));

	m_expr_neg = other.m_expr_neg;
	m_expr_pos = other.m_expr_pos;

	m_matrixRowNeg = other.m_matrixRowNeg;
	m_matrixRowPos = other.m_matrixRowPos;

	m_lastTargetReconstructed = std::move(other.m_lastTargetReconstructed);

	return *this;
}


void OptimizationProblem_Single::applyChanges(const ChangedEntries & entries)
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



// Solves using the persistent optimization problem and sparse sematics but does EXTENSIVE checks to make sure 
// the sparseness is correct
std::vector<int> OptimizationProblem_Single::solve_withChecks(const std::vector<cv::Mat> & targets)
{
	std::vector<TemporaryJoinedVector<float*>::IteratorPair> ranges;

	for (const auto & mat : targets)
	{
		const auto begin = reinterpret_cast<float*>(mat.data);
		const auto end = reinterpret_cast<float*>(mat.data) + mat.rows * mat.cols;

		ranges.push_back({ begin, end });
	}

	TemporaryJoinedVector<float*> tempVector(ranges);

	assert(tempVector.size() == M);

	std::vector<float> currentTarget(M, 0.f);
	for (auto i = 0L; i < M; ++i)
	{
		currentTarget[i] = tempVector[i];
	}

	// Current Target contains all pixel values concatenated

	std::list<size_t> activeTargetPixelCurrentFrame;
	size_t offset = 0;
	for (const auto & mat : targets)
	{
		const auto begin = reinterpret_cast<float*>(mat.data);
		const auto end = reinterpret_cast<float*>(mat.data) + mat.rows * mat.cols;

		for (auto itPixel = begin; itPixel != end; ++itPixel)
		{
			if (*itPixel != 0.f)
			{
				activeTargetPixelCurrentFrame.push_back(offset + itPixel - begin);
			}
		}

		offset += end - begin;
	}

	for (auto i = 0L; i < M; ++i)
	{
		if (std::find(activeTargetPixelCurrentFrame.begin(), activeTargetPixelCurrentFrame.end(), i) !=
			activeTargetPixelCurrentFrame.end())
		{
			assert(currentTarget[i] == 1.f);
		}
		else
		{
			assert(currentTarget[i] == 0.f);
		}
	}

	// At this point activeTargetPixelCurrentFrame is a sparse representation of currentTarget

	std::list<size_t> changedEntriesBlackToWhite;
	std::set_difference(
		activeTargetPixelCurrentFrame.begin(), activeTargetPixelCurrentFrame.end(),
		m_activeTargetPixelLastFrame.begin(), m_activeTargetPixelLastFrame.end(),
		std::back_inserter(changedEntriesBlackToWhite));

	std::list<size_t> changedEntriesWhiteToBlack;
	std::set_difference(
		m_activeTargetPixelLastFrame.begin(), m_activeTargetPixelLastFrame.end(),
		activeTargetPixelCurrentFrame.begin(), activeTargetPixelCurrentFrame.end(),
		std::back_inserter(changedEntriesWhiteToBlack));

	m_activeTargetPixelLastFrame = activeTargetPixelCurrentFrame;

	for (auto idx : changedEntriesBlackToWhite)
	{
		assert(m_lastTargetReconstructed[idx] == 0.f);
		m_lastTargetReconstructed[idx] = 1.f;

		m_matrixRowNeg[idx].setBounds(-IloInfinity, -1.f);
		m_matrixRowPos[idx].setBounds(-IloInfinity, 1.f);
	}

	for (auto idx : changedEntriesWhiteToBlack)
	{
		assert(m_lastTargetReconstructed[idx] == 1.f);
		m_lastTargetReconstructed[idx] = 0;

		m_matrixRowNeg[idx].setBounds(-IloInfinity, 0);
		m_matrixRowPos[idx].setBounds(-IloInfinity, 0);
	}

	assert(m_lastTargetReconstructed == currentTarget);

	solveCplex();
	return extractSolution();
}


// Solves using a persistent model while setting all bounds each time
std::vector<int> OptimizationProblem_Single::solve_dense(const std::vector<cv::Mat> & targets)
{
	std::vector<TemporaryJoinedVector<float*>::IteratorPair> ranges;

	for (const auto & mat : targets)
	{
		const auto begin = reinterpret_cast<float*>(mat.data);
		const auto end = reinterpret_cast<float*>(mat.data) + mat.rows * mat.cols;

		ranges.push_back({ begin, end });
	}

	TemporaryJoinedVector<float*> tempVector(ranges);

	for (size_t i = 0; i < tempVector.size(); ++i)
	{
		m_matrixRowNeg[i].setBounds(-IloInfinity, -tempVector[i]);
		m_matrixRowPos[i].setBounds(-IloInfinity, tempVector[i]);
	}

	solveCplex();
	return extractSolution();
}


std::vector<std::vector<IloExpr> *> OptimizationProblem_Single::getExpressions()
{
	return { &m_expr_neg, &m_expr_pos };
}


std::vector<std::vector<IloRange> *> OptimizationProblem_Single::getMatrixRows()
{
	return { &m_matrixRowNeg, &m_matrixRowPos };
}


std::vector<const std::vector<IloExpr> *> OptimizationProblem_Single::getExpressions() const
{
	return { &m_expr_neg, &m_expr_pos };
}


std::vector<const std::vector<IloRange> *> OptimizationProblem_Single::getMatrixRows() const
{
	return { &m_matrixRowNeg, &m_matrixRowPos };
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblem_Single::getFactory() const
{
	return std::make_unique<OptimizationProblemFactory_Single>();
}
