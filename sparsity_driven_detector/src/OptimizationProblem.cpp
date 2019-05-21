#include "OptimizationProblem.h"

#include <map>

#include "OptimizationProblemFactory.h"

#include "serialization_helper.h"
#include "SparseMatrix.h"


using Parameters = OptimizationProblem::Parameters;


std::ostream & operator<<(std::ostream &, const Parameters &)
{
	throw std::runtime_error("Not implemented");
}


std::istream & operator>>(std::istream &, Parameters &)
{
	throw std::runtime_error("Not implemented");
}


bool operator==(const Parameters & lhs, const Parameters & rhs)
{
	return /*lhs.advancedInitialization == rhs.advancedInitialization &&
		lhs.rootAlgorithm == rhs.rootAlgorithm &&*/
		lhs.beta == rhs.beta;
}


bool operator!=(const Parameters & lhs, const Parameters & rhs)
{
	return !(lhs == rhs);
}


OptimizationProblem::OptimizationProblem() :
	IOptimizationProblem(),
	env(),
	m_parameters(),
	N(0),
	M(0),
	F(),
	Lb(),
	Ub(),
	X(),
	O(),
	m_persistentModel(env),
	m_persistentCplex(m_persistentModel),
	m_activeTargetPixelLastFrame(),
	m_lastSolution()
{
	// Tests have shown that this parameter combination seems to be the fastest. 
	// It therefore does not make sense to store the parameters anywhere
	// OptimizationProblem allows to set these parameters externally after creation
	setCplexParam(IloCplex::RootAlg, IloCplex::Dual);
	setCplexParam(IloCplex::Param::Advance, 1);

	m_persistentCplex.setOut(env.getNullStream());
}


OptimizationProblem::~OptimizationProblem()
{
	env.end();
}


OptimizationProblem::OptimizationProblem(OptimizationProblem && other) noexcept(noexcept(IloEnv{ })) :
	IOptimizationProblem(),
	env(other.env),
	m_parameters(other.m_parameters),
	N(other.N),
	M(other.M),
	F(other.F),
	Lb(other.Lb),
	Ub(other.Ub),
	X(other.X),
	O(other.O),
	m_persistentModel(other.m_persistentModel),
	m_persistentCplex(other.m_persistentCplex),
	m_activeTargetPixelLastFrame(other.m_activeTargetPixelLastFrame),
	m_lastSolution(other.m_lastSolution)
{
	other.env = IloEnv{ };
}


OptimizationProblem & OptimizationProblem::operator=(OptimizationProblem && other) noexcept(noexcept(IloEnv{ }))
{
	env.end();

	env = other.env;
	other.env = IloEnv{ };

	m_parameters = other.m_parameters;

	N = other.N;
	M = other.M;
	F = other.F;
	Lb = other.Lb;
	Ub = other.Ub;

	X = other.X;
	O = other.O;

	m_persistentModel = other.m_persistentModel;
	m_persistentCplex = other.m_persistentCplex;
	m_activeTargetPixelLastFrame = other.m_activeTargetPixelLastFrame;
	m_lastSolution = other.m_lastSolution;

	return *this;
}


void OptimizationProblem::setCplexParam(IloCplex::IntParam parameter, CPXINT value) const
{
	m_persistentCplex.setParam(parameter, value);
}


void OptimizationProblem::setCplexParam(IloCplex::LongParam parameter, CPXLONG value) const
{
	m_persistentCplex.setParam(parameter, value);
}


void OptimizationProblem::setCplexParam(IloCplex::BoolParam parameter, IloBool value) const
{
	m_persistentCplex.setParam(parameter, value);
}


void OptimizationProblem::setCplexParam(IloCplex::NumParam parameter, IloNum value) const
{
	m_persistentCplex.setParam(parameter, value);
}


void OptimizationProblem::setCplexParam(IloCplex::StringParam parameter, const char * value) const
{
	m_persistentCplex.setParam(parameter, value);
}


CPXINT OptimizationProblem::getCplexParam(IloCplex::IntParam parameter) const
{
	return m_persistentCplex.getParam(parameter);
}


CPXLONG OptimizationProblem::getCplexParam(IloCplex::LongParam parameter) const
{
	return m_persistentCplex.getParam(parameter);
}


IloBool OptimizationProblem::getCplexParam(IloCplex::BoolParam parameter) const
{
	return m_persistentCplex.getParam(parameter);
}


IloNum OptimizationProblem::getCplexParam(IloCplex::NumParam parameter) const
{
	return m_persistentCplex.getParam(parameter);
}


const char * OptimizationProblem::getCplexParam(IloCplex::StringParam parameter) const
{
	return m_persistentCplex.getParam(parameter);
}


void OptimizationProblem::makeOptimizationTarget()
{
	assert(Lb.getSize() == getNumVars());
	assert(Ub.getSize() == getNumVars());
	assert(F.getSize() == getNumVars());

	X = IloNumVarArray(env, Lb, Ub, ILOFLOAT);

	for (IloInt i = 0; i < getNumVars(); ++i)
	{
		X[i].setName(cplexVarNameFromId(i + 1).c_str());
	}

	O = IloMinimize(env, IloScalProd(X, F));
	m_persistentModel.add(IloExtractable(O));
}


std::vector<int> OptimizationProblem::solve(const std::vector<cv::Mat> & target)
{
	const auto & activeTargetPixelCurrentFrame = makeSparseTarget(target);
	const auto & changes = getChangesToPreviousFrame(activeTargetPixelCurrentFrame);

	m_activeTargetPixelLastFrame = std::move(activeTargetPixelCurrentFrame);

	applyChanges(changes);

	solveCplex();

	return extractSolution();
}


std::list<size_t> OptimizationProblem::makeSparseTarget(const std::vector<cv::Mat> & target) const
{
	std::list<size_t> activeTargetPixelCurrentFrame;
	size_t offset = 0;
	for (const auto & mat : target)
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

	return activeTargetPixelCurrentFrame;
}


OptimizationProblem::ChangedEntries OptimizationProblem::getChangesToPreviousFrame(
	const std::list<size_t> & activeTargetPixelCurrentFrame) const
{
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

	return { changedEntriesBlackToWhite, changedEntriesWhiteToBlack };
}


void OptimizationProblem::solveCplex()
{
	m_persistentCplex.solve();
}


std::vector<int> OptimizationProblem::extractSolution()
{
	try
	{
		m_persistentCplex.getValues(X, m_lastSolution);
	}
	catch (IloWrongUsage & e)
	{
		e.print(std::cout);
		return { };
	}
	catch (IloException & e)
	{
		e.print(std::cout);
		return { };
	}

	std::vector<int> ret;

	for (int i = 0; i < getNumTemplates() * N; i++)
	{
		if (m_lastSolution[i] != 0)
		{
			ret.push_back(i);
		}
	}

	return ret;
}


std::string OptimizationProblem::cplexVarNameFromId(IloInt id)
{
	return "x" + std::to_string(id);
}


int OptimizationProblem::idFromCplexVarName(const std::string & varName)
{
	constexpr int idx = sizeof("x") / sizeof(char) - 1; // 0 term
	return std::stoi(varName.substr(idx));
}


bool OptimizationProblem::operator==(const IOptimizationProblem & rhs) const
{
	const auto pRhs = dynamic_cast<const OptimizationProblem *>(&rhs);

	if (!pRhs)
	{
		return false;
	}

	// Just checks for the equality of the serializable information
	if (!(*getFactory() == *pRhs->getFactory()))
	{
		return false;
	}

	if (m_parameters != pRhs->m_parameters)
	{
		return false;
	}

	if (M != pRhs->M || N != pRhs->N)
	{
		return false;
	}

	if (m_persistentCplex.getParam(IloCplex::RootAlg) !=
		pRhs->m_persistentCplex.getParam(IloCplex::RootAlg))
	{
		return false;
	}

	if (m_persistentCplex.getParam(IloCplex::Param::Advance) !=
		pRhs->m_persistentCplex.getParam(IloCplex::Param::Advance))
	{
		return false;
	}

	for (int c = 0; c < M + N; ++c)
	{
		if (F[c] != pRhs->F[c] ||
			Lb[c] != pRhs->Lb[c] ||
			Ub[c] != pRhs->Ub[c])
		{
			return false;
		}
	}

	const auto & expressions = getExpressions();
	const auto & rhsExpressions = pRhs->getExpressions();

	if (expressions.size() != rhsExpressions.size())
	{
		return false;
	}

	for (size_t idxExpressionSet = 0; idxExpressionSet < expressions.size(); ++idxExpressionSet)
	{
		const auto & expressionSet = *expressions[idxExpressionSet];
		const auto & rhsExpressionSet = *rhsExpressions[idxExpressionSet];

		if (expressionSet.size() != rhsExpressionSet.size())
		{
			return false;
		}

		for (size_t idxExpression = 0; idxExpression < expressionSet.size(); ++idxExpression)
		{
			const auto e = expressionSet[idxExpression];
			const auto rhsE = rhsExpressionSet[idxExpression];

			auto it = e.getLinearIterator();
			auto itRhs = rhsE.getLinearIterator();

			for (; it.ok() && itRhs.ok(); ++it, ++itRhs)
			{
				if (it.getCoef() != itRhs.getCoef())
				{
					return false;
				}

				if (std::strcmp(it.getVar().getName(), itRhs.getVar().getName()) != 0)
				{
					return false;
				}
			}

			if (it.ok() || itRhs.ok())
			{
				return false;
			}
		}
	}

	return true;
}


void OptimizationProblem::save(const std::string & filename) const
{
	using namespace boost::filesystem;
	using std::ofstream;

	const path temp = makeTempDir(path(filename).parent_path());

	getFactory()->save((temp / "type.txt").string());

	{
		ofstream of((temp / "meta.txt").string());
		of << M << " " << N << "\n";

		const auto & expressions = getExpressions();
		for (auto it = expressions.begin(); it != expressions.end(); ++it)
		{
			if (it != expressions.begin())
			{
				of << " ";
			}

			of << (*it)->size();
		}
	}

	{
		ofstream of((temp / "param.txt").string());
		of << m_parameters.beta;
	}

	ofstream((temp / "F.txt").string()) << F;
	ofstream((temp / "Lb.txt").string()) << Lb;
	ofstream((temp / "Ub.txt").string()) << Ub;

	{
		// Maybe we can skip this since the expressions can't be evaluated on load anyway
		std::ofstream of((temp / "expr.txt").string());

		const auto & expressions = getExpressions();

		for (const auto & expressionSet : expressions)
		{
			for (const auto & e : *expressionSet)
			{
				of << e << std::endl;
			}
		}
	}

	{
		std::ofstream of((temp / "A.txt").string());

		const auto & expressions = getExpressions();
		size_t rowOffset = 0;

		for (const auto pExpressionSet : expressions)
		{
			for (size_t idxRow = 0; idxRow < pExpressionSet->size(); ++idxRow)
			{
				const auto & row = (*pExpressionSet)[idxRow];

				for (auto it = row.getLinearIterator(); it.ok(); ++it)
				{
					const auto coef = it.getCoef();
					const auto id = idFromCplexVarName(it.getVar().getName());

					of << (idxRow + 1 + rowOffset) << " " << id << " " << coef << "\n";
				}
			}

			rowOffset += pExpressionSet->size();
		}
	}

	zipDir(temp, filename);

	remove_all(temp);
}


std::unique_ptr<OptimizationProblem> OptimizationProblem::load(const std::string & filename)
{
	using namespace boost::filesystem;

	const path temp = makeTempDir(path(filename).parent_path());

	unzipDir(filename, temp);

	std::unique_ptr<OptimizationProblem> pO = OptimizationProblemFactory::load((temp / "type.txt").string())->
		createProblemForLoading();
	auto & o = *pO; // convenience

	std::vector<size_t> setSizes;
	{
		std::ifstream is((temp / "meta.txt").string());
		is >> o.M >> o.N;
		while (!is.eof())
		{
			size_t s;
			is >> s;
			setSizes.push_back(s);
		}
	}

	{
		std::ifstream is((temp / "param.txt").string());

		ScalarType beta = 0.f;

		is >> beta;

		o.m_parameters = Parameters{ beta };
	}

	o.F = IloNumArray(o.env);
	o.Lb = IloNumArray(o.env);
	o.Ub = IloNumArray(o.env);

	o.m_lastSolution = IloNumArray(o.env, o.N + o.M);

	std::ifstream((temp / "F.txt").string()) >> o.F;
	std::ifstream((temp / "Lb.txt").string()) >> o.Lb;
	std::ifstream((temp / "Ub.txt").string()) >> o.Ub;

	o.makeOptimizationTarget();

	// temp / "expr.txt" cannot be read. Instead we have to rebuild from A in temp / "A.txt"

	{
		std::ifstream is((temp / "A.txt").string());

		SparseMatrix<ScalarType> tempMat(o.getNumExpressions(), o.getNumVars());
		tempMat.read(is);

		auto expressionSets = o.getExpressions();
		auto matrixRowSets = o.getMatrixRows();

		assert(expressionSets.size() == matrixRowSets.size());
		assert(setSizes.size() == expressionSets.size());

		size_t rowOffset = 0;

		for (size_t idxSet = 0; idxSet < expressionSets.size(); ++idxSet)
		{
			auto & expressions = expressionSets[idxSet];
			auto & matrixRows = matrixRowSets[idxSet];

			const auto setSize = setSizes[idxSet];

			for (size_t idxRow = rowOffset; idxRow < rowOffset + setSize; ++idxRow)
			{
				IloExpr expr(o.env);

				const auto & row = tempMat.getRow(idxRow);
				for (auto & entry : row)
				{
					expr += entry.val * o.X[entry.col];
				}

				expressions->push_back(expr);
				matrixRows->push_back(-IloInfinity <= expressions->back() <= 0);
				o.m_persistentModel.add(matrixRows->back());
			}

			rowOffset += setSize;
		}
	}

	// TODO: Does this need to be here
	//o.m_persistentModel.add(IloExtractable(o.O));

	return pO;
}
