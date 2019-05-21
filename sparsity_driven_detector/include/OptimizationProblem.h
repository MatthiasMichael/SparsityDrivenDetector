#pragma once

#include <list>

#ifndef IL_STD
#define IL_STD
#endif
#include <cstring>
#include <ilcplex/ilocplex.h>

#include "IOptimizationProblem.h"
#include "WorldCoordinateSystem_SDD.h" // Use consistent ScalarType

class OptimizationProblemFactory;

class OptimizationProblem : public IOptimizationProblem
{
public:

	struct Parameters
	{
		ScalarType beta;

		friend std::ostream & operator<<(std::ostream &, const Parameters &);
		friend std::istream & operator>>(std::istream &, Parameters &);

		friend bool operator==(const Parameters &, const Parameters &);
		friend bool operator!=(const Parameters &, const Parameters &);
	};

	virtual ~OptimizationProblem();

	OptimizationProblem(const OptimizationProblem &) = delete;
	OptimizationProblem(OptimizationProblem &&) noexcept(noexcept(IloEnv{}));

	OptimizationProblem & operator=(const OptimizationProblem &) = delete;
	OptimizationProblem & operator=(OptimizationProblem &&) noexcept(noexcept(IloEnv{}));

	void setCplexParam(IloCplex::IntParam parameter, CPXINT value) const;
	void setCplexParam(IloCplex::LongParam parameter, CPXLONG value) const;
	void setCplexParam(IloCplex::BoolParam parameter, IloBool value) const;
	void setCplexParam(IloCplex::NumParam parameter, IloNum value) const;
	void setCplexParam(IloCplex::StringParam parameter, const char * value) const;

	CPXINT getCplexParam(IloCplex::IntParam parameter) const;
	CPXLONG getCplexParam(IloCplex::LongParam parameter) const;
	IloBool getCplexParam(IloCplex::BoolParam parameter) const;
	IloNum getCplexParam(IloCplex::NumParam parameter) const;
	const char * getCplexParam(IloCplex::StringParam parameter) const;

	std::vector<int> solve(const std::vector<cv::Mat> & target) override;

	bool operator==(const IOptimizationProblem & rhs) const override;

	void save(const std::string & filename) const override;

	static std::unique_ptr<OptimizationProblem> load(const std::string & filename);

	friend class OptimizationProblemDecorator;

protected:
	OptimizationProblem(); 

	void makeOptimizationTarget();

	std::list<size_t> makeSparseTarget(const std::vector<cv::Mat> & target) const override;
	ChangedEntries getChangesToPreviousFrame(const std::list<size_t> & activeTargetPixelCurrentFrame) const override;

	void solveCplex() override;
	std::vector<int> extractSolution() override; 

private:
	static std::string cplexVarNameFromId(IloInt id);
	static int idFromCplexVarName(const std::string & varName);

	virtual std::vector<std::vector<IloExpr> *> getExpressions() = 0;
	virtual std::vector<std::vector<IloRange> *> getMatrixRows() = 0;

	virtual std::vector<const std::vector<IloExpr> *> getExpressions() const = 0;
	virtual std::vector<const std::vector<IloRange> *> getMatrixRows() const = 0;

	virtual std::unique_ptr<OptimizationProblemFactory> getFactory() const = 0;

	virtual IloInt getNumExpressions() const = 0;
	virtual IloInt getNumVars() const = 0;
	virtual IloInt getNumTemplates() const = 0;

protected:
	IloEnv env;

	// Start Serializable Info

	Parameters m_parameters;

	IloInt N; //< = d.getNumEntries();
	IloInt M; //< = d.getNumPixel();

	IloNumArray F;
	IloNumArray Lb;
	IloNumArray Ub;

	// End Serializable Info

	// Start Optimization Info

	IloNumVarArray X;
	IloObjective O;

	IloModel m_persistentModel;
	mutable IloCplex m_persistentCplex; //< Mutable allows setting of parameters

	// End Optimization Info

	// Intermediate Info

	std::list<size_t> m_activeTargetPixelLastFrame;

	IloNumArray m_lastSolution;
};
