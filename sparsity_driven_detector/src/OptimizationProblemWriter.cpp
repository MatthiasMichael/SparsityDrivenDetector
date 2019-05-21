#include "OptimizationProblemWriter.h"


OptimizationProblemWriter::OptimizationProblemWriter(std::unique_ptr<IOptimizationProblem> && problem,
	const std::string & folder) :
	OptimizationProblemDecorator(std::move(problem)),
	m_solveCounter(0),
	m_saveFolder(folder)
{
	// empty
}


std::vector<int> OptimizationProblemWriter::solve(const std::vector<cv::Mat> & targets)
{
	const auto ret = m_problem->solve(targets);
	++m_solveCounter;
	writeProblem(targets);
	return ret;
}

//std::vector<int> OptimizationProblemWriter::solve_dense(const std::vector<cv::Mat> & targets)
//{
//	const auto ret = m_problem->solve_dense(targets);
//	++m_solveCounter;
//	writeProblem(targets);
//	return ret;
//}


void OptimizationProblemWriter::writeProblem(const std::vector<cv::Mat> & targets)
{
	std::stringstream ss;
	ss << m_saveFolder << "/model_";
	ss.width(4);
	ss.fill('0');
	ss << m_solveCounter << ".lp";
	getCplex().exportModel(ss.str().c_str());
	std::cout << "Model Written!" << std::endl;
}
