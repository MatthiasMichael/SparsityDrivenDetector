#include "OptimizationProblemMatrixWriter.h"

#include "boost/filesystem.hpp"

#include "TemporaryJoinedVector.h"

#include "OptimizationProblemFactory.h"
#include <opencv2/imgcodecs.hpp>


OptimizationProblemMatrixWriter::OptimizationProblemMatrixWriter(std::unique_ptr<IOptimizationProblem> && problem,
                                                                 const std::string & folder) :
	OptimizationProblemWriter(std::move(problem), folder)
{
	using namespace boost::filesystem;

	const path p(m_saveFolder);
	create_directories(p);

	getFactory()->save((p / "type.txt").string());

	{
		ofstream of((p / "meta.txt").string());
		of << getM() << " " << getN() << "\n";

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
		ofstream of((p / "param.txt").string());
		//of << static_cast<int>(getParameters().rootAlgorithm) << "\n";
		//of << getParameters().advancedInitialization << "\n";
		of << getParameters().beta;
	}

	{
		std::ofstream of((p / "A.txt").string());

		const auto & expr = getExpressions();
		size_t offset = 0;

		for (auto i : expr)
		{
			auto & rowSet = *i;

			for (size_t row = 0; row < rowSet.size(); ++row)
			{
				for (auto it = rowSet[row].getLinearIterator(); it.ok(); ++it)
				{
					const auto coeff = it.getCoef();
					const auto id = idFromCplexVarName(it.getVar().getName());

					of << (row + 1 + offset) << " " << id << " " << coeff << "\n";
				}
			}

			offset += rowSet.size();
		}
	}

	OutStream((p / "F.txt").string()) << getF();
	OutStream((p / "Lb_initial.txt").string()) << getLb();
	OutStream((p / "Ub_initial.txt").string()) << getUb();
}


void OptimizationProblemMatrixWriter::writeProblem(const std::vector<cv::Mat> & targets)
{
	using namespace boost::filesystem;

	std::stringstream ss;
	ss << m_saveFolder << "/";
	ss.width(4);
	ss.fill('0');
	ss << m_solveCounter;

	const path p(ss.str());
	create_directories(p);

	for(int i = 0; i < targets.size(); ++i)
	{
		cv::Mat tempMat;
		cv::normalize(targets[i], tempMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imwrite((p / (std::string("img_") + std::to_string(i) + ".png")).string(), tempMat);
	}

	{
		std::ofstream of((p / "B.txt").string());

		of << "[";

		const auto & matrix = getMatrixRows();

		for (auto itRowSet = matrix.begin(); itRowSet != matrix.end(); ++itRowSet)
		{
			for (auto itRow = (*itRowSet)->begin(); itRow != (*itRowSet)->end(); ++itRow)
			{
				if (itRowSet != matrix.begin() || itRow != (*itRowSet)->begin())
				{
					of << ",\n";
				}
				of << itRow->getUb();
			}
		}

		of << "]";
	}

	{
		std::vector<TemporaryJoinedVector<float*>::IteratorPair> ranges;

		for (const auto & mat : targets)
		{
			const auto begin = reinterpret_cast<float*>(mat.data);
			const auto end = reinterpret_cast<float*>(mat.data) + mat.rows * mat.cols;

			ranges.emplace_back(begin, end);
		}

		TemporaryJoinedVector<float*> tempVector(ranges);

		std::ofstream of((p / "target.txt").string());

		of << "[";

		for (size_t i = 0, end = tempVector.size(); i < end; ++i)
		{
			of << tempVector[i];

			if (i == end - 1)
			{
				break;
			}

			of << ", ";
		}

		of << "]";
	}

	OutStream((p / "Lb.txt").string()) << getLb();
	OutStream((p / "Ub.txt").string()) << getUb();

	OutStream((p / "sol.txt").string()) << getLastSolution();

	getCplex().exportModel((p / "model.lp").string().c_str());
}
