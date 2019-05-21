#include "SparsityDrivenDetectorPostProcessing.h"


SparsityDrivenDetectorPostProcessing::SparsityDrivenDetectorPostProcessing(const Parameters & parameters) :
	m_parameters(parameters)
{
	// empty
}


MergedSolution SparsityDrivenDetectorPostProcessing::postProcessSolution(const Solution & solution) const
{
	return MergedSolution{ solution.framenumber, solution.timestamp, mergeActors(solution.actors) };
}


std::vector<MergedSolutionActor> SparsityDrivenDetectorPostProcessing::mergeActors(
	const std::vector<SolutionActor> & actors) const
{
	std::vector<MergedSolutionActor> mergedActors;

	for (const auto a : actors)
	{
		mergedActors.push_back(MergedSolutionActor{ a.position, a.info, { a } });
	}

	bool needAnotherMerge = true;
	while (needAnotherMerge)
	{
		std::tie(mergedActors, needAnotherMerge) = mergeActors(mergedActors);
	}

	return mergedActors;
}


bool SparsityDrivenDetectorPostProcessing::shouldMerge(const MergedSolutionActor & a,
                                                       const MergedSolutionActor & b) const
{
	if(a.baseActors.front().info.objectClass != b.baseActors.front().info.objectClass)
	{
		return false;
	}

	ScalarType minSquaredDistance = std::numeric_limits<ScalarType>::max();

	for (int i = 0; i < a.baseActors.size(); ++i)
	{
		for (int j = 0; j < b.baseActors.size(); ++j)
		{
			const auto sqn = (a.baseActors[i].position.get() - b.baseActors[j].position.get()).squaredNorm();
			minSquaredDistance = std::min(minSquaredDistance, sqn);
		}
	}

	return minSquaredDistance < m_parameters.maxFusionDistance * m_parameters.maxFusionDistance;
}


MergedSolutionActor merge(const MergedSolutionActor & a, const MergedSolutionActor & b)
{
	auto baseA = a.baseActors;
	baseA.insert(baseA.end(), b.baseActors.begin(), b.baseActors.end());

	WorldVector pos = make_named<WorldVector>(0.f, 0.f, 0.f);
	for (const auto v : baseA)
	{
		pos = pos + v.position;
	}

	pos = make_named<WorldVector>((1.f / baseA.size()) * pos.get());

	return MergedSolutionActor{ pos, a.info, baseA };
}


std::pair<std::vector<MergedSolutionActor>, bool> SparsityDrivenDetectorPostProcessing::mergeActors(
	const std::vector<MergedSolutionActor> & actors) const
{
	bool mergedSomething = false;

	std::vector<bool> used(actors.size(), false);
	std::vector<MergedSolutionActor> mergedActors;

	for (int i = 0; i < actors.size(); ++i)
	{
		if (used[i])
		{
			continue;
		}

		auto a = actors[i];

		for (int j = i + 1; j < actors.size(); ++j)
		{
			auto b = actors[j];
			if (!used[j] && shouldMerge(a, b))
			{
				a = merge(a, b);
				used[j] = true;
				mergedSomething = true;
			}
		}

		mergedActors.push_back(a);
	}

	return std::make_pair(mergedActors, mergedSomething);
}
