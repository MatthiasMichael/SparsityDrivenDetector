#pragma once

#include "WorldCoordinateSystem_SDD.h"
#include "Template.h"
#include "Solution.h"
#include "SparsityDrivenDetector.h"


struct MergedSolutionActor
{
	WorldVector position;
	Template::Info info;

	std::vector<SolutionActor> baseActors;
};

struct MergedSolution
{
	Framenumber framenumber;
	std::string timestamp;

	std::vector<MergedSolutionActor> actors;
};

class SparsityDrivenDetectorPostProcessing
{
public:
	struct Parameters
	{
		ScalarType maxFusionDistance;
	};

	SparsityDrivenDetectorPostProcessing(const Parameters & parameters);

	MergedSolution postProcessSolution(const Solution & solution) const;

private:
	std::vector<MergedSolutionActor> mergeActors(const std::vector<SolutionActor> & actors) const;
	bool shouldMerge(const MergedSolutionActor & a, const MergedSolutionActor & b) const;
	std::pair<std::vector<MergedSolutionActor>, bool> mergeActors(const std::vector<MergedSolutionActor> & actors) const;

private:
	Parameters m_parameters;
};