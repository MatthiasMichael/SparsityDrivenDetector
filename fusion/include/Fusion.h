#pragma once

#include "WorldCoordinateSystem_SDD.h"
#include "SparsityDrivenDetectorPostProcessing.h"
#include "VoxelCluster.h"


struct FusedSolutionActor
{
	MergedSolutionActor actor;
	sfs::VoxelCluster volume;
};

struct FusedSolution
{
	Framenumber framenumber;
	std::string timestamp;

	std::vector<FusedSolutionActor> actors;
};

FusedSolution fuse(const MergedSolution & solution, std::vector<sfs::VoxelCluster> volumes);