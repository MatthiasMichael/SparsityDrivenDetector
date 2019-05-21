#pragma once

#include "WorldCoordinateSystem_SDD.h"
#include "Template.h"
#include "Frame.h"


struct SolutionActor
{
	WorldVector position;
	Template::Info info;
	
	size_t gridIndex;
	size_t templateIndex;
};

struct Solution
{
	Framenumber framenumber;
	std::string timestamp;

	std::vector<SolutionActor> actors;
};