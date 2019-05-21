#pragma once

#include "FunctionalOsgDisplayDescriptor.h"

#include "DisplayableFusion.h"


inline auto fusedSolutionDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Fused Solution (Step 2)",
	[](Parametrizable & d)
	{
		
	},
	[](const Parametrizable & d, const FusedSolution & solution)
	{
		return std::make_unique<DisplayableFusion>(solution);
	}
);

