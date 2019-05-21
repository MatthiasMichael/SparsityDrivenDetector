#pragma once

#include "FunctionalOsgDisplayDescriptor.h"

#include "DisplayableGridPoints.h"
#include "DisplayableSolution.h"


inline auto gridDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Grid",
	[](Parametrizable & d)
	{
		d.addDoubleParameter("Point Size", 1, 1000, 10, 1);
	},
	[](const Parametrizable & d, const GridPoints & grid)
	{
		return std::make_unique<DisplayableGridPoints>(grid, d.getDoubleParameter("Point Size"));
	}
);

inline auto solutionDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Initial Solution (Step 0)",
	[](Parametrizable &) {},
	[](const Parametrizable &, const Solution & solution)
	{
		return std::make_unique<DisplayableSolution>(solution);
	}
);

inline auto mergedSolutionDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Merged Solution (Step 1)",
	[](Parametrizable &) {},
	[](const Parametrizable &, const MergedSolution & solution)
	{
		return std::make_unique<DisplayableMergedSolution>(solution);
	}
);
