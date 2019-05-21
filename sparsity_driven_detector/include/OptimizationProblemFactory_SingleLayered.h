#pragma once

#include "OptimizationProblemFactory.h"


class OptimizationProblemFactory_SingleLayered : public OptimizationProblemFactory
{
public:
	explicit OptimizationProblemFactory_SingleLayered(WrapInDecorators wrap = s_defaultWrap);
	std::unique_ptr<OptimizationProblemFactory> clone() const override;

	Parameters getDefaultParameters() const override;
	const char * getIdentifier() const override { return "singleLayered"; }

	std::unique_ptr<IOptimizationProblem> createProblemImpl(const Dictionary & dict, const Parameters & param) const override;
	std::unique_ptr<OptimizationProblem> createProblemForLoading() const override;

	bool operator==(const OptimizationProblemFactory & other) const override;

private:
	static inline struct Constructor
	{
		Constructor()
		{
			registerPrototype(std::make_unique<OptimizationProblemFactory_SingleLayered>());
		}
	} constructor;
};
