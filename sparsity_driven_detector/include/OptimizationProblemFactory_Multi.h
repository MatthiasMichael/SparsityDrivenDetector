#pragma once

#include "OptimizationProblemFactory.h"

class OptimizationProblemFactory_Multi : public OptimizationProblemFactory
{
public:
	explicit OptimizationProblemFactory_Multi(WrapInDecorators wrap = s_defaultWrap);

	std::unique_ptr<OptimizationProblemFactory> clone() const override;

	Parameters getDefaultParameters() const override;
	const char * getIdentifier() const override { return "multi"; }

	std::unique_ptr<IOptimizationProblem> createProblemImpl(const Dictionary & dict, const Parameters & param) const override;
	std::unique_ptr<OptimizationProblem> createProblemForLoading() const override;

	bool operator==(const OptimizationProblemFactory & other) const override;

private:
	static inline struct Constructor
	{
		Constructor()
		{
			registerPrototype(std::make_unique<OptimizationProblemFactory_Multi>());
		}

	} constructor;
};
