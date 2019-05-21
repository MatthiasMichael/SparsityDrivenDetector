#include "OptimizationProblemFactory_Multi.h"

#include "OptimizationProblem_Multi.h"


OptimizationProblemFactory_Multi::OptimizationProblemFactory_Multi(WrapInDecorators wrap) :
	OptimizationProblemFactory(wrap)
{
	// empty
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory_Multi::clone() const
{
	return std::make_unique<OptimizationProblemFactory_Multi>();
}


OptimizationProblemFactory::Parameters OptimizationProblemFactory_Multi::getDefaultParameters() const
{
	return Parameters{ 0 };
}


std::unique_ptr<IOptimizationProblem> OptimizationProblemFactory_Multi::createProblemImpl(
	const Dictionary & dict, const Parameters & param) const
{
	return std::make_unique<OptimizationProblem_Multi>(dict, param);
}


std::unique_ptr<OptimizationProblem> OptimizationProblemFactory_Multi::createProblemForLoading() const
{
	// Cannot call std::make_unique due to private default ctor
	return std::unique_ptr<OptimizationProblem>(new OptimizationProblem_Multi());
}


bool OptimizationProblemFactory_Multi::operator==(const OptimizationProblemFactory & other) const
{
	return dynamic_cast<const OptimizationProblemFactory_Multi *>(&other) != nullptr;
}
