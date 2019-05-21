#include "OptimizationProblemFactory_SingleLayered.h"

#include "OptimizationProblem_SingleLayered.h"


OptimizationProblemFactory_SingleLayered::OptimizationProblemFactory_SingleLayered(WrapInDecorators wrap) :
	OptimizationProblemFactory(wrap)
{
	// empty
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory_SingleLayered::clone() const
{
	return std::make_unique<OptimizationProblemFactory_SingleLayered>();
}


OptimizationProblemFactory::Parameters OptimizationProblemFactory_SingleLayered::getDefaultParameters() const
{
	return Parameters{ 0.1f };
}


std::unique_ptr<IOptimizationProblem> OptimizationProblemFactory_SingleLayered::createProblemImpl(const Dictionary & dict, const Parameters & param) const
{
	return std::make_unique<OptimizationProblem_SingleLayered>(dict, param);
}


std::unique_ptr<OptimizationProblem> OptimizationProblemFactory_SingleLayered::createProblemForLoading() const
{
	return std::unique_ptr<OptimizationProblem>(new OptimizationProblem_SingleLayered());
}


bool OptimizationProblemFactory_SingleLayered::operator==(const OptimizationProblemFactory & other) const
{
	return dynamic_cast<const OptimizationProblemFactory_SingleLayered *>(&other) != nullptr;
}
