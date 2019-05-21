#include "OptimizationProblemFactory_MultiLayered.h"

#include "OptimizationProblem_MultiLayered.h"


OptimizationProblemFactory_MultiLayered::OptimizationProblemFactory_MultiLayered(WrapInDecorators wrap) :
	OptimizationProblemFactory(wrap)
{
	// empty
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory_MultiLayered::clone() const
{
	return std::make_unique<OptimizationProblemFactory_MultiLayered>();
}


OptimizationProblemFactory::Parameters OptimizationProblemFactory_MultiLayered::getDefaultParameters() const
{
	return Parameters{ 0.1f };
}


std::unique_ptr<IOptimizationProblem> OptimizationProblemFactory_MultiLayered::createProblemImpl(
	const Dictionary & dict, const Parameters & param) const
{
	return std::make_unique<OptimizationProblem_MultiLayered>(dict, param);
}


std::unique_ptr<OptimizationProblem> OptimizationProblemFactory_MultiLayered::createProblemForLoading() const
{
	// Cannot call std::make_unique due to private default ctor
	return std::unique_ptr<OptimizationProblem>(new OptimizationProblem_MultiLayered());
}


bool OptimizationProblemFactory_MultiLayered::operator==(const OptimizationProblemFactory & other) const
{
	return dynamic_cast<const OptimizationProblemFactory_MultiLayered *>(&other) != nullptr;
}
