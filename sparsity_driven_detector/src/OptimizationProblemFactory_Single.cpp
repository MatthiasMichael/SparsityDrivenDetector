#include "OptimizationProblemFactory_Single.h"

#include "OptimizationProblem_Single.h"


OptimizationProblemFactory_Single::OptimizationProblemFactory_Single(WrapInDecorators wrap) :
	OptimizationProblemFactory(wrap)
{
	// empty
}


OptimizationProblemFactory::Parameters OptimizationProblemFactory_Single::getDefaultParameters() const
{
	return Parameters{ 0 };
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory_Single::clone() const
{
	return std::make_unique<OptimizationProblemFactory_Single>();
}


std::unique_ptr<IOptimizationProblem> OptimizationProblemFactory_Single::createProblemImpl(const Dictionary & dict, const Parameters & param) const
{
	return std::make_unique<OptimizationProblem_Single>(dict, param);
}


std::unique_ptr<OptimizationProblem> OptimizationProblemFactory_Single::createProblemForLoading() const
{
	// Cannot call std::make_unique due to private default ctor
	return std::unique_ptr<OptimizationProblem>(new OptimizationProblem_Single());
}


bool OptimizationProblemFactory_Single::operator==(const OptimizationProblemFactory & other) const
{
	return dynamic_cast<const OptimizationProblemFactory_Single *>(&other) != nullptr;
}