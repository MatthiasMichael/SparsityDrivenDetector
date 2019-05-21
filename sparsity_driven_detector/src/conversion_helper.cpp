#include "conversion_helper.h"

#include "OptimizationProblemDecorator.h"

const OptimizationProblem & toOptimizationProblem(const IOptimizationProblem & o)
{
	try
	{
		const auto & decorator = dynamic_cast<const OptimizationProblemDecorator &>(o);
		return decorator.getOptimizationProblem();
	}
	catch (const std::bad_cast &)
	{
		
	}

	try
	{
		const auto & optim = dynamic_cast<const OptimizationProblem &>(o);
		return optim;
	}
	catch(const std::bad_cast &)
	{
		
	}

	throw std::runtime_error("Conversion not possible.");
}
