#include "OptimizationProblemFactory.h"

OptimizationProblemFactory::WrapInDecorators OptimizationProblemFactory::s_defaultWrap = 
	[](std::unique_ptr<IOptimizationProblem> && p)
	{
		return std::move(p);
	};


OptimizationProblemFactory::OptimizationProblemFactory(WrapInDecorators wrap /*= s_defaultWrap*/) :
	m_wrap(wrap)
{
	// empty
}


std::unique_ptr<IOptimizationProblem> OptimizationProblemFactory::createProblem(const Dictionary & dict,
	const Parameters & param) const
{
	return m_wrap(createProblemImpl(dict, param));
}


void OptimizationProblemFactory::save(const std::string & filename) const
{
	std::ofstream(filename) << getIdentifier();
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory::load(const std::string & filename)
{
	std::string identifier;
	std::ifstream(filename) >> identifier;

	return fromIdentifier(identifier);
	
}


std::unique_ptr<OptimizationProblemFactory> OptimizationProblemFactory::fromIdentifier(const std::string & identifier)
{
	const auto & prototypes = getPrototypes();
	for (const auto & p : prototypes)
	{
		if (identifier == p->getIdentifier())
		{
			return p->clone();
		}
	}

	throw std::runtime_error("Invalid identifier found during load: " + identifier);
}


void OptimizationProblemFactory::registerPrototype(std::unique_ptr<OptimizationProblemFactory> && p)
{
	getPrototypes().push_back(std::move(p));
}


OptimizationProblemFactory::PrototypeList & OptimizationProblemFactory::getPrototypes()
{
	static PrototypeList prototypes;
	return prototypes;
}
