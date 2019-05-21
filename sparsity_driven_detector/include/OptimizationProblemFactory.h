#pragma once

// Please include "OptimizationProblemFactoryCollection.h" instead!

#include <memory>

#include "Dictionary.h"

#include "OptimizationProblem.h"

class OptimizationProblemFactory
{
public:
	using Parameters = OptimizationProblem::Parameters;
	using PrototypeList = std::list<std::unique_ptr<OptimizationProblemFactory>>;
	using WrapInDecorators = std::unique_ptr<IOptimizationProblem>(*)(std::unique_ptr<IOptimizationProblem> &&);

	explicit OptimizationProblemFactory(WrapInDecorators wrap = s_defaultWrap);
	virtual ~OptimizationProblemFactory() = default;

	virtual std::unique_ptr<OptimizationProblemFactory> clone() const = 0;

	virtual Parameters getDefaultParameters() const = 0;
	virtual const char * getIdentifier() const = 0;

	std::unique_ptr<IOptimizationProblem> createProblem(const Dictionary & dict, const Parameters & param) const;

	// For now loading of a problem can only generate an undecorated problem
	virtual std::unique_ptr<OptimizationProblem> createProblemForLoading() const = 0;

	virtual bool operator==(const OptimizationProblemFactory & other) const = 0;

	void setDecoratorWrapper(WrapInDecorators wrap) { m_wrap = wrap; }

	void save(const std::string & filename) const;
	
	static std::unique_ptr<OptimizationProblemFactory> load(const std::string & filename);
	static std::unique_ptr<OptimizationProblemFactory> fromIdentifier(const std::string & identifier);
	
protected:
	static void registerPrototype(std::unique_ptr<OptimizationProblemFactory> && p);

	virtual std::unique_ptr<IOptimizationProblem> createProblemImpl(const Dictionary & dict, const Parameters & param) const = 0;

	static WrapInDecorators s_defaultWrap;

private:
	static PrototypeList & getPrototypes();

	WrapInDecorators m_wrap;
};