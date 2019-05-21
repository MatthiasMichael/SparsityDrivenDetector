#pragma once

/**
 * Always include this file instead of "OptimizationProblemFactory.h"!
 * Always add header files of new Factories to this header!
 * 
 * Explanation for the interested:
 * 
 * The linker is an idiot and happily excludes static constructors in statically linked libraries,
 * if the corresponding class is not referenced explicitly.
 * Therefore we have to enforce inclusion of these symbols.
 * The simplest way is to declare the static constructor inline in the following way:
 * 
 * static inline struct Constructor
 * {
 *		Constructor() { ... Do whatever static initialization is necessary ... }
 * } constructor;
 * 
 * This however requires C++17. I'm sorry but I can't find a better solution.
 */


#include "OptimizationProblemFactory_Single.h"
#include "OptimizationProblemFactory_SingleLayered.h"
#include "OptimizationProblemFactory_Multi.h"
#include "OptimizationProblemFactory_MultiLayered.h"


inline std::unique_ptr<OptimizationProblemFactory> makeFactory(const std::string & identifier)
{
	try
	{
		return OptimizationProblemFactory::fromIdentifier(identifier);
	}
	catch (const std::runtime_error &)
	{
		std::cout << "Warning: No factory registered for '" << identifier << "'." << std::endl;
		std::cout << "         Returning default: Single." << std::endl;
		return std::make_unique<OptimizationProblemFactory_Single>();
	}
}
