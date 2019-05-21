#pragma once

#include "IOptimizationProblem.h"
#include "OptimizationProblem.h"

// An IOptimizationProblem is either a concrete implementation or a decorator that contains a
// pointer to an OptimizationProblem. This functions helps in casting...
const OptimizationProblem & toOptimizationProblem(const IOptimizationProblem & o);
