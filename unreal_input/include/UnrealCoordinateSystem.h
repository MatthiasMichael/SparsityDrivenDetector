#pragma once

#include "WorldCoordinateSystem_SDD.h" // Use the same ScalarType

struct UnrealTag {};

using UnrealCoordinateSystem = CoordinateSystem<3, ScalarType, UnrealTag>;

using UnrealVector = NamedType<UnrealCoordinateSystem::Vector, UnrealCoordinateSystem, VectorTypeSkills>;
using UnrealTransform = NamedType<UnrealCoordinateSystem::AffineTransform, UnrealCoordinateSystem>;
