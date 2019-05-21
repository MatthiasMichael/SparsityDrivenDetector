#pragma once

#include "WorldCoordinateSystem.h"
#include "vec_template.h"
#include "vec_eigen.h"

using ScalarType = float;

using WorldCoordinateSystem = WorldCoordinateSystem_T<ScalarType>;

using WorldVector = WorldVector_T<ScalarType>;
using WorldTransform = WorldTransform_T<ScalarType>;

// Template Specializations currently only used to create a detail::Face from Environment::Polygon
template<>
inline float x(const WorldVector & v)
{
	return static_cast<float>(v(0));
}


template<>
inline float y(const WorldVector & v)
{
	return static_cast<float>(v(1));
}


template<>
inline float z(const WorldVector & v)
{
	return static_cast<float>(v(2));
}


template<>
inline WorldVector make_vec3D(float x, float y, float z)
{
	return make_named<WorldVector>(x, y, z);
}