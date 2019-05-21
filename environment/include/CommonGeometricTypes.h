#pragma once

#include "GeometricTypes.h"
#include "WorldCoordinateSystem_SDD.h"
#include "GeometryUtils.h"

using Vector2 = GeometricTypes<ScalarType, 2>::Vector;
using Vector3 = GeometricTypes<ScalarType, 3>::Vector;
using Vector4 = GeometricTypes<ScalarType, 4>::Vector;

using Matrix22 = GeometricTypes<ScalarType, 2>::Matrix;
using Matrix33 = GeometricTypes<ScalarType, 3>::Matrix;
using Matrix44 = GeometricTypes<ScalarType, 4>::Matrix;

using AffineTransform = GeometricTypes<ScalarType, 3>::Transform::Affine;
using Translation = GeometricTypes<ScalarType, 3>::Transform::Translation;

using Degrees = NamedScalarTypes<ScalarType>::Degrees;
using Radians = NamedScalarTypes<ScalarType>::Radians;
using Distance = NamedVectorTypes<ScalarType, 2>::Distance;