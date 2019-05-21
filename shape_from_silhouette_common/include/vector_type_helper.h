#pragma once

#include "WorldCoordinateSystem_SDD.h"

#include "cuda_vector_functions_interop.h"

namespace detail
{
	template <typename scalar_type>
	struct type_picker { };

	template<>
	struct type_picker<float>
	{
		using type = float3;
		inline static auto make = make_float3;
	};

	template<>
	struct type_picker<double>
	{
		using type = double3;
		inline static auto make = make_double3;
	};
}

using ScalarType3 = typename detail::type_picker<ScalarType>::type;