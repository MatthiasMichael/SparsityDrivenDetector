#include "Mesh.h"


Roi3DF Mesh::calcBounds(const std::vector<Face> & faces)
{
	constexpr ScalarType maxValue = std::numeric_limits<ScalarType>::max();
	constexpr ScalarType minValue = std::numeric_limits<ScalarType>::lowest();

	auto maxPoint = make_named<WorldVector>(minValue, minValue, minValue);
	auto minPoint = make_named<WorldVector>(maxValue, maxValue, maxValue);

	for (const auto & face : faces)
	{
		for (const auto & v : face)
		{
			minPoint(0) = std::min(minPoint(0), v(0));
			minPoint(1) = std::min(minPoint(1), v(1));
			minPoint(2) = std::min(minPoint(2), v(2));

			maxPoint(0) = std::max(maxPoint(0), v(0));
			maxPoint(1) = std::max(maxPoint(1), v(1));
			maxPoint(2) = std::max(maxPoint(2), v(2));
		}
	}

	return Roi3DF(minPoint.get(), maxPoint.get());
}