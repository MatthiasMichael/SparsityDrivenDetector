#pragma once

#include <vector>

// Apparently CPLEX also defines this macro so we get a warning if we include math.h afterwards
#ifdef M_LOG2E
#undef M_LOG2E
#endif

#include "GeometryUtils.h"

#include "Environment.h"
#include "WorldCoordinateSystem_SDD.h"


class GridPoints
{
public:
	using Distance = NamedVectorTypes<ScalarType, 2>::Distance;
	using WorldVector = WorldCoordinateSystem::NamedVector;

	struct Parameters
	{
		Distance distancePoints;
		ScalarType groundPlaneHeight;

		friend bool operator==(const Parameters & lhs, const Parameters & rhs);
	};

	GridPoints(const Environment & env, const Parameters & parameters);
	GridPoints(const Roi3DF & roi, const Parameters & parameters);
	GridPoints(const Mesh & mesh, const Parameters & parameters);

	const std::vector<WorldVector> & getPoints() const { return m_points; }
	const WorldVector & getPoint(size_t idx) const { return m_points[idx]; }

	const Distance & getPointDistance() const { return m_parameters.distancePoints; }

	friend std::ostream & operator<<(std::ostream & os, const GridPoints & grid);
	friend std::istream & operator>>(std::istream & is, GridPoints & grid);

	friend bool operator==(const GridPoints & lhs, const GridPoints & rhs);

	static const ScalarType s_epsilon;

private:
	static GridPoints fromEnvironment(const Environment & environment, const Parameters & parameters);

private:
	Parameters m_parameters;

	size_t m_numPointsX;
	std::vector<WorldVector> m_points;
};
