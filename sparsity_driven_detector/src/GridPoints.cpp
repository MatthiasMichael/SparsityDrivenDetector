#include "GridPoints.h"

#include <fstream>


bool operator==(const GridPoints::Parameters & lhs, const GridPoints::Parameters & rhs)
{
	return
		lhs.distancePoints == rhs.distancePoints &&
		lhs.groundPlaneHeight == rhs.groundPlaneHeight;
}


const ScalarType GridPoints::s_epsilon = static_cast<ScalarType>(0.0001);


GridPoints::GridPoints(const Environment & env, const Parameters & parameters) :
	GridPoints(fromEnvironment(env, parameters))
{
	// empty
}


GridPoints::GridPoints(const Roi3DF & roi, const Parameters & parameters) :
	m_parameters(parameters),
	m_numPointsX(0)
{
	const ScalarType & z = parameters.groundPlaneHeight;
	const Distance & dist = parameters.distancePoints;

	if(dist(0) == 0. || dist(1) == 0.)
	{
		return;
	}

	for (ScalarType x = roi.x1, endX = roi.x2 + s_epsilon; x <= endX; x += dist(0))
	{
		++m_numPointsX;

		for (ScalarType y = roi.y1, endY = roi.y2 + s_epsilon; y <= endY; y += dist(1))
		{
			m_points.push_back(make_named<WorldVector>(x, y, z));
		}
	}
}


GridPoints::GridPoints(const Mesh & mesh, const Parameters & parameters) :
	m_parameters(parameters),
	m_numPointsX(0)
{
	const ScalarType & verticalOffset = parameters.groundPlaneHeight;
	const Distance & dist = parameters.distancePoints;

	if (dist(0) == 0. || dist(1) == 0.)
	{
		return;
	}

	const auto roi = mesh.getBoundingBox();

	for (ScalarType x = roi.x1, endX = roi.x2 + s_epsilon; x <= endX; x += dist(0))
	{
		++m_numPointsX;

		for (ScalarType y = roi.y1, endY = roi.y2 + s_epsilon; y <= endY; y += dist(1))
		{
			for (const auto & f : mesh)
			{
				const Vector3 candidatePoint(x, y, 0);
				const Vector3 direction(0, 0, -1);

				auto intersectionPoint = f.intersectInfiniteLine(candidatePoint, direction);

				if (intersectionPoint)
				{
					auto p = intersectionPoint.value();
					auto gridPoint = make_named<WorldVector>(p(0), p(1), p(2));
					m_points.push_back(gridPoint);
				}
			}
		}
	}
}


GridPoints GridPoints::fromEnvironment(const Environment & environment, const Parameters & parameters)
{
	if(!environment.getNavMesh().empty())
	{
		return GridPoints(environment.getNavMesh(), parameters);
	}

	if(!environment.getNavMesh().empty())
	{
		return GridPoints(environment.getNavMesh(), parameters);
	}

	return GridPoints(Roi3DF(0, 0, 0, 100, 100, 100), parameters); // Maybe not thought out...

}


std::ostream & operator<<(std::ostream & os, const GridPoints & grid)
{
	using std::endl;

	os << grid.m_parameters.distancePoints.get()(0) << " " << grid.m_parameters.distancePoints.get()(1) << endl;
	os << grid.m_parameters.groundPlaneHeight << endl;
	os << grid.m_numPointsX << " " << grid.m_points.size() / grid.m_numPointsX << endl;

	os << grid.m_points.size();
	for (const auto p : grid.m_points)
	{
		os << endl;
		os << p.get()(0) << " " << p.get()(1) << " " << p.get()(2);
	}

	return os;
}


std::istream & operator>>(std::istream & is, GridPoints & grid)
{
	is >> grid.m_parameters.distancePoints.get()(0) >> grid.m_parameters.distancePoints.get()(1);
	is >> grid.m_parameters.groundPlaneHeight;

	size_t dummy = 0;
	is >> grid.m_numPointsX >> dummy;

	is >> dummy;
	grid.m_points.resize(dummy);

	for (size_t i = 0; i < dummy; ++i)
	{
		is >> grid.m_points[i].get()(0) >> grid.m_points[i].get()(1) >> grid.m_points[i].get()(2);
	}

	return is;
}


bool operator==(const GridPoints & lhs, const GridPoints & rhs)
{
	return lhs.m_parameters == rhs.m_parameters && lhs.m_numPointsX == rhs.m_numPointsX && lhs.m_points == rhs.m_points;
}
