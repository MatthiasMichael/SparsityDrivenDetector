#include "Environment.h"

#include <iomanip>

#include "serialization_helper.h"


Environment::Environment(const Mesh & staticMesh, const Mesh & navMesh, const CameraSet & cameras) :
	m_staticMesh(staticMesh),
	m_navMesh(navMesh),
	m_cameras(cameras)
{
	// empty
}


void Environment::save(const std::string & filename) const
{
	using namespace boost::filesystem;

	const auto temp = makeTempDir(path(filename).parent_path());

	{
		std::ofstream os((temp / "map.txt").string());

		throw std::runtime_error("Not Implemented"); // TODO
		/*os << m_map.size() << std::endl;
		for(const auto & poly : m_map)
		{
			for(const auto & point : poly)
			{
				os << std::setprecision(std::numeric_limits<std::decay_t<decltype(point(0))>>::max_digits10);
				os << point(0) << " " << point(1) << " " << point(2) << std::endl;
			}
		}*/
	}

	{
		std::ofstream os((temp / "navMesh.txt").string());

		throw std::runtime_error("Not Implemented"); // TODO
		/*os << m_navMesh.size() << std::endl;
		for (const auto & poly : m_navMesh)
		{
			for (const auto & point : poly)
			{
				os << std::setprecision(std::numeric_limits<std::decay_t<decltype(point(0))>>::max_digits10);
				os << point(0) << " " << point(1) << " " << point(2) << std::endl;
			}
		}*/
	}

	std::ofstream((temp / "cameras.txt").string()) << m_cameras;
		
	zipDir(temp, filename);

	remove_all(temp);
}


Environment Environment::load(const std::string & filename)
{
	using namespace boost::filesystem;

	const auto temp = makeTempDir(path(filename).parent_path());

	unzipDir(filename, temp);

	Environment e; 
	throw std::runtime_error("Not Implemented"); // TODO

	/*{
		std::ifstream is((temp / "map.txt").string());
		size_t size;
		is >> size;
		e.m_map = PolygonVector(size, Polygon{3, WorldVector{}});

		for(int i = 0; i < size; ++i)
		{
			for(int j = 0; j < 3; ++j)
			{
				is >> e.m_map[i][j](0) >> e.m_map[i][j](1) >> e.m_map[i][j](2);
			}
		}
	}*/

	{
		std::ifstream is((temp / "navMesh.txt").string());
		size_t size;
		is >> size;
		throw std::runtime_error("Not Implemented"); // TODO
		/*e.m_map = PolygonVector(size, Polygon{ 3, WorldVector{} });

		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				is >> e.m_map[i][j](0) >> e.m_map[i][j](1) >> e.m_map[i][j](2);
			}
		}*/
	}

	std::ifstream((temp / "cameras.txt").string()) >> e.m_cameras;

	remove_all(temp);

	return e;
}


bool operator==(const Environment & lhs, const Environment & rhs) 
{
	const auto & lVert = lhs.m_staticMesh;
	const auto & rVert = rhs.m_staticMesh;

	if(lVert.size() != rVert.size())
	{
		return false;
	}

	throw std::runtime_error("Not implemented");

	/*for(int i_poly = 0; i_poly < lVert.size(); ++i_poly)
	{
		if(lVert[i_poly].size() != rVert[i_poly].size())
		{
			return false;
		}
		for(int i_point = 0; i_point < lVert[i_poly].size(); ++i_point)
		{
			if( !(lVert[i_poly][i_point] == rVert[i_poly][i_point]))
			{
				return false;
			}
		}
	}*/
	
	// Roi is only dependant on m_map
	/*if(lhs.m_roi != rhs.m_roi)
	{
		return false;
	}*/

	const auto & lCams = lhs.m_cameras;
	const auto & rCams = rhs.m_cameras;

	if(lCams != rCams)
	{
		return false;
	}
	/*if(lCams.size() != rCams.size())
	{
		return false;
	}

	for(int i = 0; i < lCams.size(); ++i)
	{
		if(!(lCams[i] == rCams[i]))
		{
			return false;
		}
	}*/

	return true;
}
