#include "ShapeFromSilhouette.h"

#include <fstream>

#include "ApplicationTimer.h"

#include "voxel_clustering.h"

namespace sfs
{

	struct ShapeFromSilhouette::StringConstants
	{
		inline static constexpr auto MissingSpace = "Create space before accessing values!";

		inline static constexpr auto TP_UPDATE = "ShapeFromSilhouette::ProcessInput::Update";
		inline static constexpr auto TP_FILL = "ShapeFromSilhouette::ProcessInput::Fill";
		inline static constexpr auto TP_TOTAL = "ShapeFromSilhouette::ProcessInput";
	};


	ShapeFromSilhouette::Parameters::Parameters() :
		minSegmentation(0.05),
		maxClusterDistance(3)
	{
		// empty
	}


	ShapeFromSilhouette::ShapeFromSilhouette() : m_space(nullptr)
	{
		// empty
	}


	void ShapeFromSilhouette::createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels)
	{
		m_space.reset(new Space(area, voxelSize, cameraModels));
	}


	void ShapeFromSilhouette::createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh)
	{
		m_space.reset(new Space(area, voxelSize, cameraModels, staticMesh));
	}


	void ShapeFromSilhouette::loadSpace(const std::string & filename)
	{
		std::fstream filestream(filename, std::fstream::in | std::fstream::binary);

		m_space = std::make_unique<Space>();
		readBinary(filestream, *m_space);
	}


	void ShapeFromSilhouette::saveSpace(const std::string & filename) const
	{
		if (!hasSpace())
		{
			throw std::runtime_error(StringConstants::MissingSpace);
		}

		{
			std::fstream filestream(filename, std::fstream::out | std::fstream::binary | std::fstream::trunc);
			writeBinary(filestream, *m_space);
		}

		// Test, ob die Serialisierung funktioniert.
		/*
		{
		std::fstream readStream(filename, std::fstream::in | std::fstream::binary);
		Space * s = new Space();
		readBinary(readStream, *s);

		bool same = *s == *m_space;
		if(same)
		{
		std::cout << "Serialization OK! << std::endl;
		}
		else
		{
		std::cout << "Serialization Failed! << std::endl;
		}

		delete s;
		}*/
	}


	void ShapeFromSilhouette::processInput(const std::vector<cv::Mat> & imagesSegmentation, ExtendedVoxel::GeometryPredicate predicate)
	{
		AT_START(StringConstants::TP_TOTAL);

		m_clusterObjects.clear();

		AT_START(StringConstants::TP_UPDATE);
		m_space->update(imagesSegmentation, m_parameters.minSegmentation, predicate);
		AT_STOP(StringConstants::TP_UPDATE);

		AT_START(StringConstants::TP_FILL);
		m_clusterObjects = m_space->sequentialFill(m_parameters.maxClusterDistance);
		AT_STOP(StringConstants::TP_FILL);

		AT_STOP(StringConstants::TP_TOTAL);
	}


	std::vector<const ExtendedVoxel *> ShapeFromSilhouette::getActiveExtendedVoxels() const
	{
		return hasSpace() ? m_space->getActiveExtendedVoxels() : std::vector<const ExtendedVoxel *>{ };
	}


	std::vector<const Voxel *> ShapeFromSilhouette::getActiveVoxels() const
	{
		return hasSpace() ? m_space->getActiveVoxels() : std::vector<const Voxel *>{};
	}


	std::vector<const Voxel *> ShapeFromSilhouette::getActiveVoxelsFromClusters() const
	{
		return VoxelCluster::getVoxelFromClusters(m_clusterObjects);

	}


	const CameraSet & ShapeFromSilhouette::getCameraModels() const
	{
		if (!hasSpace())
		{
			throw std::runtime_error(StringConstants::MissingSpace);
		}

		return m_space->getCameraModels();
	}


	Roi3DF ShapeFromSilhouette::getArea() const
	{
		if (!hasSpace())
		{
			throw std::runtime_error(StringConstants::MissingSpace);
		}

		return m_space->getArea();
	}


	const Space & ShapeFromSilhouette::getSpace() const
	{
		if (!hasSpace())
		{
			throw std::runtime_error(StringConstants::MissingSpace);
		}

		return *m_space;
	}
}