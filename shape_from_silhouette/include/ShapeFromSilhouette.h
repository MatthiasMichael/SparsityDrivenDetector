#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Mesh.h"
#include "Space.h"

namespace sfs
{
	class ShapeFromSilhouette
	{
	public:
		struct StringConstants;

		struct Parameters
		{
			Parameters();

			double minSegmentation;
			int maxClusterDistance;
		};

	public:
		ShapeFromSilhouette();

		void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels);
		void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh);

		void loadSpace(const std::string & filename);
		void saveSpace(const std::string & filename) const;

		bool hasSpace() const { return m_space != nullptr; }

		Parameters & getChangeableParameters() { return m_parameters; }
		void processInput(const std::vector<cv::Mat> & imagesSegmentation, ExtendedVoxel::GeometryPredicate predicate);

		std::vector<const ExtendedVoxel *> getActiveExtendedVoxels() const;

		std::vector<const Voxel *> getActiveVoxels() const;
		std::vector<const Voxel *> getActiveVoxelsFromClusters() const;

		const std::vector<VoxelCluster> & getCluster() const { return m_clusterObjects; }

		const Space & getSpace() const;
		Roi3DF getArea() const;
		const CameraSet & getCameraModels() const;

	private:
		std::unique_ptr<Space> m_space;

		Parameters m_parameters;

		std::vector<VoxelCluster> m_clusterObjects;
	};
}