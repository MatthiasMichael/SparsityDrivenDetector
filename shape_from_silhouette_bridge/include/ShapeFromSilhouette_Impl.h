#pragma once

#include <opencv2/imgproc.hpp>

#include "Roi3DF.h"

#include "cuda_vector_functions_interop.h"

#include "IdentifiableCamera.h"
#include "Mesh.h"

#include "Voxel.h"
#include "VoxelCluster.h"

namespace sfs
{
	class ShapeFromSilhouette_Impl
	{
	public:
		struct Parameters
		{
			float minSegmentation;
			int maxClusterDistance;
		};

		virtual ~ShapeFromSilhouette_Impl() = default;

		virtual void setParameters(const Parameters & parameters) = 0;

		virtual void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels) = 0;
		virtual void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh) = 0;

		virtual bool hasSpace() const = 0;

		virtual void processInput(const std::vector<cv::Mat> & imagesSegmentation) = 0;

		virtual std::vector<const Voxel *> getActiveVoxels() const = 0;
		virtual std::vector<const Voxel *> getActiveVoxelsFromClusters() const = 0;

		virtual const std::vector<VoxelCluster> & getCluster() const = 0;

		virtual Roi3DF getArea() const = 0;
		virtual const CameraSet & getCameraModels() const = 0;

		virtual void saveDebugImages(const std::string & path) const = 0;
	};
}