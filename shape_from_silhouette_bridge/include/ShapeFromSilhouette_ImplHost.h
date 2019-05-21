#pragma once

#include <memory>

#include "ShapeFromSilhouette_Impl.h"

#include "ShapeFromSilhouette.h"
#include "ExtendedVoxel.h"


namespace sfs
{
	class ShapeFromSilhouette;

	class ShapeFromSilhouette_ImplHost : public ShapeFromSilhouette_Impl
	{
	public:
		ShapeFromSilhouette_ImplHost();

		void setParameters(const Parameters & parameters) override;

		void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels) override;
		void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh) override;

		bool hasSpace() const override;

		void processInput(const std::vector<cv::Mat> & imagesSegmentation) override;

		std::vector<const Voxel *> getActiveVoxels() const override;
		std::vector<const Voxel *> getActiveVoxelsFromClusters() const override;

		const std::vector<VoxelCluster> & getCluster() const override;
		
		Roi3DF getArea() const override;
		const CameraSet & getCameraModels() const override;

		void saveDebugImages(const std::string & path) const override;
	private:
		ShapeFromSilhouette m_sfs;
		ExtendedVoxel::GeometryPredicate m_predicate;
	};
}
