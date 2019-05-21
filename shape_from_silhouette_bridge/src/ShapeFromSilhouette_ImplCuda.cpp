#include "ShapeFromSilhouette_ImplCuda.h"

#include "Face.cuh"


namespace sfs
{
	void ShapeFromSilhouette_ImplCuda::setParameters(const Parameters & parameters)
	{
		m_sfs.getChangeableParameters().minSegmentation = parameters.minSegmentation;
		m_sfs.getChangeableParameters().maxClusterDistance = parameters.maxClusterDistance;
	}


	void ShapeFromSilhouette_ImplCuda::createSpace(const Roi3DF & area, const float3 & voxelSize,
	                                               const CameraSet & cameraModels)
	{
		m_sfs.createSpace(area, voxelSize, cameraModels, { });
	}


	void ShapeFromSilhouette_ImplCuda::createSpace(const Roi3DF & area, const float3 & voxelSize,
	                                               const CameraSet & cameraModels, const Mesh & staticMesh)
	{
		std::vector<cuda::Face> faces;

		for (const auto & f : staticMesh)
		{
			faces.push_back(cuda::Face::create(f.begin(), f.end()));
		}

		m_sfs.createSpace(area, voxelSize, cameraModels, faces);
	}


	bool ShapeFromSilhouette_ImplCuda::hasSpace() const
	{
		return m_sfs.hasSpace();
	}


	void ShapeFromSilhouette_ImplCuda::processInput(const std::vector<cv::Mat> & imagesSegmentation)
	{
		m_sfs.processInput(imagesSegmentation);
	}


	std::vector<const Voxel *> ShapeFromSilhouette_ImplCuda::getActiveVoxels() const
	{
		return m_sfs.getActiveVoxels();
	}


	std::vector<const Voxel *> ShapeFromSilhouette_ImplCuda::getActiveVoxelsFromClusters() const
	{
		return m_sfs.getActiveVoxelsFromClusters();
	}


	const std::vector<VoxelCluster> & ShapeFromSilhouette_ImplCuda::getCluster() const
	{
		return m_sfs.getCluster();
	}


	Roi3DF ShapeFromSilhouette_ImplCuda::getArea() const
	{
		return m_sfs.getArea();
	}


	const CameraSet & ShapeFromSilhouette_ImplCuda::getCameraModels() const
	{
		return m_sfs.getCameraModels();
	}


	void ShapeFromSilhouette_ImplCuda::saveDebugImages(const std::string & path) const
	{
		m_sfs.getSpace().saveDebugImages(path);
	}
}
