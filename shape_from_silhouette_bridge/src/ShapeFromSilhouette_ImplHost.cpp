#include "ShapeFromSilhouette_ImplHost.h"

#include "ShapeFromSilhouette.h"
#include "VoxelPredicates.h"


namespace sfs
{
	ShapeFromSilhouette_ImplHost::ShapeFromSilhouette_ImplHost() :
		m_predicate(defaultPredicate())
	{
		// empty
	}


	void ShapeFromSilhouette_ImplHost::setParameters(const Parameters & parameters)
	{
		m_sfs.getChangeableParameters().minSegmentation = parameters.minSegmentation;
		m_sfs.getChangeableParameters().maxClusterDistance = parameters.maxClusterDistance;
	}


	void ShapeFromSilhouette_ImplHost::createSpace(const Roi3DF & area, const float3 & voxelSize,
	                                               const CameraSet & cameraModels)
	{
		m_sfs.createSpace(area, voxelSize, cameraModels);
	}


	void ShapeFromSilhouette_ImplHost::createSpace(const Roi3DF & area, const float3 & voxelSize,
	                                               const CameraSet & cameraModels, const Mesh & staticMesh)
	{
		m_sfs.createSpace(area, voxelSize, cameraModels, staticMesh);
	}


	bool ShapeFromSilhouette_ImplHost::hasSpace() const
	{
		return m_sfs.hasSpace();
	}


	void ShapeFromSilhouette_ImplHost::processInput(const std::vector<cv::Mat> & imagesSegmentation)
	{
		m_sfs.processInput(imagesSegmentation, m_predicate);
	}


	std::vector<const Voxel *> ShapeFromSilhouette_ImplHost::getActiveVoxels() const
	{
		return m_sfs.getActiveVoxels();
	}


	std::vector<const Voxel *> ShapeFromSilhouette_ImplHost::getActiveVoxelsFromClusters() const
	{
		return m_sfs.getActiveVoxelsFromClusters();
	}


	const std::vector<VoxelCluster> & ShapeFromSilhouette_ImplHost::getCluster() const
	{
		return m_sfs.getCluster();
	}


	Roi3DF ShapeFromSilhouette_ImplHost::getArea() const
	{
		return m_sfs.getArea();
	}


	const CameraSet & ShapeFromSilhouette_ImplHost::getCameraModels() const
	{
		return m_sfs.getCameraModels();
	}


	void ShapeFromSilhouette_ImplHost::saveDebugImages(const std::string & path) const
	{
		throw std::runtime_error("Not implemented");
		//m_sfs.getSpace().debugDrawAllConvexHulls(path);
	}
}
