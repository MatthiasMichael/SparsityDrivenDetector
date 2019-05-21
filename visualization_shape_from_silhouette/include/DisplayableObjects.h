#pragma once

#include <vector>

#include <osg/Material>

#include "OsgDisplayable.h"
#include "VoxelCluster.h"


class DisplayableObjects : public OsgDisplayable
{
public:
	enum ObjectDisplayKind
	{
		ActiveVoxels,
		Clusters
	};

	DisplayableObjects();
	DisplayableObjects(const std::vector<const sfs::Voxel *> & voxel, const std::vector<sfs::VoxelCluster> & clusters,
		ObjectDisplayKind displayKind);

	void setDisplayKind(ObjectDisplayKind displayKind) { m_displayKind = displayKind; }

	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	osg::ref_ptr<osg::Group> getRawVoxelGeometry() const;
	osg::ref_ptr<osg::Group> getVoxelClusterGeometry() const;

	static osg::ref_ptr<osg::Material> createVoxelMaterial();
	static osg::ref_ptr<osg::Material> createGhostVoxelMaterial();

private:
	std::vector<const sfs::Voxel *> m_voxel;
	std::vector<sfs::VoxelCluster> m_clusters;
	
	ObjectDisplayKind m_displayKind;

	static const osg::ref_ptr<osg::Material> s_voxelMaterial;
	static const osg::ref_ptr<osg::Material> s_ghostVoxelMaterial;
};