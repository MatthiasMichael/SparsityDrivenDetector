#pragma once

#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/ref_ptr>

#include "VoxelCluster.h"


class VoxelClusterGeometry : public osg::Group
{
public:
	VoxelClusterGeometry(const sfs::VoxelCluster & cluster);

	void createGeometry();
	void setMaterials(const osg::Material & material, const osg::Material & ghostMaterial);

private:
	osg::ref_ptr<osg::Geode> getVoxelGeode() const;
	osg::ref_ptr<osg::Geode> getBeamGeode() const;

private:
	sfs::VoxelCluster m_cluster;

	osg::ref_ptr<osg::Geode> m_clusterGeode;

	osg::ref_ptr<osg::Material> m_material;
	osg::ref_ptr<osg::Material> m_ghostMaterial;
};