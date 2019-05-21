#pragma once

#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/ref_ptr>

#include "Voxel.h"


class VoxelGeometry : public osg::Group
{
public:
	VoxelGeometry(const sfs::Voxel * pVoxel);

	void createGeometry();
	void setMaterial(const osg::Material & material);

private:
	const sfs::Voxel * mep_voxel;

	osg::ref_ptr<osg::Geode> m_voxelGeode;
	osg::ref_ptr<osg::Material> m_material;
};