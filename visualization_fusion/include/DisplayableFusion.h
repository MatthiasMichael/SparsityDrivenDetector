#pragma once

#include "OsgDisplayable.h"
#include "Fusion.h"
#include <osg/Material>


class DisplayableFusion : public OsgDisplayable
{
public:
	DisplayableFusion(const FusedSolution & solution);

	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	static osg::ref_ptr<osg::Material> createVoxelMaterial();
	static osg::ref_ptr<osg::Material> createGhostVoxelMaterial();
	static osg::ref_ptr<osg::Material> createIndicatorMaterial();

private:
	FusedSolution m_solution;

	static const osg::ref_ptr<osg::Material> s_voxelMaterial;
	static const osg::ref_ptr<osg::Material> s_ghostVoxelMaterial;
	static const osg::ref_ptr<osg::Material> s_indicatorMaterial;
	
};
