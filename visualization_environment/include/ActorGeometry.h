#pragma once

#include <osg/Group>
#include <osg/Material>
#include "WorldCoordinateSystem_SDD.h"


class ActorGeometry : public osg::Group
{
public:
	ActorGeometry(osg::Vec3 position, osg::Vec3 size, osg::Vec4 color, osg::Vec2 indicatorSize);

private:
	void createGeometry();

private:
	osg::Vec3 m_position;
	osg::Vec3 m_size;

	osg::Vec4 m_color;

	osg::Vec2 m_indicatorSize;

	osg::ref_ptr<osg::Material> m_material;
};
