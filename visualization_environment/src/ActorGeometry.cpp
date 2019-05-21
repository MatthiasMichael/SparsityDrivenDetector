#include "ActorGeometry.h"

#include <osg/Geode>
#include <osg/ShapeDrawable>


ActorGeometry::ActorGeometry(osg::Vec3 position, osg::Vec3 size, osg::Vec4 color, osg::Vec2 indicatorSize) :
	m_position(position),
	m_size(size),
	m_color(color),
	m_indicatorSize(indicatorSize),
	m_material(new osg::Material())
{
	createGeometry();
}


void ActorGeometry::createGeometry()
{
	osg::ref_ptr<osg::Geode> actorGeode = new osg::Geode();
	this->addChild(actorGeode);

	osg::ref_ptr<osg::Box> box = new osg::Box(m_position, m_size[0], m_size[1], m_size[2]);
	osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(box);

	const osg::Vec3 indicatorCenter = m_position + osg::Vec3(0, 0, m_indicatorSize.y() / 2 - m_position.z());
	osg::ref_ptr<osg::Cylinder> cylinder = new osg::Cylinder(indicatorCenter, m_indicatorSize.x(), m_indicatorSize.y());
	osg::ref_ptr<osg::ShapeDrawable> cd = new osg::ShapeDrawable(cylinder);

	m_material->setDiffuse(osg::Material::FRONT, m_color);
	getOrCreateStateSet()->setAttribute(m_material);

	actorGeode->addDrawable(sd);
	actorGeode->addDrawable(cd);
}
