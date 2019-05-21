#include "DisplayableGridPoints.h"

#include <osg/Geode>
#include <osg/Material>
#include <osg/Shape>
#include <osg/ShapeDrawable>

#include <boost/qvm/all.hpp>

#include "qvm_eigen.h"
#include "qvm_osg.h"
#include "GridPoints.h"


DisplayableGridPoints::DisplayableGridPoints(const GridPoints & grid, double pointSize) :
	m_grid(grid),
	m_pointSize(pointSize)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableGridPoints::getGeometry() const
{
	osg::ref_ptr<osg::Geode> pointsGroup = new osg::Geode();

	osg::ref_ptr<osg::Material> material = new osg::Material();
	//material->setDiffuse(osg::Material::FRONT, osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	//material->setSpecular(osg::Material::FRONT, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
	//material->setAmbient(osg::Material::FRONT, osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	material->setEmission(osg::Material::FRONT, osg::Vec4(0.7f, 0.0f, 0.0f, 1.0f));
	//material->setShininess(osg::Material::FRONT, 25.0);

	pointsGroup->getOrCreateStateSet()->setAttribute(material);

	for (const auto & v : m_grid.getPoints())
	{
		const osg::Vec3 position = boost::qvm::convert_to<osg::Vec3>(v.get());
		
		osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(position, m_pointSize);
		osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(sphere);

		sd->setColor(osg::Vec4(1, 1, 0, 0.5));

		pointsGroup->addDrawable(sd);
	}

	return pointsGroup;
}
