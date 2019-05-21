#include "DisplayableMesh.h"

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osgUtil/SmoothingVisitor>

#include "qvm_eigen.h"
#include "qvm_osg.h"

#include "enumerate.h"
#include "osg_utils.h"
#include <boost/qvm/quat_operations.hpp>


osg::ref_ptr<osg::Geometry> createPolygon(const Face & f, const osg::Vec4 & color)
{
	osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array();
	osg::ref_ptr<osg::DrawElementsUInt> polygonFace = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
	osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

	for(const auto & [faceIdx, vertex] : enumerate(f.getVertices()))
	{
		points->push_back(osg::Vec3(vertex(0), vertex(1), vertex(2)));
		polygonFace->push_back(faceIdx);
		colors->push_back(color);
	}

	osg::ref_ptr<osg::Geometry> osg_polygon = new osg::Geometry();
	osg_polygon->setVertexArray(points.get());
	osg_polygon->addPrimitiveSet(polygonFace);
	osg_polygon->setColorArray(colors.get());
	osg_polygon->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

	return osg_polygon;
}


osg::ref_ptr<osg::Geometry> createLines(const Face & f, const osg::Vec4 & color)
{
	osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array();
	osg::ref_ptr<osg::DrawElementsUInt> lineFace = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_LOOP, 0);
	osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

	for (const auto &[faceIdx, vertex] : enumerate(f.getVertices()))
	{
		points->push_back(osg::Vec3(vertex(0), vertex(1), vertex(2)));
		lineFace->push_back(faceIdx);
		colors->push_back(color);
	}

	osg::ref_ptr<osg::Geometry> osg_polygon = new osg::Geometry();
	osg_polygon->setVertexArray(points.get());
	osg_polygon->addPrimitiveSet(lineFace);
	osg_polygon->setColorArray(colors.get());
	osg_polygon->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

	return osg_polygon;
}



DisplayableMesh::DisplayableMesh(const Mesh & navMesh, const osg::Vec4 & color, bool showWireFrame) : 
	m_navMesh(navMesh),
	m_color(color),
	m_showWireFrame(showWireFrame)
{
	// empty
}

osg::ref_ptr<osg::Group> DisplayableMesh::getGeometry() const
{
	osg::ref_ptr<osg::Group> ret = new osg::Group();

	osg::ref_ptr<osg::Geode> faceGeode = new osg::Geode();
	ret->addChild(faceGeode);

	for (const auto & v : m_navMesh.getFaces())
	{
		osg::ref_ptr<osg::Geometry> face = createPolygon(v, m_color);
		faceGeode->addChild(face);

		osg::ref_ptr<osg::Material> material = new osg::Material();
		material->setDiffuse(osg::Material::FRONT_AND_BACK, m_color);
		//material->setSpecular(osg::Material::FRONT, osg::Vec4(0.0f, 0.1f, 0.0f, 1.0f));
		material->setAmbient(osg::Material::FRONT, m_color);
		//material->setEmission(osg::Material::FRONT, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
		material->setShininess(osg::Material::FRONT, 25.0);

		face->getOrCreateStateSet()->setAttribute(material);
		face->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);
	}

	osg::ref_ptr<osgUtil::SmoothingVisitor> visitor = new osgUtil::SmoothingVisitor;
	visitor->setCreaseAngle(0);
	faceGeode->accept(*visitor);

	if (m_showWireFrame)
	{
		osg::ref_ptr<osg::Geode> lineMeshGeode = new osg::Geode();
		ret->addChild(lineMeshGeode);

		for (const auto & v : m_navMesh.getFaces())
		{
			osg::ref_ptr<osg::Geometry> lines = createLines(v, m_color * 0.5);
			lineMeshGeode->addChild(lines);

			setAttributesNonLightingBlendable(lineMeshGeode);
			lineMeshGeode->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
		}
	}

	return ret;
}
