#pragma once

#include <OsgDisplayable.h>

#include "Environment.h"


class DisplayableMesh : public OsgDisplayable
{
public:
	DisplayableMesh(const Mesh & navMesh, const osg::Vec4 & color, bool showWireFrame);
	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	Mesh m_navMesh;
	osg::Vec4 m_color;
	bool m_showWireFrame;
};
