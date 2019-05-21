#pragma once

#include <OsgDisplayable.h>

#include "GridPoints.h"

class DisplayableGridPoints : public OsgDisplayable
{
public:
	DisplayableGridPoints(const GridPoints & grid, double pointSize);

	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	GridPoints m_grid;
	double m_pointSize;
};