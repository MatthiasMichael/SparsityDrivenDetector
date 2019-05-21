#pragma once

#include <OsgDisplayable.h>

#include "Frame.h"


class DisplayableFrame : public OsgDisplayable
{
public:
	DisplayableFrame(const Frame & frame);
	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	Frame m_frame;
};
