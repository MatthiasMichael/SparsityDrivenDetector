#include "DisplayableFrame.h"

#include <boost/qvm/all.hpp>

#include "qvm_eigen.h"
#include "qvm_osg.h"

#include "ActorGeometry.h"


DisplayableFrame::DisplayableFrame(const Frame & frame) :
	m_frame(frame)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableFrame::getGeometry() const
{
	osg::ref_ptr<osg::Group> actorsGroup = new osg::Group();

	for (const auto v : m_frame.actors)
	{
		const osg::Vec3 position = boost::qvm::convert_to<osg::Vec3>(v.state.position.get());
		const osg::Vec3 size = boost::qvm::convert_to<osg::Vec3>(v.actor.size.get());
		const osg::Vec4 color(0, 0.5, 1, 1);
		const osg::Vec2 indicatorSize(10, 400);

		osg::ref_ptr<ActorGeometry> vg = new ActorGeometry(position, size, color, indicatorSize);
		actorsGroup->addChild(vg);
	}

	return actorsGroup;
}
