#include "DisplayableSolution.h"

#include <boost/qvm/all.hpp>

#include "qvm_eigen.h"
#include "qvm_osg.h"

#include <osg/ShapeDrawable>
#include <osg/Material>

#include "ActorGeometry.h"



DisplayableSolution::DisplayableSolution(const Solution & solution) : m_solution(solution)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableSolution::getGeometry() const
{
	osg::ref_ptr<osg::Group> solutionGroup = new osg::Group();

	for (const auto s : m_solution.actors)
	{
		const osg::Vec4 color(1, 0.5, 0, 1);
		const osg::Vec3 size{ s.info.targetSize.get()(0), s.info.targetSize.get()(0), s.info.targetSize.get()(1) };
		const osg::Vec2 indicatorSize(10, 400);

		osg::Vec3 position = boost::qvm::convert_to<osg::Vec3>(s.position.get());
		position.z() += s.info.targetSize(1) / 2;

		solutionGroup->addChild(new ActorGeometry(position, size, color, indicatorSize));
	}

	return solutionGroup;
}


DisplayableMergedSolution::DisplayableMergedSolution(const MergedSolution & solution) : m_solution(solution)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableMergedSolution::getGeometry() const
{
	osg::ref_ptr<osg::Group> solutionGroup = new osg::Group();

	for (const auto s : m_solution.actors)
	{
		const osg::Vec4 color(1, 1, 0, 1);
		const osg::Vec3 size{ s.info.targetSize.get()(0), s.info.targetSize.get()(0), s.info.targetSize.get()(1) };
		const osg::Vec2 indicatorSize(10, 400);

		osg::Vec3 position = boost::qvm::convert_to<osg::Vec3>(s.position.get());
		position.z() += s.info.targetSize(1) / 2;

		solutionGroup->addChild(new ActorGeometry(position, size, color, indicatorSize));
	}

	return solutionGroup;
}
