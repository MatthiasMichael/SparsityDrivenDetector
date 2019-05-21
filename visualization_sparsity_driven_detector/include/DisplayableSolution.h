#pragma once

#include <OsgDisplayable.h>

#include "SparsityDrivenDetectorPostProcessing.h"


class DisplayableSolution : public OsgDisplayable
{
public:
	DisplayableSolution(const Solution & solution);
	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	Solution m_solution;
};


class DisplayableMergedSolution : public OsgDisplayable
{
public:
	DisplayableMergedSolution(const MergedSolution & solution);
	osg::ref_ptr<osg::Group> getGeometry() const override;

private:
	MergedSolution m_solution;
};
