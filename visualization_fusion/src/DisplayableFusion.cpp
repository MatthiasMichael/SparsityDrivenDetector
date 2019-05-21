#include "DisplayableFusion.h"

#include "ActorGeometry.h"
#include "VoxelClusterGeometry.h"

#include <boost/qvm/all.hpp>

#include "qvm_eigen.h"
#include "qvm_osg.h"
#include <osg/ShapeDrawable>

const osg::ref_ptr<osg::Material> DisplayableFusion::s_voxelMaterial = createVoxelMaterial();
const osg::ref_ptr<osg::Material> DisplayableFusion::s_ghostVoxelMaterial = createGhostVoxelMaterial();
const osg::ref_ptr<osg::Material> DisplayableFusion::s_indicatorMaterial = createIndicatorMaterial();


osg::ref_ptr<osg::Material> DisplayableFusion::createVoxelMaterial()
{
	osg::ref_ptr<osg::Material> voxelMaterial = new osg::Material();

	voxelMaterial->setDiffuse(osg::Material::FRONT, osg::Vec4(0.9f, 0.9f, 1.0f, 1.0f));
	voxelMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	voxelMaterial->setAmbient(osg::Material::FRONT, osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	voxelMaterial->setEmission(osg::Material::FRONT, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	voxelMaterial->setShininess(osg::Material::FRONT, 25.0);

	return voxelMaterial;
}


osg::ref_ptr<osg::Material> DisplayableFusion::createGhostVoxelMaterial()
{
	osg::ref_ptr<osg::Material> voxelMaterial = new osg::Material();

	voxelMaterial->setDiffuse(osg::Material::FRONT, osg::Vec4(0.6f, 0.3f, 0.3f, 1.0));
	voxelMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	voxelMaterial->setAmbient(osg::Material::FRONT, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0));
	voxelMaterial->setEmission(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	voxelMaterial->setShininess(osg::Material::FRONT, 1.0f);

	return voxelMaterial;
}


osg::ref_ptr<osg::Material> DisplayableFusion::createIndicatorMaterial()
{
	osg::ref_ptr<osg::Material> indicatorMaterial = new osg::Material();

	indicatorMaterial->setDiffuse(osg::Material::FRONT, osg::Vec4(0.0f, 1.0f, 0.0f, 1.0));
	indicatorMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	indicatorMaterial->setAmbient(osg::Material::FRONT, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0));
	indicatorMaterial->setEmission(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	indicatorMaterial->setShininess(osg::Material::FRONT, 1.0f);

	return indicatorMaterial;
}


DisplayableFusion::DisplayableFusion(const FusedSolution & solution):
	OsgDisplayable(),
	m_solution(solution)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableFusion::getGeometry() const
{
	osg::ref_ptr<osg::Group> solutionGroup = new osg::Group();

	osg::ref_ptr<osg::Group> actorGroup = new osg::Group();
	osg::ref_ptr<osg::Group> volumeGroup = new osg::Group();

	for(const auto & actor : m_solution.actors)
	{
		const auto & a = actor.actor;
		const auto & s = a.info.targetSize.get();

		const osg::Vec4 color(0, 1, 0, 1);
		const osg::Vec3 size{ s(0), s(0), s(1) };
		const osg::Vec2 indicatorSize(10, 400);

		osg::Vec3 position = boost::qvm::convert_to<osg::Vec3>(a.position.get());
		position.z() += s(1) / 2;

		const osg::Vec3 indicatorCenter = position + osg::Vec3(0, 0, indicatorSize.y() / 2 - position.z());
		osg::ref_ptr<osg::Cylinder> cylinder = new osg::Cylinder(indicatorCenter, indicatorSize.x(), indicatorSize.y());
		osg::ref_ptr<osg::ShapeDrawable> cd = new osg::ShapeDrawable(cylinder);

		cd->getOrCreateStateSet()->setAttribute(s_indicatorMaterial);

		actorGroup->addChild(cd);

		osg::ref_ptr<VoxelClusterGeometry> vcg = new VoxelClusterGeometry(actor.volume);
		vcg->setMaterials(*s_voxelMaterial, *s_ghostVoxelMaterial);
		vcg->createGeometry();
		volumeGroup->addChild(vcg);
	}

	solutionGroup->addChild(actorGroup);
	solutionGroup->addChild(volumeGroup);

	return solutionGroup;
}
