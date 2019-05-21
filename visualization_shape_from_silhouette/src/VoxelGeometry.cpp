#include "VoxelGeometry.h"

#include <osg/ShapeDrawable>
#include <osg/Material>

#include <boost/qvm/all.hpp>

#include "cuda_vector_functions_interop.h"

#include "qvm_osg.h"
#include "qvm_cuda.h"

#include "Voxel.h"



VoxelGeometry::VoxelGeometry(const sfs::Voxel * pVoxel) :
	mep_voxel(pVoxel),
	m_voxelGeode(),
	m_material(new osg::Material())
{
	// empty
}


void VoxelGeometry::createGeometry()
{
	this->removeChild(m_voxelGeode);
	m_voxelGeode.release();

	if (mep_voxel->isActive)
	{
		m_voxelGeode = new osg::Geode();
		this->addChild(m_voxelGeode);

		const osg::Vec3 center = boost::qvm::convert_to<osg::Vec3>(mep_voxel->center);
		const osg::Vec3 size = boost::qvm::convert_to<osg::Vec3>(mep_voxel->size);

		osg::ref_ptr<osg::Box> box = new osg::Box(center, size[0], size[1], size[2]);

		osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(box);

		osg::Vec4 color(1.0, 1.0, 1.0, 1);

		m_material->setDiffuse(osg::Material::FRONT, color);
		getOrCreateStateSet()->setAttribute(m_material);

		m_voxelGeode->addDrawable(sd);
	}
}


void VoxelGeometry::setMaterial(const osg::Material & material)
{
	m_material = new osg::Material(material);
	getOrCreateStateSet()->setAttribute(m_material);
}
