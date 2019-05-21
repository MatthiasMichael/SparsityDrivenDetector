#include "VoxelClusterGeometry.h"

#include <osg/ShapeDrawable>

#include <boost/qvm/all.hpp>

#include "osg_utils.h"

#include "Roi3DF_osg.h"

#include "qvm_osg.h"
#include "qvm_cuda.h"


VoxelClusterGeometry::VoxelClusterGeometry(const sfs::VoxelCluster & cluster) :
	m_cluster(cluster),
	m_clusterGeode(),
	m_material(new osg::Material())
{
	// empty
}


void VoxelClusterGeometry::createGeometry()
{
	this->removeChild(m_clusterGeode);
	m_clusterGeode.release();

	m_clusterGeode = new osg::Geode();

	if (m_cluster.empty())
		return;

	m_clusterGeode->addChild(getVoxelGeode());
	m_clusterGeode->addChild(getBeamGeode());

	this->addChild(m_clusterGeode);
}


void VoxelClusterGeometry::setMaterials(const osg::Material & material, const osg::Material & ghostMaterial)
{
	m_material = new osg::Material(material);
	m_ghostMaterial = new osg::Material(ghostMaterial);
}


osg::ref_ptr<osg::Geode> VoxelClusterGeometry::getVoxelGeode() const
{
	osg::ref_ptr<osg::Geode> voxelGeode = new osg::Geode();

	osg::ref_ptr<osg::Material> mat = m_cluster.isGhost() ? m_ghostMaterial : m_material;
	voxelGeode->getOrCreateStateSet()->setAttribute(mat);

	for (const auto v : m_cluster.getVoxel())
	{
		const osg::Vec3 center = boost::qvm::convert_to<osg::Vec3>(v->center);
		const osg::Vec3 size = boost::qvm::convert_to<osg::Vec3>(v->size);

		osg::ref_ptr<osg::Box> box = new osg::Box(center, size[0], size[1], size[2]);

		osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(box);

		voxelGeode->addDrawable(sd);
	}

	return voxelGeode;
}


osg::ref_ptr<osg::Geode> VoxelClusterGeometry::getBeamGeode() const
{
	osg::ref_ptr<osg::Geode> beamGeode = new osg::Geode();

	setAttributesNonLightingBlendable(beamGeode);

	osg::Vec3 offset = boost::qvm::convert_to<osg::Vec3>(m_cluster.getVoxel()[0]->size / 2);

	Roi3DF enlargedBox = m_cluster.getBoundingBox();
	enlargedBox.enlarge(offset.x(), offset.y(), offset.z());

	Corners<osg::Vec3>::type corners = enlargedBox.getCorners<osg::Vec3>();
	osg::Vec4 color = m_cluster.isGhost() ? osg::Vec4(0.8f, 0, 0, 1.f) : osg::Vec4(0, 0.8f, 0, 1.f);

	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (i == j)
				continue;

			bool doCreateBeam = false;
			osg::Vec3 beam = corners[j] - corners[i];
			for (int k = 0; k < 3; ++k)
			{
				if (beam[k] == 0)
				{
					doCreateBeam = true;
					break;
				}
			}
			if (doCreateBeam)
				beamGeode->addChild(createBeam(corners[i], corners[j], color));
		}
	}

	return beamGeode;
}
