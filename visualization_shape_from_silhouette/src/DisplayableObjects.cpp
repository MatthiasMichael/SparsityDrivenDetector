#include "DisplayableObjects.h"

#include "VoxelGeometry.h"
#include "VoxelClusterGeometry.h"


const osg::ref_ptr<osg::Material> DisplayableObjects::s_voxelMaterial = createVoxelMaterial();
const osg::ref_ptr<osg::Material> DisplayableObjects::s_ghostVoxelMaterial = createGhostVoxelMaterial();


osg::ref_ptr<osg::Material> DisplayableObjects::createVoxelMaterial()
{
	osg::ref_ptr<osg::Material> voxelMaterial = new osg::Material();

	voxelMaterial->setDiffuse(osg::Material::FRONT,  osg::Vec4(0.9f, 0.9f, 1.0f, 1.0f));
	voxelMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	voxelMaterial->setAmbient(osg::Material::FRONT,  osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));
	voxelMaterial->setEmission(osg::Material::FRONT, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	voxelMaterial->setShininess(osg::Material::FRONT, 25.0);

	return voxelMaterial;
}


osg::ref_ptr<osg::Material> DisplayableObjects::createGhostVoxelMaterial()
{
	osg::ref_ptr<osg::Material> voxelMaterial = new osg::Material();

	voxelMaterial->setDiffuse(osg::Material::FRONT,  osg::Vec4(0.6f, 0.3f, 0.3f, 1.0));
	voxelMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	voxelMaterial->setAmbient(osg::Material::FRONT,  osg::Vec4(0.1f, 0.1f, 0.1f, 1.0));
	voxelMaterial->setEmission(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.0, 1.0));
	voxelMaterial->setShininess(osg::Material::FRONT, 1.0f);

	return voxelMaterial;
}


DisplayableObjects::DisplayableObjects() : OsgDisplayable(), m_displayKind()
{
	// empty
}


DisplayableObjects::DisplayableObjects(const std::vector<const sfs::Voxel *> & voxel, const std::vector<sfs::VoxelCluster> & clusters, 
									   ObjectDisplayKind displayKind) :
	OsgDisplayable(), m_voxel(voxel), m_clusters(clusters), m_displayKind(displayKind)
{
	// empty
}


osg::ref_ptr<osg::Group> DisplayableObjects::getGeometry() const
{
	switch (m_displayKind)
	{
		case ActiveVoxels: return getRawVoxelGeometry(); break;
		case Clusters: return getVoxelClusterGeometry(); break;
		default: throw("Non supported display type chosen."); break;
	}
}


osg::ref_ptr<osg::Group> DisplayableObjects::getRawVoxelGeometry() const
{
	osg::ref_ptr<osg::Group> voxelGroup = new osg::Group();

	for(const auto v : m_voxel)
	{
		osg::ref_ptr<VoxelGeometry> vg = new VoxelGeometry(v);
		vg->setMaterial(*s_voxelMaterial);
		vg->createGeometry();
		voxelGroup->addChild(vg);
	}

	return voxelGroup;
}


osg::ref_ptr<osg::Group> DisplayableObjects::getVoxelClusterGeometry() const
{
	osg::ref_ptr<osg::Group> clusterGroup = new osg::Group();

	for(const auto & c : m_clusters)
	{
		osg::ref_ptr<VoxelClusterGeometry> vcg = new VoxelClusterGeometry(c);
		vcg->setMaterials(*s_voxelMaterial, *s_ghostVoxelMaterial);
		vcg->createGeometry();
		clusterGroup->addChild(vcg);
	}

	return clusterGroup;
}