#include "VoxelCluster.h"

#include <algorithm>

#include "Roi3DF_cuda.h"


namespace sfs
{
	VoxelCluster::VoxelCluster() : m_isGhost(false)
	{
		throw std::runtime_error("Don't call this!");
	}


	VoxelCluster::VoxelCluster(const Voxel * pFirstObj) : m_isGhost(false)
	{
		m_vpObject.push_back(pFirstObj);
		m_boundingBox = pFirstObj->getBoundingBox();
		m_boundingBox.z1 = 0;
	}


	VoxelCluster::VoxelCluster(const std::vector<const Voxel *> & voxel) :
		m_vpObject(voxel), m_isGhost(false)
	{
		resetBoundingBox();
	}


	double VoxelCluster::distTo(const VoxelCluster & other) const
	{
		double minDist = std::numeric_limits<double>::max();
		for (int i = 0; i < m_vpObject.size(); ++i)
		{
			for (int j = 0; j < other.m_vpObject.size(); ++j)
			{
				minDist = std::min(minDist, m_vpObject[i]->distTo(*(other.m_vpObject[j])));
			}
		}

		return minDist;
	}


	void VoxelCluster::unite(const VoxelCluster & other)
	{
		m_vpObject.insert(m_vpObject.end(), other.m_vpObject.begin(), other.m_vpObject.end());
		m_boundingBox.unite(other.getBoundingBox());
	}


	void VoxelCluster::reset()
	{
		m_vpObject.clear();
		m_boundingBox = Roi3DF();
	}


	bool VoxelCluster::contains(const Voxel * v) const
	{
		float3 offset = m_vpObject[0]->size / 2;
		Roi3DF enlargedBox = m_boundingBox;
		enlargedBox.enlarge(offset.x, offset.y, offset.z);
		return enlargedBox.contains(v->center);
	}


	bool VoxelCluster::containsActive(const Voxel * v) const
	{
		for (const auto o : m_vpObject)
		{
			if (*v == *o)
				return true;
		}
		return false;
	}


	Roi3DF VoxelCluster::getBoundingBox() const
	{
		float3 offset = m_vpObject[0]->size / 2;

		Roi3DF enlargedBox = m_boundingBox;
		enlargedBox.enlarge(offset.x, offset.y, offset.z);
		return enlargedBox;
	}


	std::vector<const Voxel *> VoxelCluster::getVoxelFromClusters(const std::vector<VoxelCluster> & clusters)
	{
		std::vector<const Voxel *> ret;

		for(const auto & c : clusters)
		{
			ret.insert(ret.end(), c.getVoxel().begin(), c.getVoxel().end());
		}

		return ret;
	}


	void VoxelCluster::addVoxelWithoutBoundingBoxChange(const Voxel * pObj)
	{
		m_vpObject.push_back(pObj);
	}


	void VoxelCluster::resetBoundingBox()
	{
		if (empty())
		{
			m_boundingBox = Roi3DF();
			return;
		}

		m_boundingBox = m_vpObject[0]->getBoundingBox();
		for (int i = 1; i < m_vpObject.size(); ++i)
		{
			m_boundingBox.unite(m_vpObject[i]->getBoundingBox());
		}
	}


	bool operator==(const VoxelCluster & left, const VoxelCluster & right)
	{
		if (left.m_vpObject.size() != right.m_vpObject.size())
			return false;

		bool identical = true;
		for (int i = 0; i < left.m_vpObject.size(); ++i)
		{
			if (*(left.m_vpObject[i]) != *(right.m_vpObject[i]))
			{
				identical = false;
				break;
			}
		}
		return identical;
	}
}
