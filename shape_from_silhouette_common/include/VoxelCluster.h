#pragma once

#include <vector>

#include "Voxel.h"


namespace sfs
{
	class VoxelCluster
	{
	public:
		VoxelCluster(); //< Only for Clustering Purposes
		explicit VoxelCluster(const Voxel * pFirstObj);
		explicit VoxelCluster(const std::vector<const Voxel *> & voxel);

		double distTo(const VoxelCluster & other) const;

		void unite(const VoxelCluster & other);
		void reset();

		void addVoxelWithoutBoundingBoxChange(const Voxel * pObj);
		void resetBoundingBox();

		bool contains(const Voxel * v) const;
		bool containsActive(const Voxel * v) const;

		size_t size() const { return m_vpObject.size(); }
		bool empty() const { return m_vpObject.empty(); }

		const std::vector<const Voxel *> & getVoxel() const { return m_vpObject; }
		std::vector<const Voxel *> & getChangeableVoxel() { return m_vpObject; }

		Roi3DF getBoundingBox() const;
		Roi3DF getPreciseBoundingBox() const { return m_boundingBox; }

		bool isGhost() const { return m_isGhost; }
		void setGhost(bool v) const { m_isGhost = v; }

		static std::vector<const Voxel *> getVoxelFromClusters(const std::vector<VoxelCluster> & clusters);
		friend bool operator==(const VoxelCluster & left, const VoxelCluster & right);

	private:
		std::vector<const Voxel *> m_vpObject;

		Roi3DF m_boundingBox;

		mutable bool m_isGhost;
	};
}
