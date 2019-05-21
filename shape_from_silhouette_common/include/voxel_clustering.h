#pragma once

#include <vector>

#include "VoxelCluster.h"

namespace sfs
{
	size_t cluster(std::vector<VoxelCluster> & vpCluster,
		const std::vector<const Voxel *> & vpObject,
		double maxClusterDist,
		size_t minNoCluster = 2,
		size_t maxNoCluster = std::numeric_limits<size_t>::max()
	);
}