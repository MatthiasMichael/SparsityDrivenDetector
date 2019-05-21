#pragma once

#include "FunctionalOsgDisplayDescriptor.h"

#include "DisplayableObjects.h"


inline auto objectsDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Objects",
	[](Parametrizable & d)
	{
		d.addEnumParameter("Display Kind", {"Voxel", "Cluster"}, 0);
	},
	[](const Parametrizable & d, 
		const std::vector<const sfs::Voxel *> & voxel, 
		const std::vector<sfs::VoxelCluster> & clusters)
	{
		const int displayKind = d.getEnumParameter("Display Kind");

		DisplayableObjects::ObjectDisplayKind enumDisplayKind = DisplayableObjects::ActiveVoxels;
		switch(displayKind)
		{
			case 0: enumDisplayKind = DisplayableObjects::ActiveVoxels; break;
			case 1: enumDisplayKind = DisplayableObjects::Clusters; break;
			default: break;
		}

		return std::make_unique<DisplayableObjects>(voxel, clusters, enumDisplayKind);
	}
);

