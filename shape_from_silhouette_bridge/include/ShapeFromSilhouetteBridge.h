#pragma once

#include "ShapeFromSilhouette_Impl.h"

namespace sfs
{
	enum ComputeLocation
	{
		Host, Device
	};

	ComputeLocation tryStringToComputeLocation(const std::string & s);

	std::unique_ptr<ShapeFromSilhouette_Impl> makeShapeFromSilhouette(ComputeLocation location);
}