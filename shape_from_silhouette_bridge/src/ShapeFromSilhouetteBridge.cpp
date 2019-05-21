#include "ShapeFromSilhouetteBridge.h"
#include "ShapeFromSilhouette_ImplHost.h"
#include "ShapeFromSilhouette_ImplCuda.h"


namespace sfs
{
	ComputeLocation tryStringToComputeLocation(const std::string & s)
	{
		if(s == "Host")
		{
			return Host;
		}

		if(s == "Device")
		{
			return Device;
		}

		throw std::runtime_error("Invalid ComputeLocation identifier");
	}


	std::unique_ptr<ShapeFromSilhouette_Impl> makeShapeFromSilhouette(ComputeLocation location)
	{
		switch(location)
		{
			case Host: return std::make_unique<ShapeFromSilhouette_ImplHost>();
			case Device: return std::make_unique<ShapeFromSilhouette_ImplCuda>();
			default: throw std::runtime_error("Unknown ComputeLocation");
		}
	}
}
