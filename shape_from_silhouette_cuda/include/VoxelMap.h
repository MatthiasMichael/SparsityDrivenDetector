#pragma once

#include <map>
#include <vector>
#include <iterator>

#include <vector_functions.h>
#include "cuda_math_utils.h"

#include "deviceStructs.h"

#include "Face.cuh"
#include "Voxel.h"
#include "VoxelCluster.h"
#include "IdentifiableCamera.h"


namespace sfs
{
	namespace cuda
	{

		class Space; // forward declaration

		class VoxelMap
		{
		public:
			struct StringConstants;

			enum IntersectionCalculation
			{
				ClusterContainsSegmentedVoxel,
				ClusterContainsVoxel,
				ViewRayIntersectsBoundingBox,
				ViewRayIntersectsBoundingBox2D
			};

		public:
			VoxelMap();
			VoxelMap(const CameraSet & cameraModels, const Space & space, const std::vector<Face> & walls);
			~VoxelMap();

			std::vector<DeviceViewRay> markGhostVoxel(std::vector<VoxelCluster> & clusters, unsigned char * p_dev_imagesSegmentation);
			std::vector<DeviceViewRay> VoxelMap::cpu_markGhostVoxel(std::vector<VoxelCluster> & clusters, unsigned char * p_dev_imagesSegmentation);

			const std::vector<int2> & getPixelMap() const { return m_host_pixelMap; }
			int2 * getDevicePixelMap() const { return m_dev_pixelMap; }

			friend bool operator==(const VoxelMap & left, const VoxelMap & right);

			friend void writeBinary(std::ostream & os, const VoxelMap & vm);
			friend void readBinary(std::istream & is, VoxelMap & vm);

		private:
			void freeDeviceMemory();

			void setImageInfo(size_t cameraIndex, const uint2 & imgSize);
			void setViewRaysToPixel(const std::vector<ViewRay_MemoryDouble> & viewRays);

			void initViewRayMapMemory();

			VoxelMap(const VoxelMap & other) { throw std::runtime_error("No copy allowed. Class manages device memory."); }
			VoxelMap & operator=(const VoxelMap & other) { throw std::runtime_error("No assignemt allowed. Class manages device memory."); }
			void setOcclusions(const std::vector<Face> & walls) const;

		private:
			// General Info
			uint m_numCameras;
			uint2 m_imgSize;

			// Maps pixel position to view Ray
			DeviceViewRay * m_dev_viewRays;
			std::vector<DeviceViewRay> m_host_viewRays;

			// Maps voxel corner to pixel
			int2 * m_dev_pixelMap;
			std::vector<int2> m_host_pixelMap;

			// Images containing indices of presumably non-ghost clusters
			unsigned char * m_dev_viewRayMapMemory;
			std::vector<unsigned char> m_host_viewRayMapMemory; //< This way we can use a single copy instruction from the GPU back to host

			//std::vector<rtcvImage8U> m_viewRayMaps; //< External pointer to m_viewRayMapMemory
		};
	}

}
