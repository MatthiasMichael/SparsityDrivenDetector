#pragma once

#include <istream>
#include <ostream>
#include <memory>
#include <vector>

#include <vector_functions.h>

#include "IdentifiableCamera.h"

#include "TemporaryVector.h"

#include "Face.cuh"
#include "Voxel.h"
#include "VoxelCluster.h"
#include "VoxelSegmentationInfo.cuh"
#include "VoxelMap.h"

#include "kernel_convexHull.cuh"
#include "kernel_spaceCreation.cuh"

namespace sfs
{
	namespace cuda
	{
		class VoxelSegmentationInfo_DeviceMemory
		{
		public:

			VoxelSegmentationInfo_DeviceMemory();
			VoxelSegmentationInfo_DeviceMemory(uint3 numVoxel, uint numCameras);
			~VoxelSegmentationInfo_DeviceMemory();

			void init(uint3 numVoxel, uint numCameras);

			ConvexVoxelProjection * getConvexHullsHost() const;

			uint * getNumImgPointsHost() const;
			uint * getNumImgRowsHost() const;

			VoxelVisibilityStatus * getVisibilitStatiHost() const;
			VoxelSegmentationStatus * getSegmentationStatusHost() const;

			inline uint getNumElements() const { return prod(m_numVoxel) * m_numCameras; }
			uint3 getNumVoxel() const { return m_numVoxel; }
			uint getNumCameras() const { return m_numCameras; }

			friend class Space;

		private:
			uint3 m_numVoxel;
			uint m_numCameras;

			ConvexVoxelProjection * p_dev_convexHulls;

			uint * p_dev_numImgPoints;
			uint * p_dev_numImgRows;

			VoxelVisibilityStatus * p_dev_visibilityStati;
			VoxelSegmentationStatus * p_dev_segmentationStati;
		};


		class ImagePoints_DeviceMemory
		{
		public:
			ImagePoints_DeviceMemory();
			ImagePoints_DeviceMemory(VoxelProjectionInfo * p_dev_voxelProjectionInfo, uint3 numVoxel, uint numCameras);
			~ImagePoints_DeviceMemory();

			void init(VoxelProjectionInfo * p_dev_voxelProjectionInfo, uint3 numVoxel, uint numCameras);

			uint getNumPointsTotal() const { return m_numPointsTotal; }

			const std::vector<uint> & getMemoryOffsetsToStart() const { return memoryOffsetToStart; }
			const std::vector<uint> & getNumElements() const { return numElements; }

			uint2 * getImgPointsHost() const;

		private:
			uint3 m_numVoxel;
			uint m_numCameras;
			uint m_numPointsTotal;

			uint2 * p_dev_imgPoints;

			std::vector<uint> memoryOffsetToStart;
			std::vector<uint> numElements;
		};


		class Space
		{
		public:
			Space();
			Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const std::vector<Face> & walls);
			~Space();

			Space(const Space &) = delete;
			Space(Space &&) = delete;

			Space & operator=(const Space &) = delete;
			Space & operator=(Space &&) = delete;

			void freeDeviceMemory();

			void update(unsigned char * pImagesSegmentation, uint2 sizeImages, uint numImages, float minPercentSegmentedPixel);
			void updateFromIntegralImage(uint * pImagesSegmentation, uint2 sizeImages, uint numImages, float minPercentSegmentedPixel);
			void updateFromSurfaceIntegralImage(cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minPercentSegmentedPixel);
			void update();

			std::vector<VoxelCluster> sequentialFill(int maxDist);

			const Voxel & getVoxel(size_t idx) const;
			const Voxel & getVoxel(float3 centerCoords) const { return getVoxel(centerCoords.x, centerCoords.y, centerCoords.z); }
			const Voxel & getVoxel(float x, float y, float z) const;

			const std::vector<const Voxel *> getVoxel(Roi3DF area) const;
			TemporaryVector<const Voxel *> getVoxel() const { return TemporaryVector<const Voxel *>(m_host_Voxel, m_host_Voxel + getNumVoxelsLinear()); }
			const std::vector<const Voxel *> getActiveVoxels() const;
			const std::vector<const Voxel *> getActiveVoxelsFromClusters(const std::vector<VoxelCluster> & clusters) const;

			Voxel * getVoxelDevice() const { return m_dev_Voxel; }

			VoxelSegmentationInfo * getVoxelSegmentationInfoHost() const;

			size_t getLinearOffset(float3 coords) const { return getLinearOffset(coords.x, coords.y, coords.z); }
			size_t getLinearOffset(float x, float y, float z) const;

			void setVoxelActive(size_t idx, bool b);
			void setVoxelActive(float x, float y, float z, bool b);

			const CameraSet & getCameraModels() const { return m_cameraModels; }
			Roi3DF getArea() const { return m_area; }

			float3 getSizeVoxels() const { return m_sizeVoxels; }
			uint2 getSizeImages() const { return m_sizeImages; }

			uint3 getNumVoxels() const { return m_numVoxels; }
			uint getNumVoxelsLinear() const { return m_numVoxels.x * m_numVoxels.y * m_numVoxels.z; }
			uint getNumCameras() const { return static_cast<uint>(m_cameraModels.size()); }

			const VoxelMap & getVoxelMap() const { return *m_voxelMap; }
			const VoxelSegmentationInfo_DeviceMemory & getDeviceMemory() const { return m_deviceMemory; }
			const ImagePoints_DeviceMemory & getImagePoints() const { return m_imgPoints; }

			void saveDebugImages(const std::string & path) const;

			friend bool operator==(const Space & left, const Space & right);

			friend void writeBinary(std::ostream & os, const Space & s);
			friend void readBinary(std::istream & is, Space & s);

		private:

			void init(const float3 & voxelSize, const std::vector<Face> & walls);
			void createVoxel();
			void createVoxelSegmentationInfo(uint2 sizeImage, uint numCameras);

		private:
			VoxelSegmentationInfo_DeviceMemory m_deviceMemory;
			ImagePoints_DeviceMemory m_imgPoints;

			VoxelSegmentationInfo * m_dev_voxelSegmentationInfo; //< Static Voxel Information residing solely on the GPU
			Voxel * m_dev_Voxel; //< Segmentation Info for each Voxel, GPU
			Voxel * m_host_Voxel; //< Segmentation Info for each Voxel, CPU

			std::unique_ptr<VoxelMap> m_voxelMap; //< VoxelMap can not be copied or assigned but also cannot be initialized right at the start

			float3 m_sizeVoxels;
			uint3 m_numVoxels;
			uint2 m_sizeImages;

			Roi3DF m_area;

			CameraSet m_cameraModels;
		};
	}
}