#pragma once

#include <vector_functions.h>

#include "cuda_math_utils.h"

#include "Voxel.h"
#include "VoxelSegmentationInfo.cuh"

#include "kernel_convexHull.cuh"

namespace sfs
{
	namespace cuda
	{

		typedef FixedSizeVector< uint2, 2 * Corners<uint2>::NumPoints > DiscreteConvexVoxelProjection;


		// Used during the creation of a VoxelSegmentationInfo encapsulation everything that is necessary to calculate the 
		// image points that are relevant to determine the activitiy of a voxel
		class VoxelProjectionInfo
		{
		public:
			__host__ __device__ VoxelProjectionInfo();

			__device__ void calcConvexHull(const Corners<int2>::type & projectedPoints);
			__device__ void calcImgPoints(uint2 * pImgPoints);

			__host__ void setOffsetImgPoints(uint offset) { m_offsetImgPoints = offset; }
			__host__ __device__ uint getOffsetImgPoints() { return m_offsetImgPoints; }
			__host__ uint2 * getImgPointsHost() const;

			__host__ __device__ const ConvexVoxelProjection & getConvexHull() const { return m_hull; }
			__host__ __device__ const BoundingBox & getBoundingBox() const { return m_boundingBox; }

			__host__ __device__ VoxelVisibilityStatus getVisibilityStatus() const { return m_visibilityStatus; }
			__host__ __device__ uint getNumImgPoints() const { return m_numImgPoints; }
			__host__ __device__ uint getNumImgRows() const { return m_boundingBox.numRows(); }
			__host__ __device__ uint2 * getImgPoints() const { return mep_imgPoints; }

			__device__ TemporaryDeviceVector<uint2> getImgPointsAsVector() const { return TemporaryDeviceVector<uint2>(mep_imgPoints, mep_imgPoints + 2 * getNumImgRows()); }

		private:
			__device__ VoxelProjectionInfo(const VoxelProjectionInfo & other) { }
			__device__ VoxelProjectionInfo & operator=(const VoxelProjectionInfo & other) { return *this; }

		private:
			ConvexVoxelProjection m_hull;
			BoundingBox m_boundingBox;

			uint m_numImgPoints; //< Total number of pixels in an image that are assigned to this voxel. NOT the length of mep_imgPoints. That is 2 * getNumImgRows()

			uint m_offsetImgPoints; //< The start of this class' area in mep_imgPoint memory
			uint2 * mep_imgPoints;

			VoxelVisibilityStatus m_visibilityStatus;
		};


		// Voxel can be either created based on space geometry information or by taking the values already present in the VoxelSegmentationInfo
		void call_createVoxel(Voxel * pVoxel, uint3 numVoxel, float3 startingPoint, float3 voxelSize);
		void call_createVoxel(VoxelSegmentationInfo * pSegmentationInfo, Voxel * pVoxel, uint3 numVoxel, uint numImages);

		void call_createVoxelProjectionInfo_convexHull(VoxelProjectionInfo * pVoxelProjectionInfo, Voxel * pVoxel, int2 * voxelMap, uint3 numVoxel, uint numCameras);
		void call_createVoxelProjectionInfo_imgPoints(VoxelProjectionInfo * pVoxelProjectionInfo, uint2 * pImgPoints, uint3 numVoxel, uint numCameras);

		void call_createVoxelSegmentationInfo(VoxelSegmentationInfo * pVoxelSegmentationInfo,
			Voxel * pVoxel,
			VoxelProjectionInfo * pVoxelProjectionInfo,
			ConvexVoxelProjection * pConvexHulls,
			uint * pNumImgPoints,
			uint * pNumImgRows,
			VoxelVisibilityStatus * pVisibilityStatus,
			VoxelSegmentationStatus * pSegmentationStatus,
			uint3 numVoxel, uint2 imgSize, uint numCameras);
	}
}