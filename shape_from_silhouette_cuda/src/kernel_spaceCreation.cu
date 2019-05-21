#include "kernel_spaceCreation.cuh"

#include <cuda.h>
#include <cuda_runtime.h> 

#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <surface_functions.h>
#include <vector_types.h>

#include "cuda_math_utils.h"
#include "cuda_error_check.h"

#include "kernel_convexHull.cuh"
#include "kernel_spaceUtils.cuh"

namespace sfs
{
	namespace cuda
	{

		__host__ __device__ VoxelProjectionInfo::VoxelProjectionInfo() : m_hull(), m_boundingBox(), m_numImgPoints(0), mep_imgPoints(nullptr), m_visibilityStatus(Visible)
		{

		}


		__device__ void VoxelProjectionInfo::calcConvexHull(const Corners<int2>::type & projectedPoints)
		{
			m_visibilityStatus = Visible;

			for (const int2 * it = projectedPoints.begin(); it != projectedPoints.end(); ++it)
			{
				if (it->x < 0 || it->y < 0)
				{
					if (it->x == INT_MIN || it->y == INT_MIN)
					{
						m_visibilityStatus = NotVisible;
					}
					else
					{
						m_visibilityStatus = Occluded;
					}
				}
			}

			if (m_visibilityStatus == Visible)
			{
				// Convert points to float so that we can calculate the convex hull which requires realValued points
				Corners<float2>::type projectedPointsAllInImage;
				for (const int2 * it = projectedPoints.begin(); it != projectedPoints.end(); ++it)
				{
					projectedPointsAllInImage.push_back(make_float2(static_cast<float>(it->x), static_cast<float>(it->y)));
				}

				m_hull = calculateConvexHull(projectedPointsAllInImage);
				m_boundingBox = BoundingBox(m_hull);

				const uint numImgRows = m_boundingBox.numRows();
			}
		}


		__device__ void VoxelProjectionInfo::calcImgPoints(uint2 * pImgPoints)
		{
			mep_imgPoints = &pImgPoints[m_offsetImgPoints];
			m_numImgPoints = 0;

			if (m_visibilityStatus != Visible)
			{
				return;
			}

			m_numImgPoints = calculatePointsOnHullEdge(m_hull, mep_imgPoints);
		}


		__host__ uint2 * VoxelProjectionInfo::getImgPointsHost() const
		{
			if (mep_imgPoints == nullptr)
			{
				return nullptr;
			}

			uint2 * pDataHost = new uint2[getNumImgRows() * 2];
			cudaSafeCall(cudaMemcpy(pDataHost, mep_imgPoints, getNumImgRows() * 2 * sizeof(uint2), cudaMemcpyDeviceToHost));
			return pDataHost;
		}


		__global__ void createVoxel(VoxelSegmentationInfo * pSegmentationInfo, Voxel * pVoxel, uint3 numVoxel, uint numImages)
		{
			uint x = blockIdx.x * blockDim.x + threadIdx.x;
			uint y = blockIdx.y * blockDim.y + threadIdx.y;
			uint z = blockIdx.z * blockDim.z + threadIdx.z;

			uint offset = x * numVoxel.y * numVoxel.z + y * numVoxel.z + z;

			if (isValidVoxelIdx(x, y, z, offset, numVoxel))
			{
				Voxel & dst = pVoxel[offset];
				VoxelSegmentationInfo & src = pSegmentationInfo[offset];

				dst.center = src.getCenter();
				dst.size = src.getSize();
				dst.isActive = false;
			}
		}


		__global__ void createVoxel(Voxel * pVoxel, uint3 numVoxel, float3 startingPoint, float3 voxelSize)
		{
			uint x = blockIdx.x * blockDim.x + threadIdx.x;
			uint y = blockIdx.y * blockDim.y + threadIdx.y;
			uint z = blockIdx.z * blockDim.z + threadIdx.z;

			uint offset = x * numVoxel.y * numVoxel.z + y * numVoxel.z + z;

			if (isValidVoxelIdx(x, y, z, offset, numVoxel))
			{
				const float3 initialOffset = voxelSize / 2;

				const float posX = startingPoint.x + initialOffset.x + x * voxelSize.x;
				const float posY = startingPoint.y + initialOffset.y + y * voxelSize.y;
				const float posZ = startingPoint.z + initialOffset.z + z * voxelSize.z;

				pVoxel[offset] = Voxel(make_float3(posX, posY, posZ), voxelSize);
			}
		}


		__global__ void createVoxelProjectionInfo_convexHull(VoxelProjectionInfo * pVoxelProjectionInfo, Voxel * pVoxel, int2 * voxelMap, uint3 numVoxel, uint numCameras)
		{
			const uint voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
			const uint cameraIdx = blockIdx.y;

			const uint voxelProjectionIdx = voxelIdx * numCameras + cameraIdx;

			if (voxelIdx < prod(numVoxel))
			{
				const Voxel & v = pVoxel[voxelIdx];
				const Corners<float3>::type & corners = v.getCorners();

				Corners<int2>::type projectedCorners;

				for (uint cornerIdx = 0; cornerIdx < corners.size(); ++cornerIdx)
				{
					const uint offsetVoxelMap = voxelIdx * corners.size() * numCameras + cornerIdx * numCameras + cameraIdx;
					const int2 pixel = voxelMap[offsetVoxelMap];

					projectedCorners.push_back(pixel);
				}

				pVoxelProjectionInfo[voxelProjectionIdx].calcConvexHull(projectedCorners);
			}
		}


		__global__ void createVoxelProjectionInfo_imgPoints(VoxelProjectionInfo * pVoxelProjectionInfo, uint2 * pImgPoints, uint3 numVoxel, uint numCameras)
		{
			const uint voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
			const uint cameraIdx = blockIdx.y;

			const uint voxelProjectionIdx = voxelIdx * numCameras + cameraIdx;

			if (voxelIdx < prod(numVoxel))
			{
				VoxelProjectionInfo & vpi = pVoxelProjectionInfo[voxelProjectionIdx];
				vpi.calcImgPoints(pImgPoints);
			}
		}


		__global__ void createVoxelSegmentationInfo(VoxelSegmentationInfo * pVoxelSegmentationInfo,
			Voxel * pVoxel,
			VoxelProjectionInfo * pVoxelProjectionInfo,
			ConvexVoxelProjection * pConvexHulls,
			uint * pNumImgPoints,
			uint * pNumImgRows,
			VoxelVisibilityStatus * pVisibilityStatus,
			VoxelSegmentationStatus * pSegmentationStatus,
			uint3 numVoxel, uint2 imgSize, uint numCameras)
		{
			const uint voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
			const uint cameraIdx = threadIdx.y;

			const uint voxelProjectionIdx = voxelIdx * numCameras + cameraIdx;

			if (voxelIdx < prod(numVoxel) && cameraIdx == 0)
			{
				VoxelSegmentationInfo & vsi = pVoxelSegmentationInfo[voxelIdx];

				const uint memoryOffset = voxelIdx * numCameras;

				const Voxel & v = pVoxel[voxelIdx];

				vsi.m_center = v.center;
				vsi.m_size = v.size;


				vsi.m_imgSize = imgSize;
				vsi.m_numImages = numCameras;

				vsi.m_numMarkedCameras = 0;

				vsi.m_visibilityStats = &pVisibilityStatus[memoryOffset];
				vsi.m_numImgPoints = &pNumImgPoints[memoryOffset];
				vsi.m_numImgRows = &pNumImgRows[memoryOffset];
				vsi.m_segmentationStats = &pSegmentationStatus[memoryOffset];
				//TODO: ConvexHull

				vsi.m_offsetImgPoints = pVoxelProjectionInfo[voxelProjectionIdx].getOffsetImgPoints();
				vsi.m_imgPoints = pVoxelProjectionInfo[voxelProjectionIdx].getImgPoints();

			}

			__syncthreads();

			if (voxelIdx < prod(numVoxel))
			{
				VoxelSegmentationInfo & vsi = pVoxelSegmentationInfo[voxelIdx];

				const VoxelProjectionInfo & vpi = pVoxelProjectionInfo[voxelProjectionIdx];

				vsi.m_visibilityStats[cameraIdx] = vpi.getVisibilityStatus();
				vsi.m_numImgPoints[cameraIdx] = vpi.getNumImgPoints();
				vsi.m_numImgRows[cameraIdx] = vpi.getNumImgRows();
				vsi.m_segmentationStats[cameraIdx] = (vpi.getVisibilityStatus() == NotVisible || vpi.getVisibilityStatus() == Occluded) ? None : NotMarked;

				if (cameraIdx == 0)
				{
					vsi.m_maxNumVisibleCameras = 0;
					for (int i = 0; i < numCameras; ++i)
					{
						if (vsi.m_visibilityStats[i] == Visible)
						{
							++vsi.m_maxNumVisibleCameras;
						}
					}
				}
			}
		}


		void call_createVoxel(VoxelSegmentationInfo * pSegmentationInfo, Voxel * pVoxel, uint3 numVoxels, uint numImages)
		{
			dim3 blockSize(32, 32, 1);
			dim3 gridSize((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

			cudaCheckError();
			createVoxel << <gridSize, blockSize >> > (pSegmentationInfo, pVoxel, numVoxels, numImages);
			cudaCheckError();
		}


		void call_createVoxel(Voxel * pVoxel, uint3 numVoxel, float3 startingPoint, float3 voxelSize)
		{
			dim3 blockSize(32, 32, 1);
			dim3 gridSize((numVoxel.x + 31) / 32, (numVoxel.y + 31) / 32, numVoxel.z);

			createVoxel << <gridSize, blockSize >> > (pVoxel, numVoxel, startingPoint, voxelSize);
			cudaCheckError();
		}


		void call_createVoxelProjectionInfo_convexHull(VoxelProjectionInfo * pVoxelProjectionInfo, Voxel * pVoxel, int2 * voxelMap, uint3 numVoxel, uint numCameras)
		{
			// Not use 1024 thread per block to avoid "too many resources requested" error
			dim3 blockSize(512, 1, 1);
			dim3 gridSize(((prod(numVoxel) + 511) / 512), numCameras);

			createVoxelProjectionInfo_convexHull << <gridSize, blockSize >> > (pVoxelProjectionInfo, pVoxel, voxelMap, numVoxel, numCameras);
			cudaCheckError();
		}


		void call_createVoxelProjectionInfo_imgPoints(VoxelProjectionInfo * pVoxelProjectionInfo, uint2 * pImgPoints, uint3 numVoxel, uint numCameras)
		{
			dim3 blockSize(1024, 1, 1);
			dim3 gridSize(((prod(numVoxel) + 1023) / 1024), numCameras);

			createVoxelProjectionInfo_imgPoints << <gridSize, blockSize >> > (pVoxelProjectionInfo, pImgPoints, numVoxel, numCameras);
			cudaCheckError();
		}



		void call_createVoxelSegmentationInfo(VoxelSegmentationInfo * pVoxelSegmentationInfo,
			Voxel * pVoxel,
			VoxelProjectionInfo * pVoxelProjectionInfo,
			ConvexVoxelProjection * pConvexHulls,
			uint * pNumImgPoints,
			uint * pNumImgRows,
			VoxelVisibilityStatus * pVisibilityStatus,
			VoxelSegmentationStatus * pSegmentationStatus,
			uint3 numVoxel, uint2 imgSize, uint numCameras)
		{
			dim3 blockSize(1024 / numCameras, numCameras, 1);
			dim3 gridSize(((prod(numVoxel) + blockSize.x - 1) / blockSize.x), 1);

			createVoxelSegmentationInfo << <gridSize, blockSize >> > (pVoxelSegmentationInfo, pVoxel, pVoxelProjectionInfo, pConvexHulls, pNumImgPoints, pNumImgRows, pVisibilityStatus, pSegmentationStatus, numVoxel, imgSize, numCameras);
			cudaCheckError();
		}

	}
}