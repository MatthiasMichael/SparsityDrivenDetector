#include <cuda.h>
#include <cuda_runtime.h> 

#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <surface_functions.h>

#include "cuda_math_utils.h"
#include "cuda_error_check.h"

#include "kernel_space.cuh"
#include "kernel_convexHull.cuh"



__device__ inline bool isValidVoxelIdx(uint x, uint y, uint z, uint offset, uint3 numVoxel)
{
	return x < numVoxel.x && y < numVoxel.y && z < numVoxel.z && offset < numVoxel.x * numVoxel.y * numVoxel.z;
}


__device__ inline bool toValidImageIdx(float x, float y, uint * retX, uint * retY, uint2 sizeImg)
{
	if(x < 0 || x >= sizeImg.x - 1 || y < 0 && y >= sizeImg.y - 1)
	{
		*retX = 0;
		*retY = 0;
		return false;
	}

	*retX = (uint)(x + 0.5f);
	*retY = (uint)(y + 0.5f);

	return true;
}


__device__ inline bool isValidImageIdx(float x, float y, uint2 sizeImg)
{
	if(x < 0 || x >= sizeImg.x || y < 0 && y >= sizeImg.y)
	{
		return false;
	}

	return true;
}


__global__ void updateSpaceFast(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, unsigned char * pImages, uint2 sizeImages, uint numImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z;
	const uint camIndex  = threadIdx.z;
	
	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];

		const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
		const unsigned char * pCurrentImage = pImages + camIndex * (sizeImages.x * sizeImages.y);

		uint segmentationCounter = 0;

		assert(imgPointsSingleCam.size() % 2 == 0);
		for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			assert(imgPointsSingleCam[i].y == imgPointsSingleCam[i+1].y);
			for(int x = imgPointsSingleCam[i].x; x <= imgPointsSingleCam[i+1].x; ++x)
			{
				const uint imgOffset = imgPointsSingleCam[i].y * sizeImages.x + x;
				const unsigned char * currentPixel = pCurrentImage + imgOffset;

				if(*currentPixel != 0)
				{
					++segmentationCounter;
				}
			}
		}
		const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

		const VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
		v.m_segmentationStats[camIndex] = s;
	}

	__syncthreads();

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel) && camIndex == 0)
	{		
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


__global__ void updateSpaceFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, uint * pImages, uint2 sizeImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z * blockDim.z + threadIdx.z;
	
	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		v.m_numMarkedCameras = 0;

		for(uint camIndex = 0; camIndex < v.m_numImages; ++camIndex)
		{
			const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
			const uint * pCurrentImage = pImages + camIndex * (sizeImages.x * sizeImages.y);

			uint segmentationCounter = 0;

			for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
			{
				const uint * pStart = pCurrentImage + imgPointsSingleCam[i].y     * sizeImages.x + imgPointsSingleCam[i].x; 
				const uint * pEnd   = pCurrentImage + imgPointsSingleCam[i + 1].y * sizeImages.x + imgPointsSingleCam[i + 1].x; 

				segmentationCounter += *pEnd - *pStart;
			}
			const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

			const VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
			v.m_segmentationStats[camIndex] = s;
			if(s == Marked)
			{
				++v.m_numMarkedCameras;
			}
		}

		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


__global__ void updateSpaceFastFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, uint * pImages, uint2 sizeImages, uint numImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z;
	const uint camIndex  = threadIdx.z;
	
	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];

		const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
		const uint * pCurrentImage = pImages + camIndex * (sizeImages.x * sizeImages.y);

		uint segmentationCounter = 0;

		for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			const uint * pStart = pCurrentImage + imgPointsSingleCam[i].y     * sizeImages.x + imgPointsSingleCam[i].x; 
			const uint * pEnd   = pCurrentImage + imgPointsSingleCam[i + 1].y * sizeImages.x + imgPointsSingleCam[i + 1].x; 

			segmentationCounter += *pEnd - *pStart;
		}
		const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

		const VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
		v.m_segmentationStats[camIndex] = s;
	}

	__syncthreads();

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel) && camIndex == 0)
	{		
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


__global__ void updateSpaceFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, cudaSurfaceObject_t images, uint2 sizeImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z * blockDim.z + threadIdx.z;
	
	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		v.m_numMarkedCameras = 0;

		for(uint camIndex = 0; camIndex < v.m_numImages; ++camIndex)
		{
			const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
			const uint offsetCurrentImage = camIndex * sizeImages.x;

			uint segmentationCounter = 0;

			for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
			{
				uint start, end;
				surf2Dread(&start, images, (offsetCurrentImage + imgPointsSingleCam[i].x) * sizeof(uint), imgPointsSingleCam[i].y ); 
				surf2Dread(&end, images, (offsetCurrentImage + imgPointsSingleCam[i+1].x) * sizeof(uint), imgPointsSingleCam[i+1].y ); 

				segmentationCounter += end - start;
			}
			const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

			VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
			v.m_segmentationStats[camIndex] = s;
			if(s == Marked)
			{
				++v.m_numMarkedCameras;
			}
		}

		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


__global__ void updateSpaceFastFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z;
	const uint camIndex  = threadIdx.z;
	
	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];

		const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
		const uint offsetCurrentImage = camIndex * sizeImages.x;

		uint segmentationCounter = 0;

		for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			uint start, end;
			surf2Dread(&start, images, (offsetCurrentImage + imgPointsSingleCam[i].x) * sizeof(uint), imgPointsSingleCam[i].y ); 
			surf2Dread(&end, images, (offsetCurrentImage + imgPointsSingleCam[i+1].x) * sizeof(uint), imgPointsSingleCam[i+1].y ); 

			segmentationCounter += end - start;
		}
		const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

		VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
		v.m_segmentationStats[camIndex] = s;
	}

	__syncthreads();

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel) && camIndex == 0)
	{		
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


__global__ void updateSpaceFastFromSurfaceIntegralImage_updateSegmentation(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z / numImages;
	const uint camIndex  = blockIdx.z % numImages;

	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{
		VoxelSegmentationInfo & v = pVoxelInfo[offset];

		const TemporaryDeviceVector<uint2> & imgPointsSingleCam = v.getImgPoints(camIndex);
		const uint offsetCurrentImage = camIndex * sizeImages.x;

		uint segmentationCounter = 0;

		for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			uint start, end;
			surf2Dread(&start, images, (offsetCurrentImage + imgPointsSingleCam[i].x) * sizeof(uint), imgPointsSingleCam[i].y ); 
			surf2Dread(&end, images, (offsetCurrentImage + imgPointsSingleCam[i+1].x) * sizeof(uint), imgPointsSingleCam[i+1].y ); 

			segmentationCounter += end - start;
		}
		const uint segmentationThreshold = v.getNumImgPoints(camIndex) * minSegmentation;

		VoxelSegmentationStatus s = segmentationCounter > segmentationThreshold ? Marked : NotMarked;
		v.m_segmentationStats[camIndex] = s;
	}
}



__global__ void updateVoxelOnly(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxel)
{
	const uint voxelId_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint voxelId_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint voxelId_z = blockIdx.z;

	const uint offset = voxelId_x * numVoxel.y * numVoxel.z + voxelId_y * numVoxel.z + voxelId_z; 

	if(isValidVoxelIdx(voxelId_x, voxelId_y, voxelId_z, offset, numVoxel))
	{		
		VoxelSegmentationInfo & v = pVoxelInfo[offset];
		const bool active = voxelPredicate_maximumVisibleActive(v);
		pVoxel[offset].m_isActive = active;
	}
}


void call_updateSpace(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, unsigned char * pImages, uint2 sizeImages, float minSegmentation)
{

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

	updateSpace<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, pImages, sizeImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFast(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, unsigned char * pImages, uint2 sizeImages, uint numImages, float minSegmentation)
{
	assert(numImages == 4);
	dim3 blockSize(16, 16, numImages);
	dim3 gridSize((numVoxels.x + 15) / 16, (numVoxels.y + 15) / 16, numVoxels.z);

	updateSpaceFast<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, pImages, sizeImages, numImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, uint * pImages, uint2 sizeImages, float minSegmentation)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

	updateSpaceFromIntegralImage<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, pImages, sizeImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFastFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, uint * pImages, uint2 sizeImages, uint numImages, float minSegmentation)
{
	assert(numImages == 4);
	dim3 blockSize(16, 16, numImages);
	dim3 gridSize((numVoxels.x + 15) / 16, (numVoxels.y + 15) / 16, numVoxels.z);

	updateSpaceFastFromIntegralImage<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, pImages, sizeImages, numImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, float minSegmentation)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

	updateSpaceFromSurfaceIntegralImage<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, images, sizeImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFastFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation)
{
	assert(numImages == 4);
	dim3 blockSize(16, 16, numImages);
	dim3 gridSize((numVoxels.x + 15) / 16, (numVoxels.y + 15) / 16, numVoxels.z);

	updateSpaceFastFromSurfaceIntegralImage<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, images, sizeImages, numImages, minSegmentation);
	cudaCheckError();
}


void call_updateSpaceFastFromSurfaceIntegralImage_2parts(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z * numImages);

	updateSpaceFastFromSurfaceIntegralImage_updateSegmentation<<<gridSize, blockSize>>>(pVoxelInfo, pVoxel, numVoxels, images, sizeImages, numImages, minSegmentation);
	cudaCheckError();

	dim3 gridSizeUpdateVoxel((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

	updateVoxelOnly<<<gridSizeUpdateVoxel, blockSize>>>(pVoxelInfo, pVoxel, numVoxels);
	cudaCheckError();
}


void call_updateSpace_updateVoxel(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSizeUpdateVoxel((numVoxels.x + 31) / 32, (numVoxels.y + 31) / 32, numVoxels.z);

	updateVoxelOnly<<<gridSizeUpdateVoxel, blockSize>>>(pVoxelInfo, pVoxel, numVoxels);
	cudaCheckError();
}
