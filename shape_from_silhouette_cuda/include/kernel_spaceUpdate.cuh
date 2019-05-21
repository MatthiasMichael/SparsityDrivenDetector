#pragma once

#include <vector_functions.h>

#include "cuda_math_utils.h"

#include "Voxel.h"
#include "VoxelSegmentationInfo.cuh"

#include "geometryPredicates.cuh"


// Different Functions to update the space. The best Option to call is
//      call_updateSpaceFastFromSurfaceIntegralImage 
// taking 13 to 14 ms on a Nvidia M1000M. 
// The other also do the job but are slower. They are still here for the sake of
// completeness and to show what has been tested.

namespace sfs
{
	namespace cuda
	{

		void call_updateSpace(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, unsigned char * pImages, uint2 sizeImages, float minSegmentation);
		void call_updateSpaceFast(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, unsigned char * pImages, uint2 sizeImages, uint numImages, float minSegmentation);
		void call_updateSpaceFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, uint * pImages, uint2 sizeImages, float minSegmentation);
		void call_updateSpaceFastFromIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, uint * pImages, uint2 sizeImages, uint numImages, float minSegmentation);
		void call_updateSpaceFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, float minSegmentation);
		void call_updateSpaceFastFromSurfaceIntegralImage(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation);
		void call_updateSpaceFastFromSurfaceIntegralImage_2parts(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels, cudaSurfaceObject_t images, uint2 sizeImages, uint numImages, float minSegmentation);

		void call_updateSpace_updateVoxel(VoxelSegmentationInfo * pVoxelInfo, Voxel * pVoxel, uint3 numVoxels);

	}
}