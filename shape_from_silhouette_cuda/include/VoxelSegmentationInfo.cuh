#pragma once

#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include "cuda_math_utils.h"

#include "Roi3DF_cuda.h"

#include "TemporaryDeviceVector.h"


namespace sfs
{
	namespace cuda
	{

		//void drawPolygon(rtcvImage8U * pImg, const std::vector<rtcvPointF> & polygon, const unsigned char color);


		__host__ __device__ Corners<float3>::type getVoxelCorners(const float3 & center, const float3 & size);


		enum VoxelSegmentationStatus
		{
			Marked,
			NotMarked,
			None //< If SegmentationStatus is None, the Visibility Status should be Not Visible or Occluded
		};


		enum VoxelVisibilityStatus
		{
			Visible, //< The entire projection of the Voxel is inside the image space
			NotVisible, //< The projection of at least one Vertex of the Voxel is outside of the image
			Occluded //< The projection of at least one Vertex of the Voxel is hidden behind a structure
		};


		class VoxelSegmentationInfo
		{
		public:
			__host__ __device__ VoxelSegmentationInfo();
			//Voxel(const osg::Vec3 & center, const osg::Vec3 & size, const std::vector<RadialCameraModel> & cameraModels, const FacesWithNormals & walls);
			__host__ ~VoxelSegmentationInfo();

			__host__ void setSimpleSerializedValues(const float3 & center, const float3 & size, const int2 & imgSize);
			__host__ void setVisibilityStats(const std::vector<VoxelVisibilityStatus> & visibilityStats);
			__host__ void setImagePoints(const std::vector<std::vector<int2>> & imgPoints, const std::vector<size_t> & numImgPoints);
			__host__ void setNonSerializableMembers(size_t numCameras, const std::vector<VoxelVisibilityStatus> & visibilityStats);

			/*__host__ void freeDeviceMemory();*/

			//void updateStatus(const std::vector<const rtcvImage8U *> & imagesSegmentation, double minPercentSegmentedPixel, GeometryPredicate predicate);
			//void updateStatus(GeometryPredicate predicate);

			__device__ float3 getSize() const { return m_size; }
			__device__ float3 getCenter() const { return m_center; }
			__device__ uint getNumImages() const { return m_numImages; }
			__device__ VoxelVisibilityStatus * getVisibilityStatus() const { return m_visibilityStats; }
			__device__ int getMaxNumVisibleCameras() const { return m_maxNumVisibleCameras; }

			__device__ TemporaryDeviceVector<uint2> getImgPoints(uint cameraIndex) const;
			__device__ uint getNumImgPoints(uint cameraIndex) const { return m_numImgPoints[cameraIndex]; }

			friend bool operator==(const VoxelSegmentationInfo & left, const VoxelSegmentationInfo & right);
			friend bool operator!=(const VoxelSegmentationInfo & left, const VoxelSegmentationInfo & right);

			friend void writeBinary(std::ostream & os, const VoxelSegmentationInfo & v);
			friend void readBinary(std::istream & is, VoxelSegmentationInfo & v);

		private:
			//std::vector<rtcvPointF> calcRawPolygon(const osg::Vec3 & center, const osg::Vec3 & size, const RadialCameraModel & cameraModel, const FacesWithNormals & walls) const;
			//bool isOccluded(const osg::Vec3 & center, const osg::Vec3 & size, const RadialCameraModel & cameraModel, const FacesWithNormals & walls) const;
			//std::vector<rtcvPointF> calcConvexHull(const std::vector<rtcvPointF> & rawPolygon) const;
			//std::vector<rtcvPoint> calcImageIndices(std::vector<rtcvPointF> & convexHull) const; // convexHull not const, otherwise loading after serialization does not compile
			//size_t calcNumImagePoints(size_t cameraIndex) const;

		public:
			float3 m_center; //< 3D-Center of the Voxel
			float3 m_size; //< 3D-Size of the Voxel in m

			uint2 m_imgSize; //< Size of the camera images showing the scene

			VoxelVisibilityStatus * m_visibilityStats; //< 1D, Information if the Voxel is visible in each camera image

			uint m_offsetImgPoints;
			uint2 * m_imgPoints; //< 2D (images, start/end)  Start- end end coordinates of each segmented pixel row for each image
			uint * m_numImgPoints; //< 1D, Number of segmented pixels since they cant be directly read off of imgPointsNew any more
			uint * m_numImgRows; //< 1D, Number of image rows for each camera, 1/2 * length of inner m_imgPoints vector

			// All members below are not necessary for serialization and do not reflect object identity (used for comparison)
			// They can either be calculated from the members above or are processing relevant.

			uint m_numImages; //< Number of Cameras that potentially provide segmentation info. Used as size info for the arrays

			VoxelSegmentationStatus * m_segmentationStats; //< Information about the pixel in each image possibly corresponding to the Voxel

			uint m_maxNumVisibleCameras; //< Maximum number of camera images in which the Voxel is visible
			uint m_numMarkedCameras; //< Number of cameras in which the Voxel is currently marked as active
		};


		std::ostream & operator <<(std::ostream & os, const VoxelVisibilityStatus & s);
		std::istream & operator >>(std::istream & is, VoxelVisibilityStatus & s);

	}
}