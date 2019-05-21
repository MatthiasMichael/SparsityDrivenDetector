#pragma once

#include <functional>
#include <vector>

#include "opencv2/imgproc.hpp"

#include "Voxel.h"

#include "Mesh.h"
#include "IdentifiableCamera.h"

namespace sfs
{

	void drawPolygon(cv::Mat * pImg, const std::vector<float2> & polygon, unsigned char color);


	class AbstractVoxelStateListener
	{
	public:
		virtual ~AbstractVoxelStateListener() = default;

		virtual void onVoxelStateChanged(bool status) = 0;
	};


	class ExtendedVoxel
	{
	public:
		using WorldCamera = IdentifiableCamera::WorldCamera;

		enum SegmentationStatus
		{
			Marked,
			NotMarked,
			None //< If SegmentationStatus is None, the Visibility Status should be Not Visible or Occluded
		};

		enum VisibilityStatus
		{
			Visible, //< The entire projection of the Voxel is inside the image space
			NotVisible, //< The projection of at least one Vertex of the Voxel is outside of the image
			Occluded //< The projection of at least one Vertex of the Voxel is hidden behind a structure
		};

		typedef std::function<bool(const std::vector<SegmentationStatus> &, const std::vector<VisibilityStatus> &)> GeometryPredicate;

	public:
		ExtendedVoxel();
		ExtendedVoxel(const float3 & center, const float3 & size, const CameraSet & cameraModels, const Mesh & walls);
		ExtendedVoxel(const Voxel & voxel, const CameraSet & cameraModels, const Mesh & walls);

		void updateStatus(const std::vector<cv::Mat> & imagesSegmentation, double minPercentSegmentedPixel, GeometryPredicate predicate);
		void updateStatus(GeometryPredicate predicate);

		double distTo(const ExtendedVoxel & other) const;

		Roi3DF getBoundingBox() const { return m_voxel.getBoundingBox(); }

		const Voxel * getVoxelPointer() const { return &m_voxel; }
		bool isActive() const { return m_voxel.isActive; }
		float3 getSize() const { return m_voxel.size; }
		const float3 & getCenter() const { return m_voxel.center; }
		int getMaxNumVisibleCameras() const { return m_maxNumVisibleCameras; }
		auto getCorners() const { return m_voxel.getCorners(); }
		size_t getNumImgPoints(size_t idxCamera) const { return m_numImgPoints[idxCamera]; }

		const std::vector<std::vector<float2>> & getConvexHull() const { return m_convexHull; }
		const std::vector<int2> & getImgPointsNew(int cameraIndex) const { return m_imgPoints[cameraIndex]; }

		// Warning: Is very slow since the function-call happens for each pixel
		// Use only for debug or non-time-critical stuff
		void loopOverImgPoints(const int cameraIndex, std::function<void(int x, int y)> f) const;

		void addStatusListener(AbstractVoxelStateListener * pListener) const { m_statusListeners.insert(pListener); }
		void removeStatusListener(AbstractVoxelStateListener * pListener) { m_statusListeners.erase(pListener); }
		void segmentationChanged(const bool status) const;

		friend bool operator==(const ExtendedVoxel & left, const ExtendedVoxel & right);
		friend bool operator!=(const ExtendedVoxel & left, const ExtendedVoxel & right);

		friend void writeBinary(std::ostream & os, const ExtendedVoxel & v);
		friend void readBinary(std::istream & is, ExtendedVoxel & v);

		// Debug Methods
		bool debugDrawPolygon();
		void setActive(bool b);

	private:
		static std::vector<float2> calcConvexHull(const std::vector<float2> & rawPolygon);
		static std::vector<int2> calcImageIndices(std::vector<float2> & convexHull); // convexHull not const, otherwise loading after serialization does not compile


		std::vector<float2> calcRawPolygon(const Voxel & voxel, const WorldCamera & cameraModel) const;
		bool isOccluded(const Voxel & voxel, const WorldCamera & cameraModel, const Mesh & walls) const;
		size_t calcNumImagePoints(size_t cameraIndex) const;
		void calcNonSerializedValues();


	private:
		Voxel m_voxel;

		int2 m_imgSize; //< Size of the camera images showing the scene

		std::vector<VisibilityStatus> m_visibilityStats; //< Information if the Voxel is visible in each camera image

		std::vector<std::vector<float2>> m_convexHull; //< Convex Hull of the projection of the Voxel in each camera image
		std::vector<std::vector<int2>> m_imgPoints; //< Start- end end coordinates of each segmented pixel row
		std::vector<size_t> m_numImgPoints; //< Number of segmented pixels since they cant be directly read off of imgPointsNew any more

		// All members below are not necessary for serialization and do not reflect object identity (used for comparison)
		// They can either be calculated from the members above or are processing relevant.

		std::vector<SegmentationStatus> m_segmentationStats; //< Information about the pixel in each image possibly corresponding to the Voxel

		mutable std::set<AbstractVoxelStateListener *> m_statusListeners;

		int m_maxNumVisibleCameras; //< Maximum number of camera images in which the Voxel is visible
		int m_numMarkedCameras; //< Number of cameras in which the Voxel is segmented
	};


	std::ostream & operator <<(std::ostream & os, const ExtendedVoxel::VisibilityStatus & s);
	std::istream & operator >>(std::istream & is, ExtendedVoxel::VisibilityStatus & s);
}