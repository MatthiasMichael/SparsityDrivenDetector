#pragma once

#include <vector>

#include <QObject>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Mesh.h"

#include "ExtendedVoxel.h"
#include "VoxelCluster.h"

namespace sfs
{
	class Space
	{
	public:
		Space() = default;
		Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels);
		Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh);

		void update(const std::vector<cv::Mat> & imagesSegmentation, double minPercentSegmentedPixel, ExtendedVoxel::GeometryPredicate p);
		void update(ExtendedVoxel::GeometryPredicate p);

		std::vector<VoxelCluster> clusterVoxels(double maxClusterDistance);
		std::vector<VoxelCluster> sequentialFill(int maxDist) const;

		const ExtendedVoxel & getVoxel(size_t idx) const;
		const ExtendedVoxel & getVoxel(float3 centerCoords) const;
		const ExtendedVoxel & getVoxel(double x, double y, double z) const;

		const std::vector<const Voxel *> & getVoxel() const { return m_rawVoxel; };
		std::vector<const Voxel *> getVoxel(Roi3DF area) const;

		std::vector<const Voxel *> getActiveVoxels() const;
		
		const std::vector<ExtendedVoxel> & getExtendedVoxel() const { return m_voxel; }
		std::vector<const ExtendedVoxel *> getExtendedVoxel(Roi3DF area) const;

		std::vector<const ExtendedVoxel *> getActiveExtendedVoxels() const;
		
		size_t getLinearOffset(float3 coords) const;
		size_t getLinearOffset(double x, double y, double z) const;

		void setVoxelActive(size_t idx, bool b);
		void setVoxelActive(double x, double y, double z, bool b);

		const CameraSet & getCameraModels() const { return m_models; }
		Roi3DF getArea() const { return m_area; }
		
		uint3 getNumVoxels() const { return m_numVoxels; }
		float3 getSizeVoxels() const { return m_sizeVoxels; }
		size_t getNumVoxelsLinear() const { return m_voxel.size(); }
		uint getNumCameras() const { return static_cast<uint>(m_models.size()); }

		//void debugDrawAllConvexHulls(const std::string & path) const;
		void debugCheckVoxelAccess();

		friend bool operator==(const Space & left, const Space & right);

		friend void writeBinary(std::ostream & os, const Space & s);
		friend void readBinary(std::istream & is, Space & s);

	private:
		void init(const float3 & voxelSize, const Mesh & walls);

	private:
		std::vector<ExtendedVoxel> m_voxel;
		std::vector<const Voxel *> m_rawVoxel; // pointer to voxel inside ExtendedVoxel

		float3 m_sizeVoxels;
		uint3 m_numVoxels;

		Roi3DF m_area;

		CameraSet m_models;
	};
}