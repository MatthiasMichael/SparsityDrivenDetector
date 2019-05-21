#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "IdentifiableCamera.h"

#include "VoxelCluster.h"

#include "Face.cuh"
#include "SpaceCuda.h"
#include "VoxelMap.h"


namespace sfs
{
	namespace cuda
	{
		// TODO: Ensure consistency of image Sizes with the one of the Camera Models

		class ShapeFromSilhouette
		{
		public:
			struct StringConstants;

			struct Parameters
			{
				Parameters();

				float minSegmentation;
				int maxClusterDistance;
			};

		public:
			ShapeFromSilhouette();
			~ShapeFromSilhouette();

			ShapeFromSilhouette(ShapeFromSilhouette &&) noexcept;

			//ShapeFromSilhouette & operator=(ShapeFromSilhouette &&) = default;

			void createSpace(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const std::vector<Face> & pWalls);

			void loadSpace(const std::string & filename);
			void saveSpace(const std::string & filename);

			bool hasSpace() const { return m_space != nullptr; }

			Parameters & getChangeableParameters() { return m_parameters; }
			void processInput(const std::vector<cv::Mat> & imagesSegmentation);

			std::vector<const Voxel *> getActiveVoxels() const { return m_space->getActiveVoxels(); }
			std::vector<const Voxel *> getActiveVoxelsFromClusters() const { return m_space->getActiveVoxelsFromClusters(m_clusterObjects); }

			const std::vector<VoxelCluster> & getCluster() const { return m_clusterObjects; }

			const Space & getSpace() const;
			Roi3DF getArea() const;
			const CameraSet & getCameraModels() const;

		private:
			bool needDeviceInitialization(const std::vector<cv::Mat> & imagesSegmentation) const;
			void initDeviceMemory(const std::vector<cv::Mat> & imagesSegmentation);
			void freeDeviceMemory();

			void copyInput(const std::vector<cv::Mat> & imagesSegmentation);


		private:
			// Processing
			std::unique_ptr<Space> m_space;

			// Retrieveable (intermediate) results
			std::vector<VoxelCluster> m_clusterObjects;

			// Parameters;
			Parameters m_parameters;

			// Cuda Processing
			int2 m_imageSize;
			size_t m_numImages;

			unsigned char * m_dev_imagesSegmentation;
			uint * m_dev_integralImages;

			cudaSurfaceObject_t m_integralImageSurface;
			cudaArray * m_integralImagesArray;

			cudaChannelFormatDesc m_integralImageSurfaceChannelDesc;
			cudaResourceDesc m_integralImageSurfaceResourceDesc;
		};
	}
}