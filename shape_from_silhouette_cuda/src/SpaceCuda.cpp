#include "SpaceCuda.h"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_error_check.h"
#include "cuda_math_utils.h"

#include "vectorOperations.h"

#include "kernel_spaceCreation.cuh"
#include "kernel_spaceUpdate.cuh"
#include "space_sequentialFill.h"


namespace sfs
{
	namespace cuda
	{
		VoxelSegmentationInfo_DeviceMemory::VoxelSegmentationInfo_DeviceMemory() :
			m_numVoxel(),
			m_numCameras(0),
			p_dev_convexHulls(nullptr),
			p_dev_numImgPoints(nullptr),
			p_dev_numImgRows(nullptr),
			p_dev_visibilityStati(nullptr),
			p_dev_segmentationStati(nullptr)
		{
			// empty
		}


		VoxelSegmentationInfo_DeviceMemory::VoxelSegmentationInfo_DeviceMemory(uint3 numVoxel, uint numCameras) :
			p_dev_convexHulls(nullptr),
			p_dev_numImgPoints(nullptr),
			p_dev_numImgRows(nullptr),
			p_dev_visibilityStati(nullptr),
			p_dev_segmentationStati(nullptr)
		{
			init(numVoxel, numCameras);
		}


		VoxelSegmentationInfo_DeviceMemory::~VoxelSegmentationInfo_DeviceMemory()
		{
			cudaSafeCall(cudaFree(p_dev_convexHulls));
			cudaSafeCall(cudaFree(p_dev_numImgPoints));
			cudaSafeCall(cudaFree(p_dev_numImgRows));
			cudaSafeCall(cudaFree(p_dev_visibilityStati));
			cudaSafeCall(cudaFree(p_dev_segmentationStati));

			p_dev_convexHulls = nullptr;
			p_dev_numImgPoints = nullptr;
			p_dev_numImgRows = nullptr;
			p_dev_visibilityStati = nullptr;
			p_dev_segmentationStati = nullptr;
		}


		void VoxelSegmentationInfo_DeviceMemory::init(uint3 numVoxel, uint numCameras)
		{
			m_numVoxel = numVoxel;
			m_numCameras = numCameras;

			cudaSafeCall(cudaFree(p_dev_convexHulls));
			cudaSafeCall(cudaMalloc((void**)&p_dev_convexHulls, getNumElements() * sizeof(ConvexVoxelProjection)));

			cudaSafeCall(cudaFree(p_dev_numImgPoints));
			cudaSafeCall(cudaMalloc((void**)&p_dev_numImgPoints, getNumElements() * sizeof(uint)));

			cudaSafeCall(cudaFree(p_dev_numImgRows));
			cudaSafeCall(cudaMalloc((void**)&p_dev_numImgRows, getNumElements() * sizeof(uint)));

			cudaSafeCall(cudaFree(p_dev_visibilityStati));
			cudaSafeCall(cudaMalloc((void**)&p_dev_visibilityStati, getNumElements() * sizeof(VoxelVisibilityStatus)));

			cudaSafeCall(cudaFree(p_dev_segmentationStati));
			cudaSafeCall(cudaMalloc((void**)&p_dev_segmentationStati, getNumElements() * sizeof(VoxelSegmentationStatus)));
		}


		ConvexVoxelProjection * VoxelSegmentationInfo_DeviceMemory::getConvexHullsHost() const
		{
			if (p_dev_convexHulls == nullptr)
			{
				return nullptr;
			}

			ConvexVoxelProjection * p_host_convexHulls = new ConvexVoxelProjection[getNumElements()];
			cudaSafeCall(cudaMemcpy(p_host_convexHulls, p_dev_convexHulls, getNumElements() * sizeof(ConvexVoxelProjection),
				cudaMemcpyDeviceToHost));

			return p_host_convexHulls;
		}


		uint * VoxelSegmentationInfo_DeviceMemory::getNumImgPointsHost() const
		{
			if (p_dev_numImgPoints == nullptr)
			{
				return nullptr;
			}

			uint * p_host_numImgPoints = new uint[getNumElements()];
			cudaSafeCall(cudaMemcpy(p_host_numImgPoints, p_dev_numImgPoints, getNumElements() * sizeof(uint),
				cudaMemcpyDeviceToHost));

			return p_host_numImgPoints;
		}


		uint * VoxelSegmentationInfo_DeviceMemory::getNumImgRowsHost() const
		{
			if (p_dev_numImgRows == nullptr)
			{
				return nullptr;
			}

			uint * p_host_numImgPoints = new uint[getNumElements()];
			cudaSafeCall(cudaMemcpy(p_host_numImgPoints, p_dev_numImgRows, getNumElements() * sizeof(uint),
				cudaMemcpyDeviceToHost));

			return p_host_numImgPoints;
		}


		VoxelVisibilityStatus * VoxelSegmentationInfo_DeviceMemory::getVisibilitStatiHost() const
		{
			if (p_dev_visibilityStati == nullptr)
			{
				return nullptr;
			}

			VoxelVisibilityStatus * p_host_visibilityStati = new VoxelVisibilityStatus[getNumElements()];
			cudaSafeCall(cudaMemcpy(p_host_visibilityStati, p_dev_visibilityStati, getNumElements() * sizeof(
				VoxelVisibilityStatus), cudaMemcpyDeviceToHost));

			return p_host_visibilityStati;
		}


		VoxelSegmentationStatus * VoxelSegmentationInfo_DeviceMemory::getSegmentationStatusHost() const
		{
			if (p_dev_segmentationStati == nullptr)
			{
				return nullptr;
			}

			VoxelSegmentationStatus * p_host_segmentationStati = new VoxelSegmentationStatus[getNumElements()];
			cudaSafeCall(cudaMemcpy(p_host_segmentationStati, p_dev_segmentationStati, getNumElements() * sizeof(
				VoxelSegmentationStatus), cudaMemcpyDeviceToHost));

			return p_host_segmentationStati;
		}


		ImagePoints_DeviceMemory::ImagePoints_DeviceMemory() :
			m_numVoxel(make_uint3(0)),
			m_numCameras(0),
			m_numPointsTotal(0),
			p_dev_imgPoints(nullptr),
			memoryOffsetToStart(),
			numElements(0)
		{
			// empty
		}


		ImagePoints_DeviceMemory::ImagePoints_DeviceMemory(VoxelProjectionInfo * p_dev_voxelProjectionInfo, uint3 numVoxel,
			uint numCameras) :
			m_numVoxel(make_uint3(0)),
			m_numCameras(0),
			m_numPointsTotal(0),
			p_dev_imgPoints(nullptr),
			memoryOffsetToStart(),
			numElements(0)
		{
			init(p_dev_voxelProjectionInfo, numVoxel, numCameras);
		}


		ImagePoints_DeviceMemory::~ImagePoints_DeviceMemory()
		{
			cudaSafeCall(cudaFree(p_dev_imgPoints));
			p_dev_imgPoints = nullptr;
		}


		void ImagePoints_DeviceMemory::init(VoxelProjectionInfo * p_dev_voxelProjectionInfo, uint3 numVoxel,
			uint numCameras)
		{
			m_numVoxel = numVoxel;
			m_numCameras = numCameras;

			const uint numVoxelProjectionInfo = prod(numVoxel) * numCameras;

			VoxelProjectionInfo * p_host_voxelProjectionInfo = (VoxelProjectionInfo *)malloc(
				numVoxelProjectionInfo * sizeof(VoxelProjectionInfo));
			cudaSafeCall(cudaMemcpy(p_host_voxelProjectionInfo, p_dev_voxelProjectionInfo, numVoxelProjectionInfo * sizeof(
				VoxelProjectionInfo), cudaMemcpyDeviceToHost));

			memoryOffsetToStart.resize(numVoxelProjectionInfo);
			numElements.resize(numVoxelProjectionInfo);

			uint accNumImgRows = 0;
			uint idx = 0;
			for (uint idxVoxel = 0; idxVoxel < prod(numVoxel); ++idxVoxel)
			{
				for (uint idxCamera = 0; idxCamera < numCameras; ++idxCamera)
				{
					VoxelProjectionInfo & vpi = p_host_voxelProjectionInfo[idxVoxel * numCameras + idxCamera];

					const uint currentOffset = 2 * accNumImgRows;
					vpi.setOffsetImgPoints(currentOffset);
					memoryOffsetToStart[idx] = currentOffset;

					accNumImgRows += vpi.getNumImgRows();
					numElements[idx] = 2 * vpi.getNumImgRows();

					++idx;
				}
			}

			m_numPointsTotal = 2 * accNumImgRows;

			cudaSafeCall(cudaMemcpy(p_dev_voxelProjectionInfo, p_host_voxelProjectionInfo, numVoxelProjectionInfo * sizeof(
				VoxelProjectionInfo), cudaMemcpyHostToDevice));

			cudaSafeCall(cudaFree(p_dev_imgPoints));
			cudaSafeCall(cudaMalloc((void**)&p_dev_imgPoints, accNumImgRows * 2 * sizeof(uint2)));

			call_createVoxelProjectionInfo_imgPoints(p_dev_voxelProjectionInfo, p_dev_imgPoints, m_numVoxel, numCameras);
			cudaCheckError();

			free(p_host_voxelProjectionInfo);
		}


		uint2 * ImagePoints_DeviceMemory::getImgPointsHost() const
		{
			if (p_dev_imgPoints == nullptr)
			{
				return nullptr;
			}

			uint2 * p_host_imgPoints = new uint2[m_numPointsTotal];
			cudaSafeCall(cudaMemcpy(p_host_imgPoints, p_dev_imgPoints, m_numPointsTotal * sizeof(uint2),
				cudaMemcpyDeviceToHost));

			return p_host_imgPoints;
		}


		bool operator==(const Space & left, const Space & right)
		{
			return left.m_dev_Voxel == right.m_dev_Voxel &&
				left.m_sizeVoxels == right.m_sizeVoxels &&
				left.m_numVoxels == right.m_numVoxels &&
				left.m_area == right.m_area &&
				left.m_cameraModels == right.m_cameraModels;
		}


		void writeBinary(std::ostream & os, const Space & s)
		{
			throw "Not implemented!";
			/*writeBinary(os, s.m_voxel);
			writeBinary(os, s.m_sizeVoxels);
			writeBinary(os, s.m_numVoxels);
			writeBinary(os, s.m_area);
			writeBinaryTrivial(os, s.m_models);
			writeBinary(os, s.m_voxelMaps);*/
		}


		void readBinary(std::istream & is, Space & s)
		{
			throw "Not implemented!";
			//std::vector<VoxelSegmentationInfo> voxelInfo;
			//readBinary(is, voxelInfo);

			//// Now we have the voxel sitting on the host with internal pointers already to device memory;
			//const size_t sizeVoxelSegmentationInfo = voxelInfo.size() * sizeof(VoxelSegmentationInfo);

			//cudaSafeCall( cudaMalloc((void**)&s.m_dev_voxelSegmentationInfo, sizeVoxelSegmentationInfo) );
			//cudaSafeCall( cudaMemcpy(s.m_dev_voxelSegmentationInfo, voxelInfo.data(), sizeVoxelSegmentationInfo, cudaMemcpyHostToDevice) );

			//readBinary(is, s.m_sizeVoxels);
			//readBinary(is, s.m_numVoxels);
			//readBinary(is, s.m_area);
			//readBinaryTrivial(is, s.m_models);

			//cudaSafeCall( cudaMalloc((void**)&s.m_dev_Voxel, voxelInfo.size() * sizeof(Voxel)) );
			//s.m_host_Voxel = (Voxel *)malloc(voxelInfo.size() * sizeof(Voxel));

			//uint3 numVoxels = make_uint3(s.m_numVoxels.x, s.m_numVoxels.y, s.m_numVoxels.z);

			//call_createVoxel(s.m_dev_voxelSegmentationInfo, s.m_dev_Voxel, numVoxels, static_cast<unsigned int>(s.m_models.size()));

			//cudaDeviceSynchronize();

			//cudaCheckError();

			//cudaSafeCall( cudaMemcpy(s.m_host_Voxel, s.m_dev_Voxel, voxelInfo.size() * sizeof(Voxel), cudaMemcpyDeviceToHost) );

			//readBinary(is, *(s.m_voxelMap));
		}


		Space::Space() :
			m_imgPoints(),
			m_dev_voxelSegmentationInfo(nullptr),
			m_dev_Voxel(nullptr),
			m_host_Voxel(nullptr),
			m_voxelMap(std::make_unique<VoxelMap>()),
			m_sizeVoxels(make_float3(0.f)),
			m_numVoxels(make_uint3(0)),
			m_sizeImages(),
			m_area(),
			m_cameraModels()
		{
			// empty; Can only be filled by stream ops
		}


		Space::Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels,
			const std::vector<Face> & walls) :
			m_imgPoints(),
			m_dev_voxelSegmentationInfo(nullptr),
			m_dev_Voxel(nullptr),
			m_host_Voxel(nullptr),
			m_voxelMap(std::make_unique<VoxelMap>()),
			m_sizeVoxels(voxelSize),
			m_numVoxels(ceil(area.size<float3>() / voxelSize)),
			m_area(area),
			m_cameraModels(cameraModels)
		{
			assert(cameraModels.size() >= 2);

			const auto imageSize = cameraModels.begin()->getImageSize();
			m_sizeImages = make_uint2(imageSize.get()(0), imageSize.get()(1));

			init(voxelSize, walls);
		}


		Space::~Space()
		{
			freeDeviceMemory();
			free(m_host_Voxel);
		}


		void Space::freeDeviceMemory()
		{
			cudaSafeCall(cudaFree(m_dev_voxelSegmentationInfo));
			cudaSafeCall(cudaFree(m_dev_Voxel));

			m_dev_voxelSegmentationInfo = nullptr;
			m_dev_Voxel = nullptr;
		}


		void Space::saveDebugImages(const std::string & path) const
		{
			//std::vector<rtcvImage32> images;

			for (const auto & cam : m_cameraModels)
			{
			//	images.push_back(rtcvImage32(cam.getImageSize().get()(0), cam.getImageSize().get()(1)));
			//	images.back().fill(0);
			}

			auto segmentationInfo = getVoxelSegmentationInfoHost();

			for (uint i = 0; i < getNumVoxelsLinear(); ++i)
			{
				auto info = segmentationInfo[i];

				uint * numImageRows = new uint[info.m_numImages];
				cudaMemcpy(numImageRows, info.m_numImgRows, info.m_numImages * sizeof(uint), cudaMemcpyDeviceToHost);

				uint totalNumImageRows = std::accumulate(numImageRows, numImageRows + info.m_numImages, 0);
				uint2 * imagePoints = new uint2[2 * totalNumImageRows];
				cudaMemcpy(imagePoints, info.m_imgPoints, 2 * totalNumImageRows * sizeof(uint2), cudaMemcpyDeviceToHost);

				uint currentRowOffset = 0;
				/*for (size_t j = 0; j < images.size(); ++j)
				{
					for (uint idx_row = 0; idx_row < numImageRows[j]; ++idx_row)
					{
						uint2 start = imagePoints[2 * (currentRowOffset + idx_row)];
						uint2 end = imagePoints[2 * (currentRowOffset + idx_row) + 1];
						assert(start.y == end.y);

						for (uint x = start.x; x < end.x; ++x)
						{
							const auto x_int = static_cast<int>(x);
							const auto y_int = static_cast<int>(start.y);
							images[j].setValue(x_int, y_int, images[j].getValue(x_int, y_int) + 1);
						}
					}
					currentRowOffset += numImageRows[j];
				}*/

				delete[] numImageRows;
				delete[] imagePoints;
			}

			/*for (size_t i = 0; i < images.size(); ++i)
			{
				std::stringstream ss;
				ss << path << "/cam_" << i << ".pgm";
				images[i].writeToFile(ss.str());
			}*/
			
		}


		void Space::init(const float3 & voxelSize, const std::vector<Face> & walls)
		{
			createVoxel();

			m_deviceMemory.init(m_numVoxels, getNumCameras());

			m_voxelMap = std::make_unique<VoxelMap>(m_cameraModels, *this, walls);

			createVoxelSegmentationInfo(m_sizeImages, getNumCameras());
		}


		void Space::update(unsigned char * pImagesSegmentation, uint2 sizeImages, uint numImages,
			float minPercentSegmentedPixel)
		{
			//call_updateSpaceFast(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels, pImagesSegmentation, sizeImages, numImages, minPercentSegmentedPixel);
			call_updateSpace(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels, pImagesSegmentation, sizeImages,
				minPercentSegmentedPixel);
			cudaDeviceSynchronize();
			cudaSafeCall(cudaMemcpy(m_host_Voxel, m_dev_Voxel, getNumVoxelsLinear() * sizeof(Voxel), cudaMemcpyDeviceToHost)
			);
		}


		void Space::updateFromIntegralImage(uint * pImagesIntegral, uint2 sizeImages, uint numImages,
			float minPercentSegmentedPixel)
		{
			//call_updateSpaceFastFromIntegralImage(m_voxelSegmentationInfo, m_devVoxel, m_numVoxels, pImagesIntegral, sizeImages, numImages, minPercentSegmentedPixel);
			call_updateSpaceFromIntegralImage(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels, pImagesIntegral,
				sizeImages, minPercentSegmentedPixel);

			cudaDeviceSynchronize();
			cudaSafeCall(cudaMemcpy(m_host_Voxel, m_dev_Voxel, getNumVoxelsLinear() * sizeof(Voxel), cudaMemcpyDeviceToHost)
			);
		}


		void Space::updateFromSurfaceIntegralImage(cudaSurfaceObject_t images, uint2 sizeImages, uint numImages,
			float minPercentSegmentedPixel)
		{
			//call_updateSpaceFastFromSurfaceIntegralImage_2parts(m_voxelSegmentationInfo, m_devVoxel, m_numVoxels, images, sizeImages, numImages, minPercentSegmentedPixel);
			//call_updateSpaceFastFromSurfaceIntegralImage(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels, images,
													   //  sizeImages, numImages, minPercentSegmentedPixel);
			call_updateSpaceFromSurfaceIntegralImage(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels, images, sizeImages, minPercentSegmentedPixel);

			cudaDeviceSynchronize();
			cudaSafeCall(cudaMemcpy(m_host_Voxel, m_dev_Voxel, getNumVoxelsLinear() * sizeof(Voxel), cudaMemcpyDeviceToHost)
			);
		}


		void Space::update()
		{
			call_updateSpace_updateVoxel(m_dev_voxelSegmentationInfo, m_dev_Voxel, m_numVoxels);

			cudaDeviceSynchronize();
			cudaSafeCall(cudaMemcpy(m_host_Voxel, m_dev_Voxel, getNumVoxelsLinear() * sizeof(Voxel), cudaMemcpyDeviceToHost)
			);
		}


		std::vector<VoxelCluster> Space::sequentialFill(int maxDist)
		{
			return ::sfs::sequentialFill(getVoxel(), m_area, m_sizeVoxels, maxDist, 
				[this](const float3 & v) { return getLinearOffset(v); });
		}


		const Voxel & Space::getVoxel(size_t idx) const
		{
			if (idx >= getNumVoxelsLinear())
			{
				throw std::out_of_range("Space does not contain this voxel!");
			}
			return m_host_Voxel[idx];
		}


		const Voxel & Space::getVoxel(float x, float y, float z) const
		{
			return getVoxel(getLinearOffset(x, y, z));
		}


		const std::vector<const Voxel *> Space::getVoxel(Roi3DF area) const
		{
			area.intersect(area);
			std::vector<const Voxel *> v;

			for (float x = area.x1; x < area.x2; x += m_sizeVoxels.x)
			{
				for (float y = area.y1; x < area.y2; y += m_sizeVoxels.y)
				{
					for (float z = area.z1; z < area.z2; z += m_sizeVoxels.z)
					{
						v.push_back(&getVoxel(x, y, z));
					}
				}
			}
			return v;
		}


		VoxelSegmentationInfo * Space::getVoxelSegmentationInfoHost() const
		{
			if (m_dev_voxelSegmentationInfo == nullptr)
			{
				return nullptr;
			}

			VoxelSegmentationInfo * p_host_VoxelSegmentationInfo = new VoxelSegmentationInfo[getNumVoxelsLinear()];
			cudaSafeCall(cudaMemcpy(p_host_VoxelSegmentationInfo, m_dev_voxelSegmentationInfo, getNumVoxelsLinear() * sizeof
			(VoxelSegmentationInfo), cudaMemcpyDeviceToHost));

			return p_host_VoxelSegmentationInfo;
		}


		void Space::setVoxelActive(float x, float y, float z, bool b)
		{
			if (!m_area.contains(x, y, z))
			{
				throw std::out_of_range("Space does not contain this voxel!");
			}

			x -= m_area.x1;
			y -= m_area.y1;
			z -= m_area.z1;

			x /= m_sizeVoxels.x;
			y /= m_sizeVoxels.y;
			z /= m_sizeVoxels.z;

			assert(x >= 0 && y >= 0 && z >= 0);

			size_t idx = static_cast<size_t>(x) * m_numVoxels.y * m_numVoxels.z + static_cast<size_t>(y) * m_numVoxels.z +
				static_cast<size_t>(z);

			m_host_Voxel[idx].isActive = true;

			// TODO: Copy back to GPU
		}


		void Space::setVoxelActive(size_t idx, bool b)
		{
			if (idx >= getNumVoxelsLinear())
			{
				throw std::out_of_range("Space does not contain this voxel!");
			}

			m_host_Voxel[idx].isActive = b;

			// TODO: Copy back to GPU
		}


		__host__ size_t Space::getLinearOffset(float x, float y, float z) const
		{
			assert(m_area.contains(x, y, z));

			if (!m_area.contains(x, y, z))
			{
				throw std::out_of_range("Space does not contain a Voxel with these coordinates!");
			}

			x -= m_area.x1;
			y -= m_area.y1;
			z -= m_area.z1;

			x /= m_sizeVoxels.x;
			y /= m_sizeVoxels.y;
			z /= m_sizeVoxels.z;

			assert(x >= 0 && y >= 0 && z >= 0);

			return static_cast<size_t>(x) * m_numVoxels.y * m_numVoxels.z + static_cast<size_t>(y) * m_numVoxels.z +
				static_cast<size_t>(z);
		}


		void Space::createVoxel()
		{
			cudaSafeCall(cudaFree(m_dev_Voxel));
			free(m_host_Voxel);

			size_t sizeVoxelBytes = getNumVoxelsLinear() * sizeof(Voxel);

			cudaSafeCall(cudaMalloc((void**)&m_dev_Voxel, sizeVoxelBytes));
			m_host_Voxel = (Voxel *)malloc(sizeVoxelBytes);

			call_createVoxel(m_dev_Voxel, m_numVoxels, m_area.start<float3>(), m_sizeVoxels);

			cudaSafeCall(cudaMemcpy(m_host_Voxel, m_dev_Voxel, sizeVoxelBytes, cudaMemcpyDeviceToHost));
		}


		void Space::createVoxelSegmentationInfo(uint2 sizeImage, uint numCameras)
		{
			VoxelProjectionInfo * p_dev_voxelProjectionInfo;
			const uint numVoxelProjectionInfo = getNumVoxelsLinear() * numCameras;
			cudaSafeCall(cudaMalloc((void**)&p_dev_voxelProjectionInfo, numVoxelProjectionInfo * sizeof(VoxelProjectionInfo)
			));

			call_createVoxelProjectionInfo_convexHull(p_dev_voxelProjectionInfo, m_dev_Voxel,
				m_voxelMap->getDevicePixelMap(), m_numVoxels, numCameras);
			cudaCheckError();

			m_imgPoints.init(p_dev_voxelProjectionInfo, getNumVoxels(), numCameras);
			// TODO: Maybe put all of the projection Info inside the image points since their only purpose is to create them

			cudaSafeCall(cudaFree(m_dev_voxelSegmentationInfo));
			cudaSafeCall(cudaMalloc((void**)&m_dev_voxelSegmentationInfo, getNumVoxelsLinear() * numCameras * sizeof(
				VoxelSegmentationInfo)));

			call_createVoxelSegmentationInfo(m_dev_voxelSegmentationInfo, m_dev_Voxel, p_dev_voxelProjectionInfo,
				m_deviceMemory.p_dev_convexHulls, m_deviceMemory.p_dev_numImgPoints,
				m_deviceMemory.p_dev_numImgRows, m_deviceMemory.p_dev_visibilityStati,
				m_deviceMemory.p_dev_segmentationStati, m_numVoxels, m_sizeImages, numCameras);

			cudaSafeCall(cudaFree(p_dev_voxelProjectionInfo));
		}


		const std::vector<const Voxel *> Space::getActiveVoxels() const
		{
			std::vector<const Voxel *> activeVoxels;
			activeVoxels.reserve(getNumVoxelsLinear());

			for (unsigned int i = 0; i < getNumVoxelsLinear(); ++i)
			{
				const Voxel * pVoxel = &m_host_Voxel[i];
				if (pVoxel->isActive)
				{
					activeVoxels.push_back(pVoxel);
				}
			}

			activeVoxels.shrink_to_fit();
			return activeVoxels;
		}


		const std::vector<const Voxel *> Space::getActiveVoxelsFromClusters(
			const std::vector<VoxelCluster> & clusters) const
		{
			std::vector<const Voxel *> activeVoxels;
			activeVoxels.reserve(getNumVoxelsLinear());

			for (const auto & c : clusters)
			{
				const std::vector<const Voxel *> & clusterVoxels = c.getVoxel();
				activeVoxels.insert(activeVoxels.end(), clusterVoxels.begin(), clusterVoxels.end());
			}

			activeVoxels.shrink_to_fit();
			return activeVoxels;
		}
	}
}
