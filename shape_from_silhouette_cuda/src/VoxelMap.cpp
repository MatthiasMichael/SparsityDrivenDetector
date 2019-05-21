#include "VoxelMap.h"

#include <climits>
#include <set>

#include <vector_types.h>

#include "cuda_error_check.h"
#include "cuda_math_utils.h"

#include "vectorOperations.h"

#include "SpaceCuda.h"

#include "kernel_VoxelMap.cuh"

#include "ApplicationTimer.h"


namespace sfs
{
	namespace cuda
	{
		struct VoxelMap::StringConstants
		{
			inline static const auto TP_CTOR = "VoxelMap::Ctor";
			inline static const auto TP_CTOR_PIXELMAP = "VoxelMap::Ctor::PixelMap";
			inline static const auto TP_CTOR_VIEWRAYMAP = "VoxelMap::Ctor::ViewRayMap";
			inline static const auto TP_CTOR_COPY = "VoxelMap::Ctor::Copy";
			inline static const auto TP_CTOR_OCCLUSIONS = "VoxelMap::Ctor::Occlusions";

			inline static const auto TP_GHOSTS = "VoxelMap::MarkGhostVoxel";
			inline static const auto TP_GHOSTS_PRE = "VoxelMap::MarkGhostVoxel::Pre";
			inline static const auto TP_GHOSTS_KERNEL = "VoxelMap::MarkGhostVoxel::Kernel";
			inline static const auto TP_GHOSTS_PAST = "VoxelMap::MarkGhostVoxel::Past";
		};

		bool operator==(const VoxelMap & left, const VoxelMap & right)
		{
			return left.m_numCameras == right.m_numCameras &&
				left.m_imgSize == right.m_imgSize &&
				left.m_dev_viewRays == right.m_dev_viewRays;
		}

		//std::ostream & operator<<(std::ostream & os, const float3 & f)
		//{
		//	os << "[" << f.x << ", " << f.y << ", " << f.z << "]";
		//	return os;
		//}


		void writeBinary(std::ostream & os, const VoxelMap & vm)
		{
			/*writeBinary(os, vm.m_cameraIndex);
			writeBinary(os, vm.m_imgSize);

			size_t s = vm.m_voxelToPoint.size();
			os.write(reinterpret_cast<const char*>(&s), sizeof(s));

			for (size_t i = 0; i < s; ++i)
			{
			writeBinaryTrivial(os, vm.m_voxelToPoint[i]);
			}

			writeBinaryTrivial(os, vm.m_viewRayToPoint);*/
		}


		void readBinary(std::istream & is, VoxelMap & vm)
		{
			// This reading method is a bit fucked up due to compatibility with the CPU Data structures
			// On the CPU-Only version there is a single Voxel map for each camera while we have one map for all cameras
			// Therefore the data of all individual maps is read and mangled into a single data structure
			// The number of cameras should be the same whether deduced from the maximal read camera index or from the total number of maps read
			// The most complicated data structure - voxelToPoint is currently not in use (too complicated and not needed for core functionality)

			//size_t numSerializedVoxelMaps;
			//is.read(reinterpret_cast<char*>(&numSerializedVoxelMaps), sizeof(numSerializedVoxelMaps));

			//size_t numCameras = 0;
			//rtcvPoint collectedImgSize(0, 0);
			//std::vector<ViewRay_MemoryDouble> collectedViewRaysToPoint;

			//for (size_t i = 0; i < numSerializedVoxelMaps; ++i)
			//{
			//	size_t cameraIndex;
			//	rtcvPoint imgSize;

			//	std::vector<std::vector<size_t>> voxelToPoint; //< currently we don't use this.

			//	std::vector<ViewRay_MemoryDouble> viewRayToPoint;

			//	readBinary(is, cameraIndex);
			//	readBinary(is, imgSize);

			//	size_t s;
			//	is.read(reinterpret_cast<char*>(&s), sizeof(s));
			//	voxelToPoint.resize(s);

			//	for (size_t i = 0; i < s; ++i)
			//	{
			//		readBinaryTrivial(is, voxelToPoint[i]);
			//	}

			//	readBinaryTrivial(is, viewRayToPoint);

			//	numCameras = std::max(numCameras, cameraIndex + 1);

			//	collectedViewRaysToPoint.insert(collectedViewRaysToPoint.end(), viewRayToPoint.begin(), viewRayToPoint.end());

			//	assert(collectedImgSize == rtcvPoint(0, 0) || imgSize == collectedImgSize);
			//	if (collectedImgSize != rtcvPoint(0, 0) && imgSize != collectedImgSize)
			//	{
			//		throw std::runtime_error("Image sizes of Voxel maps do not match. Something went wrong.");
			//	}
			//	collectedImgSize = imgSize;
			//}

			//vm.setImageInfo(numCameras, collectedImgSize);
			//vm.setViewRaysToPixel(collectedViewRaysToPoint);
		}


		VoxelMap::VoxelMap() : m_numCameras(0), m_imgSize(make_uint2(0, 0)), m_dev_viewRays(nullptr), m_dev_pixelMap(nullptr), m_dev_viewRayMapMemory(nullptr)
		{

		}


		VoxelMap::VoxelMap(const CameraSet & cameraModels, const Space & space, const std::vector<Face> & walls) :
			m_numCameras(static_cast<uint>(cameraModels.size())),
			m_imgSize(make_uint2(cameraModels.begin()->getImageSize().get()(0), cameraModels.begin()->getImageSize().get()(1))),
			m_dev_viewRays(nullptr), m_dev_pixelMap(nullptr), m_dev_viewRayMapMemory(nullptr)
		{
			AT_START(StringConstants::TP_CTOR);
			const uint numPixel = m_numCameras * m_imgSize.x * m_imgSize.y;
			const uint numVoxel = static_cast<uint>(space.getNumVoxelsLinear());
			const uint numVoxelCornerProjections = numVoxel * 8 * m_numCameras;

			// Create the Map for each Image Point
			AT_START(StringConstants::TP_CTOR_VIEWRAYMAP);
			m_host_viewRays.resize(numPixel);

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(numPixel); ++i)
			{
				const uint camIdx = i / (m_imgSize.x * m_imgSize.y);
				const uint x = (i % (m_imgSize.x * m_imgSize.y)) % m_imgSize.x;
				const uint y = (i % (m_imgSize.x * m_imgSize.y)) / m_imgSize.x;

				const auto cameraModel = *std::next(cameraModels.begin(), camIdx);
				const auto ray = cameraModel.imageToWorld(make_named<ImagePoint>(x, y));

				m_host_viewRays[i].origin = make_float3(static_cast<float>(ray.origin(0)), static_cast<float>(ray.origin(1)), static_cast<float>(ray.origin(2)));
				m_host_viewRays[i].ray = make_float3(static_cast<float>(ray.direction(0)), static_cast<float>(ray.direction(1)), static_cast<float>(ray.direction(1)));
			}
			AT_STOP(StringConstants::TP_CTOR_VIEWRAYMAP);

			// Create the map for each voxel corner
			// Here I don't want to "optimize" by letting eight voxel share a corner due to possible rounding errors
			AT_START(StringConstants::TP_CTOR_PIXELMAP);
			m_host_pixelMap.resize(numVoxelCornerProjections);

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(numVoxelCornerProjections); ++i)
			{
				const uint voxelIdx = i / (8 * m_numCameras);
				const uint cornerIdx = (i % (8 * m_numCameras)) / m_numCameras;
				const uint camIdx = (i % (8 * m_numCameras)) % m_numCameras;

				const Voxel & v = space.getVoxel(voxelIdx);
				const float3 corner = v.getCorners()[cornerIdx];

				const auto cam = *std::next(cameraModels.begin(), camIdx);
				const auto[imgPoint, visible] = cam.worldToImageWithVisibilityCheck(make_named<WorldVector>(corner.x, corner.y, corner.z));

				double row = imgPoint(1);
				double col = imgPoint(0);

				const int xInt = visible ? static_cast<int>(col) : INT_MIN;
				const int yInt = visible ? static_cast<int>(row) : INT_MIN;

				const int2 pixelPos = make_int2(xInt, yInt);
				m_host_pixelMap[i] = pixelPos;
			}
			AT_STOP(StringConstants::TP_CTOR_PIXELMAP);

			AT_START(StringConstants::TP_CTOR_COPY);
			cudaSafeCall(cudaMalloc((void**)&m_dev_viewRays, numPixel * sizeof(DeviceViewRay)));
			cudaSafeCall(cudaMemcpy(m_dev_viewRays, m_host_viewRays.data(), numPixel * sizeof(DeviceViewRay), cudaMemcpyHostToDevice));

			cudaSafeCall(cudaMalloc((void**)&m_dev_pixelMap, numVoxelCornerProjections * sizeof(int2)));
			cudaSafeCall(cudaMemcpy(m_dev_pixelMap, m_host_pixelMap.data(), numVoxelCornerProjections * sizeof(int2), cudaMemcpyHostToDevice));

			initViewRayMapMemory();

			AT_STOP(StringConstants::TP_CTOR_COPY);

			// Remove occluded pixel from the map
			if (!walls.empty())
			{
				AT_START(StringConstants::TP_CTOR_OCCLUSIONS);

				std::vector<float3> cameraCenters;
				cameraCenters.reserve(cameraModels.size());

				for (const auto & c : cameraModels)
				{
					cameraCenters.push_back(make_float3(static_cast<float>(c.getOrigin()(0)), static_cast<float>(c.getOrigin()(1)), static_cast<float>(c.getOrigin()(2))));
				}

				call_removePixelMapEntriesToNonVisiblePoints(m_dev_pixelMap, space.getVoxelDevice(), walls, cameraCenters, space.getNumVoxels());

				cudaSafeCall(cudaMemcpy(m_host_pixelMap.data(), m_dev_pixelMap, numVoxelCornerProjections * sizeof(int2), cudaMemcpyDeviceToHost));

				AT_STOP(StringConstants::TP_CTOR_OCCLUSIONS);
			}

			AT_STOP(StringConstants::TP_CTOR);
		}


		VoxelMap::~VoxelMap()
		{
			freeDeviceMemory();
		}


		std::vector<DeviceViewRay> VoxelMap::markGhostVoxel(std::vector<VoxelCluster> & clusters, unsigned char * p_dev_imagesSegmentation)
		{
			AT_START(StringConstants::TP_GHOSTS);

			AT_START(StringConstants::TP_GHOSTS_PRE);
			std::vector<DeviceVoxelCluster> deviceVoxelCluster;

			deviceVoxelCluster.reserve(clusters.size());

			std::transform(clusters.begin(), clusters.end(), std::back_inserter(deviceVoxelCluster), [](VoxelCluster & v)
			{
				v.setGhost(true); // We need to start with ghost status set to true and revert that later

				DeviceVoxelCluster dvc;
				dvc.boundingBox = v.getPreciseBoundingBox();
				dvc.isGhost = false;
				return dvc;
			});

			cudaSafeCall(cudaMemset(m_dev_viewRayMapMemory, 0, m_numCameras * m_imgSize.x * m_imgSize.y));
			AT_STOP(StringConstants::TP_GHOSTS_PRE);

			AT_START(StringConstants::TP_GHOSTS_KERNEL);
			call_checkForSingularViewRays(deviceVoxelCluster, m_dev_viewRays, p_dev_imagesSegmentation, m_dev_viewRayMapMemory, static_cast<uint>(m_numCameras), m_imgSize);

			cudaDeviceSynchronize();
			AT_STOP(StringConstants::TP_GHOSTS_KERNEL);

			AT_START(StringConstants::TP_GHOSTS_PAST);
			cudaSafeCall(cudaMemcpy(m_host_viewRayMapMemory.data(), m_dev_viewRayMapMemory, m_numCameras * m_imgSize.x * m_imgSize.y, cudaMemcpyDeviceToHost));


			/*m_viewRayMaps[0].writeToFile("C:/TEMP/VRM_0.pgm");
			m_viewRayMaps[1].writeToFile("C:/TEMP/VRM_1.pgm");
			m_viewRayMaps[2].writeToFile("C:/TEMP/VRM_2.pgm");
			m_viewRayMaps[3].writeToFile("C:/TEMP/VRM_3.pgm");

			writePGM("C:/TEMP/VRM_bin_0.pgm", m_host_viewRayMapMemory.get(), m_imgSize.x, m_imgSize.y);
			writePGM("C:/TEMP/VRM_bin_1.pgm", m_host_viewRayMapMemory.get() + m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			writePGM("C:/TEMP/VRM_bin_2.pgm", m_host_viewRayMapMemory.get() + 2 * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			writePGM("C:/TEMP/VRM_bin_3.pgm", m_host_viewRayMapMemory.get() + 3 * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);*/

			// TODO: Is it faster if we have a thread iterating over every cam and unifying the sets afterwards?
			std::set<unsigned char> nonGhostClusterIndices;
			std::vector<DeviceViewRay> singularViewRays;

			for (uint i = 0; i < m_numCameras; ++i)
			{
				for (uint j = 0; j < m_imgSize.x * m_imgSize.y; ++j)
				{
					uint offset = i * m_imgSize.x * m_imgSize.y + j;
					if (m_host_viewRayMapMemory[offset] != 0)
					{
						const unsigned char clusterIndex = m_host_viewRayMapMemory[offset] - 1; // Device Index starts at 1, 0 means no ray present here
						const auto insertInfo = nonGhostClusterIndices.insert(clusterIndex);

						if (insertInfo.second)
						{
							singularViewRays.push_back(m_host_viewRays[offset]);
							clusters[clusterIndex].setGhost(false);
						}
					}
				}
			}

			AT_STOP(StringConstants::TP_GHOSTS_PAST);

			AT_STOP(StringConstants::TP_GHOSTS);

			return singularViewRays;
		}


		void VoxelMap::setImageInfo(size_t numCameras, const uint2 & imgSize)
		{
			m_numCameras = static_cast<uint>(numCameras);
			m_imgSize = imgSize;

			initViewRayMapMemory();
		}


		void VoxelMap::setViewRaysToPixel(const std::vector<ViewRay_MemoryDouble> & viewRays)
		{
			m_host_viewRays.clear();
			m_host_viewRays.reserve(viewRays.size());

			for (const auto & v : viewRays)
			{
				m_host_viewRays.push_back(v.toDeviceViewRay());
			}

			cudaSafeCall(cudaFree(m_dev_viewRays));
			cudaSafeCall(cudaMalloc((void**)&m_dev_viewRays, m_host_viewRays.size() * sizeof(DeviceViewRay)));
			cudaSafeCall(cudaMemcpy(m_dev_viewRays, m_host_viewRays.data(), m_host_viewRays.size() * sizeof(DeviceViewRay), cudaMemcpyHostToDevice));
		}


		void VoxelMap::initViewRayMapMemory()
		{
			m_host_viewRayMapMemory.resize(m_numCameras * m_imgSize.x * m_imgSize.y);

			// Never change this method of constructing the images.
			// Successive construction somehow loses data.
			//m_viewRayMaps.resize(m_numCameras);
			for (uint i = 0; i < m_numCameras; ++i)
			{
			//	m_viewRayMaps[i].setExternalData(m_host_viewRayMapMemory.data() + i * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			}

			cudaSafeCall(cudaFree(m_dev_viewRayMapMemory));
			cudaSafeCall(cudaMalloc((void**)&m_dev_viewRayMapMemory, m_numCameras * m_imgSize.x * m_imgSize.y * sizeof(unsigned char)));
		}


		void VoxelMap::freeDeviceMemory()
		{
			cudaSafeCall(cudaFree(m_dev_viewRays));
			cudaSafeCall(cudaFree(m_dev_viewRayMapMemory));
			cudaSafeCall(cudaFree(m_dev_pixelMap));

			m_dev_viewRays = nullptr;
			m_dev_viewRayMapMemory = nullptr;
			m_dev_pixelMap = nullptr;
		}


		std::vector<DeviceViewRay> VoxelMap::cpu_markGhostVoxel(std::vector<VoxelCluster> & clusters, unsigned char * p_dev_imagesSegmentation)
		{
			// Just for debugging purposes and very slow. Please don't call
			std::vector<DeviceVoxelCluster> deviceVoxelCluster;

			deviceVoxelCluster.reserve(clusters.size());

			std::transform(clusters.begin(), clusters.end(), std::back_inserter(deviceVoxelCluster), [](const VoxelCluster & v)
			{
				DeviceVoxelCluster dvc;
				dvc.boundingBox = v.getPreciseBoundingBox();
				dvc.isGhost = false;
				return dvc;
			});

			std::unique_ptr<unsigned char[]> p_host_imagesSegmentation(new unsigned char[m_numCameras * m_imgSize.x * m_imgSize.y]);
			cudaSafeCall(cudaMemcpy(p_host_imagesSegmentation.get(), p_dev_imagesSegmentation, m_numCameras * m_imgSize.x * m_imgSize.y, cudaMemcpyDeviceToHost));

			memset(m_host_viewRayMapMemory.data(), 0, m_numCameras * m_imgSize.x * m_imgSize.y);

			//std::vector<rtcvImage8U> imagesSegmentation;
			//imagesSegmentation.resize(m_numCameras);
			for (size_t i = 0; i < m_numCameras; ++i)
			{
				//imagesSegmentation[i].setExternalData(p_host_imagesSegmentation.get() + i * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			}

			for (size_t iCam = 0; iCam < m_numCameras; ++iCam)
			{
				const size_t offsetImage = iCam * m_imgSize.x * m_imgSize.y;

				for (uint iImg = 0; iImg < m_imgSize.x * m_imgSize.y; ++iImg)
				{
					uint intersectionCounter = 0;
					for (uint iCluster = 0; iCluster < deviceVoxelCluster.size(); ++iCluster)
					{
						const bool intersectsCluster = intersectsInPlane(deviceVoxelCluster[iCluster].boundingBox, m_host_viewRays[offsetImage + iImg]);
						//if (intersectsCluster && imagesSegmentation[iCam].getValue(iImg) != 0)
						{
						//	++intersectionCounter;
							//if(counterIntersections == 1)
							//{
						//	m_host_viewRayMapMemory[offsetImage + iImg] = 255;
							//}
							/*viewRayMaps[offsetImage + offsetPixel] = counterIntersections == 1 ? idxCluster : -1;*/
						}
					}
				}
			}

			//m_viewRayMaps[0].writeToFile("C:/TEMP/VRM_0.pgm");
			//m_viewRayMaps[1].writeToFile("C:/TEMP/VRM_1.pgm");
			//m_viewRayMaps[2].writeToFile("C:/TEMP/VRM_2.pgm");
			//m_viewRayMaps[3].writeToFile("C:/TEMP/VRM_3.pgm");
			//
			//writePGM("C:/TEMP/VRM_bin_0.pgm", m_host_viewRayMapMemory.data(), m_imgSize.x, m_imgSize.y);
			//writePGM("C:/TEMP/VRM_bin_1.pgm", m_host_viewRayMapMemory.data() + m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			//writePGM("C:/TEMP/VRM_bin_2.pgm", m_host_viewRayMapMemory.data() + 2 * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);
			//writePGM("C:/TEMP/VRM_bin_3.pgm", m_host_viewRayMapMemory.data() + 3 * m_imgSize.x * m_imgSize.y, m_imgSize.x, m_imgSize.y);

			// Todo is ist faster if we have a thread iterating over every cam and unifying the sets afterwards?
			std::set<int> nonGhostClusterIndices;
			std::vector<DeviceViewRay> singularViewRays;

			for (uint i = 0; i < m_numCameras; ++i)
			{
				uint addedRays = 0;
				for (uint j = 0; j < m_imgSize.x * m_imgSize.y; ++j)
				{
					uint offset = i * m_imgSize.x * m_imgSize.y + j;
					if (m_host_viewRayMapMemory[offset] == 255)
					{
						nonGhostClusterIndices.insert(m_host_viewRayMapMemory[offset]);
						singularViewRays.push_back(m_host_viewRays[offset]);
						++addedRays;
					}
				}
			}
			std::cout << singularViewRays.size() << " View Rays" << std::endl;

			for (int i = 0; i < clusters.size(); ++i)
			{
				bool isGhost = nonGhostClusterIndices.find(i) == nonGhostClusterIndices.end();
				clusters[i].setGhost(isGhost);
			}

			return singularViewRays;
		}
	}
}