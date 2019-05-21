#include "VoxelSegmentationInfo.cuh"

#include "cuda_error_check.h"
#include "vectorOperations.h"

namespace sfs
{
	namespace cuda
	{


		std::ostream & operator<<(std::ostream & os, const VoxelVisibilityStatus & s)
		{
			os << static_cast<int>(s);
			return os;
		}


		std::istream & operator>>(std::istream & is, VoxelVisibilityStatus & s)
		{
			int i;
			is >> i;
			s = VoxelVisibilityStatus(i);
			return is;
		}


		bool operator==(const VoxelSegmentationInfo & left, const VoxelSegmentationInfo & right)
		{
			return left.m_center == right.m_center &&
				left.m_size == right.m_size &&
				left.m_imgSize == right.m_imgSize &&
				left.m_visibilityStats == right.m_visibilityStats &&
				left.m_imgPoints == right.m_imgPoints &&
				left.m_numImgPoints == right.m_numImgPoints;
		}


		bool operator!=(const VoxelSegmentationInfo & left, const VoxelSegmentationInfo & right)
		{
			return !(left == right);
		}


		// Reads a Voxel in such a way, that all pointers point to device memory, while the variables are still residing in host memory
		void readBinary(std::istream & is, VoxelSegmentationInfo & v)
		{
			// All Variables that can be read from the serialization
			float3 center;
			float3 size;
			int2 imgSize;
			std::vector<VoxelVisibilityStatus> visibilityStats;
			std::vector<std::vector<float2>> convexHull; // Convex Hull is discarded since it is only necessary for debug purposes and not used on the device right now. 
			std::vector<std::vector<int2>> imgPoints;
			std::vector<size_t> numImgPoints;

			readBinary(is, center);
			readBinary(is, size);
			readBinary(is, imgSize);
			readBinaryTrivial(is, visibilityStats);

			size_t s;
			is.read(reinterpret_cast<char*>(&s), sizeof(s));
			convexHull.resize(s);

			for (size_t i = 0; i < s; ++i)
			{
				readBinaryTrivial(is, convexHull[i]);
			}

			is.read(reinterpret_cast<char*>(&s), sizeof(s));
			imgPoints.resize(s);

			for (size_t i = 0; i < s; ++i)
			{
				readBinaryTrivial(is, imgPoints[i]);
			}

			readBinaryTrivial(is, numImgPoints);

			//v.freeDeviceMemory(); // Just in case we're reading into an existing voxel

			v.setSimpleSerializedValues(center, size, imgSize);
			v.setVisibilityStats(visibilityStats);
			v.setImagePoints(imgPoints, numImgPoints);
			v.setNonSerializableMembers(visibilityStats.size(), visibilityStats);
		}


		void writeBinary(std::ostream & os, const VoxelSegmentationInfo & v)
		{
			throw "Not implemented!";
			/*writeBinary(os, v.m_center);
			writeBinary(os, v.m_size);
			writeBinary(os, v.m_imgSize);
			writeBinaryTrivial(os, v.m_visibilityStats);

			size_t s = v.m_convexHull.size();
			os.write(reinterpret_cast<const char*>(&s), sizeof(s));

			for (size_t i = 0; i < s; ++i)
			{
				writeBinaryTrivial(os, v.m_convexHull[i]);
			}

			s = v.m_imgPoints.size();
			os.write(reinterpret_cast<const char*>(&s), sizeof(s));

			for (size_t i = 0; i < s; ++i)
			{
				writeBinaryTrivial(os, v.m_imgPoints[i]);
			}

			writeBinaryTrivial(os, v.m_numImgPoints);*/
		}


		VoxelSegmentationInfo::VoxelSegmentationInfo() :
			m_center(), m_size(), m_imgSize(),
			m_visibilityStats(nullptr),
			m_imgPoints(nullptr),
			m_numImgPoints(nullptr),
			m_numImgRows(nullptr),
			m_numImages(0),
			m_maxNumVisibleCameras()
		{
			// empty; Can only be filled via stream operators
		}


		//Voxel::Voxel(const osg::Vec3 & center, const osg::Vec3 & size, const std::vector<RadialCameraModel> & cameraModels, const FacesWithNormals & walls) :
		//	m_center(center), m_size(size), m_imgSize(),
		//	m_visibilityStats(cameraModels.size(), NotVisible), m_convexHull(), 
		//	m_segmentationStats(cameraModels.size(), None), m_imgPoints(), m_isActive(false),
		//	m_maxNumVisibleCameras(cameraModels.size()), 
		//	m_numMarkedCameras(), m_statusListeners()
		//{
		//	assert(!cameraModels.empty());
		//
		//	m_imgSize = rtcvPoint(cameraModels[0].m_imgWidth, cameraModels[0].m_imgHeight);
		//
		//	for(int i = 0; i < cameraModels.size(); ++i)
		//	{
		//		std::vector<rtcvPointF> rawPolygon = calcRawPolygon(center, size, cameraModels[i], walls);
		//		if(rawPolygon.empty()) // Voxel is outside of the camera image
		//		{
		//			m_visibilityStats[i] = NotVisible;
		//			m_segmentationStats[i] = None;
		//			--m_maxNumVisibleCameras;
		//
		//			m_convexHull.push_back(std::vector<rtcvPointF>());
		//			m_imgPoints.push_back(std::vector<rtcvPoint>());
		//			m_numImgPoints.push_back(0);
		//
		//			continue;
		//		}
		//
		//		if(isOccluded(center, size, cameraModels[i], walls)) // Voxel is hidden behind a structure
		//		{
		//			m_visibilityStats[i] = Occluded;
		//			m_segmentationStats[i] = None;
		//			--m_maxNumVisibleCameras;
		//
		//			m_convexHull.push_back(std::vector<rtcvPointF>());
		//			m_imgPoints.push_back(std::vector<rtcvPoint>());
		//			m_numImgPoints.push_back(0);
		//
		//			continue;
		//		}
		//
		//		// Otherwise the Voxel is visible and currently not marked
		//		m_visibilityStats[i] = Visible;
		//		m_segmentationStats[i] = NotMarked;
		//
		//		m_convexHull.push_back(calcConvexHull(rawPolygon));
		//		m_imgPoints.push_back(calcImageIndices(m_convexHull[i]));
		//		m_numImgPoints.push_back(calcNumImagePoints(i));
		//	}
		//}


		VoxelSegmentationInfo::~VoxelSegmentationInfo()
		{
			// empty
		}

		__device__ TemporaryDeviceVector<uint2> VoxelSegmentationInfo::getImgPoints(uint cameraIndex) const
		{
			uint2 * pStart = m_imgPoints;
			for (uint i = 0; i < cameraIndex; ++i)
			{
				pStart += m_numImgRows[i] * 2;
			}
			uint2 * pEnd = pStart + m_numImgRows[cameraIndex] * 2;
			return TemporaryDeviceVector<uint2>(pStart, pEnd);
		}

		__host__ void VoxelSegmentationInfo::setSimpleSerializedValues(const float3 & center, const float3 & size, const int2 & imgSize)
		{
			m_center = center;
			m_size = size;

			m_imgSize = make_uint2(imgSize.x, imgSize.y);
		}


		__host__ void VoxelSegmentationInfo::setVisibilityStats(const std::vector<VoxelVisibilityStatus> & visibilityStats)
		{
			const size_t sizeVisibilityStats = visibilityStats.size() * sizeof(VoxelVisibilityStatus);

			cudaSafeCall(cudaFree(m_visibilityStats));
			cudaSafeCall(cudaMalloc((void**)&m_visibilityStats, sizeVisibilityStats));
			cudaSafeCall(cudaMemcpy(m_visibilityStats, visibilityStats.data(), sizeVisibilityStats, cudaMemcpyHostToDevice));
		}


		__host__ void VoxelSegmentationInfo::setImagePoints(const std::vector<std::vector<int2>> & imgPoints, const std::vector<size_t> & numImgPoints)
		{
			// Naming here is a bit confusing, but I couldn't find a better one since there is a difference between the number of image points
			// that need to be traversed for the algorithm and the size of the respective vectors. The inner vectors of imgPoints only store
			// the start- and endpoints of each row consecutively: [start1, end1, start2, end2, start3, end3, ...]
			// Therefore the actual number of image points need to be stored separately.
			// 
			// numImgPoints: The total number of points that are in the convex hull
			// numImgRows: The length of the inner vector of imgPoints divided by 2

			std::vector<uint> numImgRows; //length of the inner imgPoints vectors

			assert(imgPoints.size() == numImgPoints.size());

			const size_t numCameras = imgPoints.size();
			size_t sizeImgPointsLinearized = 0;

			for (size_t i = 0; i < numCameras; ++i)
			{
				assert(imgPoints[i].size() % 2 == 0);
				numImgRows.push_back(static_cast<unsigned int>(imgPoints[i].size() / 2));
				sizeImgPointsLinearized += imgPoints[i].size();
			}

			std::vector<uint2> imgPointsLinearized;
			imgPointsLinearized.reserve(sizeImgPointsLinearized);

			for (const auto & v : imgPoints)
			{
				for (const auto & p : v)
				{
					uint2 point = make_uint2(p.x, p.y);
					imgPointsLinearized.push_back(point);
				}
			}

			cudaSafeCall(cudaFree(m_imgPoints));
			cudaSafeCall(cudaMalloc((void**)&m_imgPoints, sizeImgPointsLinearized * sizeof(uint2)));
			cudaSafeCall(cudaMemcpy(m_imgPoints, imgPointsLinearized.data(), sizeImgPointsLinearized * sizeof(uint2), cudaMemcpyHostToDevice));

			cudaSafeCall(cudaFree(m_numImgRows));
			cudaSafeCall(cudaMalloc((void**)&m_numImgRows, numImgRows.size() * sizeof(uint)));
			cudaSafeCall(cudaMemcpy(m_numImgRows, numImgRows.data(), numImgRows.size() * sizeof(uint), cudaMemcpyHostToDevice));

			cudaSafeCall(cudaFree(m_numImgPoints));
			cudaSafeCall(cudaMalloc((void**)&m_numImgPoints, numImgPoints.size() * sizeof(size_t)));
			cudaSafeCall(cudaMemcpy(m_numImgPoints, numImgPoints.data(), numImgPoints.size() * sizeof(size_t), cudaMemcpyHostToDevice));
		}

		__host__ void VoxelSegmentationInfo::setNonSerializableMembers(size_t numCameras, const std::vector<VoxelVisibilityStatus> & visibilityStats)
		{
			m_numImages = static_cast<unsigned int>(numCameras);

			m_maxNumVisibleCameras = static_cast<uint>(std::count_if(visibilityStats.begin(), visibilityStats.end(),
				[](const VoxelVisibilityStatus & s) { return s == Visible; }
			));

			const size_t sizeSegmentationStats = visibilityStats.size() * sizeof(VoxelSegmentationStatus); // Same size as the visibility

			cudaSafeCall(cudaFree(m_segmentationStats));
			cudaSafeCall(cudaMalloc((void**)&m_segmentationStats, sizeSegmentationStats));
			cudaSafeCall(cudaMemset(m_segmentationStats, 0, sizeSegmentationStats));

			m_numMarkedCameras = 0;
		}


		//void VoxelSegmentationInfo::freeDeviceMemory()
		//{
		//	cudaFree(m_visibilityStats);
		//	cudaFree(m_segmentationStats);
		//	cudaFree(m_imgPoints);
		//	cudaFree(m_numImgPoints);
		//	cudaFree(m_numImgRows);
		//
		//	m_visibilityStats = nullptr;
		//	m_segmentationStats = nullptr;
		//	m_imgPoints = nullptr;
		//	m_numImgPoints = nullptr;
		//	m_numImgRows = nullptr;
		//}


		//void VoxelSegmentationInfo::updateStatus(const std::vector<const rtcvImage8U *> & imagesSegmentation, double minPercentSegmentedPixel, GeometryPredicate predicate)
		//{
			/*assert(imagesSegmentation.size() == m_visibilityStats.size());

			if(m_maxNumVisibleCameras < 2)
			{
				return;
			}

			m_numMarkedCameras = 0;

			for(int cameraIndex = 0; cameraIndex < imagesSegmentation.size(); ++cameraIndex)
			{
				if(m_visibilityStats[cameraIndex] != Visible)
					continue;

				int segmentationCounter = 0;

				const std::vector<rtcvPoint> & imgPointsSingleCam = m_imgPoints[cameraIndex];

				assert(imgPointsSingleCam.size() % 2 == 0);
				for(size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
				{
					assert(imgPointsSingleCam[i].y == imgPointsSingleCam[i+1].y);
					for(int x = imgPointsSingleCam[i].x; x <= imgPointsSingleCam[i+1].x; ++x)
					{
						if(imagesSegmentation[cameraIndex]->getValue(x, imgPointsSingleCam[i].y))
						{
							++segmentationCounter;
						}
					}
				}
				const double segmentationThreshold = m_numImgPoints[cameraIndex] * (minPercentSegmentedPixel / 100.);

				SegmentationStatus s = segmentationCounter >= segmentationThreshold ? Marked : NotMarked;
				m_segmentationStats[cameraIndex] = s;
				if(s == Marked)
				{
					++m_numMarkedCameras;
				}
			}

			updateStatus(predicate);*/
			//}


			//void VoxelSegmentationInfo::updateStatus(GeometryPredicate predicate)
			//{
				/*bool active = predicate(m_segmentationStats, m_visibilityStats);

				if(active != m_isActive)
				{
					m_isActive = active;

				}

				segmentationChanged(active);*/
				//}

	}
}