#include "ExtendedVoxel.h"

#include "opencv2/imgcodecs.hpp"


#include "vectorOperations.h"

#include "convexHull_legacy.h"


//#define USE_OLD_VOXEL_INIT

namespace sfs
{
	bool contains(const cv::Size & s, const int2 & p)
	{
		return p.x >= 0 && p.y >= 0 && p.x < s.width && p.y < s.height;
	}


	void drawPolygon(cv::Mat * pImg, const std::vector<float2> & polygon, const unsigned char color)
	{
		assert(polygon.size() > 1);

		for (int i = 1; i <= polygon.size(); ++i)
		{
			int2 currentPoint = make_int2
			(
				static_cast<int>(std::floor(polygon[i % polygon.size()].x + 0.5f)),
			    static_cast<int>(std::floor(polygon[i % polygon.size()].y + 0.5f))
			);
			
			int2 lastPoint = make_int2
			(
				static_cast<int>(std::floor(polygon[i - 1].x + 0.5f)), 
				static_cast<int>(std::floor(polygon[i - 1].y + 0.5))
			);

			if (!contains(pImg->size(), currentPoint) || !contains(pImg->size(), lastPoint) || (currentPoint ==
				lastPoint))
				continue;

			cv::line(*pImg, cv::Point(lastPoint.x, lastPoint.y), cv::Point(currentPoint.x, currentPoint.y), color);
		}
	}


	std::ostream & operator<<(std::ostream & os, const ExtendedVoxel::VisibilityStatus & s)
	{
		os << static_cast<int>(s);
		return os;
	}


	std::istream & operator>>(std::istream & is, ExtendedVoxel::VisibilityStatus & s)
	{
		int i;
		is >> i;
		s = ExtendedVoxel::VisibilityStatus(i);
		return is;
	}


	bool operator==(const ExtendedVoxel & left, const ExtendedVoxel & right)
	{
		return left.m_voxel == right.m_voxel &&
			left.m_imgSize == right.m_imgSize &&
			left.m_visibilityStats == right.m_visibilityStats &&
			left.m_convexHull == right.m_convexHull &&
			left.m_imgPoints == right.m_imgPoints &&
			left.m_numImgPoints == right.m_numImgPoints;
	}


	bool operator!=(const ExtendedVoxel & left, const ExtendedVoxel & right)
	{
		return !(left == right);
	}


	void writeBinary(std::ostream & os, const ExtendedVoxel & v)
	{
		writeBinary(os, v.getCenter());
		writeBinary(os, v.getSize());
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

		writeBinaryTrivial(os, v.m_numImgPoints);
	}


	void readBinary(std::istream & is, ExtendedVoxel & v)
	{
		readBinary(is, v.m_voxel.center);
		readBinary(is, v.m_voxel.size);
		readBinary(is, v.m_imgSize);
		readBinaryTrivial(is, v.m_visibilityStats);

		size_t s;
		is.read(reinterpret_cast<char*>(&s), sizeof(s));
		v.m_convexHull.resize(s);

		for (size_t i = 0; i < s; ++i)
		{
			readBinaryTrivial(is, v.m_convexHull[i]);
		}

		is.read(reinterpret_cast<char*>(&s), sizeof(s));
		v.m_imgPoints.resize(s);

		for (size_t i = 0; i < s; ++i)
		{
			readBinaryTrivial(is, v.m_imgPoints[i]);
		}

		readBinaryTrivial(is, v.m_numImgPoints);

		v.calcNonSerializedValues();
	}


	ExtendedVoxel::ExtendedVoxel() :
		m_imgSize(),
		m_maxNumVisibleCameras(),
		m_numMarkedCameras()
	{
		// empty; Can only be filled via stream operators
	}


	ExtendedVoxel::ExtendedVoxel(
		const float3 & center,
		const float3 & size,
		const CameraSet & cameraModels,
		const Mesh & walls) :
		ExtendedVoxel(Voxel(center, size), cameraModels, walls)
	{
		// empty
	}


	ExtendedVoxel::ExtendedVoxel(const Voxel & voxel, const CameraSet & cameraModels, const Mesh & walls) :
		m_voxel(voxel),
		m_imgSize(),
		m_visibilityStats(cameraModels.size(), NotVisible),
		m_segmentationStats(cameraModels.size(), None),
		m_maxNumVisibleCameras(static_cast<int>(cameraModels.size())),
		m_numMarkedCameras()
	{
		assert(!cameraModels.empty());

		auto imageSize = cameraModels.begin()->getImageSize();
		m_imgSize = make_int2(imageSize.get()(0), imageSize.get()(1));

		for (size_t i = 0; i < cameraModels.size(); ++i)
		{
			const auto & camera = *std::next(cameraModels.begin(), i);

			std::vector<float2> rawPolygon = calcRawPolygon(m_voxel, camera);
			if (rawPolygon.empty()) // Voxel is outside of the camera image
			{
				m_visibilityStats[i] = NotVisible;
				m_segmentationStats[i] = None;
				--m_maxNumVisibleCameras;

				m_convexHull.push_back(std::vector<float2>());
				m_imgPoints.push_back(std::vector<int2>());
				m_numImgPoints.push_back(0);

				continue;
			}

			if (isOccluded(m_voxel, camera, walls)) // Voxel is hidden behind a structure
			{
				m_visibilityStats[i] = Occluded;
				m_segmentationStats[i] = None;
				--m_maxNumVisibleCameras;

				m_convexHull.push_back(std::vector<float2>());
				m_imgPoints.push_back(std::vector<int2>());
				m_numImgPoints.push_back(0);

				continue;
			}

			// Otherwise the Voxel is visible and currently not marked
			m_visibilityStats[i] = Visible;
			m_segmentationStats[i] = NotMarked;

			m_convexHull.push_back(calcConvexHull(rawPolygon));
			m_imgPoints.push_back(calcImageIndices(m_convexHull[i]));
			m_numImgPoints.push_back(calcNumImagePoints(i));
		}
	}


	void ExtendedVoxel::updateStatus(const std::vector<cv::Mat> & imagesSegmentation, double minSegmentedFactor,
	                                 GeometryPredicate predicate)
	{
		assert(imagesSegmentation.size() == m_visibilityStats.size());

		if (m_maxNumVisibleCameras < 2)
		{
			return;
		}

		m_numMarkedCameras = 0;

		for (int cameraIndex = 0; cameraIndex < imagesSegmentation.size(); ++cameraIndex)
		{
			if (m_visibilityStats[cameraIndex] != Visible)
				continue;

			int segmentationCounter = 0;

			const std::vector<int2> & imgPointsSingleCam = m_imgPoints[cameraIndex];

			assert(imgPointsSingleCam.size() % 2 == 0);
			for (size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
			{
				assert(imgPointsSingleCam[i].y == imgPointsSingleCam[i + 1].y);
				for (int x = imgPointsSingleCam[i].x; x <= imgPointsSingleCam[i + 1].x; ++x)
				{
					//if(imagesSegmentation[cameraIndex].at<unsigned char>(x, imgPointsSingleCam[i].y) > 200)
					if (imagesSegmentation[cameraIndex].at<unsigned char>(imgPointsSingleCam[i].y, x) != 0)
					{
						++segmentationCounter;
					}
				}
			}
			const double segmentationThreshold = m_numImgPoints[cameraIndex] * minSegmentedFactor;

			SegmentationStatus s = segmentationCounter >= segmentationThreshold ? Marked : NotMarked;
			m_segmentationStats[cameraIndex] = s;
			if (s == Marked)
			{
				++m_numMarkedCameras;
			}
		}

		updateStatus(predicate);
	}


	void ExtendedVoxel::updateStatus(GeometryPredicate predicate)
	{
		const bool active = predicate(m_segmentationStats, m_visibilityStats);

		if (active != m_voxel.isActive)
		{
			m_voxel.isActive = active;
		}

		segmentationChanged(active);
	}


	std::vector<float2> ExtendedVoxel::calcRawPolygon(const Voxel & voxel, const WorldCamera & cameraModel) const
	{
		std::vector<float2> rawPolygon;

		const auto & corners = voxel.getCorners();
		for (const auto & v : corners)
		{
			const auto [imagePoint, visible] = cameraModel.worldToImageWithVisibilityCheck(
				make_named<WorldVector>(v.x, v.y, v.z));

			if (!visible)
			{
				rawPolygon.clear();
				break;
			}

			rawPolygon.push_back(make_float2
			(
				static_cast<float>(imagePoint.get()(0)), 
				static_cast<float>(imagePoint.get()(1)))
			);
		}
		return rawPolygon;
	}


	std::vector<float2> ExtendedVoxel::calcConvexHull(const std::vector<float2> & rawPolygon)
	{
		std::vector<float2> convexHull;

		#ifdef USE_OLD_VOXEL_INIT
		std::vector<Vector2> eigenPolygon;

		for (const auto & p : rawPolygon)
		{
			Vector2 eigenP;
			eigenP << p.y, p.x;
			eigenPolygon.push_back(eigenP);
		}
		int lastIdx = AnalyticGeometry::sortForConvexHull(eigenPolygon);

		for (int i = 0; i < lastIdx; ++i)
		{
			convexHull.push_back(float2(eigenPolygon[i](1), eigenPolygon[i](0)));
		}
		#else
		convexHull = calculateConvexHull(rawPolygon);
		#endif

		return convexHull;
	}


	std::vector<int2> ExtendedVoxel::calcImageIndices(std::vector<float2> & convexHull)
	{
		if (convexHull.empty())
			return std::vector<int2>();

		#ifdef USE_OLD_VOXEL_INIT
		QPolygonF qPolygon;
		for (const auto & point : convexHull)
		{
			qPolygon << QPointF(point.x, point.y);
		}

		QRect boundingBox = qPolygon.boundingRect().toRect();

		std::vector<int2> indexPoints;
		indexPoints.reserve(2 * boundingBox.height());

		for (int y = boundingBox.top(); y < boundingBox.bottom(); ++y)
		{
			bool isInside = false;
			for (int x = boundingBox.left(); x < boundingBox.right(); ++x)
			{
				QPoint qp(x, y);
				int2 p(x, y);

				const bool contains = qPolygon.containsPoint(qp, Qt::WindingFill);

				if (!isInside && contains) // first segmented point in line
				{
					isInside = true;
					indexPoints.push_back(p);
				}

				if (isInside && !contains) // last segmented point in line
				{
					isInside = false;
					indexPoints.push_back(int2(x - 1, y));
					break;
				}

				if (isInside && x + 1 >= boundingBox.right()) // last segmented point in line is also at the right of bounding box
				{
					isInside = false;
					indexPoints.push_back(p);
					break;
				}
			}
		}
		indexPoints.shrink_to_fit();

		return indexPoints;
		#else
		return calculatePointsOnHullEdge(convexHull);
		#endif
	}


	bool ExtendedVoxel::isOccluded(const Voxel & voxel, const WorldCamera & cameraModel,
	                               const Mesh & walls) const
	{
		const auto cameraCenterForOcclusion = cameraModel.getOrigin();

		const auto & corners = voxel.getCorners();
		for (const auto & v : corners)
		{
			const auto & ve = make_named<WorldVector>(v.x, v.y, v.z);

			for (const auto & w : walls)
			{
				if (w.doesIntersectLine(ve.get(), cameraCenterForOcclusion.get()))
				{
					return true;
				}
			}
		}
		return false;
	}


	void ExtendedVoxel::calcNonSerializedValues()
	{
		m_segmentationStats.clear();

		for (int i = 0; i < m_visibilityStats.size(); ++i)
		{
			switch (m_visibilityStats[i])
			{
			case Visible: m_segmentationStats.push_back(NotMarked);
				break;
			case NotVisible:
			case Occluded: m_segmentationStats.push_back(None);
				break;
			default: throw std::runtime_error("Read invalid visibility status!");
				break;
			}
		}

		m_maxNumVisibleCameras =
			static_cast<int>(
			std::count_if(m_visibilityStats.begin(), m_visibilityStats.end(),
			              [](const VisibilityStatus & s) { return s == Visible; }
			));

		m_numMarkedCameras = 0;
	}


	double ExtendedVoxel::distTo(const ExtendedVoxel & other) const
	{
		assert(m_voxel.size == other.m_voxel.size); // We can only cluster voxels of identical size

		float3 distVec = m_voxel.center - other.m_voxel.center;
		distVec = distVec / m_voxel.size;
		distVec.z = 0; // Voxels above or underneath each other can always be clustered together

		return length(distVec);
	}


	void ExtendedVoxel::loopOverImgPoints(const int cameraIndex, std::function<void(int x, int y)> f) const
	{
		const std::vector<int2> & imgPointsSingleCam = m_imgPoints[cameraIndex];

		assert(imgPointsSingleCam.size() % 2 == 0);
		for (size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			assert(imgPointsSingleCam[i].y == imgPointsSingleCam[i + 1].y);
			for (int x = imgPointsSingleCam[i].x; x <= imgPointsSingleCam[i + 1].x; ++x)
			{
				f(x, imgPointsSingleCam[i].y);
			}
		}
	}


	size_t ExtendedVoxel::calcNumImagePoints(size_t cameraIndex) const
	{
		size_t numPixels = 0;
		loopOverImgPoints(static_cast<int>(cameraIndex), [&numPixels](int x, int y) { ++numPixels; });

		size_t numPixels2 = 0;
		const std::vector<int2> & imgPointsSingleCam = m_imgPoints[cameraIndex];

		assert(imgPointsSingleCam.size() % 2 == 0);
		for (size_t i = 0; i < imgPointsSingleCam.size(); i += 2)
		{
			assert(imgPointsSingleCam[i].y == imgPointsSingleCam[i + 1].y);
			numPixels2 += imgPointsSingleCam[i + 1].x - imgPointsSingleCam[i].x + 1;
		}

		if (numPixels2 != numPixels)
		{
			std::cout << numPixels2 << " -- " << numPixels << std::endl;
		}
		return numPixels;
	}


	bool ExtendedVoxel::debugDrawPolygon()
	{
		// Polygon Zeichnet sich nur, wenn es in allen Kameras sichtbar ist
		bool shouldDraw = true;
		for (int i = 0; i < m_visibilityStats.size(); ++i)
		{
			if (m_visibilityStats[i] != Visible)
			{
				shouldDraw = false;
				break;
			}
		}

		if (!shouldDraw)
			return false;

		for (int i = 0; i < m_convexHull.size(); ++i)
		{
			cv::Mat img(cv::Size(m_imgSize.y, m_imgSize.x), CV_8U, cv::Scalar(0));

			loopOverImgPoints(i, [&img](int x, int y) { img.at<unsigned char>(y, x) = 64; });

			//drawPolygon(&img, m_rawPolygon[i], 128);
			drawPolygon(&img, m_convexHull[i], 255);

			std::stringstream ss;
			ss << "C:/TEMP/PolyTest_Cam" << i << ".pgm";
			cv::imwrite(cv::String(ss.str().c_str()), img);
		}

		return true;
	}


	void ExtendedVoxel::setActive(bool b)
	{
		m_voxel.isActive = b;
		segmentationChanged(b);
	}


	void ExtendedVoxel::segmentationChanged(const bool status) const
	{
		for (AbstractVoxelStateListener * pListener : m_statusListeners)
		{
			pListener->onVoxelStateChanged(status);
		}
	}
}
