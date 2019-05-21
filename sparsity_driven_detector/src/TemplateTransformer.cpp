#include "TemplateTransformer.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "boost/filesystem.hpp"

#include "opencv2/imgcodecs.hpp"

#include "enumerate.h"
#include "listFilesInDirectory.h"

#include "serialization_helper.h"

#ifdef _MSC_VER
#define checked_iterator(ty, name, len) stdext::checked_array_iterator<ty>(name, len)
#else
#define checked_iterator(ty, name, len) std::begin(name)
#endif

TemplateTransformer::TemplateTransformer() : m_targetImageSize(make_named<ImageSize>(0, 0))
{
	// empty
	// Private so that it can only be called by the load function.
	// We don't want empty dictionaries floating around
}


TemplateTransformer::TemplateTransformer(const CameraSet & cameras, const Template & objectTemplate,
                       const std::vector<WorldVector> & points, const ImageSize & targetImageSize,
					   const Mesh & walls, bool debug_skipTransform) :
	m_targetImageSize(targetImageSize),
	m_template(objectTemplate),
	m_entries(points.size(), std::vector<unsigned char>())
{
	std::vector<cv::Mat> matPerCamera;
	for (const auto & cam : cameras)
	{
		const auto & imgSize = cam.getImageSize();
		matPerCamera.emplace_back(imgSize(1), imgSize(0), CV_8U);
	}

	const size_t numPixelsPerCam = m_targetImageSize.get().prod();
	const size_t numPixelsTotal = numPixelsPerCam * cameras.size();

	for (int idx_point = 0; idx_point < points.size(); ++idx_point)
	{
		m_entries[idx_point].resize(numPixelsTotal);

		if(debug_skipTransform)
		{
			continue;
		}

		for (size_t idx_cam = 0; idx_cam < cameras.size(); ++idx_cam)
		{
			const auto & cam = *std::next(cameras.begin(), idx_cam); // std::set has no operator[]
			matPerCamera[idx_cam].setTo(0);

			transformTemplate(objectTemplate, cam, points[idx_point], matPerCamera[idx_cam], walls);

			void * const targetData = static_cast<void*>(m_entries[idx_point].data() + numPixelsPerCam * idx_cam);

			cv::Mat target(m_targetImageSize(1), m_targetImageSize(0), CV_8U, targetData);

			cv::resize(matPerCamera[idx_cam], target, cv::Size(m_targetImageSize(0), m_targetImageSize(1)));
			cv::threshold(target, target, 50, 1, cv::THRESH_BINARY);
		}

		std::cout << "\r\tProgress: " << ((idx_point + 1.0) / points.size()) * 100.0 << "%     ";
	}
	std::cout << "\n\tDone" << std::endl;
}


std::vector<cv::Mat> TemplateTransformer::getEntryAsImages(size_t idx) const
{
	std::vector<cv::Mat> ret;
	for (int i = 0; i < getNumCameras(); ++i)
	{
		// images must not be modified only looked at
		void * data = const_cast<void*>(static_cast<const void*>(m_entries[idx].data() + getNumPixelPerCamera() * i));
		ret.emplace_back(m_targetImageSize(1), m_targetImageSize(0), CV_8U, data);
	}
	return ret;
}


void TemplateTransformer::transformTemplate(const Template & t, const IdentifiableCamera & cam, 
	const WorldVector & targetPos, cv::Mat & img, const Mesh & walls)
{
	using Distance = NamedVectorTypes<ScalarType, 3>::Distance;
	using Direction = NamedVectorTypes<ScalarType, 3>::Direction;
	using Vector3 = GeometricTypes<ScalarType, 3>::Vector;
	using Size = Template::Size;

	const Size templateImageSize = t.getImgSize();

	// topLeft, topRight, bottomRight, bottomLeft | (row, col)
	cv::Point2f templateImageCorners[] =
	{
		cv::Point2f(0, 0),
		cv::Point2f(templateImageSize(0), 0),
		cv::Point2f(templateImageSize(0), templateImageSize(1)),
		cv::Point2d(0, templateImageSize(1))
	};

	const auto dist = make_named<Distance>(targetPos.get() - cam.getOrigin().get());

	auto dir = make_named<Direction>(dist(0), dist(1), 0.f);
	dir.get() /= dir.get().norm();

	auto normal = make_named<Direction>(-dir(1), dir(0), dir(2));
	const Size templateSize = t.getInfo().targetSize;

	// topLeft, topRight, bottomRight, bottomLeft | (x, y, z)
	WorldVector templateCorners_w[] =
	{
		make_named<WorldVector>(targetPos.get() - (templateSize(0) / 2) * normal.get() + Vector3(0, 0, templateSize(1))),
		make_named<WorldVector>(targetPos.get() + (templateSize(0) / 2) * normal.get() + Vector3(0, 0, templateSize(1))),
		make_named<WorldVector>(targetPos.get() + (templateSize(0) / 2) * normal.get()),
		make_named<WorldVector>(targetPos.get() - (templateSize(0) / 2) * normal.get())
	};

	cv::Point2f transformedCorners[4];
	std::transform(std::begin(templateCorners_w), std::end(templateCorners_w), checked_iterator(cv::Point2f*, transformedCorners, 4),
	               [&](const WorldVector & v)
	               {
		               const auto imagePoint = cam.worldToImage(v);
		               return cv::Point2f(static_cast<float>(imagePoint(0)), static_cast<float>(imagePoint(1)));
	               }
	);

	const auto projectiveTransform = cv::getPerspectiveTransform(templateImageCorners, transformedCorners);

	cv::warpPerspective(t.getImg(), img, projectiveTransform, img.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

	for(const auto & face : walls)
	{
		bool occlusion = false;

		for(const auto & corner : templateCorners_w)
		{
			if(face.doesIntersectLine(corner.get(), cam.getOrigin().get()))
			{
				occlusion = true;
				break;
			}
		}

		if(!occlusion)
		{
			continue;
		}

		std::vector<cv::Point> polygonImagePoints;
		for(const auto & v : face)
		{
			const auto v_i = cam.worldToImage(make_named<WorldVector>(v)).get();
			polygonImagePoints.emplace_back(v_i(0), v_i(1));
		}

		std::vector<std::vector<cv::Point>> pointsWrapper{ polygonImagePoints };

		cv::fillPoly(img, cv::InputArrayOfArrays(pointsWrapper), 0);
		
	}

	/*cv::drawMarker(img, cv::Point(targetPos_img(0), targetPos_img(1)), 128, cv::MARKER_STAR);
	for (const auto & p : transformedCorners)
	{
		cv::drawMarker(img, cv::Point(p.x, p.y), 128);
	}*/
}


void TemplateTransformer::save(const std::string & filename) const
{
	using namespace boost::filesystem;

	const path temp = makeTempDir(path(filename).parent_path());

	{
		std::ofstream ofMeta((temp / "dict_meta.txt").string());
		ofMeta << m_targetImageSize.get()(0) << " " << m_targetImageSize.get()(1) << std::endl;
		ofMeta << m_template.getInfo().targetSize.get()(0) << " " << m_template.getInfo().targetSize.get()(1) << std::endl;
		ofMeta << getNumEntries() << " " << getNumPixel() << std::endl;
	}

	std::vector<path> baseNames;

	for (size_t i = 0; i < getNumCameras(); ++i)
	{
		std::stringstream ss;
		ss << temp.string() << "/cam_";
		ss.width(2);
		ss.fill('0');
		ss << i;

		baseNames.push_back(ss.str());
		create_directory(baseNames.back());
	}

	for (size_t iLocation = 0; iLocation < m_entries.size(); ++iLocation)
	{
		const auto mats = getEntryAsImages(iLocation);

		for (size_t iMat = 0; iMat < mats.size(); ++iMat)
		{
			cv::Mat m;
			cv::threshold(mats[iMat], m, 0, 255, cv::THRESH_BINARY);

			std::stringstream ss;
			ss << baseNames[iMat].string() << "/";
			ss.width(5);
			ss.fill('0');
			ss << iLocation << ".png";
			cv::imwrite(ss.str().c_str(), m);
		}
	}

	zipDir(temp, filename);

	remove_all(temp);
}


void TemplateTransformer::saveDebugImages(const std::string & path, int i) const
{
	std::vector<cv::Mat> images;

	for(int idx_img = 0; idx_img < getNumCameras(); ++idx_img)
	{
		images.push_back(cv::Mat(getTargetImageSizeCv(), CV_32F));
		images.back().setTo(0);
	}

	for(int idx_entry = 0; idx_entry < getNumEntries(); ++idx_entry)
	{
		const auto words = getEntryAsImages(idx_entry);

		for(const auto & [j, w] : enumerate(words))
		{
			cv::add(images[j], w, images[j], cv::noArray(), CV_32F);
		}
	}

	for(int idx_img = 0; idx_img < getNumCameras(); ++idx_img)
	{
		std::stringstream ss;
		ss << path << "/img_" << i << "_" << idx_img << ".png";
		cv::Mat img;
		cv::normalize(images[idx_img], img, 255, 0, cv::NORM_MINMAX, CV_8U);
		cv::imwrite(ss.str().c_str(), img);
	}
}


TemplateTransformer TemplateTransformer::load(const std::string & filename)
{
	using namespace boost::filesystem;

	const path temp = makeTempDir(path(filename).parent_path());

	unzipDir(filename, temp);

	TemplateTransformer dict;

	const path dictMetaPath = temp / "dict_meta.txt";

	size_t numEntries, numPixel;
	{
		std::ifstream ifMeta(dictMetaPath.string());
		ifMeta >> dict.m_targetImageSize.get()(0) >> dict.m_targetImageSize.get()(1);
		throw std::runtime_error("Not implemented");
		//ifMeta >> dict.m_template.getInfo().targetSize.get()(0) >> dict.m_template.getInfo().targetSize.get()(1);
		ifMeta >> numEntries >> numPixel;
	}

	dict.m_entries.resize(numEntries);
	for (int i = 0; i < numEntries; ++i)
	{
		dict.m_entries[i] = std::vector<unsigned char>(numPixel, 0);
	}

	int iCam = 0;

	for (directory_entry & entry : directory_iterator(temp))
	{
		if (!is_directory(entry.path()))
		{
			continue;
		}

		int iImg = 0;

		for (directory_entry & imagefile : directory_iterator(entry.path()))
		{
			if (!is_regular_file(imagefile))
			{
				throw std::runtime_error("Archive corrupted!");
			}

			cv::Mat img = cv::imread(imagefile.path().string(), cv::IMREAD_GRAYSCALE);
			cv::Mat m;
			cv::threshold(img, m, 0, 1, cv::THRESH_BINARY); 

			memcpy(dict.m_entries[iImg].data() + iCam * dict.getNumPixelPerCamera() * sizeof(unsigned char), 
				m.data, dict.getNumPixelPerCamera() * sizeof(unsigned char));

			++iImg;
		}

		++iCam;
	}

	remove_all(temp);

	return dict;
}


bool operator==(const TemplateTransformer & lhs, const TemplateTransformer & rhs)
{
	return
		lhs.m_targetImageSize == rhs.m_targetImageSize &&
		lhs.m_entries == rhs.m_entries;
}


#undef checked_iterator