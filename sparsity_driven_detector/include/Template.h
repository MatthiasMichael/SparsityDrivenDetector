#pragma once

#include "opencv2/imgproc.hpp"

#include "GeometryUtils.h"
#include "WorldCoordinateSystem_SDD.h"

class Template
{
public:
	using Size = NamedVectorTypes<ScalarType, 2>::Size; // Width - Height

	struct Info
	{
		int objectClass; //< User defined class id to say which templates represent the same class of objects
		Size targetSize; //< Size in World Coordinates (Widht/Diameter - Height). Used for dictionary creation
		Size maxSize; //< Maximum size in world coordinates. Used to assign voxel. Should maybe not be stored here
	};

	Template() = default;
	Template(const cv::Mat & img, const Info & info);

	const cv::Mat & getImg() const { return m_img; }
	Size getImgSize() const { return make_named<Size>(m_img.cols, m_img.rows); }

	const Info & getInfo() const { return m_info; }

private:
	cv::Mat m_img;
	Info m_info;
};