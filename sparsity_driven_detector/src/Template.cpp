#include "Template.h"

Template::Template(const cv::Mat & img, const Info & info) :
	m_img(cv::Mat(img.rows, img.cols, CV_8U)), m_info(info)
{
	img.convertTo(m_img, CV_8U);
	// TODO: Maybe already convert the image to binary format
}
