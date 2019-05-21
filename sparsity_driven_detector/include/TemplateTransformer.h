#pragma once

#include <vector>

#include "Environment.h"
#include "Template.h"


class TemplateTransformer
{
public:
	using WorldVector = WorldVector_T<float>;
	using ImageSize = IdentifiableCamera::Size;

	using Word = std::vector<unsigned char>;

	TemplateTransformer(const CameraSet & cameras, const Template & objectTemplate,
		const std::vector<WorldVector> & points, const ImageSize & targetImageSize,
		const Mesh & walls, bool debug_skipTransform = false);

	std::vector<cv::Mat> getEntryAsImages(size_t idx) const;

	void save(const std::string & filename) const;

	const auto & getEntries() const { return m_entries; }

	int getNumPixel() const { return static_cast<int>(m_entries.front().size()); }
	int getNumEntries() const { return static_cast<int>(m_entries.size()); }
	int getNumCameras() const { return getNumPixel() / getNumPixelPerCamera() ; }
	int getNumPixelPerCamera() const { return m_targetImageSize.get().prod(); }

	Template::Info getTemplateInfo() const { return m_template.getInfo(); }
	ImageSize getTargetImageSize() const { return m_targetImageSize; }
	cv::Size getTargetImageSizeCv() const { return cv::Size(m_targetImageSize(0), m_targetImageSize(1)); }

	void saveDebugImages(const std::string & path, int i) const;
	static TemplateTransformer load(const std::string & filename);

	friend bool operator==(const TemplateTransformer & lhs, const TemplateTransformer & rhs);

private:
	TemplateTransformer(); // Only to be used by load

	void transformTemplate(const Template & t, const IdentifiableCamera & cam, const WorldVector & targetPos, cv::Mat & img, const Mesh & walls);

private:
	ImageSize m_targetImageSize;
	Template m_template;

	std::vector<Word> m_entries;
};
