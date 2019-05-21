#pragma once

#include <vector>

#include "opencv2/imgproc.hpp"

#include "GeometryUtils.h"

#include "Scene.h"


class VideoInput
{
public:
	using PreprocessingFunction = std::function<void(const cv::Mat &, cv::Mat &)>;

	using ImageSize = NamedVectorTypes<int, 2>::Size;

	VideoInput(PreprocessingFunction f, Framenumber startOffset = make_named<Framenumber>(0));

	virtual ~VideoInput() = default;

	virtual bool advanceFrame(bool wrap = false) = 0;
	virtual bool jumpToFrame(Framenumber frameIdx) = 0;
	
	virtual Framenumber getCurrentFramenumber() const = 0;
	virtual Framenumber getMaximumFramenumber() const = 0;

	bool jumpToStart();
	
	const cv::Mat & getImage(size_t idx) const { return m_images[idx]; }
	const cv::Mat & getImageScaledForDisplay(size_t idx) const { return m_imagesScaledForDisplay[idx]; }
	const cv::Mat & getImageScaledForProcessing(size_t idx) const { return m_imagesScaledForProcessing[idx]; }
	const cv::Mat & getPreprocessedImage(size_t idx) const { return m_preprocessedImages[idx]; }

	const std::vector<cv::Mat> & getImages() const { return m_images; }
	const std::vector<cv::Mat> & getImagesScaledForDisplay() const { return m_imagesScaledForDisplay; }
	const std::vector<cv::Mat> & getImagesScaledForProcessing() const { return m_imagesScaledForProcessing; }
	const std::vector<cv::Mat> & getPreprocessedImages() const { return m_preprocessedImages; }

	static const PreprocessingFunction s_defaultPreprocessingFunction;
protected:

	PreprocessingFunction m_preprocessingFunction;

	Framenumber m_startOffset;

	std::vector<cv::Mat> m_images; // as read from wherever

	std::vector<cv::Mat> m_imagesScaledForDisplay; // as read and scaled
	std::vector<cv::Mat> m_imagesScaledForProcessing; // as read and scaled
	
	std::vector<cv::Mat> m_preprocessedImages; // scaled and then run through preprocessing
};
