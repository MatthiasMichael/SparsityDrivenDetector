#include "VideoInput_Mpeg.h"

#include "ApplicationTimer.h"


struct VideoInput_Mpeg::StringConstants
{
	inline static const auto TP_CTOR = "VideoInput_Mpeg::Ctor";

	inline static const auto TP_ADVANCE_READ = "VideoInput_Mpeg::AdvanceFrame::VideoRead";
	inline static const auto TP_ADVANCE_PREPROCESS = "VideoInput_Mpeg::AdvanceFrame::Preprocess";
};

VideoInput_Mpeg::VideoInput_Mpeg(
	const SceneInfo & sceneInfo,
	const ImageSize & displaySize, const ImageSize & processSize,
	PreprocessingFunction f, Framenumber startOffset /*= make_named<Framenumber>(0)*/
) :
	VideoInput(f, startOffset),
	m_videoReaders()
{
	AT_START(StringConstants::TP_CTOR);

	auto camInfo = sceneInfo.cameraInfo;
	std::sort(camInfo.begin(), camInfo.end(), [](const CameraInfo & a, const CameraInfo & b) { return a.id < b.id; });

	for (const auto & info : camInfo)
	{
		const std::string videoFilename = sceneInfo.filenameToCameraId.at(info.id);

		//std::cout << "Camera " << cameraId << ": " << videoFilename << std::endl;

		m_videoReaders.emplace_back(videoFilename);

		auto & currentReader = m_videoReaders.back();
		if (!currentReader.videoOpened())
		{
			AT_STOP(StringConstants::TP_CTOR);
			throw std::runtime_error("Unable to open video file.");
		}

		const ImageConverter & c_defaultSize = currentReader.registerDefaultTargetFormat(s_pixelFormat);
		const ImageConverter & c_displaySize = currentReader.registerTargetFormat({ displaySize(0), displaySize(1), s_pixelFormat });
		const ImageConverter & c_processSize = currentReader.registerTargetFormat({ processSize(0), processSize(1), s_pixelFormat });

		const auto defaultImageFormat = currentReader.getImageFormat();

		m_images.push_back(cv::Mat(defaultImageFormat.height, defaultImageFormat.width, CV_8U, c_defaultSize.getBuffer()));
		m_imagesScaledForDisplay.push_back(cv::Mat(displaySize(1), displaySize(0), CV_8U, c_displaySize.getBuffer()));
		m_imagesScaledForProcessing.push_back(cv::Mat(processSize(1), processSize(0), CV_8U, c_processSize.getBuffer()));
		m_preprocessedImages.push_back(cv::Mat());
	}

	if (m_startOffset != make_named<Framenumber>(0))
	{
		VideoInput_Mpeg::jumpToFrame(m_startOffset);
	}

	AT_STOP(StringConstants::TP_CTOR);
}


bool VideoInput_Mpeg::advanceFrame(bool wrap /*= false*/)
{
	AT_START(StringConstants::TP_ADVANCE_READ);
	for (auto & reader : m_videoReaders)
	{
		if (!advanceReader(reader, wrap))
		{
			AT_STOP(StringConstants::TP_ADVANCE_READ);
			return false;
		}
	}
	AT_STOP(StringConstants::TP_ADVANCE_READ);

	AT_START(StringConstants::TP_ADVANCE_PREPROCESS);

	assert(m_imagesScaledForProcessing.size() == m_preprocessedImages.size());

	for (size_t i = 0; i < m_imagesScaledForProcessing.size(); ++i)
	{
		m_preprocessingFunction(m_imagesScaledForProcessing[i], m_preprocessedImages[i]);
	}

	AT_STOP(StringConstants::TP_ADVANCE_PREPROCESS);

	return true;
}


bool VideoInput_Mpeg::jumpToFrame(Framenumber frameIdx)
{
	for (auto & reader : m_videoReaders)
	{
		if (reader.jumpToFrame(frameIdx.get()) == 0)
		{
			return false;
		}
	}

	assert(m_imagesScaledForProcessing.size() == m_preprocessedImages.size());

	for (size_t i = 0; i < m_imagesScaledForProcessing.size(); ++i)
	{
		m_preprocessingFunction(m_imagesScaledForProcessing[i], m_preprocessedImages[i]);
	}

	return true;
}


Framenumber VideoInput_Mpeg::getCurrentFramenumber() const
{
	assert
	(
		std::all_of(m_videoReaders.begin(), m_videoReaders.end(),
			[&](const MpegReader & r)
			{
				return r.getFrameNumber() == m_videoReaders.front().getFrameNumber();
			}
		)
	);

	return make_named<Framenumber>(m_videoReaders.front().getFrameNumber());
}


Framenumber VideoInput_Mpeg::getMaximumFramenumber() const
{
	assert
	(
		std::all_of(m_videoReaders.begin(), m_videoReaders.end(),
			[&](const MpegReader & r)
			{
				return r.getMaxFrameNumber() == m_videoReaders.front().getMaxFrameNumber();
			}
		)
	);

	return make_named<Framenumber>(m_videoReaders.front().getMaxFrameNumber());
}


bool VideoInput_Mpeg::advanceReader(MpegReader & reader, bool wrap)
{
	if (reader.advanceFrame())
	{
		return true;
	}

	if (!wrap)
	{
		return false;
	}

	if (!jumpToStart())
	{
		return false;
	}

	return reader.advanceFrame();
}
