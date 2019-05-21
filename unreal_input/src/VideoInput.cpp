#include "VideoInput.h"


const VideoInput::PreprocessingFunction VideoInput::s_defaultPreprocessingFunction =
[](const cv::Mat & input, cv::Mat & output)
{
	input.copyTo(output);
};


VideoInput::VideoInput(PreprocessingFunction f, Framenumber startOffset /*= make_named<Framenumber>(0)*/) :
	m_preprocessingFunction(f),
	m_startOffset(startOffset),
	m_images(),
	m_imagesScaledForDisplay(),
	m_imagesScaledForProcessing(),
	m_preprocessedImages()
{
	// empty
}


bool VideoInput::jumpToStart()
{
	return jumpToFrame(m_startOffset);
}
