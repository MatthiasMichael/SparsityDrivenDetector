#include "ImageSequenceReader.h"

#include "listFilesInDirectory.h"
#include <opencv2/imgcodecs.hpp>


ImageSequenceReader::ImageSequenceReader() :
	m_currentIndex(0),
	m_filenames(),
	m_frame()
{
	// empty
}


ImageSequenceReader::ImageSequenceReader(const std::string & foldername) : ImageSequenceReader()
{
	openVideo(foldername);
}


void ImageSequenceReader::openVideo(const std::string & foldername)
{
	m_currentIndex = -1;

	m_filenames = filterFileList(listFilesInDirectory(foldername), ".pgm");

	if (m_filenames.empty())
	{
		m_frame = cv::Mat{ };
		return;
	}

	m_frame = cv::imread(m_filenames.front(), cv::IMREAD_GRAYSCALE);
}


size_t ImageSequenceReader::getFrameNumber() const
{
	return m_currentIndex < 0 ? 0 : static_cast<size_t>(m_currentIndex);
}


int ImageSequenceReader::getFrameWidth() const
{
	return m_frame.cols;
}


int ImageSequenceReader::getFrameHeight() const
{
	return m_frame.rows;
}


bool ImageSequenceReader::advanceFrame()
{
	if (!videoOpened())
	{
		return false;
	}

	if (m_currentIndex != -1 && m_currentIndex >= m_filenames.size() - 1)
	{
		return false;
	}

	m_frame = cv::imread(m_filenames[++m_currentIndex], cv::IMREAD_GRAYSCALE);

	return true;
}


bool ImageSequenceReader::jumpToFrame(int frameIdx)
{
	if (!videoOpened())
	{
		return false;
	}

	m_currentIndex = frameIdx - 1;
	return advanceFrame();
}
