#pragma once

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

class ImageSequenceReader
{
public:
	ImageSequenceReader();
	ImageSequenceReader(const std::string & foldername);

	ImageSequenceReader(const ImageSequenceReader &) = delete;
	ImageSequenceReader(ImageSequenceReader &&) = delete;

	ImageSequenceReader & operator=(const ImageSequenceReader &) = delete;
	ImageSequenceReader & operator=(ImageSequenceReader &&) = delete;

	void openVideo(const std::string & foldername);
	bool videoOpened() const { return !m_filenames.empty(); }

	size_t getFrameNumber() const; //< Index of the last frame that has been read
	size_t getMaxFrameNumber() const { return m_filenames.size(); }

	int getFrameWidth() const;
	int getFrameHeight() const;

	bool advanceFrame();
	bool jumpToFrame(int frameIdx); //< Frame will already been read

	const cv::Mat & getFrame() const { return m_frame; }

private:
	int m_currentIndex;

	std::vector<std::string> m_filenames;

	cv::Mat m_frame;
};