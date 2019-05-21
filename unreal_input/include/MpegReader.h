#pragma once

#include <string>

#include <memory>
#include <unordered_map>


extern "C" 
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}


struct ImageFormat
{
	ImageFormat();
	ImageFormat(int width, int height, AVPixelFormat pixelFormat);

	bool operator==(const ImageFormat & other) const;

	int width;
	int height;
	AVPixelFormat pixelFormat;
};


class ImageConverter
{
public:
	explicit ImageConverter(ImageFormat targetFormat);
	ImageConverter(ImageFormat targetFormat, ImageFormat srcFormat);

	~ImageConverter();

	ImageConverter(const ImageConverter &) = delete;
	ImageConverter(ImageConverter &&) = delete;

	ImageConverter & operator=(const ImageConverter &) = delete;
	ImageConverter & operator=(ImageConverter &&) = delete;

	void resetConversionSource(ImageFormat srcFormat = ImageFormat());
	void convert(AVFrame * src);

	bool canConvert() const { return mp_conversionContext != nullptr; }

	unsigned char * getBuffer() const { return mp_bufferTargetFormat; }

private:
	static const int s_linesizeAlignment;

	ImageFormat m_targetFormat;
	ImageFormat m_srcFormat;

	SwsContext * mp_conversionContext;

	AVFrame * mp_frameTargetFormat;

	unsigned char * mp_bufferTargetFormat;
};


class MpegReader
{
public:
	MpegReader();
	MpegReader(const std::string & filename);

	~MpegReader();

	MpegReader(const MpegReader &) = delete;
	MpegReader(MpegReader &&) = delete;

	MpegReader & operator=(const MpegReader &) = delete;
	MpegReader & operator=(MpegReader &&) = delete;

	void openVideo(const std::string & filename);
	bool videoOpened() const { return mp_formatContext != nullptr; }

	int64_t getFrameNumber() const; //< Index of the last frame that has been read
	int64_t getMaxFrameNumber() const;
	
	int getFrameWidth() const;
	int getFrameHeight() const;

	ImageConverter & registerDefaultTargetFormat(AVPixelFormat pixelFormat);
	ImageConverter & registerTargetFormat(ImageFormat format);

	bool advanceFrame();
	bool jumpToFrame(int frameIdx); //< Frame will already been read

	ImageFormat getImageFormat() const;
	
private:
	int findFirstVideoStreamIndex() const;
	AVStream * getFirstVideoStream() const;
	void closeCurrentVideo();

private:
	AVFormatContext * mp_formatContext;
	AVCodecContext * mp_codecContext;
	
	AVFrame * mp_frameAsRead;

	std::list<ImageConverter> m_converter;

	int m_videoStreamIdx;
};