#include "MpegReader.h"

#include <cassert>


extern "C"
{
	#include "libavutil/imgutils.h"
}


#define CALL_CHECKED(func, cond, msg) if(!(func cond)){ throw std::runtime_error(msg); }
#define CALL_CHECKED_RETURN(var, func, cond, msg) var = func; if(!(var cond)){ throw std::runtime_error(msg); }
#define CALL_CHECKED_INIT(type, var, func, cond, msg) type var = func; if(!(var cond)){ throw std::runtime_error(msg); }

const int ImageConverter::s_linesizeAlignment = 1;


ImageFormat::ImageFormat() : ImageFormat(0, 0, AV_PIX_FMT_NONE)
{
	// empty
}


ImageFormat::ImageFormat(int width, int height, AVPixelFormat pixelFormat) :
	width(width),
	height(height),
	pixelFormat(pixelFormat)
{
	// empty
}


bool ImageFormat::operator==(const ImageFormat & other) const
{
	return width == other.width && height == other.height && pixelFormat == other.pixelFormat;
}


ImageConverter::ImageConverter(ImageFormat targetFormat) :
	m_targetFormat(targetFormat),
	m_srcFormat(),
	mp_conversionContext(nullptr),
	mp_frameTargetFormat(av_frame_alloc()),
	mp_bufferTargetFormat(nullptr)
{
	if (mp_frameTargetFormat == nullptr)
	{
		throw std::runtime_error("Frame BGRA allocation failed.");
	}

	CALL_CHECKED_INIT(const int, numBytes,
		av_image_get_buffer_size(targetFormat.pixelFormat, targetFormat.width, targetFormat.height, s_linesizeAlignment), > 0,
		"Could not estimate buffer size.");

	CALL_CHECKED_RETURN(mp_bufferTargetFormat,
		static_cast<unsigned char *>(av_malloc(numBytes * sizeof(uint8_t))), != nullptr,
		"Buffer allocation failed.");

	CALL_CHECKED(
		av_image_fill_arrays(mp_frameTargetFormat->data, mp_frameTargetFormat->linesize, mp_bufferTargetFormat,
		targetFormat.pixelFormat, targetFormat.width, targetFormat.height, s_linesizeAlignment), >= 0,
		"Could not associate BGRA buffer to frame.");
}


ImageConverter::ImageConverter(ImageFormat targetFormat, ImageFormat srcFormat) :
	ImageConverter(targetFormat)
{
	resetConversionSource(srcFormat);
}


ImageConverter::~ImageConverter()
{
	av_free(mp_bufferTargetFormat);

	// When I do this, avformat_close_input crashes...
	//av_free(mp_frameBGRA);

	sws_freeContext(mp_conversionContext);
}


void ImageConverter::resetConversionSource(ImageFormat srcFormat /*= ImageFormat()*/)
{
	sws_freeContext(mp_conversionContext);

	m_srcFormat = srcFormat;

	if(srcFormat == ImageFormat())
	{
		mp_conversionContext = nullptr;
		return;
	}

	CALL_CHECKED_RETURN(mp_conversionContext, sws_getContext(
		srcFormat.width, srcFormat.height, srcFormat.pixelFormat,
		m_targetFormat.width, m_targetFormat.height, m_targetFormat.pixelFormat,
		SWS_BILINEAR, nullptr, nullptr, nullptr
	), != nullptr, "Could not fetch conversion context.");
}


void ImageConverter::convert(AVFrame * src)
{
	assert(canConvert());

	sws_scale(
		mp_conversionContext,
		static_cast<uint8_t const * const *>(src->data),
		src->linesize,
		0,
		m_srcFormat.height,
		mp_frameTargetFormat->data,
		mp_frameTargetFormat->linesize);
}


MpegReader::MpegReader() :
	mp_formatContext(nullptr),
	mp_codecContext(nullptr),
	mp_frameAsRead(av_frame_alloc()),
	m_converter(),
	m_videoStreamIdx(0)
{
	if (mp_frameAsRead == nullptr)
	{
		throw std::runtime_error("Frame as read allocation failed.");
	}
}


MpegReader::MpegReader(const std::string & filename) : MpegReader()
{
	openVideo(filename);
}


MpegReader::~MpegReader()
{
	closeCurrentVideo();
}


void MpegReader::openVideo(const std::string & filename)
{
	closeCurrentVideo();

	CALL_CHECKED(avformat_open_input(&mp_formatContext, filename.c_str(), nullptr, nullptr), == 0, "Could not open video file.");
	CALL_CHECKED(avformat_find_stream_info(mp_formatContext, nullptr), >= 0, "Could not find stream information");

	//av_dump_format(mp_formatContext, 0, filename.c_str(), 0);

	CALL_CHECKED_RETURN(m_videoStreamIdx, findFirstVideoStreamIndex(), >= 0, "File does not contain any video stream.");

	const AVCodecParameters * const pCodecPar = getFirstVideoStream()->codecpar;

	CALL_CHECKED_INIT(const AVCodec *, pCodec, avcodec_find_decoder(pCodecPar->codec_id), != nullptr, "Unsupported codec.");

	CALL_CHECKED_RETURN(mp_codecContext, avcodec_alloc_context3(pCodec), != nullptr, "Codec context allocation failed.");

	CALL_CHECKED(avcodec_parameters_to_context(mp_codecContext, pCodecPar), >= 0, "Parameters to context conversion failed.");

	CALL_CHECKED(avcodec_open2(mp_codecContext, pCodec, nullptr), == 0, "Could not open codec");

	for(auto & c : m_converter)
	{
		c.resetConversionSource(getImageFormat());
	}
}


int64_t MpegReader::getFrameNumber() const
{
	if (!videoOpened())
	{
		return -1;
	}

	const auto timeBase = getFirstVideoStream()->time_base;
	const auto fps = getFirstVideoStream()->r_frame_rate;

	const auto timestamp = mp_frameAsRead->pts;

	return (timestamp * timeBase.num * fps.num) / (timeBase.den * fps.den);
}

int64_t MpegReader::getMaxFrameNumber() const
{
	return getFirstVideoStream()->nb_frames;
}


int MpegReader::getFrameWidth() const
{
	return videoOpened() ? getFirstVideoStream()->codecpar->width : -1;
}


int MpegReader::getFrameHeight() const
{
	return videoOpened() ? getFirstVideoStream()->codecpar->height : -1;
}


ImageConverter & MpegReader::registerDefaultTargetFormat(AVPixelFormat pixelFormat)
{
	auto targetImageFormat = getImageFormat();
	targetImageFormat.pixelFormat = pixelFormat;

	return registerTargetFormat(targetImageFormat);
}

ImageConverter & MpegReader::registerTargetFormat(ImageFormat format)
{
	m_converter.emplace_back(format, getImageFormat());
	
	return m_converter.back();
}


bool MpegReader::advanceFrame()
{
	AVPacket packet;

	int result_code = 0;
	do
	{
		do
		{
			if (av_read_frame(mp_formatContext, &packet) < 0)
			{
				return false;
			}

		} while (packet.stream_index != m_videoStreamIdx);

		result_code = avcodec_send_packet(mp_codecContext, &packet);

		if (result_code == 0)
		{
			result_code = avcodec_receive_frame(mp_codecContext, mp_frameAsRead);
		}

	} while (result_code == AVERROR(EAGAIN));

	if(result_code != 0)
	{
		return false;
	}

	for(auto & c : m_converter)
	{
		c.convert(mp_frameAsRead);
	}

	av_packet_unref(&packet);

	return true;
}


bool MpegReader::jumpToFrame(int frameIdx)
{
	const auto timeBase = getFirstVideoStream()->time_base;
	const auto fps = getFirstVideoStream()->r_frame_rate;

	const auto timestamp = (frameIdx * timeBase.den * fps.den) / (timeBase.num * fps.num);
	
	if (av_seek_frame(mp_formatContext, m_videoStreamIdx, timestamp, AVSEEK_FLAG_BACKWARD) < 0)
	{
		return false;
	}

	// av_seek_frame only finds the latest keyframe before the desired timestamp
	// therefore we have to read frames until we have reached frameIdx
	// sometimes however the first (few?) read(s) after issuing seek are unaffected
	// by it so we may have framenumbers > frameidx and can't check for
	// while (getFrameNumber) < frameIdx. This might cause problems if seek fails silently
	
	do 
	{
		if(!advanceFrame())
		{
			return false;
		}

	} while (getFrameNumber() != frameIdx);

	return true;
}


ImageFormat MpegReader::getImageFormat() const
{
	return videoOpened() ? ImageFormat{ getFrameWidth(), getFrameHeight(), mp_codecContext->pix_fmt } : ImageFormat();
}


int MpegReader::findFirstVideoStreamIndex() const
{
	for (unsigned int i = 0; i < mp_formatContext->nb_streams; i++)
	{
		if (mp_formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			return static_cast<int>(i);
		}
	}
	return -1;
}


AVStream * MpegReader::getFirstVideoStream() const
{
	return mp_formatContext->streams[m_videoStreamIdx];
}


void MpegReader::closeCurrentVideo()
{
	for (auto & c : m_converter)
	{
		c.resetConversionSource();
	}

	// When I do this, avformat_close_input crashes...
	//av_free(mp_frameAsRead);

	avcodec_close(mp_codecContext);

	avformat_close_input(&mp_formatContext);
}
