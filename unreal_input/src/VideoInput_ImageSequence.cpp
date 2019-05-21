#include "VideoInput_ImageSequence.h"

#include "boost/filesystem.hpp"

#include "ApplicationTimer.h"


struct VideoInput_ImageSequence::StringConstants
{
	inline static const auto TP_CTOR = "VideoInput_ImageSequence::Ctor";

	inline static const auto TP_ADVANCE_READ = "VideoInput_ImageSequence::AdvanceFrame::VideoRead";
	inline static const auto TP_ADVANCE_CONVERT = "VideoInput_ImageSequence::AdvanceFrame::Convert";
	inline static const auto TP_ADVANCE_PREPROCESS = "VideoInput_ImageSequence::AdvanceFrame::Preprocess";
};


VideoInput_ImageSequence::VideoInput_ImageSequence(
	const std::string & folder,
	const ImageSize & displaySize, const ImageSize & processSize,
	const PreprocessingFunction & f, const Framenumber & startOffset):
	VideoInput(f, startOffset)
{
	AT_START(StringConstants::TP_CTOR);

	using namespace boost::filesystem;
	for (directory_entry & e : directory_iterator(path(folder)))
	{
		if (is_directory(e))
		{
			m_reader.emplace_back(e.path().string());
			if (!m_reader.back().videoOpened())
			{
				AT_STOP(StringConstants::TP_CTOR);
				throw std::runtime_error("Could not open files from folder: " + e.path().string());
			}

			m_images.push_back(cv::Mat(processSize(1), processSize(0), CV_8U));
			m_imagesScaledForDisplay.push_back(cv::Mat(displaySize(1), displaySize(0), CV_8U));
			m_imagesScaledForProcessing.push_back(cv::Mat(processSize(1), processSize(0), CV_8U));
			m_preprocessedImages.push_back(cv::Mat{ });
		}
	}

	if (m_startOffset != make_named<Framenumber>(0))
	{
		VideoInput_ImageSequence::jumpToFrame(m_startOffset);
	}

	assert
	(
		std::all_of(m_reader.begin(), m_reader.end(),
			[&](const ImageSequenceReader & r)
			{
				return r.getFrameNumber() == m_reader.front().getFrameNumber();
			}
		)
	);

	assert
	(
		std::all_of(m_reader.begin(), m_reader.end(),
			[&](const ImageSequenceReader & r)
			{
				return r.getMaxFrameNumber() == m_reader.front().getMaxFrameNumber();
			}
		)
	);

	AT_STOP(StringConstants::TP_CTOR);
}


bool VideoInput_ImageSequence::advanceFrame(bool wrap)
{
	AT_START(StringConstants::TP_ADVANCE_READ);
	for (auto & reader : m_reader)
	{
		if (!advanceReader(reader, wrap))
		{
			AT_STOP(StringConstants::TP_ADVANCE_READ);
			return false;
		}
	}
	AT_STOP(StringConstants::TP_ADVANCE_READ);

	AT_START(StringConstants::TP_ADVANCE_CONVERT);
	for (auto [i, it] = std::tuple{ size_t{ 0 }, m_reader.begin() }; i < m_reader.size(); ++i, ++it)
	{
		auto & reader = *it;
		if (reader.getFrameHeight() != m_imagesScaledForProcessing[i].rows ||
			reader.getFrameWidth() != m_imagesScaledForProcessing[i].cols)
		{
			AT_STOP(StringConstants::TP_ADVANCE_CONVERT);
			throw std::runtime_error("WE WANT TO PROCESS THEM AS READ!!!!");
		}

		m_images[i] = reader.getFrame();

		cv::resize(reader.getFrame(), m_imagesScaledForDisplay[i], m_imagesScaledForDisplay[i].size());

		m_imagesScaledForProcessing[i] = reader.getFrame();
	}

	AT_STOP(StringConstants::TP_ADVANCE_CONVERT);

	AT_START(StringConstants::TP_ADVANCE_PREPROCESS);

	assert(m_imagesScaledForProcessing.size() == m_preprocessedImages.size());

	for (size_t i = 0; i < m_imagesScaledForProcessing.size(); ++i)
	{
		m_preprocessingFunction(m_imagesScaledForProcessing[i], m_preprocessedImages[i]);
	}

	AT_STOP(StringConstants::TP_ADVANCE_PREPROCESS);

	return true;
}


bool VideoInput_ImageSequence::advanceReader(ImageSequenceReader & reader, bool wrap)
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


bool VideoInput_ImageSequence::jumpToFrame(Framenumber frameIdx)
{
	for (auto & reader : m_reader)
	{
		if (reader.jumpToFrame(frameIdx.get()) == 0)
		{
			return false;
		}
	}

	for (size_t i = 0; i < m_imagesScaledForProcessing.size(); ++i)
	{
		m_preprocessingFunction(m_imagesScaledForProcessing[i], m_preprocessedImages[i]);
	}

	return true;
}


Framenumber VideoInput_ImageSequence::getCurrentFramenumber() const
{
	assert
	(
		std::all_of(m_reader.begin(), m_reader.end(),
			[&](const ImageSequenceReader & r)
			{
				return r.getFrameNumber() == m_reader.front().getFrameNumber();
			}
		)
	);

	return make_named<Framenumber>(m_reader.front().getFrameNumber());
}


Framenumber VideoInput_ImageSequence::getMaximumFramenumber() const
{
	assert
	(
		std::all_of(m_reader.begin(), m_reader.end(),
			[&](const ImageSequenceReader & r)
			{
				return r.getMaxFrameNumber() == m_reader.front().getMaxFrameNumber();
			}
		)
	);

	return make_named<Framenumber>(m_reader.front().getMaxFrameNumber());
}
