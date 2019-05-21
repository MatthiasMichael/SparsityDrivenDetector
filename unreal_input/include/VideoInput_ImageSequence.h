#pragma once

#include "VideoInput.h"

#include "ImageSequenceReader.h"

class VideoInput_ImageSequence : public VideoInput
{
public:
	struct StringConstants;

	VideoInput_ImageSequence(const std::string & folder,
		const ImageSize & displaySize, const ImageSize & processSize, 
		const PreprocessingFunction & f, const Framenumber & startOffset);

	VideoInput_ImageSequence(const VideoInput_ImageSequence &) = delete;

	VideoInput_ImageSequence & operator=(const VideoInput_ImageSequence &) = delete;

	bool advanceFrame(bool wrap) override;
	bool jumpToFrame(Framenumber frameIdx) override;

	Framenumber getCurrentFramenumber() const override;
	Framenumber getMaximumFramenumber() const override;

private:
	bool advanceReader(ImageSequenceReader & reader, bool wrap);

private:
	std::list<ImageSequenceReader> m_reader;
};