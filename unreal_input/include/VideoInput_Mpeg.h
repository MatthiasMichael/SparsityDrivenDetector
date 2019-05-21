#pragma once

#include "VideoInput.h"

#include "MpegReader.h"

class VideoInput_Mpeg : public VideoInput
{
public:
	struct StringConstants;

	VideoInput_Mpeg(
		const SceneInfo & sceneInfo,
		const ImageSize & displaySize, const ImageSize & processingSize,
		PreprocessingFunction f, Framenumber startOffset = make_named<Framenumber>(0)
	);

	bool advanceFrame(bool wrap) override;
	bool jumpToFrame(Framenumber frameIdx) override;
	
	Framenumber getCurrentFramenumber() const override;
	Framenumber getMaximumFramenumber() const override;

private:
	bool advanceReader(MpegReader & reader, bool wrap);

private:
	static const AVPixelFormat s_pixelFormat = AV_PIX_FMT_GRAY8;

	std::list<MpegReader> m_videoReaders;
};