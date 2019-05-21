#pragma once

#include "GridPoints.h"
#include "Environment.h"
#include "TemplateTransformer.h"
#include "Solution.h"


class Dictionary
{
public:
	using ImageSize = TemplateTransformer::ImageSize;
	using Word = TemplateTransformer::Word;

	struct Index
	{
		size_t idxTemplate;
		size_t idxPosition;
	};

	Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
		const GridPoints::Parameters & gridParameters, const ImageSize & targetImageSize, bool debug_skipTransform = false);

	Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
	           GridPoints points, const ImageSize & targetImageSize, bool debug_skipTransform = false);

	Dictionary(const Environment & environment, const Template & objectTemplate,
		const GridPoints::Parameters & gridParameters, const ImageSize & targetImageSize, bool debug_skipTransform = false);

	Dictionary(const Environment & environment, const Template & objectTemplate,
	           const GridPoints & points, const ImageSize & targetImageSize, bool debug_skipTransform = false);

	const std::vector<Word> & getEntries_single() const;
	const std::vector<Word> & getEntries(size_t idxTemplate) const;

	const Word & getWord_single(size_t idxPosition) const;
	const Word & getWord(Index i) const;

	std::vector<cv::Mat> getWordAsImages(Index i) const;

	SolutionActor getSolution(int optimizationResult) const;
	std::vector<cv::Mat> getReconstructedFrames(const std::vector<int> & optimResult) const;

	void save(const std::string & filename) const;

	const auto & getGrid() const { return m_grid; }

	int getNumTemplates() const { return static_cast<int>(m_dictionaries.size()); }

	int getNumPixel() const { return firstDict().getNumPixel(); }
	int getNumEntries() const { return firstDict().getNumEntries(); }
	int getNumCameras() const { return firstDict().getNumCameras(); }
	int getNumPixelPerCamera() const { return firstDict().getNumPixelPerCamera(); }

	ImageSize getTargetImageSize() const { return firstDict().getTargetImageSize(); }
	cv::Size getTargetImageSizeCv() const { return firstDict().getTargetImageSizeCv(); }

	Index getIndex(int linearIndex) const { return { getTemplateIndex(linearIndex), getPositionIndex(linearIndex) }; }

	void saveDebugImages(const std::string & path) const;

	static Dictionary load(const std::string & filename);

	bool operator==(const Dictionary & rhs) const;

	friend class Context;

private:
	Dictionary(); // only to be used by load

	const TemplateTransformer & firstDict() const { return m_dictionaries.front(); }

	size_t getPositionIndex(int linearIndex) const { return linearIndex % getNumEntries(); }
	size_t getTemplateIndex(int linearIndex) const { return linearIndex / getNumEntries(); }

protected:
	GridPoints m_grid;

	std::vector<TemplateTransformer> m_dictionaries;
};