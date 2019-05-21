#include "Dictionary.h"

#include <utility>

#include "listFilesInDirectory.h"

#include "serialization_helper.h"
#include <enumerate.h>


Dictionary::Dictionary() : m_grid(Roi3DF{ }, { })
{
	// empty; Only to be used by load
}

Dictionary::Dictionary(const Environment & environment, const Template & objectTemplate,
	const GridPoints::Parameters & gridParameters, const ImageSize & targetImageSize, bool debug_skipTransform) :
	Dictionary(environment, objectTemplate, GridPoints{ environment, gridParameters }, targetImageSize, debug_skipTransform)
{
	// empty
}


Dictionary::Dictionary(const Environment & environment, const Template & objectTemplate, 
	const GridPoints & points,	const ImageSize & targetImageSize, bool debug_skipTransform) :
	Dictionary(environment, std::vector<Template>{objectTemplate}, points, targetImageSize, debug_skipTransform)
{
	// empty
}



Dictionary::Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
                       const GridPoints::Parameters & gridParameters, const ImageSize & targetImageSize, bool debug_skipTransform) :
	Dictionary(environment, objectTemplates, GridPoints{ environment, gridParameters }, targetImageSize, debug_skipTransform)
{
	// empty
}


Dictionary::Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
                       GridPoints points, const ImageSize & targetImageSize, bool debug_skipTransform) :
	m_grid(std::move(points))
{	
	std::cout << "Building Dictionary\n";
	for (const auto & [i, t] : enumerate(objectTemplates))
	{
		std::cout << "\tTemplate " << i + 1 << " of " << objectTemplates.size() << std::endl;
		m_dictionaries.emplace_back(environment.getCameras(), t, m_grid.getPoints(), targetImageSize, environment.getStaticMesh(), debug_skipTransform);
	}
	std::cout << "\n\n" << std::endl;
}


const std::vector<Dictionary::Word> & Dictionary::getEntries_single() const
{
	if (m_dictionaries.size() > 1)
	{
		std::cout << "Warning: getEntries_single()." << std::endl;
		std::cout << "         Using a dictionary containing multiple templates for single template optimization." << std::endl;
	}

	return getEntries(0);
}


const std::vector<Dictionary::Word> & Dictionary::getEntries(size_t idxTemplate) const
{
	return m_dictionaries[idxTemplate].getEntries();
}


const Dictionary::Word & Dictionary::getWord_single(size_t idxPosition) const
{
	if (m_dictionaries.size() > 1)
	{
		std::cout << "Warning: getWord_single()." << std::endl;
		std::cout << "         Using a dictionary containing multiple templates for single template optimization." << std::endl;
	}

	return getWord({ 0, idxPosition });
}


const Dictionary::Word & Dictionary::getWord(Index i) const
{
	return getEntries(i.idxTemplate)[i.idxPosition];
}


std::vector<cv::Mat> Dictionary::getWordAsImages(Index i) const
{
	return m_dictionaries[i.idxTemplate].getEntryAsImages(i.idxPosition);
}


SolutionActor Dictionary::getSolution(int optimizationResult) const
{
	const Index index = { getTemplateIndex(optimizationResult), getPositionIndex(optimizationResult) };
	const auto position = m_grid.getPoint(index.idxPosition);
	const auto info = m_dictionaries[index.idxTemplate].getTemplateInfo();

	return SolutionActor{ position, info, index.idxPosition, index.idxTemplate };
}


std::vector<cv::Mat> Dictionary::getReconstructedFrames(const std::vector<int> & optimResult) const
{
	const auto numCameras = getNumCameras();
	const auto imageSize = getTargetImageSizeCv();

	std::vector<cv::Mat> reconstructedFrames;

	for (size_t i = 0; i < numCameras; ++i)
	{
		reconstructedFrames.emplace_back(imageSize, CV_8U, cv::Scalar::all(0));
	}

	for (auto linearIndex : optimResult)
	{
		const auto images = getWordAsImages(getIndex(linearIndex));

		for (size_t i = 0; i < numCameras; ++i)
		{
			reconstructedFrames[i] += images[i] * 255;
		}
	}

	return reconstructedFrames;
}


void Dictionary::save(const std::string & filename) const
{
	using namespace boost::filesystem;

	const path temp = makeTempDir(path(filename).parent_path());

	std::ofstream((temp / "grid.txt").string()) << m_grid;

	for (int i = 0; i < getNumTemplates(); ++i)
	{
		std::stringstream ss;
		ss << temp.string() << "/template_";
		ss.width(3);
		ss.fill('0');
		ss << i;

		m_dictionaries[i].save(ss.str());
	}

	zipDir(temp, filename);

	remove_all(temp);
}


void Dictionary::saveDebugImages(const std::string & path) const
{
	for(size_t idx_template = 0; idx_template < m_dictionaries.size(); ++idx_template)
	{
		m_dictionaries[idx_template].saveDebugImages(path, static_cast<int>(idx_template));
	}
}


Dictionary Dictionary::load(const std::string & filename)
{
	using namespace boost::filesystem;

	const path temp = makeTempDir(path(filename).parent_path());

	unzipDir(filename, temp);

	Dictionary dict;

	std::ifstream((temp / "grid.txt").string()) >> dict.m_grid;

	const auto zipFiles = filterFileList(listFilesInDirectory(temp.string()), "zip");

	dict.m_dictionaries.reserve(zipFiles.size());

	for (const auto & f : zipFiles)
	{
		dict.m_dictionaries.emplace_back(TemplateTransformer::load(f));
	}

	remove_all(temp);

	return dict;
}


bool Dictionary::operator==(const Dictionary & rhs) const
{
	if (!(m_grid == rhs.m_grid))
	{
		return false;
	}

	if (m_dictionaries.size() != rhs.m_dictionaries.size())
	{
		return false;
	}

	for (size_t i = 0; i < m_dictionaries.size(); ++i)
	{
		if (!(m_dictionaries[i] == rhs.m_dictionaries[i]))
		{
			return false;
		}
	}

	return true;
}
