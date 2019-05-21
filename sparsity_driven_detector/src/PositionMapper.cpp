#include "PositionMapper.h"

Dictionary::Dictionary() : m_grid(Roi3DF{}, {})
{
	// empty; Only to be used by load
}


Dictionary::Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
	const GridPoints::Parameters & gridParameters, const ImageSize & targetImageSize) :
	Dictionary(environment, objectTemplates, GridPoints{ environment, gridParameters }, targetImageSize)
{
	// empty
}


Dictionary::Dictionary(const Environment & environment, const std::vector<Template> & objectTemplates,
	const GridPoints & points, const ImageSize & targetImageSize) : 
	m_grid(points), m_dictionaries()
{
	for(const auto & t : objectTemplates)
	{
		m_dictionaries.emplace_back(environment.getCameras(), t, m_grid.getPoints(), targetImageSize);
	}
}


const std::vector<Dictionary::Word> & Dictionary::getEntries(size_t idxTemplate) const
{
	return m_dictionaries[idxTemplate].getEntries();
}

const Dictionary::Word & Dictionary::getWord(size_t idxTemplate, size_t idxPosition) const
{
	return getEntries(idxTemplate)[idxPosition];
}

std::vector<cv::Mat> Dictionary::getWordAsImages(size_t idxTemplate, size_t idxPosition) const
{
	return m_dictionaries[idxTemplate].getEntryAsImages(idxPosition);
}

void Dictionary::save(const std::string & filename) const
{
	throw std::runtime_error("Not implemented");
}

std::unique_ptr<Dictionary> Dictionary::load(const std::string & filename)
{
	throw std::runtime_error("Not implemented");
}

bool Dictionary::operator==(const Dictionary & rhs) const
{
	if(!(m_grid == rhs.m_grid))
	{
		return false;
	}

	if(m_dictionaries.size() != rhs.m_dictionaries.size())
	{
		return false;
	}

	for(size_t i = 0; i < m_dictionaries.size(); ++i)
	{
		if(!(m_dictionaries[i] == rhs.m_dictionaries[i]))
		{
			return false;
		}
	}

	return true;
}


