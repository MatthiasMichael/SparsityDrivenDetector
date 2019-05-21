#include "TimingEntry.h"

#include <stdexcept>
#include <cassert>

#include "ApplicationTimer.h"


TimingEntry::TimingEntry() : 
	m_framenumber(AT_GET_FRAME()),
	m_start(std::chrono::high_resolution_clock::now()),
	m_stop()
{
	// empty
}


bool TimingEntry::finished() const
{
	return m_stop != TimePoint{ };
}


void TimingEntry::finish()
{
	if (finished())
	{
		throw std::runtime_error("Can't finish already finished timing.");
	}

	if(AT_GET_FRAME() != m_framenumber)
	{
		throw std::runtime_error("Timing over several frames is not supported.");
	}

	m_stop = std::chrono::high_resolution_clock::now();
}


double TimingEntry::duration() const
{
	if (!finished())
	{
		throw std::runtime_error("Can't get duration of non finished timing.");
	}
	assert(m_stop >= m_start);
	return std::chrono::duration<double, std::milli>(m_stop - m_start).count();
}