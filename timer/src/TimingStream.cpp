#include "TimingStream.h"

#include <algorithm>


TimingStream::TimingStream(const std::string & name) : m_name(name)
{
	// empty
}


void TimingStream::start()
{
	if (!m_timings.empty() && !m_timings.back().finished())
	{
		throw std::runtime_error("Can't start new timing for this key while the old one is still running.");
	}

	m_timings.emplace_back();
}


void TimingStream::stop()
{
	auto & t = m_timings.back();

	if (t.finished())
	{
		throw std::runtime_error("Last timing for this key is already finished.");
	}

	t.finish();
}


double TimingStream::stopWithTime()
{
	stop();
	return m_timings.back().duration();
}


void TimingStream::print(std::ostream & os) const
{
	double total = 0;
	double min = std::numeric_limits<double>::max();
	double max = std::numeric_limits<double>::lowest();

	for (const auto & t : m_timings)
	{
		total += t.duration();
		min = std::min(min, t.duration());
		max = std::max(max, t.duration());
	}

	os << m_name << ":\n";
	os << "\tOcc:   " << m_timings.size();
	os << "\tTotal: " << total << std::endl;
	os << "\tMean:  " << total / m_timings.size() << std::endl;
	os << "\tMin:   " << min << std::endl;
	os << "\tMax:   " << max << std::endl;
}


void TimingStream::printWithoutUnfinishedTimings(std::ostream & os) const
{
	double total = 0;
	double min = std::numeric_limits<double>::max();
	double max = std::numeric_limits<double>::lowest();

	for (const auto & t : m_timings)
	{
		if(!t.finished())
		{
			continue;
		}

		total += t.duration();
		min = std::min(min, t.duration());
		max = std::max(max, t.duration());
	}

	os << m_name << ":\n";
	os << "\tOcc:   " << m_timings.size();
	os << "\tTotal: " << total << std::endl;
	os << "\tMean:  " << total / m_timings.size() << std::endl;
	os << "\tMin:   " << min << std::endl;
	os << "\tMax:   " << max << std::endl;
}
