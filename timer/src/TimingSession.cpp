#include "TimingSession.h"

TimingSession::TimingSession(const std::string & name) : 
	m_name(name), 
	m_sessionStart(std::chrono::high_resolution_clock::now())
{
	// empty
}

void TimingSession::start(const std::string & stream)
{
	try
	{
		m_streams.at(stream).start();
	}
	catch(const std::out_of_range &)
	{
		TimingStream s(stream);
		s.start();
		m_streams.emplace(stream, std::move(s));
	}
}

void TimingSession::stop(const std::string & stream)
{
	try
	{
		m_streams.at(stream).stop();
	}
	catch (const std::out_of_range &)
	{
		throw std::runtime_error("No timing for this key exists!");
	}
}

double TimingSession::stopWithTime(const std::string & stream)
{
	try
	{
		return m_streams.at(stream).stopWithTime();
	}
	catch (const std::out_of_range &)
	{
		throw std::runtime_error("No timing for this key exists!");
	}
}


std::list<const TimingStream *> TimingSession::getStreams() const
{
	std::list<const TimingStream *> ret;

	for(const auto & [key, value] : m_streams)
	{
		ret.push_back(&value);
	}

	return ret;
}


void TimingSession::print(std::ostream & os) const
{
	os << "Session: " << m_name << std::endl;
	for(const auto & [name, stream] : m_streams)
	{
		stream.print(os);
	}
}


void TimingSession::printWithoutUnfinishedTimings(std::ostream & os) const
{
	os << "Session (only finished timings): " << m_name << std::endl;
	for (const auto &[name, stream] : m_streams)
	{
		stream.printWithoutUnfinishedTimings(os);
	}
}
