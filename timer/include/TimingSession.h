#pragma once

#include <list>
#include <map>

#include "TimingStream.h"

class TimingSession
{
public:
	using TimePoint = TimingStream::TimePoint;

	TimingSession(const std::string & name);

	void start(const std::string & stream);
	void stop(const std::string & stream);
	double stopWithTime(const std::string & stream);

	std::list<const TimingStream *> getStreams() const;

	bool empty() const { return m_streams.empty(); }

	void removeStream(const std::string & stream) { m_streams.erase(stream); }

	const auto & getStart() const { return m_sessionStart; }

	void print(std::ostream & os) const;
	void printWithoutUnfinishedTimings(std::ostream & os) const;

private:
	std::string m_name;
	std::map<std::string, TimingStream> m_streams;

	TimePoint m_sessionStart;
};