#pragma once

#include <deque>
#include <string>

#include "TimingEntry.h"

class TimingStream
{
public:
	using TimePoint = TimingEntry::TimePoint;

	explicit TimingStream(const std::string & name);

	void start();
	void stop();
	double stopWithTime();

	const std::string & getName() const { return m_name; }
	const auto & getTimings() const { return m_timings; }

	void print(std::ostream & os) const;
	void printWithoutUnfinishedTimings(std::ostream & os) const;

private:
	std::string m_name;
	std::deque<TimingEntry> m_timings;
};