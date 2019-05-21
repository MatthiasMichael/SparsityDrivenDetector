#pragma once

#include <chrono>


class TimingEntry
{
public:
	using TimePoint = std::chrono::high_resolution_clock::time_point;

	TimingEntry();

	void finish();

	bool finished() const;

	double duration() const;

	const auto & getStart() const { return m_start; }
	const auto & getStop() const { return m_stop; }

	int getFramenumber() const { return m_framenumber; }

private:
	int m_framenumber;
	
	TimePoint m_start;
	TimePoint m_stop;
};
