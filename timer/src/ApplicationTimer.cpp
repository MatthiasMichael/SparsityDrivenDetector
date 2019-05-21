#include "ApplicationTimer.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdarg>

ApplicationTimer::ApplicationTimer() : m_session("DEFAULT_SESSION")
{
	// empty
}


void ApplicationTimer::startSession(const std::string & name)
{
	m_session = TimingSession(name);
}


void ApplicationTimer::start(const std::string & name)
{
	m_session.start(name);
}


void ApplicationTimer::stop(const std::string & name)
{
	m_session.stop(name);
}


double ApplicationTimer::stopWithTime(const std::string & name)
{
	return m_session.stopWithTime(name);
}


void ApplicationTimer::printStatistics(std::ostream & os) const
{
	m_session.print(os);
}


void ApplicationTimer::printStatisticsWithoutUnfinishedTimings(std::ostream & os) const
{
	m_session.printWithoutUnfinishedTimings(os);
}
