#pragma once

#include <iomanip>
#include <map>
#include <string>
#include <chrono>

#include "TimingSession.h"



class ApplicationTimer
{
public:
	static ApplicationTimer& getInstance()
	{
		static ApplicationTimer instance;
		return instance;
	}

	ApplicationTimer(ApplicationTimer const &) = delete;
	ApplicationTimer & operator=(ApplicationTimer const &) = delete;

	void startSession(const std::string & name);
	const TimingSession & getSession() const { return m_session; }

	void start(const std::string & key);
	void stop(const std::string & key);
	double stopWithTime(const std::string & key);

	void printStatistics(std::ostream & os) const;
	void printStatisticsWithoutUnfinishedTimings(std::ostream & os) const;


	void setFramenumber(int framenumber) { m_framenumber = framenumber; }
	int getFramenumber() const { return m_framenumber; }

	const auto & getSessionStart() const { return m_session.getStart();  }

private:
	ApplicationTimer();

private:
	TimingSession m_session;
	int m_framenumber;
};


template<typename T> void printElement(std::ostream & os, T t, const int& width)
{
	os << std::right << std::setw(width) << std::setfill(' ') << std::setprecision(5) << t;
}


#define AT_START_SESSION(name) ApplicationTimer::getInstance().startSession(name)
#define AT_SESSION() ApplicationTimer::getInstance().getSession()
#define AT_GET_SESSION_START() ApplicationTimer::getInstance().getSessionStart()
#define AT_START(name) ApplicationTimer::getInstance().start(name)
#define AT_STOP(name) ApplicationTimer::getInstance().stop(name)
#define AT_STOP_GET_TIME(name) ApplicationTimer::getInstance().stopWithTime(name)
#define AT_PRINT() ApplicationTimer::getInstance().printStatistics(std::cout)
#define AT_PRINT_NO_UNFINISHED() ApplicationTimer::getInstance().printStatisticsWithoutUnfinishedTimings(std::cout)
#define AT_SET_FRAME(f) ApplicationTimer::getInstance().setFramenumber(f) 
#define AT_GET_FRAME() ApplicationTimer::getInstance().getFramenumber() 
// #define AT_PRINT_LAST(...) ApplicationTimer::getInstance().printLastTimings(__VA_ARGS__, nullptr)
// #define AT_CLEAR(...) ApplicationTimer::getInstance().clear(__VA_ARGS__, nullptr)
// #define AT_CLEAR_ALL() ApplicationTimer::getInstance().clearAll()
