//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_PROFILING_TIMER_H
#define COURSE_RENDERER_PROFILING_TIMER_H

#include <ctime>
class ProfilingTimer
{
private:
	clock_t timestamp;
public:
	void start_profile();
	float stop_profile(); // returns time in milliseconds
};

#endif//COURSE_RENDERER_PROFILING_TIMER_H
