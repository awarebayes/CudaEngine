//
// Created by dev on 8/27/22.
//
#include "profiling_timer.h"

void ProfilingTimer::start_profile() {
	timestamp = clock();
}
float ProfilingTimer::stop_profile() {
	float elapsed_secs = float(clock() - timestamp) / CLOCKS_PER_SEC;
	timestamp = 0;
	return elapsed_secs;
}
