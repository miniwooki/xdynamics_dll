#ifndef XTIME_H
#define XTIME_H

#include "xdynamics_decl.h"
#include <windows.h>

class XDYNAMICS_API xTime
{
public:
	xTime();
	~xTime();

	void reset();
	void start();
	void stop();
	float getElapsedTimeF();
	double getElapsedTimeD();

private:
	bool stopped;
	LARGE_INTEGER freq;
	LARGE_INTEGER counter_ini, counter_end;
	LARGE_INTEGER getElapsed();
};

#endif