#include "xTime.h"

xTime::xTime()
{
	QueryPerformanceFrequency(&freq);
	reset();
}

xTime::~xTime()
{

}

void xTime::reset()
{
	stopped = false;
	counter_ini.QuadPart = 0; 
	counter_end.QuadPart = 0;
}

void xTime::start()
{
	stopped = false;
	QueryPerformanceCounter(&counter_ini);
}

void xTime::stop()
{
	QueryPerformanceCounter(&counter_end); 
	stopped = true;
}

float xTime::getElapsedTimeF()
{
	return ((float(getElapsed().QuadPart)) / float(freq.QuadPart));
}

double xTime::getElapsedTimeD()
{
	return ((double(getElapsed().QuadPart)) / double(freq.QuadPart));
}

LARGE_INTEGER xTime::getElapsed()
{
	LARGE_INTEGER dif; dif.QuadPart = (stopped ? counter_end.QuadPart - counter_ini.QuadPart : 0);
	return(dif);
}
