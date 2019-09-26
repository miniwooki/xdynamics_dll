#include "xLineSeries.h"

xLineSeries::xLineSeries()
	: QLineSeries()
	, end_time(0.0)
	, num_point(0)
{

}

xLineSeries::~xLineSeries()
{

}

unsigned int xLineSeries::get_num_point()
{
	return num_point;
}

double xLineSeries::get_end_time()
{
	return end_time;
}

void xLineSeries::set_num_point(unsigned int npoint)
{
	num_point = npoint;
}

void xLineSeries::set_end_time(double etime)
{
	end_time = etime;
}
