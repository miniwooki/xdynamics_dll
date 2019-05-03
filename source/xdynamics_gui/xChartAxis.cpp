#include "xChartAxis.h"

xChartAxis::xChartAxis(double _min, double _max)
	: QValueAxis(NULL)
{
	setMin(_min);
	setMax(_max);
}

xChartAxis::~xChartAxis()
{

}