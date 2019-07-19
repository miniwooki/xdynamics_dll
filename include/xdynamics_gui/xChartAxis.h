#ifndef XCHARTAXIS_H
#define XCHARTAXIS_H

#include <QtCharts/QValueAxis>

QT_CHARTS_USE_NAMESPACE

class xChartAxis : public QValueAxis
{
public:
	explicit xChartAxis(double _min, double _max);
	~xChartAxis();
};

#endif