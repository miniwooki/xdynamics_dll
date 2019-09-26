#ifndef XLINESERIES_H
#define XLINESERIES_H

#include <QtCharts>

class xLineSeries : public QLineSeries
{
public:
	xLineSeries();
	~xLineSeries();

	unsigned int get_num_point();
	double get_end_time();

	void set_num_point(unsigned int npoint);
	void set_end_time(double etime);

private:
	double end_time;
	unsigned int num_point;
};

#endif
