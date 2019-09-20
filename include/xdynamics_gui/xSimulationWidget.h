#ifndef XSIMULATIONWIDGET_H
#define XSIMULATIONWIDGET_H

#include "ui_wsimulation.h"

class wsimulation : public QWidget, public Ui::wsimulation
{
	Q_OBJECT
public:
	wsimulation(QWidget* parent = NULL);
	~wsimulation();

	void set_starting_point(QString item, unsigned int sp);
	bool get_enable_starting_point();
	unsigned int get_starting_part();
	QString get_starting_point_path();

private:
	bool is_check_starting_point;
	unsigned int starting_part;
	public slots:
	void UpdateInformation();
signals:
	void clickedSolveButton(double, unsigned int, double);
	void clickedStartPointButton();

	private slots:
	void SolveButton();
	void StartingPointButton();
	void CheckStartingPoint(bool);
	
};

#endif