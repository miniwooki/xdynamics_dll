#ifndef XSIMULATIONWIDGET_H
#define XSIMULATIONWIDGET_H

#include "ui_wsimulation.h"

class wsimulation : public QWidget, public Ui::wsimulation
{
	Q_OBJECT
public:
	wsimulation(QWidget* parent = NULL);
	~wsimulation();

	public slots:
	void UpdateInformation();
signals:
	void clickedSolveButton(double, unsigned int, double);

	private slots:
	void SolveButton();
	
};

#endif