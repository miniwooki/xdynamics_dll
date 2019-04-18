#ifndef XSIMULATIONWIDGET_H
#define XSIMULATIONWIDGET_H

#include "ui_wsimulation.h"

class wsimulation : public QWidget, public Ui::wsimulation
{
	Q_OBJECT
public:
	wsimulation(QWidget* parent = NULL);
	~wsimulation();

signals:
	void clickedSolveButton();

	private slots:
	void SolveButton();
};

#endif