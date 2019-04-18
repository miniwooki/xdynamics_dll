//#include "xdynamics_gui.h"
#include "xSimulationWidget.h"
#include "xdynamics_simulation/xSimulation.h"

wsimulation::wsimulation(QWidget* parent /* = NULL */)
	: QWidget(parent)
{
	setupUi(this);
	connect(PBSolve, SIGNAL(clicked()), this, SLOT(SolveButton()));
}

wsimulation::~wsimulation()
{

}

void wsimulation::SolveButton()
{
	QString dt = LETimeStep->text();
	xSimulation::dt = LETimeStep->text().toDouble();
	xSimulation::st = LESaveStep->text().toUInt();
	xSimulation::et = LEEndTime->text().toDouble();
	emit clickedSolveButton();
}