//#include "xdynamics_gui.h"
#include "xSimulationWidget.h"
#include "xdynamics_simulation/xSimulation.h"

wsimulation::wsimulation(QWidget* parent /* = NULL */)
	: QWidget(parent)
{
	setupUi(this);
	connect(PBSolve, SIGNAL(clicked()), this, SLOT(SolveButton()));
	connect(LETimeStep, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
	connect(LESaveStep, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
	connect(LEEndTime, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
}

wsimulation::~wsimulation()
{

}

void wsimulation::SolveButton()
{
	//QString dt = LETimeStep->text();
	double dt = LETimeStep->text().toDouble();
	unsigned int st = LESaveStep->text().toUInt();
	double et = LEEndTime->text().toDouble();
	emit clickedSolveButton(dt, st, et);
}

void wsimulation::UpdateInformation()
{
	double dt = LETimeStep->text().toDouble();
	unsigned int st = LESaveStep->text().toUInt();
	double et = LEEndTime->text().toDouble();
	unsigned int nstep = static_cast<unsigned int>((et / dt));
	unsigned int npart = static_cast<unsigned int>((nstep / st)) + 1;
	LENumSteps->setText(QString("%1").arg(nstep));
	LENumParts->setText(QString("%1").arg(npart));
}
