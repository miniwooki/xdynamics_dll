//#include "xdynamics_gui.h"
#include "xSimulationWidget.h"
#include "xdynamics_simulation/xSimulation.h"

wsimulation::wsimulation(QWidget* parent /* = NULL */)
	: QWidget(parent)
	, is_check_starting_point(false)
	, starting_part(0)
	, is_simulation_ing(false)
{
	setupUi(this);
	connect(PBSolve, SIGNAL(clicked()), this, SLOT(SolveButton()));
	connect(LETimeStep, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
	connect(LESaveStep, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
	connect(LEEndTime, SIGNAL(editingFinished()), this, SLOT(UpdateInformation()));
	connect(PB_Select_SP, SIGNAL(clicked()), this, SLOT(StartingPointButton()));
	connect(GB_StartingPoint, SIGNAL(clicked(bool)), this, SLOT(CheckStartingPoint(bool)));
}

wsimulation::~wsimulation()
{

}

void wsimulation::set_starting_point(QString item, unsigned int sp)
{
	LE_StartingPoint->setText(item);
	starting_part = sp;
}

bool wsimulation::get_enable_starting_point()
{
	return is_check_starting_point;
}

unsigned int wsimulation::get_starting_part()
{
	return starting_part;
}

QString wsimulation::get_starting_point_path()
{
	return LE_StartingPoint->text();
}

bool wsimulation::is_simulationing()
{
	return is_simulation_ing;
}

void wsimulation::set_stop_state()
{
	PBSolve->setText("Stop!!!");
	PBSolve->disconnect();
	connect(PBSolve, SIGNAL(clicked()), this, SLOT(StopButton()));
}

void wsimulation::set_start_state()
{
	PBSolve->setText("Solve");
	PBSolve->disconnect();
	connect(PBSolve, SIGNAL(clicked()), this, SLOT(SolveButton()));
}

void wsimulation::StartingPointButton()
{
	emit clickedStartPointButton();
}

void wsimulation::CheckStartingPoint(bool b)
{
	is_check_starting_point = b;
}

void wsimulation::SolveButton()
{
	//QString dt = LETimeStep->text();
	double dt = LETimeStep->text().toDouble();
	unsigned int st = LESaveStep->text().toUInt();
	double et = LEEndTime->text().toDouble();
	is_simulation_ing = true;
	emit clickedSolveButton(dt, st, et);
}

void wsimulation::StopButton()
{
	is_simulation_ing = false;
	emit clickedStopButton();
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
