#include "xdynamics_simulation/xSimulation.h"

double xSimulation::init_dt = 0.0;
double xSimulation::dt = 0.0;
xSimulation::deviceType xSimulation::dev = xSimulation::CPU;
xSimulation::MBDSolverType xSimulation::mbd_solver_type = xSimulation::MBD_SOLVER;
xSimulation::DEMSolverType xSimulation::dem_solver_type = xSimulation::DEM_SOLVER;
double xSimulation::ctime = 0.0;
double xSimulation::et = 0.0;
double xSimulation::start_time = 0.0;
unsigned int xSimulation::st = 0;
unsigned int xSimulation::npart = 0;
unsigned int xSimulation::nstep = 0;

xSimulation::xSimulation()
//	: init_dt(0)
{

}

xSimulation::~xSimulation()
{

}

bool xSimulation::Cpu()
{
	return dev == CPU;
}

bool xSimulation::Gpu()
{
	return dev == GPU;
}

void xSimulation::setCPUDevice()
{
	dev = CPU;
}

void xSimulation::setGPUDevice()
{
	dev = GPU;
}

void xSimulation::setTimeStep(double _dt)
{
	dt = _dt;
}

void xSimulation::setCurrentTime(double _ct)
{
	ctime = _ct;
}

void xSimulation::setStartTime(double _st)
{
	start_time = _st;
}

void xSimulation::setSaveStep(unsigned int _ss)
{
	st = _ss;
}

void xSimulation::setEndTime(double _et)
{
	et = _et;
}

void xSimulation::setMBDSolverType(MBDSolverType mst)
{
	mbd_solver_type = mst;
}

void xSimulation::setDEMSolverType(DEMSolverType dst)
{
	dem_solver_type = dst;
}