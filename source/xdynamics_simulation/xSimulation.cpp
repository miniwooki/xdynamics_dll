#include "xdynamics_simulation/xSimulation.h"

double xSimulation::init_dt = 0.0;
double xSimulation::dt = 0.0001;
xSimulation::deviceType xSimulation::dev = xSimulation::CPU;
xSimulation::SimulationType xSimulation::simulation_type = xSimulation::NO_DEFINED_SIMULATION_TYPE;
xSimulation::MBDSolverType xSimulation::mbd_solver_type = xSimulation::MBD_SOLVER;
xSimulation::DEMSolverType xSimulation::dem_solver_type = xSimulation::DEM_SOLVER;
xSimulation::SPHSolverType xSimulation::sph_solver_type = xSimulation::SPH_SOLVER;

bool xSimulation::triggered_stop_simulation = false;
double xSimulation::ctime = 0.0;
double xSimulation::et = 1.0;
double xSimulation::start_time = 0.0;
unsigned int xSimulation::st = 100;
unsigned int xSimulation::npart = 0;
unsigned int xSimulation::nstep = 0;

xSimulation::xSimulation()
//	: init_dt(0)
{

}

xSimulation::~xSimulation()
{

}

void xSimulation::initialize()
{
	init_dt = 0.0;
	dt = 0.0001;
	dev = xSimulation::CPU;
	mbd_solver_type = xSimulation::MBD_SOLVER;
	dem_solver_type = xSimulation::DEM_SOLVER;
	sph_solver_type = xSimulation::SPH_SOLVER;

	ctime = 0.0;
	et = 1.0;
	start_time = 0.0;
	st = 100;
	npart = 0;
	nstep = 0;
}

bool xSimulation::Cpu()
{
	return dev == CPU;
}

bool xSimulation::Gpu()
{
	return dev == GPU;
}

bool xSimulation::ConfirmStopSimulation()
{
	return triggered_stop_simulation;
}

void xSimulation::triggerStopSimulation()
{
	triggered_stop_simulation = true;
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

void xSimulation::setSPHSolverType(SPHSolverType sst)
{
	sph_solver_type = sst;
}

void xSimulation::setDEMPositionVelocity(double* p, double *v)
{
	dem_pos = p;
	dem_vel = v;
}
