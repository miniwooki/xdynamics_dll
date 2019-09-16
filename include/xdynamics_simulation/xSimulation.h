#ifndef XSIMULATION_H
#define XSIMULATION_H

#include "xdynamics_decl.h"
#include "xDynamicsError.h"
#include "xdynamics_algebra/xAlgebraMath.h"

class XDYNAMICS_API xSimulation
{
public:
	enum deviceType{ CPU = 0, GPU };
	enum SimulationType{ NO_DEFINED_SIMULATION_TYPE = -1, ONLY_MBD = 0, ONLY_DEM, ONLY_SPH, MBD_DEM, MBD_SPH, SPH_DEM };
	enum MBDSolverType{ MBD_SOLVER = 0, EXPLICIT_RK4 = 1, IMPLICIT_HHT = 2, KINEMATIC = 10 };
	enum DEMSolverType{ DEM_SOLVER = 0, EXPLICIT_VV = 1 };
	enum SPHSolverType{ SPH_SOLVER = 0, INCOMPRESSIBLESPH, WEAKELYSPH };
	//enum integratorType{ };
	xSimulation();
	~xSimulation();

	static void initialize();
	static bool Cpu();
	static bool Gpu();
	static bool ConfirmStopSimulation();
	static void triggerStopSimulation();
	static void setCPUDevice();
	static void setGPUDevice();
	static void setTimeStep(double _dt);
	static void setCurrentTime(double _ct);
	static void setStartTime(double _st);
	static void setSaveStep(unsigned int _ss);
	static void setEndTime(double _et);
	static void setMBDSolverType(MBDSolverType mst);
	static void setDEMSolverType(DEMSolverType dst);
	static void setSPHSolverType(SPHSolverType sst);
	void setDEMPositionVelocity(double* p, double *v);

	static bool triggered_stop_simulation;
	static double start_time;
	static double et;
	static double init_dt;
	static double dt;
	static double ctime;
	static unsigned int st;
	static unsigned int npart;
	static unsigned int nstep;
	static deviceType dev;
	static SimulationType simulation_type;
	static MBDSolverType mbd_solver_type;
	static DEMSolverType dem_solver_type;
	static SPHSolverType sph_solver_type;
	double* dem_pos;
	double* dem_vel;

};

#endif