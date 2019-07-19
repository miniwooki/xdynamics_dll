#include "xdynamics_global.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include "xdynamics_simulation/xDynamicsSimulator.h"

xDynamicsSimulator* GLOBAL_XSIM = 0;
xDynamicsManager* GLOBAL_XDM = 0;

void xdynamicsReset()
{
	if (GLOBAL_XSIM) delete GLOBAL_XSIM; GLOBAL_XSIM = 0;
	if (GLOBAL_XDM) delete GLOBAL_XDM; GLOBAL_XDM = 0;
}

void SET_GLOBAL_XDYNAMICS_SIMULATOR(xDynamicsSimulator* _xsim)
{
	GLOBAL_XSIM = _xsim;
}

void SET_GLOBAL_XDYNAMICS_MANAGER(xDynamicsManager* _xdm)
{
	GLOBAL_XDM = _xdm;
}