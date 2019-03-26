#ifndef XDYNAMICS_GLOBAL_H
#define XDYNAMICS_GLOBAL_H

// #include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_decl.h"
// #include "xdynamics_manager/xdynamics_manager.h"

//class xModel;
class xDynamicsManager;
class xDynamicsSimulator;

// const void* GLOBAL_XSIM = 0;
// const void* GLOBAL_XDM = 0;
inline XDYNAMICS_API void xdynamicsReset();
inline XDYNAMICS_API void SET_GLOBAL_XDYNAMICS_SIMULATOR(xDynamicsSimulator* _xsim);// uploadGlobalXSIM(_xsim)
inline XDYNAMICS_API void SET_GLOBAL_XDYNAMICS_MANAGER(xDynamicsManager* _xdm);// uploadGlobalXDM(_xdm)
/*#define XDYNAMICS_RESET xdynamicsReset();*/

#endif