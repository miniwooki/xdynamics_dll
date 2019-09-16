#ifndef XDYNAMICSERROR_H
#define XDYNAMICSERROR_H

//#include "xdynamics_manager/xdynamics_manager_decl.h"
#include "xdynamics_decl.h"
#include "xdynamics_global.h"
#include <cstring>

#define ERROR_DETECTED -99

class XDYNAMICS_API xDynamicsError
{
public:
	enum
	{
		xdynamicsSuccess = 0,
		xdynamicsErrorMultiBodySimulationHHTIterationOver = 1,
		xdynamicsErrorLinearEquationCalculation = 2,
		xdynamicsErrorMultiBodyModelInitialization = 3,
		xdynamicsErrorDiscreteElementMethodModelInitialization = 4,
		xdynamicsErrorMultiBodyModelRedundantCondition = 5,
		xdynamicsErrorIncompressibleSPHInitialization = 6,
		xdynamicsErrorExcelModelingData = 7
	};
	xDynamicsError();
	~xDynamicsError();
	static char* getErrorString();
	//static void checkXerror(int val);
	static bool _check(int result, char const *const func, const char* const file, int const line);
private:
	static char *_xdynamicsGetErrorEnum(int error);
	static char err[255];
};

#define checkXerror(val) xDynamicsError::_check ( (val), #val, __FILE__, __LINE__ )

#endif