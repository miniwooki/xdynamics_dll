#ifndef XDYNAMICSERROR_H
#define XDYNAMICSERROR_H

//#include "xdynamics_manager/xdynamics_manager_decl.h"
#include "xdynamics_decl.h"
#include "xdynamics_global.h"
#include <cstring>

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
		xdynamicsErrorMultiBodyModelRedundantCondition = 5
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
// 
// 
// 
// template<typename T>
// void check(T result, char const *const func, const char* const file, int const line)
// {
// 	if (result)
// 	{
// 		fprintf(stderr, "XDynamics error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _xdynamicsGetErrorEnum(result), func);
// 		XDYNAMICS_RESET;
// 		exit(EXIT_FAILURE);
// 	}
// }

#define checkXerror(val) xDynamicsError::_check ( (val), #val, __FILE__, __LINE__ )
//
//
//
#endif