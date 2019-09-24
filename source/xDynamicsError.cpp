#include "xDynamicsError.h"
#include "xLog.h"
#include "xdynamics_manager/xModel.h"
#include <iostream>

char xDynamicsError::err[] = "";

xDynamicsError::xDynamicsError()
{

}

xDynamicsError::~xDynamicsError()
{

}

char* xDynamicsError::getErrorString()
{
	return err;
}

bool xDynamicsError::_check(int result, char const *const func, const char* const file, int const line)
{
	if (result)
	{
		sprintf_s(err, "XDynamics error at %s:%d code=%d(%s) \"%s\" \n", file, line, result, _xdynamicsGetErrorEnum(result), func);
	//	xLog::log(err);// std::cout << err << std::endl;//printf("%s", err);
		throw runtime_error(err);
		//return result;
		//xdynamicsReset();
		//exit(0);
	}
	return result;
}

char * xDynamicsError::_xdynamicsGetErrorEnum(int error)
{
	switch (error)
	{
	case xdynamicsErrorMultiBodySimulationHHTIterationOver:
		//eStr = "N-R iteration of HHT integrator is over.";
		return "xdynamicsErrorMultiBodySimulationHHTIterationOver";
	case xdynamicsErrorLinearEquationCalculation:
		//eStr = "Calculation of linear solver(LAPACK) is failed.";
		return "xdynamicsErrorLinearEquationCalculation";
	case xdynamicsErrorMultiBodyModelInitialization:
		return "xdynamicsErrorMultiBodyModelInitialization";
	case xdynamicsErrorMultiBodyModelRedundantCondition:
		return "xdynamicsErrorMultiBodyModelRedundantCondition";
	case xdynamicsErrorExcelModelingData:
		return "xdynamicsErrorExcelModelingData";
	}
	return NULL;
}
