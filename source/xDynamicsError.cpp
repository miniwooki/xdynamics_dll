#include "xDynamicsError.h"
#include "xLog.h"
#include "xdynamics_manager/xModel.h"
#include <iostream>

wchar_t xDynamicsError::err[] = L"";

xDynamicsError::xDynamicsError()
{

}

xDynamicsError::~xDynamicsError()
{

}

wchar_t* xDynamicsError::getErrorString()
{
	return err;
}

bool xDynamicsError::_check(int result, char const *const func, const char* const file, int const line)
{
	if (result)
	{
		swprintf_s(err, L"XDynamics error at %hs:%d code=%d(%ws) \"%hs\" \n", file, line, result, _xdynamicsGetErrorEnum(result), func);
		xLog::log(err);// std::cout << err << std::endl;//printf("%s", err);
		//return result;
		//xdynamicsReset();
		//exit(0);
	}
	return result;
}

wchar_t * xDynamicsError::_xdynamicsGetErrorEnum(int error)
{
	switch (error)
	{
	case xdynamicsErrorMultiBodySimulationHHTIterationOver:
		//eStr = "N-R iteration of HHT integrator is over.";
		return L"xdynamicsErrorMultiBodySimulationHHTIterationOver";
	case xdynamicsErrorLinearEquationCalculation:
		//eStr = "Calculation of linear solver(LAPACK) is failed.";
		return L"xdynamicsErrorLinearEquationCalculation";
	case xdynamicsErrorMultiBodyModelInitialization:
		return L"xdynamicsErrorMultiBodyModelInitialization";
	case xdynamicsErrorMultiBodyModelRedundantCondition:
		return L"xdynamicsErrorMultiBodyModelRedundantCondition";
	}
	return NULL;
}
