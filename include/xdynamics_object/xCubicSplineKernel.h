#ifndef XCUBICSPLINEKERNEL_H
#define XCUBICSPLINEKERNEL_H

#include "xKernelFunction.h"

class XDYNAMICS_API xCubicSplineKernel : public xKernelFunction
{
public:
	xCubicSplineKernel();
	virtual ~xCubicSplineKernel();

	virtual void setupKernel(double gap, xKernelFunctionData& d);
	virtual double sphKernel(double QSq);
	virtual vector3d sphKernelGrad(double QSq, vector3d& distVec);
};

#endif