#ifndef XWENDLANDKERNEL_H
#define XWENDLANDKERNEL_H

#include "xKernelFunction.h"

class XDYNAMICS_API xWendlandKernel : public xKernelFunction
{
public:
	xWendlandKernel();
	virtual ~xWendlandKernel();

	virtual void setupKernel(double gap, xKernelFunctionData& d);
	virtual double sphKernel(double QSq);
	virtual vector3d sphKernelGrad(double QSq, vector3d& distVec);
};

#endif