#ifndef XQUINTICKERNEL_H
#define XQUINTICKERNEL_H

#include "xKernelFunction.h"

class XDYNAMICS_API xQuinticKernel : public xKernelFunction
{
public:
	xQuinticKernel();
	virtual ~xQuinticKernel();

	virtual void setupKernel(double gap, xKernelFunctionData& d);
	virtual double sphKernel(double QSq);
	virtual vector3d sphKernelGrad(double QSq, vector3d& distVec);
};

#endif