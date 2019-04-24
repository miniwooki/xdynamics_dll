#ifndef XQUADRATICKERNEL_H
#define XQUADRATICKERNEL_H

#include "xKernelFunction.h"

class XDYNAMICS_API xQuadraticKernel : public xKernelFunction
{
public:
	xQuadraticKernel();
	virtual ~xQuadraticKernel();
	virtual void setupKernel(double gap, xKernelFunctionData& d);
	virtual double sphKernel(double QSq);
	virtual vector3d sphKernelGrad(double QSq, vector3d& distVec);
};

#endif