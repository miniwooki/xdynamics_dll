#ifndef XKERNELFUNCTION_H
#define XKERNELFUNCTION_H

#include "xdynamics_decl.h"
#include "xdynamics_algebra/xAlgebraMath.h"

class xSmoothedParticleHydrodynamicsModel;

class XDYNAMICS_API xKernelFunction
{
public:
	xKernelFunction();
	virtual ~xKernelFunction();

	virtual void setupKernel(double gap, xKernelFunctionData& d);
	virtual double sphKernel(double QSq) = 0;
	virtual vector3d sphKernelGrad(double QSq, vector3d& distVec) = 0;

	double KernelConst();// { return kernel_const; }
	double KernelGradConst();// { return kernel_grad_const; }
	double KernelSupport();// { return kernel_support; }
	double KernelSupprotSq();// { return kernel_support_sq; }

	double h;
	double h_sq;
	double h_inv;
	double h_inv_sq;
	double h_inv_2;
	double h_inv_3;
	double h_inv_4;
	double h_inv_5;

protected:
	double kernel_support;
	double kernel_support_sq;
	double kernel_const;
	double kernel_grad_const;

	xSmoothedParticleHydrodynamicsModel *sph;
};

#endif